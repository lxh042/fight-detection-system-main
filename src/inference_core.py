import datetime
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import cv2
import mindspore as ms
import numpy as np
from mindspore import Tensor, context, nn
from mindspore.train.serialization import load

#确定version3.mindir路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
MINDIR_PATH = os.path.join(MODEL_DIR, 'version3.mindir')

#推理配置项
@dataclass
class InferenceConfig:
    source_type: str = 'video'
    video_path: Optional[str] = None
    camera_index: int = 0
    threshold: float = 0.5
    smooth_k: int = 5
    cooldown: float = 5.0
    max_frames: Optional[int] = None
    output_video: Optional[str] = None
    event_log: Optional[str] = None
    summary_json: Optional[str] = None
    log_every: int = 30


@dataclass
class InferenceSummary:
    source: str
    processed_frames: int
    incident_count: int
    elapsed_seconds: float
    avg_fps: float
    output_video: Optional[str] = None
    event_log: Optional[str] = None
    incidents: list = field(default_factory=list)

#你创建文件夹
def ensure_parent_dir(path: Optional[str]):
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

#加载 MindSpore 格式的预训练模型，设置成 CPU 推理模式，返回可直接调用的模型对象
def load_classifier():
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    graph = load(MINDIR_PATH)
    return nn.GraphCell(graph)

#根据你的配置（Config），自动选择打开摄像头还是视频文件，并做安全检查
def open_capture(config: InferenceConfig):
    if config.source_type == 'camera':
        return cv2.VideoCapture(config.camera_index), f'camera:{config.camera_index}'

    if not config.video_path:
        raise ValueError('source_type=video 时必须提供 video_path')
    if not os.path.exists(config.video_path):
        raise FileNotFoundError(f'未找到视频文件: {config.video_path}')
    return cv2.VideoCapture(config.video_path), config.video_path

#创建一个 OpenCV 视频写入器，用来保存带打斗标注的视频
def build_writer(output_path, cap, first_frame):
    if not output_path:
        return None

    ensure_parent_dir(output_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 20.0
    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#输入 41 帧的关键点序列，用模型推理，输出原始概率、平滑后概率，并判定是不是打斗
def classify_sequence(model, frame_buffer, prob_buffer, threshold, smooth_k):
    input_data = np.expand_dims(frame_buffer, axis=0).astype(np.float32)
    ms_out = model(Tensor(input_data, ms.float32))
    prob = float(ms_out.asnumpy().reshape(-1)[0])

    prob_buffer.append(prob)
    if len(prob_buffer) > smooth_k:
        prob_buffer.pop(0)

    smoothed = float(np.mean(prob_buffer))
    is_fight = smoothed > threshold
    return prob, smoothed, is_fight


def write_json(path: Optional[str], payload: dict):
    if not path:
        return

    ensure_parent_dir(path)
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def process_stream(config: InferenceConfig, extract_keypoints_fn):
    model = load_classifier()
    cap, source_desc = open_capture(config)
    if not cap.isOpened():
        raise RuntimeError(f'无法打开输入源: {source_desc}')

    frame_buffer = []
    prob_buffer = []
    incidents = []
    incident_count = 0
    last_incident_time = 0.0
    processed_frames = 0
    writer = None
    start_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frames += 1
            keypoints, result = extract_keypoints_fn(frame)
            annotated_frame = result.plot()

            label = 'warming-up'
            prob = None
            smoothed = None

            frame_buffer.append(keypoints)
            if len(frame_buffer) > 41:
                frame_buffer.pop(0)

            if len(frame_buffer) == 41:
                prob, smoothed, is_fight = classify_sequence(
                    model,
                    frame_buffer,
                    prob_buffer,
                    config.threshold,
                    config.smooth_k,
                )
                label = 'fight' if is_fight else 'no-fight'
                color = (0, 0, 255) if is_fight else (0, 255, 0)
                cv2.putText(
                    annotated_frame,
                    f'{label} (p={smoothed:.2f}, thr={config.threshold:.2f})',
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2,
                )

                current_time = time.time()
                if is_fight and (current_time - last_incident_time > config.cooldown):
                    incident_count += 1
                    last_incident_time = current_time
                    incident = {
                        'index': incident_count,
                        'frame': processed_frames,
                        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'raw_prob': round(prob, 4),
                        'smoothed_prob': round(smoothed, 4),
                        'threshold': config.threshold,
                    }
                    incidents.append(incident)
                    print(
                        f"[incident] #{incident_count} frame={processed_frames} "
                        f"time={incident['timestamp']} prob={smoothed:.4f}"
                    )

            if writer is None and config.output_video:
                writer = build_writer(config.output_video, cap, annotated_frame)
            if writer is not None:
                writer.write(annotated_frame)

            if processed_frames == 1 or (config.log_every > 0 and processed_frames % config.log_every == 0):
                if prob is None:
                    print(f'[frame {processed_frames}] collecting sequence buffer: {len(frame_buffer)}/41')
                else:
                    print(
                        f'[frame {processed_frames}] label={label} '
                        f'prob={prob:.4f} smoothed={smoothed:.4f} incidents={incident_count}'
                    )

            if config.max_frames is not None and processed_frames >= config.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    elapsed = time.time() - start_time
    fps = processed_frames / elapsed if elapsed > 0 else 0.0
    summary = InferenceSummary(
        source=source_desc,
        processed_frames=processed_frames,
        incident_count=incident_count,
        elapsed_seconds=elapsed,
        avg_fps=fps,
        output_video=config.output_video,
        event_log=config.event_log,
        incidents=incidents,
    )

    if config.event_log:
        write_json(config.event_log, {'source': source_desc, 'incidents': incidents})
    if config.summary_json:
        write_json(config.summary_json, asdict(summary))

    return summary


def print_summary(summary: InferenceSummary):
    print('\nInference summary')
    print(f'source: {summary.source}')
    print(f'processed_frames: {summary.processed_frames}')
    print(f'incident_count: {summary.incident_count}')
    print(f'elapsed_seconds: {summary.elapsed_seconds:.2f}')
    print(f'avg_fps: {summary.avg_fps:.2f}')
    if summary.output_video:
        print(f'output_video: {summary.output_video}')
    if summary.event_log:
        print(f'event_log: {summary.event_log}')