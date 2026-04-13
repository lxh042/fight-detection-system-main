import argparse
import json
import os
import sys

import cv2


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ultralytics import YOLO


YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, 'yolo11n-pose.pt')


def parse_args():
    parser = argparse.ArgumentParser(description='Export Ultralytics pose detections for a specific frame to JSON.')
    parser.add_argument('--video-path', required=True)
    parser.add_argument('--frame-index', required=True, type=int)
    parser.add_argument('--output-json', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.frame_index <= 0:
        raise ValueError('frame-index must be >= 1')
    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f'未找到视频文件: {args.video_path}')

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f'无法打开视频: {args.video_path}')

    try:
        current_index = 0
        frame = None
        while cap.isOpened():
            ok, current_frame = cap.read()
            if not ok:
                break
            current_index += 1
            if current_index == args.frame_index:
                frame = current_frame
                break
    finally:
        cap.release()

    if frame is None:
        raise RuntimeError(f'视频不足 {args.frame_index} 帧')

    model = YOLO(YOLO_MODEL_PATH)
    result = model.predict(frame, verbose=False)[0]

    boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
    scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else []
    keypoints = result.keypoints.data.cpu().numpy() if result.keypoints is not None else []

    detections = []
    count = min(len(boxes), len(scores), len(keypoints))
    for index in range(count):
        detections.append(
            {
                'score': float(scores[index]),
                'box_xyxy': [float(value) for value in boxes[index].tolist()],
                'keypoints': [[float(value) for value in kp.tolist()] for kp in keypoints[index]],
            }
        )

    payload = {
        'frame_index': args.frame_index,
        'image_width': int(frame.shape[1]),
        'image_height': int(frame.shape[0]),
        'detection_count': len(detections),
        'detections': detections,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, 'w', encoding='utf-8') as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    print(f'output_json: {args.output_json}')
    print(f'detection_count: {len(detections)}')


if __name__ == '__main__':
    main()