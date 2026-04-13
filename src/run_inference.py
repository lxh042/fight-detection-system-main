import argparse
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference_core import InferenceConfig, print_summary, process_stream
from utills.frame_to_keypoits import extract_keypoints


def parse_args():
    parser = argparse.ArgumentParser(
        description='Headless fight detection inference for video files or camera streams.'
    )
    parser.add_argument('--source-type', choices=['video', 'camera'], default='video')
    parser.add_argument('--video-path', help='Path to a local video file when source-type=video.')
    parser.add_argument('--camera-index', type=int, default=0, help='Camera index when source-type=camera.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Fight classification threshold.')
    parser.add_argument('--smooth-k', type=int, default=5, help='Sliding window size for probability smoothing.')
    parser.add_argument('--cooldown', type=float, default=5.0, help='Cooldown seconds between incident counts.')
    parser.add_argument('--max-frames', type=int, default=None, help='Optional cap on processed frames.')
    parser.add_argument('--output-video', help='Optional path to save annotated output video.')
    parser.add_argument('--event-log', help='Optional JSON path for incident records.')
    parser.add_argument('--summary-json', help='Optional JSON path for inference summary.')
    parser.add_argument('--log-every', type=int, default=30, help='Print status every N processed frames.')
    return parser.parse_args()


def main():
    args = parse_args()
    config = InferenceConfig(
        source_type=args.source_type,
        video_path=args.video_path,
        camera_index=args.camera_index,
        threshold=args.threshold,
        smooth_k=args.smooth_k,
        cooldown=args.cooldown,
        max_frames=args.max_frames,
        output_video=args.output_video,
        event_log=args.event_log,
        summary_json=args.summary_json,
        log_every=args.log_every,
    )
    summary = process_stream(config, extract_keypoints)
    print_summary(summary)


if __name__ == '__main__':
    main()