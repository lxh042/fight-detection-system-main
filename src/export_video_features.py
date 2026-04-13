import argparse
import csv
import os
import sys

import cv2


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utills.frame_to_keypoits import extract_keypoints


def parse_args():
    parser = argparse.ArgumentParser(description='Export per-frame YOLO pose features to CSV.')
    parser.add_argument('--video-path', required=True, help='Input video path.')
    parser.add_argument('--output-csv', required=True, help='Output CSV path.')
    parser.add_argument('--max-frames', type=int, default=None, help='Optional cap on processed frames.')
    parser.add_argument('--log-every', type=int, default=30, help='Print progress every N frames.')
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f'未找到视频文件: {args.video_path}')

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f'无法打开视频: {args.video_path}')

    processed_frames = 0
    feature_dim = None

    try:
        with open(args.output_csv, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                processed_frames += 1
                features, _ = extract_keypoints(frame)
                if feature_dim is None:
                    feature_dim = int(len(features))
                    writer.writerow(['frame_index', 'feature_dim'] + [f'f{i}' for i in range(feature_dim)])

                writer.writerow([processed_frames, feature_dim] + [float(value) for value in features.tolist()])

                if processed_frames == 1 or (args.log_every > 0 and processed_frames % args.log_every == 0):
                    print(f'[frame {processed_frames}] exported feature_dim={feature_dim}')

                if args.max_frames is not None and processed_frames >= args.max_frames:
                    break
    finally:
        cap.release()

    print(f'exported_frames: {processed_frames}')
    print(f'output_csv: {args.output_csv}')


if __name__ == '__main__':
    main()