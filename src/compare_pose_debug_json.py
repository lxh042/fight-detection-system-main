import argparse
import json
import os
from typing import List, Tuple

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Compare pose debug JSON exported by Python and C++.')
    parser.add_argument('--reference-json', required=True)
    parser.add_argument('--candidate-json', required=True)
    return parser.parse_args()


def load_json(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f'未找到 JSON: {path}')
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)


def to_array(values: List[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    xa1, ya1, xa2, ya2 = box_a.tolist()
    xb1, yb1, xb2, yb2 = box_b.tolist()
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return float(inter_area / union)


def match_detections(reference_detections, candidate_detections) -> List[Tuple[int, int, float]]:
    matches = []
    used_candidate_indices = set()
    for ref_index, ref_det in enumerate(reference_detections):
        ref_box = to_array(ref_det['box_xyxy'])
        best_candidate = -1
        best_iou = -1.0
        for cand_index, cand_det in enumerate(candidate_detections):
            if cand_index in used_candidate_indices:
                continue
            cand_box = to_array(cand_det['box_xyxy'])
            current_iou = box_iou(ref_box, cand_box)
            if current_iou > best_iou:
                best_iou = current_iou
                best_candidate = cand_index
        if best_candidate >= 0:
            used_candidate_indices.add(best_candidate)
            matches.append((ref_index, best_candidate, best_iou))
    return matches


def main():
    args = parse_args()
    reference = load_json(args.reference_json)
    candidate = load_json(args.candidate_json)

    print('Pose debug summary')
    print(f"frame_index: {reference['frame_index']} vs {candidate['frame_index']}")
    print(f"detection_count: {reference['detection_count']} vs {candidate['detection_count']}")

    matches = match_detections(reference['detections'], candidate['detections'])
    for ref_index, cand_index, iou in matches:
        ref_det = reference['detections'][ref_index]
        cand_det = candidate['detections'][cand_index]

        ref_box = to_array(ref_det['box_xyxy'])
        cand_box = to_array(cand_det['box_xyxy'])
        box_diff = np.abs(ref_box - cand_box)

        ref_kpts = to_array(ref_det['keypoints']).reshape(-1)
        cand_kpts = to_array(cand_det['keypoints']).reshape(-1)
        kpt_diff = np.abs(ref_kpts - cand_kpts)

        print(f'\nreference[{ref_index}] <-> candidate[{cand_index}] iou={iou:.6f}')
        print(f"score_ref={ref_det['score']:.6f} score_cpp={cand_det['score']:.6f} score_abs_diff={abs(ref_det['score'] - cand_det['score']):.6f}")
        print(f'box_mean_abs_diff={float(box_diff.mean()):.6f} box_max_abs_diff={float(box_diff.max()):.6f}')
        print(f'kpt_mean_abs_diff={float(kpt_diff.mean()):.6f} kpt_max_abs_diff={float(kpt_diff.max()):.6f}')

    if reference['detection_count'] != candidate['detection_count']:
        print('\nDetection counts differ, so IoU matching is partial.')


if __name__ == '__main__':
    main()