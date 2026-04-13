import argparse
import csv
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Compare two per-frame feature CSV files.')
    parser.add_argument('--reference-csv', required=True, help='Reference CSV, usually Python/Ultralytics output.')
    parser.add_argument('--candidate-csv', required=True, help='Candidate CSV, usually C++ output.')
    parser.add_argument('--show-top', type=int, default=5, help='Show top-N worst frames by mean abs diff.')
    return parser.parse_args()


def load_csv(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f'未找到 CSV: {path}')

    rows = []
    with open(path, 'r', encoding='utf-8', newline='') as file:
        reader = csv.reader(file)
        header = next(reader, None)
        if header is None or len(header) < 3 or header[0] != 'frame_index':
            raise ValueError(f'CSV 头格式不正确: {path}')

        for row in reader:
            if not row:
                continue
            frame_index = int(row[0])
            feature_dim = int(row[1])
            values = np.asarray([float(value) for value in row[2:]], dtype=np.float32)
            if values.size != feature_dim:
                raise ValueError(f'CSV 行特征维度不匹配: {path} frame={frame_index}')
            rows.append((frame_index, values))
    return rows


def main():
    args = parse_args()
    reference_rows = load_csv(args.reference_csv)
    candidate_rows = load_csv(args.candidate_csv)

    if len(reference_rows) != len(candidate_rows):
        raise ValueError(
            f'帧数不一致: reference={len(reference_rows)} candidate={len(candidate_rows)}'
        )

    frame_stats = []
    all_diffs = []
    for (ref_frame, ref_values), (cand_frame, cand_values) in zip(reference_rows, candidate_rows):
        if ref_frame != cand_frame:
            raise ValueError(f'帧编号不一致: reference={ref_frame} candidate={cand_frame}')
        if ref_values.shape != cand_values.shape:
            raise ValueError(f'特征形状不一致: frame={ref_frame}')

        diff = np.abs(ref_values - cand_values)
        frame_stats.append(
            {
                'frame': ref_frame,
                'mean_abs_diff': float(diff.mean()),
                'max_abs_diff': float(diff.max()),
                'l2': float(np.linalg.norm(diff)),
            }
        )
        all_diffs.append(diff)

    merged = np.concatenate(all_diffs) if all_diffs else np.zeros((0,), dtype=np.float32)
    print('Comparison summary')
    print(f'frames: {len(frame_stats)}')
    print(f'global_mean_abs_diff: {float(merged.mean()):.6f}')
    print(f'global_max_abs_diff : {float(merged.max()):.6f}')
    print(f'global_p95_abs_diff : {float(np.percentile(merged, 95)):.6f}')

    worst_frames = sorted(frame_stats, key=lambda item: item['mean_abs_diff'], reverse=True)[: args.show_top]
    print('\nWorst frames')
    for item in worst_frames:
        print(
            f"frame={item['frame']} mean_abs_diff={item['mean_abs_diff']:.6f} "
            f"max_abs_diff={item['max_abs_diff']:.6f} l2={item['l2']:.6f}"
        )


if __name__ == '__main__':
    main()