#!/usr/bin/env python3
"""Recompute the five released methods from per-trajectory metrics."""

import argparse
import csv
from pathlib import Path
from statistics import mean

METHODS = ('groot_n17', 'pi05', 'fastwam', 'uniwam_base', 'uniwam_scaled')
BINS = ('short_<1.5m', 'medium_1.5-3.5m', 'long_>=3.5m')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path,
                        default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    path = args.data_root / 'results/trajectory_metrics.csv'
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 15_000:
        raise RuntimeError(f'Expected 15,000 trajectories, got {len(rows)}')
    print('method,short_cm,mid_cm,long_cm,overall_macro_cm,'
          'short_deg,mid_deg,long_deg,overall_macro_deg')
    for method in METHODS:
        position, heading = [], []
        for distance_bin in BINS:
            sample = [row for row in rows if row['distance_bin'] == distance_bin]
            position.append(100 * mean(float(row[f'{method}_distance_error_m']) for row in sample))
            heading.append(mean(float(row[f'{method}_heading_error_deg']) for row in sample))
        values = position + [mean(position)] + heading + [mean(heading)]
        print(','.join([method] + [f'{value:.6f}' for value in values]))


if __name__ == '__main__':
    main()
