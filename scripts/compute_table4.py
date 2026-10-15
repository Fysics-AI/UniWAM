#!/usr/bin/env python3
"""Recompute MAP-Bench position and heading errors from released rollouts."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
import tarfile

import zstandard

METHODS = ('groot_n17', 'pi05', 'fastwam', 'uniwam_base', 'uniwam_scaled')
BINS = ('short_<1.5m', 'medium_1.5-3.5m', 'long_>=3.5m')


def load_bins(data_root: Path) -> dict[str, str]:
    with (data_root / 'evaluation/trajectory_ids.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    bins = {row['trajectory_id']: row['distance_bin'] for row in rows}
    if len(rows) != 15_000 or len(bins) != len(rows):
        raise ValueError('Expected 15,000 unique trajectory IDs')
    if set(bins.values()) != set(BINS):
        raise ValueError('Unexpected distance level in trajectory IDs')
    return bins


def load_errors(data_root: Path, method: str, bins: dict[str, str]) -> dict[str, list[tuple[float, float]]]:
    archive = data_root / 'archives' / f'{method}.tar.zst'
    trace_dir = Path('results/raw_traces') / method
    errors = {distance_bin: [] for distance_bin in BINS}
    seen = set()
    with archive.open('rb') as source, zstandard.ZstdDecompressor().stream_reader(source) as stream:
        with tarfile.open(fileobj=stream, mode='r|') as traces:
            for member in traces:
                if member.isdir():
                    continue
                path = Path(member.name)
                if not member.isfile() or path.parent != trace_dir or path.suffix != '.json':
                    raise ValueError(f'Unexpected archive member: {member.name}')
                with traces.extractfile(member) as item:
                    trace = json.load(item)
                identity = trace['identity']
                if identity not in bins or identity in seen:
                    raise ValueError(f'Unknown or duplicate {method} trajectory: {identity}')
                seen.add(identity)
                position = float(trace['final_operating_point_distance_m']) * 100
                heading = float(trace['final_heading_deviation_deg'])
                if not math.isfinite(position) or not math.isfinite(heading):
                    raise ValueError(f'Non-finite {method} error: {identity}')
                errors[bins[identity]].append((position, heading))
    if seen != bins.keys():
        raise ValueError(f'{method} has {len(seen)} of {len(bins)} trajectories')
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path,
                        default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    bins = load_bins(args.data_root)
    print('method,short_cm,mid_cm,long_cm,overall_macro_cm,'
          'short_deg,mid_deg,long_deg,overall_macro_deg')
    for method in METHODS:
        errors = load_errors(args.data_root, method, bins)
        position = [mean(row[0] for row in errors[distance_bin]) for distance_bin in BINS]
        heading = [mean(row[1] for row in errors[distance_bin]) for distance_bin in BINS]
        values = position + [mean(position)] + heading + [mean(heading)]
        print(','.join([method] + [f'{value:.6f}' for value in values]))


if __name__ == '__main__':
    main()
