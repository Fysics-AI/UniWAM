#!/usr/bin/env python3
"""Build the fail-closed manifest for the new camera-frame manip26 run."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validation_path(
    source_dataset: str, source_episode_index: int, pipeline_root: Path, source_root: Path
) -> Path:
    source = Path(source_dataset)
    try:
        relative = source.relative_to(source_root)
    except ValueError as exc:
        raise ValueError(f"Source dataset is outside expected AGILEX root: {source}")
    dataset = pipeline_root / relative
    episode = int(source_episode_index)
    return (
        dataset
        / "videos"
        / f"chunk-{episode // 1000:03d}"
        / f"episode_{episode:06d}"
        / "09_validation"
        / "lerobot_data"
        / f"episode_{episode:06d}.parquet"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--screening-root", type=Path, required=True)
    parser.add_argument("--source-map", type=Path, required=True)
    parser.add_argument("--source-dataset-root", type=Path, required=True)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    episodes = read_jsonl(args.episodes)
    source_map = {int(row["episode_index"]): row for row in read_jsonl(args.source_map)}
    screening = args.screening_root
    force_rows = read_jsonl(screening / "force_simple_episode_indices.jsonl")
    failed_rows = read_jsonl(screening / "failed_trajectory_exclusions.jsonl")
    force_keys = {
        (int(row["source_dataset"]), int(row["source_episode_index"]))
        for row in force_rows
        if row.get("force_simple") is True and row.get("label") == "simple_only"
    }
    failed_keys = {
        (int(row["source_dataset"]), int(row["source_episode_index"]))
        for row in failed_rows
    }
    rows: list[dict] = []
    counts: Counter[str] = Counter()
    for row in episodes:
        episode_index = int(row["episode_index"])
        source = source_map.get(episode_index)
        if source is None:
            raise KeyError(f"Missing source map row for episode_index={episode_index}")
        key = (int(row["source_dataset_index"]), int(row["source_episode_index"]))
        if int(source["source_episode_index"]) != key[1]:
            raise ValueError(f"Source-map episode mismatch for {episode_index}")
        val = validation_path(
            source["source_dataset"], key[1], args.pipeline_root, args.source_dataset_root
        )
        excluded = bool(row.get("training_excluded", False))
        failed = key in failed_keys
        forced_simple = key in force_keys
        has_validation = val.is_file()
        eligible = not excluded and not failed and has_validation
        reason = "eligible"
        if excluded:
            reason = "existing_training_excluded"
        elif failed:
            reason = "prompt_screening_failed_or_wrong_task"
        elif not has_validation:
            reason = "missing_09_validation_parquet"
        counts[reason] += 1
        rows.append(
            {
                "episode_index": int(row["episode_index"]),
                "source_dataset_index": key[0],
                "source_episode_index": key[1],
                "source_dataset": source["source_dataset"],
                "source_episode_name": source.get("source_episode_name"),
                "validation_parquet": str(val),
                "eligible": eligible,
                "filter_reason": reason,
                "prompt_mode": "forced_simple" if forced_simple else "balanced_simple_or_detailed",
            }
        )
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8"
    )
    report = {
        "status": "PASS",
        "source": str(args.episodes),
        "episode_count": len(rows),
        "force_simple_screening_rows": len(force_rows),
        "failed_screening_rows": len(failed_rows),
        "counts": dict(counts),
        "eligible_episode_count": sum(bool(row["eligible"]) for row in rows),
        "forced_simple_eligible_count": sum(
            bool(row["eligible"]) and row["prompt_mode"] == "forced_simple" for row in rows
        ),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
