#!/usr/bin/env python3
"""Read-only full audit for the 60 shared-E AgileX7000 geometry outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import backfill_agilex7000_shared_geometry as backfill
import batch_project_agilex7000_geometry as core


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=core.DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--override", type=Path, default=backfill.DEFAULT_OVERRIDE)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = (args.output_root.resolve() if args.output_root else
                   (core.DEFAULT_OUTPUT_PARENT / dataset_root.name).resolve())
    info, total_episodes, chunk_size, width, height = core.read_info(dataset_root)
    override = backfill.load_override(args.override.resolve(), dataset_root.name)
    start = int(override["episode_range"]["start_inclusive"])
    stop = int(override["episode_range"]["stop_exclusive"])
    indices = list(range(start, stop))
    provenance = core.read_provenance(dataset_root, total_episodes)
    validations = [
        backfill.validate_output(
            provenance[index], dataset_root, output_root, chunk_size, width, height, override,
        )
        for index in indices
    ]
    manifest_path = output_root / "geometry_manifest.jsonl"
    manifest = core.load_existing_manifest(manifest_path)
    manifest_errors: list[str] = []
    if len(manifest) != total_episodes:
        manifest_errors.append(f"manifest entries={len(manifest)} expected={total_episodes}")
    for index in indices:
        row = manifest.get(index, {})
        expected = {
            "calibration_status": "valid",
            "calibration_method": override["calibration_method"],
            "calibration_evidence": override["evidence"]["initializer_report"],
            "calibration_override_version": override["override_version"],
            "status": "success",
        }
        for key, value in expected.items():
            if row.get(key) != value:
                manifest_errors.append(f"episode {index} manifest {key} mismatch")
        for key in ("geometry_npz", "geometry_report", "video", "source_parquet"):
            path = Path(str(row.get(key, "")))
            if not path.is_file():
                manifest_errors.append(f"episode {index} manifest {key} is absent")
    invalid = [item for item in validations if not item["valid"]]
    status_counts = Counter(row.get("calibration_status") for row in manifest.values())
    payload = {
        "validator": "agilex7000-shared-e-readonly-validation-v1",
        "status": "success" if not invalid and not manifest_errors else "failed",
        "dataset": str(dataset_root), "output_root": str(output_root),
        "override": {"path": override["_path"], "sha256": override["_sha256"]},
        "episodes_expected": len(indices), "episodes_valid": len(validations) - len(invalid),
        "frames_validated": int(sum(item["frames"] for item in validations if item["valid"])),
        "invalid_outputs": invalid, "manifest_errors": manifest_errors,
        "manifest_entries": len(manifest), "manifest_status_counts": dict(status_counts),
        "manifest_sha256": backfill.sha256_file(manifest_path),
        "shared_calibration_signature": backfill.calibration_bundle(override).signatures["combined"],
    }
    report_path = (args.report.resolve() if args.report else
                   output_root / "shared_right_base_invariant_backfill.validation.json")
    core.atomic_write_json(report_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
