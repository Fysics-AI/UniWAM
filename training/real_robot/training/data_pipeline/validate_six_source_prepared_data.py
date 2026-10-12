#!/usr/bin/env python3
"""Validate six-source structure and, optionally, exact parent artifact identity.

This does not compare episode parquet values or VAE latent contents. A parent
artifact match is necessary, but not sufficient, for full data reproduction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")


def read_episode_lengths(root: Path) -> dict[int, int]:
    episodes_path = root / "meta/episodes.jsonl"
    require_file(episodes_path, "episodes.jsonl")
    rows = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
    lengths = {int(row["episode_index"]): int(row["length"]) for row in rows}
    if not lengths:
        raise ValueError(f"No episodes in {episodes_path}")
    return lengths


def validate_source(source: dict, horizon: int) -> dict:
    name = str(source["name"])
    root = Path(source["dataset_root"])
    window_path = Path(source["window_index"])
    phase_path = Path(source["phase_index"]) if source.get("phase_index") else None
    require_file(root / "meta/info.json", f"{name} info")
    require_file(window_path, f"{name} window index")
    lengths = read_episode_lengths(root)
    window = pq.read_table(window_path).to_pydict()
    episode = np.asarray(window["episode_index"], dtype=np.int64)
    start = np.asarray(window["start_frame"], dtype=np.int64)
    horizons = np.asarray(window["horizon"], dtype=np.int64)
    if not np.all(horizons == horizon):
        raise ValueError(f"{name}: window horizon is {sorted(set(horizons.tolist()))}")
    if "source_episode_index" not in window:
        raise ValueError(f"{name}: source_episode_index column is required")
    if not all(int(ep) in lengths for ep in episode):
        raise ValueError(f"{name}: window references an unknown episode")
    valid = start + horizon <= np.asarray([lengths[int(ep)] for ep in episode])
    if not np.all(valid):
        bad = int(np.flatnonzero(~valid)[0])
        raise ValueError(f"{name}: window crosses episode boundary at row {bad}")

    phase_rows = 0
    phase_sha256 = None
    branch_counts: dict[str, int] = {}
    if phase_path is not None:
        require_file(phase_path, f"{name} phase index")
        phase = pq.read_table(phase_path).to_pydict()
        phase_source = np.asarray(phase["source_window_index"], dtype=np.int64)
        if phase_source.size and (phase_source.min() < 0 or phase_source.max() >= episode.size):
            raise ValueError(f"{name}: phase source_window_index is out of bounds")
        phase_episode = np.asarray(phase["episode_index"], dtype=np.int64)
        phase_start = np.asarray(phase["start_frame"], dtype=np.int64)
        if not np.array_equal(phase_episode, episode[phase_source]):
            raise ValueError(f"{name}: phase episode keys do not match window index")
        if not np.array_equal(phase_start, start[phase_source]):
            raise ValueError(f"{name}: phase frame keys do not match window index")
        if not np.all(np.asarray(phase["horizon"], dtype=np.int64) == horizon):
            raise ValueError(f"{name}: phase horizon is not H{horizon}")
        branches = [str(value) for value in phase["branch"]]
        branch_counts = {branch: branches.count(branch) for branch in sorted(set(branches))}
        if set(branch_counts) - {"manip", "nav"}:
            raise ValueError(f"{name}: unexpected phase branch {sorted(set(branch_counts))}")
        phase_rows = len(branches)
        phase_sha256 = sha256(phase_path)

    sidecar_report = None
    if source.get("eef_sidecar"):
        sidecar = Path(source["eef_sidecar"])
        require_file(sidecar / "manifest.json", f"{name} EEF sidecar manifest")
        manifest = json.loads((sidecar / "manifest.json").read_text())
        expected_files = ("eef_xy_visible_normalized.npy", "feature_mask.npy")
        file_hashes = {}
        for filename in expected_files:
            path = sidecar / filename
            require_file(path, f"{name} EEF sidecar {filename}")
            file_hashes[filename] = sha256(path)
            if manifest.get("files", {}).get(filename) != file_hashes[filename]:
                raise ValueError(f"{name}: EEF sidecar {filename} differs from its manifest")
        frames = int(manifest["total_frames"])
        values = np.load(sidecar / expected_files[0], mmap_mode="r")
        mask = np.load(sidecar / expected_files[1], mmap_mode="r")
        if values.shape != (frames, 6) or values.dtype != np.float32:
            raise ValueError(f"{name}: expected EEF values [frames, 6] float32, got {values.shape}/{values.dtype}")
        if mask.shape != (frames, 6) or mask.dtype != np.bool_:
            raise ValueError(f"{name}: expected EEF mask [frames, 6] bool, got {mask.shape}/{mask.dtype}")
        if frames != sum(lengths.values()):
            raise ValueError(f"{name}: EEF sidecar frames do not match dataset episodes")
        sidecar_report = {
            "source_name": manifest.get("source_name"),
            "total_frames": frames,
            "manifest_sha256": sha256(sidecar / "manifest.json"),
            "files": file_hashes,
        }

    return {
        "name": name,
        "dataset_root": str(root),
        "window_index": str(window_path),
        "window_sha256": sha256(window_path),
        "windows": int(episode.size),
        "phase_index": str(phase_path) if phase_path else None,
        "phase_rows": phase_rows,
        "phase_sha256": phase_sha256,
        "branch_counts": branch_counts,
        "episode_count": len(lengths),
        "eef_sidecar": sidecar_report,
    }


def parent_mismatches(
    report: dict, reference: dict, index_root: Path | None, stats_root: Path
) -> tuple[list[str], list[str]]:
    mismatches = []
    storage_differences = []

    def check(label: str, actual: object, expected: object) -> None:
        if actual != expected:
            mismatches.append(f"{label}: got={actual!r}, expected={expected!r}")

    check("source_sampling_weights", report["source_sampling_weights"], reference["source_sampling_weights"])
    actual_sources = report["sources"]
    expected_sources = reference["sources"]
    check("source_count", len(actual_sources), len(expected_sources))
    for actual, expected in zip(actual_sources, expected_sources):
        name = expected["name"]
        for key in ("name", "windows", "phase_rows"):
            check(f"{name}.{key}", actual[key], expected[key])
        for label, candidate_path, candidate_sha, expected_sha, filename in (
            ("window", actual["window_index"], actual["window_sha256"], expected["window_sha256"], "window_index.parquet"),
            ("phase", actual["phase_index"], actual["phase_sha256"], expected.get("phase_sha256"), "phase_index.parquet"),
        ):
            if expected_sha is None:
                check(f"{name}.{label}_sha256", candidate_sha, None)
                continue
            if index_root is None or candidate_sha == expected_sha:
                check(f"{name}.{label}_sha256", candidate_sha, expected_sha)
                continue
            reference_path = index_root / name / filename
            require_file(reference_path, f"parent {name} {label} index")
            if sha256(reference_path) != expected_sha:
                raise ValueError(f"Parent reference index has changed: {reference_path}")
            if candidate_path is None or not pq.read_table(candidate_path).equals(
                pq.read_table(reference_path), check_metadata=True
            ):
                mismatches.append(f"{name}.{label}: parquet rows/schema differ from the parent reference")
            else:
                storage_differences.append(f"{name}.{label}: rows/schema equal; parquet bytes differ")
        sidecar = actual["eef_sidecar"]
        if sidecar is None:
            mismatches.append(f"{name}.eef_sidecar: missing")
            continue
        check(f"{name}.sidecar_source", sidecar["source_name"], expected["sidecar_source"])
        check(f"{name}.sidecar_frames", sidecar["total_frames"], expected["sidecar_frames"])
        for filename, digest in expected["sidecar_files"].items():
            check(f"{name}.sidecar_files.{filename}", sidecar["files"].get(filename), digest)
    for name, digest in reference["stats_sha256"].items():
        candidate = report["stats"].get(name, {})
        candidate_hash = candidate.get("sha256")
        if candidate_hash == digest:
            continue
        filename = (
            "camera_frame_franka_only_q01q99.json"
            if name == "franka" else "camera_frame_piper_agx_only_q01q99.json"
        )
        local_reference = stats_root / filename
        require_file(local_reference, f"parent {name} stats reference")
        if sha256(local_reference) != digest:
            raise ValueError(f"Parent stats reference has changed: {local_reference}")
        try:
            actual_payload = json.loads(Path(candidate["path"]).read_text())
            expected_payload = json.loads(local_reference.read_text())
        except (KeyError, FileNotFoundError) as exc:
            mismatches.append(f"stats.{name}: missing or unreadable")
            continue
        fields = ("state", "action", "contract")
        if any(actual_payload.get(field) != expected_payload.get(field) for field in fields):
            mismatches.append(f"stats.{name}: state/action/contract differ from parent reference")
        else:
            storage_differences.append(f"stats.{name}: contract equal; provenance bytes differ")
    for key in ("count", "sha256"):
        check(f"prompt_manifest.{key}", report["prompt_manifest"][key], reference["prompt_manifest"][key])
    return mismatches, storage_differences


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--parent-reference", type=Path)
    parser.add_argument("--parent-index-root", type=Path)
    parser.add_argument(
        "--parent-stats-root", type=Path,
        default=Path(__file__).resolve().parents[1] / "data_indices",
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    for key in ("sources", "stats", "prompt_manifest"):
        if key not in manifest:
            raise ValueError(f"Manifest missing {key!r}")
    reports = [validate_source(source, args.horizon) for source in manifest["sources"]]
    if len(reports) != 6:
        raise ValueError(f"Expected six sources, got {len(reports)}")
    stats = {}
    for name, path in manifest["stats"].items():
        stats[name] = {"path": str(path), "sha256": sha256(Path(path))}
    prompt_path = Path(manifest["prompt_manifest"])
    require_file(prompt_path, "prompt manifest")
    prompt_count = sum(1 for line in prompt_path.read_text().splitlines() if line.strip())
    output = {
        "status": "STRUCTURAL_PASS",
        "horizon": args.horizon,
        "sources": reports,
        "stats": stats,
        "prompt_manifest": {"path": str(prompt_path), "count": prompt_count, "sha256": sha256(prompt_path)},
        "source_sampling_weights": manifest.get("source_sampling_weights"),
        "episode_parquet_values_verified": False,
        "vae_latent_contents_verified": False,
    }
    if args.parent_reference:
        reference = json.loads(args.parent_reference.read_text())
        output["parent_run"] = reference["parent_run"]
        output["parent_mismatches"], output["parquet_storage_differences"] = parent_mismatches(
            output, reference, args.parent_index_root, args.parent_stats_root
        )
        output["status"] = "PARENT_ARTIFACT_MATCH" if not output["parent_mismatches"] else "PARENT_ARTIFACT_MISMATCH"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return int(output["status"] == "PARENT_ARTIFACT_MISMATCH")


if __name__ == "__main__":
    raise SystemExit(main())
