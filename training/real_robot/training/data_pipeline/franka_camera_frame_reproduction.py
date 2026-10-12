#!/usr/bin/env python3
"""Audit and reproduce Franka camera-frame columns from episode transforms.

The derive mode fits one rigid transform per arm/episode to a known prepared
reference. The resulting calibration manifest is an external data artifact,
not part of the public source release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


STATE_COLUMN = "observation.state.camera_dual_arm"
ACTION_COLUMN = "action.manip.camera_dual_arm"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def episode_paths(root: Path) -> list[Path]:
    paths = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode Parquet files in {root / 'data'}")
    return paths


def rot6d_matrix(values: np.ndarray) -> np.ndarray:
    rows = values.reshape(-1, 2, 3)
    first = rows[:, 0] / np.linalg.norm(rows[:, 0], axis=-1, keepdims=True)
    second = rows[:, 1] - np.sum(first * rows[:, 1], axis=-1, keepdims=True) * first
    second /= np.linalg.norm(second, axis=-1, keepdims=True)
    return np.stack((first, second, np.cross(first, second)), axis=-2)


def fit_transform(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    source_rotation = rot6d_matrix(source[:, 3:9])
    reference_rotation = rot6d_matrix(reference[:, 3:9])
    camera_rotations = reference_rotation @ np.swapaxes(source_rotation, -1, -2)
    left, _, right = np.linalg.svd(camera_rotations.mean(axis=0))
    orientation = left @ right
    if np.linalg.det(orientation) < 0:
        left[:, -1] *= -1
        orientation = left @ right
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = orientation
    transform[:3, 3] = np.mean(
        reference[:, :3] - source[:, :3] @ orientation.T, axis=0
    )
    return transform


def transform_arm(pose: np.ndarray, transform: np.ndarray) -> np.ndarray:
    rows = pose[:, 3:9].reshape(-1, 2, 3)
    first = rows[:, 0] / np.linalg.norm(rows[:, 0], axis=-1, keepdims=True)
    second = rows[:, 1] - np.sum(first * rows[:, 1], axis=-1, keepdims=True) * first
    second /= np.linalg.norm(second, axis=-1, keepdims=True)
    rotation = np.stack((first, second, np.cross(first, second)), axis=-2)
    result = np.empty_like(pose, dtype=np.float32)
    result[:, :3] = pose[:, :3] @ transform[:3, :3].T + transform[:3, 3]
    result[:, 3:9] = (transform[:3, :3] @ rotation)[:, :2, :].reshape(-1, 6)
    result[:, 9] = pose[:, 9]
    return result


def camera_values(absolute: np.ndarray, transforms: np.ndarray) -> np.ndarray:
    if absolute.ndim != 2 or absolute.shape[1] != 20 or transforms.shape != (2, 4, 4):
        raise ValueError(f"Expected [T,20] poses and [2,4,4] transforms, got {absolute.shape}/{transforms.shape}")
    return np.concatenate(
        (transform_arm(absolute[:, :10], transforms[0]),
         transform_arm(absolute[:, 10:], transforms[1])), axis=1
    )


def raw_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(path, columns=["observation.state", "action.manip"])
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    action = np.asarray(table["action.manip"].to_pylist(), dtype=np.float64)
    if state.shape != (table.num_rows, 23) or action.shape != (table.num_rows, 20):
        raise ValueError(f"Unexpected Franka raw state/action shapes in {path}")
    return state[:, 3:], action


def compare_values(
    source_path: Path, reference_path: Path, transforms: np.ndarray
) -> tuple[float, float]:
    state, action = raw_arrays(source_path)
    reference = pq.read_table(reference_path, columns=[STATE_COLUMN, ACTION_COLUMN])
    expected_state = np.asarray(reference[STATE_COLUMN].to_pylist(), dtype=np.float32)
    expected_action = np.asarray(reference[ACTION_COLUMN].to_pylist(), dtype=np.float32)
    if state.shape != expected_state.shape or action.shape != expected_action.shape:
        raise ValueError(f"Episode row count mismatch: {source_path}/{reference_path}")
    state_error = float(np.max(np.abs(camera_values(state, transforms) - expected_state)))
    action_error = float(np.max(np.abs(camera_values(action, transforms) - expected_action)))
    return state_error, action_error


def derive(source_root: Path, reference_root: Path, output: Path, tolerance: float) -> None:
    rows = []
    max_state_error = max_action_error = 0.0
    source_paths = episode_paths(source_root)
    for index, source_path in enumerate(source_paths):
        relative = source_path.relative_to(source_root / "data")
        reference_path = reference_root / "data" / relative
        if not reference_path.is_file():
            raise FileNotFoundError(reference_path)
        state, _ = raw_arrays(source_path)
        reference_state = np.asarray(
            pq.read_table(reference_path, columns=[STATE_COLUMN])[STATE_COLUMN].to_pylist(),
            dtype=np.float64,
        )
        if state.shape != reference_state.shape:
            raise ValueError(f"State shape mismatch in {relative}")
        transforms = np.stack([
            fit_transform(state[:, start:start + 10], reference_state[:, start:start + 10])
            for start in (0, 10)
        ])
        state_error, action_error = compare_values(source_path, reference_path, transforms)
        max_state_error = max(max_state_error, state_error)
        max_action_error = max(max_action_error, action_error)
        if max(state_error, action_error) > tolerance:
            raise ValueError(f"Camera-frame reproduction differs at {relative}: {state_error}/{action_error}")
        rows.append({
            "episode": str(relative),
            "frames": int(state.shape[0]),
            "source_sha256": sha256(source_path),
            "reference_sha256": sha256(reference_path),
            "camera_from_left_base": transforms[0].tolist(),
            "camera_from_right_base": transforms[1].tolist(),
            "max_state_error": state_error,
            "max_action_error": action_error,
        })
        if (index + 1) % 200 == 0:
            print(f"[franka-camera] verified {index + 1}/{len(source_paths)}", flush=True)
    result = {
        "status": "CAMERA_FRAME_VALUES_MATCH",
        "method": "per-episode rigid fit from known prepared parent values",
        "tolerance": tolerance,
        "source_root": str(source_root),
        "reference_root": str(reference_root),
        "source_episodes_sha256": sha256(source_root / "meta/episodes.jsonl"),
        "episode_count": len(rows),
        "max_state_error": max_state_error,
        "max_action_error": max_action_error,
        "episodes": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"CAMERA_FRAME_VALUES_MATCH episodes={len(rows)} state={max_state_error:.3g} action={max_action_error:.3g}")


def materialize(source_root: Path, manifest_path: Path, output_root: Path) -> None:
    staging = output_root.with_name(output_root.name + ".building")
    if output_root.exists() or staging.exists():
        raise FileExistsError(f"Refusing to replace {output_root} or {staging}")
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "CAMERA_FRAME_VALUES_MATCH":
        raise ValueError("Calibration manifest has not passed reference validation")
    if sha256(source_root / "meta/episodes.jsonl") != manifest["source_episodes_sha256"]:
        raise ValueError("Source episode metadata changed since transform derivation")
    entries = {row["episode"]: row for row in manifest["episodes"]}
    source_paths = episode_paths(source_root)
    if len(source_paths) != len(entries):
        raise ValueError("Source episode count changed")
    staging.mkdir(parents=True)
    shutil.copytree(source_root / "meta", staging / "meta", dirs_exist_ok=True)
    (staging / "videos").symlink_to(source_root / "videos", target_is_directory=True)
    for source_path in source_paths:
        relative = source_path.relative_to(source_root / "data")
        row = entries[str(relative)]
        if sha256(source_path) != row["source_sha256"]:
            raise ValueError(f"Source episode changed: {relative}")
        transforms = np.asarray(
            (row["camera_from_left_base"], row["camera_from_right_base"]),
            dtype=np.float64,
        )
        state, action = raw_arrays(source_path)
        table = pq.read_table(source_path)
        for name, values in (
            (STATE_COLUMN, camera_values(state, transforms)),
            (ACTION_COLUMN, camera_values(action, transforms)),
        ):
            table = table.append_column(name, pa.array(values.tolist(), type=pa.list_(pa.float32(), 20)))
        target = staging / "data" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, target, compression="zstd")
    info_path = staging / "meta/info.json"
    info = json.loads(info_path.read_text())
    names = [f"{side}_{feature}" for side in ("left", "right") for feature in (
        "x", "y", "z", "rot6d_r0c0", "rot6d_r0c1", "rot6d_r0c2",
        "rot6d_r1c0", "rot6d_r1c1", "rot6d_r1c2", "gripper",
    )]
    for key, fields in ((STATE_COLUMN, names), (ACTION_COLUMN, [f"{name}_target" for name in names])):
        info.setdefault("features", {})[key] = {"dtype": "float32", "shape": [20], "names": [fields]}
    info["uniwam_camera_frame_derivation"] = {
        "method": manifest["method"],
        "external_transform_manifest_sha256": sha256(manifest_path),
    }
    info_path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    os.replace(staging, output_root)
    print(f"FRANKA_CAMERA_FRAME_MATERIALIZED episodes={len(source_paths)} output={output_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    if args.reference_root and args.output_manifest and not args.manifest and not args.output_root:
        derive(args.source_root, args.reference_root, args.output_manifest, args.tolerance)
    elif args.manifest and args.output_root and not args.reference_root and not args.output_manifest:
        materialize(args.source_root, args.manifest, args.output_root)
    else:
        parser.error("Use --reference-root with --output-manifest, or --manifest with --output-root")


if __name__ == "__main__":
    main()
