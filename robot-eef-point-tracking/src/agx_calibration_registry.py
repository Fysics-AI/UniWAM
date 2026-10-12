#!/usr/bin/env python3
"""Calibration selection for the four final AGX LeRobot datasets.

Selections are keyed by the *source* episode id recorded in each final
``meta/episodes.jsonl``.  Final episode ids are compacted after filtering and
must never be used to select an acquisition-period calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


DEFAULT_CALIBRATION_ROOT = Path(
    os.environ.get(
        "EEF_CALIBRATION_ROOT",
        Path(__file__).resolve().parent.parent / "configs" / "calibration_template",
    )
)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    dataset_name: str
    calibration_mode: str
    calibration_file: str | None = None
    manifest_file: str | None = None
    phase_file: str | None = (
        "meta/phase_split_vlash_shared_h48_v1/phase_split_episodes.jsonl"
    )
    provisional: bool = False
    provisional_reason: str | None = None


DATASET_SPECS: dict[str, DatasetSpec] = {
    "cup": DatasetSpec(
        key="cup",
        dataset_name=(
            "agx_cup_tray_mobile_rot6d_vlash_se2_h48_phase_split_v3_"
            "action_current_v1"
        ),
        calibration_mode="fixed",
        calibration_file="mobile_aloha_front_episode2_canary.yaml",
    ),
    "white": DatasetSpec(
        key="white",
        dataset_name=(
            "agx_move_white_box_mobile_rot6d_vlash_se2_h48_phase_split_v2_"
            "action_current_v1"
        ),
        calibration_mode="manifest",
        manifest_file="move_white_box_manifest.jsonl",
    ),
    "color": DatasetSpec(
        key="color",
        dataset_name=(
            "agx_color_blocks_mobile_rot6d_vlash_se2_h48_phase_split_v3_"
            "action_current_v1"
        ),
        calibration_mode="manifest",
        manifest_file="color_blocks_manifest.jsonl",
    ),
    "ordered": DatasetSpec(
        key="ordered",
        dataset_name=(
            "agx_ordered_color_blocks_rot6d_vlash_se2_h32_v2_"
            "action_current_v1"
        ),
        calibration_mode="fixed",
        calibration_file="ordered_color_blocks_front_dino_control_debiased_v1.yaml",
        phase_file=None,
    ),
}


@dataclass(frozen=True)
class Calibration:
    dataset_key: str
    source_episode_index: int
    calibration_id: str
    calibration_unit: str
    calibration_path: Path
    calibration_sha256: str
    intrinsic: np.ndarray
    extrinsic: np.ndarray
    yaml_status: str
    manifest_status: str | None
    selection_mode: str
    provisional: bool
    provisional_reason: str | None
    calibration_method: Any | None
    evidence: Any | None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            records.append(record)
    return records


def resolve_dataset_spec(value: str) -> DatasetSpec:
    if value in DATASET_SPECS:
        return DATASET_SPECS[value]
    for spec in DATASET_SPECS.values():
        if value == spec.dataset_name:
            return spec
    choices = ", ".join(DATASET_SPECS)
    raise KeyError(f"unknown AGX dataset {value!r}; expected one of: {choices}")


class CalibrationRegistry:
    """Validated calibration lookup indexed by dataset and source episode."""

    def __init__(self, calibration_root: Path = DEFAULT_CALIBRATION_ROOT) -> None:
        self.calibration_root = Path(calibration_root).resolve()
        if not self.calibration_root.is_dir():
            raise FileNotFoundError(self.calibration_root)
        self._yaml_cache: dict[Path, tuple[dict[str, Any], str]] = {}
        self._fixed: dict[str, tuple[Path, str]] = {}
        self._manifest: dict[str, dict[int, tuple[Path, dict[str, Any]]]] = {}
        self._load()

    def _load(self) -> None:
        for spec in DATASET_SPECS.values():
            if spec.calibration_mode == "fixed":
                assert spec.calibration_file is not None
                path = self._resolve_calibration_path(
                    spec.calibration_file, self.calibration_root
                )
                self._load_yaml(path)
                self._fixed[spec.key] = (path, f"{spec.key}_fixed")
                continue

            if spec.calibration_mode != "manifest" or spec.manifest_file is None:
                raise ValueError(f"invalid calibration configuration for {spec.key}")
            manifest_path = self.calibration_root / spec.manifest_file
            if not manifest_path.is_file():
                raise FileNotFoundError(manifest_path)
            source_map: dict[int, tuple[Path, dict[str, Any]]] = {}
            for row_number, row in enumerate(read_jsonl(manifest_path), start=1):
                indices = row.get("source_episode_indices")
                if not isinstance(indices, list) or not indices:
                    raise ValueError(
                        f"{manifest_path}:{row_number} has no source_episode_indices"
                    )
                declared_count = row.get("source_episode_count")
                if declared_count is not None and int(declared_count) != len(indices):
                    raise ValueError(
                        f"{manifest_path}:{row_number} source_episode_count mismatch"
                    )
                calibration_value = row.get("calibration")
                if not isinstance(calibration_value, str):
                    raise ValueError(
                        f"{manifest_path}:{row_number} has no calibration path"
                    )
                calibration_path = self._resolve_calibration_path(
                    calibration_value, manifest_path.parent
                )
                yaml_data, _ = self._load_yaml(calibration_path)
                manifest_id = row.get("calibration_id")
                if manifest_id is not None and manifest_id != yaml_data["calibration_id"]:
                    raise ValueError(
                        f"calibration id mismatch in {manifest_path}:{row_number}: "
                        f"{manifest_id!r} != {yaml_data['calibration_id']!r}"
                    )
                for raw_index in indices:
                    source_index = int(raw_index)
                    if source_index in source_map:
                        raise ValueError(
                            f"source episode {source_index} occurs more than once in "
                            f"{manifest_path}"
                        )
                    source_map[source_index] = (calibration_path, row)
            self._manifest[spec.key] = source_map

    def _resolve_calibration_path(self, value: str, relative_to: Path) -> Path:
        supplied = Path(value)
        candidates: Iterable[Path]
        if supplied.is_absolute():
            candidates = (supplied,)
        else:
            candidates = (
                relative_to / supplied,
                relative_to.parent / supplied,
                self.calibration_root / supplied,
                self.calibration_root / supplied.name,
            )
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(
            f"cannot resolve calibration {value!r} relative to {relative_to}"
        )

    def _load_yaml(self, path: Path) -> tuple[dict[str, Any], str]:
        cached = self._yaml_cache.get(path)
        if cached is not None:
            return cached
        payload = path.read_bytes()
        data = yaml.safe_load(payload)
        if not isinstance(data, dict):
            raise ValueError(f"calibration is not a mapping: {path}")
        if not isinstance(data.get("calibration_id"), str):
            raise ValueError(f"calibration_id is missing from {path}")
        intrinsic = np.asarray(data.get("intrinsic", {}).get("matrix"), dtype=np.float64)
        extrinsic = np.asarray(data.get("extrinsic", {}).get("matrix"), dtype=np.float64)
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise ValueError(f"invalid 3x3 intrinsic in {path}")
        if extrinsic.shape != (4, 4) or not np.isfinite(extrinsic).all():
            raise ValueError(f"invalid 4x4 extrinsic in {path}")
        direction = data.get("extrinsic", {}).get("direction")
        if direction != "footprint_to_camera":
            raise ValueError(
                f"unsupported extrinsic direction {direction!r} in {path}"
            )
        digest = hashlib.sha256(payload).hexdigest()
        self._yaml_cache[path] = (data, digest)
        return data, digest

    def select(self, dataset: str, source_episode_index: int) -> Calibration:
        spec = resolve_dataset_spec(dataset)
        source_episode_index = int(source_episode_index)
        manifest_row: dict[str, Any] | None = None
        if spec.calibration_mode == "fixed":
            calibration_path, unit = self._fixed[spec.key]
        else:
            try:
                calibration_path, manifest_row = self._manifest[spec.key][
                    source_episode_index
                ]
            except KeyError as error:
                raise KeyError(
                    f"{spec.key}: source episode {source_episode_index} is absent "
                    "from the calibration manifest"
                ) from error
            unit = str(manifest_row.get("calibration_unit", "manifest_row"))

        yaml_data, digest = self._load_yaml(calibration_path)
        return Calibration(
            dataset_key=spec.key,
            source_episode_index=source_episode_index,
            calibration_id=str(yaml_data["calibration_id"]),
            calibration_unit=unit,
            calibration_path=calibration_path,
            calibration_sha256=digest,
            intrinsic=np.asarray(yaml_data["intrinsic"]["matrix"], dtype=np.float64),
            extrinsic=np.asarray(yaml_data["extrinsic"]["matrix"], dtype=np.float64),
            yaml_status=str(yaml_data.get("status", "UNKNOWN")),
            manifest_status=(
                str(manifest_row.get("status", "UNKNOWN"))
                if manifest_row is not None
                else None
            ),
            selection_mode=spec.calibration_mode,
            provisional=spec.provisional,
            provisional_reason=spec.provisional_reason,
            calibration_method=yaml_data.get("calibration_method"),
            evidence=yaml_data.get("evidence"),
        )

    def audit_final_episodes(self, dataset: str, dataset_root: Path) -> dict[str, Any]:
        spec = resolve_dataset_spec(dataset)
        episodes_path = Path(dataset_root) / "meta" / "episodes.jsonl"
        episodes = read_jsonl(episodes_path)
        seen_final: set[int] = set()
        calibration_counts: dict[str, int] = {}
        provisional_count = 0
        for row in episodes:
            if "source_episode_index" not in row:
                raise ValueError(f"source_episode_index missing in {episodes_path}")
            final_index = int(row["episode_index"])
            if final_index in seen_final:
                raise ValueError(f"duplicate final episode {final_index} in {episodes_path}")
            seen_final.add(final_index)
            calibration = self.select(spec.key, int(row["source_episode_index"]))
            calibration_counts[calibration.calibration_id] = (
                calibration_counts.get(calibration.calibration_id, 0) + 1
            )
            provisional_count += int(calibration.provisional)
        return {
            "dataset_key": spec.key,
            "dataset_name": spec.dataset_name,
            "episodes": len(episodes),
            "calibration_counts": calibration_counts,
            "provisional_episodes": provisional_count,
            "episodes_metadata": str(episodes_path),
        }
