#!/usr/bin/env python3
"""Fail closed when the six-source parent recipe uses different prepared data."""

from __future__ import annotations

import hashlib
import json
import os
import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
TASK = "uniwam_camera_frame_six_source_manip26_embodiment_stats_200k"
DATASET_NAMES = (
    "lerobot_camera_frame",
    "agx_cup_tray_mobile_rot6d_vlash_se2_h48_phase_split_v3_camera_frame",
    "agx_move_white_box_mobile_rot6d_vlash_se2_h48_phase_split_v2_camera_frame",
    "agx_color_blocks_mobile_rot6d_vlash_se2_h48_phase_split_v3_camera_frame",
    "agx_ordered_color_blocks_camera_frame_manip26_v1",
    "agilex7000_front30_camera_frame_manip26_v1",
)
DATASET_EPISODES = (943, 534, 476, 274, 187, 12150)
DATASET_FRAMES = (1372262, 566262, 574610, 434967, 144675, 10592781)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_recipe(cfg: object, reference: dict) -> list[dict]:
    train = cfg.data.train
    weights = [float(value) for value in train.source_sampling_weights]
    if weights != [float(value) for value in reference["source_sampling_weights"]]:
        raise ValueError(f"Source sampling differs from the parent run: {weights}")
    if len(train.datasets) != len(DATASET_NAMES):
        raise ValueError("The parent recipe requires exactly six sources")
    if cfg.data.norm_default_mode != "q01/q99" or bool(cfg.data.use_stepwise_action_norm):
        raise ValueError("The parent recipe requires fixed q01/q99 normalization")
    if int(cfg.model.proprio_dim) != 20 or int(cfg.model.action_dit_config.manip_action_dim) != 26:
        raise ValueError("The parent recipe requires 20D state and 26D manipulation action")

    results = []
    for index, (wrapper, expected_name, expected) in enumerate(
        zip(train.datasets, DATASET_NAMES, reference["sources"], strict=True)
    ):
        dataset = wrapper.dataset
        root = Path(str(dataset.dataset_dirs[0]))
        if root.name != expected_name:
            raise ValueError(
                f"Source {index} is {root.name!r}, expected the parent dataset {expected_name!r}"
            )
        if str(dataset.state_column) != "observation.state.camera_dual_arm" or str(
            dataset.manip_action_column
        ) != "action.manip.camera_dual_arm":
            raise ValueError(f"Source {index} changed camera-frame state/action columns")
        if int(dataset.manip_action_offset) != 1 or str(dataset.manip_relative_frame) != "fixed_reference":
            raise ValueError(f"Source {index} changed the action alignment contract")
        expected_stats = (
            "camera_frame_franka_only_q01q99.json" if index == 0
            else "camera_frame_piper_agx_only_q01q99.json"
        )
        if Path(str(dataset.pretrained_norm_stats)).name != expected_stats:
            raise ValueError(f"Source {index} selected the wrong embodiment stats")
        if str(dataset.manip_eef_xy_source_name) != str(expected["sidecar_source"]):
            raise ValueError(f"Source {index} selected the wrong EEF sidecar")
        results.append({
            "source": expected["name"], "dataset_root": root,
            "window_index": Path(str(dataset.window_index_path)),
            "phase_index": Path(str(wrapper.phase_index_path)) if wrapper.get("phase_index_path") else None,
            "stats": Path(str(dataset.pretrained_norm_stats)),
            "expected": expected,
        })
    return results


def check_files(rows: list[dict], reference: dict) -> None:
    stats_hashes = reference["stats_sha256"]
    for index, row in enumerate(rows):
        for name, path, digest in (
            ("window index", row["window_index"], row["expected"]["window_sha256"]),
            ("phase index", row["phase_index"], row["expected"].get("phase_sha256")),
        ):
            if digest is not None and (path is None or sha256(path) != digest):
                raise ValueError(f"Source {index} {name} differs from the parent reference: {path}")
        group = "franka" if index == 0 else "piper_agx"
        if sha256(row["stats"]) != stats_hashes[group]:
            raise ValueError(f"Source {index} stats differ from the parent reference: {row['stats']}")
    prompt = ROOT / "data_indices/six_source_prompts.jsonl"
    if sha256(prompt) != reference["prompt_manifest"]["sha256"]:
        raise ValueError("Prompt manifest differs from the parent reference")
    for index, row in enumerate(rows):
        root = row["dataset_root"]
        episodes = root / "meta/episodes.jsonl"
        if not (root / "meta/info.json").is_file() or not episodes.is_file():
            raise FileNotFoundError(f"Parent source {index} is unavailable: {root}")
        episode_rows = [json.loads(line) for line in episodes.read_text().splitlines() if line.strip()]
        if not episode_rows:
            raise ValueError(f"Parent source {index} has no episodes: {root}")
        frame_count = sum(int(item["length"]) for item in episode_rows)
        if len(episode_rows) != DATASET_EPISODES[index] or frame_count != DATASET_FRAMES[index]:
            raise ValueError(
                f"Source {index} has {len(episode_rows)} episodes/{frame_count} frames; "
                f"parent requires {DATASET_EPISODES[index]}/{DATASET_FRAMES[index]}"
            )
        first_episode = int(episode_rows[0]["episode_index"])
        first_parquet = root / f"data/chunk-{first_episode // 1000:03d}/episode_{first_episode:06d}.parquet"
        if not first_parquet.is_file():
            raise FileNotFoundError(f"Parent source {index} has no first episode: {first_parquet}")


def check_recomputed_stats(path: Path, rows: list[dict]) -> None:
    expected = json.loads((ROOT / "data_indices/camera_frame_piper_agx_only_q01q99.json").read_text())
    actual = json.loads(path.read_text())
    for key in ("state", "action", "contract"):
        if actual.get(key) != expected.get(key):
            raise ValueError(f"Recomputed Piper {key} stats differ from the six-source parent: {path}")
    audit = actual.get("audit", [])
    if len(audit) != 5:
        raise ValueError("Recomputed Piper stats must include exactly five mobile sources")
    for source, row in zip(audit, rows[1:], strict=True):
        window = row["phase_index"] or row["window_index"]
        if Path(str(source["root"])).resolve() != row["dataset_root"].resolve() or Path(
            str(source["window_index"])
        ).resolve() != window.resolve():
            raise ValueError(f"Recomputed Piper stats were not built from {row['source']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recomputed-stats", type=Path)
    args = parser.parse_args()
    if Path(os.environ["UNIWAM_TRAINING_ROOT"]).resolve() != ROOT:
        raise ValueError("UNIWAM_TRAINING_ROOT must point to this training directory")
    reference_path = ROOT / "data_pipeline/parent_200k_artifact_reference.json"
    reference = json.loads(reference_path.read_text())
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="train", overrides=[f"task={TASK}"])
    OmegaConf.resolve(cfg)
    rows = check_recipe(cfg, reference)
    check_files(rows, reference)
    if args.recomputed_stats is not None:
        check_recomputed_stats(args.recomputed_stats, rows)
    print(json.dumps({
        "status": "UNIWAM_PARENT_REPRO_PREFLIGHT_OK",
        "source_count": len(rows),
        "sampling_weights": list(cfg.data.train.source_sampling_weights),
        "reference_run": reference["parent_run"],
        "stats_reproduced": args.recomputed_stats is not None,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
