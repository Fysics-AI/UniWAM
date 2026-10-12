#!/usr/bin/env python3
"""Check customer Piper SFT inputs and decode one real training sample."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from uniwam.datasets.mixed_stream import MixedStreamCollator


ROOT = Path(__file__).resolve().parents[1]
TASK = "uniwam_customer_piper_manip_sft"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_parent(checkpoint: Path) -> None:
    if checkpoint.parent.name != "weights" or checkpoint.parent.parent.name != "checkpoints":
        raise ValueError("Initialization checkpoint must be under RUN/checkpoints/weights/")
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise ValueError("Initialization checkpoint is missing or empty")
    train_config = checkpoint.parents[2] / "config.yaml"
    if not train_config.is_file():
        raise FileNotFoundError(f"Missing checkpoint config: {train_config}")
    parent = OmegaConf.load(train_config)
    model = parent.model
    if int(model.proprio_dim) != 20 or int(model.action_dit_config.manip_action_dim) != 26:
        raise ValueError("Initialization checkpoint is not a 20D/26D UniWAM model")
    datasets = parent.data.train.datasets
    if not datasets:
        raise ValueError("Initialization checkpoint contains no training dataset contract")
    stats_paths = [
        Path(str(item.dataset.pretrained_norm_stats))
        for item in datasets
        if item.get("dataset", {}).get("pretrained_norm_stats") is not None
    ]
    if not stats_paths or any(not path.is_file() for path in stats_paths):
        raise ValueError("Initialization checkpoint refers to unavailable normalization statistics")


def main() -> None:
    if Path(os.environ["UNIWAM_TRAINING_ROOT"]).resolve() != ROOT:
        raise ValueError("UNIWAM_TRAINING_ROOT must point to this release's training directory")
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="train", overrides=[f"task={TASK}"])
    OmegaConf.resolve(cfg)
    check_parent(Path(str(cfg.resume)))
    data = cfg.data.train.datasets[0].dataset
    dataset_root = Path(str(data.dataset_dirs[0]))
    cache_source = Path(str(data.precomputed_latent_root)) / str(data.precomputed_latent_source_name)
    if not cache_source.is_dir():
        raise FileNotFoundError(f"Missing VAE latent source directory: {cache_source}")
    if data.get("manip_eef_xy_index_root") is not None or not data.append_missing_eef_xy_slots:
        raise ValueError("Customer SFT must mask the six auxiliary point-tracking slots")
    if float(cfg.model.loss.lambda_manip_aux_action) != 0:
        raise ValueError("Customer SFT must not supervise the point-tracking auxiliary loss")
    if not bool(cfg.model.disable_manip_aux_training):
        raise ValueError("Customer SFT must disable auxiliary training inputs and noise")
    dataset = instantiate(cfg.data.train)
    sample = dataset[0]
    collate = MixedStreamCollator(
        manip_action_dim=26, nav_action_dim=3, manip_horizon=32, nav_horizon=32,
    )
    batch = collate([sample])
    action = batch["manip_action"]
    mask = batch["manip_action_feature_mask"]
    if tuple(action.shape) != (1, 32, 26) or not bool(mask[..., :20].all()) or bool(mask[..., 20:].any()):
        raise ValueError("Customer action shape or point-tracking feature mask is wrong")
    action_abs_max = float(action[..., :20].abs().max())
    state_abs_max = float(batch["proprio"].abs().max())
    if action_abs_max >= 4.99 or state_abs_max >= 4.99:
        raise ValueError("First sample saturates the [-5,5] normalizer; check units and stats")
    print(json.dumps({
        "status": "CUSTOMER_SFT_INPUTS_OK", "windows": len(dataset),
        "source": str(dataset_root), "checkpoint": str(cfg.resume),
        "stats_sha256": sha256(Path(str(data.pretrained_norm_stats))),
        "first_action_shape": list(action.shape), "aux_supervised": False,
        "first_control_norm_abs_max": action_abs_max,
        "first_state_norm_abs_max": state_abs_max,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
