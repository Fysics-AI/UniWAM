#!/usr/bin/env python3
"""Validate a real checkpoint's train/deploy contract without loading weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from uniwam_piper_cloud.policy_rot6d import (
    AsyncPrefixContract,
    load_runtime_cfg,
    validate_checkpoint_contract,
    validate_deploy_settings,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    deploy = OmegaConf.load(args.config)
    runtime = load_runtime_cfg(
        Path(str(deploy.project_root)).resolve(), str(deploy.task),
        compile_action_infer=bool(deploy.get("compile_action_infer", False)),
        compile_vae_infer=bool(deploy.get("compile_vae_infer", False)),
    )
    contract = AsyncPrefixContract.from_deploy_cfg(deploy)
    validate_deploy_settings(deploy, contract)
    train_config = validate_checkpoint_contract(deploy, runtime, args.checkpoint, contract)
    print(json.dumps({
        "status": "UNIWAM_CLOUD_CONTRACT_OK",
        "task": str(deploy.task),
        "source_index": int(deploy.paired_source_index),
        "checkpoint": str(args.checkpoint),
        "train_config": str(train_config),
        "stats": str(deploy.dataset_stats),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
