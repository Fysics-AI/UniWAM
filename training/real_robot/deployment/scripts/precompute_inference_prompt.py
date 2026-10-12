#!/usr/bin/env python3
"""Encode the exact prefixed task string used by the cloud policy."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from omegaconf import OmegaConf


def full_instruction(config: object, task_prompt: str, robot_type: str) -> str:
    task = task_prompt.strip()
    if not task:
        raise ValueError("task_prompt must be non-empty")
    mode = str(config.get("inference_mode", "paired")).strip().lower()
    if robot_type.strip().lower() == "franka" and mode != "nav_only":
        prefix = str(config.get("instruction_prefix_by_robot", {}).get("franka", "")).strip()
    else:
        key = "nav_only" if mode == "nav_only" else "manip_only"
        prefix = str(config.get("instruction_prefix_by_mode", {}).get(key, "")).strip()
    return task if not prefix or task.startswith(prefix) else f"{prefix} {task}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-prompt", required=True)
    parser.add_argument("--robot-type", default="piper")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    instruction = full_instruction(config, args.task_prompt, args.robot_type)
    prompt = "A video recorded from a robot's point of view executing the following instruction: " + instruction
    training_root = Path(str(config.project_root))
    encoder = training_root / "data_pipeline/encode_six_source_prompt_cache.py"
    if not encoder.is_file():
        raise FileNotFoundError(encoder)
    with tempfile.TemporaryDirectory(prefix="uniwam_prompt_") as temporary:
        manifest = Path(temporary) / "prompt.jsonl"
        manifest.write_text(json.dumps({"prompt": prompt}, ensure_ascii=True) + "\n")
        subprocess.run(
            [sys.executable, str(encoder), "--manifest", str(manifest), "--output", str(args.output)],
            check=True,
        )
    print(f"UNIWAM_PROMPT_CACHE_READY task={args.task_prompt!r} mode={config.get('inference_mode')}")


if __name__ == "__main__":
    main()
