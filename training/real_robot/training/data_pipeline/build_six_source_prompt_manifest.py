#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
MANIP_PREFIX = "Two black Piper arms are mounted on the near edge of a mobile AgileX platform. A fixed D435 main camera above and between the arms looks forward in their reaching direction, away from the robot; two D435 wrist views are shown. Express dual-arm end-effector states and actions in this fixed camera frame."
NAV_PREFIX = "Two black Piper arms are mounted on a mobile AgileX platform. A front D455 navigation camera looks forward in the travel direction, away from the arms; two D435 wrist views are shown, but the manipulation main view is absent. Navigation actions are relative planar motion."
FRANKA_PREFIX = "Two white Franka arms stand across the table from a fixed D455 camera. Express end-effector states and actions in this fixed camera frame."


def add(prompts: set[str], prefix: str, task: str) -> None:
    prompts.add(DEFAULT_PROMPT.format(task=f"{prefix} {task.strip()}"))


def tasks(root: Path) -> list[str]:
    return [str(json.loads(line)["task"]).strip() for line in (root / "meta/tasks.jsonl").read_text().splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    prompts: set[str] = set()
    add(prompts, FRANKA_PREFIX, "Build a stable pyramid from six colored cups gathered from both sides: three cups on the bottom, two in the middle, and one on top.")
    manifest = json.loads(args.manifest.read_text())
    if len(manifest["sources"]) != 6:
        raise ValueError("Expected six sources in the prompt manifest")
    mobile_roots = [Path(row["dataset_root"]) for row in manifest["sources"][1:4]]
    for root in mobile_roots:
        for task in tasks(root):
            add(prompts, MANIP_PREFIX, task)
            add(prompts, NAV_PREFIX, task)
    ordered_prefix = MANIP_PREFIX
    ordered_variant = Path(manifest["prompt_inputs"]["ordered_variants"])
    for line in ordered_variant.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            add(prompts, ordered_prefix, row["detailed_prompt"])
            add(prompts, ordered_prefix, row["simple_prompt"])
    seven = Path(manifest["sources"][5]["dataset_root"])
    seven_prefix = MANIP_PREFIX
    for task in tasks(seven):
        add(prompts, seven_prefix, task)
    variants = Path(manifest["prompt_inputs"]["agilex7000_variants"])
    for line in variants.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            add(prompts, seven_prefix, row["detailed_prompt"])
            add(prompts, seven_prefix, row["simple_prompt"])
    ordered = sorted(prompts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps({"prompt": p}, ensure_ascii=True) + "\n" for p in ordered))
    report = {
        "status": "PASS",
        "prompt_count": len(ordered),
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "cache_contract": "sha256(DEFAULT_PROMPT(task_with_camera_frame_prefix))",
        "source_prefixes": {"franka": FRANKA_PREFIX, "agilex_manip": MANIP_PREFIX, "agilex_nav": NAV_PREFIX},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
