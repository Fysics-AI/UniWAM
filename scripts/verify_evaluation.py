#!/usr/bin/env python3
"""Verify the standalone geometry-only evaluation component."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--data-root", type=Path, required=True)
args = parser.parse_args()
ROOT = args.data_root.expanduser().resolve()
benchmark_path = ROOT / "evaluation/benchmark_15k.json"
benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
groups = benchmark["groups"]
assert len(groups) == 15_000, len(groups)
ids = [f"{group['group_id']}::canonical_00" for group in groups]
assert len(set(ids)) == 15_000
assert len(list((ROOT / "evaluation/scene_contexts").glob("*.json"))) == 299

for group in groups:
    canonical = group["canonical"]
    for key in ("start_base_xz", "start_theta_rad", "start_camera_xz", "frame_count"):
        assert key in canonical, (group["group_id"], key)
    for key in ("goal_base_xz", "operating_camera_xz", "goal_theta_rad"):
        assert key in group, (group["group_id"], key)
    assert "plan_summary_json" not in group
    assert "video_path" not in canonical
    assert "annotation_path" not in canonical

print("PASS: standalone geometry-only evaluation package")
print("PASS: 15,000 unique trajectories, 299 scene contexts, no source episode dependency")
