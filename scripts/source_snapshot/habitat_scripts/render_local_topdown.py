#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import magnum as mn
import numpy as np

import habitat_sim
from habitat_sim import CameraSensorSpec, Configuration, SensorType, SimulatorConfiguration
from habitat_sim.agent import AgentConfiguration
from habitat_sim.utils.common import quat_from_angle_axis


def load_navmesh_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("navmesh_coverage_trajectory", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class ViewRecord:
    trajectory_json_path: Path
    trajectory_data: dict
    node_row: dict
    view_row: dict
    floor_dir: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Match Gemini observation bboxes to rendered semantic targets and export one local RGB topdown per observation."
    )
    parser.add_argument(
        "--scene-dir",
        required=True,
        help="Scene output directory containing floor_* folders and the Gemini annotation JSON.",
    )
    parser.add_argument(
        "--gemini-json",
        required=True,
        help="Path to Gemini annotation JSON list.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Default: <scene-dir>/gemini_local_topdown_matches_rgb_v1",
    )
    parser.add_argument("--max-observations", type=int, default=3, help="How many annotation entries to process.")
    parser.add_argument("--max-targets-per-observation", type=int, default=8)
    parser.add_argument("--target-name", default="", help="Optional lowercase substring filter for target name.")
    parser.add_argument("--llm-bbox-scale", type=float, default=1000.0, help="Scale used by Gemini bboxes.")
    parser.add_argument("--match-min-score", type=float, default=0.25)
    parser.add_argument(
        "--min-match-iou",
        type=float,
        default=0.08,
        help="Minimum IoU between API bbox and semantic bbox in first-person matching; lower matches are dropped.",
    )
    parser.add_argument("--semantic-min-area-px", type=int, default=32)
    parser.add_argument("--semantic-min-side-px", type=int, default=4)
    parser.add_argument("--semantic-open-kernel", type=int, default=3)
    parser.add_argument("--semantic-close-kernel", type=int, default=5)
    parser.add_argument("--semantic-min-fill-ratio", type=float, default=0.05)
    parser.add_argument(
        "--semantic-ignore-labels",
        default="background,unknown,wall,ceiling,floor",
    )
    parser.add_argument("--topdown-width", type=int, default=1024)
    parser.add_argument("--topdown-height", type=int, default=1024)
    parser.add_argument("--topdown-hfov-deg", type=float, default=100.0)
    parser.add_argument("--topdown-margin-m", type=float, default=0.75)
    parser.add_argument("--topdown-min-above-m", type=float, default=1.5)
    parser.add_argument("--topdown-max-above-m", type=float, default=4.0)
    parser.add_argument("--topdown-crop-fraction", type=float, default=0.72)
    parser.add_argument("--topdown-max-growth-iters", type=int, default=4)
    parser.add_argument("--topdown-growth-factor", type=float, default=1.20)
    parser.add_argument("--topdown-cut-above-ground-m", type=float, default=1.8)
    parser.add_argument("--topdown-cut-depth-margin-m", type=float, default=0.10)
    parser.add_argument("--topdown-cut-inpaint-radius-px", type=int, default=5)
    parser.add_argument("--topdown-ceiling-margin-m", type=float, default=0.25)
    parser.add_argument("--topdown-ray-start-offset-m", type=float, default=0.05)
    parser.add_argument(
        "--topdown-view-margin-ratio",
        type=float,
        default=0.035,
        help="Required image border margin ratio when auto-adjusting camera height for coverage.",
    )
    parser.add_argument(
        "--topdown-camera-height-scale",
        type=float,
        default=0.90,
        help="Scale factor applied to desired topdown camera height before coverage checks (smaller => lower view).",
    )
    parser.add_argument(
        "--topdown-stand-y-fraction",
        type=float,
        default=0.82,
        help="Stand marker vertical anchor in topdown image (0=top, 1=bottom) when aligned to first-person forward direction.",
    )
    parser.add_argument("--max-target-distance-m", type=float, default=2.5)
    parser.add_argument(
        "--require-topdown-semantic",
        type=int,
        default=1,
        help="If 1, only keep targets with topdown semantic-mask bbox; drop fallback world bbox targets.",
    )
    parser.add_argument(
        "--topdown-only",
        type=int,
        default=1,
        help="If 1, only save local_topdown_rgb.png (+meta/summary). If 0, also save observation/panel debug images.",
    )
    parser.add_argument(
        "--topdown-trim-invalid",
        type=int,
        default=1,
        help="If 1, trim invalid gray border in topdown and letterbox back to output size.",
    )
    parser.add_argument("--topdown-trim-margin-px", type=int, default=10)
    parser.add_argument("--topdown-trim-min-valid-px", type=int, default=1024)
    parser.add_argument("--topdown-pad-gray-value", type=int, default=230)
    parser.add_argument("--local-nav-meters-per-pixel", type=float, default=0.03)
    parser.add_argument("--local-nav-margin-m", type=float, default=0.70)
    parser.add_argument("--local-nav-min-side-m", type=float, default=3.2)
    parser.add_argument("--local-nav-dot-stride-px", type=int, default=18)
    parser.add_argument("--local-nav-min-clearance-m", type=float, default=0.18)
    return parser


def normalize_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def sanitize_text(text: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9]+", "_", str(text).strip()).strip("_").lower()
    return out or "item"


def bbox_iou_xyxy(box_a: list[int], box_b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 1e-8:
        return 0.0
    return float(inter / denom)


def bbox_center_xy(box: list[int]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def center_similarity(box_a: list[int], box_b: list[int], width: int, height: int) -> float:
    ax, ay = bbox_center_xy(box_a)
    bx, by = bbox_center_xy(box_b)
    dx = ax - bx
    dy = ay - by
    diag = math.hypot(float(width), float(height))
    if diag <= 1e-8:
        return 0.0
    return max(0.0, 1.0 - math.hypot(dx, dy) / diag)


def canonical_tokens(text: str) -> set[str]:
    drop = {
        "a",
        "an",
        "the",
        "small",
        "large",
        "white",
        "black",
        "gray",
        "grey",
        "wooden",
        "glass",
        "paneled",
        "standing",
        "open",
        "closed",
        "brown",
        "silver",
        "metal",
    }
    words = re.sub(r"[^a-zA-Z0-9]+", " ", str(text).lower()).split()
    out = set()
    for word in words:
        if word in drop:
            continue
        if word.endswith("s") and len(word) > 3:
            word = word[:-1]
        out.add(word)
    return out


def label_similarity(name_a: str, name_b: str) -> float:
    tokens_a = canonical_tokens(name_a)
    tokens_b = canonical_tokens(name_b)
    if not tokens_a or not tokens_b:
        return 0.0
    inter = len(tokens_a & tokens_b)
    if inter == 0:
        joined_a = " ".join(sorted(tokens_a))
        joined_b = " ".join(sorted(tokens_b))
        if joined_a in joined_b or joined_b in joined_a:
            return 0.5
        return 0.0
    union = len(tokens_a | tokens_b)
    return float(inter / max(1, union))


def scale_llm_bbox(raw_bbox: list[float], width: int, height: int, scale: float) -> list[int]:
    if float(scale) <= 0.0:
        raise ValueError("llm bbox scale must be positive")
    x1, y1, x2, y2 = [float(v) for v in raw_bbox]
    if max(abs(v) for v in [x1, y1, x2, y2]) <= 1.5:
        x1 *= float(width)
        x2 *= float(width)
        y1 *= float(height)
        y2 *= float(height)
    else:
        x1 = x1 / float(scale) * float(width)
        x2 = x2 / float(scale) * float(width)
        y1 = y1 / float(scale) * float(height)
        y2 = y2 / float(scale) * float(height)
    x1 = int(np.clip(round(x1), 0, max(0, width - 1)))
    x2 = int(np.clip(round(x2), x1 + 1, max(1, width)))
    y1 = int(np.clip(round(y1), 0, max(0, height - 1)))
    y2 = int(np.clip(round(y2), y1 + 1, max(1, height)))
    return [x1, y1, x2, y2]


def parse_semantic_components(
    *,
    semantic_obs: np.ndarray,
    semantic_id_to_label: dict[int, str],
    ignore_labels: set[str],
    min_area_px: int,
    min_side_px: int,
    open_kernel: int,
    close_kernel: int,
    min_fill_ratio: float,
) -> list[dict]:
    rows: list[dict] = []
    open_struct = np.ones((int(open_kernel), int(open_kernel)), dtype=np.uint8) if int(open_kernel) >= 2 else None
    close_struct = np.ones((int(close_kernel), int(close_kernel)), dtype=np.uint8) if int(close_kernel) >= 2 else None
    semantic = np.asarray(semantic_obs, dtype=np.int32)
    for sid in sorted(int(x) for x in np.unique(semantic) if int(x) > 0):
        label = str(semantic_id_to_label.get(int(sid), "Unknown")).strip()
        key = label.lower()
        if key in ignore_labels or key in {"unknown", "background"}:
            continue
        raw_mask = (semantic == int(sid)).astype(np.uint8)
        if int(np.count_nonzero(raw_mask)) < int(min_area_px):
            continue
        mask = raw_mask.copy()
        if open_struct is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_struct)
        if close_struct is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_struct)
        if int(np.count_nonzero(mask)) < int(min_area_px):
            continue
        num_labels, cc, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for comp_id in range(1, int(num_labels)):
            area = int(stats[comp_id, cv2.CC_STAT_AREA])
            w = int(stats[comp_id, cv2.CC_STAT_WIDTH])
            h = int(stats[comp_id, cv2.CC_STAT_HEIGHT])
            fill_ratio = float(area) / max(1.0, float(w * h))
            if area < int(min_area_px) or min(w, h) < int(min_side_px) or fill_ratio < float(min_fill_ratio):
                continue
            x = int(stats[comp_id, cv2.CC_STAT_LEFT])
            y = int(stats[comp_id, cv2.CC_STAT_TOP])
            comp_mask = (cc == comp_id).astype(np.uint8)
            rows.append(
                {
                    "label": label,
                    "semantic_ids": [int(sid)],
                    "bbox_xyxy": [x, y, x + w, y + h],
                    "bbox_wh": [w, h],
                    "area_px": area,
                    "fill_ratio": fill_ratio,
                    "mask": comp_mask,
                }
            )
    rows.sort(key=lambda row: int(row["area_px"]), reverse=True)
    return rows


def build_view_index(scene_dir: Path) -> tuple[dict[str, ViewRecord], dict[str, dict]]:
    mapping: dict[str, ViewRecord] = {}
    trajectory_by_path: dict[str, dict] = {}
    for traj_json_path in sorted(scene_dir.glob("floor_*/trajectory_nodes.json")):
        traj_data = json.loads(traj_json_path.read_text(encoding="utf-8"))
        trajectory_by_path[str(traj_json_path.resolve())] = traj_data
        for node_row in traj_data.get("nodes", []):
            for view_row in node_row.get("views", []):
                rgb_path = normalize_path(view_row["rgb_png"])
                mapping[rgb_path] = ViewRecord(
                    trajectory_json_path=traj_json_path.resolve(),
                    trajectory_data=traj_data,
                    node_row=node_row,
                    view_row=view_row,
                    floor_dir=traj_json_path.resolve().parent,
                )
    return mapping, trajectory_by_path


def build_topdown_semantic_simulator(
    *,
    scene_dataset_config: str,
    scene_id: str,
    width: int,
    height: int,
    hfov_deg: float,
) -> habitat_sim.Simulator:
    backend_cfg = SimulatorConfiguration()
    backend_cfg.scene_dataset_config_file = str(scene_dataset_config)
    backend_cfg.scene_id = str(scene_id)
    backend_cfg.load_semantic_mesh = True

    rgb_sensor = CameraSensorSpec()
    rgb_sensor.uuid = "topdown_rgb"
    rgb_sensor.sensor_type = SensorType.COLOR
    rgb_sensor.resolution = [int(height), int(width)]
    rgb_sensor.hfov = float(hfov_deg)
    rgb_sensor.position = [0.0, 0.0, 0.0]

    sem_sensor = CameraSensorSpec()
    sem_sensor.uuid = "topdown_semantic"
    sem_sensor.sensor_type = SensorType.SEMANTIC
    sem_sensor.resolution = [int(height), int(width)]
    sem_sensor.hfov = float(hfov_deg)
    sem_sensor.position = [0.0, 0.0, 0.0]

    agent_cfg = AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_sensor, sem_sensor]
    return habitat_sim.Simulator(Configuration(backend_cfg, [agent_cfg]))


def build_dual_agent_simulator(
    *,
    scene_dataset_config: str,
    scene_id: str,
    obs_width: int,
    obs_height: int,
    sensor_height_m: float,
    obs_hfov_deg: float,
    topdown_width: int,
    topdown_height: int,
    topdown_hfov_deg: float,
) -> habitat_sim.Simulator:
    backend_cfg = SimulatorConfiguration()
    backend_cfg.scene_dataset_config_file = str(scene_dataset_config)
    backend_cfg.scene_id = str(scene_id)
    backend_cfg.load_semantic_mesh = True

    rgb_sensor = CameraSensorSpec()
    rgb_sensor.uuid = "rgb"
    rgb_sensor.sensor_type = SensorType.COLOR
    rgb_sensor.resolution = [int(obs_height), int(obs_width)]
    rgb_sensor.hfov = float(obs_hfov_deg)
    rgb_sensor.position = [0.0, float(sensor_height_m), 0.0]

    depth_sensor = CameraSensorSpec()
    depth_sensor.uuid = "depth"
    depth_sensor.sensor_type = SensorType.DEPTH
    depth_sensor.resolution = [int(obs_height), int(obs_width)]
    depth_sensor.hfov = float(obs_hfov_deg)
    depth_sensor.position = [0.0, float(sensor_height_m), 0.0]

    sem_sensor = CameraSensorSpec()
    sem_sensor.uuid = "semantic"
    sem_sensor.sensor_type = SensorType.SEMANTIC
    sem_sensor.resolution = [int(obs_height), int(obs_width)]
    sem_sensor.hfov = float(obs_hfov_deg)
    sem_sensor.position = [0.0, float(sensor_height_m), 0.0]

    top_rgb_sensor = CameraSensorSpec()
    top_rgb_sensor.uuid = "topdown_rgb"
    top_rgb_sensor.sensor_type = SensorType.COLOR
    top_rgb_sensor.resolution = [int(topdown_height), int(topdown_width)]
    top_rgb_sensor.hfov = float(topdown_hfov_deg)
    top_rgb_sensor.position = [0.0, 0.0, 0.0]

    top_depth_sensor = CameraSensorSpec()
    top_depth_sensor.uuid = "topdown_depth"
    top_depth_sensor.sensor_type = SensorType.DEPTH
    top_depth_sensor.resolution = [int(topdown_height), int(topdown_width)]
    top_depth_sensor.hfov = float(topdown_hfov_deg)
    top_depth_sensor.position = [0.0, 0.0, 0.0]

    top_sem_sensor = CameraSensorSpec()
    top_sem_sensor.uuid = "topdown_semantic"
    top_sem_sensor.sensor_type = SensorType.SEMANTIC
    top_sem_sensor.resolution = [int(topdown_height), int(topdown_width)]
    top_sem_sensor.hfov = float(topdown_hfov_deg)
    top_sem_sensor.position = [0.0, 0.0, 0.0]

    agent_cfg = AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_sensor, depth_sensor, sem_sensor, top_rgb_sensor, top_depth_sensor, top_sem_sensor]
    return habitat_sim.Simulator(Configuration(backend_cfg, [agent_cfg]))


def render_topdown_rgb_semantic(
    *,
    sim: habitat_sim.Simulator,
    position_world: np.ndarray,
    rotation_quat: Any,
    agent_index: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    agent = sim.get_agent(int(agent_index))
    state = agent.get_state()
    state.position = np.asarray(position_world, dtype=np.float32)
    state.rotation = rotation_quat
    agent.set_state(state, reset_sensors=True)
    obs = sim.get_sensor_observations()
    rgb = np.asarray(obs["topdown_rgb"], dtype=np.uint8)
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    depth = np.asarray(obs["topdown_depth"], dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    semantic = np.asarray(obs["topdown_semantic"], dtype=np.int32)
    render_camera = agent._sensors["topdown_rgb"].render_camera  # noqa: SLF001
    return rgb, semantic, depth, render_camera


def render_first_person_observation(
    sim: habitat_sim.Simulator,
    *,
    position_world: np.ndarray,
    yaw_rad: float,
    pitch_down_deg: float,
    agent_index: int = 0,
) -> dict:
    agent = sim.get_agent(int(agent_index))
    state = agent.get_state()
    state.position = np.asarray(position_world, dtype=np.float32)
    q_yaw = quat_from_angle_axis(float(yaw_rad), np.array([0.0, 1.0, 0.0], dtype=np.float32))
    q_pitch = quat_from_angle_axis(-math.radians(float(pitch_down_deg)), np.array([1.0, 0.0, 0.0], dtype=np.float32))
    state.rotation = q_yaw * q_pitch
    agent.set_state(state, reset_sensors=True)
    obs = sim.get_sensor_observations()
    rgb = np.asarray(obs["rgb"], dtype=np.uint8)
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    render_camera = agent._sensors["rgb"].render_camera  # noqa: SLF001
    return {
        "rgb": rgb,
        "depth": np.asarray(obs["depth"], dtype=np.float32),
        "semantic": np.asarray(obs["semantic"], dtype=np.int32),
        "render_camera": render_camera,
    }


def topdown_quat_no_tilt(yaw_deg: float = 0.0):
    yaw = math.radians(float(yaw_deg))
    q_yaw = quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    q_pitch = quat_from_angle_axis(-0.5 * math.pi, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    return q_yaw * q_pitch


def upward_hit_distance(
    sim: habitat_sim.Simulator,
    origin_world: np.ndarray,
    *,
    start_offset_m: float,
) -> float | None:
    if hasattr(sim, "get_physics_simulation_library"):
        try:
            physics_lib = sim.get_physics_simulation_library()
            if "NoPhysics" in str(physics_lib):
                return None
        except Exception:
            return None
    origin = np.asarray(origin_world, dtype=np.float32)
    ray = habitat_sim.geo.Ray(
        mn.Vector3(float(origin[0]), float(origin[1]) + float(start_offset_m), float(origin[2])),
        mn.Vector3(0.0, 1.0, 0.0),
    )
    result = sim.cast_ray(ray)
    hits = getattr(result, "hits", None)
    if hits is None:
        return None
    valid: list[float] = []
    for hit in hits:
        dist = float(hit.hit_dist)
        if math.isfinite(dist) and dist > 1e-3:
            valid.append(dist)
    if not valid:
        return None
    return min(valid)


def compute_node_topdown_camera_above(
    *,
    node_y: float,
    floor_levels: list[float] | tuple[float, ...],
    desired_above_m: float,
    upward_hit_dist_m: float | None,
    ceiling_margin_m: float,
) -> float:
    desired = max(0.8, float(desired_above_m))
    limits = [desired]
    higher = sorted(float(y) for y in floor_levels if float(y) > float(node_y) + 0.75)
    if higher:
        limits.append(float(higher[0]) - float(node_y) - 0.35)
    if upward_hit_dist_m is not None and math.isfinite(float(upward_hit_dist_m)):
        limits.append(float(upward_hit_dist_m) - float(ceiling_margin_m))
    return max(0.8, min(float(v) for v in limits))


def palette_rgb(index: int) -> tuple[int, int, int]:
    palette = [
        (255, 196, 0),
        (0, 220, 255),
        (120, 255, 120),
        (255, 120, 120),
        (220, 120, 255),
        (120, 200, 255),
        (255, 160, 90),
        (180, 255, 210),
    ]
    return palette[int(index) % len(palette)]


def blend_mask_rgb(image: np.ndarray, mask: np.ndarray, color_rgb: tuple[int, int, int], alpha: float) -> np.ndarray:
    out = np.asarray(image, dtype=np.uint8).copy()
    mask_bool = np.asarray(mask, dtype=np.uint8) > 0
    if not np.any(mask_bool):
        return out
    color_arr = np.asarray(color_rgb, dtype=np.float32).reshape(1, 3)
    base = out[mask_bool].astype(np.float32)
    out[mask_bool] = np.clip(np.round((1.0 - float(alpha)) * base + float(alpha) * color_arr), 0, 255).astype(np.uint8)
    return out


def build_left_right_panel(topdown_rgb: np.ndarray, obs_rgb: np.ndarray) -> np.ndarray:
    left = np.asarray(topdown_rgb, dtype=np.uint8)
    right = np.asarray(obs_rgb, dtype=np.uint8)
    out_h = max(int(left.shape[0]), int(right.shape[0]))
    left_w = max(1, int(round(float(left.shape[1]) * float(out_h) / max(1.0, float(left.shape[0])))))
    right_w = max(1, int(round(float(right.shape[1]) * float(out_h) / max(1.0, float(right.shape[0])))))
    left_rs = cv2.resize(left, (left_w, out_h), interpolation=cv2.INTER_LINEAR)
    right_rs = cv2.resize(right, (right_w, out_h), interpolation=cv2.INTER_LINEAR)
    gap = 8
    panel = np.zeros((out_h, left_w + gap + right_w, 3), dtype=np.uint8)
    panel[:, :left_w] = left_rs
    panel[:, left_w + gap : left_w + gap + right_w] = right_rs
    cv2.putText(panel, "topdown", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, lineType=cv2.LINE_AA)
    cv2.putText(panel, "topdown", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.putText(panel, "observation", (left_w + gap + 18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, lineType=cv2.LINE_AA)
    cv2.putText(panel, "observation", (left_w + gap + 18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    return panel


def compute_multi_target_crop_world_bounds(
    *,
    node_world: np.ndarray,
    target_rows: list[dict],
    margin_m: float,
    min_side_m: float,
    view_yaw_deg: float,
    stand_y_fraction: float,
) -> dict:
    anchor = float(np.clip(float(stand_y_fraction), 0.60, 0.92))
    yaw = math.radians(float(view_yaw_deg))
    # First-person-consistent local frame:
    # right  = camera right on XZ plane
    # forward = camera forward on XZ plane
    right_x = float(math.cos(yaw))
    right_z = float(-math.sin(yaw))
    forward_x = float(-math.sin(yaw))
    forward_z = float(-math.cos(yaw))
    nx = float(node_world[0])
    nz = float(node_world[2])

    # Collect local-plane (u, f) points where:
    # - u axis: camera right
    # - f axis: camera forward
    # Stand point is (0, 0). We later map it to lower-middle of the output.
    local_u: list[float] = [0.0]
    local_f: list[float] = [0.0]

    def _append_point(world_x: float, world_z: float) -> None:
        dx = float(world_x) - nx
        dz = float(world_z) - nz
        local_u.append(dx * right_x + dz * right_z)
        local_f.append(dx * forward_x + dz * forward_z)

    for row in target_rows:
        bounds_xz = choose_target_crop_bounds_xz(row)
        x0, z0, x1, z1 = [float(v) for v in bounds_xz]
        _append_point(float(x0), float(z0))
        _append_point(float(x0), float(z1))
        _append_point(float(x1), float(z0))
        _append_point(float(x1), float(z1))

    margin = float(max(0.05, margin_m))
    max_abs_u = max(abs(float(v)) for v in local_u) + margin
    max_fwd = max(float(v) for v in local_f) + margin
    max_back = max(0.0, -(min(float(v) for v in local_f) - margin))

    # Enforce:
    # 1) include all targets
    # 2) square metric scale in topdown
    # 3) stand appears near lower-center according to anchor
    side_need_u = 2.0 * max_abs_u
    side_need_fwd = max(0.0, max_fwd) / max(1e-3, anchor)
    side_need_back = max(0.0, max_back) / max(1e-3, (1.0 - anchor))
    side = max(float(min_side_m), float(side_need_u), float(side_need_fwd), float(side_need_back), 1.0)

    u_min = -0.5 * side
    u_max = 0.5 * side
    f_front = anchor * side
    f_back = -(1.0 - anchor) * side

    # Convert oriented local crop corners back to world XZ for metadata.
    corners_local = [
        (u_min, f_front),
        (u_max, f_front),
        (u_min, f_back),
        (u_max, f_back),
    ]
    corners_world = [
        (
            nx + right_x * float(u) + forward_x * float(f),
            nz + right_z * float(u) + forward_z * float(f),
        )
        for (u, f) in corners_local
    ]
    xs = [float(p[0]) for p in corners_world]
    zs = [float(p[1]) for p in corners_world]
    x_min = min(xs)
    x_max = max(xs)
    z_min = min(zs)
    z_max = max(zs)
    cx = 0.5 * (x_min + x_max)
    cz = 0.5 * (z_min + z_max)
    return {
        "x_min": float(x_min),
        "x_max": float(x_max),
        "z_min": float(z_min),
        "z_max": float(z_max),
        "center_x": float(cx),
        "center_z": float(cz),
        "side_m": float(side),
        "mapping": "oriented_forward",
        "stand_x": float(nx),
        "stand_z": float(nz),
        "right_x": float(right_x),
        "right_z": float(right_z),
        "forward_x": float(forward_x),
        "forward_z": float(forward_z),
        "u_min": float(u_min),
        "u_max": float(u_max),
        "f_front": float(f_front),
        "f_back": float(f_back),
        "stand_y_fraction": float(anchor),
    }


def estimate_topdown_camera_above_m(
    *,
    crop_side_m: float,
    hfov_deg: float,
    crop_fraction: float,
    min_above_m: float,
    max_above_m: float,
) -> float:
    half_side = 0.5 * float(crop_side_m)
    eff_fraction = float(np.clip(float(crop_fraction), 0.25, 0.95))
    hfov_rad = math.radians(float(hfov_deg))
    denom = max(1e-6, eff_fraction * math.tan(0.5 * hfov_rad))
    required = half_side / denom
    return float(np.clip(max(float(min_above_m), required), float(min_above_m), float(max_above_m)))


def estimate_topdown_camera_above_from_node(
    *,
    node_world: np.ndarray,
    crop_world: dict,
    width: int,
    height: int,
    hfov_deg: float,
    crop_fraction: float,
    min_above_m: float,
    max_above_m: float,
) -> float:
    eff_fraction = float(np.clip(float(crop_fraction), 0.25, 0.95))
    dx = max(abs(float(crop_world["x_min"]) - float(node_world[0])), abs(float(crop_world["x_max"]) - float(node_world[0])))
    dz = max(abs(float(crop_world["z_min"]) - float(node_world[2])), abs(float(crop_world["z_max"]) - float(node_world[2])))
    hfov_rad = math.radians(float(hfov_deg))
    vfov_rad = 2.0 * math.atan(math.tan(0.5 * hfov_rad) * (float(height) / max(1.0, float(width))))
    req_x = dx / max(1e-6, eff_fraction * math.tan(0.5 * hfov_rad))
    req_z = dz / max(1e-6, eff_fraction * math.tan(0.5 * vfov_rad))
    required = max(float(min_above_m), float(req_x), float(req_z))
    return float(np.clip(required, float(min_above_m), float(max_above_m)))


def compute_center_crop_box(width: int, height: int, crop_fraction: float) -> list[int]:
    frac = float(np.clip(float(crop_fraction), 0.25, 1.0))
    crop_w = max(32, int(round(float(width) * frac)))
    crop_h = max(32, int(round(float(height) * frac)))
    x0 = max(0, (int(width) - crop_w) // 2)
    y0 = max(0, (int(height) - crop_h) // 2)
    return [int(x0), int(y0), int(x0 + crop_w), int(y0 + crop_h)]


def project_world_to_pixel_exact(render_camera: Any, world_xyz: np.ndarray, width: int, height: int) -> dict:
    v = mn.Vector4(float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2]), 1.0)
    clip = render_camera.projection_matrix * (render_camera.camera_matrix * v)
    w = float(clip[3])
    if abs(w) < 1e-8:
        return {
            "valid": False,
            "u": float(width * 0.5),
            "v": float(height * 0.5),
            "u_draw": int(width // 2),
            "v_draw": int(height // 2),
            "visible_in_image": False,
        }
    ndc_x = float(clip[0]) / w
    ndc_y = float(clip[1]) / w
    ndc_z = float(clip[2]) / w
    u = (ndc_x + 1.0) * 0.5 * float(width - 1)
    vv = (1.0 - (ndc_y + 1.0) * 0.5) * float(height - 1)
    visible = (0.0 <= u < float(width)) and (0.0 <= vv < float(height)) and (-1.0 <= ndc_z <= 1.0)
    return {
        "valid": bool(np.isfinite(u) and np.isfinite(vv)),
        "u": float(u),
        "v": float(vv),
        "u_draw": int(np.clip(round(u), 0, max(0, int(width) - 1))),
        "v_draw": int(np.clip(round(vv), 0, max(0, int(height) - 1))),
        "visible_in_image": bool(visible),
    }


def build_world_xz_to_fullpixel_homography(
    *,
    render_camera: Any,
    plane_y: float,
    center_x: float,
    center_z: float,
    span_m: float,
    width: int,
    height: int,
) -> np.ndarray | None:
    span = max(1.0, float(span_m))
    for _ in range(6):
        half = 0.5 * span
        src_xz = np.asarray(
            [
                [float(center_x) - half, float(center_z) - half],
                [float(center_x) + half, float(center_z) - half],
                [float(center_x) - half, float(center_z) + half],
                [float(center_x) + half, float(center_z) + half],
            ],
            dtype=np.float32,
        )
        dst_uv: list[list[float]] = []
        ok = True
        for wx, wz in src_xz:
            proj = project_world_to_pixel_exact(
                render_camera,
                np.asarray([float(wx), float(plane_y), float(wz)], dtype=np.float32),
                int(width),
                int(height),
            )
            if not bool(proj.get("valid", False)):
                ok = False
                break
            dst_uv.append([float(proj["u"]), float(proj["v"])])
        if ok:
            h = cv2.getPerspectiveTransform(src_xz.astype(np.float32), np.asarray(dst_uv, dtype=np.float32))
            if h is not None and np.all(np.isfinite(h)):
                return np.asarray(h, dtype=np.float32)
        span *= 0.72
    return None


def world_xz_grid_to_fullpixel_map(
    *,
    grid_x: np.ndarray,
    grid_z: np.ndarray,
    world_xz_to_fullpixel_h: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    h = np.asarray(world_xz_to_fullpixel_h, dtype=np.float32)
    gx = np.asarray(grid_x, dtype=np.float32)
    gz = np.asarray(grid_z, dtype=np.float32)
    denom = h[2, 0] * gx + h[2, 1] * gz + h[2, 2]
    denom = np.where(np.abs(denom) < 1e-8, np.nan, denom)
    map_x = (h[0, 0] * gx + h[0, 1] * gz + h[0, 2]) / denom
    map_y = (h[1, 0] * gx + h[1, 1] * gz + h[1, 2]) / denom
    return map_x.astype(np.float32), map_y.astype(np.float32)


def compute_trim_letterbox_transform(
    *,
    valid_mask: np.ndarray,
    out_width: int,
    out_height: int,
    margin_px: int,
    min_valid_px: int,
) -> dict:
    mask = np.asarray(valid_mask, dtype=bool)
    out_w = int(out_width)
    out_h = int(out_height)
    src_h, src_w = mask.shape[:2]
    ys, xs = np.where(mask)
    if int(xs.size) < int(max(1, min_valid_px)):
        return {
            "enabled": False,
            "src_width": int(src_w),
            "src_height": int(src_h),
            "out_width": int(out_w),
            "out_height": int(out_h),
            "valid_px": int(xs.size),
        }
    pad = max(0, int(margin_px))
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(int(src_w), int(xs.max()) + 1 + pad)
    y1 = min(int(src_h), int(ys.max()) + 1 + pad)
    crop_w = max(1, int(x1 - x0))
    crop_h = max(1, int(y1 - y0))
    scale = min(float(out_w) / float(crop_w), float(out_h) / float(crop_h))
    render_w = max(1, int(round(float(crop_w) * float(scale))))
    render_h = max(1, int(round(float(crop_h) * float(scale))))
    off_x = max(0, (int(out_w) - int(render_w)) // 2)
    off_y = max(0, (int(out_h) - int(render_h)) // 2)
    return {
        "enabled": True,
        "src_width": int(src_w),
        "src_height": int(src_h),
        "out_width": int(out_w),
        "out_height": int(out_h),
        "x0": int(x0),
        "y0": int(y0),
        "x1": int(x1),
        "y1": int(y1),
        "crop_width": int(crop_w),
        "crop_height": int(crop_h),
        "render_width": int(render_w),
        "render_height": int(render_h),
        "offset_x": int(off_x),
        "offset_y": int(off_y),
        "scale": float(scale),
        "valid_px": int(xs.size),
    }


def apply_trim_letterbox_rgb(
    *,
    image_rgb: np.ndarray,
    valid_mask: np.ndarray,
    trim_transform: dict,
    pad_gray_value: int,
) -> tuple[np.ndarray, np.ndarray]:
    img = np.asarray(image_rgb, dtype=np.uint8)
    mask = np.asarray(valid_mask, dtype=bool)
    if not bool(trim_transform.get("enabled", False)):
        return img.copy(), mask.copy()

    x0 = int(trim_transform["x0"])
    y0 = int(trim_transform["y0"])
    x1 = int(trim_transform["x1"])
    y1 = int(trim_transform["y1"])
    rw = int(trim_transform["render_width"])
    rh = int(trim_transform["render_height"])
    ox = int(trim_transform["offset_x"])
    oy = int(trim_transform["offset_y"])
    out_w = int(trim_transform["out_width"])
    out_h = int(trim_transform["out_height"])
    pad_val = int(np.clip(int(pad_gray_value), 0, 255))

    crop_img = img[y0:y1, x0:x1]
    crop_mask = mask[y0:y1, x0:x1]
    resized_img = cv2.resize(crop_img, (rw, rh), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize((crop_mask.astype(np.uint8) * 255), (rw, rh), interpolation=cv2.INTER_NEAREST) > 0

    out_img = np.full((out_h, out_w, 3), int(pad_val), dtype=np.uint8)
    out_mask = np.zeros((out_h, out_w), dtype=bool)
    out_img[oy : oy + rh, ox : ox + rw] = resized_img
    out_mask[oy : oy + rh, ox : ox + rw] = resized_mask
    return out_img, out_mask


def map_base_point_to_trimmed(
    *,
    xy: tuple[int, int],
    trim_transform: dict,
) -> tuple[int, int]:
    x = float(xy[0])
    y = float(xy[1])
    if not bool(trim_transform.get("enabled", False)):
        out_w = int(trim_transform.get("out_width", 1))
        out_h = int(trim_transform.get("out_height", 1))
        return (
            int(np.clip(round(x), 0, max(0, out_w - 1))),
            int(np.clip(round(y), 0, max(0, out_h - 1))),
        )
    x0 = float(trim_transform["x0"])
    y0 = float(trim_transform["y0"])
    cw = float(max(1, int(trim_transform["crop_width"])))
    ch = float(max(1, int(trim_transform["crop_height"])))
    rw = float(max(1, int(trim_transform["render_width"])))
    rh = float(max(1, int(trim_transform["render_height"])))
    ox = float(trim_transform["offset_x"])
    oy = float(trim_transform["offset_y"])
    out_w = int(trim_transform["out_width"])
    out_h = int(trim_transform["out_height"])
    tx = ox + ((x - x0 + 0.5) * rw / cw) - 0.5
    ty = oy + ((y - y0 + 0.5) * rh / ch) - 0.5
    return (
        int(np.clip(round(tx), 0, max(0, out_w - 1))),
        int(np.clip(round(ty), 0, max(0, out_h - 1))),
    )


def map_trimmed_point_to_base(
    *,
    xy: tuple[int, int],
    trim_transform: dict,
) -> tuple[int, int]:
    x = float(xy[0])
    y = float(xy[1])
    src_w = int(trim_transform.get("src_width", trim_transform.get("out_width", 1)))
    src_h = int(trim_transform.get("src_height", trim_transform.get("out_height", 1)))
    if not bool(trim_transform.get("enabled", False)):
        return (
            int(np.clip(round(x), 0, max(0, src_w - 1))),
            int(np.clip(round(y), 0, max(0, src_h - 1))),
        )
    x0 = float(trim_transform["x0"])
    y0 = float(trim_transform["y0"])
    cw = float(max(1, int(trim_transform["crop_width"])))
    ch = float(max(1, int(trim_transform["crop_height"])))
    rw = float(max(1, int(trim_transform["render_width"])))
    rh = float(max(1, int(trim_transform["render_height"])))
    ox = float(trim_transform["offset_x"])
    oy = float(trim_transform["offset_y"])
    bx = x0 + ((x - ox + 0.5) * cw / rw) - 0.5
    by = y0 + ((y - oy + 0.5) * ch / rh) - 0.5
    return (
        int(np.clip(round(bx), 0, max(0, src_w - 1))),
        int(np.clip(round(by), 0, max(0, src_h - 1))),
    )


def map_base_mask_to_trimmed(
    *,
    base_mask: np.ndarray,
    trim_transform: dict,
) -> np.ndarray:
    src = (np.asarray(base_mask, dtype=np.uint8) > 0)
    if not bool(trim_transform.get("enabled", False)):
        return src.astype(np.uint8)
    x0 = int(trim_transform["x0"])
    y0 = int(trim_transform["y0"])
    x1 = int(trim_transform["x1"])
    y1 = int(trim_transform["y1"])
    rw = int(trim_transform["render_width"])
    rh = int(trim_transform["render_height"])
    ox = int(trim_transform["offset_x"])
    oy = int(trim_transform["offset_y"])
    out_w = int(trim_transform["out_width"])
    out_h = int(trim_transform["out_height"])
    crop = src[y0:y1, x0:x1]
    resized = cv2.resize((crop.astype(np.uint8) * 255), (rw, rh), interpolation=cv2.INTER_NEAREST) > 0
    out = np.zeros((out_h, out_w), dtype=np.uint8)
    out[oy : oy + rh, ox : ox + rw] = resized.astype(np.uint8)
    return out


def map_base_bbox_to_trimmed(
    *,
    box_xyxy: list[int] | None,
    trim_transform: dict,
) -> list[int] | None:
    if box_xyxy is None:
        return None
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    p1 = map_base_point_to_trimmed(xy=(int(x1), int(y1)), trim_transform=trim_transform)
    p2 = map_base_point_to_trimmed(xy=(int(x2), int(y2)), trim_transform=trim_transform)
    ox1 = int(min(p1[0], p2[0]))
    oy1 = int(min(p1[1], p2[1]))
    ox2 = int(max(p1[0], p2[0]) + 1)
    oy2 = int(max(p1[1], p2[1]) + 1)
    out_w = int(trim_transform.get("out_width", 1))
    out_h = int(trim_transform.get("out_height", 1))
    ox1 = int(np.clip(ox1, 0, max(0, out_w - 1)))
    oy1 = int(np.clip(oy1, 0, max(0, out_h - 1)))
    ox2 = int(np.clip(max(ox1 + 1, ox2), ox1 + 1, max(1, out_w)))
    oy2 = int(np.clip(max(oy1 + 1, oy2), oy1 + 1, max(1, out_h)))
    return [ox1, oy1, ox2, oy2]


def project_footprint_bbox_exact(
    *,
    render_camera: Any,
    footprint_xz: list[float],
    world_y: float,
    width: int,
    height: int,
) -> list[int] | None:
    x0, z0, x1, z1 = [float(v) for v in footprint_xz]
    corners = [
        np.asarray([x0, float(world_y), z0], dtype=np.float32),
        np.asarray([x0, float(world_y), z1], dtype=np.float32),
        np.asarray([x1, float(world_y), z0], dtype=np.float32),
        np.asarray([x1, float(world_y), z1], dtype=np.float32),
    ]
    proj = [project_world_to_pixel_exact(render_camera, c, int(width), int(height)) for c in corners]
    valid = [p for p in proj if bool(p["valid"])]
    if not valid:
        return None
    xs = [float(p["u"]) for p in valid]
    ys = [float(p["v"]) for p in valid]
    x_min = int(np.clip(math.floor(min(xs)), 0, max(0, int(width) - 1)))
    y_min = int(np.clip(math.floor(min(ys)), 0, max(0, int(height) - 1)))
    x_max = int(np.clip(math.ceil(max(xs)), x_min + 1, max(1, int(width))))
    y_max = int(np.clip(math.ceil(max(ys)), y_min + 1, max(1, int(height))))
    return [x_min, y_min, x_max, y_max]


def compute_projected_crop_box(
    *,
    projected_points: list[dict],
    projected_boxes: list[list[int] | None],
    width: int,
    height: int,
    pad_px: int,
    min_crop_px: int,
) -> list[int]:
    xs: list[float] = []
    ys: list[float] = []
    for p in projected_points:
        if bool(p.get("valid", False)):
            xs.append(float(p["u"]))
            ys.append(float(p["v"]))
    for box in projected_boxes:
        if box is None:
            continue
        xs.extend([float(box[0]), float(box[2])])
        ys.extend([float(box[1]), float(box[3])])
    if not xs or not ys:
        return compute_center_crop_box(int(width), int(height), 0.72)
    x_min = min(xs) - float(pad_px)
    x_max = max(xs) + float(pad_px)
    y_min = min(ys) - float(pad_px)
    y_max = max(ys) + float(pad_px)
    side = max(float(min_crop_px), float(x_max - x_min), float(y_max - y_min))
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    half = 0.5 * side
    bx0 = int(np.clip(math.floor(cx - half), 0, max(0, int(width) - 1)))
    by0 = int(np.clip(math.floor(cy - half), 0, max(0, int(height) - 1)))
    bx1 = int(np.clip(math.ceil(cx + half), bx0 + 1, max(1, int(width))))
    by1 = int(np.clip(math.ceil(cy + half), by0 + 1, max(1, int(height))))
    return [bx0, by0, bx1, by1]


def expand_square_crop_box(
    *,
    box_xyxy: list[int],
    width: int,
    height: int,
    pad_px: int,
    min_crop_px: int,
) -> list[int]:
    x0, y0, x1, y1 = [int(v) for v in box_xyxy]
    x0 = max(0, x0 - int(pad_px))
    y0 = max(0, y0 - int(pad_px))
    x1 = min(int(width), x1 + int(pad_px))
    y1 = min(int(height), y1 + int(pad_px))
    side = max(int(min_crop_px), x1 - x0, y1 - y0)
    cx = 0.5 * float(x0 + x1)
    cy = 0.5 * float(y0 + y1)
    half = 0.5 * float(side)
    bx0 = int(np.clip(math.floor(cx - half), 0, max(0, int(width) - 1)))
    by0 = int(np.clip(math.floor(cy - half), 0, max(0, int(height) - 1)))
    bx1 = int(np.clip(math.ceil(cx + half), bx0 + 1, max(1, int(width))))
    by1 = int(np.clip(math.ceil(cy + half), by0 + 1, max(1, int(height))))
    return [bx0, by0, bx1, by1]


def extract_topdown_semantic_target(
    *,
    semantic_obs: np.ndarray,
    semantic_ids: list[int],
    open_kernel: int,
    close_kernel: int,
    min_area_px: int,
    preferred_bbox_xyxy: list[int] | None = None,
    preferred_center_xy: tuple[float, float] | None = None,
) -> dict | None:
    semantic = np.asarray(semantic_obs, dtype=np.int32)
    id_set = {int(v) for v in semantic_ids if int(v) > 0}
    if not id_set:
        return None
    mask = np.isin(semantic, list(id_set)).astype(np.uint8)
    if int(np.count_nonzero(mask)) < int(min_area_px):
        return None
    if int(open_kernel) >= 2:
        k = np.ones((int(open_kernel), int(open_kernel)), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if int(close_kernel) >= 2:
        k = np.ones((int(close_kernel), int(close_kernel)), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    if int(np.count_nonzero(mask)) < int(min_area_px):
        return None
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if int(nlab) <= 1:
        return None
    best_lab = -1
    best_area = -1
    best_iou = -1.0
    best_dist = float("inf")
    for lab in range(1, int(nlab)):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < int(min_area_px):
            continue
        x = int(stats[lab, cv2.CC_STAT_LEFT])
        y = int(stats[lab, cv2.CC_STAT_TOP])
        w = int(stats[lab, cv2.CC_STAT_WIDTH])
        h = int(stats[lab, cv2.CC_STAT_HEIGHT])
        bbox = [int(x), int(y), int(x + w), int(y + h)]
        cx = 0.5 * float(bbox[0] + bbox[2])
        cy = 0.5 * float(bbox[1] + bbox[3])
        iou_pref = bbox_iou_xyxy(bbox, preferred_bbox_xyxy) if preferred_bbox_xyxy is not None else 0.0
        if preferred_center_xy is not None:
            dist_pref = math.hypot(cx - float(preferred_center_xy[0]), cy - float(preferred_center_xy[1]))
        else:
            dist_pref = 0.0

        # Prefer overlap with expected projected footprint first, then center distance, then area.
        better = False
        if preferred_bbox_xyxy is not None:
            if iou_pref > best_iou + 1e-6:
                better = True
            elif abs(iou_pref - best_iou) <= 1e-6 and (dist_pref < best_dist - 1e-6):
                better = True
            elif abs(iou_pref - best_iou) <= 1e-6 and abs(dist_pref - best_dist) <= 1e-6 and area > best_area:
                better = True
        elif preferred_center_xy is not None:
            if dist_pref < best_dist - 1e-6:
                better = True
            elif abs(dist_pref - best_dist) <= 1e-6 and area > best_area:
                better = True
        else:
            if area > best_area:
                better = True
        if better:
            best_lab = int(lab)
            best_area = int(area)
            best_iou = float(iou_pref)
            best_dist = float(dist_pref)
    if best_lab < 1 or best_area < int(min_area_px):
        return None
    keep = (labels == best_lab).astype(np.uint8)
    ys, xs = np.where(keep > 0)
    if ys.size == 0:
        return None
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return {
        "mask": keep,
        "bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "center_xy": [int(round(0.5 * (x0 + x1))), int(round(0.5 * (y0 + y1)))],
        "area_px": int(best_area),
        "preferred_bbox_iou": float(max(0.0, best_iou)),
        "preferred_center_dist_px": float(best_dist if math.isfinite(best_dist) else 0.0),
    }


def unproject_pixel_to_plane_exact(render_camera: Any, u: float, v: float, plane_y: float) -> np.ndarray | None:
    ray = render_camera.unproject(mn.Vector2i(int(round(float(u))), int(round(float(v)))))
    t = render_camera.node.absolute_translation
    origin = np.asarray([float(t[0]), float(t[1]), float(t[2])], dtype=np.float64)
    direction = np.asarray([float(ray.direction[0]), float(ray.direction[1]), float(ray.direction[2])], dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        return None
    direction = direction / norm
    if abs(float(direction[1])) < 1e-8:
        return None
    travel = (float(plane_y) - float(origin[1])) / float(direction[1])
    if travel <= 0.0:
        return None
    point = origin + direction * travel
    if not np.all(np.isfinite(point)):
        return None
    return point.astype(np.float32)


def output_pixel_to_world_on_plane(
    *,
    render_camera: Any,
    output_xy: tuple[int, int],
    crop_box_xyxy: list[int],
    out_width: int,
    out_height: int,
    plane_y: float,
) -> np.ndarray | None:
    x0, y0, x1, y1 = [int(v) for v in crop_box_xyxy]
    crop_w = max(1, x1 - x0)
    crop_h = max(1, y1 - y0)
    full_u = float(x0) + (float(output_xy[0]) + 0.5) * float(crop_w) / float(max(1, out_width))
    full_v = float(y0) + (float(output_xy[1]) + 0.5) * float(crop_h) / float(max(1, out_height))
    return unproject_pixel_to_plane_exact(render_camera, full_u, full_v, float(plane_y))


def project_topdown_world_xz_to_full_pixel(
    *,
    world_x: float,
    world_z: float,
    camera_center_world: np.ndarray,
    camera_above_m: float,
    width: int,
    height: int,
    hfov_deg: float,
) -> tuple[float, float]:
    dx = float(world_x) - float(camera_center_world[0])
    dz = float(world_z) - float(camera_center_world[2])
    hfov_rad = math.radians(float(hfov_deg))
    fx = 0.5 * float(width) / math.tan(0.5 * hfov_rad)
    fy = fx
    cx = 0.5 * (float(width) - 1.0)
    cy = 0.5 * (float(height) - 1.0)
    depth = max(1e-4, float(camera_above_m))
    u = cx + fx * dx / depth
    v = cy - fy * dz / depth
    return u, v


def project_full_pixel_to_output(
    *,
    full_xy: tuple[float, float],
    crop_box_xyxy: list[int],
    out_width: int,
    out_height: int,
) -> tuple[int, int]:
    x0, y0, x1, y1 = [int(v) for v in crop_box_xyxy]
    crop_w = max(1, x1 - x0)
    crop_h = max(1, y1 - y0)
    px = int(np.clip(round((float(full_xy[0]) - float(x0)) * float(out_width) / float(crop_w)), 0, max(0, out_width - 1)))
    py = int(np.clip(round((float(full_xy[1]) - float(y0)) * float(out_height) / float(crop_h)), 0, max(0, out_height - 1)))
    return px, py


def project_topdown_world_xz_to_output(
    *,
    world_x: float,
    world_z: float,
    camera_center_world: np.ndarray,
    camera_above_m: float,
    full_width: int,
    full_height: int,
    hfov_deg: float,
    crop_box_xyxy: list[int],
    out_width: int,
    out_height: int,
) -> tuple[int, int]:
    full_xy = project_topdown_world_xz_to_full_pixel(
        world_x=float(world_x),
        world_z=float(world_z),
        camera_center_world=camera_center_world,
        camera_above_m=float(camera_above_m),
        width=int(full_width),
        height=int(full_height),
        hfov_deg=float(hfov_deg),
    )
    return project_full_pixel_to_output(
        full_xy=full_xy,
        crop_box_xyxy=crop_box_xyxy,
        out_width=int(out_width),
        out_height=int(out_height),
    )


def project_topdown_footprint_bbox_to_output(
    *,
    footprint_xz: list[float],
    camera_center_world: np.ndarray,
    camera_above_m: float,
    full_width: int,
    full_height: int,
    hfov_deg: float,
    crop_box_xyxy: list[int],
    out_width: int,
    out_height: int,
) -> list[int] | None:
    x0, z0, x1, z1 = [float(v) for v in footprint_xz]
    corners = [(x0, z0), (x0, z1), (x1, z0), (x1, z1)]
    full_pts = [
        project_topdown_world_xz_to_full_pixel(
            world_x=wx,
            world_z=wz,
            camera_center_world=camera_center_world,
            camera_above_m=float(camera_above_m),
            width=int(full_width),
            height=int(full_height),
            hfov_deg=float(hfov_deg),
        )
        for wx, wz in corners
    ]
    xs = [float(p[0]) for p in full_pts]
    ys = [float(p[1]) for p in full_pts]
    full_box = [
        int(math.floor(min(xs))),
        int(math.floor(min(ys))),
        int(math.ceil(max(xs))),
        int(math.ceil(max(ys))),
    ]
    x1o, y1o = project_full_pixel_to_output(
        full_xy=(float(full_box[0]), float(full_box[1])),
        crop_box_xyxy=crop_box_xyxy,
        out_width=int(out_width),
        out_height=int(out_height),
    )
    x2o, y2o = project_full_pixel_to_output(
        full_xy=(float(full_box[2]), float(full_box[3])),
        crop_box_xyxy=crop_box_xyxy,
        out_width=int(out_width),
        out_height=int(out_height),
    )
    x2o = max(x1o + 1, x2o)
    y2o = max(y1o + 1, y2o)
    return [int(x1o), int(y1o), int(x2o), int(y2o)]


def build_target_match_candidates(
    *,
    targets: list[dict],
    sem_items: list[dict],
    image_width: int,
    image_height: int,
    llm_bbox_scale: float,
    min_match_iou: float,
) -> list[dict]:
    candidates: list[dict] = []
    for target_idx, target in enumerate(targets):
        llm_name = str(target.get("name", "")).strip()
        llm_bbox_xyxy = scale_llm_bbox(
            list(target["bbox"]),
            width=int(image_width),
            height=int(image_height),
            scale=float(llm_bbox_scale),
        )
        for item_idx, item in enumerate(sem_items):
            iou = bbox_iou_xyxy(llm_bbox_xyxy, item["bbox_xyxy"])
            if float(iou) < float(min_match_iou):
                continue
            center_score = center_similarity(llm_bbox_xyxy, item["bbox_xyxy"], image_width, image_height)
            # Match by geometry only: overlap is primary, center is only a weak tie-breaker.
            score = 100.0 * iou + 0.1 * center_score
            candidates.append(
                {
                    "target_idx": int(target_idx),
                    "item_idx": int(item_idx),
                    "llm_name": llm_name,
                    "llm_bbox_xyxy": [int(v) for v in llm_bbox_xyxy],
                    "iou": float(iou),
                    "score": float(score),
                    "score_breakdown": {
                        "iou": float(iou),
                        "label_similarity": 0.0,
                        "center_similarity": float(center_score),
                        "label_priority": 0.0,
                    },
                }
            )
    candidates.sort(
        key=lambda row: (
            float(row["iou"]),
            float(row["score_breakdown"]["center_similarity"]),
            float(row["score"]),
        ),
        reverse=True,
    )
    return candidates


def assign_targets_to_semantic_items(
    *,
    targets: list[dict],
    sem_items: list[dict],
    image_width: int,
    image_height: int,
    llm_bbox_scale: float,
    match_min_score: float,
    min_match_iou: float,
) -> list[dict]:
    candidates = build_target_match_candidates(
        targets=targets,
        sem_items=sem_items,
        image_width=int(image_width),
        image_height=int(image_height),
        llm_bbox_scale=float(llm_bbox_scale),
        min_match_iou=float(min_match_iou),
    )
    used_targets: set[int] = set()
    used_items: set[int] = set()
    matches: list[dict] = []
    for cand in candidates:
        if int(cand["target_idx"]) in used_targets or int(cand["item_idx"]) in used_items:
            continue
        if float(cand["iou"]) < float(min_match_iou):
            continue
        if float(cand["score"]) < float(match_min_score):
            continue
        target = dict(targets[int(cand["target_idx"])])
        item = dict(sem_items[int(cand["item_idx"])])
        target["name"] = str(cand["llm_name"])
        matches.append(
            {
                "target_idx": int(cand["target_idx"]),
                "llm_target": {
                    "name": str(cand["llm_name"]),
                    "bbox_raw": list(target.get("bbox", [])),
                    "bbox_xyxy": [int(v) for v in cand["llm_bbox_xyxy"]],
                },
                "matched_item": item,
                "score": float(cand["score"]),
                "score_breakdown": cand["score_breakdown"],
            }
        )
        used_targets.add(int(cand["target_idx"]))
        used_items.add(int(cand["item_idx"]))
    matches.sort(key=lambda row: int(row["target_idx"]))
    return matches


def extract_object_records(sim: habitat_sim.Simulator, semantic_ids: list[int]) -> list[dict]:
    out: list[dict] = []
    id_set = {int(v) for v in semantic_ids}
    if sim.semantic_scene is None:
        return out
    for obj in sim.semantic_scene.objects:
        raw_id = getattr(obj, "id", None)
        sid = None
        if raw_id is not None:
            text = str(raw_id)
            match = re.search(r"_(\d+)$", text)
            if match:
                sid = int(match.group(1))
            elif text.isdigit():
                sid = int(text)
        if sid is None or int(sid) not in id_set:
            continue
        aabb = getattr(obj, "aabb", None)
        if aabb is None:
            continue
        center = getattr(aabb, "center", None)
        if center is None:
            continue
        if callable(center):
            center = center()
        aabb_min = getattr(aabb, "min", None)
        aabb_max = getattr(aabb, "max", None)
        if callable(aabb_min):
            aabb_min = aabb_min()
        if callable(aabb_max):
            aabb_max = aabb_max()
        footprint_xz = None
        if aabb_min is not None and aabb_max is not None:
            mnv = np.asarray([float(aabb_min[0]), float(aabb_min[1]), float(aabb_min[2])], dtype=np.float32)
            mxv = np.asarray([float(aabb_max[0]), float(aabb_max[1]), float(aabb_max[2])], dtype=np.float32)
            footprint_xz = [
                float(min(mnv[0], mxv[0])),
                float(min(mnv[2], mxv[2])),
                float(max(mnv[0], mxv[0])),
                float(max(mnv[2], mxv[2])),
            ]
        else:
            left = getattr(aabb, "left", None)
            right = getattr(aabb, "right", None)
            back = getattr(aabb, "back", None)
            front = getattr(aabb, "front", None)
            if None not in {left, right, back, front}:
                footprint_xz = [
                    float(min(left, right)),
                    float(min(back, front)),
                    float(max(left, right)),
                    float(max(back, front)),
                ]
        if footprint_xz is None:
            footprint_xz = [float(center[0]) - 0.15, float(center[2]) - 0.15, float(center[0]) + 0.15, float(center[2]) + 0.15]
        if abs(float(footprint_xz[2]) - float(footprint_xz[0])) < 1e-4 or abs(float(footprint_xz[3]) - float(footprint_xz[1])) < 1e-4:
            footprint_xz = [float(center[0]) - 0.15, float(center[2]) - 0.15, float(center[0]) + 0.15, float(center[2]) + 0.15]
        out.append(
            {
                "semantic_id": int(sid),
                "center_world": np.asarray([float(center[0]), float(center[1]), float(center[2])], dtype=np.float32),
                "footprint_xz": footprint_xz,
                "height_range_y": [
                    float(getattr(aabb, "bottom")),
                    float(getattr(aabb, "top")),
                ],
            }
        )
    return out


def choose_target_object_record(ref_world: np.ndarray, object_rows: list[dict], nav_y: float | None = None) -> dict | None:
    if not object_rows:
        return None
    ref = np.asarray(ref_world, dtype=np.float32)
    candidates = list(object_rows)
    if nav_y is not None:
        floor_rows = [
            row
            for row in candidates
            if target_world_is_on_current_floor(np.asarray(row["center_world"], dtype=np.float32), nav_y=float(nav_y))
        ]
        if floor_rows:
            candidates = floor_rows
    return min(
        candidates,
        key=lambda row: float(np.linalg.norm((np.asarray(row["center_world"], dtype=np.float32) - ref)[[0, 2]])),
    )


def sanitize_target_footprint_xz(target_object: dict | None, target_world: np.ndarray) -> list[float] | None:
    if target_object is None:
        return None
    raw = target_object.get("footprint_xz")
    if raw is None or len(raw) != 4:
        return None
    x0, z0, x1, z1 = [float(v) for v in raw]
    tx = float(target_world[0])
    tz = float(target_world[2])
    if not np.all(np.isfinite([x0, z0, x1, z1, tx, tz])):
        return None
    lo_x = min(x0, x1)
    hi_x = max(x0, x1)
    lo_z = min(z0, z1)
    hi_z = max(z0, z1)
    width = hi_x - lo_x
    depth = hi_z - lo_z
    if width < 0.05 or depth < 0.05:
        return None
    if width > 4.5 or depth > 4.5:
        return None
    cx = 0.5 * (lo_x + hi_x)
    cz = 0.5 * (lo_z + hi_z)
    if math.hypot(cx - tx, cz - tz) > 2.0:
        return None
    return [lo_x, lo_z, hi_x, hi_z]


def target_world_is_on_current_floor(
    target_world: np.ndarray,
    *,
    nav_y: float,
    max_below_m: float = 0.35,
    max_above_m: float = 2.2,
) -> bool:
    dy = float(target_world[1]) - float(nav_y)
    return (-float(max_below_m)) <= dy <= float(max_above_m)


def project_topdown_world_xz_to_pixel(
    *,
    world_x: float,
    world_z: float,
    node_world: np.ndarray,
    camera_above_m: float,
    width: int,
    height: int,
    hfov_deg: float,
) -> tuple[float, float]:
    dx = float(world_x) - float(node_world[0])
    dz = float(world_z) - float(node_world[2])
    hfov_rad = math.radians(float(hfov_deg))
    fx = 0.5 * float(width) / math.tan(0.5 * hfov_rad)
    fy = fx
    cx = 0.5 * (float(width) - 1.0)
    cy = 0.5 * (float(height) - 1.0)
    depth = max(1e-4, float(camera_above_m))
    u = cx + fx * dx / depth
    v = cy - fy * dz / depth
    return u, v


def project_topdown_footprint_bbox(
    *,
    footprint_xz: list[float],
    node_world: np.ndarray,
    camera_above_m: float,
    width: int,
    height: int,
    hfov_deg: float,
) -> list[int] | None:
    x0, z0, x1, z1 = [float(v) for v in footprint_xz]
    corners = [
        (x0, z0),
        (x0, z1),
        (x1, z0),
        (x1, z1),
    ]
    pixels = [
        project_topdown_world_xz_to_pixel(
            world_x=wx,
            world_z=wz,
            node_world=node_world,
            camera_above_m=camera_above_m,
            width=width,
            height=height,
            hfov_deg=hfov_deg,
        )
        for wx, wz in corners
    ]
    xs = [float(p[0]) for p in pixels]
    ys = [float(p[1]) for p in pixels]
    pad = 6.0
    x_min = int(np.clip(math.floor(min(xs) - pad), 0, max(0, width - 1)))
    y_min = int(np.clip(math.floor(min(ys) - pad), 0, max(0, height - 1)))
    x_max = int(np.clip(math.ceil(max(xs) + pad), x_min + 1, max(1, width)))
    y_max = int(np.clip(math.ceil(max(ys) + pad), y_min + 1, max(1, height)))
    if x_max <= 0 or y_max <= 0 or x_min >= width or y_min >= height:
        return None
    return [x_min, y_min, x_max, y_max]


def estimate_target_world_from_depth(
    *,
    matched_item: dict,
    depth_obs: np.ndarray | None,
    node_world: np.ndarray,
    sensor_height_m: float,
    yaw_deg: float,
    pitch_down_deg: float,
    intrinsics: dict | None,
) -> np.ndarray | None:
    if depth_obs is None:
        return None
    mask = np.asarray(matched_item.get("mask"), dtype=np.uint8) > 0
    if mask.size == 0 or not np.any(mask):
        return None
    depth = np.asarray(depth_obs, dtype=np.float32)
    valid = mask & np.isfinite(depth) & (depth > 0.05)
    if not np.any(valid):
        return None

    ys, xs = np.nonzero(valid)
    depth_values = depth[valid]
    pick_idx = int(np.argmin(np.abs(depth_values - np.median(depth_values))))
    py = int(ys[pick_idx])
    px = int(xs[pick_idx])
    z = float(depth_values[pick_idx])

    width = int(depth.shape[1])
    height = int(depth.shape[0])
    fx = fy = None
    cx = cy = None
    if intrinsics:
        fx = intrinsics.get("fx")
        fy = intrinsics.get("fy")
        cx = intrinsics.get("cx")
        cy = intrinsics.get("cy")
    if fx is None or fy is None or cx is None or cy is None:
        hfov_deg = float(intrinsics.get("hfov_deg", 90.0)) if intrinsics else 90.0
        hfov_rad = math.radians(hfov_deg)
        fx = 0.5 * float(width) / math.tan(0.5 * hfov_rad)
        fy = float(fx)
        cx = 0.5 * (float(width) - 1.0)
        cy = 0.5 * (float(height) - 1.0)

    x_cam = (float(px) - float(cx)) * z / float(fx)
    y_cam = (float(py) - float(cy)) * z / float(fy)
    point_cam = np.asarray([x_cam, y_cam, -z], dtype=np.float32)

    pitch_rad = -math.radians(float(pitch_down_deg))
    cp = math.cos(pitch_rad)
    sp = math.sin(pitch_rad)
    rot_pitch = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, cp, -sp],
            [0.0, sp, cp],
        ],
        dtype=np.float32,
    )
    yaw_rad = math.radians(float(yaw_deg))
    cyaw = math.cos(yaw_rad)
    syaw = math.sin(yaw_rad)
    rot_yaw = np.asarray(
        [
            [cyaw, 0.0, syaw],
            [0.0, 1.0, 0.0],
            [-syaw, 0.0, cyaw],
        ],
        dtype=np.float32,
    )
    point_world = rot_yaw @ (rot_pitch @ point_cam)
    camera_world = np.asarray(node_world, dtype=np.float32) + np.asarray([0.0, float(sensor_height_m), 0.0], dtype=np.float32)
    return camera_world + point_world


def refine_match_mask(
    *,
    matched_item: dict,
    llm_bbox_xyxy: list[int],
    min_area_px: int = 24,
    pad_px: int = 6,
) -> np.ndarray:
    mask = np.asarray(matched_item.get("mask"), dtype=np.uint8) > 0
    if mask.size == 0 or not np.any(mask):
        return np.zeros_like(mask, dtype=np.uint8)
    h, w = mask.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in llm_bbox_xyxy]
    x0 = max(0, x0 - int(pad_px))
    y0 = max(0, y0 - int(pad_px))
    x1 = min(w, x1 + int(pad_px))
    y1 = min(h, y1 + int(pad_px))
    bbox_mask = np.zeros_like(mask, dtype=bool)
    bbox_mask[y0:y1, x0:x1] = True
    refined = mask & bbox_mask
    if int(np.count_nonzero(refined)) < int(min_area_px):
        refined = mask.copy()
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(refined.astype(np.uint8), connectivity=8)
    if int(nlab) <= 1:
        return refined.astype(np.uint8)
    cx = 0.5 * float(llm_bbox_xyxy[0] + llm_bbox_xyxy[2])
    cy = 0.5 * float(llm_bbox_xyxy[1] + llm_bbox_xyxy[3])
    best_lab = -1
    best_score = None
    for lab in range(1, int(nlab)):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < int(min_area_px):
            continue
        lx = float(stats[lab, cv2.CC_STAT_LEFT])
        ly = float(stats[lab, cv2.CC_STAT_TOP])
        lw = float(stats[lab, cv2.CC_STAT_WIDTH])
        lh = float(stats[lab, cv2.CC_STAT_HEIGHT])
        dist = math.hypot((lx + 0.5 * lw) - cx, (ly + 0.5 * lh) - cy)
        score = (dist, -area)
        if best_score is None or score < best_score:
            best_score = score
            best_lab = int(lab)
    if best_lab < 1:
        return refined.astype(np.uint8)
    return (labels == best_lab).astype(np.uint8)


def backproject_depth_mask_to_world_points(
    *,
    depth_obs: np.ndarray | None,
    mask_u8: np.ndarray,
    node_world: np.ndarray,
    sensor_height_m: float,
    yaw_deg: float,
    pitch_down_deg: float,
    intrinsics: dict | None,
    max_points: int = 2500,
) -> np.ndarray | None:
    if depth_obs is None:
        return None
    mask = np.asarray(mask_u8, dtype=np.uint8) > 0
    if mask.size == 0 or not np.any(mask):
        return None
    depth = np.asarray(depth_obs, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    valid = mask & np.isfinite(depth) & (depth > 0.05)
    if not np.any(valid):
        return None
    ys, xs = np.nonzero(valid)
    z = depth[valid].astype(np.float32)
    if int(xs.size) > int(max_points):
        sample_idx = np.linspace(0, int(xs.size) - 1, int(max_points), dtype=np.int32)
        xs = xs[sample_idx]
        ys = ys[sample_idx]
        z = z[sample_idx]

    width = int(depth.shape[1])
    height = int(depth.shape[0])
    fx = fy = cx = cy = None
    if intrinsics:
        fx = intrinsics.get("fx")
        fy = intrinsics.get("fy")
        cx = intrinsics.get("cx")
        cy = intrinsics.get("cy")
    if fx is None or fy is None or cx is None or cy is None:
        hfov_deg = float(intrinsics.get("hfov_deg", 90.0)) if intrinsics else 90.0
        hfov_rad = math.radians(hfov_deg)
        fx = 0.5 * float(width) / math.tan(0.5 * hfov_rad)
        fy = float(fx)
        cx = 0.5 * (float(width) - 1.0)
        cy = 0.5 * (float(height) - 1.0)

    x_cam = (xs.astype(np.float32) - float(cx)) * z / float(fx)
    y_cam = (ys.astype(np.float32) - float(cy)) * z / float(fy)
    pts_cam = np.stack([x_cam, y_cam, -z], axis=1)

    pitch_rad = -math.radians(float(pitch_down_deg))
    cp = math.cos(pitch_rad)
    sp = math.sin(pitch_rad)
    rot_pitch = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, cp, -sp],
            [0.0, sp, cp],
        ],
        dtype=np.float32,
    )
    yaw_rad = math.radians(float(yaw_deg))
    cyaw = math.cos(yaw_rad)
    syaw = math.sin(yaw_rad)
    rot_yaw = np.asarray(
        [
            [cyaw, 0.0, syaw],
            [0.0, 1.0, 0.0],
            [-syaw, 0.0, cyaw],
        ],
        dtype=np.float32,
    )
    rot = rot_yaw @ rot_pitch
    camera_world = np.asarray(node_world, dtype=np.float32) + np.asarray([0.0, float(sensor_height_m), 0.0], dtype=np.float32)
    pts_world = (rot @ pts_cam.T).T + camera_world.reshape(1, 3)
    pts_world = pts_world[np.all(np.isfinite(pts_world), axis=1)]
    if pts_world.size == 0:
        return None
    return pts_world.astype(np.float32)


def summarize_target_world_points(
    points_world: np.ndarray | None,
    *,
    min_points: int = 12,
) -> dict | None:
    if points_world is None:
        return None
    pts = np.asarray(points_world, dtype=np.float32)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if int(len(pts)) < int(min_points):
        return None
    xz = pts[:, [0, 2]]
    med = np.median(xz, axis=0)
    dist = np.linalg.norm(xz - med.reshape(1, 2), axis=1)
    keep_thr = max(float(np.percentile(dist, 85.0)), 0.08) + 0.08
    keep = dist <= keep_thr
    if int(np.count_nonzero(keep)) >= int(max(min_points, 0.3 * len(pts))):
        pts = pts[keep]
    px = np.percentile(pts[:, 0], [5.0, 50.0, 95.0])
    py = np.percentile(pts[:, 1], [10.0, 50.0, 90.0])
    pz = np.percentile(pts[:, 2], [5.0, 50.0, 95.0])
    x0, x1 = float(px[0]), float(px[2])
    z0, z1 = float(pz[0]), float(pz[2])
    if (x1 - x0) < 0.12:
        cx = float(px[1])
        x0 = cx - 0.06
        x1 = cx + 0.06
    if (z1 - z0) < 0.12:
        cz = float(pz[1])
        z0 = cz - 0.06
        z1 = cz + 0.06
    sample_count = min(len(pts), 256)
    sample_idx = np.linspace(0, len(pts) - 1, sample_count, dtype=np.int32)
    pts_sample = pts[sample_idx]
    return {
        "center_world": np.asarray([float(px[1]), float(py[1]), float(pz[1])], dtype=np.float32),
        "bounds_xz": [float(x0), float(z0), float(x1), float(z1)],
        "points_world_sampled": pts_sample.astype(np.float32),
        "point_count": int(len(pts)),
    }


def project_world_points_bbox_exact(
    *,
    render_camera: Any,
    world_points: np.ndarray | list[list[float]],
    width: int,
    height: int,
    pad_px: int = 4,
) -> dict | None:
    pts = np.asarray(world_points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] != 3:
        return None
    if int(pts.shape[0]) > 3000:
        sample_idx = np.linspace(0, int(pts.shape[0]) - 1, 3000, dtype=np.int32)
        pts = pts[sample_idx]
    proj_rows = [project_world_to_pixel_exact(render_camera, p, int(width), int(height)) for p in pts]
    valid = [row for row in proj_rows if bool(row.get("valid", False))]
    if not valid:
        return None
    xs = np.asarray([float(row["u"]) for row in valid], dtype=np.float32)
    ys = np.asarray([float(row["v"]) for row in valid], dtype=np.float32)
    x0 = int(np.clip(math.floor(float(xs.min()) - float(pad_px)), 0, max(0, int(width) - 1)))
    y0 = int(np.clip(math.floor(float(ys.min()) - float(pad_px)), 0, max(0, int(height) - 1)))
    x1 = int(np.clip(math.ceil(float(xs.max()) + float(pad_px)), x0 + 1, max(1, int(width))))
    y1 = int(np.clip(math.ceil(float(ys.max()) + float(pad_px)), y0 + 1, max(1, int(height))))
    return {
        "bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "center_xy": [int(np.clip(round(float(np.median(xs))), 0, max(0, int(width) - 1))), int(np.clip(round(float(np.median(ys))), 0, max(0, int(height) - 1)))],
        "projected_point_count": int(len(valid)),
    }


def choose_target_world_xz(node_world: np.ndarray, centers_world: list[np.ndarray]) -> np.ndarray:
    if not centers_world:
        return np.asarray(node_world, dtype=np.float32).copy()
    ref = np.asarray(node_world, dtype=np.float32)
    best = min(centers_world, key=lambda p: float(np.linalg.norm((np.asarray(p) - ref)[[0, 2]])))
    return np.asarray(best, dtype=np.float32)


def build_local_observation_fov_geometry(
    *,
    crop_world: dict,
    target_rows: list[dict],
    obs_hfov_deg: float,
) -> dict | None:
    stand_x = float(crop_world["stand_x"])
    stand_z = float(crop_world["stand_z"])
    right_x = float(crop_world["right_x"])
    right_z = float(crop_world["right_z"])
    forward_x = float(crop_world["forward_x"])
    forward_z = float(crop_world["forward_z"])
    u_min = float(crop_world["u_min"])
    u_max = float(crop_world["u_max"])
    f_front = float(crop_world["f_front"])

    forward_reach_m = 0.9
    for row in target_rows:
        target_world = np.asarray(row.get("target_world"), dtype=np.float32)
        if target_world.shape != (3,) or not np.all(np.isfinite(target_world)):
            continue
        dx = float(target_world[0]) - stand_x
        dz = float(target_world[2]) - stand_z
        f = dx * forward_x + dz * forward_z
        if np.isfinite(f):
            forward_reach_m = max(forward_reach_m, float(f) + 0.35)
    forward_reach_m = float(np.clip(forward_reach_m, 0.9, max(1.0, f_front - 0.08)))

    half_angle_rad = 0.5 * math.radians(float(np.clip(float(obs_hfov_deg), 10.0, 160.0)))
    half_span_m = float(forward_reach_m * math.tan(half_angle_rad))
    max_half_span_m = max(0.20, 0.5 * max(0.30, (u_max - u_min) - 0.10))
    half_span_m = float(np.clip(half_span_m, 0.20, max_half_span_m))

    def _world_point(u: float, f: float) -> list[float]:
        return [
            float(stand_x + right_x * float(u) + forward_x * float(f)),
            float(stand_z + right_z * float(u) + forward_z * float(f)),
        ]

    return {
        "obs_hfov_deg": float(obs_hfov_deg),
        "forward_reach_m": float(forward_reach_m),
        "half_span_m": float(half_span_m),
        "polygon_world_xz": [
            [float(stand_x), float(stand_z)],
            _world_point(-half_span_m, forward_reach_m),
            _world_point(half_span_m, forward_reach_m),
        ],
        "forward_tip_world_xz": _world_point(0.0, min(float(f_front) - 0.06, float(forward_reach_m) * 0.72)),
    }


def build_floor_nav_cache(pathfinder, *, nav_y: float, meters_per_pixel: float) -> dict:
    lower_b, upper_b = pathfinder.get_bounds()
    lower = np.asarray(lower_b, dtype=np.float32)
    upper = np.asarray(upper_b, dtype=np.float32)
    mpp = max(0.02, float(meters_per_pixel))
    try:
        nav_grid = np.asarray(pathfinder.get_topdown_view(float(mpp), float(nav_y)), dtype=np.uint8)
    except TypeError:
        nav_grid = np.asarray(pathfinder.get_topdown_view(float(mpp), float(nav_y), 0.05), dtype=np.uint8)
    nav_bool = nav_grid > 0
    clearance_px = cv2.distanceTransform(nav_bool.astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    return {
        "nav_y": float(nav_y),
        "meters_per_pixel": float(mpp),
        "lower": lower,
        "upper": upper,
        "nav_bool": nav_bool,
        "clearance_px": clearance_px,
    }


def nav_world_to_grid(nav_cache: dict, x: float, z: float) -> tuple[int, int]:
    lower = np.asarray(nav_cache["lower"], dtype=np.float32)
    nav_bool = np.asarray(nav_cache["nav_bool"], dtype=bool)
    h, w = nav_bool.shape[:2]
    mpp = float(nav_cache["meters_per_pixel"])
    col = int(np.clip(round((float(x) - float(lower[0])) / mpp - 0.5), 0, w - 1))
    row = int(np.clip(round((float(z) - float(lower[2])) / mpp - 0.5), 0, h - 1))
    return col, row


def planar_distance_to_footprint(node_world: np.ndarray, footprint_xz: list[float]) -> float:
    x0, z0, x1, z1 = [float(v) for v in footprint_xz]
    px = float(node_world[0])
    pz = float(node_world[2])
    dx = max(x0 - px, 0.0, px - x1)
    dz = max(z0 - pz, 0.0, pz - z1)
    return float(math.hypot(dx, dz))


def planar_distance_to_target(node_world: np.ndarray, target_world: np.ndarray, target_object: dict | None) -> float:
    sane_footprint = sanitize_target_footprint_xz(target_object, np.asarray(target_world, dtype=np.float32))
    if sane_footprint is not None:
        return planar_distance_to_footprint(node_world, sane_footprint)
    delta = np.asarray(target_world, dtype=np.float32) - np.asarray(node_world, dtype=np.float32)
    return float(np.linalg.norm(delta[[0, 2]]))


def estimate_target_draw_radius_m(target_object: dict | None) -> float:
    if target_object is None:
        return 0.35
    raw = target_object.get("footprint_xz") if isinstance(target_object, dict) else None
    if raw is None or len(raw) != 4:
        return 0.35
    x0, z0, x1, z1 = [float(v) for v in raw]
    width = abs(x1 - x0)
    depth = abs(z1 - z0)
    if width < 0.05 or depth < 0.05 or width > 4.5 or depth > 4.5:
        return 0.35
    radius = 0.25 * max(width, depth)
    return float(np.clip(radius, 0.25, 0.70))


def choose_target_crop_bounds_xz(row: dict) -> list[float]:
    target_world = np.asarray(row["target_world"], dtype=np.float32)
    visible_bounds = row.get("target_surface_bounds_xz")
    if visible_bounds is not None and len(visible_bounds) == 4:
        vals = [float(v) for v in visible_bounds]
        if np.all(np.isfinite(vals)) and vals[2] > vals[0] and vals[3] > vals[1]:
            return vals

    target_object = row.get("target_object")
    sane_footprint = sanitize_target_footprint_xz(target_object, target_world)
    if sane_footprint is None:
        radius_m = estimate_target_draw_radius_m(target_object)
        cx = float(target_world[0])
        cz = float(target_world[2])
        return [cx - radius_m, cz - radius_m, cx + radius_m, cz + radius_m]

    x0, z0, x1, z1 = [float(v) for v in sane_footprint]
    cx = float(target_world[0])
    cz = float(target_world[2])
    half_x = min(max(abs(float(x1) - cx), abs(cx - float(x0))), 0.85)
    half_z = min(max(abs(float(z1) - cz), abs(cz - float(z0))), 0.85)
    half_x = max(0.20, float(half_x))
    half_z = max(0.20, float(half_z))
    return [cx - half_x, cz - half_z, cx + half_x, cz + half_z]


def compute_local_crop_world_bounds(
    *,
    node_world: np.ndarray,
    target_world: np.ndarray,
    target_object: dict | None,
    margin_m: float,
    min_side_m: float,
) -> dict:
    target_radius_m = estimate_target_draw_radius_m(target_object)
    xs = [float(node_world[0]), float(target_world[0]) - target_radius_m, float(target_world[0]) + target_radius_m]
    zs = [float(node_world[2]), float(target_world[2]) - target_radius_m, float(target_world[2]) + target_radius_m]
    x_min = min(xs) - float(margin_m)
    x_max = max(xs) + float(margin_m)
    z_min = min(zs) - float(margin_m)
    z_max = max(zs) + float(margin_m)
    side = max(float(min_side_m), float(x_max - x_min), float(z_max - z_min))
    cx = 0.5 * (x_min + x_max)
    cz = 0.5 * (z_min + z_max)
    half = 0.5 * side
    return {
        "x_min": float(cx - half),
        "x_max": float(cx + half),
        "z_min": float(cz - half),
        "z_max": float(cz + half),
        "side_m": float(side),
    }


def world_xz_to_rectified_output(
    *,
    world_x: float,
    world_z: float,
    crop_world: dict,
    out_width: int,
    out_height: int,
) -> tuple[int, int]:
    if str(crop_world.get("mapping", "")) == "oriented_forward":
        stand_x = float(crop_world["stand_x"])
        stand_z = float(crop_world["stand_z"])
        right_x = float(crop_world["right_x"])
        right_z = float(crop_world["right_z"])
        forward_x = float(crop_world["forward_x"])
        forward_z = float(crop_world["forward_z"])
        u_min = float(crop_world["u_min"])
        u_max = float(crop_world["u_max"])
        f_front = float(crop_world["f_front"])
        f_back = float(crop_world["f_back"])
        dx = float(world_x) - stand_x
        dz = float(world_z) - stand_z
        u = dx * right_x + dz * right_z
        f = dx * forward_x + dz * forward_z
        px = int(
            np.clip(
                round((u - u_min) * float(out_width - 1) / max(1e-6, u_max - u_min)),
                0,
                max(0, int(out_width) - 1),
            )
        )
        py = int(
            np.clip(
                round((f_front - f) * float(out_height - 1) / max(1e-6, f_front - f_back)),
                0,
                max(0, int(out_height) - 1),
            )
        )
        return px, py

    x_min = float(crop_world["x_min"])
    x_max = float(crop_world["x_max"])
    z_min = float(crop_world["z_min"])
    z_max = float(crop_world["z_max"])
    px = int(
        np.clip(
            round((float(world_x) - x_min) * float(out_width - 1) / max(1e-6, x_max - x_min)),
            0,
            max(0, int(out_width) - 1),
        )
    )
    py = int(
        np.clip(
            round((float(world_z) - z_min) * float(out_height - 1) / max(1e-6, z_max - z_min)),
            0,
            max(0, int(out_height) - 1),
        )
    )
    return px, py


def world_xz_bbox_to_rectified_output(
    *,
    bounds_xz: list[float],
    crop_world: dict,
    out_width: int,
    out_height: int,
) -> list[int]:
    x0, z0, x1, z1 = [float(v) for v in bounds_xz]
    corners = [
        (min(x0, x1), min(z0, z1)),
        (min(x0, x1), max(z0, z1)),
        (max(x0, x1), min(z0, z1)),
        (max(x0, x1), max(z0, z1)),
    ]
    pts = [
        world_xz_to_rectified_output(
            world_x=float(cx),
            world_z=float(cz),
            crop_world=crop_world,
            out_width=int(out_width),
            out_height=int(out_height),
        )
        for (cx, cz) in corners
    ]
    xs = [int(p[0]) for p in pts]
    ys = [int(p[1]) for p in pts]
    return [int(min(xs)), int(min(ys)), int(max(xs) + 1), int(max(ys) + 1)]


def rectified_output_to_world_xz(
    *,
    output_xy: tuple[int, int],
    crop_world: dict,
    out_width: int,
    out_height: int,
) -> np.ndarray:
    if str(crop_world.get("mapping", "")) == "oriented_forward":
        stand_x = float(crop_world["stand_x"])
        stand_z = float(crop_world["stand_z"])
        right_x = float(crop_world["right_x"])
        right_z = float(crop_world["right_z"])
        forward_x = float(crop_world["forward_x"])
        forward_z = float(crop_world["forward_z"])
        u_min = float(crop_world["u_min"])
        u_max = float(crop_world["u_max"])
        f_front = float(crop_world["f_front"])
        f_back = float(crop_world["f_back"])
        u = u_min + (float(output_xy[0]) + 0.5) * (u_max - u_min) / max(1.0, float(out_width))
        f = f_front - (float(output_xy[1]) + 0.5) * (f_front - f_back) / max(1.0, float(out_height))
        world_x = stand_x + right_x * u + forward_x * f
        world_z = stand_z + right_z * u + forward_z * f
        return np.asarray([float(world_x), float(world_z)], dtype=np.float32)

    x_min = float(crop_world["x_min"])
    x_max = float(crop_world["x_max"])
    z_min = float(crop_world["z_min"])
    z_max = float(crop_world["z_max"])
    world_x = x_min + (float(output_xy[0]) + 0.5) * (x_max - x_min) / max(1.0, float(out_width))
    world_z = z_min + (float(output_xy[1]) + 0.5) * (z_max - z_min) / max(1.0, float(out_height))
    return np.asarray([float(world_x), float(world_z)], dtype=np.float32)


def compute_topdown_hit_world_y_map(
    *,
    depth_obs: np.ndarray,
    camera_center_world: np.ndarray,
    hfov_deg: float,
) -> np.ndarray:
    depth = np.asarray(depth_obs, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    h, w = depth.shape[:2]
    hfov_rad = math.radians(float(hfov_deg))
    fx = 0.5 * float(w) / math.tan(0.5 * hfov_rad)
    fy = float(fx)
    cx = 0.5 * (float(w) - 1.0)
    cy = 0.5 * (float(h) - 1.0)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    dir_x = (xs - float(cx)) / float(fx)
    dir_y = (ys - float(cy)) / float(fy)
    dir_z = -np.ones_like(dir_x, dtype=np.float32)
    dir_norm = np.sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z)
    dir_norm = np.maximum(dir_norm, 1e-6)
    point_cam_z = dir_z / dir_norm * depth
    hit_world_y = float(camera_center_world[1]) + point_cam_z
    hit_world_y[~np.isfinite(depth)] = np.nan
    return hit_world_y.astype(np.float32)


def build_rectified_world_xz_grids(
    *,
    crop_world: dict,
    out_width: int,
    out_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    out_w = int(out_width)
    out_h = int(out_height)
    if str(crop_world.get("mapping", "")) == "oriented_forward":
        u_min = float(crop_world["u_min"])
        u_max = float(crop_world["u_max"])
        f_front = float(crop_world["f_front"])
        f_back = float(crop_world["f_back"])
        stand_x = float(crop_world["stand_x"])
        stand_z = float(crop_world["stand_z"])
        right_x = float(crop_world["right_x"])
        right_z = float(crop_world["right_z"])
        forward_x = float(crop_world["forward_x"])
        forward_z = float(crop_world["forward_z"])

        us = np.linspace(u_min, u_max, out_w, dtype=np.float32)
        fs = np.linspace(f_front, f_back, out_h, dtype=np.float32)
        grid_u, grid_f = np.meshgrid(us, fs)
        grid_x = stand_x + right_x * grid_u + forward_x * grid_f
        grid_z = stand_z + right_z * grid_u + forward_z * grid_f
        return grid_x.astype(np.float32), grid_z.astype(np.float32)

    xs = np.linspace(float(crop_world["x_min"]), float(crop_world["x_max"]), out_w, dtype=np.float32)
    zs = np.linspace(float(crop_world["z_min"]), float(crop_world["z_max"]), out_h, dtype=np.float32)
    grid_x, grid_z = np.meshgrid(xs, zs)
    return grid_x.astype(np.float32), grid_z.astype(np.float32)


def build_rectified_local_topdown_rgb(
    *,
    full_rgb: np.ndarray,
    full_valid_mask: np.ndarray,
    world_xz_to_fullpixel_h: np.ndarray | None,
    camera_center_world: np.ndarray,
    camera_above_m: float,
    hfov_deg: float,
    crop_world: dict,
    out_width: int,
    out_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    grid_x, grid_z = build_rectified_world_xz_grids(
        crop_world=crop_world,
        out_width=int(out_width),
        out_height=int(out_height),
    )

    if world_xz_to_fullpixel_h is not None:
        map_x, map_y = world_xz_grid_to_fullpixel_map(
            grid_x=grid_x,
            grid_z=grid_z,
            world_xz_to_fullpixel_h=np.asarray(world_xz_to_fullpixel_h, dtype=np.float32),
        )
    else:
        full_h, full_w = full_rgb.shape[:2]
        hfov_rad = math.radians(float(hfov_deg))
        fx = 0.5 * float(full_w) / math.tan(0.5 * hfov_rad)
        fy = float(fx)
        cx = 0.5 * (float(full_w) - 1.0)
        cy = 0.5 * (float(full_h) - 1.0)
        depth = max(1e-4, float(camera_above_m))
        map_x = cx + fx * (grid_x - float(camera_center_world[0])) / depth
        map_y = cy - fy * (grid_z - float(camera_center_world[2])) / depth

    vis = cv2.remap(
        np.asarray(full_rgb, dtype=np.uint8),
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(230, 230, 230),
    )
    valid = cv2.remap(
        (np.asarray(full_valid_mask, dtype=np.uint8) * 255),
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid_bool = valid > 0
    if np.any(~valid_bool):
        rgb_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        inpaint_mask = ((~valid_bool).astype(np.uint8) * 255)
        keep_mask = valid_bool & np.any(vis > 0, axis=2)
        if np.any(keep_mask):
            fill_color = np.median(vis[keep_mask], axis=0).astype(np.uint8)
        else:
            fill_color = np.asarray([230, 230, 230], dtype=np.uint8)
        if float(np.count_nonzero(inpaint_mask)) / float(max(1, inpaint_mask.size)) < 0.45:
            rgb_bgr = cv2.inpaint(rgb_bgr, inpaint_mask, 5.0, cv2.INPAINT_TELEA)
        else:
            rgb_bgr[inpaint_mask > 0] = fill_color[::-1]
        vis = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    return vis, valid_bool


def rectify_full_topdown_mask(
    *,
    full_mask: np.ndarray,
    world_xz_to_fullpixel_h: np.ndarray | None,
    camera_center_world: np.ndarray,
    camera_above_m: float,
    hfov_deg: float,
    crop_world: dict,
    out_width: int,
    out_height: int,
) -> np.ndarray:
    mask_u8 = (np.asarray(full_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    grid_x, grid_z = build_rectified_world_xz_grids(
        crop_world=crop_world,
        out_width=int(out_width),
        out_height=int(out_height),
    )

    if world_xz_to_fullpixel_h is not None:
        map_x, map_y = world_xz_grid_to_fullpixel_map(
            grid_x=grid_x,
            grid_z=grid_z,
            world_xz_to_fullpixel_h=np.asarray(world_xz_to_fullpixel_h, dtype=np.float32),
        )
    else:
        full_h, full_w = mask_u8.shape[:2]
        hfov_rad = math.radians(float(hfov_deg))
        fx = 0.5 * float(full_w) / math.tan(0.5 * hfov_rad)
        fy = float(fx)
        cx = 0.5 * (float(full_w) - 1.0)
        cy = 0.5 * (float(full_h) - 1.0)
        depth = max(1e-4, float(camera_above_m))
        map_x = cx + fx * (grid_x - float(camera_center_world[0])) / depth
        map_y = cy - fy * (grid_z - float(camera_center_world[2])) / depth
    rect = cv2.remap(
        mask_u8,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return (rect > 0).astype(np.uint8)


def scale_local_point_to_output(x: float, y: float, crop_w: int, crop_h: int, out_w: int, out_h: int) -> tuple[int, int]:
    px = int(np.clip(round(float(x) * float(out_w) / max(1.0, float(crop_w))), 0, max(0, out_w - 1)))
    py = int(np.clip(round(float(y) * float(out_h) / max(1.0, float(crop_h))), 0, max(0, out_h - 1)))
    return px, py


def scale_local_bbox_to_output(box_xyxy: list[int], crop_w: int, crop_h: int, out_w: int, out_h: int) -> list[int]:
    x1, y1 = scale_local_point_to_output(box_xyxy[0], box_xyxy[1], crop_w, crop_h, out_w, out_h)
    x2, y2 = scale_local_point_to_output(box_xyxy[2], box_xyxy[3], crop_w, crop_h, out_w, out_h)
    x2 = max(x1 + 1, x2)
    y2 = max(y1 + 1, y2)
    return [x1, y1, x2, y2]


def render_local_nav_topdown(
    *,
    nav_cache: dict,
    node_world: np.ndarray,
    target_world: np.ndarray,
    target_name: str,
    target_object: dict | None,
    out_width: int,
    out_height: int,
    margin_m: float,
    min_side_m: float,
    dot_stride_px: int,
    min_clearance_m: float,
) -> tuple[np.ndarray, dict]:
    crop_world = compute_local_crop_world_bounds(
        node_world=node_world,
        target_world=target_world,
        target_object=target_object,
        margin_m=float(margin_m),
        min_side_m=float(min_side_m),
    )
    nav_bool = np.asarray(nav_cache["nav_bool"], dtype=bool)
    clearance_px = np.asarray(nav_cache["clearance_px"], dtype=np.float32)
    h, w = nav_bool.shape[:2]
    mpp = float(nav_cache["meters_per_pixel"])

    c0, r0 = nav_world_to_grid(nav_cache, crop_world["x_min"], crop_world["z_min"])
    c1, r1 = nav_world_to_grid(nav_cache, crop_world["x_max"], crop_world["z_max"])
    col_min = max(0, min(c0, c1))
    col_max = min(w, max(c0, c1) + 1)
    row_min = max(0, min(r0, r1))
    row_max = min(h, max(r0, r1) + 1)
    crop_nav = nav_bool[row_min:row_max, col_min:col_max]
    crop_clearance = clearance_px[row_min:row_max, col_min:col_max]
    crop_h, crop_w = crop_nav.shape[:2]
    if crop_h <= 1 or crop_w <= 1:
        raise RuntimeError("local nav crop is empty")

    base = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
    base[:] = (44, 44, 44)
    if np.any(crop_nav):
        local_clear = crop_clearance.copy()
        local_clear[~crop_nav] = 0.0
        if float(np.max(local_clear)) > 1e-6:
            norm = np.clip(local_clear / float(np.max(local_clear)), 0.0, 1.0)
        else:
            norm = np.zeros_like(local_clear, dtype=np.float32)
        nav_gray = np.clip(np.round(180.0 + 50.0 * norm), 0, 255).astype(np.uint8)
        for ch in range(3):
            base[..., ch] = np.where(crop_nav, nav_gray, base[..., ch])

    vis = cv2.resize(base, (int(out_width), int(out_height)), interpolation=cv2.INTER_NEAREST)

    node_col, node_row = nav_world_to_grid(nav_cache, float(node_world[0]), float(node_world[2]))
    stand_raw_x = node_col - col_min
    stand_raw_y = node_row - row_min
    stand_x, stand_y = scale_local_point_to_output(stand_raw_x, stand_raw_y, crop_w, crop_h, int(out_width), int(out_height))

    dot_stride = max(6, int(dot_stride_px))
    min_clearance_px = float(min_clearance_m) / max(mpp, 1e-6)
    for raw_y in range(dot_stride // 2, crop_h, dot_stride):
        for raw_x in range(dot_stride // 2, crop_w, dot_stride):
            if not bool(crop_nav[raw_y, raw_x]):
                continue
            if float(crop_clearance[raw_y, raw_x]) < min_clearance_px:
                continue
            px, py = scale_local_point_to_output(raw_x, raw_y, crop_w, crop_h, int(out_width), int(out_height))
            cv2.circle(vis, (px, py), 2, (90, 200, 90), -1, lineType=cv2.LINE_AA)

    target_bbox_out = None
    target_center_out = None
    if target_object is not None:
        target_col, target_row = nav_world_to_grid(nav_cache, float(target_world[0]), float(target_world[2]))
        draw_radius_px = max(3, int(round(estimate_target_draw_radius_m(target_object) / max(mpp, 1e-6))))
        raw_box = [
            int(target_col - col_min - draw_radius_px),
            int(target_row - row_min - draw_radius_px),
            int(target_col - col_min + draw_radius_px + 1),
            int(target_row - row_min + draw_radius_px + 1),
        ]
        target_bbox_out = scale_local_bbox_to_output(raw_box, crop_w, crop_h, int(out_width), int(out_height))
        target_center_out = scale_local_point_to_output(target_col - col_min, target_row - row_min, crop_w, crop_h, int(out_width), int(out_height))
    else:
        target_col, target_row = nav_world_to_grid(nav_cache, float(target_world[0]), float(target_world[2]))
        target_center_out = scale_local_point_to_output(target_col - col_min, target_row - row_min, crop_w, crop_h, int(out_width), int(out_height))

    if target_bbox_out is not None:
        tx1, ty1, tx2, ty2 = [int(v) for v in target_bbox_out]
        overlay = vis.copy()
        cv2.circle(overlay, (int(round(0.5 * (tx1 + tx2))), int(round(0.5 * (ty1 + ty2)))), max(6, int(round(0.3 * max(tx2 - tx1, ty2 - ty1)))), (0, 220, 255), -1, lineType=cv2.LINE_AA)
        vis = cv2.addWeighted(overlay, 0.18, vis, 0.82, 0.0)
        cv2.circle(vis, (int(round(0.5 * (tx1 + tx2))), int(round(0.5 * (ty1 + ty2)))), max(6, int(round(0.3 * max(tx2 - tx1, ty2 - ty1)))), (0, 220, 255), 2, lineType=cv2.LINE_AA)
        target_center_out = (int(round(0.5 * (tx1 + tx2))), int(round(0.5 * (ty1 + ty2))))

    cv2.drawMarker(vis, (stand_x, stand_y), (255, 60, 60), markerType=cv2.MARKER_CROSS, markerSize=26, thickness=2)
    cv2.circle(vis, (stand_x, stand_y), 10, (255, 60, 60), 2, lineType=cv2.LINE_AA)
    cv2.putText(vis, "stand", (stand_x + 8, max(20, stand_y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, lineType=cv2.LINE_AA)
    cv2.putText(vis, "stand", (stand_x + 8, max(20, stand_y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 60, 60), 1, lineType=cv2.LINE_AA)

    if target_center_out is not None:
        cv2.line(vis, (stand_x, stand_y), (int(target_center_out[0]), int(target_center_out[1])), (0, 220, 255), 2, lineType=cv2.LINE_AA)
        cv2.circle(vis, (int(target_center_out[0]), int(target_center_out[1])), 6, (0, 220, 255), -1, lineType=cv2.LINE_AA)
        cv2.putText(vis, str(target_name), (int(target_center_out[0]) + 8, max(20, int(target_center_out[1]) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, lineType=cv2.LINE_AA)
        cv2.putText(vis, str(target_name), (int(target_center_out[0]) + 8, max(20, int(target_center_out[1]) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, lineType=cv2.LINE_AA)

    dist_m = planar_distance_to_target(node_world, target_world, target_object)
    cv2.putText(vis, f"local nav topdown d={dist_m:.2f}m side={crop_world['side_m']:.2f}m", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2, lineType=cv2.LINE_AA)
    cv2.putText(vis, f"local nav topdown d={dist_m:.2f}m side={crop_world['side_m']:.2f}m", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    return vis, {
        "target_distance_m": float(dist_m),
        "crop_world_bounds_xz": crop_world,
        "target_bbox_xyxy": [int(v) for v in target_bbox_out] if target_bbox_out is not None else None,
        "target_center_xy": [int(target_center_out[0]), int(target_center_out[1])] if target_center_out is not None else None,
        "stand_xy": [int(stand_x), int(stand_y)],
        "meters_per_pixel": float(mpp),
    }


def draw_observation_overlay(rgb: np.ndarray, matches: list[dict]) -> np.ndarray:
    vis = np.asarray(rgb, dtype=np.uint8).copy()
    for idx, match in enumerate(matches):
        color = palette_rgb(idx)
        llm_target = dict(match["llm_target"])

        lx1, ly1, lx2, ly2 = [int(v) for v in llm_target["bbox_xyxy"]]
        cv2.rectangle(vis, (lx1, ly1), (lx2, ly2), (0, 0, 0), 4, lineType=cv2.LINE_AA)
        cv2.rectangle(vis, (lx1, ly1), (lx2, ly2), color, 2, lineType=cv2.LINE_AA)
    return vis


def render_local_rgb_topdown(
    *,
    sim: habitat_sim.Simulator,
    nav_cache: dict,
    node_world: np.ndarray,
    nav_y: float,
    floor_levels: list[float] | tuple[float, ...],
    forward_world: np.ndarray | None,
    target_rows: list[dict],
    out_width: int,
    out_height: int,
    hfov_deg: float,
    margin_m: float,
    min_side_m: float,
    min_above_m: float,
    max_above_m: float,
    crop_fraction: float,
    cut_above_ground_m: float,
    cut_depth_margin_m: float,
    cut_inpaint_radius_px: int,
    dot_stride_px: int,
    min_clearance_m: float,
    semantic_open_kernel: int,
    semantic_close_kernel: int,
    semantic_min_area_px: int,
    topdown_ceiling_margin_m: float,
    topdown_ray_start_offset_m: float,
    view_yaw_deg: float,
    obs_hfov_deg: float,
    topdown_stand_y_fraction: float,
    topdown_view_margin_ratio: float,
    topdown_camera_height_scale: float,
    topdown_trim_invalid: bool,
    topdown_trim_margin_px: int,
    topdown_trim_min_valid_px: int,
    topdown_pad_gray_value: int,
    require_topdown_semantic: bool,
) -> tuple[np.ndarray, dict]:
    crop_world = compute_multi_target_crop_world_bounds(
        node_world=node_world,
        target_rows=target_rows,
        margin_m=float(margin_m),
        min_side_m=float(min_side_m),
        view_yaw_deg=float(view_yaw_deg),
        stand_y_fraction=float(topdown_stand_y_fraction),
    )
    desired_camera_above_m = estimate_topdown_camera_above_from_node(
        node_world=np.asarray(node_world, dtype=np.float32),
        crop_world=crop_world,
        width=int(out_width),
        height=int(out_height),
        hfov_deg=float(hfov_deg),
        crop_fraction=float(crop_fraction),
        min_above_m=float(min_above_m),
        max_above_m=float(max_above_m),
    )
    desired_camera_above_m = float(
        np.clip(
            float(desired_camera_above_m) * float(np.clip(float(topdown_camera_height_scale), 0.65, 1.25)),
            float(min_above_m),
            float(max_above_m),
        )
    )
    upward_hit_dist_m = upward_hit_distance(
        sim,
        np.asarray(node_world, dtype=np.float32),
        start_offset_m=float(topdown_ray_start_offset_m),
    )
    camera_above_limit_m = compute_node_topdown_camera_above(
        node_y=float(node_world[1]),
        floor_levels=[float(v) for v in floor_levels],
        desired_above_m=float(max_above_m),
        upward_hit_dist_m=upward_hit_dist_m,
        ceiling_margin_m=float(topdown_ceiling_margin_m),
    )
    camera_above_m = min(
        float(camera_above_limit_m),
        compute_node_topdown_camera_above(
            node_y=float(node_world[1]),
            floor_levels=[float(v) for v in floor_levels],
            desired_above_m=float(desired_camera_above_m),
            upward_hit_dist_m=upward_hit_dist_m,
            ceiling_margin_m=float(topdown_ceiling_margin_m),
        ),
    )
    full_rgb = None
    full_sem = None
    full_depth = None
    render_camera = None
    full_h = None
    full_w = None
    stand_proj_full = None
    target_center_proj_full: list[dict] = []
    target_point_bbox_full_list: list[dict | None] = []
    target_bbox_full_list: list[list[int] | None] = []
    margin_ratio = float(np.clip(float(topdown_view_margin_ratio), 0.01, 0.20))
    view_margin_px = max(12, int(round(margin_ratio * float(max(out_width, out_height)))))
    for _ in range(max(1, int(max_above_m > 0)) + int(4)):
        camera_center_world = np.asarray(
            [float(node_world[0]), float(nav_y) + float(camera_above_m), float(node_world[2])],
            dtype=np.float32,
        )
        full_rgb, full_sem, full_depth, render_camera = render_topdown_rgb_semantic(
            sim=sim,
            position_world=camera_center_world,
            rotation_quat=topdown_quat_no_tilt(0.0),
        )
        full_h, full_w = full_rgb.shape[:2]
        stand_proj_full = project_world_to_pixel_exact(render_camera, np.asarray(node_world, dtype=np.float32), int(full_w), int(full_h))
        target_center_proj_full = []
        target_point_bbox_full_list = []
        target_bbox_full_list = []
        all_ok = bool(stand_proj_full["valid"])
        if all_ok:
            all_ok = (
                view_margin_px <= float(stand_proj_full["u"]) < float(full_w - view_margin_px)
                and view_margin_px <= float(stand_proj_full["v"]) < float(full_h - view_margin_px)
            )
        for row in target_rows:
            target_world = np.asarray(row["target_world"], dtype=np.float32)
            proj = project_world_to_pixel_exact(render_camera, target_world, int(full_w), int(full_h))
            target_center_proj_full.append(proj)
            point_bbox_full = None
            if row.get("target_world_points_xyz") is not None:
                point_bbox_full = project_world_points_bbox_exact(
                    render_camera=render_camera,
                    world_points=np.asarray(row["target_world_points_xyz"], dtype=np.float32),
                    width=int(full_w),
                    height=int(full_h),
                    pad_px=5,
                )
            target_point_bbox_full_list.append(point_bbox_full)
            bbox_full = None
            if point_bbox_full is not None:
                bbox_full = list(point_bbox_full["bbox_xyxy"])
            else:
                target_object = row.get("target_object")
                if target_object is not None:
                    bbox_full = project_footprint_bbox_exact(
                        render_camera=render_camera,
                        footprint_xz=[float(v) for v in target_object["footprint_xz"]],
                        world_y=float(nav_y),
                        width=int(full_w),
                        height=int(full_h),
                    )
            target_bbox_full_list.append(bbox_full)
            if not bool(proj["valid"]):
                all_ok = False
                continue
            if not (view_margin_px <= float(proj["u"]) < float(full_w - view_margin_px) and view_margin_px <= float(proj["v"]) < float(full_h - view_margin_px)):
                all_ok = False
        if all_ok or float(camera_above_m) >= float(camera_above_limit_m) - 1e-6:
            break
        next_desired = min(float(max_above_m), float(camera_above_m) * float(1.15))
        camera_above_m = min(
            float(camera_above_limit_m),
            compute_node_topdown_camera_above(
                node_y=float(node_world[1]),
                floor_levels=[float(v) for v in floor_levels],
                desired_above_m=float(next_desired),
                upward_hit_dist_m=upward_hit_dist_m,
                ceiling_margin_m=float(topdown_ceiling_margin_m),
            ),
        )

    assert full_rgb is not None and full_sem is not None and full_depth is not None and render_camera is not None and full_h is not None and full_w is not None and stand_proj_full is not None
    world_xz_to_fullpixel_h = build_world_xz_to_fullpixel_homography(
        render_camera=render_camera,
        plane_y=float(nav_y),
        center_x=float(node_world[0]),
        center_z=float(node_world[2]),
        span_m=float(crop_world.get("side_m", 3.0)),
        width=int(full_w),
        height=int(full_h),
    )
    depth_keep_mask = np.ones((int(full_h), int(full_w)), dtype=bool)
    removed_mask_global = np.zeros((int(full_h), int(full_w)), dtype=bool)
    cut_meta = {
        "enabled": bool(float(cut_above_ground_m) > 0.0),
        "cut_above_ground_m": float(cut_above_ground_m),
        "cut_world_y_threshold_m": None,
        "removed_fraction": 0.0,
    }
    if float(cut_above_ground_m) > 0.0:
        full_depth = np.asarray(full_depth, dtype=np.float32)
        if full_depth.ndim == 3 and full_depth.shape[-1] == 1:
            full_depth = full_depth[..., 0]
        valid_depth = np.isfinite(full_depth) & (full_depth > 0.05)
        hit_world_y = compute_topdown_hit_world_y_map(
            depth_obs=full_depth,
            camera_center_world=camera_center_world,
            hfov_deg=float(hfov_deg),
        )
        max_world_y = float(nav_y) + float(cut_above_ground_m)
        depth_keep_mask = (~valid_depth) | (hit_world_y <= float(max_world_y))
        removed_mask = valid_depth & (~depth_keep_mask)
        removed_mask_global = removed_mask.copy()
        removed_fraction = float(np.count_nonzero(removed_mask)) / float(max(1, removed_mask.size))
        cut_meta["cut_world_y_threshold_m"] = float(max_world_y)
        cut_meta["removed_fraction"] = float(removed_fraction)
        if np.any(removed_mask):
            rgb_bgr = cv2.cvtColor(full_rgb, cv2.COLOR_RGB2BGR)
            inpaint_mask = (removed_mask.astype(np.uint8) * 255)
            if removed_fraction < 0.90:
                rgb_bgr = cv2.inpaint(rgb_bgr, inpaint_mask, float(max(1, int(cut_inpaint_radius_px))), cv2.INPAINT_TELEA)
            else:
                keep_mask = depth_keep_mask & np.any(full_rgb > 0, axis=2)
                if np.any(keep_mask):
                    fill_color = np.median(full_rgb[keep_mask], axis=0).astype(np.uint8)
                else:
                    fill_color = np.asarray([230, 230, 230], dtype=np.uint8)
                rgb_bgr[inpaint_mask > 0] = fill_color[::-1]
            full_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            full_sem = np.asarray(full_sem, dtype=np.int32).copy()
            full_sem[removed_mask] = 0

    invalid_mask = (~np.isfinite(full_depth)) | (np.asarray(full_depth, dtype=np.float32) <= 0.05) | (~np.any(full_rgb > 0, axis=2))
    if np.any(invalid_mask):
        invalid_fraction = float(np.count_nonzero(invalid_mask)) / float(max(1, invalid_mask.size))
        rgb_bgr = cv2.cvtColor(full_rgb, cv2.COLOR_RGB2BGR)
        inpaint_mask = (invalid_mask.astype(np.uint8) * 255)
        if invalid_fraction < 0.90:
            rgb_bgr = cv2.inpaint(rgb_bgr, inpaint_mask, 5.0, cv2.INPAINT_TELEA)
        else:
            keep_mask = (~invalid_mask) & np.any(full_rgb > 0, axis=2)
            if np.any(keep_mask):
                fill_color = np.median(full_rgb[keep_mask], axis=0).astype(np.uint8)
            else:
                fill_color = np.asarray([220, 220, 220], dtype=np.uint8)
            rgb_bgr[inpaint_mask > 0] = fill_color[::-1]
        full_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

    # Treat pixels removed by topdown cut as invalid for trim/crop, so artificial gray
    # fill from inpaint does not expand the retained topdown area.
    full_valid_mask = depth_keep_mask & (~invalid_mask) & (~removed_mask_global)
    topdown_sem_min_area = max(12, int(semantic_min_area_px) // 2)
    target_top_sem_rows: list[dict | None] = []
    for idx, row in enumerate(target_rows):
        preferred_bbox = target_bbox_full_list[idx] if idx < len(target_bbox_full_list) else None
        preferred_center = None
        if idx < len(target_center_proj_full):
            proj_row = target_center_proj_full[idx]
            if bool(proj_row.get("valid", False)):
                preferred_center = (float(proj_row["u"]), float(proj_row["v"]))
        sem_row = extract_topdown_semantic_target(
            semantic_obs=full_sem,
            semantic_ids=[int(v) for v in row["matched_item"]["semantic_ids"]],
            open_kernel=max(1, int(semantic_open_kernel) - 2),
            close_kernel=max(1, int(semantic_close_kernel) - 2),
            min_area_px=int(topdown_sem_min_area),
            preferred_bbox_xyxy=preferred_bbox,
            preferred_center_xy=preferred_center,
        )
        target_top_sem_rows.append(sem_row)
    vis_base, rectified_valid_mask_base = build_rectified_local_topdown_rgb(
        full_rgb=full_rgb,
        full_valid_mask=full_valid_mask,
        world_xz_to_fullpixel_h=world_xz_to_fullpixel_h,
        camera_center_world=camera_center_world,
        camera_above_m=float(camera_above_m),
        hfov_deg=float(hfov_deg),
        crop_world=crop_world,
        out_width=int(out_width),
        out_height=int(out_height),
    )
    if bool(topdown_trim_invalid):
        trim_transform = compute_trim_letterbox_transform(
            valid_mask=rectified_valid_mask_base,
            out_width=int(out_width),
            out_height=int(out_height),
            margin_px=int(topdown_trim_margin_px),
            min_valid_px=int(topdown_trim_min_valid_px),
        )
    else:
        trim_transform = {
            "enabled": False,
            "src_width": int(out_width),
            "src_height": int(out_height),
            "out_width": int(out_width),
            "out_height": int(out_height),
            "valid_px": int(np.count_nonzero(rectified_valid_mask_base)),
        }
    vis, rectified_valid_mask = apply_trim_letterbox_rgb(
        image_rgb=vis_base,
        valid_mask=rectified_valid_mask_base,
        trim_transform=trim_transform,
        pad_gray_value=int(topdown_pad_gray_value),
    )

    mpp = float(nav_cache["meters_per_pixel"])
    min_clearance_px = float(min_clearance_m) / max(mpp, 1e-6)
    c0, r0 = nav_world_to_grid(nav_cache, crop_world["x_min"], crop_world["z_min"])
    c1, r1 = nav_world_to_grid(nav_cache, crop_world["x_max"], crop_world["z_max"])
    nav_bool = np.asarray(nav_cache["nav_bool"], dtype=bool)
    clearance_px = np.asarray(nav_cache["clearance_px"], dtype=np.float32)
    h, w = nav_bool.shape[:2]
    col_min = max(0, min(c0, c1))
    col_max = min(w, max(c0, c1) + 1)
    row_min = max(0, min(r0, r1))
    row_max = min(h, max(r0, r1) + 1)
    dot_spacing_m = max(0.20, float(dot_stride_px) * float(crop_world["side_m"]) / max(1.0, float(out_width)))
    dot_stride_cells = max(1, int(round(dot_spacing_m / max(mpp, 1e-6))))
    for row in range(row_min, row_max, dot_stride_cells):
        for col in range(col_min, col_max, dot_stride_cells):
            if not bool(nav_bool[row, col]):
                continue
            if float(clearance_px[row, col]) < min_clearance_px:
                continue
            wx = float(nav_cache["lower"][0]) + (float(col) + 0.5) * mpp
            wz = float(nav_cache["lower"][2]) + (float(row) + 0.5) * mpp
            px, py = world_xz_to_rectified_output(
                world_x=float(wx),
                world_z=float(wz),
                crop_world=crop_world,
                out_width=int(out_width),
                out_height=int(out_height),
            )
            px, py = map_base_point_to_trimmed(xy=(int(px), int(py)), trim_transform=trim_transform)
            cv2.circle(vis, (px, py), 2, (80, 220, 80), -1, lineType=cv2.LINE_AA)

    stand_xy_base = world_xz_to_rectified_output(
        world_x=float(node_world[0]),
        world_z=float(node_world[2]),
        crop_world=crop_world,
        out_width=int(out_width),
        out_height=int(out_height),
    )
    stand_xy = map_base_point_to_trimmed(
        xy=(int(stand_xy_base[0]), int(stand_xy_base[1])),
        trim_transform=trim_transform,
    )
    obs_fov_meta = build_local_observation_fov_geometry(
        crop_world=crop_world,
        target_rows=target_rows,
        obs_hfov_deg=float(obs_hfov_deg),
    )
    forward_xy = None
    obs_fov_polygon_xy = None
    if obs_fov_meta is not None:
        poly_base = [
            world_xz_to_rectified_output(
                world_x=float(pt[0]),
                world_z=float(pt[1]),
                crop_world=crop_world,
                out_width=int(out_width),
                out_height=int(out_height),
            )
            for pt in obs_fov_meta["polygon_world_xz"]
        ]
        poly_trim = [
            map_base_point_to_trimmed(
                xy=(int(pt[0]), int(pt[1])),
                trim_transform=trim_transform,
            )
            for pt in poly_base
        ]
        poly_np = np.asarray(poly_trim, dtype=np.int32).reshape(-1, 1, 2)
        overlay = vis.copy()
        cv2.fillConvexPoly(overlay, poly_np, (255, 175, 70), lineType=cv2.LINE_AA)
        vis = cv2.addWeighted(overlay, 0.12, vis, 0.88, 0.0)
        cv2.polylines(vis, [poly_np], True, (255, 145, 40), 2, lineType=cv2.LINE_AA)
        forward_tip_base = world_xz_to_rectified_output(
            world_x=float(obs_fov_meta["forward_tip_world_xz"][0]),
            world_z=float(obs_fov_meta["forward_tip_world_xz"][1]),
            crop_world=crop_world,
            out_width=int(out_width),
            out_height=int(out_height),
        )
        forward_xy = map_base_point_to_trimmed(
            xy=(int(forward_tip_base[0]), int(forward_tip_base[1])),
            trim_transform=trim_transform,
        )
        cv2.arrowedLine(
            vis,
            stand_xy,
            forward_xy,
            (255, 145, 40),
            3,
            line_type=cv2.LINE_AA,
            tipLength=0.18,
        )
        obs_fov_polygon_xy = [[int(pt[0]), int(pt[1])] for pt in poly_trim]
    cv2.drawMarker(vis, stand_xy, (255, 60, 60), markerType=cv2.MARKER_CROSS, markerSize=28, thickness=2)
    cv2.circle(vis, stand_xy, 10, (255, 60, 60), 2, lineType=cv2.LINE_AA)
    cv2.putText(vis, "stand", (stand_xy[0] + 8, max(20, stand_xy[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2, lineType=cv2.LINE_AA)
    cv2.putText(vis, "stand", (stand_xy[0] + 8, max(20, stand_xy[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 60, 60), 1, lineType=cv2.LINE_AA)
    stand_backproject_xz = rectified_output_to_world_xz(
        output_xy=(int(stand_xy_base[0]), int(stand_xy_base[1])),
        crop_world=crop_world,
        out_width=int(out_width),
        out_height=int(out_height),
    )
    stand_backproject_world = np.asarray(
        [float(stand_backproject_xz[0]), float(nav_y), float(stand_backproject_xz[1])],
        dtype=np.float32,
    )
    stand_backproject_error_m = float(
        np.linalg.norm((np.asarray(stand_backproject_world, dtype=np.float32) - np.asarray(node_world, dtype=np.float32))[[0, 2]])
    )

    target_meta: list[dict] = []
    for idx, row in enumerate(target_rows):
        color = palette_rgb(idx)
        llm_name = str(row["llm_target"]["name"])
        target_bbox_xyxy = None
        target_center_xy = None
        marker_radius_px = 8
        topdown_sem_area_px = 0
        bbox_status = "topdown_semantic_missing"
        sem_row = target_top_sem_rows[idx] if idx < len(target_top_sem_rows) else None
        if sem_row is not None:
            rect_sem_base = rectify_full_topdown_mask(
                full_mask=np.asarray(sem_row["mask"], dtype=np.uint8),
                world_xz_to_fullpixel_h=world_xz_to_fullpixel_h,
                camera_center_world=camera_center_world,
                camera_above_m=float(camera_above_m),
                hfov_deg=float(hfov_deg),
                crop_world=crop_world,
                out_width=int(out_width),
                out_height=int(out_height),
            )
            rect_sem_base = (np.asarray(rect_sem_base, dtype=np.uint8) > 0) & np.asarray(rectified_valid_mask_base, dtype=bool)
            if int(np.count_nonzero(rect_sem_base)) >= int(topdown_sem_min_area):
                nlab, labels, stats, _ = cv2.connectedComponentsWithStats(rect_sem_base.astype(np.uint8), connectivity=8)
                if int(nlab) > 1:
                    target_world = np.asarray(row["target_world"], dtype=np.float32)
                    ref_xy_base = world_xz_to_rectified_output(
                        world_x=float(target_world[0]),
                        world_z=float(target_world[2]),
                        crop_world=crop_world,
                        out_width=int(out_width),
                        out_height=int(out_height),
                    )
                    best_lab = -1
                    best_area = -1
                    best_dist = float("inf")
                    for lab in range(1, int(nlab)):
                        area = int(stats[lab, cv2.CC_STAT_AREA])
                        if area < int(topdown_sem_min_area):
                            continue
                        x = int(stats[lab, cv2.CC_STAT_LEFT])
                        y = int(stats[lab, cv2.CC_STAT_TOP])
                        w_box = int(stats[lab, cv2.CC_STAT_WIDTH])
                        h_box = int(stats[lab, cv2.CC_STAT_HEIGHT])
                        cx = float(x) + 0.5 * float(w_box)
                        cy = float(y) + 0.5 * float(h_box)
                        dist = math.hypot(cx - float(ref_xy_base[0]), cy - float(ref_xy_base[1]))
                        if dist < best_dist - 1e-6 or (abs(dist - best_dist) <= 1e-6 and area > best_area):
                            best_dist = float(dist)
                            best_area = int(area)
                            best_lab = int(lab)
                    if best_lab >= 1:
                        rect_sem_base = labels == best_lab
                rect_sem = map_base_mask_to_trimmed(
                    base_mask=np.asarray(rect_sem_base, dtype=np.uint8),
                    trim_transform=trim_transform,
                )
                rect_sem = (np.asarray(rect_sem, dtype=np.uint8) > 0) & np.asarray(rectified_valid_mask, dtype=bool)
                ys, xs = np.where(rect_sem)
                if int(xs.size) >= int(topdown_sem_min_area):
                    x0 = int(xs.min())
                    y0 = int(ys.min())
                    x1 = int(xs.max()) + 1
                    y1 = int(ys.max()) + 1
                    target_bbox_xyxy = [int(x0), int(y0), int(x1), int(y1)]
                    topdown_sem_area_px = int(xs.size)
                    bbox_status = "ok_topdown_semantic"
                    vis = blend_mask_rgb(vis, rect_sem.astype(np.uint8), color, 0.16)
                    marker_radius_px = int(
                        np.clip(round(0.28 * max(target_bbox_xyxy[2] - target_bbox_xyxy[0], target_bbox_xyxy[3] - target_bbox_xyxy[1])), 7, 18)
                    )

        if target_bbox_xyxy is None and bool(require_topdown_semantic):
            target_world = np.asarray(row["target_world"], dtype=np.float32)
            target_center_xy_base = world_xz_to_rectified_output(
                world_x=float(target_world[0]),
                world_z=float(target_world[2]),
                crop_world=crop_world,
                out_width=int(out_width),
                out_height=int(out_height),
            )
            target_center_xy = map_base_point_to_trimmed(
                xy=(int(target_center_xy_base[0]), int(target_center_xy_base[1])),
                trim_transform=trim_transform,
            )
            target_meta.append(
                {
                    "llm_name": llm_name,
                    "topdown_center_xy": [int(target_center_xy[0]), int(target_center_xy[1])],
                    "topdown_bbox_xyxy": None,
                    "topdown_semantic_area_px": int(topdown_sem_area_px),
                    "topdown_bbox_status": "dropped_no_topdown_semantic",
                }
            )
            continue

        if target_bbox_xyxy is None:
            target_bounds_xz = row.get("target_world_bounds_xz")
            if target_bounds_xz is None:
                target_object = row.get("target_object")
                if target_object is not None:
                    target_bounds_xz = [float(v) for v in target_object["footprint_xz"]]
            if target_bounds_xz is not None:
                target_bbox_base = world_xz_bbox_to_rectified_output(
                    bounds_xz=[float(v) for v in target_bounds_xz],
                    crop_world=crop_world,
                    out_width=int(out_width),
                    out_height=int(out_height),
                )
                target_bbox_xyxy = map_base_bbox_to_trimmed(
                    box_xyxy=[int(v) for v in target_bbox_base],
                    trim_transform=trim_transform,
                )
                bb_w = max(1, int(target_bbox_xyxy[2] - target_bbox_xyxy[0]))
                bb_h = max(1, int(target_bbox_xyxy[3] - target_bbox_xyxy[1]))
                bb_ratio = float(bb_w * bb_h) / float(max(1, int(out_width) * int(out_height)))
                if bb_ratio <= 0.35:
                    marker_radius_px = int(np.clip(round(0.28 * max(bb_w, bb_h)), 7, 18))
                    bbox_status = "fallback_world_bbox"
                else:
                    target_bbox_xyxy = None
                    bbox_status = "fallback_bbox_too_large"

        if target_bbox_xyxy is not None:
            target_center_xy = (
                int(round(0.5 * float(target_bbox_xyxy[0] + target_bbox_xyxy[2]))),
                int(round(0.5 * float(target_bbox_xyxy[1] + target_bbox_xyxy[3]))),
            )
        else:
            target_world = np.asarray(row["target_world"], dtype=np.float32)
            target_center_xy_base = world_xz_to_rectified_output(
                world_x=float(target_world[0]),
                world_z=float(target_world[2]),
                crop_world=crop_world,
                out_width=int(out_width),
                out_height=int(out_height),
            )
            target_center_xy = map_base_point_to_trimmed(
                xy=(int(target_center_xy_base[0]), int(target_center_xy_base[1])),
                trim_transform=trim_transform,
            )
            bbox_status = "fallback_world_center"

        use_bbox = bool(target_bbox_xyxy is not None)
        if use_bbox:
            tx1, ty1, tx2, ty2 = [int(v) for v in target_bbox_xyxy]
            overlay = vis.copy()
            cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), color, -1, lineType=cv2.LINE_AA)
            vis = cv2.addWeighted(overlay, 0.14, vis, 0.86, 0.0)
            cv2.rectangle(vis, (tx1, ty1), (tx2, ty2), color, 2, lineType=cv2.LINE_AA)
        cv2.line(vis, stand_xy, target_center_xy, color, 2, lineType=cv2.LINE_AA)
        if not use_bbox:
            overlay = vis.copy()
            cv2.circle(overlay, target_center_xy, int(marker_radius_px), color, -1, lineType=cv2.LINE_AA)
            vis = cv2.addWeighted(overlay, 0.18, vis, 0.82, 0.0)
            target_bbox_xyxy = None
        cv2.circle(vis, target_center_xy, int(marker_radius_px), color, -1, lineType=cv2.LINE_AA)
        cv2.circle(vis, target_center_xy, int(marker_radius_px + 2), color, 2, lineType=cv2.LINE_AA)
        cv2.putText(
            vis,
            llm_name,
            (target_center_xy[0] + 8, max(20, target_center_xy[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 0, 0),
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            llm_name,
            (target_center_xy[0] + 8, max(20, target_center_xy[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            1,
            lineType=cv2.LINE_AA,
        )
        target_meta.append(
            {
                "llm_name": llm_name,
                "topdown_center_xy": [int(target_center_xy[0]), int(target_center_xy[1])],
                "topdown_bbox_xyxy": [int(v) for v in target_bbox_xyxy] if target_bbox_xyxy is not None else None,
                "topdown_semantic_area_px": int(topdown_sem_area_px),
                "topdown_bbox_status": str(bbox_status),
            }
        )

    cv2.putText(
        vis,
        f"rgb topdown h={float(camera_above_m):.2f}m targets={len(target_rows)}",
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (0, 0, 0),
        2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        f"rgb topdown h={float(camera_above_m):.2f}m targets={len(target_rows)}",
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        1,
        lineType=cv2.LINE_AA,
    )
    return vis, {
        "camera_center_world_xyz": [float(x) for x in camera_center_world.tolist()],
        "camera_above_ground_m": float(camera_above_m),
        "camera_above_limit_m": float(camera_above_limit_m),
        "camera_above_desired_m": float(desired_camera_above_m),
        "view_margin_ratio": float(margin_ratio),
        "view_margin_px": int(view_margin_px),
        "camera_height_scale": float(np.clip(float(topdown_camera_height_scale), 0.65, 1.25)),
        "upward_hit_dist_m": float(upward_hit_dist_m) if upward_hit_dist_m is not None else None,
        "observation_fov": {
            "obs_hfov_deg": float(obs_fov_meta["obs_hfov_deg"]),
            "forward_reach_m": float(obs_fov_meta["forward_reach_m"]),
            "half_span_m": float(obs_fov_meta["half_span_m"]),
            "polygon_world_xz": [[float(v) for v in pt] for pt in obs_fov_meta["polygon_world_xz"]],
            "polygon_xy": [[int(v) for v in pt] for pt in obs_fov_polygon_xy] if obs_fov_polygon_xy is not None else None,
            "forward_tip_world_xz": [float(v) for v in obs_fov_meta["forward_tip_world_xz"]],
            "forward_tip_xy": [int(forward_xy[0]), int(forward_xy[1])] if forward_xy is not None else None,
        }
        if obs_fov_meta is not None
        else None,
        "crop_world_bounds_xz": crop_world,
        "crop_box_xyxy_fullres": None,
        "topdown_cut": cut_meta,
        "stand_xy": [int(stand_xy[0]), int(stand_xy[1])],
        "stand_xy_base": [int(stand_xy_base[0]), int(stand_xy_base[1])],
        "trim_letterbox": trim_transform,
        "forward_xy": [int(forward_xy[0]), int(forward_xy[1])] if forward_xy is not None else None,
        "stand_backproject_world_xyz": [float(x) for x in stand_backproject_world.tolist()] if stand_backproject_world is not None else None,
        "stand_backproject_error_m": float(stand_backproject_error_m) if stand_backproject_error_m is not None else None,
        "meters_per_pixel_nav": float(mpp),
        "world_xz_to_fullpixel_homography": world_xz_to_fullpixel_h.tolist() if world_xz_to_fullpixel_h is not None else None,
        "rectified_valid_fraction_before_trim": float(np.count_nonzero(rectified_valid_mask_base)) / float(max(1, rectified_valid_mask_base.size)),
        "rectified_valid_fraction": float(np.count_nonzero(rectified_valid_mask)) / float(max(1, rectified_valid_mask.size)),
        "targets": target_meta,
    }


def main() -> None:
    args = build_parser().parse_args()
    scene_dir = Path(args.scene_dir).expanduser().resolve()
    gemini_json = Path(args.gemini_json).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (scene_dir / "gemini_local_topdown_matches_rgb_v1")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not gemini_json.exists():
        raise FileNotFoundError(f"gemini json not found: {gemini_json}")

    navmesh_module = load_navmesh_module(Path(__file__).with_name("navmesh_coverage_trajectory.py"))
    view_index, _ = build_view_index(scene_dir)
    annotations = json.loads(gemini_json.read_text(encoding="utf-8"))
    target_name_filter = str(args.target_name).strip().lower()
    ignore_labels = navmesh_module.parse_label_allow_deny_list(str(args.semantic_ignore_labels))

    sim_cache: dict[tuple[str, str, int, int, float, float, int, int, float], habitat_sim.Simulator] = {}
    nav_cache_by_scene: dict[tuple[str, str, float, float], dict] = {}
    summary_rows: list[dict] = []

    try:
        obs_count = 0
        obs_output_index = 0
        for ann in annotations:
            image_path = normalize_path(ann["image_path"])
            view_rec = view_index.get(image_path)
            if view_rec is None:
                continue
            obs_count += 1
            if obs_count > int(args.max_observations):
                break

            rgb_file = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if rgb_file is None:
                continue
            image_height, image_width = rgb_file.shape[:2]

            traj = view_rec.trajectory_data
            cam = traj["camera_observation"]
            scene_id = str(traj["scene_id"])
            dataset_config = str(traj["dataset_config"])
            node_world = np.asarray(view_rec.node_row["world_xyz"], dtype=np.float32)
            view_yaw_deg = float(view_rec.view_row["yaw_deg"])
            sensor_height_m = float(cam["sensor_height_m"])
            hfov_deg = float(cam["hfov_deg"])
            nav_y = float(traj.get("nav_y", node_world[1]))
            floor_levels = [float(v) for v in traj.get("detected_floor_levels_y", [])]

            sim_key = (
                dataset_config,
                scene_id,
                int(cam["node_image_width"]),
                int(cam["node_image_height"]),
                sensor_height_m,
                hfov_deg,
                int(args.topdown_width),
                int(args.topdown_height),
                float(args.topdown_hfov_deg),
            )
            if sim_key not in sim_cache:
                sim_cache[sim_key] = build_dual_agent_simulator(
                    scene_dataset_config=dataset_config,
                    scene_id=scene_id,
                    obs_width=int(cam["node_image_width"]),
                    obs_height=int(cam["node_image_height"]),
                    sensor_height_m=sensor_height_m,
                    obs_hfov_deg=hfov_deg,
                    topdown_width=int(args.topdown_width),
                    topdown_height=int(args.topdown_height),
                    topdown_hfov_deg=float(args.topdown_hfov_deg),
                )
            sim = sim_cache[sim_key]
            nav_key = (
                dataset_config,
                scene_id,
                round(float(nav_y), 4),
                float(args.local_nav_meters_per_pixel),
            )
            if nav_key not in nav_cache_by_scene:
                nav_cache_by_scene[nav_key] = build_floor_nav_cache(
                    sim.pathfinder,
                    nav_y=float(nav_y),
                    meters_per_pixel=float(args.local_nav_meters_per_pixel),
                )
            obs = render_first_person_observation(
                sim,
                position_world=node_world,
                yaw_rad=math.radians(view_yaw_deg),
                pitch_down_deg=float(cam["pitch_down_deg"]),
            )
            obs_render_camera = obs.get("render_camera")
            forward_world = None
            if obs_render_camera is not None:
                forward_world = unproject_pixel_to_plane_exact(
                    obs_render_camera,
                    float(image_width - 1) * 0.5,
                    float(image_height - 1) * 0.5,
                    float(nav_y),
                )
            sid_to_label = navmesh_module.build_semantic_id_to_label(sim)
            sem_items = parse_semantic_components(
                semantic_obs=obs["semantic"],
                semantic_id_to_label=sid_to_label,
                ignore_labels=ignore_labels,
                min_area_px=int(args.semantic_min_area_px),
                min_side_px=int(args.semantic_min_side_px),
                open_kernel=int(args.semantic_open_kernel),
                close_kernel=int(args.semantic_close_kernel),
                min_fill_ratio=float(args.semantic_min_fill_ratio),
            )
            ann_targets = []
            for target in ann.get("targets", []):
                llm_name = str(target.get("name", "")).strip()
                if not llm_name:
                    continue
                if target_name_filter and target_name_filter not in llm_name.lower():
                    continue
                ann_targets.append(dict(target))
                if len(ann_targets) >= int(args.max_targets_per_observation):
                    break
            if not ann_targets:
                continue

            matched_rows = assign_targets_to_semantic_items(
                targets=ann_targets,
                sem_items=sem_items,
                image_width=int(image_width),
                image_height=int(image_height),
                llm_bbox_scale=float(args.llm_bbox_scale),
                match_min_score=float(args.match_min_score),
                min_match_iou=float(args.min_match_iou),
            )
            if not matched_rows:
                continue

            kept_rows: list[dict] = []
            for match in matched_rows:
                best_item = dict(match["matched_item"])
                refined_mask = refine_match_mask(
                    matched_item=best_item,
                    llm_bbox_xyxy=[int(v) for v in match["llm_target"]["bbox_xyxy"]],
                    min_area_px=max(24, int(args.semantic_min_area_px)),
                )
                depth_target_world = estimate_target_world_from_depth(
                    matched_item={**best_item, "mask": refined_mask},
                    depth_obs=obs.get("depth"),
                    node_world=node_world,
                    sensor_height_m=sensor_height_m,
                    yaw_deg=view_yaw_deg,
                    pitch_down_deg=float(cam["pitch_down_deg"]),
                    intrinsics=view_rec.view_row.get("intrinsics"),
                )
                target_world_points = backproject_depth_mask_to_world_points(
                    depth_obs=obs.get("depth"),
                    mask_u8=refined_mask,
                    node_world=node_world,
                    sensor_height_m=sensor_height_m,
                    yaw_deg=view_yaw_deg,
                    pitch_down_deg=float(cam["pitch_down_deg"]),
                    intrinsics=view_rec.view_row.get("intrinsics"),
                    max_points=2500,
                )
                target_summary = summarize_target_world_points(target_world_points)
                if depth_target_world is None and target_summary is None:
                    continue

                ref_world = None
                if depth_target_world is not None:
                    ref_world = np.asarray(depth_target_world, dtype=np.float32)
                elif target_summary is not None:
                    ref_world = np.asarray(target_summary["center_world"], dtype=np.float32)
                else:
                    ref_world = np.asarray(node_world, dtype=np.float32)

                object_rows = extract_object_records(sim, [int(v) for v in best_item.get("semantic_ids", [])])
                target_object_row = choose_target_object_record(
                    np.asarray(ref_world, dtype=np.float32),
                    object_rows,
                    nav_y=float(nav_y),
                )
                # Strict mode: we only keep targets that can be grounded to HM3D semantic object records.
                if target_object_row is None:
                    continue

                target_world = np.asarray(target_object_row["center_world"], dtype=np.float32)
                target_object = {
                    "semantic_id": int(target_object_row["semantic_id"]),
                    "footprint_xz": [float(v) for v in target_object_row["footprint_xz"]],
                    "height_range_y": [float(v) for v in target_object_row.get("height_range_y", [target_world[1], target_world[1]])],
                    "source": "scene_semantic_object",
                }
                fx0, fz0, fx1, fz1 = [float(v) for v in target_object["footprint_xz"]]
                f_w = abs(float(fx1) - float(fx0))
                f_d = abs(float(fz1) - float(fz0))
                if max(float(f_w), float(f_d)) > 4.2 or (float(f_w) * float(f_d)) > 10.0:
                    continue
                if not target_world_is_on_current_floor(target_world, nav_y=float(nav_y)):
                    continue
                target_distance_m = planar_distance_to_target(node_world, target_world, target_object)
                if float(target_distance_m) > float(args.max_target_distance_m):
                    continue
                match["target_world"] = [float(x) for x in np.asarray(target_world, dtype=np.float32).tolist()]
                match["target_world_from_depth"] = [float(x) for x in np.asarray(depth_target_world, dtype=np.float32).tolist()] if depth_target_world is not None else None
                match["target_distance_m"] = float(target_distance_m)
                match["target_object"] = target_object
                match["target_world_bounds_xz"] = [float(v) for v in target_object["footprint_xz"]]
                match["target_world_points_xyz"] = target_summary["points_world_sampled"].tolist() if target_summary is not None else None
                match["target_world_point_count"] = int(target_summary["point_count"]) if target_summary is not None else 0
                kept_rows.append(match)
            if not kept_rows:
                continue

            topdown_overlay, topdown_meta = render_local_rgb_topdown(
                sim=sim,
                nav_cache=nav_cache_by_scene[nav_key],
                node_world=node_world,
                nav_y=float(nav_y),
                floor_levels=floor_levels,
                forward_world=None if forward_world is None else np.asarray(forward_world, dtype=np.float32),
                target_rows=kept_rows,
                out_width=int(args.topdown_width),
                out_height=int(args.topdown_height),
                hfov_deg=float(args.topdown_hfov_deg),
                margin_m=float(args.topdown_margin_m),
                min_side_m=float(args.local_nav_min_side_m),
                min_above_m=float(args.topdown_min_above_m),
                max_above_m=float(args.topdown_max_above_m),
                crop_fraction=float(args.topdown_crop_fraction),
                cut_above_ground_m=float(args.topdown_cut_above_ground_m),
                cut_depth_margin_m=float(args.topdown_cut_depth_margin_m),
                cut_inpaint_radius_px=int(args.topdown_cut_inpaint_radius_px),
                dot_stride_px=int(args.local_nav_dot_stride_px),
                min_clearance_m=float(args.local_nav_min_clearance_m),
                semantic_open_kernel=int(args.semantic_open_kernel),
                semantic_close_kernel=int(args.semantic_close_kernel),
                semantic_min_area_px=int(args.semantic_min_area_px),
                topdown_ceiling_margin_m=float(args.topdown_ceiling_margin_m),
                topdown_ray_start_offset_m=float(args.topdown_ray_start_offset_m),
                view_yaw_deg=float(view_yaw_deg),
                obs_hfov_deg=float(hfov_deg),
                topdown_stand_y_fraction=float(args.topdown_stand_y_fraction),
                topdown_view_margin_ratio=float(args.topdown_view_margin_ratio),
                topdown_camera_height_scale=float(args.topdown_camera_height_scale),
                topdown_trim_invalid=bool(int(args.topdown_trim_invalid) == 1),
                topdown_trim_margin_px=int(args.topdown_trim_margin_px),
                topdown_trim_min_valid_px=int(args.topdown_trim_min_valid_px),
                topdown_pad_gray_value=int(args.topdown_pad_gray_value),
                require_topdown_semantic=bool(int(args.require_topdown_semantic) == 1),
            )
            topdown_targets_all = list(topdown_meta.get("targets", []))
            active_indices = list(range(min(len(kept_rows), len(topdown_targets_all))))
            if bool(int(args.require_topdown_semantic) == 1):
                active_indices = [
                    i
                    for i in active_indices
                    if str(topdown_targets_all[i].get("topdown_bbox_status", "")) == "ok_topdown_semantic"
                ]
            if not active_indices:
                continue
            kept_rows = [kept_rows[i] for i in active_indices]
            topdown_targets_filtered = [topdown_targets_all[i] for i in active_indices]
            topdown_meta = {**topdown_meta, "targets": topdown_targets_filtered}

            obs_dir = out_dir / f"obs_{obs_output_index:04d}_{sanitize_text(Path(image_path).stem)}"
            obs_dir.mkdir(parents=True, exist_ok=True)
            topdown_overlay_path = obs_dir / "local_topdown_rgb.png"
            cv2.imwrite(str(topdown_overlay_path), cv2.cvtColor(topdown_overlay, cv2.COLOR_RGB2BGR))

            obs_overlay_path = None
            panel_path = None
            if int(args.topdown_only) != 1:
                obs_overlay = draw_observation_overlay(obs["rgb"], kept_rows)
                obs_overlay_path = obs_dir / "observation_match.png"
                panel_path = obs_dir / "panel_left_topdown_right_obs.png"
                cv2.imwrite(str(obs_overlay_path), cv2.cvtColor(obs_overlay, cv2.COLOR_RGB2BGR))
                panel = build_left_right_panel(topdown_overlay, obs_overlay)
                cv2.imwrite(str(panel_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

            payload_targets: list[dict] = []
            for idx, row in enumerate(kept_rows):
                llm_name = str(row["llm_target"]["name"])
                topdown_target = topdown_targets_filtered[idx] if idx < len(topdown_targets_filtered) else {}
                target_object = row.get("target_object")
                matched_item = dict(row["matched_item"])
                payload_targets.append(
                    {
                        "llm_target": row["llm_target"],
                        "matched_semantic": {
                            "label": str(matched_item["label"]),
                            "bbox_xyxy": [int(v) for v in matched_item["bbox_xyxy"]],
                            "semantic_ids": [int(v) for v in matched_item["semantic_ids"]],
                            "area_px": int(matched_item["area_px"]),
                            "score": float(row["score"]),
                            "score_breakdown": row["score_breakdown"],
                        },
                        "target_world_xyz_estimate": [float(x) for x in row["target_world"]],
                        "target_world_xyz_from_depth": row["target_world_from_depth"],
                        "target_world_bounds_xz": row.get("target_world_bounds_xz"),
                        "target_world_point_count": int(row.get("target_world_point_count", 0)),
                        "target_distance_m": float(row["target_distance_m"]),
                        "target_object_semantic_id": int(target_object["semantic_id"]) if target_object is not None else None,
                        "target_object_footprint_xz": [float(v) for v in target_object["footprint_xz"]] if target_object is not None else None,
                        "topdown_center_xy": topdown_target.get("topdown_center_xy"),
                        "topdown_bbox_xyxy": topdown_target.get("topdown_bbox_xyxy"),
                        "topdown_semantic_area_px": topdown_target.get("topdown_semantic_area_px"),
                        "topdown_bbox_status": topdown_target.get("topdown_bbox_status"),
                    }
                )

            crop_meta = dict(topdown_meta["crop_world_bounds_xz"])
            trim_meta = dict(topdown_meta.get("trim_letterbox", {}))
            if str(crop_meta.get("mapping", "")) == "oriented_forward":
                pixel_to_world_meta = {
                    "type": "rectified_forward_aligned_xz",
                    "mapping": "oriented_forward",
                    "stand_x": float(crop_meta["stand_x"]),
                    "stand_z": float(crop_meta["stand_z"]),
                    "right_x": float(crop_meta["right_x"]),
                    "right_z": float(crop_meta["right_z"]),
                    "forward_x": float(crop_meta["forward_x"]),
                    "forward_z": float(crop_meta["forward_z"]),
                    "u_min": float(crop_meta["u_min"]),
                    "u_max": float(crop_meta["u_max"]),
                    "f_front": float(crop_meta["f_front"]),
                    "f_back": float(crop_meta["f_back"]),
                    "stand_y_fraction": float(crop_meta.get("stand_y_fraction", 0.82)),
                    "out_width": int(args.topdown_width),
                    "out_height": int(args.topdown_height),
                    "pixel_center_offset": 0.5,
                    "axis": "x_right_y_down_forward_up",
                }
            else:
                pixel_to_world_meta = {
                    "type": "rectified_linear_xz",
                    "x_min": float(crop_meta["x_min"]),
                    "x_max": float(crop_meta["x_max"]),
                    "z_min": float(crop_meta["z_min"]),
                    "z_max": float(crop_meta["z_max"]),
                    "out_width": int(args.topdown_width),
                    "out_height": int(args.topdown_height),
                    "pixel_center_offset": 0.5,
                    "axis": "x_right_z_down",
                }
            if bool(trim_meta.get("enabled", False)):
                pixel_to_world_meta["output_to_base_rectified"] = {
                    "enabled": True,
                    "x0": int(trim_meta["x0"]),
                    "y0": int(trim_meta["y0"]),
                    "x1": int(trim_meta["x1"]),
                    "y1": int(trim_meta["y1"]),
                    "crop_width": int(trim_meta["crop_width"]),
                    "crop_height": int(trim_meta["crop_height"]),
                    "render_width": int(trim_meta["render_width"]),
                    "render_height": int(trim_meta["render_height"]),
                    "offset_x": int(trim_meta["offset_x"]),
                    "offset_y": int(trim_meta["offset_y"]),
                    "src_width": int(trim_meta["src_width"]),
                    "src_height": int(trim_meta["src_height"]),
                    "out_width": int(trim_meta["out_width"]),
                    "out_height": int(trim_meta["out_height"]),
                    "pixel_center_offset": 0.5,
                }
            else:
                pixel_to_world_meta["output_to_base_rectified"] = {
                    "enabled": False,
                    "src_width": int(trim_meta.get("src_width", args.topdown_width)),
                    "src_height": int(trim_meta.get("src_height", args.topdown_height)),
                    "out_width": int(trim_meta.get("out_width", args.topdown_width)),
                    "out_height": int(trim_meta.get("out_height", args.topdown_height)),
                    "pixel_center_offset": 0.5,
                }

            payload = {
                "source_image_path": image_path,
                "trajectory_json": str(view_rec.trajectory_json_path),
                "scene_id": scene_id,
                "node_index": int(view_rec.node_row["node_index"]),
                "view_index": int(view_rec.view_row["view_index"]),
                "view_yaw_deg": float(view_yaw_deg),
                "view_pitch_down_deg": float(cam["pitch_down_deg"]),
                "nav_y": float(nav_y),
                "node_world_xyz": [float(x) for x in node_world.tolist()],
                "camera_world_xyz": [float(x) for x in np.asarray(view_rec.view_row["camera_world_xyz"], dtype=np.float32).tolist()],
                "first_person_view": {
                    "rgb_png": image_path,
                    "depth_npy": str(view_rec.view_row.get("depth_npy")),
                    "yaw_deg": float(view_yaw_deg),
                    "pitch_down_deg": float(cam["pitch_down_deg"]),
                    "sensor_height_m": float(sensor_height_m),
                    "image_wh": [int(image_width), int(image_height)],
                    "intrinsics": dict(view_rec.view_row.get("intrinsics", {})),
                },
                "topdown": {
                    "mode": "rgb_local_crop",
                    "camera_center_world_xyz": topdown_meta["camera_center_world_xyz"],
                    "camera_above_ground_m": float(topdown_meta["camera_above_ground_m"]),
                    "hfov_deg": float(args.topdown_hfov_deg),
                    "crop_fraction": float(args.topdown_crop_fraction),
                    "cut": topdown_meta.get("topdown_cut"),
                    "crop_world_bounds_xz": crop_meta,
                    "pixel_to_world_xz": pixel_to_world_meta,
                    "stand_xy": [int(v) for v in topdown_meta["stand_xy"]],
                    "stand_xy_base": topdown_meta.get("stand_xy_base"),
                    "stand_backproject_world_xyz": topdown_meta.get("stand_backproject_world_xyz"),
                    "stand_backproject_error_m": topdown_meta.get("stand_backproject_error_m"),
                    "meters_per_pixel_nav": float(topdown_meta["meters_per_pixel_nav"]),
                    "view_margin_ratio": topdown_meta.get("view_margin_ratio"),
                    "view_margin_px": topdown_meta.get("view_margin_px"),
                    "camera_height_scale": topdown_meta.get("camera_height_scale"),
                    "observation_fov": topdown_meta.get("observation_fov"),
                    "forward_xy": topdown_meta.get("forward_xy"),
                    "world_xz_to_fullpixel_homography": topdown_meta.get("world_xz_to_fullpixel_homography"),
                    "trim_letterbox": trim_meta,
                    "rectified_valid_fraction_before_trim": topdown_meta.get("rectified_valid_fraction_before_trim"),
                    "rectified_valid_fraction": topdown_meta.get("rectified_valid_fraction"),
                },
                "targets": payload_targets,
                "artifacts": {
                    "local_topdown_rgb_png": str(topdown_overlay_path),
                },
            }
            if obs_overlay_path is not None:
                payload["artifacts"]["observation_match_png"] = str(obs_overlay_path)
            if panel_path is not None:
                payload["artifacts"]["panel_left_topdown_right_obs_png"] = str(panel_path)
            meta_path = obs_dir / "meta.json"
            meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            summary_rows.append(payload)
            obs_output_index += 1

        summary_path = out_dir / "summary.json"
        summary_path.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"out_dir": str(out_dir), "summary_json": str(summary_path), "count": len(summary_rows)}, ensure_ascii=False, indent=2))
    finally:
        for sim in sim_cache.values():
            sim.close()


if __name__ == "__main__":
    main()
