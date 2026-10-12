#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import magnum as mn
import numpy as np


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render first-person views from precomputed observer candidates.")
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--observer-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--scripts-dir", type=Path, default=Path(__file__).resolve().parent.parent / "habitat_scripts")
    parser.add_argument("--obs-width", type=int, default=1280)
    parser.add_argument("--obs-height", type=int, default=1080)
    parser.add_argument("--obs-hfov-deg", type=float, default=118.850803)
    parser.add_argument("--vfov-min-deg", type=float, default=90.0)
    parser.add_argument("--vfov-max-deg", type=float, default=110.0)
    parser.add_argument("--vfov-step-deg", type=float, default=5.0)
    parser.add_argument("--sensor-height-m", type=float, default=1.5)
    parser.add_argument("--pitch-min-deg", type=float, default=0.0)
    parser.add_argument("--pitch-max-deg", type=float, default=45.0)
    parser.add_argument("--pitch-step-deg", type=float, default=2.0)
    parser.add_argument("--yaw-search-deg", type=float, default=20.0)
    parser.add_argument("--yaw-step-deg", type=float, default=2.0)
    parser.add_argument("--target-op-v-ratio", type=float, default=0.72)
    parser.add_argument("--target-center-v-ratio", type=float, default=0.34)
    parser.add_argument("--min-v-ratio", type=float, default=0.15)
    parser.add_argument("--bottom-safe-ratio", type=float, default=0.90)
    parser.add_argument("--op-min-u-ratio", type=float, default=0.18)
    parser.add_argument("--op-max-u-ratio", type=float, default=0.82)
    parser.add_argument("--op-min-v-ratio", type=float, default=0.50)
    parser.add_argument("--op-max-v-ratio", type=float, default=0.88)
    parser.add_argument("--depth-tol-m", type=float, default=0.12)
    parser.add_argument("--observer-min-distance-m", type=float, default=0.5)
    parser.add_argument("--observer-max-distance-m", type=float, default=4.0)
    parser.add_argument("--observer-fallback-distances-m", default="1.0,1.25,1.5,2.0,2.5,3.0,3.5,4.0")
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--diversity-top-k", type=int, default=3)
    parser.add_argument("--selection-seed", type=int, default=7)
    parser.add_argument("--verbose", type=int, default=1)
    return parser


def image_xy_to_world_xz(px: float, py: float, p2w: dict) -> tuple[float, float]:
    u = float(p2w["u_min"]) + (float(px) + 0.5) * (float(p2w["u_max"]) - float(p2w["u_min"])) / float(p2w["out_width"])
    f = float(p2w["f_front"]) - (float(py) + 0.5) * (float(p2w["f_front"]) - float(p2w["f_back"])) / float(p2w["out_height"])
    wx = float(p2w["stand_x"]) + float(p2w["right_x"]) * u + float(p2w["forward_x"]) * f
    wz = float(p2w["stand_z"]) + float(p2w["right_z"]) * u + float(p2w["forward_z"]) * f
    return float(wx), float(wz)


def world_xz_to_image_xy(wx: float, wz: float, p2w: dict) -> tuple[int, int]:
    dx = float(wx) - float(p2w["stand_x"])
    dz = float(wz) - float(p2w["stand_z"])
    u = dx * float(p2w["right_x"]) + dz * float(p2w["right_z"])
    f = dx * float(p2w["forward_x"]) + dz * float(p2w["forward_z"])
    px = (u - float(p2w["u_min"])) * float(p2w["out_width"]) / max(1e-6, float(p2w["u_max"]) - float(p2w["u_min"])) - 0.5
    py = (float(p2w["f_front"]) - f) * float(p2w["out_height"]) / max(1e-6, float(p2w["f_front"]) - float(p2w["f_back"])) - 0.5
    return int(round(px)), int(round(py))


def parse_float_list(spec: str) -> list[float]:
    return [float(v.strip()) for v in str(spec).split(',') if str(v).strip()]


def build_distance_schedule(initial_distance_m: float, min_distance_m: float, max_distance_m: float, fallback_spec: str) -> list[float]:
    out = [float(initial_distance_m)]
    extras = sorted(parse_float_list(fallback_spec))
    for dist in extras:
        if dist < float(min_distance_m) or dist > float(max_distance_m):
            continue
        if dist <= float(initial_distance_m) + 0.05:
            continue
        out.append(float(dist))
    return out


def yaw_between(src_world_xyz: np.ndarray, dst_world_xyz: np.ndarray) -> float:
    dx = float(dst_world_xyz[0] - src_world_xyz[0])
    dz = float(dst_world_xyz[2] - src_world_xyz[2])
    return float((-math.degrees(math.atan2(dx, -dz))) % 360.0)


def vfov_to_hfov_deg(vfov_deg: float, width: int, height: int) -> float:
    aspect = float(width) / max(1.0, float(height))
    return float(math.degrees(2.0 * math.atan(math.tan(math.radians(float(vfov_deg) * 0.5)) * aspect)))


def project_world_to_pixel_exact(render_camera: Any, world_xyz: np.ndarray, width: int, height: int) -> dict:
    v = mn.Vector4(float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2]), 1.0)
    clip = render_camera.projection_matrix * (render_camera.camera_matrix * v)
    w = float(clip[3])
    if abs(w) < 1e-8:
        return {"valid": False, "u": 0.0, "v": 0.0, "visible_in_image": False}
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
        "u_draw": int(np.clip(round(u), 0, max(0, width - 1))),
        "v_draw": int(np.clip(round(vv), 0, max(0, height - 1))),
        "visible_in_image": bool(visible),
    }


def expected_depth_from_camera(render_camera: Any, world_xyz: np.ndarray) -> float:
    view_p = render_camera.camera_matrix * mn.Vector4(float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2]), 1.0)
    return float(-view_p[2])


def precise_snap(pathfinder: Any, wx: float, wz: float, ref_y: float) -> np.ndarray | None:
    radii = [0.0, 0.02, 0.04, 0.06, 0.10]
    angles = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
    best = None
    best_d2 = float("inf")
    for r in radii:
        for a in angles:
            raw = np.array([wx + r * math.cos(float(a)), ref_y, wz + r * math.sin(float(a))], dtype=np.float32)
            sp = np.asarray(pathfinder.snap_point(raw), dtype=np.float32)
            if sp.shape != (3,) or not np.all(np.isfinite(sp)):
                continue
            if not bool(pathfinder.is_navigable(sp)):
                continue
            if abs(float(sp[1]) - float(ref_y)) > 0.4:
                continue
            d2 = (float(sp[0]) - float(wx)) ** 2 + (float(sp[2]) - float(wz)) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = sp.copy()
    return best


def draw_topdown_panel(base_rgb: np.ndarray, observer_xy: tuple[int, int], op_xy: tuple[int, int]) -> np.ndarray:
    vis = base_rgb.copy()
    cv2.drawMarker(vis, observer_xy, (255, 80, 80), cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
    cv2.circle(vis, observer_xy, 9, (255, 80, 80), 2, cv2.LINE_AA)
    cv2.drawMarker(vis, op_xy, (0, 255, 0), cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
    cv2.circle(vis, op_xy, 9, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.arrowedLine(vis, observer_xy, op_xy, (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.05)
    return vis


def preferred_op_slot(candidate_index: int) -> tuple[float, float]:
    return (0.50, 0.82)


def resolve_target_focus_world_xyz(sample_target: dict, fallback_xz: np.ndarray, fallback_y: float) -> np.ndarray:
    for key in ["target_surface_center_world", "target_world_xyz_from_depth", "target_world_xyz_estimate"]:
        value = sample_target.get(key)
        if isinstance(value, list) and len(value) == 3:
            arr = np.asarray(value, dtype=np.float32)
            if arr.shape == (3,) and np.all(np.isfinite(arr)):
                return arr
    return np.asarray([float(fallback_xz[0]), float(fallback_y), float(fallback_xz[1])], dtype=np.float32)


def compute_pitch_down_deg(observer_world_xyz: np.ndarray, target_world_xyz: np.ndarray) -> float:
    dx = float(target_world_xyz[0] - observer_world_xyz[0])
    dz = float(target_world_xyz[2] - observer_world_xyz[2])
    dy = float(observer_world_xyz[1] - target_world_xyz[1])
    horiz = math.hypot(dx, dz)
    if horiz <= 1e-6:
        return 0.0
    return float(math.degrees(math.atan2(dy, horiz)))


def build_pitch_schedule(base_pitch_down_deg: float, pitch_max_deg: float, step_deg: float) -> list[float]:
    start = float(np.clip(base_pitch_down_deg, -15.0, float(pitch_max_deg)))
    step = max(1e-3, float(step_deg))
    values = [round(start, 6)]
    p = start + step
    while p <= float(pitch_max_deg) + 1e-6:
        values.append(round(p, 6))
        p += step
    return values


def rgb_neighbor_diff_score(rgb: np.ndarray) -> float:
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return 1e9
    dx = float(np.abs(arr[:, 1:] - arr[:, :-1]).mean())
    dy = float(np.abs(arr[1:] - arr[:-1]).mean())
    return 0.5 * (dx + dy)


def looks_like_corrupted_render(rgb: np.ndarray) -> bool:
    # Habitat render failures occasionally produce full-frame colored noise.
    return bool(rgb_neighbor_diff_score(rgb) > 20.0)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def choose_view(
    *,
    base_mod: Any,
    sim: Any,
    observer_world_xyz: np.ndarray,
    target_semantic_ids: list[int],
    operating_world_xyz: np.ndarray,
    base_yaw_deg: float,
    base_pitch_down_deg: float,
    vfov_deg: float,
    width: int,
    height: int,
    pitch_min_deg: float,
    pitch_max_deg: float,
    pitch_step_deg: float,
    yaw_search_deg: float,
    yaw_step_deg: float,
    preferred_op_u_ratio: float,
    preferred_op_v_ratio: float,
    target_center_v_ratio: float,
    min_v_ratio: float,
    bottom_safe_ratio: float,
    op_min_u_ratio: float,
    op_max_u_ratio: float,
    op_min_v_ratio: float,
    op_max_v_ratio: float,
    depth_tol_m: float,
    diversity_top_k: int,
    selection_seed: int,
) -> dict | None:
    # Final policy requested by the user:
    # - heading stays fixed to the semantic target center
    # - start from the geometric pitch that faces the target center
    # - increase pitch by +5 deg only when the operating point is too low / not visible
    # - only after this vfov fails should the caller expand vfov and retry
    op_bottom_guard = min(0.80, float(bottom_safe_ratio))
    pitch_values = build_pitch_schedule(float(base_pitch_down_deg), float(pitch_max_deg), max(5.0, float(pitch_step_deg)))

    for pitch_down_deg in pitch_values:
        obs = base_mod.render_first_person_observation(
            sim,
            position_world=observer_world_xyz,
            yaw_rad=math.radians(float(base_yaw_deg)),
            pitch_down_deg=float(pitch_down_deg),
        )
        render_camera = obs['render_camera']
        depth = np.asarray(obs['depth'], dtype=np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        semantic = np.asarray(obs['semantic'], dtype=np.int32)
        if semantic.ndim == 3 and semantic.shape[-1] == 1:
            semantic = semantic[..., 0]
        target_mask = np.isin(semantic, np.asarray(target_semantic_ids, dtype=np.int32))
        target_area_px = int(np.count_nonzero(target_mask))
        bbox = mask_bbox(target_mask)
        if bbox is None or target_area_px <= 0:
            continue
        x0, y0, x1, y1 = bbox
        target_complete = x0 > 0 and y0 > 0 and x1 < int(width - 1) and y1 < int(height - 1)
        if not target_complete:
            # Current vfov is too small for a complete target view.
            continue
        u_t = 0.5 * float(x0 + x1) / max(1.0, float(width - 1))
        v_t = 0.5 * float(y0 + y1) / max(1.0, float(height - 1))
        target_xy = [int(round(0.5 * float(x0 + x1))), int(round(0.5 * float(y0 + y1)))]

        proj_op = project_world_to_pixel_exact(render_camera, operating_world_xyz, int(width), int(height))
        op_visible = bool(proj_op['visible_in_image'])
        if op_visible:
            u_op = float(proj_op['u']) / max(1.0, float(width - 1))
            v_op = float(proj_op['v']) / max(1.0, float(height - 1))
        else:
            u_op = 0.5
            v_op = 1.0

        d_obs = None
        d_exp = None
        depth_err = 1e6
        if op_visible:
            d_obs = float(depth[int(proj_op['v_draw']), int(proj_op['u_draw'])])
            d_exp = expected_depth_from_camera(render_camera, operating_world_xyz)
            if math.isfinite(d_obs) and math.isfinite(d_exp) and d_obs > 0.05 and d_exp > 0.0:
                depth_err = abs(d_obs - d_exp)

        rgb = np.asarray(obs['rgb'], dtype=np.uint8)
        if rgb.ndim == 3 and rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        if looks_like_corrupted_render(rgb):
            continue

        row = {
            'projection_mode': 'direct',
            'yaw_deg': float(base_yaw_deg),
            'yaw_offset_deg': 0.0,
            'pitch_down_deg': float(pitch_down_deg),
            'operating_rgb_xy': [int(proj_op['u_draw']), int(proj_op['v_draw'])] if op_visible else [0, 0],
            'target_rgb_xy': target_xy,
            'target_bbox_xyxy': [int(x0), int(y0), int(x1), int(y1)],
            'operating_u_ratio': float(u_op),
            'operating_v_ratio': float(v_op),
            'target_u_ratio': float(u_t),
            'target_v_ratio': float(v_t),
            'target_area_px': int(target_area_px),
            'depth_obs_m': float(d_obs) if d_obs is not None and math.isfinite(d_obs) else None,
            'depth_expected_m': float(d_exp) if d_exp is not None and math.isfinite(d_exp) else None,
            'target_visible': True,
            'rgb': rgb,
            'depth_err': float(depth_err),
            'score': float(max(0.0, pitch_down_deg)) + (float(vfov_deg) - 90.0) * 0.2,
            'rgb_neighbor_diff_score': float(rgb_neighbor_diff_score(rgb)),
        }

        if op_visible and float(v_op) < float(op_bottom_guard):
            return {'best': row, 'rgb': rgb}

    return None


def main() -> None:
    args = build_parser().parse_args()
    verbose = bool(int(args.verbose))
    sample_dir = args.sample_dir.expanduser().resolve()
    observer_json = args.observer_json.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_meta = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
    observer_data = json.loads(observer_json.read_text(encoding="utf-8"))
    source_meta = json.loads(Path(sample_meta["source_meta_json"]).read_text(encoding="utf-8"))
    p2w = sample_meta["topdown"]["pixel_to_world_xz"]

    target_index = int(observer_data.get("target_index", 0))
    operating_xy = observer_data["operating_point_topdown_xy"]
    op_world_xz = np.asarray(image_xy_to_world_xz(operating_xy[0], operating_xy[1], p2w), dtype=np.float32)
    target_row = source_meta["targets"][target_index]
    target_center_xy = sample_meta["targets"][target_index]["center_xy"]
    target_center_world_xz = np.asarray(image_xy_to_world_xz(target_center_xy[0], target_center_xy[1], p2w), dtype=np.float32)
    target_focus_world_xyz = resolve_target_focus_world_xyz(
        target_row,
        target_center_world_xz,
        float(source_meta["nav_y"]),
    )
    target_semantic_ids = [int(v) for v in target_row.get("matched_semantic", {}).get("semantic_ids", []) if int(v) > 0]
    if not target_semantic_ids:
        if target_row.get("target_object_semantic_id") is not None:
            target_semantic_ids = [int(target_row["target_object_semantic_id"])]
    if not target_semantic_ids:
        raise RuntimeError("failed to resolve target semantic ids")

    floor_rgb = cv2.cvtColor(cv2.imread(str(sample_dir / "local_topdown_rgb.png")), cv2.COLOR_BGR2RGB)
    traj = json.loads(Path(source_meta["trajectory_json"]).read_text(encoding="utf-8"))
    scene_source = str(traj.get("scene_path", "")).strip() or str(traj["scene_id"])
    base_mod = load_module(args.scripts_dir / "render_local_topdown.py", "_base_obscand")
    vfov_values = []
    vf = float(args.vfov_min_deg)
    while vf <= float(args.vfov_max_deg) + 1e-6:
        vfov_values.append(round(vf, 6))
        vf += float(args.vfov_step_deg)
    if not vfov_values:
        vfov_values = [float(args.vfov_max_deg)]

    sim_cache: dict[float, Any] = {}

    def get_sim_for_hfov(hfov_deg: float) -> Any:
        key = round(float(hfov_deg), 6)
        if key not in sim_cache:
            sim_cache[key] = base_mod.build_dual_agent_simulator(
                scene_dataset_config=str(traj["dataset_config"]),
                scene_id=scene_source,
                obs_width=int(args.obs_width),
                obs_height=int(args.obs_height),
                sensor_height_m=float(args.sensor_height_m),
                obs_hfov_deg=float(hfov_deg),
                topdown_width=64,
                topdown_height=64,
                topdown_hfov_deg=100.0,
            )
        return sim_cache[key]

    try:
        probe_sim = get_sim_for_hfov(float(args.obs_hfov_deg))
        pf = probe_sim.pathfinder
        op_world_xyz = precise_snap(pf, float(op_world_xz[0]), float(op_world_xz[1]), float(source_meta["nav_y"]))
        if op_world_xyz is None:
            raise RuntimeError("failed to snap operating point")
        rows = []
        for idx, cand in enumerate(observer_data.get("observer_points", [])[: int(args.max_candidates)]):
            observer_world_xz = np.asarray(cand["observer_world_xz"], dtype=np.float32)
            observer_world_xyz = precise_snap(pf, float(observer_world_xz[0]), float(observer_world_xz[1]), float(source_meta["nav_y"]))
            if observer_world_xyz is None:
                rows.append({"candidate_index": idx, "status": "failed_snap", "observer_point": cand})
                continue
            preferred_u_ratio, preferred_v_ratio = preferred_op_slot(int(idx))
            base_vec_xz = np.asarray([
                float(observer_world_xyz[0] - op_world_xyz[0]),
                float(observer_world_xyz[2] - op_world_xyz[2]),
            ], dtype=np.float32)
            base_dist = float(np.linalg.norm(base_vec_xz))
            if base_dist < 1e-6:
                rows.append({"candidate_index": idx, "status": "failed_snap", "observer_point": cand})
                continue
            base_dir_xz = base_vec_xz / base_dist
            distance_schedule = build_distance_schedule(
                initial_distance_m=float(base_dist),
                min_distance_m=float(args.observer_min_distance_m),
                max_distance_m=float(args.observer_max_distance_m),
                fallback_spec=str(args.observer_fallback_distances_m),
            )
            if verbose:
                print(
                    f"[candidate {idx}] base_dist={base_dist:.2f} schedule={distance_schedule} pref=({preferred_u_ratio:.2f},{preferred_v_ratio:.2f})",
                    flush=True,
                )
            pitch = None
            chosen_vfov = None
            chosen_hfov = None
            chosen_observer_world_xyz = None
            chosen_observer_world_xz = None
            chosen_observer_xy = None
            chosen_observer_distance_m = None
            for trial_dist in distance_schedule:
                trial_x = float(op_world_xyz[0]) + float(base_dir_xz[0]) * float(trial_dist)
                trial_z = float(op_world_xyz[2]) + float(base_dir_xz[1]) * float(trial_dist)
                trial_world_xyz = precise_snap(pf, float(trial_x), float(trial_z), float(source_meta["nav_y"]))
                if trial_world_xyz is None:
                    continue
                trial_world_xz = np.asarray([float(trial_world_xyz[0]), float(trial_world_xyz[2])], dtype=np.float32)
                base_yaw_deg = yaw_between(trial_world_xyz, target_focus_world_xyz)
                base_pitch_down_deg = compute_pitch_down_deg(trial_world_xyz, target_focus_world_xyz)
                local_pitch = None
                local_vfov = None
                local_hfov = None
                for vfov_deg in vfov_values:
                    hfov_deg = vfov_to_hfov_deg(float(vfov_deg), int(args.obs_width), int(args.obs_height))
                    sim = get_sim_for_hfov(float(hfov_deg))
                    local_pitch = choose_view(
                        base_mod=base_mod,
                        sim=sim,
                        observer_world_xyz=trial_world_xyz,
                        target_semantic_ids=target_semantic_ids,
                        operating_world_xyz=op_world_xyz,
                        base_yaw_deg=float(base_yaw_deg),
                        base_pitch_down_deg=float(base_pitch_down_deg),
                        vfov_deg=float(vfov_deg),
                        width=int(args.obs_width),
                        height=int(args.obs_height),
                        pitch_min_deg=float(args.pitch_min_deg),
                        pitch_max_deg=float(args.pitch_max_deg),
                        pitch_step_deg=float(args.pitch_step_deg),
                        yaw_search_deg=float(args.yaw_search_deg),
                        yaw_step_deg=float(args.yaw_step_deg),
                        preferred_op_u_ratio=float(preferred_u_ratio),
                        preferred_op_v_ratio=float(preferred_v_ratio),
                        target_center_v_ratio=float(args.target_center_v_ratio),
                        min_v_ratio=float(args.min_v_ratio),
                        bottom_safe_ratio=float(args.bottom_safe_ratio),
                        op_min_u_ratio=float(args.op_min_u_ratio),
                        op_max_u_ratio=float(args.op_max_u_ratio),
                        op_min_v_ratio=float(args.op_min_v_ratio),
                        op_max_v_ratio=float(args.op_max_v_ratio),
                        depth_tol_m=float(args.depth_tol_m),
                        diversity_top_k=int(args.diversity_top_k),
                        selection_seed=int(args.selection_seed) + int(idx),
                    )
                    if local_pitch is not None:
                        local_vfov = float(vfov_deg)
                        local_hfov = float(hfov_deg)
                        break
                if local_pitch is None:
                    continue
                pitch = local_pitch
                chosen_vfov = local_vfov
                chosen_hfov = local_hfov
                chosen_observer_world_xyz = trial_world_xyz
                chosen_observer_world_xz = trial_world_xz
                chosen_observer_xy = world_xz_to_image_xy(float(trial_world_xz[0]), float(trial_world_xz[1]), p2w)
                chosen_observer_distance_m = float(trial_dist)
                break
            if pitch is None:
                rows.append({
                    "candidate_index": idx,
                    "status": "failed_pitch",
                    "distance_schedule_m": distance_schedule,
                    **cand,
                })
                continue

            topdown_panel = draw_topdown_panel(
                floor_rgb,
                observer_xy=tuple(int(v) for v in chosen_observer_xy),
                op_xy=tuple(int(v) for v in operating_xy),
            )
            right = pitch["rgb"].copy()
            op_uv = tuple(int(v) for v in pitch["best"]["operating_rgb_xy"])
            tx0, ty0, tx1, ty1 = [int(v) for v in pitch["best"]["target_bbox_xyxy"]]
            tgt_uv = tuple(int(v) for v in pitch["best"]["target_rgb_xy"])
            cv2.rectangle(right, (tx0, ty0), (tx1, ty1), (255, 220, 0), 2, cv2.LINE_AA)
            cv2.drawMarker(right, tgt_uv, (255, 220, 0), cv2.MARKER_TILTED_CROSS, 20, 2, cv2.LINE_AA)
            cv2.drawMarker(right, op_uv, (0, 255, 0), cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
            cv2.circle(right, op_uv, 8, (0, 255, 0), 2, cv2.LINE_AA)

            h = max(topdown_panel.shape[0], right.shape[0])
            left_w = int(round(topdown_panel.shape[1] * h / max(1, topdown_panel.shape[0])))
            right_w = int(round(right.shape[1] * h / max(1, right.shape[0])))
            left_rs = cv2.resize(topdown_panel, (left_w, h), interpolation=cv2.INTER_LINEAR)
            right_rs = cv2.resize(right, (right_w, h), interpolation=cv2.INTER_LINEAR)
            panel = np.zeros((h, left_w + 8 + right_w, 3), dtype=np.uint8)
            panel[:, :left_w] = left_rs
            panel[:, left_w + 8:left_w + 8 + right_w] = right_rs

            cand_dir = out_dir / f"candidate_{idx:02d}"
            cand_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(cand_dir / "topdown_observer_and_operating.png"), cv2.cvtColor(topdown_panel, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(cand_dir / "observation_rgb.png"), cv2.cvtColor(pitch["rgb"], cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(cand_dir / "observation_with_points.png"), cv2.cvtColor(right, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(cand_dir / "side_by_side.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

            row = {
                "candidate_index": int(idx),
                "status": "ok",
                "observer_point": cand,
                "observer_point_used": {
                    "observer_topdown_xy": [int(chosen_observer_xy[0]), int(chosen_observer_xy[1])],
                    "observer_world_xz": [float(v) for v in chosen_observer_world_xz.tolist()],
                    "distance_from_operating_m": float(chosen_observer_distance_m),
                },
                "observer_world_xyz": [float(v) for v in chosen_observer_world_xyz.tolist()],
                "operating_world_xyz": [float(v) for v in op_world_xyz.tolist()],
                "observation": {
                    "projection_mode": str(pitch["best"].get("projection_mode", "strict")),
                    "yaw_deg": float(pitch["best"]["yaw_deg"]),
                    "yaw_offset_deg": float(pitch["best"]["yaw_offset_deg"]),
                    "pitch_down_deg": float(pitch["best"]["pitch_down_deg"]),
                    "operating_rgb_xy": [int(v) for v in pitch["best"]["operating_rgb_xy"]],
                    "target_rgb_xy": [int(v) for v in pitch["best"]["target_rgb_xy"]],
                    "target_bbox_xyxy": [int(v) for v in pitch["best"]["target_bbox_xyxy"]],
                    "operating_u_ratio": float(pitch["best"]["operating_u_ratio"]),
                    "operating_v_ratio": float(pitch["best"]["operating_v_ratio"]),
                    "target_u_ratio": float(pitch["best"]["target_u_ratio"]),
                    "target_v_ratio": float(pitch["best"]["target_v_ratio"]),
                    "depth_obs_m": float(pitch["best"]["depth_obs_m"]) if pitch["best"]["depth_obs_m"] is not None else None,
                    "depth_expected_m": float(pitch["best"]["depth_expected_m"]) if pitch["best"]["depth_expected_m"] is not None else None,
                    "operating_visible": True,
                    "target_visible": bool(pitch["best"].get("target_visible", True)),
                    "image_wh": [int(args.obs_width), int(args.obs_height)],
                    "hfov_deg": float(chosen_hfov),
                    "vfov_deg": float(chosen_vfov),
                    "target_semantic_ids": [int(v) for v in target_semantic_ids],
                    "target_focus_world_xyz": [float(v) for v in target_focus_world_xyz.tolist()],
                    "rgb_neighbor_diff_score": float(pitch["best"].get("rgb_neighbor_diff_score", 0.0)),
                    "score": float(idx) * 100.0 + float(pitch["best"]["score"]),
                },
                "artifacts": {
                    "topdown_observer_and_operating": str(cand_dir / "topdown_observer_and_operating.png"),
                    "observation_rgb": str(cand_dir / "observation_rgb.png"),
                    "observation_with_points": str(cand_dir / "observation_with_points.png"),
                    "side_by_side": str(cand_dir / "side_by_side.png"),
                },
            }
            (cand_dir / "result.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
            rows.append(row)

        (out_dir / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"out_dir": str(out_dir), "count": len(rows)}, ensure_ascii=False, indent=2))
    finally:
        sim_cache.clear()


if __name__ == "__main__":
    main()
