#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


HERE = Path(__file__).resolve().parent
RENDER_FP_MOD = load_module(HERE / 'render_fp_from_observer_candidates.py', '_accepted_base_renderfp')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Generate accepted-view and operating-point-view base package for accepted targets.')
    p.add_argument('--stage3-root', type=Path, required=True, help='Root containing final_observations/ and plan/')
    p.add_argument('--accepted-file', type=Path, required=True, help='accepted_target_uids.txt from API filtering')
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--scripts-dir', type=Path, default=HERE.parent / 'habitat_scripts')
    p.add_argument('--obs-width', type=int, default=1280)
    p.add_argument('--obs-height', type=int, default=1080)
    p.add_argument('--sensor-height-m', type=float, default=1.5)
    p.add_argument('--mask-open-k', type=int, default=0)
    p.add_argument('--mask-close-k', type=int, default=0)
    p.add_argument('--mask-min-area-px', type=int, default=1)
    p.add_argument('--operating-heading-mode', choices=['semantic_target', 'edge_normal'], default='semantic_target')
    p.add_argument('--operating-pitch-deg', type=float, default=40.0)
    p.add_argument('--operating-pitch-max-deg', type=float, default=55.0)
    p.add_argument('--operating-pitch-step-deg', type=float, default=5.0)
    p.add_argument('--operating-vfov-default', type=float, default=100.0)
    p.add_argument('--operating-vfov-max', type=float, default=110.0)
    p.add_argument('--operating-min-occupancy-ratio', type=float, default=0.60)
    p.add_argument('--operating-backoff-max-m', type=float, default=1.5)
    p.add_argument('--operating-backoff-step-m', type=float, default=0.25)
    p.add_argument('--max-targets', type=int, default=0)
    p.add_argument('--accepted-only', type=int, default=0)
    p.add_argument('--resume', type=int, default=1)
    p.add_argument('--verbose', type=int, default=1)
    return p


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def load_accepted_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    out = []
    seen = set()
    for v in ids:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def clean_target_component(
    semantic_obs: np.ndarray,
    target_semantic_ids: list[int],
    preferred_center_xy: tuple[float, float] | None,
    open_k: int,
    close_k: int,
    min_area_px: int,
) -> dict | None:
    semantic = np.asarray(semantic_obs, dtype=np.int32)
    ids = [int(v) for v in target_semantic_ids if int(v) > 0]
    if not ids:
        return None
    mask = np.isin(semantic, ids).astype(np.uint8)
    if int(np.count_nonzero(mask)) <= 0:
        return None
    # Use the HM3D semantic instance mask directly. Do not select a single connected
    # component, otherwise thin / fragmented objects such as cabinets get truncated.
    comp = mask.copy().astype(np.uint8)
    if int(open_k) >= 2:
        opened = cv2.morphologyEx(comp, cv2.MORPH_OPEN, np.ones((int(open_k), int(open_k)), dtype=np.uint8))
        if int(np.count_nonzero(opened)) > 0:
            comp = opened
    if int(close_k) >= 2:
        closed = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, np.ones((int(close_k), int(close_k)), dtype=np.uint8))
        if int(np.count_nonzero(closed)) > 0:
            comp = closed

    ys, xs = np.nonzero(comp)
    if xs.size == 0 or ys.size == 0:
        return None
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    center = [int(round(xs.mean())), int(round(ys.mean()))]
    return {
        'mask': comp,
        'bbox_xyxy': bbox,
        'center_xy': center,
        'area_px': int(np.count_nonzero(comp)),
    }


def render_clean_view(
    *,
    base_mod: Any,
    sim: Any,
    camera_world_xyz: np.ndarray,
    yaw_deg: float,
    pitch_down_deg: float,
    target_semantic_ids: list[int],
    target_focus_world_xyz: np.ndarray,
    operating_world_xyz: np.ndarray | None,
    width: int,
    height: int,
    open_k: int,
    close_k: int,
    min_area_px: int,
) -> dict:
    obs = base_mod.render_first_person_observation(
        sim,
        position_world=camera_world_xyz,
        yaw_rad=math.radians(float(yaw_deg)),
        pitch_down_deg=float(pitch_down_deg),
    )
    rgb = np.asarray(obs['rgb'], dtype=np.uint8)
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    semantic = np.asarray(obs['semantic'], dtype=np.int32)
    if semantic.ndim == 3 and semantic.shape[-1] == 1:
        semantic = semantic[..., 0]
    render_camera = obs['render_camera']

    target_proj = RENDER_FP_MOD.project_world_to_pixel_exact(render_camera, target_focus_world_xyz, int(width), int(height))
    preferred_center = None
    if bool(target_proj.get('valid', False)):
        preferred_center = (float(target_proj['u']), float(target_proj['v']))
    clean = clean_target_component(
        semantic_obs=semantic,
        target_semantic_ids=target_semantic_ids,
        preferred_center_xy=preferred_center,
        open_k=int(open_k),
        close_k=int(close_k),
        min_area_px=int(min_area_px),
    )
    if clean is None:
        raise RuntimeError('failed to build clean target mask bbox')

    op_proj = None
    if operating_world_xyz is not None:
        op_proj = RENDER_FP_MOD.project_world_to_pixel_exact(render_camera, operating_world_xyz, int(width), int(height))
        if not bool(op_proj.get('visible_in_image', False)):
            op_proj = None

    return {
        'rgb': rgb,
        'depth': np.asarray(obs['depth'], dtype=np.float32),
        'target_mask_u8': clean['mask'].astype(np.uint8),
        'target_bbox_xyxy': clean['bbox_xyxy'],
        'target_center_xy': clean['center_xy'],
        'target_mask_area_px': clean['area_px'],
        'target_focus_rgb_xy': [int(target_proj['u_draw']), int(target_proj['v_draw'])] if bool(target_proj.get('valid', False)) else None,
        'target_focus_u_ratio': float(target_proj['u']) / max(1.0, float(width - 1)) if bool(target_proj.get('valid', False)) else None,
        'target_focus_v_ratio': float(target_proj['v']) / max(1.0, float(height - 1)) if bool(target_proj.get('valid', False)) else None,
        'target_focus_visible': bool(target_proj.get('visible_in_image', False)),
        'operating_rgb_xy': [int(op_proj['u_draw']), int(op_proj['v_draw'])] if op_proj is not None else None,
    }


def render_raw_rgb_view(*, base_mod: Any, sim: Any, camera_world_xyz: np.ndarray, yaw_deg: float, pitch_down_deg: float) -> np.ndarray:
    obs = base_mod.render_first_person_observation(
        sim,
        position_world=camera_world_xyz,
        yaw_rad=math.radians(float(yaw_deg)),
        pitch_down_deg=float(pitch_down_deg),
    )
    rgb = np.asarray(obs['rgb'], dtype=np.uint8)
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    return rgb


def draw_overlay(rgb: np.ndarray, *, bbox_xyxy: list[int], center_xy: list[int], op_xy: list[int] | None) -> np.ndarray:
    out = np.asarray(rgb, dtype=np.uint8).copy()
    x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]
    cx, cy = [int(v) for v in center_xy]
    cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), (0, 220, 255), 2, cv2.LINE_AA)
    cv2.drawMarker(out, (cx, cy), (0, 220, 255), cv2.MARKER_TILTED_CROSS, 20, 2, cv2.LINE_AA)
    if op_xy is not None:
        ox, oy = [int(v) for v in op_xy]
        cv2.drawMarker(out, (ox, oy), (0, 255, 0), cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
        cv2.circle(out, (ox, oy), 7, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def pixel_offset_to_angle_deg(offset_px: float, image_extent_px: int, fov_deg: float) -> float:
    half_extent = max(1.0, float(image_extent_px) * 0.5)
    half_fov_rad = math.radians(float(fov_deg) * 0.5)
    return math.degrees(math.atan((float(offset_px) / half_extent) * math.tan(half_fov_rad)))


def evaluate_operating_candidate(candidate: dict, width: int, height: int) -> dict | None:
    x0, y0, x1, y1 = [int(v) for v in candidate['target_bbox_xyxy']]
    box_w = max(1, x1 - x0)
    box_h = max(1, y1 - y0)
    width_ratio = float(box_w) / max(1.0, float(width))
    height_ratio = float(box_h) / max(1.0, float(height))
    area_ratio = float(candidate.get('target_mask_area_px', 0)) / max(1.0, float(width * height))
    occupancy_ratio = max(width_ratio, height_ratio)
    center_x, center_y = [int(v) for v in candidate['target_center_xy']]
    u = float(center_x) / max(1.0, float(width - 1))
    v = float(center_y) / max(1.0, float(height - 1))
    center_ok = 0.08 <= u <= 0.92 and 0.10 <= v <= 0.90
    central_x0 = int(round(0.12 * float(width)))
    central_y0 = int(round(0.12 * float(height)))
    central_x1 = int(round(0.88 * float(width)))
    central_y1 = int(round(0.88 * float(height)))
    overlap_w = max(0, min(x1, central_x1) - max(x0, central_x0))
    overlap_h = max(0, min(y1, central_y1) - max(y0, central_y0))
    central_overlap_ok = overlap_w > 0 and overlap_h > 0
    visible_enough = area_ratio >= 0.003 or width_ratio >= 0.10 or height_ratio >= 0.10
    too_large = width_ratio >= 0.98 or height_ratio >= 0.98 or area_ratio >= 0.85
    acceptable = visible_enough and (center_ok or central_overlap_ok) and not too_large
    score = abs(u - 0.5) + abs(v - 0.52) + 0.10 * max(width_ratio, height_ratio)
    return {
        'u': u,
        'v': v,
        'width_ratio': width_ratio,
        'height_ratio': height_ratio,
        'area_ratio': area_ratio,
        'occupancy_ratio': occupancy_ratio,
        'visible_enough': visible_enough,
        'too_large': too_large,
        'acceptable': acceptable,
        'score': float(score),
    }


def build_sim_for_target(base_mod: Any, traj: dict, scene_source: str, width: int, height: int, sensor_height_m: float, hfov_deg: float):
    return base_mod.build_dual_agent_simulator(
        scene_dataset_config=str(traj['dataset_config']),
        scene_id=scene_source,
        obs_width=int(width),
        obs_height=int(height),
        sensor_height_m=float(sensor_height_m),
        obs_hfov_deg=float(hfov_deg),
        topdown_width=64,
        topdown_height=64,
        topdown_hfov_deg=100.0,
    )


def compute_pitch_from_camera_eye(base_world_xyz: np.ndarray, target_world_xyz: np.ndarray, sensor_height_m: float) -> float:
    eye_xyz = np.asarray([
        float(base_world_xyz[0]),
        float(base_world_xyz[1]) + float(sensor_height_m),
        float(base_world_xyz[2]),
    ], dtype=np.float32)
    return float(RENDER_FP_MOD.compute_pitch_down_deg(eye_xyz, target_world_xyz))


def build_operating_pitch_candidates(*, base_pitch_deg: float, fallback_pitch_deg: float, pitch_max_deg: float, step_deg: float) -> list[float]:
    values: list[float] = []
    clipped_max = max(0.0, float(pitch_max_deg))
    clipped_step = max(1.0, float(step_deg))

    def add(value: float) -> None:
        clipped = float(np.clip(float(value), 0.0, clipped_max))
        if any(abs(clipped - existing) <= 1e-6 for existing in values):
            return
        values.append(clipped)

    base = float(np.clip(float(base_pitch_deg), 0.0, clipped_max))
    fallback = float(np.clip(float(fallback_pitch_deg), 0.0, clipped_max))
    add(base)
    add(fallback)

    delta = clipped_step
    while delta <= clipped_max + 1e-6:
        add(base + delta)
        add(base - delta)
        delta += clipped_step

    return values if values else [fallback]


def build_operating_camera_candidates(*, pathfinder: Any, operating_world_xyz: np.ndarray, heading_focus_world_xyz: np.ndarray,
                                      ref_y: float, backoff_max_m: float, backoff_step_m: float) -> list[dict]:
    candidates: list[dict] = []

    def append_candidate(camera_world_xyz: np.ndarray | None, backoff_m: float) -> None:
        if camera_world_xyz is None:
            return
        camera = np.asarray(camera_world_xyz, dtype=np.float32)
        cam_xz = np.asarray([float(camera[0]), float(camera[2])], dtype=np.float32)
        for existing in candidates:
            prev = np.asarray(existing['camera_world_xyz'], dtype=np.float32)
            prev_xz = np.asarray([float(prev[0]), float(prev[2])], dtype=np.float32)
            if float(np.linalg.norm(cam_xz - prev_xz)) <= 1e-3:
                return
        candidates.append({
            'camera_world_xyz': camera.copy(),
            'camera_on_operating_point': bool(float(backoff_m) <= 1e-6),
            'camera_backoff_m': float(max(0.0, backoff_m)),
        })

    append_candidate(np.asarray(operating_world_xyz, dtype=np.float32), 0.0)

    focus_vec = np.asarray([
        float(heading_focus_world_xyz[0]) - float(operating_world_xyz[0]),
        float(heading_focus_world_xyz[2]) - float(operating_world_xyz[2]),
    ], dtype=np.float32)
    focus_norm = float(np.linalg.norm(focus_vec))
    if focus_norm <= 1e-6:
        return candidates
    focus_dir = focus_vec / focus_norm

    max_backoff = max(0.0, float(backoff_max_m))
    step = max(0.05, float(backoff_step_m))
    backoff_m = step
    while backoff_m <= max_backoff + 1e-6:
        raw_x = float(operating_world_xyz[0]) - float(focus_dir[0]) * float(backoff_m)
        raw_z = float(operating_world_xyz[2]) - float(focus_dir[1]) * float(backoff_m)
        snapped = RENDER_FP_MOD.precise_snap(pathfinder, raw_x, raw_z, float(ref_y))
        append_candidate(snapped, float(backoff_m))
        backoff_m += step

    return candidates


def operating_candidate_sort_score(*, candidate_eval: dict | None, camera_backoff_m: float, vfov_deg: float,
                                   vfov_default_deg: float, occupancy_threshold: float) -> float:
    if candidate_eval is None:
        return 1e9
    score = float(candidate_eval.get('score', 1e6))
    occupancy_ratio = float(candidate_eval.get('occupancy_ratio', 0.0))
    if not bool(candidate_eval.get('acceptable', False)):
        score += 10.0
    if bool(candidate_eval.get('too_large', False)):
        score += 2.0
    score += max(0.0, float(occupancy_threshold) - occupancy_ratio) * 5.0
    score += max(0.0, float(camera_backoff_m)) * 0.05
    score += max(0.0, float(vfov_deg) - float(vfov_default_deg)) * 0.02
    return float(score)


def compute_heading_from_operating_edge(*, annotation_target: dict, p2w: dict, target_center_world_xz: np.ndarray, nav_y: float) -> np.ndarray | None:
    edge = annotation_target.get('operating_edge_line') if isinstance(annotation_target, dict) else None
    if not isinstance(edge, list) or len(edge) != 2:
        return None
    p0, p1 = edge
    if not (isinstance(p0, list) and len(p0) == 2 and isinstance(p1, list) and len(p1) == 2):
        return None
    px0 = int(round(float(p0[0]) * float(p2w['out_width']) / 999.0))
    py0 = int(round(float(p0[1]) * float(p2w['out_height']) / 999.0))
    px1 = int(round(float(p1[0]) * float(p2w['out_width']) / 999.0))
    py1 = int(round(float(p1[1]) * float(p2w['out_height']) / 999.0))
    px0 = int(np.clip(px0, 0, max(0, int(p2w['out_width']) - 1)))
    py0 = int(np.clip(py0, 0, max(0, int(p2w['out_height']) - 1)))
    px1 = int(np.clip(px1, 0, max(0, int(p2w['out_width']) - 1)))
    py1 = int(np.clip(py1, 0, max(0, int(p2w['out_height']) - 1)))
    wx0, wz0 = RENDER_FP_MOD.image_xy_to_world_xz(float(px0), float(py0), p2w)
    wx1, wz1 = RENDER_FP_MOD.image_xy_to_world_xz(float(px1), float(py1), p2w)
    edge_vec = np.asarray([float(wx1 - wx0), float(wz1 - wz0)], dtype=np.float32)
    edge_norm = float(np.linalg.norm(edge_vec))
    if edge_norm < 1e-6:
        return None
    edge_dir = edge_vec / edge_norm
    normal_a = np.asarray([-edge_dir[1], edge_dir[0]], dtype=np.float32)
    normal_b = -normal_a
    midpoint_xz = np.asarray([(float(wx0) + float(wx1)) * 0.5, (float(wz0) + float(wz1)) * 0.5], dtype=np.float32)
    target_vec = np.asarray(target_center_world_xz, dtype=np.float32) - midpoint_xz
    normal = normal_a if float(np.dot(normal_a, target_vec)) >= float(np.dot(normal_b, target_vec)) else normal_b
    heading_point_xz = midpoint_xz + normal
    return np.asarray([float(heading_point_xz[0]), float(nav_y), float(heading_point_xz[1])], dtype=np.float32)


def main() -> None:
    args = build_parser().parse_args()
    stage3_root = args.stage3_root.expanduser().resolve()
    accepted_file = args.accepted_file.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    final_root = stage3_root / 'final_observations'
    plan_root = stage3_root / 'plan'
    if not final_root.exists():
        raise FileNotFoundError(final_root)
    if not plan_root.exists():
        raise FileNotFoundError(plan_root)

    accepted_ids = load_accepted_ids(accepted_file)
    if int(args.max_targets) > 0:
        accepted_ids = accepted_ids[: int(args.max_targets)]

    base_mod = load_module(args.scripts_dir / 'render_local_topdown.py', '_accepted_base_scene')
    done = 0
    skipped = 0
    errors = 0
    for target_uid in accepted_ids:
        target_dir = out_root / target_uid.split('__', 1)[0] / target_uid
        accepted_rgb_path = target_dir / 'accepted_observation_rgb.png'
        accepted_overlay_path = target_dir / 'accepted_observation_overlay.png'
        accepted_meta_path = target_dir / 'accepted_observation_meta.json'
        operating_rgb_path = target_dir / 'operating_point_view_rgb.png'
        operating_overlay_path = target_dir / 'operating_point_view_overlay.png'
        operating_meta_path = target_dir / 'operating_point_view_meta.json'
        if bool(int(args.accepted_only)):
            ready = accepted_rgb_path.exists() and accepted_meta_path.exists() and accepted_overlay_path.exists()
        else:
            ready = accepted_rgb_path.exists() and accepted_meta_path.exists() and accepted_overlay_path.exists() and operating_rgb_path.exists() and operating_meta_path.exists() and operating_overlay_path.exists()
        if bool(int(args.resume)) and ready:
            skipped += 1
            continue
        manifest_path = final_root / f'{target_uid}__manifest.json'
        if not manifest_path.exists():
            errors += 1
            continue
        try:
            manifest = load_json(manifest_path)
            plan = load_json(plan_root / target_uid / 'result.json')
            expanded_meta = load_json(Path(plan['expanded_sample_dir']) / 'meta.json')
            stage_meta = load_json(Path(expanded_meta['source_meta_json']))
            traj = load_json(Path(stage_meta['trajectory_json']))
            scene_source = str(traj.get('scene_path', '')).strip() or str(traj['scene_id'])
            target_index = int(manifest['target_index'])
            nav_y = float(stage_meta['nav_y'])
            target_row = stage_meta['targets'][target_index]
            fallback_xz = np.asarray(target_row.get('topdown_center_world_xz', [0.0, 0.0]), dtype=np.float32)
            target_focus_world_xyz = RENDER_FP_MOD.resolve_target_focus_world_xyz(target_row, fallback_xz, nav_y)
            target_semantic_ids = [int(v) for v in manifest.get('target_semantic_ids', []) if int(v) > 0]
            if not target_semantic_ids:
                raise RuntimeError('missing target_semantic_ids')

            src_img = cv2.imread(str(Path(manifest['final_observation_with_points_png']).expanduser().resolve()))
            if src_img is not None:
                height, width = src_img.shape[:2]
            else:
                width = int(args.obs_width)
                height = int(args.obs_height)
            accepted_hfov = float(manifest['fp_hfov_deg'])
            accepted_vfov = float(manifest['fp_vfov_deg'])

            sim = build_sim_for_target(
                base_mod=base_mod,
                traj=traj,
                scene_source=scene_source,
                width=int(width),
                height=int(height),
                sensor_height_m=float(args.sensor_height_m),
                hfov_deg=float(accepted_hfov),
            )
            try:
                pf = sim.pathfinder
                accepted_obs_xz = np.asarray(manifest['observer_point_used']['observer_world_xz'], dtype=np.float32)
                accepted_camera_xyz = RENDER_FP_MOD.precise_snap(pf, float(accepted_obs_xz[0]), float(accepted_obs_xz[1]), nav_y)
                if accepted_camera_xyz is None:
                    raise RuntimeError('failed to snap accepted observer')
                operating_world_xz = np.asarray(plan['operating_point_world_xz'], dtype=np.float32)
                operating_world_xyz = RENDER_FP_MOD.precise_snap(pf, float(operating_world_xz[0]), float(operating_world_xz[1]), nav_y)
                if operating_world_xyz is None:
                    raise RuntimeError('failed to snap operating point')

                accepted_view = render_clean_view(
                    base_mod=base_mod,
                    sim=sim,
                    camera_world_xyz=accepted_camera_xyz,
                    yaw_deg=float(manifest['fp_yaw_deg']),
                    pitch_down_deg=float(manifest['fp_pitch_down_deg']),
                    target_semantic_ids=target_semantic_ids,
                    target_focus_world_xyz=target_focus_world_xyz,
                    operating_world_xyz=operating_world_xyz,
                    width=int(width),
                    height=int(height),
                    open_k=int(args.mask_open_k),
                    close_k=int(args.mask_close_k),
                    min_area_px=int(args.mask_min_area_px),
                )

                target_center_world_xz = np.asarray(target_row.get('topdown_center_world_xz', fallback_xz), dtype=np.float32)
                initial_focus_world_xyz = target_focus_world_xyz.copy()
                heading_focus_world_xyz = initial_focus_world_xyz.copy()
                if str(args.operating_heading_mode) == 'edge_normal':
                    p2w = stage_meta['topdown']['pixel_to_world_xz']
                    heading_from_edge = compute_heading_from_operating_edge(
                        annotation_target=plan.get('annotation_target', {}),
                        p2w=p2w,
                        target_center_world_xz=target_center_world_xz,
                        nav_y=nav_y,
                    )
                    if heading_from_edge is not None:
                        heading_focus_world_xyz = np.asarray([
                            float(heading_from_edge[0]),
                            float(initial_focus_world_xyz[1]),
                            float(heading_from_edge[2]),
                        ], dtype=np.float32)
                op_yaw_deg = RENDER_FP_MOD.yaw_between(operating_world_xyz, heading_focus_world_xyz)
                base_pitch_deg = compute_pitch_from_camera_eye(
                    operating_world_xyz,
                    initial_focus_world_xyz,
                    float(args.sensor_height_m),
                )
                op_vfov_candidates = []
                for vfov_deg in [float(args.operating_vfov_default), float(args.operating_vfov_max)]:
                    if not any(abs(vfov_deg - existing) <= 1e-6 for existing in op_vfov_candidates):
                        op_vfov_candidates.append(vfov_deg)

                operating_view = None
                fallback_operating_view = None
                fallback_camera_xyz = operating_world_xyz.copy()
                fallback_camera_on_operating_point = True
                fallback_camera_backoff_m = 0.0
                chosen_hfov = float(accepted_hfov)
                chosen_vfov = float(op_vfov_candidates[0])
                chosen_pitch_deg = float(base_pitch_deg)
                chosen_yaw_deg = float(op_yaw_deg)
                chosen_eval = None
                operating_skip_reason = None
                chosen_camera_world_xyz = operating_world_xyz.copy()
                chosen_camera_on_operating_point = True
                chosen_camera_backoff_m = 0.0
                fallback_score = 1e9

                for vfov_deg in op_vfov_candidates:
                    hfov_deg = RENDER_FP_MOD.vfov_to_hfov_deg(vfov_deg, int(width), int(height))
                    if abs(float(hfov_deg) - float(chosen_hfov)) > 1e-3:
                        sim.close()
                        sim = build_sim_for_target(
                            base_mod=base_mod,
                            traj=traj,
                            scene_source=scene_source,
                            width=int(width),
                            height=int(height),
                            sensor_height_m=float(args.sensor_height_m),
                            hfov_deg=float(hfov_deg),
                        )
                        pf = sim.pathfinder
                        accepted_camera_xyz = RENDER_FP_MOD.precise_snap(pf, float(accepted_obs_xz[0]), float(accepted_obs_xz[1]), nav_y)
                        operating_world_xyz = RENDER_FP_MOD.precise_snap(pf, float(operating_world_xz[0]), float(operating_world_xz[1]), nav_y)
                        if accepted_camera_xyz is None or operating_world_xyz is None:
                            raise RuntimeError('failed to re-snap after hfov switch')

                    camera_candidates = build_operating_camera_candidates(
                        pathfinder=pf,
                        operating_world_xyz=operating_world_xyz,
                        heading_focus_world_xyz=heading_focus_world_xyz,
                        ref_y=nav_y,
                        backoff_max_m=float(args.operating_backoff_max_m),
                        backoff_step_m=float(args.operating_backoff_step_m),
                    )

                    for camera_info in camera_candidates:
                        camera_world_xyz = np.asarray(camera_info['camera_world_xyz'], dtype=np.float32)
                        yaw_deg = float(RENDER_FP_MOD.yaw_between(camera_world_xyz, heading_focus_world_xyz))
                        camera_base_pitch_deg = compute_pitch_from_camera_eye(
                            camera_world_xyz,
                            initial_focus_world_xyz,
                            float(args.sensor_height_m),
                        )
                        pitch_candidates = build_operating_pitch_candidates(
                            base_pitch_deg=float(camera_base_pitch_deg),
                            fallback_pitch_deg=float(args.operating_pitch_deg),
                            pitch_max_deg=float(args.operating_pitch_max_deg),
                            step_deg=float(args.operating_pitch_step_deg),
                        )

                        for pitch_deg in pitch_candidates:
                            candidate = None
                            candidate_eval = None
                            try:
                                candidate = render_clean_view(
                                    base_mod=base_mod,
                                    sim=sim,
                                    camera_world_xyz=camera_world_xyz,
                                    yaw_deg=float(yaw_deg),
                                    pitch_down_deg=float(pitch_deg),
                                    target_semantic_ids=target_semantic_ids,
                                    target_focus_world_xyz=initial_focus_world_xyz,
                                    operating_world_xyz=None,
                                    width=int(width),
                                    height=int(height),
                                    open_k=int(args.mask_open_k),
                                    close_k=int(args.mask_close_k),
                                    min_area_px=int(args.mask_min_area_px),
                                )
                            except Exception:
                                candidate = None

                            if candidate is None:
                                continue

                            candidate_eval = evaluate_operating_candidate(candidate, int(width), int(height))
                            candidate_score = operating_candidate_sort_score(
                                candidate_eval=candidate_eval,
                                camera_backoff_m=float(camera_info['camera_backoff_m']),
                                vfov_deg=float(vfov_deg),
                                vfov_default_deg=float(args.operating_vfov_default),
                                occupancy_threshold=float(args.operating_min_occupancy_ratio),
                            )
                            if candidate_score < fallback_score:
                                fallback_score = float(candidate_score)
                                fallback_operating_view = candidate
                                fallback_camera_xyz = camera_world_xyz.copy()
                                fallback_camera_on_operating_point = bool(camera_info['camera_on_operating_point'])
                                fallback_camera_backoff_m = float(camera_info['camera_backoff_m'])
                                chosen_hfov = float(hfov_deg)
                                chosen_vfov = float(vfov_deg)
                                chosen_pitch_deg = float(pitch_deg)
                                chosen_yaw_deg = float(yaw_deg)
                                chosen_eval = candidate_eval

                            occupancy_ratio = float(candidate_eval['occupancy_ratio']) if candidate_eval is not None else 0.0
                            candidate_ok = bool(candidate_eval is not None and candidate_eval.get('acceptable', False))
                            if candidate_ok and occupancy_ratio >= float(args.operating_min_occupancy_ratio):
                                operating_view = candidate
                                chosen_camera_world_xyz = camera_world_xyz.copy()
                                chosen_camera_on_operating_point = bool(camera_info['camera_on_operating_point'])
                                chosen_camera_backoff_m = float(camera_info['camera_backoff_m'])
                                chosen_hfov = float(hfov_deg)
                                chosen_vfov = float(vfov_deg)
                                chosen_pitch_deg = float(pitch_deg)
                                chosen_yaw_deg = float(yaw_deg)
                                chosen_eval = candidate_eval
                                break

                        if operating_view is not None:
                            break

                    if operating_view is not None:
                        break

                if operating_view is None and fallback_operating_view is not None:
                    operating_view = fallback_operating_view
                    chosen_camera_world_xyz = fallback_camera_xyz.copy()
                    chosen_camera_on_operating_point = bool(fallback_camera_on_operating_point)
                    chosen_camera_backoff_m = float(fallback_camera_backoff_m)
                    operating_skip_reason = 'forced_saved_for_inspection'
                if operating_view is None and operating_skip_reason is None:
                    operating_skip_reason = 'target_not_well_centered_from_operating_point'
                    raw_rgb = render_raw_rgb_view(
                        base_mod=base_mod,
                        sim=sim,
                        camera_world_xyz=chosen_camera_world_xyz,
                        yaw_deg=float(chosen_yaw_deg),
                        pitch_down_deg=float(chosen_pitch_deg),
                    )
                    operating_view = {
                        'rgb': raw_rgb,
                        'target_bbox_xyxy': None,
                        'target_center_xy': None,
                        'target_mask_area_px': 0,
                    }
                    chosen_hfov = float(RENDER_FP_MOD.vfov_to_hfov_deg(float(op_vfov_candidates[0]), int(width), int(height)))
                    chosen_vfov = float(op_vfov_candidates[0])
                    chosen_pitch_deg = float(chosen_pitch_deg)
                    chosen_yaw_deg = float(chosen_yaw_deg)
                target_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(accepted_rgb_path), cv2.cvtColor(accepted_view['rgb'], cv2.COLOR_RGB2BGR))
                cv2.imwrite(
                    str(accepted_overlay_path),
                    cv2.cvtColor(
                        draw_overlay(
                            accepted_view['rgb'],
                            bbox_xyxy=accepted_view['target_bbox_xyxy'],
                            center_xy=accepted_view['target_center_xy'],
                            op_xy=accepted_view['operating_rgb_xy'],
                        ),
                        cv2.COLOR_RGB2BGR,
                    ),
                )

                accepted_meta = {
                    'view_type': 'accepted_observation',
                    'target_uid': target_uid,
                    'scene_id': manifest['scene_id'],
                    'obs_id': manifest['obs_id'],
                    'target_index': int(manifest['target_index']),
                    'target_name': manifest['target_name'],
                    'source_stage3_manifest': str(manifest_path),
                    'source_plan_json': str(plan_root / target_uid / 'result.json'),
                    'image_path': str(accepted_rgb_path),
                    'overlay_image_path': str(accepted_overlay_path),
                    'image_wh': [int(width), int(height)],
                    'camera_world_xyz': [float(v) for v in accepted_camera_xyz.tolist()],
                    'camera_world_xz': [float(accepted_camera_xyz[0]), float(accepted_camera_xyz[2])],
                    'yaw_deg': float(manifest['fp_yaw_deg']),
                    'pitch_down_deg': float(manifest['fp_pitch_down_deg']),
                    'hfov_deg': float(accepted_hfov),
                    'vfov_deg': float(accepted_vfov),
                    'target_semantic_ids': [int(v) for v in target_semantic_ids],
                    'target_focus_world_xyz': [float(v) for v in target_focus_world_xyz.tolist()],
                    'heading_focus_world_xyz': [float(v) for v in heading_focus_world_xyz.tolist()],
                    'target_bbox_xyxy': [int(v) for v in accepted_view['target_bbox_xyxy']],
                    'target_center_xy': [int(v) for v in accepted_view['target_center_xy']],
                    'target_mask_area_px': int(accepted_view['target_mask_area_px']),
                    'operating_point_world_xyz': [float(v) for v in operating_world_xyz.tolist()],
                    'operating_point_world_xz': [float(v) for v in operating_world_xz.tolist()],
                    'operating_point_rgb_xy': accepted_view['operating_rgb_xy'],
                    'observer_point_used': manifest['observer_point_used'],
                    'observer_point_original': manifest['observer_point_original'],
                }
                atomic_write_json(accepted_meta_path, accepted_meta)

                if bool(int(args.accepted_only)):
                    done += 1
                    if bool(int(args.verbose)):
                        print(f'[done-accepted-only] {target_uid}', flush=True)
                    continue

                if operating_view is None:
                    atomic_write_json(operating_meta_path, {
                        'view_type': 'operating_point_view',
                        'target_uid': target_uid,
                        'scene_id': manifest['scene_id'],
                        'obs_id': manifest['obs_id'],
                        'target_index': int(manifest['target_index']),
                        'target_name': manifest['target_name'],
                        'source_stage3_manifest': str(manifest_path),
                        'source_plan_json': str(plan_root / target_uid / 'result.json'),
                        'image_wh': [int(width), int(height)],
                        'camera_world_xyz': [float(v) for v in chosen_camera_world_xyz.tolist()],
                        'camera_world_xz': [float(chosen_camera_world_xyz[0]), float(chosen_camera_world_xyz[2])],
                        'camera_on_operating_point': bool(chosen_camera_on_operating_point),
                        'camera_backoff_m': float(chosen_camera_backoff_m),
                        'status': 'skipped',
                        'skip_reason': str(operating_skip_reason),
                        'target_semantic_ids': [int(v) for v in target_semantic_ids],
                        'target_focus_world_xyz': [float(v) for v in initial_focus_world_xyz.tolist()],
                        'heading_focus_world_xyz': [float(v) for v in heading_focus_world_xyz.tolist()],
                        'operating_point_world_xyz': [float(v) for v in operating_world_xyz.tolist()],
                        'operating_point_world_xz': [float(v) for v in operating_world_xz.tolist()],
                        'operating_point_rgb_xy': None,
                    })
                    skipped += 1
                    if bool(int(args.verbose)):
                        print(f'[skip-operating] {target_uid}: {operating_skip_reason}', flush=True)
                    done += 1
                    continue

                cv2.imwrite(str(operating_rgb_path), cv2.cvtColor(operating_view['rgb'], cv2.COLOR_RGB2BGR))
                if operating_view.get('target_bbox_xyxy') is not None and operating_view.get('target_center_xy') is not None:
                    operating_overlay_rgb = draw_overlay(
                        operating_view['rgb'],
                        bbox_xyxy=operating_view['target_bbox_xyxy'],
                        center_xy=operating_view['target_center_xy'],
                        op_xy=None,
                    )
                else:
                    operating_overlay_rgb = operating_view['rgb']
                cv2.imwrite(str(operating_overlay_path), cv2.cvtColor(operating_overlay_rgb, cv2.COLOR_RGB2BGR))
                operating_meta = {
                    'view_type': 'operating_point_view',
                    'target_uid': target_uid,
                    'scene_id': manifest['scene_id'],
                    'obs_id': manifest['obs_id'],
                    'target_index': int(manifest['target_index']),
                    'target_name': manifest['target_name'],
                    'source_stage3_manifest': str(manifest_path),
                    'source_plan_json': str(plan_root / target_uid / 'result.json'),
                    'image_path': str(operating_rgb_path),
                    'overlay_image_path': str(operating_overlay_path),
                    'image_wh': [int(width), int(height)],
                    'camera_world_xyz': [float(v) for v in chosen_camera_world_xyz.tolist()],
                    'camera_world_xz': [float(chosen_camera_world_xyz[0]), float(chosen_camera_world_xyz[2])],
                    'camera_on_operating_point': bool(chosen_camera_on_operating_point),
                    'camera_backoff_m': float(chosen_camera_backoff_m),
                    'status': 'saved' if operating_skip_reason is None else 'forced_saved',
                    'selection_note': (
                        'camera_backoff_used'
                        if operating_skip_reason is None and not bool(chosen_camera_on_operating_point)
                        else (None if operating_skip_reason is None else str(operating_skip_reason))
                    ),
                    'yaw_deg': float(chosen_yaw_deg),
                    'pitch_down_deg': float(chosen_pitch_deg),
                    'hfov_deg': float(chosen_hfov),
                    'vfov_deg': float(chosen_vfov),
                    'heading_mode': str(args.operating_heading_mode),
                    'target_occupancy_ratio': None if chosen_eval is None else float(chosen_eval['occupancy_ratio']),
                    'target_bbox_width_ratio': None if chosen_eval is None else float(chosen_eval['width_ratio']),
                    'target_bbox_height_ratio': None if chosen_eval is None else float(chosen_eval['height_ratio']),
                    'target_area_ratio': None if chosen_eval is None else float(chosen_eval['area_ratio']),
                    'vfov_default_deg': float(args.operating_vfov_default),
                    'vfov_max_deg': float(args.operating_vfov_max),
                    'occupancy_threshold': float(args.operating_min_occupancy_ratio),
                    'target_semantic_ids': [int(v) for v in target_semantic_ids],
                    'target_focus_world_xyz': [float(v) for v in initial_focus_world_xyz.tolist()],
                    'heading_focus_world_xyz': [float(v) for v in heading_focus_world_xyz.tolist()],
                    'target_bbox_xyxy': [int(v) for v in operating_view['target_bbox_xyxy']] if operating_view.get('target_bbox_xyxy') is not None else None,
                    'target_center_xy': [int(v) for v in operating_view['target_center_xy']] if operating_view.get('target_center_xy') is not None else None,
                    'target_mask_area_px': int(operating_view['target_mask_area_px']),
                    'operating_point_world_xyz': [float(v) for v in operating_world_xyz.tolist()],
                    'operating_point_world_xz': [float(v) for v in operating_world_xz.tolist()],
                    'operating_point_rgb_xy': None,
                }
                atomic_write_json(operating_meta_path, operating_meta)
                done += 1
                if bool(int(args.verbose)):
                    print(f'[done] {target_uid}', flush=True)
            finally:
                sim.close()
        except Exception as e:
            errors += 1
            err_path = target_dir / 'error.json'
            atomic_write_json(err_path, {
                'target_uid': target_uid,
                'error': str(e),
            })
            if bool(int(args.verbose)):
                print(f'[error] {target_uid}: {e}', flush=True)

    summary = {
        'stage3_root': str(stage3_root),
        'accepted_file': str(accepted_file),
        'out_root': str(out_root),
        'requested_count': len(accepted_ids),
        'done': int(done),
        'skipped': int(skipped),
        'errors': int(errors),
    }
    atomic_write_json(out_root / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
