#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import heapq
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import habitat_sim
import numpy as np


HERE = Path(__file__).resolve().parent
TEMP_VIDEO_FOURCC = 'mp4v'
FINAL_VIDEO_CODEC = 'h264_avc1'


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Render observer->operating curved-motion video with operating-point projection.')
    p.add_argument('--stage3-root', type=Path, default=Path('.'))
    p.add_argument('--accepted-file', type=Path, default=Path('accepted_target_uids.txt'))
    p.add_argument('--target-uid', type=str, default='00009-vLpv2VX547B__obs_0037_view_03_yaw290_rgb__t00__dining_table')
    p.add_argument('--scripts-dir', type=Path, default=HERE.parent / 'habitat_scripts')
    p.add_argument('--framework-final-dir', type=Path, default=HERE)
    p.add_argument('--out-root', type=Path, default=None)
    p.add_argument('--sensor-height-m', type=float, default=1.5)
    p.add_argument('--frame-count', type=int, default=72)
    p.add_argument('--fps', type=int, default=15)
    p.add_argument('--control-scale', type=float, default=0.45)
    p.add_argument('--topdown-debug', type=int, default=1)
    p.add_argument('--storyboard-frames', type=int, default=8)
    p.add_argument('--override-vfov-deg', type=float, default=0.0)
    p.add_argument('--override-pitch-deg', type=float, default=-1.0)
    p.add_argument('--dynamic-pitch', type=int, default=0)
    p.add_argument('--dynamic-pitch-bias-deg', type=float, default=0.0)
    p.add_argument('--dynamic-pitch-max-deg', type=float, default=60.0)
    p.add_argument('--dynamic-pitch-step-deg', type=float, default=3.0)
    p.add_argument('--force-render-width', type=int, default=0)
    p.add_argument('--force-render-height', type=int, default=0)
    p.add_argument('--override-hfov-deg', type=float, default=0.0)
    p.add_argument('--target-bottom-safe-ratio', type=float, default=0.92)
    p.add_argument('--target-center-v-ratio', type=float, default=0.74)
    p.add_argument('--require-curve-navmask', type=int, default=1)
    p.add_argument('--endpoint-offset-m', type=float, default=0.0)
    p.add_argument('--stop-on-op-invisible', type=int, default=0)
    p.add_argument('--require-start-facing-visible', type=int, default=0)
    p.add_argument('--start-facing-border-margin-px', type=int, default=8)
    p.add_argument('--path-shape', choices=['curve', 'shortest_path', 'safe_nav_curve', 'structured_approach', 'controller_approach', 'staged_approach', 'normal_close_approach', 'twist_replay'], default='curve')
    p.add_argument('--terminal-approach-dist-m', type=float, default=0.50)
    p.add_argument('--terminal-approach-min-dist-m', type=float, default=0.25)
    p.add_argument('--align-start-ratio', type=float, default=0.12)
    p.add_argument('--align-terminal-ratio', type=float, default=0.14)
    p.add_argument('--final-approach-ratio', type=float, default=0.18)
    p.add_argument('--controller-forward-speed-mps', type=float, default=0.28)
    p.add_argument('--controller-final-speed-mps', type=float, default=0.16)
    p.add_argument('--controller-max-yaw-rate-degps', type=float, default=45.0)
    p.add_argument('--controller-yaw-gain', type=float, default=2.2)
    p.add_argument('--controller-curve-lookahead-m', type=float, default=0.55)
    p.add_argument('--staged-target-speed-mps', type=float, default=0.28)
    p.add_argument('--staged-final-speed-mps', type=float, default=0.16)
    p.add_argument('--staged-max-yaw-rate-degps', type=float, default=60.0)
    p.add_argument('--staged-auto-frame-count', type=int, default=0)
    p.add_argument('--observer-world-xz', type=str, default='', help='Optional observer x,z override, e.g. "1.0,2.0".')
    p.add_argument('--start-yaw-deg', type=float, default=None, help='Optional start yaw override for the first video frame/path tangent.')
    p.add_argument('--start-pitch-deg', type=float, default=None, help='Optional start pitch override; applied before --override-pitch-deg.')
    p.add_argument('--replay-actions-npy', type=Path, default=None, help='Body-frame Twist actions, shape [T,3]=[linear_x_mps,linear_y_mps,angular_z_radps].')
    p.add_argument('--run-tag', type=str, default='', help='Optional suffix subdirectory under target uid for variants.')
    return p


def parse_xz_pair(text: str) -> np.ndarray | None:
    text = str(text or '').strip()
    if not text:
        return None
    parts = [p.strip() for p in text.split(',') if p.strip()]
    if len(parts) != 2:
        raise ValueError(f'expected x,z pair, got {text!r}')
    return np.asarray([float(parts[0]), float(parts[1])], dtype=np.float32)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def load_ids(path: Path) -> list[str]:
    rows = [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    seen = set()
    out = []
    for row in rows:
        if row in seen:
            continue
        seen.add(row)
        out.append(row)
    return out


def normalize_xz(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n < 1e-8:
        return np.asarray([0.0, 0.0], dtype=np.float32)
    return (vec / n).astype(np.float32)


def yaw_to_forward_xz(yaw_deg: float) -> np.ndarray:
    yaw_rad = math.radians(float(yaw_deg))
    return np.asarray([-math.sin(yaw_rad), -math.cos(yaw_rad)], dtype=np.float32)


def direction_to_yaw_deg(dir_xz: np.ndarray) -> float:
    dx = float(dir_xz[0])
    dz = float(dir_xz[1])
    return float((-math.degrees(math.atan2(dx, -dz))) % 360.0)


def signed_angle_delta_deg(start_deg: float, end_deg: float) -> float:
    return float((float(end_deg) - float(start_deg) + 180.0) % 360.0 - 180.0)


def build_base_action_rows(frame_rows: list[dict], fps: float) -> tuple[list[dict], np.ndarray, np.ndarray]:
    dt = 1.0 / max(1.0, float(fps))
    rows: list[dict] = []
    world_action_array = np.zeros((len(frame_rows), 3), dtype=np.float32)
    body_action_array = np.zeros((len(frame_rows), 3), dtype=np.float32)
    for idx, frame in enumerate(frame_rows):
        curr_xz = frame.get('camera_eye_world_xz') or frame.get('camera_world_xz')
        if idx == 0:
            vx = 0.0
            vy = 0.0
            vyaw = 0.0
        else:
            prev = frame_rows[idx - 1]
            prev_xz = prev.get('camera_eye_world_xz') or prev.get('camera_world_xz')
            vx = (float(curr_xz[0]) - float(prev_xz[0])) / dt
            vy = (float(curr_xz[1]) - float(prev_xz[1])) / dt
            vyaw = signed_angle_delta_deg(float(prev.get('yaw_deg', 0.0)), float(frame.get('yaw_deg', 0.0))) / dt
        action_yaw_deg = float(frame.get('yaw_deg', 0.0)) if idx == 0 else float(frame_rows[idx - 1].get('yaw_deg', 0.0))
        yaw_rad = math.radians(action_yaw_deg)
        forward = np.asarray([-math.sin(yaw_rad), -math.cos(yaw_rad)], dtype=np.float32)
        left = np.asarray([-math.cos(yaw_rad), math.sin(yaw_rad)], dtype=np.float32)
        world_velocity = np.asarray([float(vx), float(vy)], dtype=np.float32)
        world_action_array[idx] = [float(vx), float(vy), float(vyaw)]
        body_action_array[idx] = [
            float(np.dot(world_velocity, forward)),
            float(np.dot(world_velocity, left)),
            float(math.radians(float(vyaw))),
        ]
        rows.append({
            'frame_index': int(frame.get('frame_index', idx)),
            't_sec': float(idx) / max(1.0, float(fps)),
            'x': float(curr_xz[0]),
            'z': float(curr_xz[1]),
            'yaw_deg': float(frame.get('yaw_deg', 0.0)),
            'pitch_down_deg': float(frame.get('pitch_down_deg', 0.0)),
            'vx_mps': float(vx),
            'vy_mps': float(vy),
            'vyaw_degps': float(vyaw),
            'linear_x_mps': float(body_action_array[idx, 0]),
            'linear_y_mps': float(body_action_array[idx, 1]),
            'angular_z_radps': float(body_action_array[idx, 2]),
        })
    return rows, world_action_array, body_action_array


def write_base_actions(*, target_dir: Path, frame_rows: list[dict], fps: float) -> dict:
    action_rows, world_action_array, body_action_array = build_base_action_rows(frame_rows, fps=fps)
    json_path = target_dir / 'actions_30hz.json'
    csv_path = target_dir / 'actions_30hz.csv'
    world_npy_path = target_dir / 'actions_30hz.npy'
    body_npy_path = target_dir / 'actions_body_twist_30hz.npy'
    np.save(str(world_npy_path), world_action_array.astype(np.float32))
    np.save(str(body_npy_path), body_action_array.astype(np.float32))
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'frame_index', 't_sec', 'x', 'z', 'yaw_deg', 'pitch_down_deg',
            'vx_mps', 'vy_mps', 'vyaw_degps',
            'linear_x_mps', 'linear_y_mps', 'angular_z_radps',
        ])
        writer.writeheader()
        writer.writerows(action_rows)
    speeds = np.linalg.norm(body_action_array[:, :2], axis=1) if body_action_array.size else np.zeros((0,), dtype=np.float32)
    abs_angular_z = np.abs(body_action_array[:, 2]) if body_action_array.size else np.zeros((0,), dtype=np.float32)
    abs_vyaw_degps = np.abs(world_action_array[:, 2]) if world_action_array.size else np.zeros((0,), dtype=np.float32)
    payload = {
        'fps': float(fps),
        'dt': 1.0 / max(1.0, float(fps)),
        'coordinate_frame': 'body/base_link',
        'action_order': ['linear_x_mps', 'linear_y_mps', 'angular_z_radps'],
        'legacy_world_coordinate_frame': 'world_xz_eye_locked',
        'legacy_world_action_order': ['vx_mps', 'vy_mps', 'vyaw_degps'],
        'count': int(len(action_rows)),
        'max_speed_mps': float(np.max(speeds)) if speeds.size else 0.0,
        'mean_speed_mps': float(np.mean(speeds)) if speeds.size else 0.0,
        'max_abs_vyaw_degps': float(np.max(abs_vyaw_degps)) if abs_vyaw_degps.size else 0.0,
        'mean_abs_vyaw_degps': float(np.mean(abs_vyaw_degps)) if abs_vyaw_degps.size else 0.0,
        'max_abs_angular_z_radps': float(np.max(abs_angular_z)) if abs_angular_z.size else 0.0,
        'mean_abs_angular_z_radps': float(np.mean(abs_angular_z)) if abs_angular_z.size else 0.0,
        'rows': action_rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return {
        'json': str(json_path),
        'csv': str(csv_path),
        'npy': str(body_npy_path),
        'body_twist_npy': str(body_npy_path),
        'legacy_world_npy': str(world_npy_path),
        'summary': {k: payload[k] for k in [
            'fps', 'dt', 'coordinate_frame', 'action_order', 'count',
            'max_speed_mps', 'mean_speed_mps',
            'max_abs_vyaw_degps', 'mean_abs_vyaw_degps',
            'max_abs_angular_z_radps', 'mean_abs_angular_z_radps',
        ]},
    }


def integrate_body_twist_actions(*, start_xz: np.ndarray, start_yaw_deg: float,
                                 actions: np.ndarray, fps: float) -> tuple[np.ndarray, list[float]]:
    dt = 1.0 / max(1.0, float(fps))
    pos = np.asarray(start_xz, dtype=np.float32).copy()
    yaw = float(start_yaw_deg)
    points = [pos.copy()]
    yaws = [float(yaw)]
    arr = np.asarray(actions, dtype=np.float32)
    for idx in range(1, arr.shape[0]):
        linear_x, linear_y, angular_z = [float(v) for v in arr[idx].tolist()]
        yaw_rad = math.radians(float(yaw))
        forward = np.asarray([-math.sin(yaw_rad), -math.cos(yaw_rad)], dtype=np.float32)
        left = np.asarray([-math.cos(yaw_rad), math.sin(yaw_rad)], dtype=np.float32)
        velocity = forward * float(linear_x) + left * float(linear_y)
        pos = (pos + velocity * dt).astype(np.float32)
        yaw = float((yaw + math.degrees(float(angular_z) * dt)) % 360.0)
        points.append(pos.copy())
        yaws.append(float(yaw))
    return np.asarray(points, dtype=np.float32), yaws


def build_twist_replay_samples(*, start_xz: np.ndarray, start_yaw_deg: float,
                               actions_npy: Path, fps: float) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, float]], dict]:
    actions = np.load(str(actions_npy)).astype(np.float32)
    if actions.ndim != 2 or actions.shape[1] != 3:
        raise ValueError(f'expected actions shape [T,3], got {actions.shape}')
    points_xz, yaws = integrate_body_twist_actions(
        start_xz=np.asarray(start_xz, dtype=np.float32),
        start_yaw_deg=float(start_yaw_deg),
        actions=actions,
        fps=float(fps),
    )
    samples = []
    for idx, point in enumerate(points_xz):
        yaw = float(yaws[idx])
        samples.append((point.astype(np.float32), yaw_to_forward_xz(yaw), float(idx) / max(1.0, float(points_xz.shape[0] - 1))))
    return points_xz, samples, {
        'path_planner': 'twist_replay',
        'replay_actions_npy': str(actions_npy),
        'replay_action_order': ['linear_x_mps', 'linear_y_mps', 'angular_z_radps'],
        'replay_action_frame': 'body/base_link',
        'replay_frame_count': int(points_xz.shape[0]),
        'yaw_override_deg': [float(v) for v in yaws],
    }


def rect_distance_xz(point_xz: np.ndarray, rect_min_x: float, rect_max_x: float, rect_min_z: float, rect_max_z: float) -> float:
    x = float(point_xz[0])
    z = float(point_xz[1])
    dx = 0.0 if rect_min_x <= x <= rect_max_x else min(abs(x - rect_min_x), abs(x - rect_max_x))
    dz = 0.0 if rect_min_z <= z <= rect_max_z else min(abs(z - rect_min_z), abs(z - rect_max_z))
    return float(math.hypot(dx, dz))


def topdown_bbox_world_rect(target_row: dict, p2w: dict) -> tuple[float, float, float, float] | None:
    bbox = target_row.get('topdown_bbox_xyxy')
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return None
    x0, y0, x1, y1 = [int(v) for v in bbox]

    def image_xy_to_world_xz(px: int, py: int) -> tuple[float, float]:
        u = float(p2w['u_min']) + (float(px) + 0.5) * (float(p2w['u_max']) - float(p2w['u_min'])) / float(p2w['out_width'])
        f = float(p2w['f_front']) - (float(py) + 0.5) * (float(p2w['f_front']) - float(p2w['f_back'])) / float(p2w['out_height'])
        wx = float(p2w['stand_x']) + float(p2w['right_x']) * u + float(p2w['forward_x']) * f
        wz = float(p2w['stand_z']) + float(p2w['right_z']) * u + float(p2w['forward_z']) * f
        return float(wx), float(wz)

    pts = [
        image_xy_to_world_xz(x0, y0),
        image_xy_to_world_xz(x1, y0),
        image_xy_to_world_xz(x0, y1),
        image_xy_to_world_xz(x1, y1),
    ]
    xs = [p[0] for p in pts]
    zs = [p[1] for p in pts]
    return float(min(xs)), float(max(xs)), float(min(zs)), float(max(zs))


def find_offset_endpoint_on_navmesh(
    *,
    pathfinder: Any,
    render_fp: Any,
    operating_xz: np.ndarray,
    heading_focus_xz: np.ndarray,
    nav_y: float,
    desired_offset_m: float,
    target_rect_world: tuple[float, float, float, float] | None = None,
    step_m: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    base = np.asarray(operating_xz, dtype=np.float32)
    normal = normalize_xz(np.asarray(heading_focus_xz, dtype=np.float32) - base)
    if float(np.linalg.norm(normal)) < 1e-8:
        return None
    tangent = np.asarray([-normal[1], normal[0]], dtype=np.float32)
    desired = base + normal * float(desired_offset_m)

    radius_values = []
    r = max(0.05, float(desired_offset_m) - 0.05)
    while r <= float(desired_offset_m) + 0.30 + 1e-6:
        radius_values.append(round(r, 6))
        r += float(step_m)
    angle_values = [0.0]
    ang = 10.0
    while ang <= 120.0 + 1e-6:
        angle_values.extend([ang, -ang])
        ang += 10.0

    best = None
    best_score = float('inf')
    base_bbox_dist = None
    if target_rect_world is not None:
        base_bbox_dist = rect_distance_xz(base, *target_rect_world)
    for radius in radius_values:
        for ang_deg in angle_values:
            ang = math.radians(float(ang_deg))
            direction = normalize_xz(normal * math.cos(ang) + tangent * math.sin(ang))
            raw = base + direction * float(radius)
            snapped = render_fp.precise_snap(pathfinder, float(raw[0]), float(raw[1]), float(nav_y))
            if snapped is None:
                continue
            snapped_xz = np.asarray([float(snapped[0]), float(snapped[2])], dtype=np.float32)
            delta = snapped_xz - base
            longitudinal = float(np.dot(delta, normal))
            lateral_abs = abs(float(np.dot(delta, tangent)))
            if longitudinal <= 0.02:
                continue
            radial = float(np.linalg.norm(delta))
            snap_err = float(np.linalg.norm(snapped_xz - raw))
            bbox_gain = 0.0
            bbox_dist = None
            if target_rect_world is not None and base_bbox_dist is not None:
                bbox_dist = rect_distance_xz(snapped_xz, *target_rect_world)
                bbox_gain = float(bbox_dist - base_bbox_dist)
            angle_cost = abs(float(ang_deg)) / 120.0
            score = abs(radial - float(desired_offset_m)) * 3.0 + angle_cost * 1.5 + snap_err * 2.0 - bbox_gain * 2.5
            if score < best_score:
                best_score = score
                best = {
                    'raw_xz': raw.copy(),
                    'snapped_world_xyz': np.asarray(snapped, dtype=np.float32).copy(),
                    'snapped_xz': snapped_xz.copy(),
                    'longitudinal_m': float(longitudinal),
                    'lateral_m': float(lateral_abs),
                    'radial_m': float(radial),
                    'angle_deg': float(ang_deg),
                    'snap_error_m': float(snap_err),
                    'bbox_distance_m': None if bbox_dist is None else float(bbox_dist),
                    'bbox_distance_gain_m': float(bbox_gain),
                    'score': float(score),
                }
    if best is None:
        return None
    raw_xz = np.asarray(best['raw_xz'], dtype=np.float32)
    snapped_world_xyz = np.asarray(best['snapped_world_xyz'], dtype=np.float32)
    snapped_xz = np.asarray(best['snapped_xz'], dtype=np.float32)
    best['raw_xz'] = [float(v) for v in raw_xz.tolist()]
    best['snapped_world_xyz'] = [float(v) for v in snapped_world_xyz.tolist()]
    best['snapped_xz'] = [float(v) for v in snapped_xz.tolist()]
    return desired.astype(np.float32), snapped_xz.astype(np.float32), best


def build_motion_samples(
    *,
    pathfinder: Any,
    path_shape: str,
    obs_world_xyz: np.ndarray,
    endpoint_world_xyz: np.ndarray,
    start_yaw_deg: float,
    heading_focus_world_xyz: np.ndarray,
    frame_count: int,
    control_scale: float,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, float]]]:
    if str(path_shape) == 'shortest_path':
        shortest = habitat_sim.ShortestPath()
        shortest.requested_start = obs_world_xyz
        shortest.requested_end = endpoint_world_xyz
        found = bool(pathfinder.find_path(shortest))
        if not found or shortest.points is None or len(shortest.points) <= 0:
            path_points_xz = np.asarray([
                [float(obs_world_xyz[0]), float(obs_world_xyz[2])],
                [float(endpoint_world_xyz[0]), float(endpoint_world_xyz[2])],
            ], dtype=np.float32)
        else:
            path_points_xz = np.asarray([[float(p[0]), float(p[2])] for p in shortest.points], dtype=np.float32)
        return path_points_xz, sample_polyline_path(path_points_xz, int(frame_count))

    path_points_xz = np.asarray([
        [float(obs_world_xyz[0]), float(obs_world_xyz[2])],
        [float(endpoint_world_xyz[0]), float(endpoint_world_xyz[2])],
    ], dtype=np.float32)
    start_dir = yaw_to_forward_xz(float(start_yaw_deg))
    end_dir = normalize_xz(np.asarray([
        float(heading_focus_world_xyz[0]) - float(endpoint_world_xyz[0]),
        float(heading_focus_world_xyz[2]) - float(endpoint_world_xyz[2]),
    ], dtype=np.float32))
    travel_dist = float(np.linalg.norm(path_points_xz[1] - path_points_xz[0]))
    tangent_len = max(0.30, min(1.20, float(control_scale) * max(0.5, travel_dist)))
    p0 = path_points_xz[0].astype(np.float32)
    p3 = path_points_xz[1].astype(np.float32)
    p1 = (p0 + start_dir * float(tangent_len)).astype(np.float32)
    p2 = (p3 - end_dir * float(tangent_len)).astype(np.float32)
    return path_points_xz, sample_bezier_arc(p0, p1, p2, p3, int(frame_count))


def bezier_point(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: float) -> np.ndarray:
    u = 1.0 - float(t)
    return (
        (u ** 3) * p0
        + 3.0 * (u ** 2) * float(t) * p1
        + 3.0 * u * (float(t) ** 2) * p2
        + (float(t) ** 3) * p3
    ).astype(np.float32)


def bezier_tangent(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: float) -> np.ndarray:
    u = 1.0 - float(t)
    tangent = (
        3.0 * (u ** 2) * (p1 - p0)
        + 6.0 * u * float(t) * (p2 - p1)
        + 3.0 * (float(t) ** 2) * (p3 - p2)
    )
    return normalize_xz(np.asarray(tangent, dtype=np.float32))


def sample_bezier_arc(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, frame_count: int) -> list[tuple[np.ndarray, np.ndarray, float]]:
    dense_n = max(200, int(frame_count) * 8)
    ts = np.linspace(0.0, 1.0, dense_n, dtype=np.float32)
    pts = np.asarray([bezier_point(p0, p1, p2, p3, float(t)) for t in ts], dtype=np.float32)
    deltas = pts[1:] - pts[:-1]
    seg_lens = np.linalg.norm(deltas, axis=1)
    cum = np.concatenate([np.asarray([0.0], dtype=np.float32), np.cumsum(seg_lens, dtype=np.float32)])
    total = float(cum[-1])
    if total <= 1e-8:
        tangent = normalize_xz(p3 - p0)
        return [(p0.copy(), tangent.copy(), 0.0) for _ in range(int(frame_count))]
    targets = np.linspace(0.0, total, int(frame_count), dtype=np.float32)
    out = []
    for d in targets:
        idx = int(np.searchsorted(cum, d, side='right') - 1)
        idx = max(0, min(idx, dense_n - 2))
        d0 = float(cum[idx])
        d1 = float(cum[idx + 1])
        alpha = 0.0 if abs(d1 - d0) < 1e-8 else float((float(d) - d0) / (d1 - d0))
        t = float(ts[idx] * (1.0 - alpha) + ts[idx + 1] * alpha)
        point = bezier_point(p0, p1, p2, p3, t)
        tangent = bezier_tangent(p0, p1, p2, p3, t)
        out.append((point, tangent, t))
    return out


def sample_polyline_path(points_xz: np.ndarray, frame_count: int) -> list[tuple[np.ndarray, np.ndarray, float]]:
    pts = np.asarray(points_xz, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] <= 0 or pts.shape[1] != 2:
        raise RuntimeError('invalid polyline points')
    if pts.shape[0] == 1:
        return [(pts[0].copy(), np.asarray([0.0, -1.0], dtype=np.float32), 0.0) for _ in range(int(frame_count))]
    deltas = pts[1:] - pts[:-1]
    seg_lens = np.linalg.norm(deltas, axis=1)
    cum = np.concatenate([np.asarray([0.0], dtype=np.float32), np.cumsum(seg_lens, dtype=np.float32)])
    total = float(cum[-1])
    if total <= 1e-8:
        tangent = normalize_xz(pts[-1] - pts[0])
        return [(pts[0].copy(), tangent.copy(), 0.0) for _ in range(int(frame_count))]
    targets = np.linspace(0.0, total, int(frame_count), dtype=np.float32)
    tangent_window_m = min(0.45, max(0.12, total / max(6.0, float(frame_count) * 0.25)))

    def point_at_distance(distance_m: float) -> np.ndarray:
        d = float(np.clip(float(distance_m), 0.0, total))
        j = int(np.searchsorted(cum, d, side='right') - 1)
        j = max(0, min(j, pts.shape[0] - 2))
        jd0 = float(cum[j])
        jd1 = float(cum[j + 1])
        ja = 0.0 if abs(jd1 - jd0) < 1e-8 else float((d - jd0) / (jd1 - jd0))
        return np.asarray((1.0 - ja) * pts[j] + ja * pts[j + 1], dtype=np.float32)

    out = []
    for d in targets:
        idx = int(np.searchsorted(cum, d, side='right') - 1)
        idx = max(0, min(idx, pts.shape[0] - 2))
        d0 = float(cum[idx])
        d1 = float(cum[idx + 1])
        alpha = 0.0 if abs(d1 - d0) < 1e-8 else float((float(d) - d0) / (d1 - d0))
        point = (1.0 - alpha) * pts[idx] + alpha * pts[idx + 1]
        back = point_at_distance(float(d) - float(tangent_window_m))
        ahead = point_at_distance(float(d) + float(tangent_window_m))
        tangent = normalize_xz(ahead - back)
        if float(np.linalg.norm(tangent)) < 1e-8:
            tangent = normalize_xz(pts[idx + 1] - pts[idx])
        out.append((np.asarray(point, dtype=np.float32), tangent, float(d) / max(1e-8, total)))
    return out


def dedupe_path_points_xz(points_xz: np.ndarray, min_step_m: float = 0.03) -> np.ndarray:
    pts = np.asarray(points_xz, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] <= 1:
        return pts
    kept = [pts[0]]
    for point in pts[1:]:
        if float(np.linalg.norm(point - kept[-1])) >= float(min_step_m):
            kept.append(point)
    if float(np.linalg.norm(pts[-1] - kept[-1])) > 1e-6:
        kept.append(pts[-1])
    return np.asarray(kept, dtype=np.float32)


def path_xz_is_navsafe(points_xz: np.ndarray, p2w: dict, navmask: np.ndarray, step_m: float = 0.02) -> bool:
    pts = dedupe_path_points_xz(points_xz, min_step_m=1e-4)
    if pts.shape[0] <= 1:
        return True
    dense = []
    for start, end in zip(pts[:-1], pts[1:]):
        seg = sample_segment_xz(start, end, step_m=step_m)
        if dense:
            seg = seg[1:]
        dense.extend(seg.tolist())
    if not dense:
        dense = pts.tolist()
    return bool(validate_curve_points_against_navmask(
        draw_points_xz=np.asarray(dense, dtype=np.float32),
        p2w=p2w,
        navmask=navmask,
    ).get('valid', False))


def chaikin_smooth_path_xz_safe(points_xz: np.ndarray, p2w: dict, navmask: np.ndarray,
                                iterations: int = 4) -> tuple[np.ndarray, dict]:
    current = dedupe_path_points_xz(points_xz)
    if current.shape[0] <= 2:
        return current, {'smooth_curve_used': False, 'smooth_curve_reason': 'too_few_points'}
    accepted = current
    accepted_iterations = 0
    for _ in range(max(0, int(iterations))):
        nxt = [accepted[0]]
        for p0, p1 in zip(accepted[:-1], accepted[1:]):
            q = (0.78 * p0 + 0.22 * p1).astype(np.float32)
            r = (0.22 * p0 + 0.78 * p1).astype(np.float32)
            nxt.extend([q, r])
        nxt.append(accepted[-1])
        candidate = dedupe_path_points_xz(np.asarray(nxt, dtype=np.float32), min_step_m=0.015)
        if not path_xz_is_navsafe(candidate, p2w=p2w, navmask=navmask, step_m=0.02):
            break
        accepted = candidate
        accepted_iterations += 1
    return accepted, {
        'smooth_curve_used': bool(accepted_iterations > 0),
        'smooth_curve_iterations': int(accepted_iterations),
        'smooth_curve_input_points': int(current.shape[0]),
        'smooth_curve_output_points': int(accepted.shape[0]),
    }


def angle_lerp_deg(start_deg: float, end_deg: float, alpha: float) -> float:
    delta = signed_angle_delta_deg(start_deg, end_deg)
    return float((float(start_deg) + delta * float(np.clip(alpha, 0.0, 1.0))) % 360.0)


def smoothstep(alpha: float) -> float:
    a = float(np.clip(alpha, 0.0, 1.0))
    return float(a * a * (3.0 - 2.0 * a))


def sample_segment_xz(start_xz: np.ndarray, end_xz: np.ndarray, step_m: float = 0.02) -> np.ndarray:
    start = np.asarray(start_xz, dtype=np.float32)
    end = np.asarray(end_xz, dtype=np.float32)
    dist = float(np.linalg.norm(end - start))
    count = max(2, int(math.ceil(dist / max(1e-6, float(step_m)))) + 1)
    alphas = np.linspace(0.0, 1.0, count, dtype=np.float32)
    return np.asarray([(1.0 - float(a)) * start + float(a) * end for a in alphas], dtype=np.float32)


def find_terminal_preop_xz(*, op_xz: np.ndarray, target_focus_xz: np.ndarray, navmask: np.ndarray,
                           p2w: dict, desired_dist_m: float, min_dist_m: float) -> tuple[np.ndarray, dict] | None:
    op = np.asarray(op_xz, dtype=np.float32)
    focus = np.asarray(target_focus_xz, dtype=np.float32)
    approach_dir = normalize_xz(focus - op)
    if float(np.linalg.norm(approach_dir)) < 1e-8:
        return None
    desired = max(float(min_dist_m), float(desired_dist_m))
    preferred = [0.12, 0.16, 0.20, 0.25, 0.30, 0.40, 0.50]
    distances = []
    for dist in preferred:
        if float(min_dist_m) - 1e-6 <= float(dist) <= desired + 1e-6:
            distances.append(float(dist))
    for dist in np.linspace(float(min_dist_m), desired, 8, dtype=np.float32):
        distances.append(float(dist))
    distances = sorted({round(float(v), 4) for v in distances})
    for dist in distances:
        pre = (op - approach_dir * float(dist)).astype(np.float32)
        terminal_segment = sample_segment_xz(pre, op, step_m=0.02)
        if validate_curve_points_against_navmask(draw_points_xz=terminal_segment, p2w=p2w, navmask=navmask)['valid']:
            return pre, {
                'terminal_approach_dist_m': float(dist),
                'terminal_approach_dir_xz': [float(v) for v in approach_dir.tolist()],
                'terminal_yaw_deg': float(direction_to_yaw_deg(approach_dir)),
                'terminal_segment_check_points': int(terminal_segment.shape[0]),
            }
    return None


def point_at_fraction_on_polyline(points_xz: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray, int]:
    pts = dedupe_path_points_xz(points_xz, min_step_m=1e-4)
    if pts.shape[0] <= 1:
        return pts[0].copy(), np.asarray([0.0, -1.0], dtype=np.float32), 0
    deltas = pts[1:] - pts[:-1]
    seg_lens = np.linalg.norm(deltas, axis=1)
    cum = np.concatenate([np.asarray([0.0], dtype=np.float32), np.cumsum(seg_lens, dtype=np.float32)])
    total = float(cum[-1])
    if total <= 1e-8:
        return pts[0].copy(), normalize_xz(pts[-1] - pts[0]), 0
    distance = float(np.clip(float(fraction), 0.0, 1.0)) * total
    idx = int(np.searchsorted(cum, distance, side='right') - 1)
    idx = max(0, min(idx, pts.shape[0] - 2))
    d0 = float(cum[idx])
    d1 = float(cum[idx + 1])
    alpha = 0.0 if abs(d1 - d0) < 1e-8 else float((distance - d0) / (d1 - d0))
    point = ((1.0 - alpha) * pts[idx] + alpha * pts[idx + 1]).astype(np.float32)
    tangent = normalize_xz(pts[min(idx + 1, pts.shape[0] - 1)] - pts[max(0, idx - 1)])
    if float(np.linalg.norm(tangent)) < 1e-8:
        tangent = normalize_xz(pts[idx + 1] - pts[idx])
    return point, tangent, idx


def build_terminal_bezier_transition_xz(*, route_points_xz: np.ndarray, preop_xz: np.ndarray,
                                        terminal_dir: np.ndarray, p2w: dict, navmask: np.ndarray) -> tuple[np.ndarray, dict]:
    route = dedupe_path_points_xz(route_points_xz)
    preop = np.asarray(preop_xz, dtype=np.float32)
    terminal = normalize_xz(np.asarray(terminal_dir, dtype=np.float32))
    if route.shape[0] <= 2:
        return route, {'terminal_bezier_used': False, 'terminal_bezier_reason': 'too_few_route_points'}
    best = None
    for fraction in [0.45, 0.52, 0.60, 0.68, 0.75, 0.82]:
        anchor, anchor_tangent, anchor_idx = point_at_fraction_on_polyline(route, fraction)
        chord = float(np.linalg.norm(preop - anchor))
        if chord < 0.18:
            continue
        p0 = anchor
        p3 = preop
        p1 = (p0 + anchor_tangent * min(0.75, max(0.18, chord * 0.42))).astype(np.float32)
        p2 = (p3 - terminal * min(0.75, max(0.16, chord * 0.46))).astype(np.float32)
        dense_n = max(40, int(chord / 0.025))
        ts = np.linspace(0.0, 1.0, dense_n, dtype=np.float32)
        bezier_pts = np.asarray([bezier_point(p0, p1, p2, p3, float(t)) for t in ts], dtype=np.float32)
        prefix = route[:max(1, anchor_idx + 1)]
        candidate = dedupe_path_points_xz(np.vstack([prefix, bezier_pts]), min_step_m=0.015)
        if not path_xz_is_navsafe(candidate, p2w=p2w, navmask=navmask, step_m=0.018):
            continue
        score = chord - 0.08 * candidate.shape[0]
        if best is None or score > best[0]:
            best = (score, candidate, {
                'terminal_bezier_used': True,
                'terminal_bezier_fraction': float(fraction),
                'terminal_bezier_anchor_idx': int(anchor_idx),
                'terminal_bezier_chord_m': float(chord),
                'terminal_bezier_points': int(candidate.shape[0]),
            })
    if best is not None:
        return best[1], best[2]
    return route, {'terminal_bezier_used': False, 'terminal_bezier_reason': 'no_navsafe_bezier_candidate'}


def sample_controller_follow_path(*, path_points_xz: np.ndarray, start_yaw_deg: float, frame_count: int, fps: int,
                                  final_yaw_deg: float, p2w: dict, navmask: np.ndarray,
                                  forward_speed_mps: float, final_speed_mps: float,
                                  max_yaw_rate_degps: float, yaw_gain: float,
                                  final_segment_m: float, curve_lookahead_m: float) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, float]], dict]:
    path = dedupe_path_points_xz(path_points_xz, min_step_m=0.01)
    if path.shape[0] < 2:
        tangent = yaw_to_forward_xz(start_yaw_deg)
        samples = [(path[0].copy(), tangent.copy(), 0.0) for _ in range(int(frame_count))]
        return path, samples, {'path_planner': 'controller_follow', 'controller_fallback': 'too_few_points'}

    sample_count = max(int(frame_count), 2)
    ref_samples = sample_polyline_path(path, sample_count)
    ref_points = np.asarray([row[0] for row in ref_samples], dtype=np.float32)
    ref_tangents = np.asarray([row[1] for row in ref_samples], dtype=np.float32)
    deltas = ref_points[1:] - ref_points[:-1]
    seg_lens = np.linalg.norm(deltas, axis=1)
    cum = np.concatenate([np.asarray([0.0], dtype=np.float32), np.cumsum(seg_lens, dtype=np.float32)])
    total_len = float(cum[-1])
    dt = 1.0 / max(1.0, float(fps))
    pos = ref_points[0].copy()
    yaw = float(start_yaw_deg)
    progress_m = 0.0
    out_points: list[np.ndarray] = [pos.copy()]
    out_yaws: list[float] = [yaw]

    def point_tangent_at(distance_m: float) -> tuple[np.ndarray, np.ndarray]:
        d = float(np.clip(distance_m, 0.0, total_len))
        idx = int(np.searchsorted(cum, d, side='right') - 1)
        idx = max(0, min(idx, ref_points.shape[0] - 2))
        d0 = float(cum[idx])
        d1 = float(cum[idx + 1])
        alpha = 0.0 if abs(d1 - d0) < 1e-8 else float((d - d0) / (d1 - d0))
        point = ((1.0 - alpha) * ref_points[idx] + alpha * ref_points[idx + 1]).astype(np.float32)
        tangent = normalize_xz(((1.0 - alpha) * ref_tangents[idx] + alpha * ref_tangents[idx + 1]).astype(np.float32))
        if float(np.linalg.norm(tangent)) < 1e-8:
            tangent = normalize_xz(ref_points[idx + 1] - ref_points[idx])
        return point, tangent

    for _ in range(1, sample_count):
        remaining = max(0.0, total_len - progress_m)
        is_final = remaining <= max(0.10, float(final_segment_m) + 0.04)
        lookahead_point, lookahead_tangent = point_tangent_at(progress_m + float(curve_lookahead_m))
        path_yaw = float(direction_to_yaw_deg(lookahead_tangent))
        if is_final:
            final_alpha = float(np.clip(1.0 - remaining / max(0.10, float(final_segment_m) + 0.04), 0.0, 1.0))
            target_yaw = angle_lerp_deg(path_yaw, float(final_yaw_deg), smoothstep(final_alpha))
        else:
            target_yaw = path_yaw
        yaw_error = signed_angle_delta_deg(yaw, target_yaw)
        max_step = float(max_yaw_rate_degps) * dt
        yaw_step = float(np.clip(float(yaw_gain) * yaw_error * dt, -max_step, max_step))
        yaw = float((yaw + yaw_step) % 360.0)
        speed = float(final_speed_mps if is_final else forward_speed_mps)
        step_m = min(speed * dt, max(0.0, total_len - progress_m))
        progress_m = min(total_len, progress_m + step_m)
        point, _ = point_tangent_at(progress_m)
        pos = point.copy()
        out_points.append(pos)
        out_yaws.append(yaw)

    out_points_np = np.asarray(out_points, dtype=np.float32)
    if not path_xz_is_navsafe(out_points_np, p2w=p2w, navmask=navmask, step_m=0.02):
        samples = sample_polyline_path(path, int(frame_count))
        return path, samples, {'path_planner': 'controller_follow', 'controller_fallback': 'navmask_validation_failed'}

    while out_points_np.shape[0] < int(frame_count):
        out_points_np = np.vstack([out_points_np, out_points_np[-1:]])
        out_yaws.append(float(out_yaws[-1]))
    if out_points_np.shape[0] > int(frame_count):
        out_points_np = out_points_np[:int(frame_count)]
        out_yaws = out_yaws[:int(frame_count)]

    samples = []
    for idx, point in enumerate(out_points_np):
        next_point = out_points_np[min(idx + 1, out_points_np.shape[0] - 1)]
        tangent = yaw_to_forward_xz(out_yaws[idx]) if idx == out_points_np.shape[0] - 1 else normalize_xz(next_point - point)
        if float(np.linalg.norm(tangent)) < 1e-8:
            tangent = yaw_to_forward_xz(out_yaws[idx])
        samples.append((point.astype(np.float32), tangent.astype(np.float32), float(idx) / max(1.0, float(out_points_np.shape[0] - 1))))
    return out_points_np, samples, {
        'path_planner': 'controller_follow',
        'controller_forward_speed_mps': float(forward_speed_mps),
        'controller_final_speed_mps': float(final_speed_mps),
        'controller_max_yaw_rate_degps': float(max_yaw_rate_degps),
        'controller_yaw_gain': float(yaw_gain),
        'controller_curve_lookahead_m': float(curve_lookahead_m),
        'controller_total_path_len_m': float(total_len),
        'yaw_override_deg': [float(v) for v in out_yaws],
    }


def point_at_distance_on_polyline(points_xz: np.ndarray, distance_m: float) -> tuple[np.ndarray, np.ndarray]:
    pts = dedupe_path_points_xz(points_xz, min_step_m=1e-4)
    if pts.shape[0] <= 1:
        return pts[0].copy(), np.asarray([0.0, -1.0], dtype=np.float32)
    deltas = pts[1:] - pts[:-1]
    seg_lens = np.linalg.norm(deltas, axis=1)
    cum = np.concatenate([np.asarray([0.0], dtype=np.float32), np.cumsum(seg_lens, dtype=np.float32)])
    total = float(cum[-1])
    if total <= 1e-8:
        return pts[0].copy(), normalize_xz(pts[-1] - pts[0])
    d = float(np.clip(distance_m, 0.0, total))
    idx = int(np.searchsorted(cum, d, side='right') - 1)
    idx = max(0, min(idx, pts.shape[0] - 2))
    d0 = float(cum[idx])
    d1 = float(cum[idx + 1])
    alpha = 0.0 if abs(d1 - d0) < 1e-8 else float((d - d0) / (d1 - d0))
    point = ((1.0 - alpha) * pts[idx] + alpha * pts[idx + 1]).astype(np.float32)
    tangent = normalize_xz(pts[idx + 1] - pts[idx])
    return point, tangent


def sample_staged_approach_path(*, route_points_xz: np.ndarray, op_xz: np.ndarray, semantic_focus_xz: np.ndarray,
                                terminal_focus_xz: np.ndarray, start_yaw_deg: float, frame_count: int,
                                final_turn_dist_m: float, fps: float = 30.0,
                                target_speed_mps: float = 0.28, final_speed_mps: float = 0.16,
                                max_yaw_rate_degps: float = 60.0,
                                auto_frame_count: bool = False) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, float]], dict]:
    route = dedupe_path_points_xz(route_points_xz, min_step_m=0.01)
    start = route[0].astype(np.float32)
    op = np.asarray(op_xz, dtype=np.float32)
    semantic_focus = np.asarray(semantic_focus_xz, dtype=np.float32)
    terminal_focus = np.asarray(terminal_focus_xz, dtype=np.float32)
    semantic_dir = normalize_xz(semantic_focus - start)
    if float(np.linalg.norm(semantic_dir)) < 1e-8:
        semantic_dir = normalize_xz(route[min(1, route.shape[0] - 1)] - start)
    target_heading_yaw = float(direction_to_yaw_deg(semantic_dir))
    terminal_dir = normalize_xz(terminal_focus - op)
    terminal_yaw = float(direction_to_yaw_deg(terminal_dir))

    deltas = route[1:] - route[:-1]
    total_len = float(np.linalg.norm(deltas, axis=1).sum()) if route.shape[0] > 1 else 0.0
    final_dist = float(np.clip(float(final_turn_dist_m), 0.08, max(0.08, min(0.30, total_len * 0.45))))
    final_start_dist = max(0.0, total_len - final_dist)

    fps_safe = max(1.0, float(fps))
    target_speed = max(0.03, float(target_speed_mps))
    final_speed = max(0.03, float(final_speed_mps))
    max_yaw_rate = max(5.0, float(max_yaw_rate_degps))
    start_yaw_delta = abs(signed_angle_delta_deg(float(start_yaw_deg), target_heading_yaw))
    final_yaw_delta = abs(signed_angle_delta_deg(target_heading_yaw, terminal_yaw))
    if bool(auto_frame_count):
        yaw_safety = 1.6
        align_n = max(6, int(math.ceil(start_yaw_delta / max_yaw_rate * fps_safe * yaw_safety)))
        move_n = max(8, int(math.ceil(final_start_dist / target_speed * fps_safe)))
        final_n_by_dist = int(math.ceil(final_dist / final_speed * fps_safe))
        final_n_by_yaw = int(math.ceil(final_yaw_delta / max_yaw_rate * fps_safe * yaw_safety))
        final_n = max(8, final_n_by_dist, final_n_by_yaw)
        frame_count = int(align_n + move_n + final_n)
    else:
        align_n = max(6, min(16, int(round(frame_count * 0.10))))
        final_n = max(8, int(round(frame_count * 0.18)))
        move_n = max(8, int(frame_count) - align_n - final_n)
        if align_n + move_n + final_n != int(frame_count):
            move_n = max(8, int(frame_count) - align_n - final_n)

    samples: list[tuple[np.ndarray, np.ndarray, float]] = []
    yaw_rows: list[float] = []
    for idx in range(align_n):
        alpha = smoothstep(float(idx + 1) / float(align_n))
        yaw = angle_lerp_deg(float(start_yaw_deg), target_heading_yaw, alpha)
        samples.append((start.copy(), yaw_to_forward_xz(yaw), 0.0))
        yaw_rows.append(float(yaw))

    if move_n > 0:
        for idx in range(move_n):
            alpha = float(idx + 1) / float(move_n)
            dist = final_start_dist * alpha
            point, _ = point_at_distance_on_polyline(route, dist)
            yaw = float(target_heading_yaw)
            samples.append((point.astype(np.float32), yaw_to_forward_xz(yaw), float(dist / max(1e-8, total_len))))
            yaw_rows.append(float(yaw))

    for idx in range(final_n):
        alpha = smoothstep(float(idx + 1) / float(final_n))
        dist = final_start_dist + final_dist * alpha
        point, _ = point_at_distance_on_polyline(route, dist)
        yaw = angle_lerp_deg(target_heading_yaw, terminal_yaw, alpha)
        samples.append((point.astype(np.float32), yaw_to_forward_xz(yaw), float(dist / max(1e-8, total_len))))
        yaw_rows.append(float(yaw))

    if len(samples) > int(frame_count):
        samples = samples[:int(frame_count)]
        yaw_rows = yaw_rows[:int(frame_count)]
    while len(samples) < int(frame_count):
        samples.append((op.copy(), terminal_dir.copy(), 1.0))
        yaw_rows.append(float(terminal_yaw))
    if samples:
        samples[-1] = (op.copy(), terminal_dir.copy(), 1.0)
        yaw_rows[-1] = float(terminal_yaw)

    return np.asarray([row[0] for row in samples], dtype=np.float32), samples, {
        'path_planner': 'staged_approach',
        'start_align_frames': int(align_n),
        'move_frames': int(move_n),
        'final_turn_frames': int(final_n),
        'auto_frame_count': bool(auto_frame_count),
        'effective_frame_count': int(len(samples)),
        'fps': float(fps_safe),
        'target_speed_mps': float(target_speed),
        'final_speed_mps': float(final_speed),
        'max_yaw_rate_degps': float(max_yaw_rate),
        'total_path_len_m': float(total_len),
        'final_start_dist_m': float(final_start_dist),
        'final_turn_dist_m': float(final_dist),
        'start_yaw_delta_deg': float(start_yaw_delta),
        'final_yaw_delta_deg': float(final_yaw_delta),
        'target_heading_yaw_deg': float(target_heading_yaw),
        'terminal_yaw_deg': float(terminal_yaw),
        'yaw_override_deg': [float(v) for v in yaw_rows],
    }


def cubic_bezier_points_xz(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, count: int) -> np.ndarray:
    pts = []
    for alpha in np.linspace(0.0, 1.0, max(2, int(count)), dtype=np.float32):
        a = float(alpha)
        pts.append(
            ((1.0 - a) ** 3) * p0
            + 3.0 * ((1.0 - a) ** 2) * a * p1
            + 3.0 * (1.0 - a) * (a ** 2) * p2
            + (a ** 3) * p3
        )
    return np.asarray(pts, dtype=np.float32)


def sample_normal_close_approach_path(*, start_xz: np.ndarray, op_xz: np.ndarray, semantic_focus_xz: np.ndarray,
                                      terminal_focus_xz: np.ndarray, start_yaw_deg: float, frame_count: int,
                                      fps: float, target_speed_mps: float, final_speed_mps: float,
                                      max_yaw_rate_degps: float, p2w: dict, navmask: np.ndarray) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, float]], dict]:
    start = np.asarray(start_xz, dtype=np.float32)
    op = np.asarray(op_xz, dtype=np.float32)
    semantic_focus = np.asarray(semantic_focus_xz, dtype=np.float32)
    terminal_focus = np.asarray(terminal_focus_xz, dtype=np.float32)
    start_to_semantic = normalize_xz(semantic_focus - start)
    if float(np.linalg.norm(start_to_semantic)) < 1e-8:
        start_to_semantic = normalize_xz(op - start)
    terminal_dir = normalize_xz(terminal_focus - op)
    if float(np.linalg.norm(terminal_dir)) < 1e-8:
        terminal_dir = start_to_semantic.copy()
    start_face_yaw = float(direction_to_yaw_deg(start_to_semantic))
    terminal_yaw = float(direction_to_yaw_deg(terminal_dir))

    total_dist = float(np.linalg.norm(op - start))
    desired_final = min(0.22, max(0.08, total_dist * 0.28))

    chord = float(np.linalg.norm(op - start))
    start_handle_dir = yaw_to_forward_xz(float(start_yaw_deg))
    route_info: dict = {'ok': True, 'path_planner': 'direct_smooth_bezier', 'direct_line_used': True}
    if chord < 1e-4:
        path = np.asarray([start, op], dtype=np.float32)
    else:
        dense_count = max(160, int(math.ceil(chord / 0.004)))
        handle = float(np.clip(chord * 0.62, 0.10, 0.58))
        best_curve = None
        best_info = None
        for scale in [1.0, 0.85, 0.70, 0.55, 0.40, 0.25]:
            cand = cubic_bezier_points_xz(
                start,
                (start + start_handle_dir * handle * scale).astype(np.float32),
                (op - terminal_dir * handle * scale).astype(np.float32),
                op,
                count=dense_count,
            )
            nav_info = validate_curve_points_against_navmask(draw_points_xz=cand, p2w=p2w, navmask=navmask)
            if bool(nav_info.get('valid', False)):
                best_curve = cand
                best_info = {
                    'ok': True,
                    'path_planner': 'single_cubic_bezier_navsafe',
                    'bezier_handle_m': float(handle * scale),
                    'bezier_dense_points': int(cand.shape[0]),
                    'direct_line_used': False,
                }
                break
        if best_curve is not None:
            path = best_curve
            route_info = best_info or route_info
        else:
            route_points_xz, route_info = build_safe_nav_curve_path_xz_coarse(
                start_xz=start,
                goal_xz=op,
                p2w=p2w,
                navmask=navmask,
                safety_margin_px=1,
                coarse_scale=4,
            )
            route_info['direct_line_used'] = False
            if bool(route_info.get('ok', False)):
                smooth_route, smooth_info = chaikin_smooth_path_xz_safe(
                    route_points_xz,
                    p2w=p2w,
                    navmask=navmask,
                    iterations=5,
                )
                path = smooth_route
                route_info.update(smooth_info)
            elif path_xz_is_navsafe(np.asarray([start, op], dtype=np.float32), p2w=p2w, navmask=navmask, step_m=0.02):
                path = sample_segment_xz(start, op, step_m=0.006)
                route_info = {'ok': True, 'path_planner': 'direct_line_navsafe_fallback', 'direct_line_used': True}
            else:
                path = sample_segment_xz(start, op, step_m=0.006)
                route_info = {'ok': False, 'path_planner': 'unsafe_direct_fallback', 'direct_line_used': True}

    path = dedupe_path_points_xz(path, min_step_m=0.005)
    path_check = validate_curve_points_against_navmask(draw_points_xz=path, p2w=p2w, navmask=navmask)
    if not bool(path_check.get('valid', False)):
        path = dedupe_path_points_xz(sample_segment_xz(start, op, step_m=0.006), min_step_m=0.005)
        path_check = validate_curve_points_against_navmask(draw_points_xz=path, p2w=p2w, navmask=navmask)

    deltas = path[1:] - path[:-1]
    seg_lens = np.linalg.norm(deltas, axis=1)
    cum = np.concatenate([np.asarray([0.0], dtype=np.float32), np.cumsum(seg_lens, dtype=np.float32)])
    path_len = float(cum[-1])
    def point_at_distance(distance_m: float) -> np.ndarray:
        if path.shape[0] <= 1 or path_len <= 1e-8:
            return start.copy()
        d = float(np.clip(distance_m, 0.0, path_len))
        idx = int(np.searchsorted(cum, d, side='right') - 1)
        idx = max(0, min(idx, path.shape[0] - 2))
        d0 = float(cum[idx])
        d1 = float(cum[idx + 1])
        alpha = 0.0 if abs(d1 - d0) < 1e-8 else float((d - d0) / (d1 - d0))
        return ((1.0 - alpha) * path[idx] + alpha * path[idx + 1]).astype(np.float32)

    def tangent_at_distance(distance_m: float) -> np.ndarray:
        if path.shape[0] <= 1 or path_len <= 1e-8:
            return start_to_semantic.copy()
        window = min(0.08, max(0.025, path_len * 0.08))
        before = point_at_distance(float(distance_m) - window)
        after = point_at_distance(float(distance_m) + window)
        tangent = normalize_xz(after - before)
        if float(np.linalg.norm(tangent)) < 1e-8:
            tangent = start_to_semantic.copy()
        return tangent

    fps_safe = max(1.0, float(fps))
    max_yaw_rate = max(5.0, float(max_yaw_rate_degps))
    final_path_len = min(float(desired_final), max(0.0, path_len))
    preop = point_at_distance(max(0.0, path_len - final_path_len))
    target_v = max(0.03, float(target_speed_mps))
    move_n = max(16, int(math.ceil(path_len / target_v * fps_safe)) + 1)
    samples = sample_polyline_path(path, move_n)
    yaw_rows = [float(direction_to_yaw_deg(tangent_xz)) for _point, tangent_xz, _t_value in samples]
    if yaw_rows:
        final_blend_n = min(len(yaw_rows), max(8, int(round(0.75 * fps_safe))))
        blend_start = max(0, len(yaw_rows) - final_blend_n)
        bridge_yaw = float(yaw_rows[blend_start])
        for idx in range(blend_start, len(yaw_rows)):
            denom = max(1, len(yaw_rows) - blend_start - 1)
            alpha = smoothstep(float(idx - blend_start) / float(denom))
            yaw_rows[idx] = angle_lerp_deg(bridge_yaw, terminal_yaw, alpha)
        samples = [
            (point.copy(), yaw_to_forward_xz(float(yaw_rows[idx])), float(t_value))
            for idx, (point, _tangent, t_value) in enumerate(samples)
        ]

    if samples:
        samples[-1] = (op.copy(), yaw_to_forward_xz(terminal_yaw), 1.0)
        yaw_rows[-1] = float(terminal_yaw)

    stop_n = max(3, int(round(0.20 * fps_safe)))
    for _ in range(stop_n):
        samples.append((op.copy(), yaw_to_forward_xz(terminal_yaw), 1.0))
        yaw_rows.append(float(terminal_yaw))

    return path, samples, {
        'path_planner': 'normal_close_approach',
        'start_align_frames': 0,
        'move_frames': int(move_n),
        'final_turn_frames': 0,
        'final_approach_frames': 0,
        'stop_tail_frames': int(stop_n),
        'effective_frame_count': int(len(samples)),
        'start_yaw_deg': float(start_yaw_deg),
        'target_heading_yaw_deg': float(start_face_yaw),
        'terminal_yaw_deg': float(terminal_yaw),
        'start_yaw_delta_deg': float(abs(signed_angle_delta_deg(float(start_yaw_deg), start_face_yaw))),
        'path_len_m': float(path_len),
        'terminal_approach_dist_m': float(final_path_len),
        'preop_world_xz': [float(v) for v in preop.tolist()],
        'continuous_curve_motion': True,
        'arc_length_uniform_sampling': True,
        'local_retime': False,
        'target_speed_mps': float(target_v),
        'adaptive_yaw_rate_limit_degps': float(max_yaw_rate),
        'final_yaw_blend_frames': int(min(len(yaw_rows), max(8, int(round(0.75 * fps_safe))))),
        'route_info': route_info,
        'path_navmask_valid': bool(path_check.get('valid', False)),
        'yaw_override_deg': [float(v) for v in yaw_rows],
    }


def sample_structured_approach_path(*, start_xz: np.ndarray, op_xz: np.ndarray, preop_xz: np.ndarray,
                                    target_focus_xz: np.ndarray, semantic_focus_xz: np.ndarray,
                                    start_yaw_deg: float, frame_count: int, align_start_ratio: float,
                                    align_terminal_ratio: float, final_approach_ratio: float,
                                    p2w: dict, navmask: np.ndarray) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, float]], dict]:
    frame_count = max(8, int(frame_count))
    start = np.asarray(start_xz, dtype=np.float32)
    preop = np.asarray(preop_xz, dtype=np.float32)
    op = np.asarray(op_xz, dtype=np.float32)
    target_focus = np.asarray(target_focus_xz, dtype=np.float32)
    semantic_focus = np.asarray(semantic_focus_xz, dtype=np.float32)
    terminal_dir = normalize_xz(target_focus - op)
    terminal_yaw = float(direction_to_yaw_deg(terminal_dir))
    semantic_dir = normalize_xz(semantic_focus - start)
    if float(np.linalg.norm(semantic_dir)) < 1e-8:
        semantic_dir = normalize_xz(preop - start)
    if float(np.linalg.norm(semantic_dir)) < 1e-8:
        semantic_dir = terminal_dir.copy()
    start_face_yaw = float(direction_to_yaw_deg(semantic_dir))

    start_align_n = max(8, min(16, int(round(frame_count * 0.12))))
    final_n = max(6, int(round(frame_count * max(0.08, float(final_approach_ratio)))))
    post_curve_face_n = 0
    move_n = max(8, frame_count - start_align_n - final_n - post_curve_face_n)
    total_n = start_align_n + move_n + post_curve_face_n + final_n
    if total_n != frame_count:
        move_n = max(8, move_n + frame_count - total_n)

    mid_points_xz, planning_info = build_safe_nav_curve_path_xz_coarse(
        start_xz=start,
        goal_xz=preop,
        p2w=p2w,
        navmask=navmask,
        safety_margin_px=1,
        coarse_scale=4,
    )
    if not bool(planning_info.get('ok', False)):
        mid_points_xz = np.asarray([start, preop], dtype=np.float32)
    smooth_mid_points_xz, smooth_curve_info = chaikin_smooth_path_xz_safe(
        mid_points_xz,
        p2w=p2w,
        navmask=navmask,
        iterations=4,
    )
    if smooth_mid_points_xz.shape[0] >= 2:
        mid_points_xz = smooth_mid_points_xz
    planning_info.update(smooth_curve_info)
    terminal_transition_points_xz, terminal_transition_info = build_terminal_bezier_transition_xz(
        route_points_xz=mid_points_xz,
        preop_xz=preop,
        terminal_dir=terminal_dir,
        p2w=p2w,
        navmask=navmask,
    )
    if terminal_transition_points_xz.shape[0] >= 2:
        mid_points_xz = terminal_transition_points_xz
    planning_info.update(terminal_transition_info)
    # Use the first short forward segment after turning as a stable straight prefix,
    # then curve naturally into the terminal Bezier and finish with a very short vertical stop.
    facing_walk_m = 0.28
    facing_goal = start + semantic_dir * float(facing_walk_m)
    if not path_xz_is_navsafe(np.asarray([start, facing_goal], dtype=np.float32), p2w=p2w, navmask=navmask, step_m=0.02):
        for dist in [0.22, 0.18, 0.14, 0.10]:
            candidate = start + semantic_dir * float(dist)
            if path_xz_is_navsafe(np.asarray([start, candidate], dtype=np.float32), p2w=p2w, navmask=navmask, step_m=0.02):
                facing_goal = candidate.astype(np.float32)
                break
        else:
            facing_goal = start.copy()
    pre_curve_anchor = facing_goal.astype(np.float32)
    warmup_dist = min(0.08, max(0.0, float(np.linalg.norm(pre_curve_anchor - start)) * 0.35))
    warmup_end = (start + semantic_dir * float(warmup_dist)).astype(np.float32)
    if not path_xz_is_navsafe(np.asarray([start, warmup_end], dtype=np.float32), p2w=p2w, navmask=navmask, step_m=0.02):
        warmup_end = start.copy()
    terminal_prefix, prefix_info = build_safe_nav_curve_path_xz_coarse(
        start_xz=pre_curve_anchor,
        goal_xz=preop,
        p2w=p2w,
        navmask=navmask,
        safety_margin_px=1,
        coarse_scale=4,
    )
    if not bool(prefix_info.get('ok', False)):
        terminal_prefix = np.asarray([pre_curve_anchor, preop], dtype=np.float32)
    terminal_prefix, terminal_prefix_smooth_info = build_terminal_bezier_transition_xz(
        route_points_xz=terminal_prefix,
        preop_xz=preop,
        terminal_dir=terminal_dir,
        p2w=p2w,
        navmask=navmask,
    )
    prefix_polyline = dedupe_path_points_xz(np.vstack([warmup_end.reshape(1, 2), pre_curve_anchor.reshape(1, 2), terminal_prefix]), min_step_m=0.01)
    mid_samples = sample_polyline_path(prefix_polyline, move_n)
    mid_yaws_raw = [float(direction_to_yaw_deg(tangent)) for _, tangent, _ in mid_samples]
    if mid_yaws_raw:
        max_yaw_step_deg = 1.6
        mid_yaws = [float(start_face_yaw)]
        for raw_yaw in mid_yaws_raw[1:]:
            delta = signed_angle_delta_deg(mid_yaws[-1], raw_yaw)
            clipped_delta = float(np.clip(delta, -max_yaw_step_deg, max_yaw_step_deg))
            mid_yaws.append(float((mid_yaws[-1] + clipped_delta) % 360.0))
    else:
        mid_yaws = []

    samples: list[tuple[np.ndarray, np.ndarray, float]] = []
    yaw_rows: list[float] = []
    for idx in range(start_align_n):
        alpha = smoothstep(float(idx + 1) / float(start_align_n))
        yaw = angle_lerp_deg(float(start_yaw_deg), start_face_yaw, alpha)
        point = ((1.0 - alpha) * start + alpha * warmup_end).astype(np.float32)
        samples.append((point, yaw_to_forward_xz(yaw), 0.0))
        yaw_rows.append(float(yaw))

    mid_start_idx = len(samples)
    for sample_idx, (point, tangent, t_value) in enumerate(mid_samples):
        yaw = float(mid_yaws[sample_idx]) if sample_idx < len(mid_yaws) else float(direction_to_yaw_deg(tangent))
        samples.append((np.asarray(point, dtype=np.float32), yaw_to_forward_xz(yaw), float(t_value)))
        yaw_rows.append(float(yaw))

    last_mid_yaw = yaw_rows[-1] if yaw_rows else start_face_yaw
    for idx in range(post_curve_face_n):
        alpha = smoothstep(float(idx + 1) / float(post_curve_face_n))
        yaw = angle_lerp_deg(last_mid_yaw, terminal_yaw, alpha)
        samples.append((preop.copy(), yaw_to_forward_xz(yaw), 1.0))
        yaw_rows.append(float(yaw))

    for idx in range(final_n):
        alpha = smoothstep(float(idx + 1) / float(final_n))
        point = ((1.0 - alpha) * preop + alpha * op).astype(np.float32)
        yaw = angle_lerp_deg(last_mid_yaw, terminal_yaw, alpha)
        samples.append((point, yaw_to_forward_xz(yaw), 1.0))
        yaw_rows.append(float(yaw))

    if len(samples) > frame_count:
        samples = samples[:frame_count]
        yaw_rows = yaw_rows[:frame_count]
    while len(samples) < frame_count:
        samples.append((op.copy(), terminal_dir.copy(), 1.0))
        yaw_rows.append(float(terminal_yaw))

    path_points = np.asarray([row[0] for row in samples], dtype=np.float32)
    info = {
        'path_planner': 'structured_approach',
        'start_align_frames': int(start_align_n),
        'mid_move_frames': int(move_n),
        'terminal_align_frames': int(post_curve_face_n),
        'final_approach_frames': int(final_n),
        'start_move_yaw_deg': float(start_face_yaw),
        'terminal_yaw_deg': float(terminal_yaw),
        'preop_world_xz': [float(v) for v in preop.tolist()],
        'operating_world_xz': [float(v) for v in op.tolist()],
        'terminal_approach_dir_xz': [float(v) for v in terminal_dir.tolist()],
        'mid_path_planning_info': planning_info,
        'prefix_path_planning_info': prefix_info,
        'prefix_terminal_transition_info': terminal_prefix_smooth_info,
        'semantic_focus_world_xz': [float(v) for v in semantic_focus.tolist()],
        'facing_walk_world_xz': [float(v) for v in pre_curve_anchor.tolist()],
        'warmup_end_world_xz': [float(v) for v in warmup_end.tolist()],
        'yaw_override_deg': [float(v) for v in yaw_rows],
        'mid_start_frame_index': int(mid_start_idx),
    }
    return path_points, samples, info


def safe_navmask_with_margin(navmask: np.ndarray, margin_px: int = 2) -> np.ndarray:
    safe = np.asarray(navmask, dtype=np.uint8)
    k = max(0, int(margin_px))
    if k <= 0:
        return safe.astype(bool)
    kernel = np.ones((2 * k + 1, 2 * k + 1), dtype=np.uint8)
    eroded = cv2.erode(safe, kernel, iterations=1)
    return eroded.astype(bool)


def nearest_safe_pixel(px: int, py: int, safe_mask: np.ndarray, max_radius: int = 60) -> tuple[int, int] | None:
    h, w = safe_mask.shape[:2]
    if 0 <= px < w and 0 <= py < h and bool(safe_mask[py, px]):
        return int(px), int(py)
    best = None
    best_d2 = None
    for radius in range(1, int(max_radius) + 1):
        x0 = max(0, int(px) - radius)
        x1 = min(w - 1, int(px) + radius)
        y0 = max(0, int(py) - radius)
        y1 = min(h - 1, int(py) + radius)
        candidates = []
        for x in range(x0, x1 + 1):
            candidates.append((x, y0))
            candidates.append((x, y1))
        for y in range(y0 + 1, y1):
            candidates.append((x0, y))
            candidates.append((x1, y))
        for x, y in candidates:
            if not bool(safe_mask[y, x]):
                continue
            d2 = (int(x) - int(px)) ** 2 + (int(y) - int(py)) ** 2
            if best_d2 is None or d2 < best_d2:
                best = (int(x), int(y))
                best_d2 = int(d2)
        if best is not None:
            return best
    return None


def astar_on_safe_mask(start_xy: tuple[int, int], goal_xy: tuple[int, int], safe_mask: np.ndarray) -> list[tuple[int, int]] | None:
    h, w = safe_mask.shape[:2]
    sx, sy = [int(v) for v in start_xy]
    gx, gy = [int(v) for v in goal_xy]
    if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h):
        return None
    if not bool(safe_mask[sy, sx]) or not bool(safe_mask[gy, gx]):
        return None

    distance = cv2.distanceTransform(np.asarray(safe_mask, dtype=np.uint8), cv2.DIST_L2, 3)
    max_clearance = max(1.0, float(distance.max()))
    neighbors = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
    ]

    def heuristic(x: int, y: int) -> float:
        return float(math.hypot(int(x) - gx, int(y) - gy))

    start = (sx, sy)
    goal = (gx, gy)
    open_heap: list[tuple[float, float, tuple[int, int]]] = [(heuristic(sx, sy), 0.0, start)]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    best_g: dict[tuple[int, int], float] = {start: 0.0}
    closed: set[tuple[int, int]] = set()

    while open_heap:
        _f, g_score, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            path = [current]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            path.reverse()
            return path
        closed.add(current)
        cx, cy = current
        for dx, dy, step_cost in neighbors:
            nx = cx + dx
            ny = cy + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if not bool(safe_mask[ny, nx]):
                continue
            clearance = float(distance[ny, nx]) / max_clearance
            clearance_penalty = 0.25 * (1.0 - min(1.0, clearance))
            ng = float(g_score) + float(step_cost) + float(clearance_penalty)
            node = (int(nx), int(ny))
            if ng >= best_g.get(node, float('inf')):
                continue
            came_from[node] = current
            best_g[node] = ng
            heapq.heappush(open_heap, (ng + heuristic(nx, ny), ng, node))
    return None


def line_pixels_safe(a: tuple[int, int], b: tuple[int, int], safe_mask: np.ndarray, step_px: float = 1.0) -> bool:
    ax, ay = [float(v) for v in a]
    bx, by = [float(v) for v in b]
    dist = float(math.hypot(bx - ax, by - ay))
    steps = max(1, int(math.ceil(dist / max(0.5, float(step_px)))))
    h, w = safe_mask.shape[:2]
    for idx in range(steps + 1):
        t = float(idx) / float(steps)
        x = int(round(ax * (1.0 - t) + bx * t))
        y = int(round(ay * (1.0 - t) + by * t))
        if not (0 <= x < w and 0 <= y < h and bool(safe_mask[y, x])):
            return False
    return True


def shortcut_safe_path(path_xy: list[tuple[int, int]], safe_mask: np.ndarray) -> list[tuple[int, int]]:
    if len(path_xy) <= 2:
        return list(path_xy)
    out = [path_xy[0]]
    anchor = 0
    while anchor < len(path_xy) - 1:
        next_idx = anchor + 1
        for candidate in range(len(path_xy) - 1, anchor, -1):
            if line_pixels_safe(path_xy[anchor], path_xy[candidate], safe_mask):
                next_idx = candidate
                break
        out.append(path_xy[next_idx])
        anchor = next_idx
    return out


def densify_polyline_xy(points_xy: list[tuple[int, int]], step_px: float = 3.0) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.shape[0] <= 1:
        return pts
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        dist = float(np.linalg.norm(b - a))
        steps = max(1, int(math.ceil(dist / max(0.5, float(step_px)))))
        for idx in range(1, steps + 1):
            t = float(idx) / float(steps)
            out.append((1.0 - t) * a + t * b)
    return np.asarray(out, dtype=np.float32)


def smooth_xy_path_safe(points_xy: list[tuple[int, int]], safe_mask: np.ndarray, iterations: int = 2) -> list[tuple[int, int]]:
    pts = densify_polyline_xy(points_xy, step_px=4.0)
    if pts.shape[0] <= 2:
        return [(int(round(x)), int(round(y))) for x, y in pts.tolist()]
    for _ in range(max(0, int(iterations))):
        new_pts = [pts[0]]
        for idx in range(1, pts.shape[0] - 1):
            new_pts.append(0.25 * pts[idx - 1] + 0.50 * pts[idx] + 0.25 * pts[idx + 1])
        new_pts.append(pts[-1])
        candidate = np.asarray(new_pts, dtype=np.float32)
        rounded = [(int(round(x)), int(round(y))) for x, y in candidate.tolist()]
        if all(line_pixels_safe(a, b, safe_mask) for a, b in zip(rounded[:-1], rounded[1:])):
            pts = candidate
        else:
            break
    return [(int(round(x)), int(round(y))) for x, y in pts.tolist()]


def build_safe_nav_curve_path_xz(*, start_xz: np.ndarray, goal_xz: np.ndarray, p2w: dict,
                                 navmask: np.ndarray, safety_margin_px: int = 2) -> tuple[np.ndarray, dict]:
    start_xy = world_xz_to_image_xy_local(float(start_xz[0]), float(start_xz[1]), p2w)
    goal_xy = world_xz_to_image_xy_local(float(goal_xz[0]), float(goal_xz[1]), p2w)
    safe_mask = safe_navmask_with_margin(navmask, margin_px=int(safety_margin_px))
    h, w = safe_mask.shape[:2]
    route_span = max(abs(int(start_xy[0]) - int(goal_xy[0])), abs(int(start_xy[1]) - int(goal_xy[1])))
    margin = int(max(80, min(260, route_span * 0.65 + 80)))
    x0 = max(0, min(int(start_xy[0]), int(goal_xy[0])) - margin)
    x1 = min(w - 1, max(int(start_xy[0]), int(goal_xy[0])) + margin)
    y0 = max(0, min(int(start_xy[1]), int(goal_xy[1])) - margin)
    y1 = min(h - 1, max(int(start_xy[1]), int(goal_xy[1])) + margin)
    local_safe = safe_mask[y0:y1 + 1, x0:x1 + 1]
    local_start_xy = (int(start_xy[0]) - x0, int(start_xy[1]) - y0)
    local_goal_xy = (int(goal_xy[0]) - x0, int(goal_xy[1]) - y0)
    start_safe = nearest_safe_pixel(int(local_start_xy[0]), int(local_start_xy[1]), local_safe)
    goal_safe = nearest_safe_pixel(int(local_goal_xy[0]), int(local_goal_xy[1]), local_safe)
    if start_safe is None or goal_safe is None:
        return np.asarray([start_xz, goal_xz], dtype=np.float32), {
            'ok': False,
            'fallback_reason': 'start_or_goal_not_in_safe_mask',
            'start_xy': [int(start_xy[0]), int(start_xy[1])],
            'goal_xy': [int(goal_xy[0]), int(goal_xy[1])],
            'start_safe_xy': None if start_safe is None else [int(start_safe[0]), int(start_safe[1])],
            'goal_safe_xy': None if goal_safe is None else [int(goal_safe[0]), int(goal_safe[1])],
            'local_window_xyxy': [int(x0), int(y0), int(x1), int(y1)],
        }
    raw_path = astar_on_safe_mask(start_safe, goal_safe, local_safe)
    if raw_path is None:
        return np.asarray([start_xz, goal_xz], dtype=np.float32), {
            'ok': False,
            'fallback_reason': 'astar_failed',
            'start_xy': [int(start_xy[0]), int(start_xy[1])],
            'goal_xy': [int(goal_xy[0]), int(goal_xy[1])],
            'start_safe_xy': [int(start_safe[0]) + int(x0), int(start_safe[1]) + int(y0)],
            'goal_safe_xy': [int(goal_safe[0]) + int(x0), int(goal_safe[1]) + int(y0)],
            'local_window_xyxy': [int(x0), int(y0), int(x1), int(y1)],
        }
    shortcut = shortcut_safe_path(raw_path, local_safe)
    smooth = smooth_xy_path_safe(shortcut, local_safe, iterations=2)
    points_xz = []
    for px, py in smooth:
        wx, wz = image_xy_to_world_xz_local(int(px) + int(x0), int(py) + int(y0), p2w)
        points_xz.append([float(wx), float(wz)])
    if len(points_xz) < 2:
        points_xz = [start_xz.tolist(), goal_xz.tolist()]
    return np.asarray(points_xz, dtype=np.float32), {
        'ok': True,
        'path_planner': 'safe_nav_curve_astar_shortcut_smooth',
        'safety_margin_px': int(safety_margin_px),
        'local_window_xyxy': [int(x0), int(y0), int(x1), int(y1)],
        'local_window_shape_hw': [int(local_safe.shape[0]), int(local_safe.shape[1])],
        'start_xy': [int(start_xy[0]), int(start_xy[1])],
        'goal_xy': [int(goal_xy[0]), int(goal_xy[1])],
        'start_safe_xy': [int(start_safe[0]) + int(x0), int(start_safe[1]) + int(y0)],
        'goal_safe_xy': [int(goal_safe[0]) + int(x0), int(goal_safe[1]) + int(y0)],
        'astar_pixel_count': int(len(raw_path)),
        'shortcut_pixel_count': int(len(shortcut)),
        'smooth_pixel_count': int(len(smooth)),
    }


def downsample_safe_mask_for_planning(mask: np.ndarray, scale: int) -> np.ndarray:
    scale = max(1, int(scale))
    if scale <= 1:
        return np.asarray(mask, dtype=bool)
    h, w = mask.shape[:2]
    pad_h = int(math.ceil(h / scale) * scale - h)
    pad_w = int(math.ceil(w / scale) * scale - w)
    padded = np.pad(np.asarray(mask, dtype=bool), ((0, pad_h), (0, pad_w)), mode='constant', constant_values=False)
    coarse = padded.reshape(padded.shape[0] // scale, scale, padded.shape[1] // scale, scale).all(axis=(1, 3))
    return np.asarray(coarse, dtype=bool)


def coarse_xy_to_local_xy(point: tuple[int, int], scale: int, local_shape: tuple[int, int]) -> tuple[int, int]:
    h, w = local_shape
    x = min(w - 1, max(0, int(point[0]) * int(scale) + int(scale) // 2))
    y = min(h - 1, max(0, int(point[1]) * int(scale) + int(scale) // 2))
    return int(x), int(y)


def build_safe_nav_curve_path_xz_coarse(*, start_xz: np.ndarray, goal_xz: np.ndarray, p2w: dict,
                                        navmask: np.ndarray, safety_margin_px: int = 1,
                                        coarse_scale: int = 4) -> tuple[np.ndarray, dict]:
    start_xy = world_xz_to_image_xy_local(float(start_xz[0]), float(start_xz[1]), p2w)
    goal_xy = world_xz_to_image_xy_local(float(goal_xz[0]), float(goal_xz[1]), p2w)
    base_safe_mask = safe_navmask_with_margin(navmask, margin_px=int(safety_margin_px))
    h, w = base_safe_mask.shape[:2]
    route_span = max(abs(int(start_xy[0]) - int(goal_xy[0])), abs(int(start_xy[1]) - int(goal_xy[1])))
    margin = int(max(96, min(360, route_span * 0.85 + 120)))
    x0 = max(0, min(int(start_xy[0]), int(goal_xy[0])) - margin)
    x1 = min(w - 1, max(int(start_xy[0]), int(goal_xy[0])) + margin)
    y0 = max(0, min(int(start_xy[1]), int(goal_xy[1])) - margin)
    y1 = min(h - 1, max(int(start_xy[1]), int(goal_xy[1])) + margin)
    local_safe = np.asarray(base_safe_mask[y0:y1 + 1, x0:x1 + 1], dtype=bool)
    if not local_safe.any():
        local_safe = np.asarray(navmask[y0:y1 + 1, x0:x1 + 1], dtype=bool)

    local_start_xy = (int(start_xy[0]) - x0, int(start_xy[1]) - y0)
    local_goal_xy = (int(goal_xy[0]) - x0, int(goal_xy[1]) - y0)
    scale_values = [max(1, int(coarse_scale)), 2, 1]
    seen_scales: set[int] = set()
    attempts = []
    for scale in scale_values:
        if scale in seen_scales:
            continue
        seen_scales.add(scale)
        coarse_mask = downsample_safe_mask_for_planning(local_safe, scale)
        if not coarse_mask.any():
            attempts.append({'scale': int(scale), 'status': 'empty_coarse_mask'})
            continue
        coarse_start = (int(round(local_start_xy[0] / float(scale))), int(round(local_start_xy[1] / float(scale))))
        coarse_goal = (int(round(local_goal_xy[0] / float(scale))), int(round(local_goal_xy[1] / float(scale))))
        start_safe = nearest_safe_pixel(coarse_start[0], coarse_start[1], coarse_mask, max_radius=max(8, int(80 / scale)))
        goal_safe = nearest_safe_pixel(coarse_goal[0], coarse_goal[1], coarse_mask, max_radius=max(8, int(80 / scale)))
        if start_safe is None or goal_safe is None:
            attempts.append({'scale': int(scale), 'status': 'start_or_goal_not_safe'})
            continue
        raw_coarse = astar_on_safe_mask(start_safe, goal_safe, coarse_mask)
        if raw_coarse is None:
            attempts.append({'scale': int(scale), 'status': 'astar_failed'})
            continue

        raw_local = [coarse_xy_to_local_xy(p, scale, local_safe.shape[:2]) for p in raw_coarse]
        raw_full = [(int(px) + int(x0), int(py) + int(y0)) for px, py in raw_local]
        raw_check = all(line_pixels_safe(a, b, navmask) for a, b in zip(raw_full[:-1], raw_full[1:]))
        shortcut_full = raw_full
        if raw_check and len(raw_full) > 18:
            sparse_step = max(2, int(round(len(raw_full) / 18.0)))
            sparse = raw_full[::sparse_step]
            if sparse[-1] != raw_full[-1]:
                sparse.append(raw_full[-1])
            if all(line_pixels_safe(a, b, navmask) for a, b in zip(sparse[:-1], sparse[1:])):
                shortcut_full = sparse
        smooth_full = smooth_xy_path_safe(shortcut_full, np.asarray(navmask, dtype=bool), iterations=4)
        smooth_ok = all(line_pixels_safe(a, b, navmask) for a, b in zip(smooth_full[:-1], smooth_full[1:]))
        final_xy = smooth_full if smooth_ok else shortcut_full
        final_ok = all(line_pixels_safe(a, b, navmask) for a, b in zip(final_xy[:-1], final_xy[1:]))
        if not final_ok:
            attempts.append({'scale': int(scale), 'status': 'fullres_validation_failed', 'raw_points': int(len(raw_full))})
            continue
        points_xz = []
        for px, py in final_xy:
            wx, wz = image_xy_to_world_xz_local(int(px), int(py), p2w)
            points_xz.append([float(wx), float(wz)])
        if len(points_xz) < 2:
            attempts.append({'scale': int(scale), 'status': 'too_few_points'})
            continue
        return np.asarray(points_xz, dtype=np.float32), {
            'ok': True,
            'path_planner': 'safe_nav_curve_coarse_navmask_astar',
            'coarse_scale': int(scale),
            'safety_margin_px': int(safety_margin_px),
            'local_window_xyxy': [int(x0), int(y0), int(x1), int(y1)],
            'local_window_shape_hw': [int(local_safe.shape[0]), int(local_safe.shape[1])],
            'coarse_shape_hw': [int(coarse_mask.shape[0]), int(coarse_mask.shape[1])],
            'astar_pixel_count': int(len(raw_coarse)),
            'raw_fullres_point_count': int(len(raw_full)),
            'shortcut_point_count': int(len(shortcut_full)),
            'smooth_point_count': int(len(smooth_full)),
            'final_point_count': int(len(final_xy)),
            'smooth_used': bool(smooth_ok),
            'attempts': attempts,
        }

    return np.asarray([start_xz, goal_xz], dtype=np.float32), {
        'ok': False,
        'path_planner': 'safe_nav_curve_coarse_navmask_astar',
        'fallback_reason': 'all_coarse_astar_attempts_failed',
        'safety_margin_px': int(safety_margin_px),
        'local_window_xyxy': [int(x0), int(y0), int(x1), int(y1)],
        'local_window_shape_hw': [int(local_safe.shape[0]), int(local_safe.shape[1])],
        'attempts': attempts,
    }


def build_safe_curve_from_navmesh_polyline(*, path_points_xz: np.ndarray, p2w: dict,
                                           navmask: np.ndarray, safety_margin_px: int = 1) -> tuple[np.ndarray, dict]:
    pts = np.asarray(path_points_xz, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return pts, {'ok': False, 'fallback_reason': 'invalid_navmesh_polyline'}
    safe_mask = safe_navmask_with_margin(navmask, margin_px=int(safety_margin_px))
    points_xy = [world_xz_to_image_xy_local(float(p[0]), float(p[1]), p2w) for p in pts]
    if not all(0 <= x < safe_mask.shape[1] and 0 <= y < safe_mask.shape[0] and bool(safe_mask[y, x]) for x, y in points_xy):
        safe_mask = np.asarray(navmask, dtype=bool)
    shortcut = shortcut_safe_path(points_xy, safe_mask)
    smooth = smooth_xy_path_safe(shortcut, safe_mask, iterations=2)
    candidate_xz = []
    for px, py in smooth:
        wx, wz = image_xy_to_world_xz_local(int(px), int(py), p2w)
        candidate_xz.append([float(wx), float(wz)])
    candidate = np.asarray(candidate_xz, dtype=np.float32)
    if candidate.shape[0] < 2:
        return pts, {
            'ok': False,
            'fallback_reason': 'smoothed_path_too_short',
            'path_planner': 'habitat_shortest_path',
        }
    candidate_check = validate_curve_points_against_navmask(draw_points_xz=candidate, p2w=p2w, navmask=navmask)
    if not bool(candidate_check.get('valid', False)):
        return pts, {
            'ok': True,
            'path_planner': 'habitat_shortest_path_unsmoothed_fallback',
            'safety_margin_px': int(safety_margin_px),
            'input_point_count': int(pts.shape[0]),
            'rejected_smooth_invalid_count': int(candidate_check.get('invalid_count', 0)),
            'rejected_smooth_first_invalid_point': candidate_check.get('first_invalid_point'),
        }
    return candidate, {
        'ok': True,
        'path_planner': 'habitat_shortest_path_safe_shortcut_smooth',
        'safety_margin_px': int(safety_margin_px),
        'input_point_count': int(pts.shape[0]),
        'shortcut_pixel_count': int(len(shortcut)),
        'smooth_pixel_count': int(len(smooth)),
    }


def fit_image(img: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(float(width) / max(1.0, float(w)), float(height) / max(1.0, float(h)))
    new_w = max(1, int(round(float(w) * scale)))
    new_h = max(1, int(round(float(h) * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    x0 = (width - new_w) // 2
    y0 = (height - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def make_storyboard(frames: list[np.ndarray], labels: list[str], cell_w: int, cell_h: int, cols: int = 4) -> np.ndarray:
    if not frames:
        return np.full((cell_h, cell_w, 3), 245, dtype=np.uint8)
    cols = max(1, int(cols))
    rows = int(math.ceil(len(frames) / float(cols)))
    gap = 14
    canvas = np.full((rows * cell_h + (rows + 1) * gap, cols * cell_w + (cols + 1) * gap, 3), 250, dtype=np.uint8)
    for idx, frame in enumerate(frames):
        row = idx // cols
        col = idx % cols
        x0 = gap + col * (cell_w + gap)
        y0 = gap + row * (cell_h + gap)
        card = fit_image(frame, cell_w, cell_h)
        label = labels[idx] if idx < len(labels) else f'frame {idx}'
        cv2.putText(card, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(card, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 1, cv2.LINE_AA)
        canvas[y0:y0 + cell_h, x0:x0 + cell_w] = card
    return canvas


def draw_text_block(img: np.ndarray, lines: list[str], x: int, y: int, gap: int = 22) -> None:
    yy = y
    for line in lines:
        cv2.putText(img, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(img, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (30, 30, 30), 1, cv2.LINE_AA)
        yy += gap


def transcode_to_h264_avc1(*, input_path: Path, output_path: Path) -> None:
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        raise RuntimeError('ffmpeg is required to write H.264/avc1 mp4 output')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        '-y',
        '-i', str(input_path),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        '-tag:v', 'avc1',
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def build_pitch_candidates(*, base_pitch_deg: float, target_focus_pitch_deg: float, bias_deg: float,
                           max_pitch_deg: float, step_deg: float) -> list[float]:
    values: list[float] = []
    max_pitch = max(0.0, float(max_pitch_deg))
    step = max(0.5, float(step_deg))

    def add(value: float) -> None:
        clipped = float(np.clip(float(value), 0.0, max_pitch))
        if any(abs(clipped - existing) <= 1e-6 for existing in values):
            return
        values.append(clipped)

    start_pitch = max(float(base_pitch_deg) + float(bias_deg), float(target_focus_pitch_deg) + float(bias_deg))
    if not math.isfinite(float(start_pitch)):
        start_pitch = float(base_pitch_deg)
    add(start_pitch)
    pitch = start_pitch + step
    while pitch <= max_pitch + 1e-6:
        add(pitch)
        pitch += step
    if not values:
        add(base_pitch_deg)
    return values


def evaluate_target_frame(clean: dict | None, width: int, height: int, *, target_bottom_safe_ratio: float,
                          target_center_v_ratio: float) -> dict:
    if clean is None:
        return {
            'target_visible': False,
            'center_v_ratio': None,
            'bottom_v_ratio': None,
            'area_ratio': 0.0,
            'acceptable': False,
            'score': 1e9,
        }

    x0, y0, x1, y1 = [int(v) for v in clean['bbox_xyxy']]
    center_v_ratio = float(clean['center_xy'][1]) / max(1.0, float(height - 1))
    bottom_v_ratio = float(max(0, y1 - 1)) / max(1.0, float(height - 1))
    area_ratio = float(clean.get('area_px', 0)) / max(1.0, float(width * height))
    acceptable = bottom_v_ratio <= float(target_bottom_safe_ratio) and center_v_ratio <= float(target_center_v_ratio)
    score = (
        max(0.0, bottom_v_ratio - float(target_bottom_safe_ratio)) * 8.0
        + max(0.0, center_v_ratio - float(target_center_v_ratio)) * 5.0
        - min(area_ratio, 0.05)
    )
    return {
        'target_visible': True,
        'center_v_ratio': float(center_v_ratio),
        'bottom_v_ratio': float(bottom_v_ratio),
        'area_ratio': float(area_ratio),
        'acceptable': bool(acceptable),
        'score': float(score),
    }


def build_curve_navmask(*, pathfinder: Any, nav_y: float, p2w: dict) -> np.ndarray:
    plan_mod = load_module(HERE / 'plan_candidates_on_expanded_topdown.py', '_curve_video_planmod')
    nav_cache = plan_mod.build_nav_cache_union(
        pathfinder=pathfinder,
        nav_y=float(nav_y),
        sample_band_m=0.3,
        height_slack_m=0.6,
    )
    return np.asarray(plan_mod.build_local_navmask(nav_cache, p2w), dtype=bool)


def world_xz_to_image_xy_local(world_x: float, world_z: float, p2w: dict) -> tuple[int, int]:
    u = (float(world_x) - float(p2w['stand_x'])) * float(p2w['right_x']) + (float(world_z) - float(p2w['stand_z'])) * float(p2w['right_z'])
    f = (float(world_x) - float(p2w['stand_x'])) * float(p2w['forward_x']) + (float(world_z) - float(p2w['stand_z'])) * float(p2w['forward_z'])
    px = (u - float(p2w['u_min'])) * float(p2w['out_width']) / max(1e-6, float(p2w['u_max']) - float(p2w['u_min'])) - 0.5
    py = (float(p2w['f_front']) - f) * float(p2w['out_height']) / max(1e-6, float(p2w['f_front']) - float(p2w['f_back'])) - 0.5
    return int(round(px)), int(round(py))


def image_xy_to_world_xz_local(px: int, py: int, p2w: dict) -> tuple[float, float]:
    u = float(p2w['u_min']) + (float(px) + 0.5) * (float(p2w['u_max']) - float(p2w['u_min'])) / float(p2w['out_width'])
    f = float(p2w['f_front']) - (float(py) + 0.5) * (float(p2w['f_front']) - float(p2w['f_back'])) / float(p2w['out_height'])
    wx = float(p2w['stand_x']) + float(p2w['right_x']) * u + float(p2w['forward_x']) * f
    wz = float(p2w['stand_z']) + float(p2w['right_z']) * u + float(p2w['forward_z']) * f
    return float(wx), float(wz)


def validate_curve_points_against_navmask(*, draw_points_xz: np.ndarray, p2w: dict, navmask: np.ndarray) -> dict:
    invalid_points: list[dict] = []
    for idx, point_xz in enumerate(np.asarray(draw_points_xz, dtype=np.float32)):
        px, py = world_xz_to_image_xy_local(float(point_xz[0]), float(point_xz[1]), p2w)
        in_nav = 0 <= int(px) < navmask.shape[1] and 0 <= int(py) < navmask.shape[0] and bool(navmask[int(py), int(px)])
        if not in_nav:
            invalid_points.append({
                'index': int(idx),
                'world_xz': [float(point_xz[0]), float(point_xz[1])],
                'topdown_xy': [int(px), int(py)],
            })
    return {
        'valid': len(invalid_points) == 0,
        'invalid_count': int(len(invalid_points)),
        'first_invalid_point': None if not invalid_points else invalid_points[0],
        'invalid_points': invalid_points[:16],
    }


def compute_eye_locked_agent_base_world_xyz(*, nominal_base_world_xyz: np.ndarray, yaw_deg: float,
                                            pitch_down_deg: float, sensor_height_m: float) -> tuple[np.ndarray, np.ndarray]:
    nominal_base = np.asarray(nominal_base_world_xyz, dtype=np.float32)
    sensor_height = max(0.0, float(sensor_height_m))
    desired_eye_world_xyz = nominal_base + np.asarray([0.0, sensor_height, 0.0], dtype=np.float32)
    pitch_rad = math.radians(float(max(0.0, pitch_down_deg)))
    forward_xz = yaw_to_forward_xz(float(yaw_deg))
    forward_offset_m = float(sensor_height) * math.sin(pitch_rad)
    vertical_offset_m = float(sensor_height) * math.cos(pitch_rad)
    rotated_sensor_offset_world = np.asarray([
        float(forward_xz[0]) * float(forward_offset_m),
        float(vertical_offset_m),
        float(forward_xz[1]) * float(forward_offset_m),
    ], dtype=np.float32)
    render_base_world_xyz = desired_eye_world_xyz - rotated_sensor_offset_world
    return render_base_world_xyz.astype(np.float32), desired_eye_world_xyz.astype(np.float32)


def render_first_person_observation_eye_locked(*, base_mod: Any, sim: Any, nominal_base_world_xyz: np.ndarray,
                                               yaw_deg: float, pitch_down_deg: float, sensor_height_m: float) -> dict:
    render_base_world_xyz, desired_eye_world_xyz = compute_eye_locked_agent_base_world_xyz(
        nominal_base_world_xyz=np.asarray(nominal_base_world_xyz, dtype=np.float32),
        yaw_deg=float(yaw_deg),
        pitch_down_deg=float(pitch_down_deg),
        sensor_height_m=float(sensor_height_m),
    )
    obs = base_mod.render_first_person_observation(
        sim,
        position_world=render_base_world_xyz,
        yaw_rad=math.radians(float(yaw_deg)),
        pitch_down_deg=float(pitch_down_deg),
    )
    obs['nominal_base_world_xyz'] = np.asarray(nominal_base_world_xyz, dtype=np.float32).copy()
    obs['render_base_world_xyz'] = render_base_world_xyz.astype(np.float32).copy()
    obs['eye_world_xyz'] = desired_eye_world_xyz.astype(np.float32).copy()
    return obs


def render_first_person_observation_with_geometry(*, args: Any, base_mod: Any, sim: Any,
                                                  camera_world_xyz: np.ndarray, yaw_deg: float,
                                                  pitch_down_deg: float) -> dict:
    return render_first_person_observation_eye_locked(
        base_mod=base_mod,
        sim=sim,
        nominal_base_world_xyz=np.asarray(camera_world_xyz, dtype=np.float32),
        yaw_deg=float(yaw_deg),
        pitch_down_deg=float(pitch_down_deg),
        sensor_height_m=float(args.sensor_height_m),
    )


def render_with_pitch_policy(*, args: Any, base_mod: Any, base_views: Any, render_fp: Any, sim: Any,
                             camera_world_xyz: np.ndarray, yaw_deg: float, base_pitch_deg: float,
                             target_focus_world_xyz: np.ndarray, target_semantic_ids: list[int],
                             op_world_xyz: np.ndarray, width: int, height: int) -> dict:
    target_focus_pitch_deg = float(base_views.compute_pitch_from_camera_eye(
        np.asarray(camera_world_xyz, dtype=np.float32),
        np.asarray(target_focus_world_xyz, dtype=np.float32),
        float(args.sensor_height_m),
    ))

    if bool(int(args.dynamic_pitch)):
        pitch_candidates = build_pitch_candidates(
            base_pitch_deg=float(base_pitch_deg),
            target_focus_pitch_deg=float(target_focus_pitch_deg),
            bias_deg=float(args.dynamic_pitch_bias_deg),
            max_pitch_deg=float(args.dynamic_pitch_max_deg),
            step_deg=float(args.dynamic_pitch_step_deg),
        )
        pitch_mode = 'dynamic'
    else:
        pitch_candidates = [float(base_pitch_deg)]
        pitch_mode = 'fixed'

    best: dict | None = None
    best_score = 1e9
    for pitch_down_deg in pitch_candidates:
        try:
            obs = render_first_person_observation_with_geometry(
                args=args,
                base_mod=base_mod,
                sim=sim,
                camera_world_xyz=np.asarray(camera_world_xyz, dtype=np.float32),
                yaw_deg=float(yaw_deg),
                pitch_down_deg=float(pitch_down_deg),
            )
        except Exception:
            continue
        rgb = np.asarray(obs['rgb'], dtype=np.uint8)
        if rgb.ndim == 3 and rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        semantic = np.asarray(obs['semantic'], dtype=np.int32)
        if semantic.ndim == 3 and semantic.shape[-1] == 1:
            semantic = semantic[..., 0]
        render_camera = obs['render_camera']
        op_proj = render_fp.project_world_to_pixel_exact(render_camera, op_world_xyz, int(width), int(height))
        clean = base_views.clean_target_component(
            semantic_obs=semantic,
            target_semantic_ids=target_semantic_ids,
            preferred_center_xy=None,
            open_k=0,
            close_k=0,
            min_area_px=1,
        )
        eval_info = evaluate_target_frame(
            clean,
            int(width),
            int(height),
            target_bottom_safe_ratio=float(args.target_bottom_safe_ratio),
            target_center_v_ratio=float(args.target_center_v_ratio),
        )
        row = {
            'obs': obs,
            'rgb': rgb,
            'semantic': semantic,
            'render_camera': render_camera,
            'op_proj': op_proj,
            'clean': clean,
            'pitch_down_deg': float(pitch_down_deg),
            'pitch_mode': str(pitch_mode),
            'target_focus_pitch_deg': float(target_focus_pitch_deg),
            'eval': eval_info,
            'camera_nominal_base_world_xyz': np.asarray(obs['nominal_base_world_xyz'], dtype=np.float32).copy(),
            'camera_render_base_world_xyz': np.asarray(obs['render_base_world_xyz'], dtype=np.float32).copy(),
            'camera_eye_world_xyz': np.asarray(obs['eye_world_xyz'], dtype=np.float32).copy(),
        }
        score = float(eval_info['score'])
        if score < best_score:
            best_score = score
            best = row
        if bool(eval_info['acceptable']):
            return row

    if best is None:
        fallback_pitch = 0.0 if not math.isfinite(float(base_pitch_deg)) else float(base_pitch_deg)
        obs = render_first_person_observation_with_geometry(
            args=args,
            base_mod=base_mod,
            sim=sim,
            camera_world_xyz=np.asarray(camera_world_xyz, dtype=np.float32),
            yaw_deg=float(yaw_deg),
            pitch_down_deg=float(fallback_pitch),
        )
        rgb = np.asarray(obs['rgb'], dtype=np.uint8)
        if rgb.ndim == 3 and rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        semantic = np.asarray(obs['semantic'], dtype=np.int32)
        if semantic.ndim == 3 and semantic.shape[-1] == 1:
            semantic = semantic[..., 0]
        render_camera = obs['render_camera']
        op_proj = render_fp.project_world_to_pixel_exact(render_camera, op_world_xyz, int(width), int(height))
        clean = base_views.clean_target_component(
            semantic_obs=semantic,
            target_semantic_ids=target_semantic_ids,
            preferred_center_xy=None,
            open_k=0,
            close_k=0,
            min_area_px=1,
        )
        return {
            'obs': obs,
            'rgb': rgb,
            'semantic': semantic,
            'render_camera': render_camera,
            'op_proj': op_proj,
            'clean': clean,
            'pitch_down_deg': float(fallback_pitch),
            'pitch_mode': 'fixed_fallback',
            'target_focus_pitch_deg': float(target_focus_pitch_deg) if math.isfinite(float(target_focus_pitch_deg)) else None,
            'camera_nominal_base_world_xyz': np.asarray(obs['nominal_base_world_xyz'], dtype=np.float32).copy(),
            'camera_render_base_world_xyz': np.asarray(obs['render_base_world_xyz'], dtype=np.float32).copy(),
            'camera_eye_world_xyz': np.asarray(obs['eye_world_xyz'], dtype=np.float32).copy(),
            'eval': evaluate_target_frame(
                clean,
                int(width),
                int(height),
                target_bottom_safe_ratio=float(args.target_bottom_safe_ratio),
                target_center_v_ratio=float(args.target_center_v_ratio),
            ),
        }
    return best


def build_labeled_frame(*, rgb: np.ndarray, clean: dict | None, op_proj: dict, lines: list[str]) -> np.ndarray:
    overlay = np.asarray(rgb, dtype=np.uint8).copy()
    op_visible = bool(op_proj.get('valid', False)) and bool(op_proj.get('visible_in_image', False))
    if op_visible:
        cv2.drawMarker(
            overlay,
            (int(op_proj['u_draw']), int(op_proj['v_draw'])),
            (0, 255, 0),
            cv2.MARKER_CROSS,
            22,
            2,
            cv2.LINE_AA,
        )
        cv2.circle(overlay, (int(op_proj['u_draw']), int(op_proj['v_draw'])), 7, (0, 255, 0), 2, cv2.LINE_AA)
    if clean is not None:
        x0, y0, x1, y1 = [int(v) for v in clean['bbox_xyxy']]
        cx, cy = [int(v) for v in clean['center_xy']]
        cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (255, 220, 0), 2, cv2.LINE_AA)
        cv2.drawMarker(overlay, (cx, cy), (255, 220, 0), cv2.MARKER_TILTED_CROSS, 18, 2, cv2.LINE_AA)
    if lines:
        draw_text_block(overlay, lines, 18, 30)
    return overlay


def render_start_facing_debug_frame(*, args: Any, base_mod: Any, base_views: Any, render_fp: Any, sim: Any,
                                    start_world_xyz: np.ndarray, heading_focus_world_xyz: np.ndarray,
                                    target_focus_world_xyz: np.ndarray, target_semantic_ids: list[int],
                                    op_world_xyz: np.ndarray, pitch_down_deg: float,
                                    width: int, height: int) -> dict:
    start_facing_yaw = float(render_fp.yaw_between(start_world_xyz, heading_focus_world_xyz))
    frame_render = render_with_pitch_policy(
        args=args,
        base_mod=base_mod,
        base_views=base_views,
        render_fp=render_fp,
        sim=sim,
        camera_world_xyz=np.asarray(start_world_xyz, dtype=np.float32),
        yaw_deg=float(start_facing_yaw),
        base_pitch_deg=float(pitch_down_deg),
        target_focus_world_xyz=np.asarray(target_focus_world_xyz, dtype=np.float32),
        target_semantic_ids=target_semantic_ids,
        op_world_xyz=np.asarray(op_world_xyz, dtype=np.float32),
        width=int(width),
        height=int(height),
    )
    target_eval = dict(frame_render['eval'])
    op_proj = frame_render['op_proj']
    op_visible = bool(op_proj.get('valid', False)) and bool(op_proj.get('visible_in_image', False))
    border_margin = max(0, int(args.start_facing_border_margin_px))
    clean = frame_render['clean']
    target_not_clipped = False
    target_bbox_xyxy = None
    if clean is not None:
        x0, y0, x1, y1 = [int(v) for v in clean['bbox_xyxy']]
        target_bbox_xyxy = [int(x0), int(y0), int(x1), int(y1)]
        target_not_clipped = (
            int(x0) >= border_margin
            and int(y0) >= border_margin
            and int(x1) <= int(width) - border_margin
            and int(y1) <= int(height) - border_margin
        )
    acceptable = bool(target_eval.get('target_visible', False)) and bool(target_not_clipped) and bool(op_visible)
    labeled = build_labeled_frame(
        rgb=frame_render['rgb'],
        clean=clean,
        op_proj=op_proj,
        lines=[
            'start facing target',
            f'yaw={start_facing_yaw:.1f}',
            f'pitch={float(frame_render["pitch_down_deg"]):.1f}',
            f'target_visible={1 if target_eval.get("target_visible", False) else 0}',
            f'target_not_clipped={1 if target_not_clipped else 0}',
            f'op_visible={1 if op_visible else 0}',
        ],
    )
    clean_rgb = np.asarray(frame_render['rgb'], dtype=np.uint8).copy()
    return {
        'yaw_deg': float(start_facing_yaw),
        'pitch_down_deg': float(frame_render['pitch_down_deg']),
        'target_eval': target_eval,
        'target_bbox_xyxy': target_bbox_xyxy,
        'target_not_clipped': bool(target_not_clipped),
        'op_visible': bool(op_visible),
        'acceptable': bool(acceptable),
        'labeled_rgb': labeled,
        'clean_rgb': clean_rgb,
        'op_rgb_xy': [int(op_proj['u_draw']), int(op_proj['v_draw'])] if op_visible else None,
    }


def main() -> None:
    args = build_parser().parse_args()
    stage3_root = args.stage3_root.expanduser().resolve()
    accepted_file = args.accepted_file.expanduser().resolve()
    if not args.target_uid:
        ids = load_ids(accepted_file)
        if not ids:
            raise RuntimeError('accepted file is empty')
        target_uid = ids[0]
    else:
        target_uid = str(args.target_uid)

    out_root = args.out_root.expanduser().resolve() if args.out_root is not None else Path(
        f'outputs/observer_to_operating_curve_video_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    target_dir = out_root / target_uid.split('__', 1)[0] / target_uid
    run_tag = str(args.run_tag).strip()
    if run_tag:
        target_dir = target_dir / run_tag
    target_dir.mkdir(parents=True, exist_ok=True)

    base_views = load_module(args.framework_final_dir / 'generate_accepted_base_views.py', '_curve_video_baseviews')
    render_fp = load_module(args.framework_final_dir / 'render_fp_from_observer_candidates.py', '_curve_video_renderfp')
    base_mod = base_views.load_module(args.scripts_dir / 'render_local_topdown.py', '_curve_video_scene')

    manifest = load_json(stage3_root / 'final_observations' / f'{target_uid}__manifest.json')
    plan = load_json(stage3_root / 'plan' / target_uid / 'result.json')
    expanded_meta = load_json(Path(plan['expanded_sample_dir']) / 'meta.json')
    stage_meta = load_json(Path(expanded_meta['source_meta_json']))
    traj = load_json(Path(stage_meta['trajectory_json']))
    target_row = stage_meta['targets'][int(plan['target_index'])]

    accepted_rgb = cv2.imread(str(Path(manifest['final_observation_with_points_png'])), cv2.IMREAD_COLOR)
    if accepted_rgb is None:
        raise FileNotFoundError(manifest['final_observation_with_points_png'])
    width = int(accepted_rgb.shape[1])
    height = int(accepted_rgb.shape[0])
    if int(args.force_render_width) > 0:
        width = int(args.force_render_width)
    if int(args.force_render_height) > 0:
        height = int(args.force_render_height)
    nav_y = float(stage_meta['nav_y'])

    observer_xz = np.asarray(manifest['observer_point_used']['observer_world_xz'], dtype=np.float32)
    observer_override_xz = parse_xz_pair(str(args.observer_world_xz))
    if observer_override_xz is not None:
        observer_xz = observer_override_xz.astype(np.float32)
    operating_xz = np.asarray(plan['operating_point_world_xz'], dtype=np.float32)
    start_yaw_deg = float(manifest['fp_yaw_deg'])
    pitch_down_deg = float(manifest['fp_pitch_down_deg'])
    if args.start_yaw_deg is not None:
        start_yaw_deg = float(args.start_yaw_deg)
    if args.start_pitch_deg is not None:
        pitch_down_deg = float(args.start_pitch_deg)
    hfov_deg = float(manifest['fp_hfov_deg'])
    vfov_deg = float(manifest['fp_vfov_deg'])
    if float(args.override_hfov_deg) > 0.0:
        hfov_deg = float(args.override_hfov_deg)
    if float(args.override_pitch_deg) > 0.0:
        pitch_down_deg = float(args.override_pitch_deg)
    if float(args.override_vfov_deg) > 0.0:
        vfov_deg = float(args.override_vfov_deg)
        hfov_deg = float(render_fp.vfov_to_hfov_deg(float(vfov_deg), int(width), int(height)))
    target_semantic_ids = [int(v) for v in manifest.get('target_semantic_ids', []) if int(v) > 0]

    target_center_world_xz = np.asarray(target_row.get('topdown_center_world_xz', operating_xz.tolist()), dtype=np.float32)
    target_center_world_xyz = np.asarray([
        float(target_center_world_xz[0]),
        float(nav_y),
        float(target_center_world_xz[1]),
    ], dtype=np.float32)
    target_focus_world_xyz = render_fp.resolve_target_focus_world_xyz(target_row, target_center_world_xz, nav_y)
    target_rect_world = topdown_bbox_world_rect(target_row, stage_meta['topdown']['pixel_to_world_xz'])
    heading_focus_world_xyz = base_views.compute_heading_from_operating_edge(
        annotation_target=plan.get('annotation_target', {}),
        p2w=stage_meta['topdown']['pixel_to_world_xz'],
        target_center_world_xz=target_center_world_xz,
        nav_y=nav_y,
    )
    if heading_focus_world_xyz is None:
        heading_focus_world_xyz = target_focus_world_xyz.copy()
    heading_focus_xz = np.asarray([float(heading_focus_world_xyz[0]), float(heading_focus_world_xyz[2])], dtype=np.float32)
    endpoint_xz = operating_xz.copy()
    endpoint_raw_xz = operating_xz.copy()
    endpoint_search_info = None
    scene_source = str(traj.get('scene_path', '')).strip() or str(traj['scene_id'])
    sim = base_views.build_sim_for_target(
        base_mod=base_mod,
        traj=traj,
        scene_source=scene_source,
        width=int(width),
        height=int(height),
        sensor_height_m=float(args.sensor_height_m),
        hfov_deg=float(hfov_deg),
    )
    topdown_rgb = cv2.cvtColor(cv2.imread(str(Path(plan['expanded_sample_dir']) / 'local_topdown_rgb.png')), cv2.COLOR_BGR2RGB)
    accepted_overlay_bgr = cv2.imread(str(Path(manifest['final_observation_with_points_png'])), cv2.IMREAD_COLOR)
    if accepted_overlay_bgr is None:
        accepted_overlay = accepted_rgb.copy()
    else:
        accepted_overlay = cv2.cvtColor(accepted_overlay_bgr, cv2.COLOR_BGR2RGB)
    p2w = expanded_meta['topdown']['pixel_to_world_xz']
    op_world_xyz = None
    obs_world_xyz = None
    path_points_xz = None
    frame_rows = []
    storyboard_frames = []
    storyboard_labels = []
    fp_video_path = target_dir / 'observer_to_operating_fp.mp4'
    debug_video_path = target_dir / 'observer_to_operating_debug.mp4'
    fp_video_temp_path = target_dir / 'observer_to_operating_fp_tmp_mp4v.mp4'
    debug_video_temp_path = target_dir / 'observer_to_operating_debug_tmp_mp4v.mp4'
    topdown_path_png = target_dir / 'planned_curve_topdown.png'
    storyboard_path_png = target_dir / 'observer_to_operating_storyboard.png'
    final_fp_frame_path = target_dir / 'final_fp_frame.png'
    final_debug_frame_path = target_dir / 'final_debug_frame.png'
    start_facing_debug_frame_path = target_dir / 'start_facing_debug_frame.png'
    start_facing_clean_frame_path = target_dir / 'start_facing_clean_frame.png'
    fourcc = cv2.VideoWriter_fourcc(*TEMP_VIDEO_FOURCC)
    fp_writer = None
    debug_writer = None
    last_clean_frame = None
    last_debug_panel = None
    start_facing_info = None
    curve_navmask = None
    curve_navcheck = {'valid': True, 'invalid_count': 0, 'first_invalid_point': None, 'invalid_points': []}
    path_planning_info = {'path_shape': str(args.path_shape)}
    endpoint_strategy = 'operating_point'
    endpoint_candidate_count = 0
    try:
        pf = sim.pathfinder
        obs_world_xyz = render_fp.precise_snap(pf, float(observer_xz[0]), float(observer_xz[1]), nav_y)
        op_world_xyz = render_fp.precise_snap(pf, float(operating_xz[0]), float(operating_xz[1]), nav_y)
        endpoint_world_xyz = None
        if obs_world_xyz is None or op_world_xyz is None:
            raise RuntimeError('failed to snap observer or operating point')

        start_facing_info = render_start_facing_debug_frame(
            args=args,
            base_mod=base_mod,
            base_views=base_views,
            render_fp=render_fp,
            sim=sim,
            start_world_xyz=np.asarray(obs_world_xyz, dtype=np.float32),
            heading_focus_world_xyz=np.asarray(heading_focus_world_xyz, dtype=np.float32),
            target_focus_world_xyz=np.asarray(target_focus_world_xyz, dtype=np.float32),
            target_semantic_ids=target_semantic_ids,
            op_world_xyz=np.asarray(op_world_xyz, dtype=np.float32),
            pitch_down_deg=float(pitch_down_deg),
            width=int(width),
            height=int(height),
        )
        cv2.imwrite(
            str(start_facing_debug_frame_path),
            cv2.cvtColor(np.asarray(start_facing_info['labeled_rgb'], dtype=np.uint8), cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            str(start_facing_clean_frame_path),
            cv2.cvtColor(np.asarray(start_facing_info['clean_rgb'], dtype=np.uint8), cv2.COLOR_RGB2BGR),
        )
        if bool(int(args.require_start_facing_visible)) and not bool(start_facing_info.get('acceptable', False)):
            raise RuntimeError('start_facing_target_or_operating_point_not_visible')

        if endpoint_world_xyz is None and float(args.endpoint_offset_m) > 0.0:
            endpoint_strategy = 'offset_from_operating_edge'
            search = find_offset_endpoint_on_navmesh(
                pathfinder=pf,
                render_fp=render_fp,
                operating_xz=operating_xz,
                heading_focus_xz=heading_focus_xz,
                nav_y=nav_y,
                desired_offset_m=float(args.endpoint_offset_m),
                target_rect_world=target_rect_world,
            )
            if search is not None:
                endpoint_raw_xz, endpoint_xz, endpoint_search_info = search
                endpoint_world_xyz = render_fp.precise_snap(pf, float(endpoint_xz[0]), float(endpoint_xz[1]), nav_y)
        if endpoint_world_xyz is None:
            endpoint_strategy = 'operating_point'
            endpoint_world_xyz = render_fp.precise_snap(pf, float(endpoint_xz[0]), float(endpoint_xz[1]), nav_y)
        if endpoint_world_xyz is None:
            raise RuntimeError('failed to snap endpoint')

        if str(args.path_shape) == 'twist_replay':
            if args.replay_actions_npy is None:
                raise ValueError('--replay-actions-npy is required for --path-shape twist_replay')
            start_xz_for_replay = np.asarray([float(obs_world_xyz[0]), float(obs_world_xyz[2])], dtype=np.float32)
            path_points_xz, samples, path_planning_info = build_twist_replay_samples(
                start_xz=start_xz_for_replay,
                start_yaw_deg=float(start_yaw_deg),
                actions_npy=args.replay_actions_npy.expanduser().resolve(),
                fps=float(args.fps),
            )
        elif str(args.path_shape) == 'safe_nav_curve':
            curve_navmask = build_curve_navmask(pathfinder=pf, nav_y=nav_y, p2w=p2w)
            start_xz_for_plan = np.asarray([float(obs_world_xyz[0]), float(obs_world_xyz[2])], dtype=np.float32)
            endpoint_xz_for_plan = np.asarray([float(endpoint_world_xyz[0]), float(endpoint_world_xyz[2])], dtype=np.float32)
            path_points_xz, path_planning_info = build_safe_nav_curve_path_xz_coarse(
                start_xz=start_xz_for_plan,
                goal_xz=endpoint_xz_for_plan,
                p2w=p2w,
                navmask=curve_navmask,
                safety_margin_px=1,
                coarse_scale=4,
            )
            if not bool(path_planning_info.get('ok', False)):
                raise RuntimeError(f"safe_nav_curve_failed:{path_planning_info.get('fallback_reason')}")
            samples = sample_polyline_path(path_points_xz, int(args.frame_count))
        elif str(args.path_shape) == 'structured_approach':
            curve_navmask = build_curve_navmask(pathfinder=pf, nav_y=nav_y, p2w=p2w)
            start_xz_for_plan = np.asarray([float(obs_world_xyz[0]), float(obs_world_xyz[2])], dtype=np.float32)
            op_xz_for_plan = np.asarray([float(endpoint_world_xyz[0]), float(endpoint_world_xyz[2])], dtype=np.float32)
            terminal_focus_xz_for_plan = np.asarray([float(heading_focus_world_xyz[0]), float(heading_focus_world_xyz[2])], dtype=np.float32)
            preop = find_terminal_preop_xz(
                op_xz=op_xz_for_plan,
                target_focus_xz=terminal_focus_xz_for_plan,
                navmask=curve_navmask,
                p2w=p2w,
                desired_dist_m=float(args.terminal_approach_dist_m),
                min_dist_m=float(args.terminal_approach_min_dist_m),
            )
            if preop is None:
                raise RuntimeError('structured_approach_failed:terminal_preop_not_navsafe')
            preop_xz, terminal_info = preop
            path_points_xz, samples, path_planning_info = sample_structured_approach_path(
                start_xz=start_xz_for_plan,
                op_xz=op_xz_for_plan,
                preop_xz=preop_xz,
                target_focus_xz=terminal_focus_xz_for_plan,
                semantic_focus_xz=target_center_world_xz,
                start_yaw_deg=float(start_yaw_deg),
                frame_count=int(args.frame_count),
                align_start_ratio=float(args.align_start_ratio),
                align_terminal_ratio=float(args.align_terminal_ratio),
                final_approach_ratio=float(args.final_approach_ratio),
                p2w=p2w,
                navmask=curve_navmask,
            )
            path_planning_info.update(terminal_info)
        elif str(args.path_shape) == 'controller_approach':
            curve_navmask = build_curve_navmask(pathfinder=pf, nav_y=nav_y, p2w=p2w)
            start_xz_for_plan = np.asarray([float(obs_world_xyz[0]), float(obs_world_xyz[2])], dtype=np.float32)
            op_xz_for_plan = np.asarray([float(endpoint_world_xyz[0]), float(endpoint_world_xyz[2])], dtype=np.float32)
            terminal_focus_xz_for_plan = np.asarray([float(heading_focus_world_xyz[0]), float(heading_focus_world_xyz[2])], dtype=np.float32)
            preop = find_terminal_preop_xz(
                op_xz=op_xz_for_plan,
                target_focus_xz=terminal_focus_xz_for_plan,
                navmask=curve_navmask,
                p2w=p2w,
                desired_dist_m=float(args.terminal_approach_dist_m),
                min_dist_m=float(args.terminal_approach_min_dist_m),
            )
            if preop is None:
                raise RuntimeError('controller_approach_failed:terminal_preop_not_navsafe')
            preop_xz, terminal_info = preop
            direct_route_xz = np.asarray([start_xz_for_plan, preop_xz], dtype=np.float32)
            if path_xz_is_navsafe(direct_route_xz, p2w=p2w, navmask=curve_navmask, step_m=0.02):
                route_points_xz = direct_route_xz
                route_info = {
                    'ok': True,
                    'path_planner': 'direct_line_navsafe',
                    'direct_line_used': True,
                    'direct_line_to_preop_m': float(np.linalg.norm(preop_xz - start_xz_for_plan)),
                }
            else:
                route_points_xz, route_info = build_safe_nav_curve_path_xz_coarse(
                    start_xz=start_xz_for_plan,
                    goal_xz=preop_xz,
                    p2w=p2w,
                    navmask=curve_navmask,
                    safety_margin_px=1,
                    coarse_scale=4,
                )
                route_info['direct_line_used'] = False
                if not bool(route_info.get('ok', False)):
                    route_points_xz = direct_route_xz
            route_points_xz, terminal_transition_info = build_terminal_bezier_transition_xz(
                route_points_xz=route_points_xz,
                preop_xz=preop_xz,
                terminal_dir=normalize_xz(terminal_focus_xz_for_plan - op_xz_for_plan),
                p2w=p2w,
                navmask=curve_navmask,
            )
            full_path_points_xz = dedupe_path_points_xz(np.vstack([
                route_points_xz,
                sample_segment_xz(preop_xz, op_xz_for_plan, step_m=0.02)[1:],
            ]), min_step_m=0.01)
            path_points_xz, samples, path_planning_info = sample_controller_follow_path(
                path_points_xz=full_path_points_xz,
                start_yaw_deg=float(start_yaw_deg),
                frame_count=int(args.frame_count),
                fps=int(args.fps),
                final_yaw_deg=float(render_fp.yaw_between(op_world_xyz, heading_focus_world_xyz)),
                p2w=p2w,
                navmask=curve_navmask,
                forward_speed_mps=float(args.controller_forward_speed_mps),
                final_speed_mps=float(args.controller_final_speed_mps),
                max_yaw_rate_degps=float(args.controller_max_yaw_rate_degps),
                yaw_gain=float(args.controller_yaw_gain),
                final_segment_m=float(terminal_info.get('terminal_approach_dist_m', 0.08)),
                curve_lookahead_m=float(args.controller_curve_lookahead_m),
            )
            path_planning_info.update({
                'route_info': route_info,
                'terminal_transition_info': terminal_transition_info,
            })
            path_planning_info.update(terminal_info)
        elif str(args.path_shape) == 'staged_approach':
            curve_navmask = build_curve_navmask(pathfinder=pf, nav_y=nav_y, p2w=p2w)
            start_xz_for_plan = np.asarray([float(obs_world_xyz[0]), float(obs_world_xyz[2])], dtype=np.float32)
            op_xz_for_plan = np.asarray([float(endpoint_world_xyz[0]), float(endpoint_world_xyz[2])], dtype=np.float32)
            terminal_focus_xz_for_plan = np.asarray([float(heading_focus_world_xyz[0]), float(heading_focus_world_xyz[2])], dtype=np.float32)
            preop = find_terminal_preop_xz(
                op_xz=op_xz_for_plan,
                target_focus_xz=terminal_focus_xz_for_plan,
                navmask=curve_navmask,
                p2w=p2w,
                desired_dist_m=0.30,
                min_dist_m=0.20,
            )
            if preop is None:
                preop = find_terminal_preop_xz(
                    op_xz=op_xz_for_plan,
                    target_focus_xz=terminal_focus_xz_for_plan,
                    navmask=curve_navmask,
                    p2w=p2w,
                    desired_dist_m=0.20,
                    min_dist_m=0.08,
                )
            if preop is None:
                raise RuntimeError('staged_approach_failed:terminal_preop_not_navsafe')
            preop_xz, terminal_info = preop
            direct_route_xz = np.asarray([start_xz_for_plan, preop_xz], dtype=np.float32)
            if path_xz_is_navsafe(direct_route_xz, p2w=p2w, navmask=curve_navmask, step_m=0.02):
                route_points_xz = direct_route_xz
                route_info = {
                    'ok': True,
                    'path_planner': 'direct_line_navsafe',
                    'direct_line_used': True,
                    'direct_line_to_preop_m': float(np.linalg.norm(preop_xz - start_xz_for_plan)),
                }
            else:
                route_points_xz, route_info = build_safe_nav_curve_path_xz_coarse(
                    start_xz=start_xz_for_plan,
                    goal_xz=preop_xz,
                    p2w=p2w,
                    navmask=curve_navmask,
                    safety_margin_px=1,
                    coarse_scale=4,
                )
                route_info['direct_line_used'] = False
                if not bool(route_info.get('ok', False)):
                    route_points_xz = direct_route_xz
            final_segment = sample_segment_xz(preop_xz, op_xz_for_plan, step_m=0.02)
            full_route_xz = dedupe_path_points_xz(np.vstack([route_points_xz, final_segment[1:]]), min_step_m=0.01)
            path_points_xz, samples, path_planning_info = sample_staged_approach_path(
                route_points_xz=full_route_xz,
                op_xz=op_xz_for_plan,
                semantic_focus_xz=target_center_world_xz,
                terminal_focus_xz=terminal_focus_xz_for_plan,
                start_yaw_deg=float(start_yaw_deg),
                frame_count=int(args.frame_count),
                final_turn_dist_m=float(terminal_info.get('terminal_approach_dist_m', 0.20)),
                fps=float(args.fps),
                target_speed_mps=float(args.staged_target_speed_mps),
                final_speed_mps=float(args.staged_final_speed_mps),
                max_yaw_rate_degps=float(args.staged_max_yaw_rate_degps),
                auto_frame_count=bool(int(args.staged_auto_frame_count)),
            )
            path_planning_info.update({
                'route_info': route_info,
                'preop_world_xz': [float(v) for v in preop_xz.tolist()],
            })
            path_planning_info.update(terminal_info)
        elif str(args.path_shape) == 'normal_close_approach':
            curve_navmask = build_curve_navmask(pathfinder=pf, nav_y=nav_y, p2w=p2w)
            start_xz_for_plan = np.asarray([float(obs_world_xyz[0]), float(obs_world_xyz[2])], dtype=np.float32)
            op_xz_for_plan = np.asarray([float(endpoint_world_xyz[0]), float(endpoint_world_xyz[2])], dtype=np.float32)
            terminal_focus_xz_for_plan = np.asarray([float(heading_focus_world_xyz[0]), float(heading_focus_world_xyz[2])], dtype=np.float32)
            path_points_xz, samples, path_planning_info = sample_normal_close_approach_path(
                start_xz=start_xz_for_plan,
                op_xz=op_xz_for_plan,
                semantic_focus_xz=target_center_world_xz,
                terminal_focus_xz=terminal_focus_xz_for_plan,
                start_yaw_deg=float(start_yaw_deg),
                frame_count=int(args.frame_count),
                fps=float(args.fps),
                target_speed_mps=float(args.staged_target_speed_mps),
                final_speed_mps=float(args.staged_final_speed_mps),
                max_yaw_rate_degps=float(args.staged_max_yaw_rate_degps),
                p2w=p2w,
                navmask=curve_navmask,
            )
        else:
            path_points_xz, samples = build_motion_samples(
                pathfinder=pf,
                path_shape=str(args.path_shape),
                obs_world_xyz=np.asarray(obs_world_xyz, dtype=np.float32),
                endpoint_world_xyz=np.asarray(endpoint_world_xyz, dtype=np.float32),
                start_yaw_deg=float(start_yaw_deg),
                heading_focus_world_xyz=np.asarray(heading_focus_world_xyz, dtype=np.float32),
                frame_count=int(args.frame_count),
                control_scale=float(args.control_scale),
            )

        poly_pts = []
        nav_checked_shapes = {'curve', 'safe_nav_curve', 'structured_approach', 'normal_close_approach', 'twist_replay'}
        draw_path_shapes = {'shortest_path', 'safe_nav_curve', 'structured_approach', 'normal_close_approach', 'twist_replay'}
        draw_points = path_points_xz if str(args.path_shape) in draw_path_shapes else np.asarray([point for point, _, _ in samples], dtype=np.float32)
        if str(args.path_shape) in nav_checked_shapes and curve_navmask is None:
            curve_navmask = build_curve_navmask(pathfinder=pf, nav_y=nav_y, p2w=p2w)
        if str(args.path_shape) in nav_checked_shapes:
            curve_navcheck = validate_curve_points_against_navmask(draw_points_xz=draw_points, p2w=p2w, navmask=curve_navmask)
        for point_xz in draw_points:
            px, py = render_fp.world_xz_to_image_xy(float(point_xz[0]), float(point_xz[1]), p2w)
            poly_pts.append([int(px), int(py)])
        path_vis = topdown_rgb.copy()
        if len(poly_pts) >= 2:
            cv2.polylines(path_vis, [np.asarray(poly_pts, dtype=np.int32)], False, (255, 255, 255), 2, cv2.LINE_AA)
        if not bool(curve_navcheck.get('valid', True)):
            for row in curve_navcheck.get('invalid_points', []):
                px, py = [int(v) for v in row['topdown_xy']]
                cv2.circle(path_vis, (px, py), 4, (0, 0, 255), -1, cv2.LINE_AA)
        start_xy = render_fp.world_xz_to_image_xy(float(observer_xz[0]), float(observer_xz[1]), p2w)
        operating_xy = render_fp.world_xz_to_image_xy(float(operating_xz[0]), float(operating_xz[1]), p2w)
        end_xy = render_fp.world_xz_to_image_xy(float(endpoint_xz[0]), float(endpoint_xz[1]), p2w)
        raw_end_xy = render_fp.world_xz_to_image_xy(float(endpoint_raw_xz[0]), float(endpoint_raw_xz[1]), p2w)
        cv2.circle(path_vis, start_xy, 8, (255, 80, 80), 2, cv2.LINE_AA)
        cv2.circle(path_vis, operating_xy, 8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(path_vis, raw_end_xy, 8, (255, 120, 255), 2, cv2.LINE_AA)
        cv2.circle(path_vis, end_xy, 8, (255, 0, 255), 2, cv2.LINE_AA)
        heading_xy = render_fp.world_xz_to_image_xy(float(heading_focus_world_xyz[0]), float(heading_focus_world_xyz[2]), p2w)
        cv2.arrowedLine(path_vis, end_xy, heading_xy, (255, 220, 0), 2, cv2.LINE_AA, tipLength=0.08)
        cv2.imwrite(str(topdown_path_png), cv2.cvtColor(path_vis, cv2.COLOR_RGB2BGR))

        if str(args.path_shape) in nav_checked_shapes and bool(int(args.require_curve_navmask)) and not bool(curve_navcheck.get('valid', True)):
            raise RuntimeError('curve_path_leaves_navmask')

        fp_writer = cv2.VideoWriter(str(fp_video_temp_path), fourcc, float(args.fps), (int(width), int(height)))
        if bool(int(args.topdown_debug)):
            debug_writer = cv2.VideoWriter(str(debug_video_temp_path), fourcc, float(args.fps), (int(height + 8 + width + 8 + width), int(height)))

        stop_reason = 'completed'
        for frame_idx, (raw_xz, tangent_xz, t_value) in enumerate(samples):
            cam_world_xyz = render_fp.precise_snap(pf, float(raw_xz[0]), float(raw_xz[1]), nav_y)
            if cam_world_xyz is None:
                stop_reason = 'snap_failed'
                break
            yaw_deg = float(direction_to_yaw_deg(tangent_xz))
            yaw_override = path_planning_info.get('yaw_override_deg') if isinstance(path_planning_info, dict) else None
            if isinstance(yaw_override, list) and frame_idx < len(yaw_override):
                yaw_deg = float(yaw_override[frame_idx])
            if frame_idx == len(samples) - 1 and str(args.path_shape) not in {'safe_nav_curve', 'structured_approach', 'normal_close_approach'}:
                yaw_deg = float(render_fp.yaw_between(cam_world_xyz, heading_focus_world_xyz))

            frame_render = render_with_pitch_policy(
                args=args,
                base_mod=base_mod,
                base_views=base_views,
                render_fp=render_fp,
                sim=sim,
                camera_world_xyz=np.asarray(cam_world_xyz, dtype=np.float32),
                yaw_deg=float(yaw_deg),
                base_pitch_deg=float(pitch_down_deg),
                target_focus_world_xyz=np.asarray(target_focus_world_xyz, dtype=np.float32),
                target_semantic_ids=target_semantic_ids,
                op_world_xyz=np.asarray(op_world_xyz, dtype=np.float32),
                width=int(width),
                height=int(height),
            )
            rgb = frame_render['rgb']
            semantic = frame_render['semantic']
            render_camera = frame_render['render_camera']
            op_proj = frame_render['op_proj']
            frame_pitch_deg = float(frame_render['pitch_down_deg'])
            frame_pitch_mode = str(frame_render['pitch_mode'])
            target_eval = dict(frame_render['eval'])
            frame_render_base_world_xyz = np.asarray(frame_render['camera_render_base_world_xyz'], dtype=np.float32)
            frame_eye_world_xyz = np.asarray(frame_render['camera_eye_world_xyz'], dtype=np.float32)
            op_visible = bool(op_proj.get('valid', False)) and bool(op_proj.get('visible_in_image', False))
            if bool(int(args.stop_on_op_invisible)) and not op_visible:
                stop_reason = 'operating_point_not_visible'
                break

            clean = frame_render['clean']
            overlay = build_labeled_frame(
                rgb=rgb,
                clean=clean,
                op_proj=op_proj,
                lines=[
                    f'frame={frame_idx:03d}',
                    f't={t_value:.3f}',
                    f'yaw={yaw_deg:.1f}',
                    f'pitch={frame_pitch_deg:.1f}',
                    f'vfov={vfov_deg:.1f}',
                    f'pitch_mode={frame_pitch_mode}',
                    f'target_visible={1 if target_eval["target_visible"] else 0}',
                    f'op_visible={1 if op_visible else 0}',
                ],
            )

            fp_writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            last_clean_frame = rgb.copy()

            keep_story = False
            if int(args.storyboard_frames) > 0:
                max_story = max(1, int(args.storyboard_frames))
                if len(storyboard_frames) < max_story:
                    keep_story = True
                else:
                    target_slots = np.linspace(0, max(0, len(samples) - 1), max_story)
                    nearest = min(float(abs(frame_idx - slot)) for slot in target_slots)
                    keep_story = nearest < 0.5
            if keep_story:
                storyboard_frames.append(overlay.copy())
                storyboard_labels.append(f'f={frame_idx}  t={t_value:.2f}')

            debug_frame_path = None
            if debug_writer is not None:
                topdown_frame = path_vis.copy()
                cur_xy = render_fp.world_xz_to_image_xy(float(cam_world_xyz[0]), float(cam_world_xyz[2]), p2w)
                cv2.circle(topdown_frame, cur_xy, 7, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.arrowedLine(topdown_frame, cur_xy, render_fp.world_xz_to_image_xy(float(cam_world_xyz[0] + tangent_xz[0]), float(cam_world_xyz[2] + tangent_xz[1]), p2w), (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.18)
                left = fit_image(topdown_frame, int(height), int(height))
                mid = fit_image(accepted_overlay, int(width), int(height))
                panel = np.zeros((int(height), int(height + 8 + width + 8 + width), 3), dtype=np.uint8)
                panel[:, :int(height)] = left
                panel[:, int(height) + 8:int(height) + 8 + int(width)] = mid
                panel[:, int(height) + 8 + int(width) + 8:] = overlay
                cv2.putText(panel, 'topdown curve', (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(panel, 'topdown curve', (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (30, 30, 30), 1, cv2.LINE_AA)
                cv2.putText(panel, 'accepted observation', (int(height) + 26, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(panel, 'accepted observation', (int(height) + 26, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (30, 30, 30), 1, cv2.LINE_AA)
                cv2.putText(panel, 'curve motion view', (int(height) + 8 + int(width) + 26, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(panel, 'curve motion view', (int(height) + 8 + int(width) + 26, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (30, 30, 30), 1, cv2.LINE_AA)
                debug_writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
                last_debug_panel = panel.copy()

            frame_rows.append({
                'frame_index': int(frame_idx),
                'curve_t': float(t_value),
                'camera_world_xyz': [float(v) for v in cam_world_xyz.tolist()],
                'camera_world_xz': [float(cam_world_xyz[0]), float(cam_world_xyz[2])],
                'camera_nominal_base_world_xyz': [float(v) for v in cam_world_xyz.tolist()],
                'camera_nominal_base_world_xz': [float(cam_world_xyz[0]), float(cam_world_xyz[2])],
                'camera_render_base_world_xyz': [float(v) for v in frame_render_base_world_xyz.tolist()],
                'camera_render_base_world_xz': [float(frame_render_base_world_xyz[0]), float(frame_render_base_world_xyz[2])],
                'camera_eye_world_xyz': [float(v) for v in frame_eye_world_xyz.tolist()],
                'camera_eye_world_xz': [float(frame_eye_world_xyz[0]), float(frame_eye_world_xyz[2])],
                'yaw_deg': float(yaw_deg),
                'pitch_down_deg': float(frame_pitch_deg),
                'pitch_mode': str(frame_pitch_mode),
                'target_visible': bool(target_eval['target_visible']),
                'target_center_v_ratio': None if target_eval['center_v_ratio'] is None else float(target_eval['center_v_ratio']),
                'target_bottom_v_ratio': None if target_eval['bottom_v_ratio'] is None else float(target_eval['bottom_v_ratio']),
                'target_area_ratio': float(target_eval['area_ratio']),
                'operating_point_visible': bool(op_visible),
                'operating_rgb_xy': [int(op_proj['u_draw']), int(op_proj['v_draw'])] if op_visible else None,
            })
    finally:
        if fp_writer is not None:
            fp_writer.release()
        if debug_writer is not None:
            debug_writer.release()
        sim.close()

    if fp_writer is not None and fp_video_temp_path.exists():
        transcode_to_h264_avc1(input_path=fp_video_temp_path, output_path=fp_video_path)
        fp_video_temp_path.unlink(missing_ok=True)
    if debug_writer is not None and debug_video_temp_path.exists():
        transcode_to_h264_avc1(input_path=debug_video_temp_path, output_path=debug_video_path)
        debug_video_temp_path.unlink(missing_ok=True)

    if storyboard_frames:
        storyboard = make_storyboard(storyboard_frames, storyboard_labels, cell_w=520, cell_h=360, cols=4)
        cv2.imwrite(str(storyboard_path_png), cv2.cvtColor(storyboard, cv2.COLOR_RGB2BGR))
    else:
        storyboard_path_png = None
    if last_clean_frame is not None:
        cv2.imwrite(str(final_fp_frame_path), cv2.cvtColor(last_clean_frame, cv2.COLOR_RGB2BGR))
    else:
        final_fp_frame_path = None
    if last_debug_panel is not None:
        cv2.imwrite(str(final_debug_frame_path), cv2.cvtColor(last_debug_panel, cv2.COLOR_RGB2BGR))
    else:
        final_debug_frame_path = None
    actions_info = write_base_actions(target_dir=target_dir, frame_rows=frame_rows, fps=float(args.fps))

    summary = {
        'target_uid': target_uid,
        'out_dir': str(target_dir),
        'fp_video_path': str(fp_video_path),
        'debug_video_path': str(debug_video_path) if debug_writer is not None else None,
        'video_codec': str(FINAL_VIDEO_CODEC),
        'actions_json': actions_info['json'],
        'actions_csv': actions_info['csv'],
        'actions_npy': actions_info['npy'],
        'actions_body_twist_npy': actions_info['body_twist_npy'],
        'actions_legacy_world_npy': actions_info['legacy_world_npy'],
        'actions_summary': actions_info['summary'],
        'topdown_curve_png': str(topdown_path_png),
        'storyboard_png': str(storyboard_path_png) if storyboard_path_png is not None else None,
        'final_fp_frame_png': str(final_fp_frame_path) if final_fp_frame_path is not None else None,
        'final_debug_frame_png': str(final_debug_frame_path) if final_debug_frame_path is not None else None,
        'start_facing_clean_frame_png': str(start_facing_clean_frame_path) if start_facing_clean_frame_path.exists() else None,
        'start_facing_debug_frame_png': str(start_facing_debug_frame_path) if start_facing_debug_frame_path.exists() else None,
        'start_facing_info': None if start_facing_info is None else {
            'yaw_deg': float(start_facing_info['yaw_deg']),
            'pitch_down_deg': float(start_facing_info['pitch_down_deg']),
            'target_visible': bool(start_facing_info['target_eval'].get('target_visible', False)),
            'target_not_clipped': bool(start_facing_info.get('target_not_clipped', False)),
            'operating_point_visible': bool(start_facing_info.get('op_visible', False)),
            'acceptable': bool(start_facing_info.get('acceptable', False)),
            'target_bbox_xyxy': start_facing_info.get('target_bbox_xyxy'),
            'operating_rgb_xy': start_facing_info.get('op_rgb_xy'),
        },
        'observer_world_xz': [float(v) for v in observer_xz.tolist()],
        'operating_world_xz': [float(v) for v in operating_xz.tolist()],
        'target_center_world_xyz': [float(v) for v in target_center_world_xyz.tolist()],
        'target_focus_world_xyz': [float(v) for v in target_focus_world_xyz.tolist()],
        'target_center_world_xz': [float(v) for v in target_center_world_xz.tolist()],
        'endpoint_raw_world_xz': [float(v) for v in endpoint_raw_xz.tolist()],
        'endpoint_world_xz': [float(v) for v in endpoint_xz.tolist()],
        'endpoint_strategy': str(endpoint_strategy),
        'endpoint_candidate_count': int(endpoint_candidate_count),
        'endpoint_to_operating_distance_m': float(np.linalg.norm(endpoint_xz.astype(np.float32) - operating_xz.astype(np.float32))),
        'heading_focus_world_xyz': [float(v) for v in heading_focus_world_xyz.tolist()],
        'start_yaw_deg': float(start_yaw_deg),
        'end_heading_yaw_deg': float(render_fp.yaw_between(op_world_xyz, heading_focus_world_xyz)) if op_world_xyz is not None else None,
        'pitch_down_deg': float(pitch_down_deg),
        'dynamic_pitch': bool(int(args.dynamic_pitch)),
        'dynamic_pitch_bias_deg': float(args.dynamic_pitch_bias_deg),
        'dynamic_pitch_max_deg': float(args.dynamic_pitch_max_deg),
        'dynamic_pitch_step_deg': float(args.dynamic_pitch_step_deg),
        'vfov_deg': float(vfov_deg),
        'hfov_deg': float(hfov_deg),
        'override_pitch_deg': float(args.override_pitch_deg),
        'override_vfov_deg': float(args.override_vfov_deg),
        'endpoint_offset_m': float(args.endpoint_offset_m),
        'endpoint_search_info': endpoint_search_info,
        'path_mode': 'curve_follow' if str(args.path_shape) == 'curve' else 'navmesh_shortest_path',
        'curve_navmask_required': bool(int(args.require_curve_navmask)),
        'curve_navmask_valid': bool(curve_navcheck.get('valid', True)),
        'curve_navmask_invalid_count': int(curve_navcheck.get('invalid_count', 0)),
        'curve_navmask_first_invalid_point': curve_navcheck.get('first_invalid_point'),
        'path_planning_info': path_planning_info,
        'path_points_world_xz': [[float(v) for v in row] for row in path_points_xz.tolist()] if path_points_xz is not None else None,
        'frame_count_requested': int(args.frame_count),
        'frame_count_rendered': int(len(frame_rows)),
        'stop_reason': stop_reason,
        'camera_geometry_mode': 'eye_locked_sensor_height',
        'frames': frame_rows,
    }
    (target_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
