#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

FINAL_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = FINAL_DIR.parent / 'habitat_scripts'
STAGE3_ROOT = Path('.')
OLD_BATCH_SUMMARY = Path('')
BASELINE_V2_ROOT = Path('')
API_ACCEPTED_RESULTS = Path('')
EXCLUDED_TARGET_UIDS_FILE = FINAL_DIR / 'hm3d_excluded_target_uids.txt'
MULTIVIEW_ROOTS = [
    Path(value).expanduser()
    for value in os.environ.get('UNIWAM_MULTIVIEW_ROOTS', '').split(os.pathsep)
    if value.strip()
]
_MULTIVIEW_INDEX: dict[str, list[dict]] | None = None


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def load_excluded_target_uids() -> set[str]:
    if not EXCLUDED_TARGET_UIDS_FILE.is_file():
        return set()
    return {
        line.strip() for line in EXCLUDED_TARGET_UIDS_FILE.read_text(encoding='utf-8').splitlines()
        if line.strip() and not line.lstrip().startswith('#')
    }


def load_module(path: Path, name: str) -> Any:
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def wrap_rad(a: float) -> float:
    return float((float(a) + math.pi) % (2.0 * math.pi) - math.pi)


def angle_diff_rad(a: float, b: float) -> float:
    return wrap_rad(float(b) - float(a))


def yaw_deg_to_forward_xz(yaw_deg: float) -> np.ndarray:
    r = math.radians(float(yaw_deg))
    return np.asarray([-math.sin(r), -math.cos(r)], dtype=np.float32)


def hm3d_yaw_deg_to_planar_theta(yaw_deg: float) -> float:
    f = yaw_deg_to_forward_xz(float(yaw_deg))
    return float(math.atan2(float(f[1]), float(f[0])))


def planar_theta_to_hm3d_yaw_deg(theta: float) -> float:
    dx = math.cos(float(theta))
    dz = math.sin(float(theta))
    return float((-math.degrees(math.atan2(dx, -dz))) % 360.0)


def normalize(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n > 1e-8:
        return (v / n).astype(np.float32)
    if fallback is None:
        return np.asarray([1.0, 0.0], dtype=np.float32)
    return normalize(fallback)


def bezier_point(p0, p1, p2, p3, u):
    u = float(u)
    q = 1.0 - u
    return ((q**3) * p0 + 3.0 * (q**2) * u * p1 + 3.0 * q * (u**2) * p2 + (u**3) * p3).astype(np.float32)


def bezier_tangent(p0, p1, p2, p3, u):
    u = float(u)
    q = 1.0 - u
    return normalize(3.0 * (q**2) * (p1 - p0) + 6.0 * q * u * (p2 - p1) + 3.0 * (u**2) * (p3 - p2), fallback=p3 - p0)


def interp_vec(cum: np.ndarray, values: np.ndarray, s: float) -> np.ndarray:
    s = float(s)
    if s <= 0.0:
        return values[0].copy()
    if s >= float(cum[-1]):
        return values[-1].copy()
    idx = int(np.searchsorted(cum, s, side='right') - 1)
    idx = max(0, min(idx, len(cum) - 2))
    s0, s1 = float(cum[idx]), float(cum[idx + 1])
    a = 0.0 if abs(s1 - s0) < 1e-9 else (s - s0) / (s1 - s0)
    return (values[idx] * (1.0 - a) + values[idx + 1] * a).astype(np.float32)


def interp_angle(cum: np.ndarray, angles: np.ndarray, s: float) -> float:
    arr = np.asarray(angles, dtype=np.float32).reshape(-1, 1)
    return float(interp_vec(cum, arr, float(s))[0])


def unwrap_rads(vals: np.ndarray) -> np.ndarray:
    out = np.asarray(vals, dtype=np.float64).copy()
    for i in range(1, out.shape[0]):
        out[i] = out[i - 1] + angle_diff_rad(out[i - 1], out[i])
    return out.astype(np.float32)


def ramp_toward_scalar(current: float, target: float, max_delta: float) -> float:
    current = float(current)
    target = float(target)
    max_delta = abs(float(max_delta))
    if abs(target - current) <= max_delta:
        return target
    return current + math.copysign(max_delta, target - current)


def command_smoothness_info(commands: np.ndarray) -> dict:
    cmds = np.asarray(commands, dtype=np.float32)
    if cmds.shape[0] < 2:
        return {
            'max_abs_dvyaw_per_frame': 0.0,
            'max_abs_dvx_per_frame': 0.0,
            'vyaw_large_sign_flip_count': 0,
            'vyaw_small_sign_flip_count': 0,
        }
    dvx = np.diff(cmds[:, 0])
    dvyaw = np.diff(cmds[:, 1])
    vyaw_a = cmds[:-1, 1]
    vyaw_b = cmds[1:, 1]
    large_flip = (vyaw_a * vyaw_b < 0.0) & (np.abs(vyaw_a) > 0.04) & (np.abs(vyaw_b) > 0.04)
    small_flip = (vyaw_a * vyaw_b < 0.0) & (np.abs(vyaw_a) > 0.005) & (np.abs(vyaw_b) > 0.005)
    return {
        'max_abs_dvyaw_per_frame': float(np.max(np.abs(dvyaw))) if dvyaw.size else 0.0,
        'max_abs_dvx_per_frame': float(np.max(np.abs(dvx))) if dvx.size else 0.0,
        'vyaw_large_sign_flip_count': int(np.count_nonzero(large_flip)),
        'vyaw_small_sign_flip_count': int(np.count_nonzero(small_flip)),
    }


def segment_name_for_frame(segment_rows: list[dict], frame_idx: int) -> str:
    frame_idx = int(frame_idx)
    best = 'start'
    for row in segment_rows:
        try:
            idx = int(row.get('idx', -1))
        except Exception:
            continue
        if idx <= frame_idx:
            best = str(row.get('segment', best))
        else:
            break
    return best


def rotation_profile(delta: float, *, rate: float, max_vyaw: float, max_vyaw_accel: float) -> list[float]:
    delta = float(delta)
    if abs(delta) <= 1e-6:
        return []
    sign = 1.0 if delta >= 0.0 else -1.0
    target = abs(delta)
    dt = 1.0 / float(rate)
    max_step = float(max_vyaw_accel) * dt
    for n in range(3, 6000):
        mids = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
        shape = np.sin(np.pi * mids)
        area = float(np.sum(shape) * dt)
        if area <= 1e-9:
            continue
        amp = target / area
        vals = amp * shape
        joined = np.concatenate([np.asarray([0.0]), vals, np.asarray([0.0])])
        if amp <= float(max_vyaw) + 1e-9 and float(np.max(np.abs(np.diff(joined)))) <= max_step + 1e-9:
            return [float(sign * v) for v in vals.tolist()]
    raise RuntimeError(f'failed to make rotation profile for delta={delta:.4f} rad')


def build_body_center_bezier(*, start_base_xy: np.ndarray, start_theta: float,
                             goal_base_xy: np.ndarray, goal_theta: float,
                             handle_scale: float, handle_max: float,
                             path_samples: int) -> dict:
    p0 = np.asarray(start_base_xy, dtype=np.float32)
    p5 = np.asarray(goal_base_xy, dtype=np.float32)
    dist = float(np.linalg.norm(p5 - p0))
    handle = max(0.10, min(float(handle_scale) * dist, float(handle_max), max(0.10, dist / 3.0)))
    f0 = np.asarray([math.cos(start_theta), math.sin(start_theta)], dtype=np.float32)
    f1 = np.asarray([math.cos(goal_theta), math.sin(goal_theta)], dtype=np.float32)
    controls = np.asarray([
        p0,
        p0 + handle * f0,
        p0 + 2.0 * handle * f0,
        p5 - 2.0 * handle * f1,
        p5 - handle * f1,
        p5,
    ], dtype=np.float32)

    def point(u: float) -> np.ndarray:
        row = controls.copy()
        for level in range(1, row.shape[0]):
            row[:row.shape[0] - level] = (1.0 - u) * row[:row.shape[0] - level] + u * row[1:row.shape[0] - level + 1]
        return row[0].astype(np.float32)

    deriv_controls = 5.0 * (controls[1:] - controls[:-1])

    def tangent(u: float) -> np.ndarray:
        row = deriv_controls.copy()
        for level in range(1, row.shape[0]):
            row[:row.shape[0] - level] = (1.0 - u) * row[:row.shape[0] - level] + u * row[1:row.shape[0] - level + 1]
        return normalize(row[0], fallback=p5 - p0)

    dense_n = max(1000, int(path_samples) * 8)
    us = np.linspace(0.0, 1.0, dense_n, dtype=np.float32)
    pts = np.asarray([point(float(u)) for u in us], dtype=np.float32)
    tans = np.asarray([tangent(float(u)) for u in us], dtype=np.float32)
    theta = np.asarray([math.atan2(float(t[1]), float(t[0])) for t in tans], dtype=np.float32)
    theta = unwrap_rads(theta)
    theta[-1] = theta[-2] + angle_diff_rad(float(theta[-2]), float(goal_theta)) if theta.shape[0] > 1 else float(goal_theta)
    seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    cum = np.concatenate([np.asarray([0.0], dtype=np.float32), np.cumsum(seg, dtype=np.float32)])
    return {'controls': controls, 'pts': pts, 'theta': theta, 'cum': cum, 'length_m': float(cum[-1]), 'handle_m': float(handle)}


def retime_nonholonomic(*, curve: dict, rate: float, max_vx: float, max_vyaw: float,
                        max_vx_accel: float, max_vyaw_accel: float | None = None,
                        decel_margin: float = 0.0, stop_tail: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    dt = 1.0 / float(rate)
    cum = curve['cum']
    pts = curve['pts']
    theta_dense = curve['theta']
    total = float(cum[-1])
    dv = 0.98 * float(max_vx_accel) * dt
    max_frames = int(max(1000, math.ceil(total / max(1e-6, max_vx) * float(rate) * 8.0 + 600)))
    best = None
    for n_frames in range(1, max_frames):
        envelope = np.asarray(
            [min(float(max_vx), dv * float(i + 1), dv * float(n_frames - i)) for i in range(n_frames)],
            dtype=np.float32,
        )
        capacity = float(np.sum(envelope) * dt)
        if capacity + 1e-8 < total:
            continue
        scale = 0.0 if capacity <= 1e-8 else min(1.0, total / capacity)
        speeds = envelope * float(scale)
        s_rows = [0.0]
        s = 0.0
        for speed in speeds:
            s = min(total, s + float(speed) * dt)
            s_rows.append(float(s))
        decorative_xy = np.asarray([interp_vec(cum, pts, ss) for ss in s_rows], dtype=np.float32)
        decorative_theta = np.asarray([interp_angle(cum, theta_dense, ss) for ss in s_rows], dtype=np.float32)
        commands = np.zeros((len(s_rows), 2), dtype=np.float32)
        for i in range(1, len(s_rows)):
            delta = decorative_xy[i] - decorative_xy[i - 1]
            prev_theta = float(decorative_theta[i - 1])
            world_vx = float(delta[0]) / dt
            world_vy = float(delta[1]) / dt
            vx = world_vx * math.cos(prev_theta) + world_vy * math.sin(prev_theta)
            vyaw = angle_diff_rad(float(decorative_theta[i - 1]), float(decorative_theta[i])) / dt
            commands[i, 0] = float(np.clip(vx, -float(max_vx), float(max_vx)))
            commands[i, 1] = float(np.clip(vyaw, -float(max_vyaw), float(max_vyaw)))
        info = command_smoothness_info(commands)
        yaw_accel_ok = True
        if max_vyaw_accel is not None:
            yaw_accel_ok = info['max_abs_dvyaw_per_frame'] <= float(max_vyaw_accel) * dt + 1e-5
        if float(np.max(np.abs(commands[:, 1]))) <= float(max_vyaw) + 1e-5 and yaw_accel_ok:
            best = (commands, decorative_xy, decorative_theta, info)
            break
    if best is None:
        raise RuntimeError(f'failed to retime curve under vx/vyaw constraints, length={total:.3f}m')
    commands, decorative_xy, decorative_theta, smooth_info = best
    stop_frames = 0 if float(stop_tail) <= 0.0 else max(3, int(float(stop_tail) * float(rate)))
    if stop_frames > 0:
        stop_xy = np.repeat(decorative_xy[-1:].copy(), stop_frames, axis=0)
        stop_theta = np.repeat(decorative_theta[-1:].copy(), stop_frames, axis=0)
        decorative_xy = np.concatenate([decorative_xy, stop_xy], axis=0)
        decorative_theta = np.concatenate([decorative_theta, stop_theta], axis=0)
        commands = np.concatenate([commands, np.zeros((stop_frames, 2), dtype=np.float32)], axis=0)
    # Authoritative executable trajectory: integrate generated vx/vyaw.
    sim_xy = np.zeros_like(decorative_xy)
    sim_theta = np.zeros_like(decorative_theta)
    sim_xy[0] = decorative_xy[0]
    sim_theta[0] = decorative_theta[0]
    for i in range(1, len(commands)):
        vx, vyaw = float(commands[i, 0]), float(commands[i, 1])
        th = float(sim_theta[i - 1])
        sim_xy[i, 0] = float(sim_xy[i - 1, 0]) + vx * math.cos(th) * dt
        sim_xy[i, 1] = float(sim_xy[i - 1, 1]) + vx * math.sin(th) * dt
        sim_theta[i] = wrap_rad(th + vyaw * dt)
    meta = {'frame_count': int(len(commands)), 'duration_s': float((len(commands) - 1) / float(rate)), 'stop_frames': int(stop_frames), 'smoothness': smooth_info}
    return commands, sim_xy, sim_theta, meta



def clean_tiny_tail_actions(commands: np.ndarray, *, vx_eps: float = 0.006, vyaw_eps: float = 0.015, min_tail_frames: int = 3) -> np.ndarray:
    """Remove visually useless tiny terminal pulses from generator output.

    The Direct-CAN Bezier generator can emit a few sub-millimeter correction pulses
    near the end. They make the action plot jitter but do not move the base
    meaningfully. Keep the main trajectory intact and zero only the terminal low
    amplitude region after the last meaningful command.
    """
    out = np.asarray(commands, dtype=np.float32).copy()
    if out.size == 0:
        return out
    meaningful = np.where((np.abs(out[:, 0]) >= float(vx_eps)) | (np.abs(out[:, 1]) >= float(vyaw_eps)))[0]
    if meaningful.size == 0:
        out[:] = 0.0
        return out
    last = int(meaningful[-1])
    if out.shape[0] - last - 1 >= int(min_tail_frames):
        out[last + 1:] = 0.0
    # Also suppress isolated tiny commands in the last 2 seconds.
    tail_start = max(0, out.shape[0] - 60)
    tiny = (np.abs(out[tail_start:, 0]) < float(vx_eps)) & (np.abs(out[tail_start:, 1]) < float(vyaw_eps))
    out[tail_start:][tiny] = 0.0
    return out

def xz_to_xy(xz: np.ndarray) -> np.ndarray:
    return np.asarray([float(xz[0]), float(xz[1])], dtype=np.float32)


def xy_to_xz(xy: np.ndarray) -> np.ndarray:
    return np.asarray([float(xy[0]), float(xy[1])], dtype=np.float32)


def theta_to_forward(theta: float) -> np.ndarray:
    return np.asarray([math.cos(float(theta)), math.sin(float(theta))], dtype=np.float32)


def compute_operating_edge_normal_theta(
    *,
    annotation_target: dict,
    p2w: dict | None,
    render_fp: Any,
    operating_world_xz: np.ndarray,
    target_center_world_xz: np.ndarray,
) -> dict | None:
    edge = annotation_target.get('operating_edge_line') if isinstance(annotation_target, dict) else None
    if not isinstance(edge, list) or len(edge) != 2 or p2w is None:
        return None
    p0, p1 = edge
    if not (isinstance(p0, list) and len(p0) == 2 and isinstance(p1, list) and len(p1) == 2):
        return None
    out_w = int(p2w.get('out_width', 0))
    out_h = int(p2w.get('out_height', 0))
    if out_w <= 1 or out_h <= 1:
        return None
    px0 = int(round(float(p0[0]) * float(out_w) / 999.0))
    py0 = int(round(float(p0[1]) * float(out_h) / 999.0))
    px1 = int(round(float(p1[0]) * float(out_w) / 999.0))
    py1 = int(round(float(p1[1]) * float(out_h) / 999.0))
    px0 = int(np.clip(px0, 0, out_w - 1))
    py0 = int(np.clip(py0, 0, out_h - 1))
    px1 = int(np.clip(px1, 0, out_w - 1))
    py1 = int(np.clip(py1, 0, out_h - 1))
    wx0, wz0 = render_fp.image_xy_to_world_xz(float(px0), float(py0), p2w)
    wx1, wz1 = render_fp.image_xy_to_world_xz(float(px1), float(py1), p2w)
    edge_vec = np.asarray([float(wx1 - wx0), float(wz1 - wz0)], dtype=np.float32)
    edge_len = float(np.linalg.norm(edge_vec))
    if edge_len <= 1e-6:
        return None
    edge_dir = edge_vec / edge_len
    normal_a = np.asarray([-float(edge_dir[1]), float(edge_dir[0])], dtype=np.float32)
    normal_b = -normal_a
    edge_mid = np.asarray([(float(wx0) + float(wx1)) * 0.5, (float(wz0) + float(wz1)) * 0.5], dtype=np.float32)
    op_to_center = np.asarray(target_center_world_xz, dtype=np.float32) - np.asarray(operating_world_xz, dtype=np.float32)
    dot_a = float(np.dot(normal_a, op_to_center))
    dot_b = float(np.dot(normal_b, op_to_center))
    normal = normal_a if dot_a >= dot_b else normal_b
    theta = float(math.atan2(float(normal[1]), float(normal[0])))
    return {
        'theta_rad': theta,
        'yaw_deg': float(planar_theta_to_hm3d_yaw_deg(theta)),
        'normal_xz': [float(normal[0]), float(normal[1])],
        'edge_world_xz': [[float(wx0), float(wz0)], [float(wx1), float(wz1)]],
        'edge_mid_world_xz': [float(edge_mid[0]), float(edge_mid[1])],
        'operating_to_target_center_xz': [float(op_to_center[0]), float(op_to_center[1])],
        'dot_normal_a_to_op_target_center': dot_a,
        'dot_normal_b_to_op_target_center': dot_b,
        'sign_source': 'operating_edge_line_normal_selected_by_operating_point_to_target_center',
    }


def camera_xz_from_base_xy(base_xy: np.ndarray, theta: float, camera_offset_m: float) -> np.ndarray:
    return np.asarray([
        float(base_xy[0]) + float(camera_offset_m) * math.cos(float(theta)),
        float(base_xy[1]) + float(camera_offset_m) * math.sin(float(theta)),
    ], dtype=np.float32)


def integrate_constant_twist_step(
    base_xy: np.ndarray,
    theta: float,
    vx: float,
    vyaw: float,
    dt: float,
) -> tuple[np.ndarray, float]:
    """Integrate one constant planar Twist command without Euler drift."""
    theta = float(theta)
    vx = float(vx)
    vyaw = float(vyaw)
    dt = float(dt)
    next_theta_unwrapped = theta + vyaw * dt
    if abs(vyaw) <= 1e-10:
        delta_x = vx * math.cos(theta) * dt
        delta_y = vx * math.sin(theta) * dt
    else:
        radius = vx / vyaw
        delta_x = radius * (math.sin(next_theta_unwrapped) - math.sin(theta))
        delta_y = -radius * (math.cos(next_theta_unwrapped) - math.cos(theta))
    return np.asarray([
        float(base_xy[0]) + delta_x,
        float(base_xy[1]) + delta_y,
    ], dtype=np.float32), wrap_rad(next_theta_unwrapped)


def plan_staged_direct_commands(
    *,
    start_base_xy: np.ndarray,
    start_theta: float,
    goal_camera_xz: np.ndarray,
    final_theta: float,
    camera_offset_m: float,
    rate: float,
    max_vx: float,
    max_vyaw: float,
    max_vx_accel: float,
    final_approach_m: float,
    stop_tail: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    dt = 1.0 / float(rate)
    base_rows = [np.asarray(start_base_xy, dtype=np.float32)]
    theta_rows = [float(start_theta)]
    command_rows = [np.asarray([0.0, 0.0], dtype=np.float32)]
    segment_rows: list[dict] = []

    def append_command(vx: float, vyaw: float, segment: str) -> None:
        prev_base = base_rows[-1]
        prev_theta = float(theta_rows[-1])
        vx = float(np.clip(vx, -float(max_vx), float(max_vx)))
        vyaw = float(np.clip(vyaw, -float(max_vyaw), float(max_vyaw)))
        next_base = np.asarray([
            float(prev_base[0]) + vx * math.cos(prev_theta) * dt,
            float(prev_base[1]) + vx * math.sin(prev_theta) * dt,
        ], dtype=np.float32)
        next_theta = wrap_rad(prev_theta + vyaw * dt)
        base_rows.append(next_base)
        theta_rows.append(next_theta)
        command_rows.append(np.asarray([vx, vyaw], dtype=np.float32))
        segment_rows.append({'idx': int(len(command_rows) - 1), 'segment': segment, 'vx': vx, 'vyaw': vyaw})

    def append_rotate(theta_target: float, segment: str) -> None:
        for _ in range(2000):
            delta = angle_diff_rad(float(theta_rows[-1]), float(theta_target))
            if abs(delta) <= 1e-4:
                theta_rows[-1] = wrap_rad(float(theta_target))
                break
            vyaw = math.copysign(min(float(max_vyaw), abs(delta) / dt), delta)
            append_command(0.0, vyaw, segment)

    def append_drive(distance: float, segment: str) -> None:
        distance = max(0.0, float(distance))
        if distance <= 1e-5:
            append_command(0.0, 0.0, f'{segment}_settle')
            return
        dv = float(max_vx_accel) * dt
        speeds = None
        for n_frames in range(1, 20000):
            envelope = np.asarray(
                [
                    min(
                        float(max_vx),
                        dv * float(i + 1),
                        dv * float(n_frames - i),
                    )
                    for i in range(n_frames)
                ],
                dtype=np.float32,
            )
            capacity = float(np.sum(envelope) * dt)
            if capacity + 1e-8 >= distance:
                scale = 0.0 if capacity <= 1e-8 else distance / capacity
                speeds = envelope * float(min(1.0, scale))
                break
        if speeds is None:
            raise RuntimeError(f'append_drive failed to fit distance={distance:.4f}m')
        for vx in speeds:
            append_command(float(vx), 0.0, segment)
        append_command(0.0, 0.0, f'{segment}_settle')

    def cubic_point(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, u: float) -> np.ndarray:
        q = 1.0 - float(u)
        return (
            (q**3) * p0
            + 3.0 * (q**2) * float(u) * p1
            + 3.0 * q * (float(u) ** 2) * p2
            + (float(u) ** 3) * p3
        ).astype(np.float32)

    def cubic_tangent(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, u: float) -> np.ndarray:
        q = 1.0 - float(u)
        return normalize(
            3.0 * (q**2) * (p1 - p0)
            + 6.0 * q * float(u) * (p2 - p1)
            + 3.0 * (float(u) ** 2) * (p3 - p2),
            fallback=p3 - p0,
        )

    def append_bezier_segment(
        start_xy: np.ndarray,
        start_theta_seg: float,
        end_xy: np.ndarray,
        end_theta_seg: float,
        segment: str,
    ) -> None:
        p0 = np.asarray(start_xy, dtype=np.float32)
        p3 = np.asarray(end_xy, dtype=np.float32)
        dist = float(np.linalg.norm(p3 - p0))
        if dist <= 1e-4:
            append_rotate(end_theta_seg, f'{segment}_rotate_only')
            return
        handle = max(0.04, min(0.65 * dist, 0.22))
        p1 = p0 + handle * np.asarray([math.cos(float(start_theta_seg)), math.sin(float(start_theta_seg))], dtype=np.float32)
        p2 = p3 - handle * np.asarray([math.cos(float(end_theta_seg)), math.sin(float(end_theta_seg))], dtype=np.float32)
        samples = max(80, int(math.ceil(dist / 0.006)))
        us = np.linspace(0.0, 1.0, samples, dtype=np.float32)
        pts = np.asarray([cubic_point(p0, p1, p2, p3, float(u)) for u in us], dtype=np.float32)
        tangents = np.asarray([cubic_tangent(p0, p1, p2, p3, float(u)) for u in us], dtype=np.float32)
        yaws = np.asarray([math.atan2(float(t[1]), float(t[0])) for t in tangents], dtype=np.float32)
        yaws = unwrap_rads(yaws)
        yaws[-1] = yaws[-2] + angle_diff_rad(float(yaws[-2]), float(end_theta_seg)) if yaws.shape[0] > 1 else float(end_theta_seg)
        seg_len = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        cum = np.concatenate([np.asarray([0.0], dtype=np.float32), np.cumsum(seg_len, dtype=np.float32)])
        total = float(cum[-1])
        dv = float(max_vx_accel) * dt
        speeds = None
        for n_frames in range(1, 20000):
            envelope = np.asarray(
                [
                    min(
                        float(max_vx),
                        dv * float(i + 1),
                        dv * float(n_frames - i),
                    )
                    for i in range(n_frames)
                ],
                dtype=np.float32,
            )
            capacity = float(np.sum(envelope) * dt)
            if capacity + 1e-8 >= total:
                scale = 0.0 if capacity <= 1e-8 else total / capacity
                speeds = envelope * float(min(1.0, scale))
                break
        if speeds is None:
            raise RuntimeError(f'append_bezier_segment failed to fit length={total:.4f}m')
        s = 0.0
        for speed in speeds:
            next_s = min(total, s + float(speed) * dt)
            target_xy = interp_vec(cum, pts, next_s)
            target_theta = interp_angle(cum, yaws, next_s)
            prev_xy = base_rows[-1]
            prev_theta = float(theta_rows[-1])
            delta = target_xy - prev_xy
            world_vx = float(delta[0]) / dt
            world_vy = float(delta[1]) / dt
            vx = world_vx * math.cos(prev_theta) + world_vy * math.sin(prev_theta)
            vyaw = angle_diff_rad(prev_theta, target_theta) / dt
            append_command(vx, vyaw, segment)
            s = next_s
        if abs(angle_diff_rad(float(theta_rows[-1]), float(end_theta_seg))) > 1e-3:
            append_rotate(end_theta_seg, f'{segment}_final_yaw_trim')
        append_command(0.0, 0.0, f'{segment}_settle')

    goal_camera = np.asarray(goal_camera_xz, dtype=np.float32)
    final_forward = np.asarray([math.cos(float(final_theta)), math.sin(float(final_theta))], dtype=np.float32)
    final_base_xy = goal_camera - float(camera_offset_m) * final_forward
    total_dist = float(np.linalg.norm(final_base_xy - np.asarray(start_base_xy, dtype=np.float32)))
    final_approach = min(float(final_approach_m), max(0.05, total_dist * 0.35))
    pre_final_base = final_base_xy - final_approach * final_forward
    terminal_curve_m = min(0.45, max(0.18, total_dist * 0.18))
    terminal_curve_start = pre_final_base - terminal_curve_m * final_forward

    first_vec = terminal_curve_start - np.asarray(start_base_xy, dtype=np.float32)
    first_dist = float(np.linalg.norm(first_vec))
    if first_dist > 1e-4:
        first_theta = float(math.atan2(float(first_vec[1]), float(first_vec[0])))
    else:
        first_theta = float(final_theta)

    append_rotate(first_theta, 'rotate_to_direct_line')
    append_drive(first_dist, 'drive_direct_line')
    append_bezier_segment(terminal_curve_start, first_theta, pre_final_base, final_theta, 'terminal_bezier_turn')
    append_drive(float(final_approach), 'final_forward_approach')

    stop_frames = max(3, int(float(stop_tail) * float(rate)))
    for _ in range(stop_frames):
        append_command(0.0, 0.0, 'stop_tail')

    base_xy = np.asarray(base_rows, dtype=np.float32)
    theta = np.asarray(theta_rows, dtype=np.float32)
    commands = np.asarray(command_rows, dtype=np.float32)
    meta = {
        'mode': 'staged_direct_drive_terminal_bezier_final_approach',
        'final_approach_m': float(final_approach),
        'terminal_curve_m': float(terminal_curve_m),
        'first_line_distance_m': float(first_dist),
        'final_base_xy': [float(v) for v in final_base_xy.tolist()],
        'pre_final_base_xy': [float(v) for v in pre_final_base.tolist()],
        'terminal_curve_start_xy': [float(v) for v in terminal_curve_start.tolist()],
        'segments': segment_rows,
    }
    return commands, base_xy, theta, meta


def drive_profile(distance: float, *, rate: float, max_vx: float, max_vx_accel: float) -> list[float]:
    distance = max(0.0, float(distance))
    if distance <= 1e-7:
        return []
    dt = 1.0 / float(rate)
    dv = float(max_vx_accel) * dt
    for n_frames in range(1, 30000):
        envelope = np.asarray(
            [min(float(max_vx), dv * float(i + 1), dv * float(n_frames - i)) for i in range(n_frames)],
            dtype=np.float64,
        )
        capacity = float(np.sum(envelope) * dt)
        if capacity + 1e-10 >= distance:
            scale = 0.0 if capacity <= 1e-10 else distance / capacity
            return [float(v) for v in (envelope * min(1.0, scale)).tolist()]
    raise RuntimeError(f'failed to make drive profile for distance={distance:.4f}m')


def plan_poseaware_line_final_approach_commands(
    *,
    start_base_xy: np.ndarray,
    start_theta: float,
    face_target_theta: float,
    goal_camera_xz: np.ndarray,
    final_theta: float,
    camera_offset_m: float,
    rate: float,
    max_vx: float,
    max_vyaw: float,
    max_vx_accel: float,
    max_vyaw_accel: float,
    approach_candidates_m: tuple[float, ...],
    stop_tail: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    dt = 1.0 / float(rate)
    base_rows = [np.asarray(start_base_xy, dtype=np.float32)]
    theta_rows = [float(start_theta)]
    command_rows = [np.asarray([0.0, 0.0], dtype=np.float32)]
    segment_rows: list[dict] = []

    def append(vx: float, vyaw: float, segment: str) -> None:
        prev_base = np.asarray(base_rows[-1], dtype=np.float32)
        prev_theta = float(theta_rows[-1])
        vx = float(np.clip(vx, 0.0, float(max_vx)))
        vyaw = float(np.clip(vyaw, -float(max_vyaw), float(max_vyaw)))
        next_base = np.asarray([
            float(prev_base[0]) + vx * math.cos(prev_theta) * dt,
            float(prev_base[1]) + vx * math.sin(prev_theta) * dt,
        ], dtype=np.float32)
        next_theta = wrap_rad(prev_theta + vyaw * dt)
        base_rows.append(next_base)
        theta_rows.append(next_theta)
        command_rows.append(np.asarray([vx, vyaw], dtype=np.float32))
        segment_rows.append({'idx': int(len(command_rows) - 1), 'segment': segment, 'vx': vx, 'vyaw': vyaw})

    def append_rotation(theta_target: float, segment: str) -> None:
        delta = angle_diff_rad(float(theta_rows[-1]), float(theta_target))
        for vyaw in rotation_profile(delta, rate=rate, max_vyaw=max_vyaw, max_vyaw_accel=max_vyaw_accel):
            append(0.0, float(vyaw), segment)
        theta_rows[-1] = wrap_rad(float(theta_target))

    def append_drive(distance: float, segment: str) -> None:
        for vx in drive_profile(distance, rate=rate, max_vx=max_vx, max_vx_accel=max_vx_accel):
            append(float(vx), 0.0, segment)

    goal_camera = np.asarray(goal_camera_xz, dtype=np.float32)
    final_forward = theta_to_forward(float(final_theta))
    final_base = goal_camera - float(camera_offset_m) * final_forward
    total_dist = float(np.linalg.norm(final_base - np.asarray(start_base_xy, dtype=np.float32)))
    candidates = [float(v) for v in approach_candidates_m if float(v) > 0.0] or [0.2]
    requested_approach = float(candidates[0])
    preferred_approach = min(requested_approach, max(0.05, total_dist * 0.45))
    min_approach = max(0.04, min(preferred_approach, min(candidates)) - 0.02)
    max_approach = min(max(0.08, total_dist * 0.70), max(preferred_approach, requested_approach) + 0.02)

    selected_approach = min(max(float(preferred_approach), float(min_approach)), float(max_approach))
    pre_final_base = final_base - selected_approach * final_forward
    first_vec = pre_final_base - np.asarray(start_base_xy, dtype=np.float32)
    first_dist = float(np.linalg.norm(first_vec))
    first_theta = float(math.atan2(float(first_vec[1]), float(first_vec[0]))) if first_dist > 1e-6 else float(final_theta)

    append_rotation(first_theta, 'rotate_to_pre_op_standoff')
    append_drive(first_dist, 'smooth_forward_to_pre_op_standoff')
    before_final_rotate_base = np.asarray(base_rows[-1], dtype=np.float32).copy()
    before_final_rotate_theta = float(theta_rows[-1])
    append_rotation(float(final_theta), 'rotate_to_operating_edge_normal')
    before_approach_base = np.asarray(base_rows[-1], dtype=np.float32).copy()
    lateral_axis = np.asarray([-math.sin(float(final_theta)), math.cos(float(final_theta))], dtype=np.float32)
    final_approach_lateral_error = float(np.dot(final_base - before_approach_base, lateral_axis))
    final_approach_dist = float(np.dot(final_base - before_approach_base, final_forward))
    append_drive(max(0.0, final_approach_dist), 'short_perpendicular_approach')

    stop_frames = max(3, int(float(stop_tail) * float(rate)))
    for _ in range(stop_frames):
        append(0.0, 0.0, 'stop_tail')

    base_xy = np.asarray(base_rows, dtype=np.float32)
    theta = np.asarray(theta_rows, dtype=np.float32)
    commands = np.asarray(command_rows, dtype=np.float32)
    meta = {
        'mode': 'poseaware_rotate_to_pre_op_standoff_rotate_operating_forward_approach',
        'goal_base_xy': [float(v) for v in final_base.tolist()],
        'pre_final_base_xy': [float(v) for v in pre_final_base.tolist()],
        'selected_approach_m': float(selected_approach),
        'requested_approach_m': float(requested_approach),
        'preferred_approach_m': float(preferred_approach),
        'approach_window_m': [float(min_approach), float(max_approach)],
        'first_line_distance_m': float(first_dist),
        'first_line_theta_rad': float(first_theta),
        'face_target_theta_rad': float(face_target_theta),
        'pre_op_standoff_camera_xz': [float(v) for v in (np.asarray(pre_final_base, dtype=np.float32) + float(camera_offset_m) * final_forward).tolist()],
        'first_line_target': 'pre_op_standoff_point',
        'before_final_rotate_base_xy': [float(v) for v in before_final_rotate_base.tolist()],
        'before_final_rotate_theta_rad': float(before_final_rotate_theta),
        'final_approach_dist_m': float(final_approach_dist),
        'final_approach_lateral_error_m': float(final_approach_lateral_error),
        'segments': segment_rows,
        'action_smoothness': command_smoothness_info(commands),
    }
    return commands, base_xy, theta, meta


def plan_face_target_then_global_bezier_commands(
    *,
    start_base_xy: np.ndarray,
    start_theta: float,
    face_target_theta: float,
    goal_camera_xz: np.ndarray,
    final_theta: float,
    camera_offset_m: float,
    rate: float,
    max_vx: float,
    max_vyaw: float,
    max_vx_accel: float,
    max_vyaw_accel: float,
    curve_handle_scale: float,
    curve_handle_max: float,
    path_samples: int,
    retime_decel_margin: float,
    approach_candidates_m: tuple[float, ...],
    stop_tail: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    dt = 1.0 / float(rate)
    command_rows = [np.asarray([0.0, 0.0], dtype=np.float32)]
    base_rows = [np.asarray(start_base_xy, dtype=np.float32)]
    theta_rows = [float(start_theta)]
    segment_rows: list[dict] = []

    def append_rotate(theta_target: float, segment: str) -> None:
        delta = angle_diff_rad(float(theta_rows[-1]), float(theta_target))
        for vyaw in rotation_profile(delta, rate=rate, max_vyaw=max_vyaw, max_vyaw_accel=max_vyaw_accel):
            base_rows.append(base_rows[-1].copy())
            theta_rows.append(wrap_rad(float(theta_rows[-1]) + vyaw * dt))
            command_rows.append(np.asarray([0.0, vyaw], dtype=np.float32))
            segment_rows.append({'idx': int(len(command_rows) - 1), 'segment': segment, 'vx': 0.0, 'vyaw': float(vyaw)})
        theta_rows[-1] = wrap_rad(float(theta_target))

    goal_camera = np.asarray(goal_camera_xz, dtype=np.float32)
    final_forward = np.asarray([math.cos(float(final_theta)), math.sin(float(final_theta))], dtype=np.float32)
    goal_base_xy = goal_camera - float(camera_offset_m) * final_forward

    append_rotate(float(face_target_theta), 'rotate_to_face_target')
    base_after_rotate = np.asarray(base_rows[-1], dtype=np.float32)
    theta_after_rotate = float(theta_rows[-1])
    approach_m = float(approach_candidates_m[0]) if approach_candidates_m else 0.4
    approach_m = max(0.05, approach_m)
    total_dist = float(np.linalg.norm(goal_base_xy - base_after_rotate))
    approach_m = min(approach_m, max(0.05, total_dist * 0.45))
    pre_goal_base_xy = goal_base_xy - approach_m * final_forward
    curve_goal_base_xy = pre_goal_base_xy - 0.30 * approach_m * final_forward
    curve = build_body_center_bezier(
        start_base_xy=base_after_rotate,
        start_theta=theta_after_rotate,
        goal_base_xy=np.asarray(curve_goal_base_xy, dtype=np.float32),
        goal_theta=float(final_theta),
        handle_scale=float(curve_handle_scale),
        handle_max=float(curve_handle_max),
        path_samples=int(path_samples),
    )
    bezier_commands, bezier_xy, bezier_theta, bezier_meta = retime_nonholonomic(
        curve=curve,
        rate=float(rate),
        max_vx=float(max_vx),
        max_vyaw=float(max_vyaw),
        max_vx_accel=float(max_vx_accel),
        max_vyaw_accel=float(max_vyaw_accel),
        decel_margin=float(retime_decel_margin),
        stop_tail=0.0,
    )
    for j in range(1, bezier_commands.shape[0]):
        command_rows.append(bezier_commands[j].astype(np.float32))
        base_rows.append(bezier_xy[j].astype(np.float32))
        theta_rows.append(float(bezier_theta[j]))
        segment_rows.append({
            'idx': int(len(command_rows) - 1),
            'segment': 'global_bezier_to_operating_point',
            'vx': float(bezier_commands[j, 0]),
            'vyaw': float(bezier_commands[j, 1]),
        })

    append_rotate(float(final_theta), 'rotate_to_edge_normal')

    current_base = np.asarray(base_rows[-1], dtype=np.float32)
    current_theta = float(theta_rows[-1])
    lateral = float(np.dot(goal_base_xy - current_base, np.asarray([-math.sin(current_theta), math.cos(current_theta)], dtype=np.float32)))
    distance = float(np.dot(goal_base_xy - current_base, np.asarray([math.cos(current_theta), math.sin(current_theta)], dtype=np.float32)))
    if distance > 1e-5:
        dv = 0.98 * float(max_vx_accel) * dt
        speeds = None
        for n_frames in range(1, 20000):
            envelope = np.asarray(
                [
                    min(
                        float(max_vx),
                        dv * float(i + 1),
                        dv * float(n_frames - i),
                    )
                    for i in range(n_frames)
                ],
                dtype=np.float32,
            )
            capacity = float(np.sum(envelope) * dt)
            if capacity + 1e-8 >= distance:
                scale = 0.0 if capacity <= 1e-8 else distance / capacity
                speeds = envelope * float(min(1.0, scale))
                break
        if speeds is None:
            raise RuntimeError(f'final approach failed to fit distance={distance:.4f}m')
        for vx in speeds:
            prev_base = np.asarray(base_rows[-1], dtype=np.float32)
            prev_theta = float(theta_rows[-1])
            vx = float(np.clip(float(vx), 0.0, float(max_vx)))
            next_base = np.asarray([
                float(prev_base[0]) + vx * math.cos(prev_theta) * dt,
                float(prev_base[1]) + vx * math.sin(prev_theta) * dt,
            ], dtype=np.float32)
            base_rows.append(next_base)
            theta_rows.append(prev_theta)
            command_rows.append(np.asarray([vx, 0.0], dtype=np.float32))
            segment_rows.append({'idx': int(len(command_rows) - 1), 'segment': 'short_perpendicular_approach', 'vx': float(vx), 'vyaw': 0.0})

    stop_frames = max(3, int(float(stop_tail) * float(rate)))
    for _ in range(stop_frames):
        base_rows.append(np.asarray(base_rows[-1], dtype=np.float32).copy())
        theta_rows.append(float(theta_rows[-1]))
        command_rows.append(np.asarray([0.0, 0.0], dtype=np.float32))
        segment_rows.append({'idx': int(len(command_rows) - 1), 'segment': 'stop_tail', 'vx': 0.0, 'vyaw': 0.0})

    commands = np.asarray(command_rows, dtype=np.float32)
    base_xy = np.asarray(base_rows, dtype=np.float32)
    theta = np.asarray(theta_rows, dtype=np.float32)
    meta = {
        'mode': 'strict_face_target_bezier_edge_normal_short_perpendicular',
        'goal_base_xy': [float(v) for v in goal_base_xy.tolist()],
        'pre_goal_base_xy': [float(v) for v in pre_goal_base_xy.tolist()],
        'curve_goal_base_xy': [float(v) for v in curve_goal_base_xy.tolist()],
        'selected_approach_m': float(approach_m),
        'final_approach_dist_m': float(distance),
        'final_approach_lateral_error_m': float(lateral),
        'curve_length_m': float(curve.get('length_m', 0.0)),
        'curve_handle_m': float(curve.get('handle_m', 0.0)),
        'bezier_frame_count': int(bezier_meta.get('frame_count', 0)),
        'segments': segment_rows,
        'action_smoothness': command_smoothness_info(commands),
    }
    return commands, base_xy, theta, meta


def write_world_plan_diagnostic(
    out_path: Path,
    *,
    base_xy: np.ndarray,
    cam_xy: np.ndarray,
    op_xz: np.ndarray,
    target_xz: np.ndarray,
    start_yaw_deg: float,
    final_yaw_deg: float,
    target_uid: str,
) -> None:
    pts = np.concatenate([
        np.asarray(base_xy, dtype=np.float32),
        np.asarray(cam_xy, dtype=np.float32),
        np.asarray(op_xz, dtype=np.float32).reshape(1, 2),
        np.asarray(target_xz, dtype=np.float32).reshape(1, 2),
    ], axis=0)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    pad = max(0.6, float(np.max(mx - mn)) * 0.15)
    mn -= pad
    mx += pad
    width, height = 1200, 900
    img = np.full((height, width, 3), 245, dtype=np.uint8)

    def to_img(p: np.ndarray) -> tuple[int, int]:
        x = (float(p[0]) - float(mn[0])) / max(1e-6, float(mx[0] - mn[0]))
        y = (float(p[1]) - float(mn[1])) / max(1e-6, float(mx[1] - mn[1]))
        return int(round(x * (width - 1))), int(round((1.0 - y) * (height - 1)))

    def put(text: str, org: tuple[int, int], color=(0, 0, 0), scale=0.55) -> None:
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    def poly(path: np.ndarray, color: tuple[int, int, int], thick: int) -> None:
        arr = np.asarray([to_img(p) for p in path], dtype=np.int32)
        if len(arr) >= 2:
            cv2.polylines(img, [arr], False, color, thick, cv2.LINE_AA)

    def point(name: str, p: np.ndarray, color: tuple[int, int, int], r: int = 7) -> None:
        q = to_img(np.asarray(p, dtype=np.float32))
        cv2.circle(img, q, r, color, -1, cv2.LINE_AA)
        put(name, (q[0] + 8, q[1] - 8), color, 0.55)

    def yaw_forward(yaw_deg: float) -> np.ndarray:
        yaw_rad = math.radians(float(yaw_deg))
        return np.asarray([-math.sin(yaw_rad), -math.cos(yaw_rad)], dtype=np.float32)

    def arrow(name: str, p: np.ndarray, yaw_deg: float, color: tuple[int, int, int]) -> None:
        start = np.asarray(p, dtype=np.float32)
        end = start + yaw_forward(yaw_deg) * 0.75
        a, b = to_img(start), to_img(end)
        cv2.arrowedLine(img, a, b, color, 3, cv2.LINE_AA, tipLength=0.25)
        put(name, (b[0] + 6, b[1] + 6), color, 0.48)

    for gx in np.linspace(float(mn[0]), float(mx[0]), 8):
        cv2.line(img, to_img(np.asarray([gx, mn[1]], dtype=np.float32)), to_img(np.asarray([gx, mx[1]], dtype=np.float32)), (220, 220, 220), 1)
    for gy in np.linspace(float(mn[1]), float(mx[1]), 8):
        cv2.line(img, to_img(np.asarray([mn[0], gy], dtype=np.float32)), to_img(np.asarray([mx[0], gy], dtype=np.float32)), (220, 220, 220), 1)
    poly(base_xy, (255, 128, 0), 3)
    poly(cam_xy, (0, 200, 200), 3)
    point('start_cam', cam_xy[0], (0, 180, 0))
    point('op/camera_goal', op_xz, (0, 0, 255), 9)
    point('final_cam', cam_xy[-1], (0, 255, 255), 7)
    point('final_base', base_xy[-1], (255, 128, 0), 7)
    point('target', target_xz, (255, 0, 0), 9)
    arrow('start_yaw', cam_xy[0], start_yaw_deg, (0, 160, 0))
    arrow('final_edge_normal', cam_xy[-1], final_yaw_deg, (255, 0, 0))
    put('pose-aware trajectory: smooth rotate -> smooth transition -> edge-normal rotate -> short approach', (15, 30), (0, 0, 0), 0.62)
    put(target_uid, (15, 60), (0, 0, 0), 0.52)
    put(f'camera->op={float(np.linalg.norm(cam_xy[-1] - op_xz)):.3f}m  base->op={float(np.linalg.norm(base_xy[-1] - op_xz)):.3f}m', (15, 86), (0, 0, 0), 0.52)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def draw_text(img, text, org, scale=0.48, color=(30, 30, 30)):
    x, y = org
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def fit_image(img: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(float(width) / max(1.0, w), float(height) / max(1.0, h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    x0 = (width - nw) // 2
    y0 = (height - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def transcode_to_h264(input_path: Path, output_path: Path) -> None:
    subprocess.run([
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-i', str(input_path),
        '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
        '-pix_fmt', 'yuv420p', '-profile:v', 'baseline',
        '-level', '3.1', '-movflags', '+faststart', '-tag:v', 'avc1', str(output_path),
    ], check=True)


def make_action_panel(commands: np.ndarray, idx: int, width: int, height: int, max_vx: float, max_vyaw: float) -> np.ndarray:
    panel = np.full((height, width, 3), 255, dtype=np.uint8)
    n = int(commands.shape[0])
    draw_text(panel, f'Direct-CAN commands {idx}/{n-1}', (12, 24))
    draw_text(panel, f'vx={commands[idx,0]:+.3f} m/s  vyaw={commands[idx,1]:+.3f} rad/s', (12, 48))
    x0, y0 = 24, 78
    pw, ph = width - 48, height - 108
    cv2.rectangle(panel, (x0, y0), (x0 + pw, y0 + ph), (200, 200, 200), 1)
    cv2.line(panel, (x0, y0 + ph // 2), (x0 + pw, y0 + ph // 2), (220, 220, 220), 1)
    for vals, lim, color in [(commands[:, 0], max_vx, (0, 130, 255)), (commands[:, 1], max_vyaw, (255, 80, 80))]:
        xs = np.linspace(x0, x0 + pw, n)
        ys = y0 + ph // 2 - np.clip(vals / max(1e-6, lim), -1, 1) * ph * 0.42
        cv2.polylines(panel, [np.stack([xs, ys], 1).astype(np.int32)], False, color, 2, cv2.LINE_AA)
    xc = int(round(x0 + pw * idx / max(1, n - 1)))
    cv2.line(panel, (xc, y0), (xc, y0 + ph), (0, 0, 0), 1)
    draw_text(panel, 'blue=vx, red=vyaw', (12, height - 14), 0.42)
    return panel


def load_case_context(target_uid: str, modules: dict, width: int, height: int, hfov_deg: float, camera_height_m: float) -> dict:
    curve = modules['curve']
    base_views = modules['base_views']
    render_fp = modules['render_fp']
    base_mod = modules['base_mod']
    manifest = load_json(STAGE3_ROOT / 'final_observations' / f'{target_uid}__manifest.json')
    plan = load_json(STAGE3_ROOT / 'plan' / target_uid / 'result.json')
    expanded_meta = load_json(Path(plan['expanded_sample_dir']) / 'meta.json')
    stage_meta = load_json(Path(expanded_meta['source_meta_json']))
    traj = load_json(Path(stage_meta['trajectory_json']))
    target_row = stage_meta['targets'][int(plan['target_index'])]
    nav_y = float(stage_meta['nav_y'])
    scene_source = str(traj.get('scene_path', '')).strip() or str(traj['scene_id'])
    sim = base_views.build_sim_for_target(
        base_mod=base_mod,
        traj=traj,
        scene_source=scene_source,
        width=int(width),
        height=int(height),
        sensor_height_m=float(camera_height_m),
        hfov_deg=float(hfov_deg),
    )
    old_summary = None
    for row in load_json(OLD_BATCH_SUMMARY).get('rows', []):
        if row.get('target_uid') == target_uid:
            old_summary = load_json(Path(row['normal_summary_json']))
            break
    p2w_path = Path(plan['expanded_sample_dir']) / 'pixel_to_world.json'
    if p2w_path.exists():
        p2w = load_json(p2w_path)
    else:
        p2w = expanded_meta.get('topdown', {}).get('pixel_to_world_xz')
    target_semantic_ids = [int(v) for v in manifest.get('target_semantic_ids', []) if int(v) > 0]
    if old_summary is None:
        observer_xz = np.asarray(manifest['observer_point_used']['observer_world_xz'], dtype=np.float32)
        operating_xz = np.asarray(plan['operating_point_world_xz'], dtype=np.float32)
        target_center_world_xz = np.asarray(target_row.get('topdown_center_world_xz') or operating_xz, dtype=np.float32)
        target_focus_world_xyz = render_fp.resolve_target_focus_world_xyz(target_row, target_center_world_xz, nav_y)
        operating_world_xyz = render_fp.precise_snap(sim.pathfinder, float(operating_xz[0]), float(operating_xz[1]), nav_y)
        if operating_world_xyz is None:
            operating_world_xyz = np.asarray([float(operating_xz[0]), nav_y, float(operating_xz[1])], dtype=np.float32)
        heading_focus_world_xyz = base_views.compute_heading_from_operating_edge(
            annotation_target=plan.get('annotation_target', {}),
            p2w=p2w,
            target_center_world_xz=np.asarray(target_center_world_xz, dtype=np.float32),
            nav_y=float(nav_y),
        ) if p2w is not None else None
        if heading_focus_world_xyz is None:
            heading_focus_world_xyz = target_focus_world_xyz.copy()
        old_summary = {
            'target_uid': target_uid,
            'observer_world_xz': [float(v) for v in observer_xz.tolist()],
            'operating_world_xz': [float(v) for v in operating_xz.tolist()],
            'target_center_world_xz': [float(v) for v in target_center_world_xz.tolist()],
            'target_focus_world_xyz': [float(v) for v in np.asarray(target_focus_world_xyz, dtype=np.float32).tolist()],
            'heading_focus_world_xyz': [float(v) for v in np.asarray(heading_focus_world_xyz, dtype=np.float32).tolist()],
            'start_yaw_deg': float(manifest.get('fp_yaw_deg', render_fp.yaw_between(np.asarray([observer_xz[0], nav_y, observer_xz[1]], dtype=np.float32), np.asarray(target_focus_world_xyz, dtype=np.float32)))),
            'end_heading_yaw_deg': float(render_fp.yaw_between(np.asarray(operating_world_xyz, dtype=np.float32), np.asarray(heading_focus_world_xyz, dtype=np.float32))),
            'topdown_curve_png': str(plan.get('artifacts', {}).get('topdown_observer_candidates', '')),
        }
    else:
        target_center_world_xz = np.asarray(old_summary.get('target_center_world_xz') or target_row.get('topdown_center_world_xz'), dtype=np.float32)
        target_focus_world_xyz = np.asarray(old_summary.get('target_focus_world_xyz'), dtype=np.float32)
        heading_focus_world_xyz = np.asarray(old_summary.get('heading_focus_world_xyz'), dtype=np.float32)
    if not target_semantic_ids:
        matched = target_row.get('matched_semantic', {}) if isinstance(target_row, dict) else {}
        target_semantic_ids = [int(v) for v in matched.get('semantic_ids', []) if int(v) > 0]
    if not target_semantic_ids and isinstance(target_row, dict) and target_row.get('target_object_semantic_id') is not None:
        target_semantic_ids = [int(target_row['target_object_semantic_id'])]
    return {
        'sim': sim, 'manifest': manifest, 'plan': plan, 'stage_meta': stage_meta, 'traj': traj,
        'old_summary': old_summary, 'nav_y': nav_y, 'target_center_world_xz': target_center_world_xz,
        'target_focus_world_xyz': target_focus_world_xyz, 'heading_focus_world_xyz': heading_focus_world_xyz,
        'target_semantic_ids': target_semantic_ids, 'p2w': p2w,
    }


def build_topdown_canvas(summary: dict, plan: dict | None = None) -> np.ndarray:
    candidates = []
    if isinstance(plan, dict):
        try:
            expanded_dir = plan.get('expanded_sample_dir')
            if expanded_dir:
                meta_path = Path(expanded_dir) / 'meta.json'
                if meta_path.exists():
                    expanded_meta = load_json(meta_path)
                    candidates.append(expanded_meta.get('artifacts', {}).get('local_topdown_rgb_png'))
                    candidates.append(expanded_meta.get('topdown', {}).get('source_floor_rgb_png'))
        except Exception:
            pass
    p = Path(summary.get('topdown_curve_png', ''))
    candidates.append(str(p) if p else '')
    for candidate in candidates:
        if not candidate:
            continue
        pp = Path(str(candidate))
        if not pp.exists():
            continue
        img = cv2.imread(str(pp), cv2.IMREAD_COLOR)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.full((520, 520, 3), 245, dtype=np.uint8)


def world_to_img(render_fp: Any, p2w: dict | None, x: float, z: float) -> tuple[int, int] | None:
    if not p2w:
        return None
    return render_fp.world_xz_to_image_xy(float(x), float(z), p2w)


def draw_plan_topdown(base_img: np.ndarray, render_fp: Any, p2w: dict | None,
                      base_xy: np.ndarray, cam_xy: np.ndarray, decorative_xy: np.ndarray,
                      target_uid: str) -> np.ndarray:
    img = np.asarray(base_img, dtype=np.uint8).copy()
    if p2w:
        pts_base = []
        pts_cam = []
        for p in base_xy:
            q = world_to_img(render_fp, p2w, float(p[0]), float(p[1]))
            if q: pts_base.append(q)
        for p in cam_xy:
            q = world_to_img(render_fp, p2w, float(p[0]), float(p[1]))
            if q: pts_cam.append(q)
        if len(pts_base) >= 2:
            cv2.polylines(img, [np.asarray(pts_base, dtype=np.int32)], False, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.polylines(img, [np.asarray(pts_base, dtype=np.int32)], False, (255, 80, 40), 2, cv2.LINE_AA)
        if len(pts_cam) >= 2:
            cv2.polylines(img, [np.asarray(pts_cam, dtype=np.int32)], False, (0, 255, 255), 2, cv2.LINE_AA)
        if pts_base:
            cv2.circle(img, pts_base[0], 6, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(img, pts_base[-1], 6, (255, 0, 0), -1, cv2.LINE_AA)
        if pts_cam:
            cv2.circle(img, pts_cam[-1], 7, (255, 0, 255), -1, cv2.LINE_AA)
    draw_text(img, 'blue/orange=base center, yellow=camera path', (12, 24), 0.48)
    draw_text(img, target_uid[:56], (12, 48), 0.42)
    return img


def variant_slug(variant: dict) -> str:
    variant_id = str(variant.get('variant_id', 'v00'))
    variant_source = str(variant.get('source', 'canonical_v2'))
    return ''.join(ch if ch.isalnum() or ch in {'_', '-'} else '_' for ch in f'{variant_id}__{variant_source}')


def variant_output_dir(out_root: Path, target_uid: str, variant: dict) -> Path:
    return Path(out_root) / target_uid.split('__', 1)[0] / target_uid / variant_slug(variant)


def target_angle_offset_stats(cam_xy: np.ndarray, theta: np.ndarray, target_xz: np.ndarray) -> dict:
    offsets = []
    for cam, th in zip(np.asarray(cam_xy, dtype=np.float32), np.asarray(theta, dtype=np.float32)):
        vec = np.asarray(target_xz, dtype=np.float32) - np.asarray(cam, dtype=np.float32)
        if float(np.linalg.norm(vec)) <= 1e-6:
            continue
        target_theta = math.atan2(float(vec[1]), float(vec[0]))
        offsets.append(abs(float(math.degrees(angle_diff_rad(float(th), target_theta)))))
    if not offsets:
        return {'max_deg': 0.0, 'median_deg': 0.0}
    return {'max_deg': float(max(offsets)), 'median_deg': float(np.median(offsets))}


def start_visibility_score(info: dict, original_yaw_deg: float) -> tuple:
    if not bool(info.get('target_visible', False)):
        return (0, 0.0, 0.0, -9999.0)
    u = float(info.get('target_center_u_ratio', 0.0) or 0.0)
    v = float(info.get('target_center_v_ratio', 0.0) or 0.0)
    center_dist = math.sqrt((u - 0.5) ** 2 + (v - 0.5) ** 2)
    yaw_penalty = yaw_diff_deg(float(info.get('yaw_deg', original_yaw_deg)), float(original_yaw_deg))
    return (
        1,
        float(info.get('target_area_ratio', 0.0) or 0.0),
        -center_dist,
        -yaw_penalty,
    )


def rgb_near_black_ratio(rgb: np.ndarray, *, max_value: int = 4, stride: int = 8) -> float:
    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] < 3:
        return 1.0
    sampled = image[::max(1, int(stride)), ::max(1, int(stride)), :3]
    return float(np.mean(np.all(sampled <= int(max_value), axis=2)))


def evaluate_start_heading_visibility(
    *,
    args,
    modules: dict,
    sim: Any,
    pf: Any,
    ctx: dict,
    obs_cam_xz: np.ndarray,
    nav_y: float,
    yaw_deg: float,
) -> dict:
    render_fp = modules['render_fp']
    base_mod = modules['base_mod']
    base_views = modules['base_views']
    cam_world = render_fp.precise_snap(pf, float(obs_cam_xz[0]), float(obs_cam_xz[1]), nav_y)
    if cam_world is None:
        cam_world = np.asarray([float(obs_cam_xz[0]), nav_y, float(obs_cam_xz[1])], dtype=np.float32)
    render_base_world, desired_eye_world = modules['curve'].compute_eye_locked_agent_base_world_xyz(
        nominal_base_world_xyz=np.asarray([float(cam_world[0]), float(nav_y), float(cam_world[2])], dtype=np.float32),
        yaw_deg=float(yaw_deg),
        pitch_down_deg=float(args.pitch_deg),
        sensor_height_m=float(args.camera_height_m),
    )
    obs = base_mod.render_first_person_observation(
        sim,
        position_world=np.asarray(render_base_world, dtype=np.float32),
        yaw_rad=math.radians(float(yaw_deg)),
        pitch_down_deg=float(args.pitch_deg),
    )
    near_black_ratio = rgb_near_black_ratio(obs['rgb'], max_value=4, stride=8)
    max_near_black_ratio = float(getattr(args, 'max_near_black_ratio', 0.40))
    semantic = np.asarray(obs['semantic'], dtype=np.int32)
    if semantic.ndim == 3 and semantic.shape[-1] == 1:
        semantic = semantic[..., 0]
    clean = None
    error = ''
    try:
        pref = render_fp.project_world_to_pixel_exact(obs['render_camera'], ctx['target_focus_world_xyz'], int(args.width), int(args.height))
        pref_xy = (float(pref['u']), float(pref['v'])) if bool(pref.get('valid', False)) else None
        clean = base_views.clean_target_component(
            semantic_obs=semantic,
            target_semantic_ids=ctx['target_semantic_ids'],
            preferred_center_xy=pref_xy,
            open_k=3,
            close_k=7,
            min_area_px=16,
        )
    except Exception as exc:
        error = repr(exc)
        clean = None
    bbox = [0, 0, 0, 0]
    center = [0, 0]
    area_px = 0
    area_ratio = 0.0
    u_ratio = 0.0
    v_ratio = 0.0
    central_30 = False
    if clean is not None:
        bbox = [int(v) for v in clean.get('bbox_xyxy', [0, 0, 0, 0])]
        center = [int(v) for v in clean.get('center_xy', [0, 0])]
        if len(center) != 2:
            center = [0, 0]
        area_px = int(clean.get('area_px', 0))
        area_ratio = float(area_px) / max(1.0, float(int(args.width) * int(args.height)))
        u_ratio = float(center[0]) / max(1.0, float(int(args.width) - 1))
        v_ratio = float(center[1]) / max(1.0, float(int(args.height) - 1))
        central_30 = 0.35 <= u_ratio <= 0.65 and 0.35 <= v_ratio <= 0.65
    return {
        'yaw_deg': float(yaw_deg) % 360.0,
        'target_visible': bool(clean is not None),
        'target_bbox_xyxy': bbox,
        'target_center_xy': center,
        'target_area_px': int(area_px),
        'target_area_ratio': float(area_ratio),
        'target_center_u_ratio': float(u_ratio),
        'target_center_v_ratio': float(v_ratio),
        'target_center_in_central_30pct': bool(central_30),
        'semantic_clean_error': error,
        'rgb_near_black_ratio': near_black_ratio,
        'rgb_reconstruction_threshold': {
            'near_black_rgb_max_value': 4,
            'sample_stride': 8,
            'max_near_black_ratio': max_near_black_ratio,
        },
        'eye_locked_render_base_world_xz': [float(render_base_world[0]), float(render_base_world[2])],
        'eye_locked_camera_eye_world_xz': [float(desired_eye_world[0]), float(desired_eye_world[2])],
    }


def tune_start_heading(
    *,
    args,
    modules: dict,
    sim: Any,
    pf: Any,
    ctx: dict,
    obs_cam_xz: np.ndarray,
    nav_y: float,
    start_yaw_deg: float,
    face_target_yaw_deg: float,
) -> dict:
    original = float(start_yaw_deg) % 360.0
    first = evaluate_start_heading_visibility(
        args=args, modules=modules, sim=sim, pf=pf, ctx=ctx,
        obs_cam_xz=obs_cam_xz, nav_y=nav_y, yaw_deg=original,
    )
    min_area = float(args.min_start_target_area_ratio)
    require_central = bool(args.require_start_target_central_30pct)
    def ok(info: dict) -> bool:
        return (
            bool(info.get('target_visible', False))
            and float(info.get('target_area_ratio', 0.0) or 0.0) >= min_area
            and (not require_central or bool(info.get('target_center_in_central_30pct', False)))
        )
    candidates = [first]
    if (not bool(args.autotune_start_heading)) or ok(first):
        first['source'] = 'original_start_yaw'
        return {
            'enabled': bool(args.autotune_start_heading),
            'adjusted': False,
            'original_yaw_deg': float(original),
            'selected_yaw_deg': float(original),
            'selected': first,
            'candidates_evaluated': candidates,
            'requirements': {
                'min_start_target_area_ratio': min_area,
                'require_start_target_central_30pct': require_central,
            },
        }

    offsets = [0.0]
    step = max(0.5, float(args.start_heading_search_step_deg))
    limit = max(0.0, float(args.start_heading_search_degrees))
    cur = step
    while cur <= limit + 1e-6:
        offsets.extend([cur, -cur])
        cur += step
    yaw_pool: list[float] = []
    seen: set[float] = set()
    for base in [original, float(face_target_yaw_deg) % 360.0]:
        for off in offsets:
            yaw = (base + off) % 360.0
            key = round(yaw, 3)
            if key in seen:
                continue
            seen.add(key)
            yaw_pool.append(yaw)
    for yaw in yaw_pool:
        if abs(yaw_diff_deg(yaw, original)) < 1e-6:
            continue
        info = evaluate_start_heading_visibility(
            args=args, modules=modules, sim=sim, pf=pf, ctx=ctx,
            obs_cam_xz=obs_cam_xz, nav_y=nav_y, yaw_deg=yaw,
        )
        info['source'] = 'autotune_search'
        candidates.append(info)
        if ok(info) and bool(args.stop_start_heading_search_on_first_ok):
            break
    valid = [c for c in candidates if ok(c)]
    selected = max(valid, key=lambda c: start_visibility_score(c, original)) if valid else max(candidates, key=lambda c: start_visibility_score(c, original))
    return {
        'enabled': True,
        'adjusted': bool(abs(yaw_diff_deg(float(selected['yaw_deg']), original)) > 1e-6),
        'original_yaw_deg': float(original),
        'face_target_yaw_deg': float(face_target_yaw_deg) % 360.0,
        'selected_yaw_deg': float(selected['yaw_deg']) % 360.0,
        'selected': selected,
        'candidates_evaluated': candidates,
        'requirements': {
            'search_degrees': float(args.start_heading_search_degrees),
            'search_step_deg': float(args.start_heading_search_step_deg),
            'min_start_target_area_ratio': min_area,
            'require_start_target_central_30pct': require_central,
        },
    }


def render_case(args, target_uid: str, modules: dict, variant: dict | None = None) -> dict:
    render_fp = modules['render_fp']
    base_mod = modules['base_mod']
    base_views = modules['base_views']
    ctx = load_case_context(target_uid, modules, args.width, args.height, args.hfov_deg, args.camera_height_m)
    sim = ctx['sim']
    pf = sim.pathfinder
    old = ctx['old_summary']
    nav_y = float(ctx['nav_y'])
    op_xz = np.asarray(old['operating_world_xz'], dtype=np.float32)
    obs_cam_xz = np.asarray(old['observer_world_xz'], dtype=np.float32)
    start_yaw_deg = float(old['start_yaw_deg'])
    variant = dict(variant or {'variant_id': 'v00', 'source': 'canonical_v2'})
    if 'camera_world_xz' in variant:
        obs_cam_xz = np.asarray(variant['camera_world_xz'], dtype=np.float32)
    if 'yaw_deg' in variant:
        start_yaw_deg = float(variant['yaw_deg'])
    # Goal camera is exactly over the operating point. Final yaw uses the
    # annotated operating-edge normal, with its sign selected by op->target.
    target_focus_world_xyz = np.asarray(old.get('target_focus_world_xyz') or ctx.get('target_focus_world_xyz'), dtype=np.float32)
    target_center_xz = np.asarray(old.get('target_center_world_xz') or ctx.get('target_center_world_xz') or [float(target_focus_world_xyz[0]), float(target_focus_world_xyz[2])], dtype=np.float32)
    target_focus_xz = target_center_xz
    center_yaw_deg = float(render_fp.yaw_between(
        np.asarray([float(op_xz[0]), float(nav_y), float(op_xz[1])], dtype=np.float32),
        np.asarray([float(target_center_xz[0]), float(nav_y), float(target_center_xz[1])], dtype=np.float32),
    ))
    center_theta = hm3d_yaw_deg_to_planar_theta(center_yaw_deg)
    edge_normal = compute_operating_edge_normal_theta(
        annotation_target=ctx['plan'].get('annotation_target', {}),
        p2w=ctx['p2w'],
        render_fp=render_fp,
        operating_world_xz=op_xz,
        target_center_world_xz=target_center_xz,
    )
    if edge_normal is None:
        goal_theta = float(center_theta)
        goal_yaw_deg = float(center_yaw_deg)
        final_heading_source = {
            'source': 'fallback_op_to_target_center',
            'reason': 'missing_or_invalid_operating_edge_line',
            'center_theta_rad': float(center_theta),
            'center_yaw_deg': float(center_yaw_deg),
        }
    else:
        goal_theta = float(edge_normal['theta_rad'])
        goal_yaw_deg = float(edge_normal['yaw_deg'])
        final_heading_source = {
            'source': 'operating_edge_line_normal',
            'edge_normal': edge_normal,
            'center_reference_theta_rad': float(center_theta),
            'center_reference_yaw_deg': float(center_yaw_deg),
            'normal_vs_op_to_target_delta_deg': float(math.degrees(angle_diff_rad(goal_theta, center_theta))),
        }
    face_target_yaw_deg = float(render_fp.yaw_between(
        np.asarray([float(obs_cam_xz[0]), float(nav_y), float(obs_cam_xz[1])], dtype=np.float32),
        np.asarray([float(target_center_xz[0]), float(nav_y), float(target_center_xz[1])], dtype=np.float32),
    ))
    start_heading_autotune = tune_start_heading(
        args=args,
        modules=modules,
        sim=sim,
        pf=pf,
        ctx=ctx,
        obs_cam_xz=obs_cam_xz,
        nav_y=nav_y,
        start_yaw_deg=float(start_yaw_deg),
        face_target_yaw_deg=float(face_target_yaw_deg),
    )
    start_yaw_deg = float(start_heading_autotune['selected_yaw_deg'])
    start_theta = hm3d_yaw_deg_to_planar_theta(start_yaw_deg)
    start_base_xy = xz_to_xy(obs_cam_xz) - float(args.camera_forward_offset_m) * np.asarray([math.cos(start_theta), math.sin(start_theta)], dtype=np.float32)
    goal_base_xy = xz_to_xy(op_xz) - float(args.camera_forward_offset_m) * np.asarray([math.cos(goal_theta), math.sin(goal_theta)], dtype=np.float32)
    face_target_theta = hm3d_yaw_deg_to_planar_theta(face_target_yaw_deg)
    commands_raw, base_xy_raw, theta_raw, meta = plan_poseaware_line_final_approach_commands(
        start_base_xy=start_base_xy,
        start_theta=start_theta,
        face_target_theta=face_target_theta,
        goal_camera_xz=xz_to_xy(op_xz),
        final_theta=goal_theta,
        camera_offset_m=float(args.camera_forward_offset_m),
        rate=args.rate,
        max_vx=args.max_vx,
        max_vyaw=args.max_vyaw,
        max_vx_accel=args.max_vx_accel,
        max_vyaw_accel=args.max_vyaw_accel,
        approach_candidates_m=tuple(float(v.strip()) for v in str(args.approach_candidates_m).split(',') if v.strip()),
        stop_tail=args.stop_tail,
    )
    commands = commands_raw.astype(np.float32).copy()
    # Re-integrate after cleaning so rendered poses and saved commands are identical.
    dt = 1.0 / float(args.rate)
    base_xy = np.zeros_like(base_xy_raw)
    theta = np.zeros_like(theta_raw)
    base_xy[0] = base_xy_raw[0]
    theta[0] = theta_raw[0]
    exact_twist_integration = bool(getattr(args, 'exact_constant_twist_integration', False))
    for ii in range(1, len(commands)):
        vx_i, vyaw_i = float(commands[ii, 0]), float(commands[ii, 1])
        th_i = float(theta[ii - 1])
        if exact_twist_integration:
            base_xy[ii], theta[ii] = integrate_constant_twist_step(
                base_xy[ii - 1], th_i, vx_i, vyaw_i, dt,
            )
        else:
            base_xy[ii, 0] = float(base_xy[ii - 1, 0]) + vx_i * math.cos(th_i) * dt
            base_xy[ii, 1] = float(base_xy[ii - 1, 1]) + vx_i * math.sin(th_i) * dt
            theta[ii] = wrap_rad(th_i + vyaw_i * dt)
    cam_xy = np.asarray([camera_xz_from_base_xy(base_xy[i], float(theta[i]), args.camera_forward_offset_m) for i in range(base_xy.shape[0])], dtype=np.float32)
    out_dir = variant_output_dir(Path(args.out_root), target_uid, variant)
    out_dir.mkdir(parents=True, exist_ok=True)
    segment_names = [str(row.get('segment', '')) for row in meta.get('segments', [])]
    pre_final_base_for_check = np.asarray(meta.get('pre_final_base_xy', [0.0, 0.0]), dtype=np.float32)
    first_drive_offsets = []
    for idx, name in enumerate(segment_names, start=1):
        if name == 'smooth_forward_to_pre_op_standoff' and idx < len(base_xy):
            vec = pre_final_base_for_check - np.asarray(base_xy[idx], dtype=np.float32)
            # Near the standoff point the residual vector is almost zero, so its
            # angle becomes numerically meaningless and can flip by 180 deg.
            if float(np.linalg.norm(vec)) > 0.03:
                first_drive_offsets.append(abs(float(math.degrees(angle_diff_rad(float(theta[idx]), math.atan2(float(vec[1]), float(vec[0])))))))
    geometry_checks = {
        'selected_approach_ok': bool(float(meta.get('approach_window_m', [0.0, 999.0])[0]) <= float(meta.get('selected_approach_m', 999.0)) <= float(meta.get('approach_window_m', [0.0, 999.0])[1])),
        'final_approach_lateral_ok': bool(abs(float(meta.get('final_approach_lateral_error_m', 999.0))) <= float(args.max_final_lateral_error_m)),
        'final_approach_forward_ok': bool(float(meta.get('final_approach_dist_m', -999.0)) >= 0.04),
        'first_drive_faces_pre_op_standoff_ok': bool((max(first_drive_offsets) if first_drive_offsets else 0.0) <= 1.0),
    }
    geometry_prefilter = {
        'pass': bool(all(geometry_checks.values())),
        'checks': geometry_checks,
        'target_angle_offset_all': target_angle_offset_stats(cam_xy, theta, target_center_xz),
        'first_drive_max_pre_op_standoff_angle_offset_deg': float(max(first_drive_offsets) if first_drive_offsets else 0.0),
        'thresholds': {
            'approach_window_m': [float(v) for v in meta.get('approach_window_m', [0.0, 999.0])],
            'max_final_lateral_error_m': float(args.max_final_lateral_error_m),
            'min_final_approach_forward_m': 0.04,
            'max_first_drive_pre_op_standoff_angle_offset_deg': 1.0,
        },
    }
    if bool(args.prefilter_geometry) and not geometry_prefilter['pass']:
        diagnostic_png = out_dir / 'staged_world_plan_diagnostic.png'
        diagnostic_png_written = False
        if bool(args.write_diagnostic_png):
            write_world_plan_diagnostic(
                diagnostic_png,
                base_xy=base_xy,
                cam_xy=cam_xy,
                op_xz=xz_to_xy(op_xz),
                target_xz=target_focus_xz,
                start_yaw_deg=float(start_yaw_deg),
                final_yaw_deg=float(goal_yaw_deg),
                target_uid=target_uid,
            )
            diagnostic_png_written = True
        summary = {
            'target_uid': target_uid,
            'variant': variant,
            'out_dir': str(out_dir),
            'error': 'geometry_prefilter_rejected_before_habitat_render',
            'planner_meta': meta,
            'geometry_prefilter': geometry_prefilter,
            'start_yaw_deg': float(start_yaw_deg),
            'face_semantic_target_yaw_deg': float(face_target_yaw_deg),
            'final_yaw_deg': float(goal_yaw_deg),
            'target_center_world_xz': [float(v) for v in target_center_xz.tolist()],
            'staged_world_plan_diagnostic_png': str(diagnostic_png) if diagnostic_png_written else '',
            'trajectory_filter': {'pass': False, 'checks': geometry_checks},
        }
        if not bool(getattr(args, 'defer_summary_write', False)):
            (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        sim.close()
        return summary
    tmp_fp = out_dir / 'ranger_fp_tmp_mp4v.mp4'
    tmp_dbg = out_dir / 'ranger_debug_tmp_mp4v.mp4'
    pitch_tag = f'{float(args.pitch_deg):04.1f}'.replace('.', 'p')
    fp_out = out_dir / f'ranger_fp_h122_pitch{pitch_tag}_848x480.mp4'
    aux_camera_specs = []
    aux_sim = None
    if bool(args.write_aux_front_cameras):
        for name, lateral_m in [('left', float(args.aux_camera_lateral_m)), ('right', -float(args.aux_camera_lateral_m))]:
            aux_camera_specs.append({
                'name': name,
                'lateral_m': lateral_m,
                'forward_m': float(args.aux_camera_forward_m),
                'height_m': float(args.aux_camera_height_m),
                'pitch_down_deg': float(args.aux_camera_pitch_deg),
                'height_errors_m': [],
                'planar_errors_m': [],
                'tmp': out_dir / f'ranger_{name}_front_tmp_mp4v.mp4',
                'out': out_dir / f'ranger_{name}_wrist_relmain_f010_l035_h100_pitch15_424x240.mp4',
            })
        scene_source = str(ctx['traj'].get('scene_path', '')).strip() or str(ctx['traj']['scene_id'])
        aux_sim = base_views.build_sim_for_target(
            base_mod=base_mod,
            traj=ctx['traj'],
            scene_source=scene_source,
            width=int(args.aux_camera_width),
            height=int(args.aux_camera_height),
            sensor_height_m=float(args.aux_camera_height_m),
            hfov_deg=float(args.aux_camera_hfov_deg),
        )
    dbg_out = out_dir / 'ranger_topdown_fp_actions.mp4'
    fp_writer = cv2.VideoWriter(str(tmp_fp), cv2.VideoWriter_fourcc(*'mp4v'), float(args.rate), (int(args.width), int(args.height)))
    for spec in aux_camera_specs:
        spec['writer'] = cv2.VideoWriter(str(spec['tmp']), cv2.VideoWriter_fourcc(*'mp4v'), float(args.rate), (int(args.aux_camera_width), int(args.aux_camera_height)))
    debug_writer = None
    if bool(args.write_debug_video):
        debug_writer = cv2.VideoWriter(str(tmp_dbg), cv2.VideoWriter_fourcc(*'mp4v'), float(args.rate), (int(args.height + 8 + args.width + 8 + 420), int(args.height)))
    diagnostic_png = out_dir / 'staged_world_plan_diagnostic.png'
    diagnostic_png_written = False
    if bool(args.write_diagnostic_png):
        write_world_plan_diagnostic(
            diagnostic_png,
            base_xy=base_xy,
            cam_xy=cam_xy,
            op_xz=xz_to_xy(op_xz),
            target_xz=target_focus_xz,
            start_yaw_deg=float(start_yaw_deg),
            final_yaw_deg=float(goal_yaw_deg),
            target_uid=target_uid,
        )
        diagnostic_png_written = True
    top = None
    if bool(args.write_debug_video):
        top = draw_plan_topdown(build_topdown_canvas(old, ctx['plan']), render_fp, ctx['p2w'], base_xy, cam_xy, base_xy, target_uid)
    frames = []
    for i in range(commands.shape[0]):
        segment_name = segment_name_for_frame(meta.get('segments', []), i)
        cam_xz = cam_xy[i]
        cam_world = render_fp.precise_snap(pf, float(cam_xz[0]), float(cam_xz[1]), nav_y)
        if cam_world is None:
            cam_world = np.asarray([float(cam_xz[0]), nav_y, float(cam_xz[1])], dtype=np.float32)
        cam_eye = np.asarray([float(cam_world[0]), float(nav_y) + float(args.camera_height_m), float(cam_world[2])], dtype=np.float32)
        yaw_deg = planar_theta_to_hm3d_yaw_deg(float(theta[i]))
        # Habitat rotates the sensor offset together with pitch. If we pass the
        # desired standing point directly as agent position, a pitched camera at
        # height 1.22m moves forward by about 1.22*sin(30deg)=0.61m. Use the
        # previously validated eye-locked compensation so the camera eye stays
        # exactly at the desired standing/camera point.
        render_base_world, desired_eye_world = modules['curve'].compute_eye_locked_agent_base_world_xyz(
            nominal_base_world_xyz=np.asarray([float(cam_world[0]), float(nav_y), float(cam_world[2])], dtype=np.float32),
            yaw_deg=float(yaw_deg),
            pitch_down_deg=float(args.pitch_deg),
            sensor_height_m=float(args.camera_height_m),
        )
        obs = base_mod.render_first_person_observation(
            sim,
            position_world=np.asarray(render_base_world, dtype=np.float32),
            yaw_rad=math.radians(float(yaw_deg)),
            pitch_down_deg=float(args.pitch_deg),
        )
        rgb = np.asarray(obs['rgb'], dtype=np.uint8)
        if rgb.ndim == 3 and rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        frame_near_black_ratio = rgb_near_black_ratio(rgb, max_value=4, stride=8)
        semantic = np.asarray(obs['semantic'], dtype=np.int32)
        if semantic.ndim == 3 and semantic.shape[-1] == 1:
            semantic = semantic[..., 0]
        render_camera = obs['render_camera']
        op_world = np.asarray([float(op_xz[0]), float(nav_y), float(op_xz[1])], dtype=np.float32)
        op_proj = render_fp.project_world_to_pixel_exact(render_camera, op_world, int(args.width), int(args.height))
        clean = None
        semantic_clean_error = ''
        try:
            pref = render_fp.project_world_to_pixel_exact(render_camera, ctx['target_focus_world_xyz'], int(args.width), int(args.height))
            pref_xy = (float(pref['u']), float(pref['v'])) if bool(pref.get('valid', False)) else None
            clean = base_views.clean_target_component(
                semantic_obs=semantic,
                target_semantic_ids=ctx['target_semantic_ids'],
                preferred_center_xy=pref_xy,
                open_k=3,
                close_k=7,
                min_area_px=16,
            )
        except Exception as exc:
            semantic_clean_error = repr(exc)
            clean = None
        fp_writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        forward = np.asarray([math.cos(float(theta[i])), math.sin(float(theta[i]))], dtype=np.float32)
        # HM3D XZ is left-handed as a 2D plotting plane here. Camera right is
        # (-forward_z, forward_x), so body-left is the opposite direction.
        left = np.asarray([forward[1], -forward[0]], dtype=np.float32)
        for spec in aux_camera_specs:
            # Extrinsics are relative to the current main-camera optical center,
            # not the robot base center.
            aux_xz = np.asarray(cam_xy[i], dtype=np.float32) + float(spec['forward_m']) * forward + float(spec['lateral_m']) * left
            aux_render_base, _ = modules['curve'].compute_eye_locked_agent_base_world_xyz(
                nominal_base_world_xyz=np.asarray([float(aux_xz[0]), float(nav_y), float(aux_xz[1])], dtype=np.float32),
                yaw_deg=float(yaw_deg),
                pitch_down_deg=float(spec['pitch_down_deg']),
                sensor_height_m=float(spec['height_m']),
            )
            aux_obs = base_mod.render_first_person_observation(
                aux_sim,
                position_world=np.asarray(aux_render_base, dtype=np.float32),
                yaw_rad=math.radians(float(yaw_deg)),
                pitch_down_deg=float(spec['pitch_down_deg']),
            )
            actual_t = aux_obs['render_camera'].node.absolute_translation
            actual_eye = np.asarray([float(actual_t[0]), float(actual_t[1]), float(actual_t[2])], dtype=np.float64)
            spec['height_errors_m'].append(float(actual_eye[1] - (float(nav_y) + float(spec['height_m']))))
            spec['planar_errors_m'].append(float(np.linalg.norm(actual_eye[[0, 2]] - np.asarray(aux_xz, dtype=np.float64))))
            aux_rgb = np.asarray(aux_obs['rgb'], dtype=np.uint8)
            if aux_rgb.ndim == 3 and aux_rgb.shape[-1] == 4:
                aux_rgb = aux_rgb[..., :3]
            spec['writer'].write(cv2.cvtColor(aux_rgb, cv2.COLOR_RGB2BGR))
        if debug_writer is not None and top is not None:
            overlay = rgb.copy()
            if clean is not None:
                x0, y0, x1, y1 = [int(v) for v in clean['bbox_xyxy']]
                cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (0, 220, 255), 2, cv2.LINE_AA)
            if bool(op_proj.get('visible_in_image', False)):
                cv2.drawMarker(overlay, (int(op_proj['u_draw']), int(op_proj['v_draw'])), (0, 255, 0), cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
            draw_text(overlay, f'f={i:03d} yaw={yaw_deg:.1f} pitch={args.pitch_deg:.1f}', (14, 28), 0.55)
            draw_text(overlay, f'vx={commands[i,0]:+.3f} vyaw={commands[i,1]:+.3f}', (14, 54), 0.55)
            top_frame = top.copy()
            if ctx['p2w']:
                q = world_to_img(render_fp, ctx['p2w'], float(base_xy[i,0]), float(base_xy[i,1]))
                cq = world_to_img(render_fp, ctx['p2w'], float(cam_xy[i,0]), float(cam_xy[i,1]))
                if q:
                    cv2.circle(top_frame, q, 6, (255, 0, 0), -1, cv2.LINE_AA)
                if cq:
                    cv2.circle(top_frame, cq, 5, (0, 255, 255), -1, cv2.LINE_AA)
            panel = np.zeros((int(args.height), int(args.height + 8 + args.width + 8 + 420), 3), dtype=np.uint8)
            panel[:, :int(args.height)] = fit_image(top_frame, int(args.height), int(args.height))
            panel[:, int(args.height)+8:int(args.height)+8+int(args.width)] = overlay
            panel[:, int(args.height)+8+int(args.width)+8:] = make_action_panel(commands, i, 420, int(args.height), args.max_vx, args.max_vyaw)
            debug_writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            if i in {0, commands.shape[0] // 2, commands.shape[0] - 1}:
                cv2.imwrite(str(out_dir / f'frame_{i:04d}_debug.png'), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        target_bbox = [0, 0, 0, 0] if clean is None else [int(v) for v in clean['bbox_xyxy']]
        target_center_xy = [0, 0]
        target_area_px = 0
        target_area_ratio = 0.0
        target_center_u_ratio = 0.0
        target_center_v_ratio = 0.0
        target_center_in_central_30pct = False
        if clean is not None:
            target_center_xy = [int(v) for v in clean.get('center_xy', [])]
            if len(target_center_xy) != 2:
                target_center_xy = [0, 0]
            target_area_px = int(clean.get('area_px', 0))
            target_area_ratio = float(target_area_px) / max(1.0, float(int(args.width) * int(args.height)))
            if len(target_center_xy) == 2:
                target_center_u_ratio = float(target_center_xy[0]) / max(1.0, float(int(args.width) - 1))
                target_center_v_ratio = float(target_center_xy[1]) / max(1.0, float(int(args.height) - 1))
                target_center_in_central_30pct = (
                    0.35 <= target_center_u_ratio <= 0.65
                    and 0.35 <= target_center_v_ratio <= 0.65
                )
        op_pixel_xy = [0, 0]
        op_u_ratio = 0.0
        op_v_ratio = 0.0
        op_is_visible = bool(op_proj.get('visible_in_image', False))
        if op_is_visible:
            op_u = float(op_proj.get('u_draw', op_proj.get('u', 0.0)))
            op_v = float(op_proj.get('v_draw', op_proj.get('v', 0.0)))
            op_pixel_xy = [int(round(op_u)), int(round(op_v))]
            op_u_ratio = float(op_u / max(1.0, float(int(args.width) - 1)))
            op_v_ratio = float(op_v / max(1.0, float(int(args.height) - 1)))
        frames.append({
            'idx': int(i), 't': float(i / float(args.rate)),
            'segment': segment_name,
            'base_x': float(base_xy[i,0]), 'base_y': float(base_xy[i,1]), 'base_theta_rad': float(theta[i]),
            'camera_x': float(cam_xy[i,0]), 'camera_y': float(cam_xy[i,1]),
            'camera_world_xz': [float(cam_xy[i,0]), float(cam_xy[i,1])],
            'eye_locked_render_base_world_xz': [float(render_base_world[0]), float(render_base_world[2])],
            'eye_locked_camera_eye_world_xz': [float(desired_eye_world[0]), float(desired_eye_world[2])],
            'hm3d_yaw_deg': float(yaw_deg),
            'vx': float(commands[i,0]), 'vyaw': float(commands[i,1]),
            'op_visible': bool(op_is_visible),
            'op_pixel_xy': op_pixel_xy,
            'op_u_ratio': op_u_ratio,
            'op_v_ratio': op_v_ratio,
            'target_visible': bool(clean is not None),
            'target_bbox_xyxy': target_bbox,
            'target_center_xy': target_center_xy,
            'target_area_px': int(target_area_px),
            'target_area_ratio': float(target_area_ratio),
            'target_center_u_ratio': target_center_u_ratio,
            'target_center_v_ratio': target_center_v_ratio,
            'target_center_in_central_30pct': bool(target_center_in_central_30pct),
            'semantic_clean_error': semantic_clean_error,
            'rgb_near_black_ratio': frame_near_black_ratio,
        })
    fp_writer.release()
    for spec in aux_camera_specs:
        spec['writer'].release()
    if aux_sim is not None:
        aux_sim.close()
    if debug_writer is not None:
        debug_writer.release()
    transcode_to_h264(tmp_fp, fp_out)
    for spec in aux_camera_specs:
        transcode_to_h264(spec['tmp'], spec['out'])
        spec['tmp'].unlink(missing_ok=True)
    if debug_writer is not None:
        transcode_to_h264(tmp_dbg, dbg_out)
    tmp_fp.unlink(missing_ok=True)
    tmp_dbg.unlink(missing_ok=True)
    write_action_artifacts = bool(getattr(args, 'write_action_artifacts', True))
    actions_path = out_dir / 'actions_vx_vyaw_30hz.npy'
    commands_path = out_dir / 'commands.csv'
    if write_action_artifacts:
        np.save(str(actions_path), commands.astype(np.float32))
    frame_annotations_jsonl = out_dir / 'frame_annotations.jsonl'
    with frame_annotations_jsonl.open('w', encoding='utf-8') as f:
        for row in frames:
            f.write(json.dumps({
                'idx': int(row['idx']),
                't': float(row['t']),
                'segment': str(row.get('segment') or ''),
                'target_visible': bool(row.get('target_visible', False)),
                'target_bbox_xyxy': [int(v) for v in (row.get('target_bbox_xyxy') or [0, 0, 0, 0])],
                'target_center_xy': [int(v) for v in (row.get('target_center_xy') or [0, 0])],
                'target_area_px': int(row.get('target_area_px', 0)),
                'target_area_ratio': float(row.get('target_area_ratio', 0.0)),
                'target_center_u_ratio': float(row.get('target_center_u_ratio', 0.0) or 0.0),
                'target_center_v_ratio': float(row.get('target_center_v_ratio', 0.0) or 0.0),
                'op_visible': bool(row.get('op_visible', False)),
                'op_pixel_xy': [int(v) for v in (row.get('op_pixel_xy') or [0, 0])],
                'op_u_ratio': float(row.get('op_u_ratio', 0.0) or 0.0),
                'op_v_ratio': float(row.get('op_v_ratio', 0.0) or 0.0),
                'vx': float(row.get('vx', 0.0)),
                'vyaw': float(row.get('vyaw', 0.0)),
                'camera_world_xz': [float(v) for v in (row.get('camera_world_xz') or [0.0, 0.0])],
                'hm3d_yaw_deg': float(row.get('hm3d_yaw_deg', 0.0)),
                'rgb_near_black_ratio': float(row.get('rgb_near_black_ratio', 1.0)),
            }, ensure_ascii=False) + '\n')
    if write_action_artifacts:
        with commands_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['idx','t','vx','vyaw','sim_x','sim_y','sim_theta_rad','camera_x','camera_y'])
            writer.writeheader()
            for i in range(commands.shape[0]):
                writer.writerow({'idx': i, 't': i / float(args.rate), 'vx': float(commands[i,0]), 'vyaw': float(commands[i,1]), 'sim_x': float(base_xy[i,0]), 'sim_y': float(base_xy[i,1]), 'sim_theta_rad': float(theta[i]), 'camera_x': float(cam_xy[i,0]), 'camera_y': float(cam_xy[i,1])})
    final_camera_error = float(np.linalg.norm(cam_xy[-1] - op_xz))
    visible_flags = [bool(f.get('target_visible', False)) for f in frames]
    visible_count = int(sum(1 for v in visible_flags if v))
    total_frames = max(1, int(len(frames)))
    early_n = max(1, int(math.ceil(total_frames * float(args.semantic_early_fraction))))
    early_visible_count = int(sum(1 for v in visible_flags[:early_n] if v))
    pre_final_frames = [
        f for f in frames
        if str(f.get('segment', '')) not in {'short_perpendicular_approach', 'stop_tail'}
    ]
    pre_final_visible_flags = [bool(f.get('target_visible', False)) for f in pre_final_frames]
    pre_final_visible_count = int(sum(1 for v in pre_final_visible_flags if v))
    pre_final_total = max(1, int(len(pre_final_frames)))
    mid_start = max(0, int(math.floor(pre_final_total * 0.25)))
    mid_end = max(mid_start + 1, int(math.ceil(pre_final_total * 0.75)))
    mid_visible_flags = pre_final_visible_flags[mid_start:mid_end]
    mid_visible_count = int(sum(1 for v in mid_visible_flags if v))
    rotate_frames = [f for f in frames if str(f.get('segment', '')) in {'rotate_to_pre_op_standoff', 'rotate_to_operating_edge_normal'}]
    rotate_visible_flags = [bool(f.get('target_visible', False)) for f in rotate_frames]
    rotate_visible_count = int(sum(1 for v in rotate_visible_flags if v))
    central_flags = [bool(f.get('target_center_in_central_30pct', False)) for f in frames]
    semantic_visibility = {
        'target_semantic_ids': [int(v) for v in ctx['target_semantic_ids']],
        'visible_count': int(visible_count),
        'total_frames': int(total_frames),
        'visible_ratio': float(visible_count / total_frames),
        'early_fraction': float(args.semantic_early_fraction),
        'early_frame_count': int(early_n),
        'early_visible_count': int(early_visible_count),
        'early_visible_ratio': float(early_visible_count / max(1, early_n)),
        'start_visible': bool(visible_flags[0]) if visible_flags else False,
        'pre_final_visible_count': int(pre_final_visible_count),
        'pre_final_frame_count': int(pre_final_total),
        'pre_final_visible_ratio': float(pre_final_visible_count / pre_final_total),
        'mid_pre_final_visible_count': int(mid_visible_count),
        'mid_pre_final_frame_count': int(max(1, len(mid_visible_flags))),
        'mid_pre_final_visible_ratio': float(mid_visible_count / max(1, len(mid_visible_flags))),
        'rotate_visible_count': int(rotate_visible_count),
        'rotate_frame_count': int(max(1, len(rotate_visible_flags))),
        'rotate_visible_ratio': float(rotate_visible_count / max(1, len(rotate_visible_flags))),
        'central_30pct_count': int(sum(1 for v in central_flags if v)),
        'central_30pct_ratio': float(sum(1 for v in central_flags if v) / total_frames),
        'max_target_area_ratio': float(max([float(f.get('target_area_ratio', 0.0) or 0.0) for f in frames] or [0.0])),
        'median_visible_target_area_ratio': float(np.median([float(f.get('target_area_ratio', 0.0) or 0.0) for f in frames if bool(f.get('target_visible', False))])) if visible_count > 0 else 0.0,
    }
    near_black_ratios = [float(f.get('rgb_near_black_ratio', 1.0)) for f in frames]
    max_near_black_ratio = float(getattr(args, 'max_near_black_ratio', 0.40))
    rgb_reconstruction_quality = {
        'near_black_rgb_max_value': 4,
        'sample_stride': 8,
        'max_allowed_near_black_ratio': max_near_black_ratio,
        'max_near_black_ratio': float(max(near_black_ratios or [1.0])),
        'median_near_black_ratio': float(np.median(near_black_ratios or [1.0])),
        'bad_frame_count': int(sum(value > max_near_black_ratio for value in near_black_ratios)),
        'frame_count': len(near_black_ratios),
        'pass': bool(near_black_ratios and max(near_black_ratios) <= max_near_black_ratio),
    }
    summary = {
        'target_uid': target_uid,
        'variant': variant,
        'out_dir': str(out_dir),
        'fp_video': str(fp_out),
        'aux_camera_videos': {str(spec['name']): str(spec['out']) for spec in aux_camera_specs},
        'debug_video': str(dbg_out) if debug_writer is not None else '',
        'frame_annotations_jsonl': str(frame_annotations_jsonl),
        'commands_csv': str(commands_path) if write_action_artifacts else '',
        'actions_vx_vyaw_npy': str(actions_path) if write_action_artifacts else '',
        'rate_hz': float(args.rate),
        'dt': float(1.0 / float(args.rate)),
        'action_order': ['vx_mps', 'vyaw_radps'],
        'vy': 0.0,
        'camera_model': {
            'height_m': float(args.camera_height_m), 'forward_offset_m': float(args.camera_forward_offset_m),
            'pitch_down_deg': float(args.pitch_deg), 'width': int(args.width), 'height': int(args.height),
            'fx': float(args.fx), 'fy': float(args.fy), 'cx': float(args.cx), 'cy': float(args.cy),
            'hfov_used_deg': float(args.hfov_deg),
        },
        'aux_camera_models': {
            str(spec['name']): {
                'height_m': float(spec['height_m']),
                'forward_m_from_main_camera': float(spec['forward_m']),
                'lateral_m_from_main_camera': float(spec['lateral_m']),
                'pitch_down_deg': float(spec['pitch_down_deg']),
                'width': int(args.aux_camera_width), 'height': int(args.aux_camera_height),
                'K': [[float(args.aux_camera_fx), 0.0, float(args.aux_camera_cx)],
                      [0.0, float(args.aux_camera_fy), float(args.aux_camera_cy)],
                      [0.0, 0.0, 1.0]],
                'hfov_used_deg': float(args.aux_camera_hfov_deg),
                'vfov_reference_deg': float(args.aux_camera_vfov_deg),
                'distortion': [0.0, 0.0, 0.0, 0.0, 0.0],
                'extrinsics_reference': 'current_main_camera_optical_center_body_frame',
                'eye_locked_pitch': True,
                'actual_optical_center_audit': {
                    'frame_count': int(len(spec['height_errors_m'])),
                    'max_abs_height_error_m': float(max([abs(v) for v in spec['height_errors_m']] or [0.0])),
                    'mean_abs_height_error_m': float(np.mean(np.abs(spec['height_errors_m']))) if spec['height_errors_m'] else 0.0,
                    'max_planar_position_error_m': float(max(spec['planar_errors_m'] or [0.0])),
                    'height_reference': 'nav_y + requested_camera_height_m',
                },
            } for spec in aux_camera_specs
        },
        'planner_params': {
            'mode': str(meta.get('mode', 'poseaware_exact_rotate_line_rotate_final_approach')),
            'max_vx': float(args.max_vx), 'max_vyaw': float(args.max_vyaw), 'max_vx_accel': float(args.max_vx_accel),
            'max_vyaw_accel': float(args.max_vyaw_accel),
            'max_allowed_dvx_per_frame': float(args.max_vx_accel / args.rate),
            'max_allowed_dvyaw_per_frame': float(args.max_vyaw_accel / args.rate),
            'curve_handle_scale': 0.0, 'curve_handle_max': 0.0,
            'curve_length_m': 0.0,
            'curve_handle_m': 0.0,
            'selected_approach_m': float(meta.get('selected_approach_m', 0.0)),
            'final_heading_source': final_heading_source,
            'face_target_yaw_deg': float(face_target_yaw_deg),
            'start_heading_autotune': start_heading_autotune,
            'stop_tail': float(args.stop_tail),
        },
        'start': {'camera_xz': [float(v) for v in obs_cam_xz.tolist()], 'base_xy': [float(v) for v in start_base_xy.tolist()], 'yaw_deg': float(start_yaw_deg), 'theta_rad': float(start_theta)},
        'goal': {'camera_xz': [float(v) for v in op_xz.tolist()], 'base_xy': [float(v) for v in goal_base_xy.tolist()], 'yaw_deg': float(goal_yaw_deg), 'theta_rad': float(goal_theta)},
        'target_center_world_xz': [float(v) for v in target_center_xz.tolist()],
        'target_focus_world_xyz': [float(v) for v in target_focus_world_xyz.tolist()],
        'staged_plan_meta': meta,
        'staged_world_plan_diagnostic_png': str(diagnostic_png) if diagnostic_png_written else '',
        'frame_count': int(commands.shape[0]), 'duration_s': float((commands.shape[0]-1)/float(args.rate)),
        'max_abs_vx': float(np.max(np.abs(commands[:,0]))), 'max_abs_vyaw': float(np.max(np.abs(commands[:,1]))),
        'action_smoothness': command_smoothness_info(commands),
        'tail_cleaning': {'enabled': False, 'note': 'pose-aware generator emits explicit zero stop_tail; saved poses are re-integrated from saved commands'},
        'final_camera_to_operating_error_m': final_camera_error,
        'geometry_prefilter': geometry_prefilter,
        'semantic_visibility': semantic_visibility,
        'rgb_reconstruction_quality': rgb_reconstruction_quality,
        'frames': frames,
    }
    smooth = summary['action_smoothness']
    approach_window = meta.get('approach_window_m', [0.0, 999.0])
    selected_approach = float(meta.get('selected_approach_m', 999.0))
    checks = {
        'final_camera_error_ok': bool(final_camera_error <= float(args.max_final_error_m)),
        'final_approach_lateral_ok': bool(abs(float(meta.get('final_approach_lateral_error_m', 999.0))) <= float(args.max_final_lateral_error_m)),
        'approach_distance_ok': bool(float(approach_window[0]) <= selected_approach <= float(approach_window[1])),
        'vx_nonnegative_ok': bool(float(np.min(commands[:, 0])) >= -1e-6),
        'vx_limit_ok': bool(float(np.max(commands[:, 0])) <= float(args.max_vx) + 1e-6),
        'vyaw_limit_ok': bool(float(np.max(np.abs(commands[:, 1]))) <= float(args.max_vyaw) + 1e-6),
        'dvx_accel_ok': bool(float(smooth['max_abs_dvx_per_frame']) <= float(args.max_vx_accel / args.rate) + 1e-5),
        'dvyaw_accel_ok': bool(float(smooth['max_abs_dvyaw_per_frame']) <= float(args.max_vyaw_accel / args.rate) + 1e-5),
        'vyaw_sign_flip_ok': bool(int(smooth['vyaw_small_sign_flip_count']) == 0 and int(smooth['vyaw_large_sign_flip_count']) == 0),
        'vy_zero_ok': True,
        'semantic_start_visible_ok': bool(semantic_visibility['start_visible']),
        'semantic_visible_ratio_ok': bool(float(semantic_visibility['visible_ratio']) >= float(args.min_semantic_visible_ratio)),
        'semantic_early_visible_ratio_ok': bool(float(semantic_visibility['early_visible_ratio']) >= float(args.min_semantic_early_visible_ratio)),
        'semantic_pre_final_visible_ratio_ok': bool(float(semantic_visibility['pre_final_visible_ratio']) >= float(args.min_semantic_pre_final_visible_ratio)),
        'semantic_mid_pre_final_visible_ratio_ok': bool(float(semantic_visibility['mid_pre_final_visible_ratio']) >= float(args.min_semantic_mid_visible_ratio)),
    }
    summary['trajectory_filter'] = {
        'pass': bool(all(checks.values())),
        'checks': checks,
        'thresholds': {
            'max_final_error_m': float(args.max_final_error_m),
            'max_final_lateral_error_m': float(args.max_final_lateral_error_m),
            'max_abs_dvx_per_frame': float(args.max_vx_accel / args.rate),
            'max_abs_dvyaw_per_frame': float(args.max_vyaw_accel / args.rate),
            'min_semantic_visible_ratio': float(args.min_semantic_visible_ratio),
            'semantic_early_fraction': float(args.semantic_early_fraction),
            'min_semantic_early_visible_ratio': float(args.min_semantic_early_visible_ratio),
            'min_semantic_pre_final_visible_ratio': float(args.min_semantic_pre_final_visible_ratio),
            'min_semantic_mid_visible_ratio': float(args.min_semantic_mid_visible_ratio),
            'approach_window_m': [float(approach_window[0]), float(approach_window[1])],
        },
    }
    if not bool(getattr(args, 'defer_summary_write', False)):
        (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    sim.close()
    return summary


def baseline_v2_summary_path(target_uid: str) -> Path:
    return BASELINE_V2_ROOT / target_uid.split('__', 1)[0] / target_uid / 'summary.json'


def load_api_accepted_uids() -> list[str]:
    uids: list[str] = []
    for path in sorted(API_ACCEPTED_RESULTS.glob('*.json')):
        try:
            row = load_json(path)
        except Exception:
            continue
        if row.get('keep') is True and row.get('target_uid'):
            uids.append(str(row['target_uid']))
    return uids


def yaw_diff_deg(a: float, b: float) -> float:
    return abs(math.degrees(angle_diff_rad(math.radians(float(a)), math.radians(float(b)))))


def is_distinct_start(candidate: dict, selected: list[dict], *, min_pos_m: float = 0.20, min_yaw_deg: float = 12.0) -> bool:
    cand_xz = np.asarray(candidate['camera_world_xz'], dtype=np.float32)
    cand_yaw = float(candidate['yaw_deg'])
    for prev in selected:
        prev_xz = np.asarray(prev.get('camera_world_xz', prev.get('start_camera_xz', [9999.0, 9999.0])), dtype=np.float32)
        prev_yaw = float(prev.get('yaw_deg', prev.get('start_yaw_deg', 9999.0)))
        if float(np.linalg.norm(cand_xz - prev_xz)) < float(min_pos_m) and yaw_diff_deg(cand_yaw, prev_yaw) < float(min_yaw_deg):
            return False
    return True


def load_multiview_index() -> dict[str, list[dict]]:
    global _MULTIVIEW_INDEX
    if _MULTIVIEW_INDEX is not None:
        return _MULTIVIEW_INDEX
    index: dict[str, list[dict]] = {}
    seen_by_uid: dict[str, set[tuple[float, float, float]]] = {}
    for root in MULTIVIEW_ROOTS:
        logs = root / 'logs'
        if not logs.exists():
            continue
        for path in sorted(logs.glob('summary_shard_*_of_7.json')):
            try:
                data = load_json(path)
            except Exception:
                continue
            for row in data.get('rows', []):
                target_uid = row.get('target_uid')
                if not target_uid:
                    continue
                for view in row.get('views', []):
                    xz = view.get('camera_world_xz')
                    if not (isinstance(xz, list) and len(xz) == 2):
                        continue
                    if view.get('yaw_deg') is None:
                        continue
                    if not bool(view.get('camera_op_navline_ok', False)):
                        continue
                    if not bool(view.get('target_bbox_complete', False)):
                        continue
                    if not bool(view.get('operating_visible', False)):
                        continue
                    if int(view.get('target_area_px') or 0) < 1000:
                        continue
                    yaw = float(view['yaw_deg']) % 360.0
                    key = (round(float(xz[0]), 3), round(float(xz[1]), 3), round(yaw, 1))
                    seen = seen_by_uid.setdefault(str(target_uid), set())
                    if key in seen:
                        continue
                    seen.add(key)
                    path_t = float(view.get('path_t') or 0.0)
                    index.setdefault(str(target_uid), []).append({
                        'source': 'multiview_start',
                        'camera_world_xz': [float(xz[0]), float(xz[1])],
                        'yaw_deg': yaw,
                        'source_root': str(root),
                        'source_summary_json': str(path),
                        'source_view_index': int(view.get('view_index', -1)),
                        'source_rgb_png': str(view.get('rgb_png') or view.get('image_path') or ''),
                        'path_t': path_t,
                        'distance_from_operating_m': float(view.get('distance_from_operating_m') or 0.0),
                        'target_area_px': int(view.get('target_area_px') or 0),
                        'camera_op_navline_ok': bool(view.get('camera_op_navline_ok', False)),
                        'target_bbox_complete': bool(view.get('target_bbox_complete', False)),
                        'operating_visible': bool(view.get('operating_visible', False)),
                    })
    _MULTIVIEW_INDEX = index
    return index


def load_multiview_start_candidates(target_uid: str, canonical: dict) -> list[dict]:
    raw = [dict(v) for v in load_multiview_index().get(str(target_uid), [])]
    candidates: list[dict] = []
    canonical_xz = np.asarray(canonical['camera_world_xz'], dtype=np.float32)
    for cand in raw:
        xz = np.asarray(cand['camera_world_xz'], dtype=np.float32)
        dist_to_canonical = float(np.linalg.norm(xz - canonical_xz))
        path_t = float(cand.get('path_t') or 0.0)
        cand['_score'] = (
            (2.0 if path_t > 0.05 else 0.0)
            + min(1.0, dist_to_canonical)
            + min(1.0, float(cand.get('target_area_px') or 0) / 80000.0)
        )
        candidates.append(cand)
    candidates.sort(key=lambda v: (-float(v.get('_score', 0.0)), str(v.get('source_root', '')), int(v.get('source_view_index', 9999))))
    for cand in candidates:
        cand.pop('_score', None)
    return candidates


def compute_face_target_yaw_from_xz(camera_world_xz: np.ndarray, target_center_world_xz: np.ndarray) -> float:
    dx = float(target_center_world_xz[0]) - float(camera_world_xz[0])
    dz = float(target_center_world_xz[1]) - float(camera_world_xz[1])
    return float((-math.degrees(math.atan2(dx, -dz))) % 360.0)


def load_target_center_from_baseline(target_uid: str) -> np.ndarray:
    summary = load_json(baseline_v2_summary_path(target_uid))
    center = summary.get('target_center_world_xz')
    if isinstance(center, list) and len(center) == 2:
        return np.asarray(center, dtype=np.float32)
    return np.asarray(summary['goal']['camera_xz'], dtype=np.float32)


def load_target_variant_context(target_uid: str) -> dict:
    baseline_path = baseline_v2_summary_path(target_uid)
    if baseline_path.exists():
        summary = load_json(baseline_path)
        center = summary.get('target_center_world_xz')
        if not (isinstance(center, list) and len(center) == 2):
            center = summary['goal']['camera_xz']
        return {
            'canonical_camera_world_xz': [float(v) for v in summary['start']['camera_xz']],
            'source_yaw_deg': float(summary['start']['yaw_deg']),
            'target_center_world_xz': [float(center[0]), float(center[1])],
            'source_summary_json': str(baseline_path),
            'source': 'baseline_v2_summary',
        }

    manifest = load_json(STAGE3_ROOT / 'final_observations' / f'{target_uid}__manifest.json')
    plan = load_json(STAGE3_ROOT / 'plan' / target_uid / 'result.json')
    expanded_meta = load_json(Path(plan['expanded_sample_dir']) / 'meta.json')
    stage_meta = load_json(Path(expanded_meta['source_meta_json']))
    target_row = stage_meta['targets'][int(plan['target_index'])]
    observer_xz = manifest['observer_point_used']['observer_world_xz']
    target_center = target_row.get('topdown_center_world_xz') or plan.get('operating_point_world_xz')
    return {
        'canonical_camera_world_xz': [float(observer_xz[0]), float(observer_xz[1])],
        'source_yaw_deg': float(manifest.get('fp_yaw_deg', 0.0)),
        'target_center_world_xz': [float(target_center[0]), float(target_center[1])],
        'source_manifest_json': str(STAGE3_ROOT / 'final_observations' / f'{target_uid}__manifest.json'),
        'source_plan_json': str(STAGE3_ROOT / 'plan' / target_uid / 'result.json'),
        'source': 'api_stage3_manifest_plan',
    }


def load_midpoint_yaw_candidates(target_uid: str, target_center_world_xz: np.ndarray) -> list[dict]:
    path = baseline_v2_summary_path(target_uid)
    if not path.exists():
        return []
    summary = load_json(path)
    frames = summary.get('frames') or []
    if len(frames) < 3:
        return []
    pts = []
    for frame in frames:
        xz = frame.get('camera_world_xz')
        if isinstance(xz, list) and len(xz) == 2:
            pts.append([float(xz[0]), float(xz[1])])
        else:
            pts.append([float(frame['camera_x']), float(frame['camera_y'])])
    pts_np = np.asarray(pts, dtype=np.float32)
    seg = np.linalg.norm(pts_np[1:] - pts_np[:-1], axis=1)
    cum = np.concatenate([np.asarray([0.0], dtype=np.float32), np.cumsum(seg, dtype=np.float32)])
    total = float(cum[-1])
    if total <= 1e-5:
        return []
    goal = np.asarray(summary['goal']['camera_xz'], dtype=np.float32)
    out: list[dict] = []
    specs = [0.18, 0.25, 0.30, 0.36, 0.42, 0.48, 0.55, 0.62, 0.68, 0.74, 0.80]
    used_idx: set[int] = set()
    for frac in specs:
        target_s = float(frac) * total
        idx = int(np.searchsorted(cum, target_s, side='left'))
        idx = max(1, min(idx, len(frames) - 2))
        if idx in used_idx:
            continue
        used_idx.add(idx)
        frame = frames[idx]
        xz = pts_np[idx]
        if float(np.linalg.norm(xz - goal)) < 0.45:
            continue
        base_yaw = float(frame.get('hm3d_yaw_deg', summary['start']['yaw_deg']))
        face_yaw = compute_face_target_yaw_from_xz(xz, np.asarray(target_center_world_xz, dtype=np.float32))
        out.append({
            'source': 'midpoint_yaw_replan',
            'camera_world_xz': [float(xz[0]), float(xz[1])],
            'yaw_deg': float(face_yaw),
            'source_baseline_summary_json': str(path),
            'source_frame_idx': int(idx),
            'source_frame_t': float(frame.get('t', idx / float(summary.get('rate_hz', 30.0)))),
            'source_frame_yaw_deg': base_yaw,
            'yaw_policy': 'recomputed_face_semantic_target',
            'path_fraction_by_distance': float(cum[idx] / total),
        })
    return out


def build_variants_for_target(target_uid: str, variants_per_target: int) -> list[dict]:
    ctx = load_target_variant_context(target_uid)
    target_center_world_xz = np.asarray(ctx['target_center_world_xz'], dtype=np.float32)
    canonical = {
        'camera_world_xz': ctx['canonical_camera_world_xz'],
        'yaw_deg': compute_face_target_yaw_from_xz(
            np.asarray(ctx['canonical_camera_world_xz'], dtype=np.float32),
            target_center_world_xz,
        ),
    }
    variants: list[dict] = [{
        'variant_id': 'v00',
        'source': 'canonical_v2_face_semantic_target',
        'camera_world_xz': [float(v) for v in canonical['camera_world_xz']],
        'yaw_deg': float(canonical['yaw_deg']),
        'source_yaw_deg': float(ctx['source_yaw_deg']),
        'yaw_policy': 'recomputed_face_semantic_target',
        'variant_context_source': str(ctx.get('source', '')),
        'source_baseline_summary_json': str(ctx.get('source_summary_json', '')),
        'source_manifest_json': str(ctx.get('source_manifest_json', '')),
        'source_plan_json': str(ctx.get('source_plan_json', '')),
    }]
    selected_for_distinct = [canonical]

    target_count = int(variants_per_target)

    for cand in load_multiview_start_candidates(target_uid, canonical):
        if len(variants) >= target_count:
            break
        if not is_distinct_start(cand, selected_for_distinct):
            continue
        cand = dict(cand)
        cand['variant_id'] = f'v{len(variants):02d}'
        cand['source_yaw_deg'] = float(cand['yaw_deg'])
        cand['yaw_deg'] = compute_face_target_yaw_from_xz(np.asarray(cand['camera_world_xz'], dtype=np.float32), target_center_world_xz)
        cand['yaw_policy'] = 'recomputed_face_semantic_target'
        variants.append(cand)
        selected_for_distinct.append(cand)

    for cand in load_midpoint_yaw_candidates(target_uid, target_center_world_xz):
        if len(variants) >= target_count:
            break
        if not is_distinct_start(cand, selected_for_distinct, min_pos_m=0.08, min_yaw_deg=20.0):
            continue
        cand = dict(cand)
        cand['variant_id'] = f'v{len(variants):02d}'
        variants.append(cand)
        selected_for_distinct.append(cand)

    # Last-resort fill keeps the pilot at a stable count without changing the planner.
    midpoint_pool = load_midpoint_yaw_candidates(target_uid, target_center_world_xz)
    while len(variants) < target_count and midpoint_pool:
        cand = dict(midpoint_pool[(len(variants) - 1) % len(midpoint_pool)])
        cand['variant_id'] = f'v{len(variants):02d}'
        cand['source'] = 'midpoint_yaw_replan_fill'
        cand['yaw_policy'] = 'recomputed_face_semantic_target_fill'
        variants.append(cand)
    return variants[:target_count]


def apply_shard(uids: list[str], shard_index: int, num_shards: int) -> list[str]:
    num_shards = int(num_shards)
    shard_index = int(shard_index)
    if num_shards <= 1:
        return list(uids)
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f'shard_index must be in [0, {num_shards - 1}], got {shard_index}')
    return [uid for i, uid in enumerate(uids) if i % num_shards == shard_index]


def compact_summary(row: dict) -> dict:
    if not isinstance(row, dict):
        return {'error': 'non_dict_summary', 'value': repr(row)}
    keys = [
        'target_uid', 'variant', 'out_dir', 'error', 'fp_video', 'aux_camera_videos', 'debug_video',
        'frame_annotations_jsonl', 'commands_csv', 'actions_vx_vyaw_npy',
        'rate_hz', 'dt', 'frame_count', 'duration_s', 'action_order', 'vy',
        'camera_model', 'aux_camera_models', 'planner_params', 'start', 'goal',
        'target_center_world_xz', 'target_focus_world_xyz',
        'final_camera_to_operating_error_m', 'geometry_prefilter',
        'semantic_visibility', 'trajectory_filter',
        'staged_world_plan_diagnostic_png',
    ]
    out = {k: row[k] for k in keys if k in row}
    variant = row.get('variant') if isinstance(row.get('variant'), dict) else {}
    out['variant_id'] = str(variant.get('variant_id', ''))
    out['variant_source'] = str(variant.get('source', ''))
    if 'staged_plan_meta' in row and isinstance(row['staged_plan_meta'], dict):
        meta = row['staged_plan_meta']
        out['staged_plan_meta_compact'] = {
            'mode': str(meta.get('mode', '')),
            'selected_approach_m': float(meta.get('selected_approach_m', 0.0) or 0.0),
            'first_line_distance_m': float(meta.get('first_line_distance_m', 0.0) or 0.0),
            'final_approach_dist_m': float(meta.get('final_approach_dist_m', 0.0) or 0.0),
            'final_approach_lateral_error_m': float(meta.get('final_approach_lateral_error_m', 0.0) or 0.0),
        }
    return out


def output_summary_path(args) -> Path:
    if int(args.num_shards) > 1:
        return Path(args.out_root) / f'summary_shard_{int(args.shard_index):03d}_of_{int(args.num_shards):03d}.json'
    target_variant_pairs_file = Path(args.target_variant_pairs_file)
    if str(target_variant_pairs_file) not in {'', '.'} and target_variant_pairs_file.name.startswith('worker_'):
        return Path(args.out_root) / f'summary_{target_variant_pairs_file.stem}.json'
    target_uids_file = Path(args.target_uids_file)
    if str(target_uids_file) not in {'', '.'} and target_uids_file.name.startswith('worker_'):
        return Path(args.out_root) / f'summary_{target_uids_file.stem}.json'
    return Path(args.out_root) / 'summary.json'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-root', type=Path, default=Path('outputs/ranger_probe'))
    p.add_argument('--case-count', type=int, default=0)
    p.add_argument('--max-targets', type=int, default=0, help='Optional global target limit; 0 means no explicit max.')
    p.add_argument('--target-uids', type=str, default='')
    p.add_argument('--target-uids-file', type=Path, default=Path(''))
    p.add_argument('--target-variant-pairs-file', type=Path, default=Path(''), help='Optional lines of: target_uid variant_id.')
    p.add_argument('--use-api-accepted', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--min-multiview-candidates', type=int, default=0)
    p.add_argument('--require-full-variant-count', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--skip-existing', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--checkpoint-every', type=int, default=10)
    p.add_argument('--rate', type=float, default=30.0)
    p.add_argument('--max-vx', type=float, default=0.120)
    p.add_argument('--max-vyaw', type=float, default=0.180)
    p.add_argument('--max-vx-accel', type=float, default=0.220)
    p.add_argument('--max-vyaw-accel', type=float, default=0.600)
    p.add_argument('--approach-candidates-m', type=str, default='0.2,0.12')
    p.add_argument('--curve-handle-scale', type=float, default=0.65)
    p.add_argument('--curve-handle-max', type=float, default=0.55)
    p.add_argument('--path-samples', type=int, default=240)
    p.add_argument('--retime-decel-margin', type=float, default=0.010)
    p.add_argument('--stop-tail', type=float, default=0.25)
    p.add_argument('--max-final-error-m', type=float, default=0.03)
    p.add_argument('--max-final-lateral-error-m', type=float, default=0.03)
    p.add_argument('--min-semantic-visible-ratio', type=float, default=0.30)
    p.add_argument('--semantic-early-fraction', type=float, default=0.30)
    p.add_argument('--min-semantic-early-visible-ratio', type=float, default=0.80)
    p.add_argument('--min-semantic-pre-final-visible-ratio', type=float, default=0.40)
    p.add_argument('--min-semantic-mid-visible-ratio', type=float, default=0.40)
    p.add_argument('--autotune-start-heading', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--start-heading-search-degrees', type=float, default=35.0)
    p.add_argument('--start-heading-search-step-deg', type=float, default=5.0)
    p.add_argument('--stop-start-heading-search-on-first-ok', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--min-start-target-area-ratio', type=float, default=0.0)
    p.add_argument('--require-start-target-central-30pct', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--camera-height-m', type=float, default=1.22)
    p.add_argument('--camera-forward-offset-m', type=float, default=0.26)
    p.add_argument('--pitch-deg', type=float, default=22.0)
    p.add_argument('--write-aux-front-cameras', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--aux-camera-height-m', type=float, default=1.0)
    p.add_argument('--aux-camera-forward-m', type=float, default=0.10)
    p.add_argument('--aux-camera-lateral-m', type=float, default=0.35)
    p.add_argument('--aux-camera-pitch-deg', type=float, default=15.0)
    p.add_argument('--aux-camera-width', type=int, default=424)
    p.add_argument('--aux-camera-height', type=int, default=240)
    p.add_argument('--aux-camera-fx', type=float, default=301.3548)
    p.add_argument('--aux-camera-fy', type=float, default=301.2285)
    p.add_argument('--aux-camera-cx', type=float, default=215.6338)
    p.add_argument('--aux-camera-cy', type=float, default=125.8129)
    p.add_argument('--aux-camera-hfov-deg', type=float, default=70.25)
    p.add_argument('--aux-camera-vfov-deg', type=float, default=43.44)
    p.add_argument('--width', type=int, default=848)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--fx', type=float, default=428.101715)
    p.add_argument('--fy', type=float, default=427.644989)
    p.add_argument('--cx', type=float, default=423.457764)
    p.add_argument('--cy', type=float, default=236.585236)
    p.add_argument('--hfov-deg', type=float, default=89.451536)
    p.add_argument('--variants-per-target', type=int, default=1)
    p.add_argument('--candidate-variants-per-target', type=int, default=0)
    p.add_argument('--stop-after-accepted-per-target', type=int, default=0)
    p.add_argument('--variant-ids', type=str, default='', help='Optional comma-separated variant ids to render, e.g. v01,v04.')
    p.add_argument('--prefilter-geometry', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--write-debug-video', action='store_true', default=False)
    p.add_argument('--write-diagnostic-png', action='store_true', default=False)
    args = p.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    target_uids_file = Path(args.target_uids_file)
    target_variant_pairs_file = Path(args.target_variant_pairs_file)
    has_target_variant_pairs_file = (
        str(target_variant_pairs_file) not in {'', '.'} and target_variant_pairs_file.is_file()
    )
    target_variant_ids: dict[str, set[str]] = {}
    if has_target_variant_pairs_file:
        for line in target_variant_pairs_file.read_text(encoding='utf-8').splitlines():
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            target_variant_ids.setdefault(parts[0], set()).add(parts[1])
    has_target_uids_file = str(target_uids_file) not in {'', '.'} and target_uids_file.is_file()
    explicit_uids = bool(args.target_uids.strip()) or has_target_uids_file or has_target_variant_pairs_file
    if has_target_variant_pairs_file:
        all_uids = sorted(target_variant_ids)
        uid_source = str(target_variant_pairs_file)
    elif has_target_uids_file:
        all_uids = [line.strip() for line in Path(args.target_uids_file).read_text(encoding='utf-8').splitlines() if line.strip()]
        uid_source = str(args.target_uids_file)
    elif args.target_uids.strip():
        all_uids = [x.strip() for x in args.target_uids.split(',') if x.strip()]
        uid_source = 'target_uids_arg'
    elif bool(args.use_api_accepted):
        all_uids = load_api_accepted_uids()
        uid_source = str(API_ACCEPTED_RESULTS)
    else:
        rows = load_json(OLD_BATCH_SUMMARY).get('rows', [])
        all_uids = [r['target_uid'] for r in rows]
        uid_source = str(OLD_BATCH_SUMMARY)

    raw_uid_count = int(len(all_uids))
    excluded_target_uids = load_excluded_target_uids()
    excluded_uid_count = sum(uid in excluded_target_uids for uid in all_uids)
    all_uids = [uid for uid in all_uids if uid not in excluded_target_uids]
    if int(args.min_multiview_candidates) > 0:
        mv_index = load_multiview_index()
        all_uids = [uid for uid in all_uids if len(mv_index.get(uid, [])) >= int(args.min_multiview_candidates)]
    after_multiview_filter_count = int(len(all_uids))

    if int(args.max_targets) > 0:
        all_uids = all_uids[:int(args.max_targets)]
    elif (not explicit_uids) and int(args.case_count) > 0:
        all_uids = all_uids[:int(args.case_count)]

    pre_shard_uid_count = int(len(all_uids))
    uids = apply_shard(all_uids, int(args.shard_index), int(args.num_shards))
    summaries = []
    per_target_counts: dict[str, dict] = {}
    summary_path = output_summary_path(args)

    def write_batch_summary() -> None:
        accepted = [row for row in summaries if isinstance(row, dict) and row.get('trajectory_filter', {}).get('pass') is True]
        rejected = [row for row in summaries if not (isinstance(row, dict) and row.get('trajectory_filter', {}).get('pass') is True)]
        out = {
            'out_root': str(args.out_root),
            'summary_json': str(summary_path),
            'uid_source': uid_source,
            'raw_uid_count': int(raw_uid_count),
            'excluded_uid_count': int(excluded_uid_count),
            'excluded_target_uids_file': str(EXCLUDED_TARGET_UIDS_FILE),
            'after_multiview_filter_count': int(after_multiview_filter_count),
            'pre_shard_uid_count': int(pre_shard_uid_count),
            'target_count': int(len(uids)),
            'shard_index': int(args.shard_index),
            'num_shards': int(args.num_shards),
            'variants_per_target': int(args.variants_per_target),
            'candidate_variants_per_target': int(args.candidate_variants_per_target),
            'stop_after_accepted_per_target': int(args.stop_after_accepted_per_target),
            'min_multiview_candidates': int(args.min_multiview_candidates),
            'require_full_variant_count': bool(args.require_full_variant_count),
            'skip_existing': bool(args.skip_existing),
            'write_debug_video': bool(args.write_debug_video),
            'write_diagnostic_png': bool(args.write_diagnostic_png),
            'camera_model': {
                'height_m': float(args.camera_height_m),
                'forward_offset_m': float(args.camera_forward_offset_m),
                'pitch_down_deg': float(args.pitch_deg),
                'width': int(args.width),
                'height': int(args.height),
                'fx': float(args.fx),
                'fy': float(args.fy),
                'cx': float(args.cx),
                'cy': float(args.cy),
                'hfov_used_deg': float(args.hfov_deg),
            },
            'row_count': int(len(summaries)),
            'accepted_count': int(len(accepted)),
            'rejected_count': int(len(rejected)),
            'per_target_counts': per_target_counts,
            'rows': summaries,
            'accepted_rows': accepted,
            'rejected_rows': rejected,
        }
        summary_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')

    # Resolve exclusions before importing Habitat or loading any scene/planning code.
    modules = {}
    if uids:
        modules['curve'] = load_module(FINAL_DIR / 'render_observer_to_operating_curve_video.py', '_ranger_probe_curve')
        modules['base_views'] = modules['curve'].load_module(FINAL_DIR / 'generate_accepted_base_views.py', '_ranger_probe_baseviews')
        modules['render_fp'] = modules['curve'].load_module(FINAL_DIR / 'render_fp_from_observer_candidates.py', '_ranger_probe_renderfp')
        modules['base_mod'] = modules['base_views'].load_module(SCRIPTS_DIR / 'render_local_topdown.py', '_ranger_probe_scene')

    for processed_idx, uid in enumerate(uids, start=1):
        try:
            candidate_count = int(args.candidate_variants_per_target) if int(args.candidate_variants_per_target) > 0 else int(args.variants_per_target)
            variants = build_variants_for_target(uid, candidate_count)
            if target_variant_ids:
                keep = target_variant_ids.get(uid, set())
                variants = [v for v in variants if str(v.get('variant_id', '')) in keep]
            if args.variant_ids.strip():
                keep = {x.strip() for x in args.variant_ids.split(',') if x.strip()}
                variants = [v for v in variants if str(v.get('variant_id', '')) in keep]
            if bool(args.require_full_variant_count) and len(variants) < int(args.variants_per_target):
                summaries.append({
                    'target_uid': uid,
                    'error': 'not_enough_source_variants',
                    'candidate_count': int(len(variants)),
                    'required_count': int(args.variants_per_target),
                })
                per_target_counts[uid] = {
                    'accepted_count': 0,
                    'rejected_count': 1,
                    'candidate_count': int(len(variants)),
                    'stopped_after_accepted': int(args.stop_after_accepted_per_target),
                }
                continue
        except Exception as e:
            summaries.append({'target_uid': uid, 'error': f'build_variants_failed: {repr(e)}'})
            continue
        accepted_for_uid = 0
        rejected_for_uid = 0
        skipped_for_uid = 0
        for variant in variants:
            if bool(args.skip_existing):
                existing_summary_path = variant_output_dir(Path(args.out_root), uid, variant) / 'summary.json'
                if existing_summary_path.exists():
                    try:
                        existing = compact_summary(load_json(existing_summary_path))
                    except Exception as e:
                        existing = {'target_uid': uid, 'variant': variant, 'error': f'load_existing_summary_failed: {repr(e)}'}
                    existing['skipped_existing'] = True
                    summaries.append(existing)
                    skipped_for_uid += 1
                    if isinstance(existing, dict) and existing.get('trajectory_filter', {}).get('pass') is True:
                        accepted_for_uid += 1
                    else:
                        rejected_for_uid += 1
                    if int(args.stop_after_accepted_per_target) > 0 and accepted_for_uid >= int(args.stop_after_accepted_per_target):
                        break
                    continue
            try:
                summary = render_case(args, uid, modules, variant)
                compact = compact_summary(summary)
                summaries.append(compact)
                if isinstance(compact, dict) and compact.get('trajectory_filter', {}).get('pass') is True:
                    accepted_for_uid += 1
                else:
                    rejected_for_uid += 1
            except Exception as e:
                summaries.append({'target_uid': uid, 'variant': variant, 'error': repr(e)})
                rejected_for_uid += 1
            if int(args.stop_after_accepted_per_target) > 0 and accepted_for_uid >= int(args.stop_after_accepted_per_target):
                break
        per_target_counts[uid] = {
            'accepted_count': int(accepted_for_uid),
            'rejected_count': int(rejected_for_uid),
            'skipped_existing_count': int(skipped_for_uid),
            'candidate_count': int(len(variants)),
            'stopped_after_accepted': int(args.stop_after_accepted_per_target),
        }
        if int(args.checkpoint_every) > 0 and processed_idx % int(args.checkpoint_every) == 0:
            write_batch_summary()
    write_batch_summary()
    print(summary_path)


if __name__ == '__main__':
    main()
