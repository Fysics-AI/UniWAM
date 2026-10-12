#!/usr/bin/env python3
"""Build a phase-routed virtual sampling index for mobile AGX demonstrations.

The source LeRobot dataset is never rewritten.  This script creates a compact
index of virtual branch samples that reference the existing window rows:

* navigation primary samples come from the initial base-motion phase;
* manipulation primary samples come from the post-navigation arm phase;
* controlled cross-branch hold samples teach base/arms to remain still.

The intended trajectory order is navigation followed by manipulation.  Episodes
that violate that contract are reported explicitly rather than silently forced
into one phase.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from uniwam.datasets.lerobot.rot6d import rotation_geodesic_angle_np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument(
        "--window-index-path",
        type=Path,
        default=None,
        help="Optional external window index aligned with the existing latent-cache row order.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--exclude-episode",
        action="append",
        type=int,
        default=[],
        help="Exclude an entire converted episode from the routed training index.",
    )

    # Phase segmentation is intentionally stricter than the old event-only stop detector.
    parser.add_argument("--linear-motion-threshold", type=float, default=0.01)
    parser.add_argument("--yaw-motion-threshold", type=float, default=0.02)
    parser.add_argument("--min-motion-run-frames", type=int, default=5)
    parser.add_argument("--max-static-gap-frames", type=int, default=5)
    parser.add_argument("--min-terminal-static-frames", type=int, default=45)

    # Existing navigation event policy.
    parser.add_argument("--nav-turn-weight", type=float, default=5.0)
    parser.add_argument("--nav-stop-pre-frames", type=int, default=90)
    parser.add_argument("--nav-stop-post-frames", type=int, default=0)
    parser.add_argument("--nav-stop-weight", type=float, default=2.0)

    # Existing manipulation-event policy, expressed as additive bonuses to base weight 1.
    parser.add_argument("--xyz-motion-threshold", type=float, default=1.0e-4)
    parser.add_argument("--rotation-motion-threshold", type=float, default=1.0e-4)
    parser.add_argument("--gripper-motion-threshold", type=float, default=5.0e-5)
    parser.add_argument("--gripper-smooth-window", type=int, default=5)
    parser.add_argument("--gripper-event-delta", type=float, default=1.0e-2)
    parser.add_argument("--event-cooldown", type=int, default=8)
    parser.add_argument("--manip-event-pre-frames", type=int, default=60)
    parser.add_argument("--manip-event-post-frames", type=int, default=90)
    parser.add_argument("--direction-window", type=int, default=6)
    parser.add_argument("--direction-angle-deg", type=float, default=75.0)
    parser.add_argument("--direction-min-displacement", type=float, default=5.0e-3)
    parser.add_argument("--speed-smooth-window", type=int, default=5)
    parser.add_argument("--speed-change-min-delta", type=float, default=3.0e-3)
    parser.add_argument("--gripper-bonus", type=float, default=3.0)
    parser.add_argument("--direction-bonus", type=float, default=1.5)
    parser.add_argument("--speed-bonus", type=float, default=1.0)

    # These are final probability masses after all event multipliers, not raw row rates.
    parser.add_argument("--target-nav-hold-mass", type=float, default=0.20)
    parser.add_argument("--target-manip-hold-mass", type=float, default=0.15)
    parser.add_argument(
        "--target-nav-branch-mass",
        type=float,
        default=0.40,
        help=(
            "Final navigation probability mass within a mobile source. Set to 0 "
            "for manipulation-only sources. This calibration is applied after "
            "event and cross-hold weighting."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("target_nav_hold_mass", "target_manip_hold_mass"):
        value = float(getattr(args, name))
        if not 0.0 < value < 0.5:
            raise ValueError(f"{name} must be in (0, 0.5), got {value}.")
    if not 0.0 <= float(args.target_nav_branch_mass) < 1.0:
        raise ValueError(
            "target_nav_branch_mass must be in [0, 1), got "
            f"{args.target_nav_branch_mass}."
        )
    for name in (
        "horizon",
        "min_motion_run_frames",
        "max_static_gap_frames",
        "min_terminal_static_frames",
        "event_cooldown",
    ):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"{name} must be non-negative.")


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int, bool]]:
    if mask.ndim != 1:
        raise ValueError(f"Expected a one-dimensional mask, got {mask.shape}.")
    if mask.size == 0:
        return []
    changes = np.flatnonzero(mask[1:] != mask[:-1]) + 1
    bounds = np.concatenate(([0], changes, [mask.size]))
    return [
        (int(bounds[index]), int(bounds[index + 1]), bool(mask[bounds[index]]))
        for index in range(bounds.size - 1)
    ]


def remove_short_true_runs(mask: np.ndarray, min_length: int) -> np.ndarray:
    result = mask.astype(bool, copy=True)
    for start, end, value in contiguous_runs(result):
        if value and end - start < min_length:
            result[start:end] = False
    return result


def fill_short_false_gaps(mask: np.ndarray, max_length: int) -> np.ndarray:
    result = mask.astype(bool, copy=True)
    runs = contiguous_runs(result)
    for run_index, (start, end, value) in enumerate(runs):
        if value or end - start > max_length:
            continue
        if run_index == 0 or run_index == len(runs) - 1:
            continue
        if runs[run_index - 1][2] and runs[run_index + 1][2]:
            result[start:end] = True
    return result


def robust_base_motion(nav: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if nav.ndim != 2 or nav.shape[1] != 3:
        raise ValueError(f"Expected action.nav [T,3], got {nav.shape}.")
    linear = np.linalg.norm(nav[:, :2], axis=1) > float(args.linear_motion_threshold)
    turning = np.abs(nav[:, 2]) >= float(args.yaw_motion_threshold)
    raw = linear | turning
    filtered = remove_short_true_runs(raw, int(args.min_motion_run_frames))
    filtered = fill_short_false_gaps(filtered, int(args.max_static_gap_frames))
    filtered = remove_short_true_runs(filtered, int(args.min_motion_run_frames))
    return filtered, turning


def phase_boundary(base_active: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    active_indices = np.flatnonzero(base_active)
    if active_indices.size == 0:
        return {"status": "no_navigation_motion"}
    first_motion = int(active_indices[0])
    last_motion = int(active_indices[-1])
    split_frame = last_motion + 1
    terminal_static = int(base_active.size - split_frame)
    if terminal_static < int(args.min_terminal_static_frames):
        return {
            "status": "insufficient_terminal_static",
            "first_motion_frame": first_motion,
            "last_motion_frame": last_motion,
            "terminal_static_frames": terminal_static,
        }
    return {
        "status": "ok",
        "first_motion_frame": first_motion,
        "last_motion_frame": last_motion,
        "split_frame": split_frame,
        "terminal_static_frames": terminal_static,
    }


def canonical_quaternion(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms < 1.0e-8):
        raise ValueError("Encountered a zero-norm quaternion.")
    result = values / norms
    flip = result[..., 3] < 0.0
    result[flip] *= -1.0
    return result


def quaternion_angle(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = canonical_quaternion(left)
    right = canonical_quaternion(right)
    dot = np.abs(np.sum(left * right, axis=-1))
    return 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))


def centered_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    window = max(1, int(window))
    if window == 1 or values.size == 0:
        return values.copy()
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(window, dtype=np.float64) / float(window), mode="valid")


def nonmax_suppress(candidates: list[tuple[int, float]], length: int, cooldown: int) -> np.ndarray:
    result = np.zeros(length, dtype=bool)
    for index, _score in sorted(candidates, key=lambda item: item[1], reverse=True):
        if not result[max(0, index - cooldown) : min(length, index + cooldown + 1)].any():
            result[index] = True
    return result


def detect_gripper_events(grippers: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    total = max(0, grippers.shape[0] - 1)
    events = np.zeros(total, dtype=bool)
    if total == 0:
        return events
    for arm_index in range(grippers.shape[1]):
        smooth = centered_average(grippers[:, arm_index], int(args.gripper_smooth_window))
        delta = np.diff(smooth)
        moving = np.abs(delta) > float(args.gripper_motion_threshold)
        candidates: list[tuple[int, float]] = []
        cursor = 0
        while cursor < total:
            if not moving[cursor]:
                cursor += 1
                continue
            sign = 1.0 if delta[cursor] >= 0.0 else -1.0
            start = cursor
            cursor += 1
            while cursor < total and moving[cursor] and delta[cursor] * sign > 0.0:
                cursor += 1
            end = cursor - 1
            net = float(smooth[end + 1] - smooth[start])
            if abs(net) >= float(args.gripper_event_delta):
                local = int(np.argmax(np.abs(delta[start : end + 1])))
                candidates.append((start + local, abs(net)))
        events |= nonmax_suppress(candidates, total, int(args.event_cooldown))
    return events


def detect_direction_events(vectors: list[np.ndarray], args: argparse.Namespace) -> np.ndarray:
    total = vectors[0].shape[0] if vectors else 0
    events = np.zeros(total, dtype=bool)
    window = max(1, int(args.direction_window))
    cosine_threshold = float(np.cos(np.deg2rad(float(args.direction_angle_deg))))
    for values in vectors:
        candidates: list[tuple[int, float]] = []
        for index in range(window, total - window + 1):
            before = values[index - window : index].mean(axis=0)
            after = values[index : index + window].mean(axis=0)
            before_norm = float(np.linalg.norm(before))
            after_norm = float(np.linalg.norm(after))
            if min(before_norm, after_norm) < float(args.direction_min_displacement):
                continue
            cosine = float(np.clip(np.dot(before, after) / (before_norm * after_norm), -1.0, 1.0))
            if cosine <= cosine_threshold:
                candidates.append((index, float(np.arccos(cosine))))
        events |= nonmax_suppress(candidates, total, int(args.event_cooldown))
    return events


def detect_speed_events(speeds: list[np.ndarray], args: argparse.Namespace) -> np.ndarray:
    total = speeds[0].shape[0] if speeds else 0
    events = np.zeros(total, dtype=bool)
    for values in speeds:
        smooth = centered_average(values, int(args.speed_smooth_window))
        delta = np.zeros_like(smooth)
        if total > 1:
            delta[1:] = np.abs(np.diff(smooth))
        candidates = [
            (int(index), float(delta[index]))
            for index in np.flatnonzero(delta > float(args.speed_change_min_delta))
        ]
        events |= nonmax_suppress(candidates, total, int(args.event_cooldown))
    return events


def manipulation_activity(
    state: np.ndarray, action: np.ndarray, args: argparse.Namespace
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if state.ndim != 2 or action.ndim != 2 or state.shape[0] != action.shape[0]:
        raise ValueError(
            f"State/action must be aligned matrices, got {state.shape}/{action.shape}."
        )
    if state.shape[1] == 19 and action.shape[1] == 16:
        left_position, left_rotation = state[:, 3:6], state[:, 6:10]
        left_target_position, left_target_rotation = action[:, 0:3], action[:, 3:7]
        left_gripper, left_target_gripper = state[:, 10], action[:, 7]
        right_position, right_rotation = state[:, 11:14], state[:, 14:18]
        right_target_position, right_target_rotation = action[:, 8:11], action[:, 11:15]
        right_gripper, right_target_gripper = state[:, 18], action[:, 15]
        left_angle = quaternion_angle(left_rotation, left_target_rotation)
        right_angle = quaternion_angle(right_rotation, right_target_rotation)
    elif state.shape[1] == 23 and action.shape[1] == 20:
        left_position, left_rotation = state[:, 3:6], state[:, 6:12]
        left_target_position, left_target_rotation = action[:, 0:3], action[:, 3:9]
        left_gripper, left_target_gripper = state[:, 12], action[:, 9]
        right_position, right_rotation = state[:, 13:16], state[:, 16:22]
        right_target_position, right_target_rotation = action[:, 10:13], action[:, 13:19]
        right_gripper, right_target_gripper = state[:, 22], action[:, 19]
        left_angle = rotation_geodesic_angle_np(left_rotation, left_target_rotation)
        right_angle = rotation_geodesic_angle_np(right_rotation, right_target_rotation)
    else:
        raise ValueError(
            "Expected quaternion [T,19]/[T,16] or rot6d [T,23]/[T,20] state/action, "
            f"got {state.shape}/{action.shape}."
        )

    left_delta = left_target_position - left_position
    right_delta = right_target_position - right_position
    left_speed = np.linalg.norm(left_delta, axis=1)
    right_speed = np.linalg.norm(right_delta, axis=1)
    gripper_delta = np.maximum(
        np.abs(left_target_gripper - left_gripper),
        np.abs(right_target_gripper - right_gripper),
    )
    active = (
        (np.maximum(left_speed, right_speed) > float(args.xyz_motion_threshold))
        | (np.maximum(left_angle, right_angle) > float(args.rotation_motion_threshold))
        | (gripper_delta > float(args.gripper_motion_threshold))
    )
    gripper_events = detect_gripper_events(
        np.stack((left_gripper, right_gripper), axis=-1), args
    )
    direction_events = detect_direction_events([left_delta, right_delta], args)
    speed_events = detect_speed_events([left_speed, right_speed], args)
    return active, gripper_events, direction_events, speed_events


def expand_event_context(events: np.ndarray, pre_frames: int, post_frames: int) -> np.ndarray:
    result = np.zeros(events.shape, dtype=bool)
    for index in np.flatnonzero(events):
        start = max(0, int(index) - int(pre_frames))
        end = min(events.size, int(index) + int(post_frames) + 1)
        result[start:end] = True
    return result


def any_in_window(values: np.ndarray, start: int, horizon: int) -> bool:
    return bool(values[start : start + horizon].any())


def episode_path(root: Path, episode_index: int) -> Path:
    return root / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"


def add_candidate(
    records: list[dict[str, Any]],
    *,
    source_window_index: int,
    episode_index: int,
    start_frame: int,
    horizon: int,
    phase_split_frame: int,
    branch: str,
    sample_kind: str,
    role: str,
    raw_weight: float,
    nav_active_count: int,
    manip_active_count: int,
    has_nav_turn: bool,
    in_nav_stop_pre_context: bool,
    in_gripper_event_context: bool,
    in_direction_event_context: bool,
    in_speed_event_context: bool,
) -> None:
    records.append(
        {
            "source_window_index": source_window_index,
            "episode_index": episode_index,
            "start_frame": start_frame,
            "horizon": horizon,
            "phase_split_frame": phase_split_frame,
            "branch": branch,
            "sample_kind": sample_kind,
            "role": role,
            "raw_sampling_weight": np.float32(raw_weight),
            "sampling_weight": np.float32(raw_weight),
            "nav_active_count": np.int16(nav_active_count),
            "manip_active_count": np.int16(manip_active_count),
            "has_nav_turn": has_nav_turn,
            "in_nav_stop_pre_context": in_nav_stop_pre_context,
            "in_gripper_event_context": in_gripper_event_context,
            "in_direction_event_context": in_direction_event_context,
            "in_speed_event_context": in_speed_event_context,
        }
    )


def apply_hold_mass_target(
    records: list[dict[str, Any]], branch: str, target_hold_mass: float
) -> dict[str, float | str | None]:
    primary = [record for record in records if record["branch"] == branch and record["role"] == "primary"]
    holds = [record for record in records if record["branch"] == branch and record["role"] == "cross_hold"]
    primary_mass = float(sum(float(record["raw_sampling_weight"]) for record in primary))
    hold_mass = float(sum(float(record["raw_sampling_weight"]) for record in holds))
    if primary_mass <= 0.0 or hold_mass <= 0.0:
        return {
            "status": "not_applicable",
            "primary_raw_mass": primary_mass,
            "hold_raw_mass": hold_mass,
            "hold_scale": None,
            "final_hold_mass": 0.0,
            "actual_hold_mass_fraction": None,
        }
    scale = target_hold_mass * primary_mass / ((1.0 - target_hold_mass) * hold_mass)
    for record in holds:
        record["sampling_weight"] = np.float32(float(record["raw_sampling_weight"]) * scale)
    final_hold_mass = hold_mass * scale
    return {
        "status": "applied",
        "primary_raw_mass": primary_mass,
        "hold_raw_mass": hold_mass,
        "hold_scale": scale,
        "final_hold_mass": final_hold_mass,
        "actual_hold_mass_fraction": final_hold_mass / (primary_mass + final_hold_mass),
    }


def apply_branch_mass_target(
    records: list[dict[str, Any]], target_nav_mass: float
) -> dict[str, float | str | None]:
    """Calibrate nav/manip mass without changing either branch's internal mix."""
    nav = [record for record in records if record["branch"] == "nav"]
    manip = [record for record in records if record["branch"] == "manip"]
    nav_mass = float(sum(float(record["sampling_weight"]) for record in nav))
    manip_mass = float(sum(float(record["sampling_weight"]) for record in manip))
    if target_nav_mass == 0.0:
        if nav_mass > 0.0:
            raise ValueError(
                "target_nav_branch_mass=0 is only valid when the source has no nav records."
            )
        return {
            "status": "manipulation_only",
            "nav_mass_before": nav_mass,
            "manip_mass_before": manip_mass,
            "nav_scale": None,
            "actual_nav_mass_fraction": 0.0,
        }
    if nav_mass <= 0.0 or manip_mass <= 0.0:
        raise ValueError(
            "A positive target_nav_branch_mass requires both nav and manip candidates: "
            f"nav_mass={nav_mass}, manip_mass={manip_mass}."
        )
    nav_scale = target_nav_mass * manip_mass / ((1.0 - target_nav_mass) * nav_mass)
    for record in nav:
        record["sampling_weight"] = np.float32(
            float(record["sampling_weight"]) * nav_scale
        )
    nav_mass_after = nav_mass * nav_scale
    return {
        "status": "applied",
        "nav_mass_before": nav_mass,
        "manip_mass_before": manip_mass,
        "nav_scale": nav_scale,
        "nav_mass_after": nav_mass_after,
        "actual_nav_mass_fraction": nav_mass_after / (nav_mass_after + manip_mass),
    }


def build_index(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    root = args.dataset_root.resolve()
    windows_path = (
        root / "meta/window_index.parquet"
        if args.window_index_path is None
        else args.window_index_path.resolve()
    )
    if not windows_path.is_file():
        raise FileNotFoundError(f"Missing {windows_path}")
    windows = pq.read_table(windows_path, columns=["episode_index", "start_frame", "horizon"])
    episode_column = np.asarray(windows["episode_index"].to_numpy(), dtype=np.int64)
    start_column = np.asarray(windows["start_frame"].to_numpy(), dtype=np.int64)
    horizon_column = np.asarray(windows["horizon"].to_numpy(), dtype=np.int64)
    if not np.all(horizon_column == int(args.horizon)):
        raise ValueError(
            f"Expected all window horizons={args.horizon}, got {sorted(set(horizon_column.tolist()))[:8]}."
        )

    records: list[dict[str, Any]] = []
    episode_reports: list[dict[str, Any]] = []
    all_episode_indices_raw = sorted(set(episode_column.tolist()))
    excluded_episodes = sorted(set(int(value) for value in args.exclude_episode))
    missing_exclusions = set(excluded_episodes) - set(all_episode_indices_raw)
    if missing_exclusions:
        raise ValueError(
            f"Excluded episodes are absent from the window table: {sorted(missing_exclusions)}."
        )
    all_episode_indices = [
        episode for episode in all_episode_indices_raw if episode not in set(excluded_episodes)
    ]
    if not all_episode_indices:
        raise ValueError("Episode exclusions removed the entire source.")
    for count, episode_index in enumerate(all_episode_indices, start=1):
        path = episode_path(root, int(episode_index))
        if not path.is_file():
            raise FileNotFoundError(f"Missing episode parquet: {path}")
        schema_names = set(pq.read_schema(path).names)
        nav_column = "action.nav_velocity" if "action.nav_velocity" in schema_names else "action.nav"
        table = pq.read_table(path, columns=["observation.state", nav_column, "action.manip"])
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        nav = np.asarray(table[nav_column].to_pylist(), dtype=np.float32)
        manip = np.asarray(table["action.manip"].to_pylist(), dtype=np.float32)
        if state.shape[0] != nav.shape[0] or state.shape[0] != manip.shape[0]:
            raise ValueError(f"Episode {episode_index} has inconsistent feature lengths.")

        base_active, turn = robust_base_motion(nav, args)
        boundary = phase_boundary(base_active, args)
        manip_active, gripper_events, direction_events, speed_events = manipulation_activity(
            state, manip, args
        )
        overlap_frames = int(np.count_nonzero(base_active & manip_active))
        episode_mask = episode_column == int(episode_index)
        row_indices = np.flatnonzero(episode_mask)
        report: dict[str, Any] = {
            "episode_index": int(episode_index),
            "frame_count": int(state.shape[0]),
            "window_count": int(row_indices.size),
            "base_active_frames": int(base_active.sum()),
            "manip_active_frames": int(manip_active.sum()),
            "simultaneous_base_manip_frames": overlap_frames,
            "gripper_event_count": int(gripper_events.sum()),
            "direction_event_count": int(direction_events.sum()),
            "speed_event_count": int(speed_events.sum()),
            **boundary,
        }
        gripper_context = expand_event_context(
            gripper_events, int(args.manip_event_pre_frames), int(args.manip_event_post_frames)
        )
        direction_context = expand_event_context(
            direction_events, int(args.manip_event_pre_frames), int(args.manip_event_post_frames)
        )
        speed_context = expand_event_context(
            speed_events, int(args.manip_event_pre_frames), int(args.manip_event_post_frames)
        )

        # Several recordings stored in a mobile source contain only the post-arrival
        # manipulation segment. They are useful manipulation data, not failed rows.
        # They deliberately do not enter nav-hold: without a demonstrated navigation
        # phase, they would inflate static-base supervision in an unrelated domain.
        if boundary["status"] == "no_navigation_motion":
            counts = Counter()
            for source_window_index in row_indices.tolist():
                start = int(start_column[source_window_index])
                horizon = int(horizon_column[source_window_index])
                manip_count = int(manip_active[start : start + horizon].sum())
                if manip_count == 0:
                    continue
                in_gripper = any_in_window(gripper_context, start, horizon)
                in_direction = any_in_window(direction_context, start, horizon)
                in_speed = any_in_window(speed_context, start, horizon)
                manip_weight = 1.0
                if in_gripper:
                    manip_weight += float(args.gripper_bonus)
                if in_direction:
                    manip_weight += float(args.direction_bonus)
                if in_speed:
                    manip_weight += float(args.speed_bonus)
                add_candidate(
                    records,
                    source_window_index=source_window_index,
                    episode_index=int(episode_index),
                    start_frame=start,
                    horizon=horizon,
                    phase_split_frame=-1,
                    branch="manip",
                    sample_kind="manip_active_no_nav",
                    role="primary",
                    raw_weight=manip_weight,
                    nav_active_count=0,
                    manip_active_count=manip_count,
                    has_nav_turn=False,
                    in_nav_stop_pre_context=False,
                    in_gripper_event_context=in_gripper,
                    in_direction_event_context=in_direction,
                    in_speed_event_context=in_speed,
                )
                counts["manip_active_no_nav"] += 1
            report["candidate_counts"] = dict(counts)
            report["routing"] = "manipulation_only"
            episode_reports.append(report)
            continue

        if boundary["status"] != "ok":
            episode_reports.append(report)
            continue

        split = int(boundary["split_frame"])
        stop_pre_start = max(int(boundary["first_motion_frame"]), split - int(args.nav_stop_pre_frames))
        stop_post_end = min(
            int(state.shape[0]) - 1, split + int(args.nav_stop_post_frames)
        )

        counts = Counter()
        for source_window_index in row_indices.tolist():
            start = int(start_column[source_window_index])
            horizon = int(horizon_column[source_window_index])
            nav_count = int(base_active[start : start + horizon].sum())
            manip_count = int(manip_active[start : start + horizon].sum())
            has_turn = any_in_window(turn, start, horizon)
            in_stop_pre = stop_pre_start <= start < split
            in_stop_context = stop_pre_start <= start <= stop_post_end
            in_gripper = any_in_window(gripper_context, start, horizon)
            in_direction = any_in_window(direction_context, start, horizon)
            in_speed = any_in_window(speed_context, start, horizon)
            nav_weight = 1.0
            if has_turn:
                nav_weight *= float(args.nav_turn_weight)
            if in_stop_context:
                nav_weight *= float(args.nav_stop_weight)
            manip_weight = 1.0
            if in_gripper:
                manip_weight += float(args.gripper_bonus)
            if in_direction:
                manip_weight += float(args.direction_bonus)
            if in_speed:
                manip_weight += float(args.speed_bonus)

            # A boundary-crossing window remains a navigation sample.  Its future contains
            # the braking action; later arm motion must not relabel it as manipulation.
            if start < split and nav_count > 0:
                kind = "nav_stop_pre" if in_stop_pre else "nav_motion"
                add_candidate(
                    records,
                    source_window_index=source_window_index,
                    episode_index=int(episode_index),
                    start_frame=start,
                    horizon=horizon,
                    phase_split_frame=split,
                    branch="nav",
                    sample_kind=kind,
                    role="primary",
                    raw_weight=nav_weight,
                    nav_active_count=nav_count,
                    manip_active_count=manip_count,
                    has_nav_turn=has_turn,
                    in_nav_stop_pre_context=in_stop_context,
                    in_gripper_event_context=in_gripper,
                    in_direction_event_context=in_direction,
                    in_speed_event_context=in_speed,
                )
                counts[kind] += 1

                if manip_count == 0:
                    add_candidate(
                        records,
                        source_window_index=source_window_index,
                        episode_index=int(episode_index),
                        start_frame=start,
                        horizon=horizon,
                        phase_split_frame=split,
                        branch="manip",
                        sample_kind="manip_hold_in_nav",
                        role="cross_hold",
                        raw_weight=nav_weight,
                        nav_active_count=nav_count,
                        manip_active_count=manip_count,
                        has_nav_turn=has_turn,
                        in_nav_stop_pre_context=in_stop_context,
                        in_gripper_event_context=in_gripper,
                        in_direction_event_context=in_direction,
                        in_speed_event_context=in_speed,
                    )
                    counts["manip_hold_in_nav"] += 1

            if start >= split and manip_count > 0 and nav_count == 0:
                add_candidate(
                    records,
                    source_window_index=source_window_index,
                    episode_index=int(episode_index),
                    start_frame=start,
                    horizon=horizon,
                    phase_split_frame=split,
                    branch="manip",
                    sample_kind="manip_active",
                    role="primary",
                    raw_weight=manip_weight,
                    nav_active_count=nav_count,
                    manip_active_count=manip_count,
                    has_nav_turn=has_turn,
                    in_nav_stop_pre_context=in_stop_context,
                    in_gripper_event_context=in_gripper,
                    in_direction_event_context=in_direction,
                    in_speed_event_context=in_speed,
                )
                counts["manip_active"] += 1

                add_candidate(
                    records,
                    source_window_index=source_window_index,
                    episode_index=int(episode_index),
                    start_frame=start,
                    horizon=horizon,
                    phase_split_frame=split,
                    branch="nav",
                    sample_kind="nav_hold_in_manip",
                    role="cross_hold",
                    raw_weight=(
                        manip_weight * float(args.nav_stop_weight)
                        if in_stop_context
                        else manip_weight
                    ),
                    nav_active_count=nav_count,
                    manip_active_count=manip_count,
                    has_nav_turn=has_turn,
                    in_nav_stop_pre_context=in_stop_context,
                    in_gripper_event_context=in_gripper,
                    in_direction_event_context=in_direction,
                    in_speed_event_context=in_speed,
                )
                counts["nav_hold_in_manip"] += 1

        report["candidate_counts"] = dict(counts)
        episode_reports.append(report)
        if count % 50 == 0 or count == len(all_episode_indices):
            print(f"processed {count}/{len(all_episode_indices)} episodes", flush=True)

    nav_ratio = apply_hold_mass_target(records, "nav", float(args.target_nav_hold_mass))
    manip_ratio = apply_hold_mass_target(records, "manip", float(args.target_manip_hold_mass))
    branch_ratio = apply_branch_mass_target(records, float(args.target_nav_branch_mass))
    kinds = Counter(record["sample_kind"] for record in records)
    kind_mass: Counter[str] = Counter()
    for record in records:
        kind_mass[record["sample_kind"]] += float(record["sampling_weight"])
    summary = {
        "source_name": args.source_name,
        "dataset_root": str(root),
        "window_count": int(windows.num_rows),
        "excluded_episodes": excluded_episodes,
        "excluded_window_count": int(np.isin(episode_column, excluded_episodes).sum()),
        "virtual_candidate_count": int(len(records)),
        "parameters": {
            "horizon": int(args.horizon),
            "fps": int(args.fps),
            "linear_motion_threshold": float(args.linear_motion_threshold),
            "yaw_motion_threshold": float(args.yaw_motion_threshold),
            "min_motion_run_frames": int(args.min_motion_run_frames),
            "max_static_gap_frames": int(args.max_static_gap_frames),
            "min_terminal_static_frames": int(args.min_terminal_static_frames),
            "nav_stop_pre_frames": int(args.nav_stop_pre_frames),
            "nav_stop_post_frames": int(args.nav_stop_post_frames),
            "nav_stop_weight": float(args.nav_stop_weight),
            "target_nav_hold_mass": float(args.target_nav_hold_mass),
            "target_manip_hold_mass": float(args.target_manip_hold_mass),
            "target_nav_branch_mass": float(args.target_nav_branch_mass),
            "exclude_episode": excluded_episodes,
        },
        "branch_mass": {"nav": nav_ratio, "manip": manip_ratio},
        "branch_mix": branch_ratio,
        "candidate_count_by_kind": dict(kinds),
        "final_sampling_mass_by_kind": dict(kind_mass),
        "episode_status_counts": dict(Counter(report["status"] for report in episode_reports)),
        "episodes_with_simultaneous_base_manip_motion": int(
            sum(report["simultaneous_base_manip_frames"] > 0 for report in episode_reports)
        ),
        "simultaneous_base_manip_frame_count": int(
            sum(report["simultaneous_base_manip_frames"] for report in episode_reports)
        ),
    }
    return records, episode_reports, summary


def write_outputs(
    output_dir: Path,
    records: list[dict[str, Any]],
    episode_reports: list[dict[str, Any]],
    summary: dict[str, Any],
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "phase_branch_index.parquet"
    report_path = output_dir / "phase_split_report.json"
    episodes_path = output_dir / "phase_split_episodes.jsonl"
    existing = [path for path in (index_path, report_path, episodes_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing outputs: {existing}")
    if not records:
        raise RuntimeError("No virtual branch candidates were produced.")
    pq.write_table(pa.Table.from_pylist(records), index_path, compression="zstd")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with episodes_path.open("w", encoding="utf-8") as handle:
        for report in episode_reports:
            handle.write(json.dumps(report, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    validate_args(args)
    records, episode_reports, summary = build_index(args)
    write_outputs(args.output_dir.resolve(), records, episode_reports, summary, args.overwrite)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
