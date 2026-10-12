#!/usr/bin/env python3
"""Batched closed-loop HM3D evaluation for policies with a batch WebSocket API.

Each active episode keeps its own Habitat simulator state.  Rendering is done
once for the active batch, followed by one policy request with
``{"batch": [observation, ...]}``.  Actions and stop/collision handling stay
per episode, so batching cannot mix trajectories.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import hm3d_benchmark as bench


DEFAULT_DATA_ROOT = Path(__import__("os").environ.get("MAP_BENCH_DATA_ROOT", ".")).resolve()
DEFAULT_BENCHMARK = DEFAULT_DATA_ROOT / "evaluation" / "benchmark_15k.json"
DEFAULT_OUTPUT = Path("outputs/closedloop_batch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Data checkout containing evaluation/scene_contexts (or set MAP_BENCH_DATA_ROOT).",
    )
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=18018)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--policy-retries", type=int, default=3)
    parser.add_argument(
        "--policy-one-shot",
        action="store_true",
        help="Open one WebSocket per inference request and close it before Habitat execution.",
    )
    parser.add_argument(
        "--render-lock-path",
        type=Path,
        default=None,
        help="Serialize Habitat GPU rendering across workers sharing one GPU.",
    )
    parser.add_argument(
        "--source-prefetch-workers",
        type=int,
        default=0,
        help="Decode source-video first frames ahead of episode initialization.",
    )
    parser.add_argument("--group-limit", type=int, default=0)
    parser.add_argument("--group-offset", type=int, default=0)
    parser.add_argument("--target-shard-count", type=int, default=1)
    parser.add_argument("--target-shard-index", type=int, default=0)
    parser.add_argument("--variant-mode", choices=("canonical", "all"), default="all")
    parser.add_argument("--write-video", action="store_true")
    parser.add_argument("--save-start-audit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--near-zero-ratio", type=float, default=0.80)
    parser.add_argument("--near-zero-position-m", type=float, default=0.05)
    parser.add_argument("--near-zero-heading-rad", type=float, default=0.05)
    parser.add_argument("--near-zero-consecutive", type=int, default=3)
    parser.add_argument("--success-position-m", type=float, default=0.50)
    parser.add_argument("--success-heading-deg", type=float, default=15.0)
    parser.add_argument("--start-camera-tolerance-m", type=float, default=0.003)
    parser.add_argument("--start-yaw-tolerance-deg", type=float, default=0.02)
    parser.add_argument("--start-pixel-mae-limit", type=float, default=25.0)
    parser.add_argument("--base-seed", type=int, default=20260807)
    parser.add_argument("--fixed-diffusion-seed", action="store_true")
    parser.add_argument("--action-horizon", type=int, choices=(16, 32), default=32)
    parser.add_argument("--camera-width", type=int, default=566)
    parser.add_argument("--camera-height", type=int, default=320)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def safe_name(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in value)
    return f"{stem[:120]}__{digest}"


def episode_result_path(
    output_dir: Path,
    item: tuple[dict, str, int, dict],
) -> Path:
    group, family, variant_index, _ = item
    identity = f"{group['group_id']}::{family}_{variant_index:02d}"
    return output_dir / "episodes" / safe_name(identity) / "result.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_policy_client(host: str, port: int) -> Any:
    for path in (HERE.parent / "openpi_client_src",):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    return WebsocketClientPolicy(host=host, port=port)


def close_policy_client(client: Any) -> None:
    if client is None:
        return
    websocket = getattr(client, "_ws", None)
    if websocket is None:
        return
    try:
        websocket.close()
    except Exception:
        pass


@contextmanager
def render_guard(args: argparse.Namespace):
    lock_file = getattr(args, "render_lock_file", None)
    if lock_file is None:
        yield
        return
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def install_render_guard(scene: Any, args: argparse.Namespace) -> None:
    raw_render = scene.render

    def guarded_render(base_xz: np.ndarray, theta: float):
        with render_guard(args):
            rendered = raw_render(base_xz, theta)
            rgb = np.asarray(rendered[0])
            if rgb.size == 0 or not np.any(rgb):
                raise RuntimeError("Habitat returned an all-black RGB frame")
            return rendered

    scene.render = guarded_render


class SourceFramePrefetcher:
    def __init__(self, decode: Any, workers: int) -> None:
        self._decode = decode
        self._executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="source-frame",
        )
        self._futures: dict[str, Future] = {}

    def prepare(self, items: list[tuple[dict, str, int, dict]]) -> None:
        self._futures = {
            str(episode["video_path"]): self._executor.submit(
                self._decode, str(episode["video_path"])
            )
            for _, _, _, episode in items
            if episode.get("video_path")
        }

    def decode(self, video_path: str) -> np.ndarray:
        future = self._futures.pop(str(video_path), None)
        if future is None:
            return self._decode(video_path)
        return future.result()

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def flatten_episodes(benchmark: dict, variant_mode: str) -> list[tuple[dict, str, int, dict]]:
    episodes = []
    for group in benchmark["groups"]:
        episodes.append((group, "canonical", 0, group["canonical"]))
        if variant_mode == "all":
            episodes.extend(
                (group, "initial_yaw", index, episode)
                for index, episode in enumerate(group.get("initial_yaw", []), start=1)
            )
    return episodes


def target_balanced_groups(
    groups: list[dict], shard_count: int, shard_index: int
) -> tuple[list[dict], dict]:
    """Assign whole targets to shards using predicted closed-loop work."""
    if shard_count <= 0:
        raise ValueError("target-shard-count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError("target-shard-index must be in [0, target-shard-count)")

    by_target: dict[str, list[dict]] = {}
    for group in groups:
        by_target.setdefault(str(group["target_uid"]), []).append(group)

    def group_work(group: dict) -> int:
        chunks = int(group.get("reference_chunk_count", 1))
        return max(1, 2 * chunks)

    targets = [
        (target_uid, target_groups, sum(group_work(group) for group in target_groups))
        for target_uid, target_groups in by_target.items()
    ]
    targets.sort(key=lambda value: (-value[2], value[0]))

    shard_loads = [0 for _ in range(shard_count)]
    shard_targets: list[list[tuple[str, list[dict], int]]] = [
        [] for _ in range(shard_count)
    ]
    for target in targets:
        destination = min(range(shard_count), key=lambda index: (shard_loads[index], index))
        shard_targets[destination].append(target)
        shard_loads[destination] += target[2]

    selected = shard_targets[shard_index]
    selected_groups = [group for _, target_groups, _ in selected for group in target_groups]
    plan = {
        "count": shard_count,
        "index": shard_index,
        "target_count": len(selected),
        "group_count": len(selected_groups),
        "estimated_max_chunk_calls": shard_loads[shard_index],
        "all_shard_estimated_max_chunk_calls": shard_loads,
        "target_uids": [target_uid for target_uid, _, _ in selected],
    }
    return selected_groups, plan


def atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def make_state(
    scene: Any,
    group: dict,
    family: str,
    variant_index: int,
    episode: dict,
    args: argparse.Namespace,
) -> dict:
    identity = f"{group['group_id']}::{family}_{variant_index:02d}"
    output_dir = args.output_dir / "episodes" / safe_name(identity)
    result_path = output_dir / "result.json"
    synthetic = "start_base_xz" in episode
    if synthetic:
        summary_path = Path(group.get("plan_summary_json") or args.benchmark)
        summary = {}
        canonical_summary_path = summary_path
        canonical_summary = {}
    else:
        summary_path, summary = bench.episode_summary(episode)
        canonical_summary_path, canonical_summary = bench.episode_summary(group["canonical"])
    base_xz, theta, rgb, start_audit = bench.start_pose_audit(
        scene, episode, summary, output_dir, args
    )
    if synthetic:
        goal_base_xz = np.asarray(group["goal_base_xz"], dtype=np.float32)
        operating_xz = np.asarray(group["operating_camera_xz"], dtype=np.float32)
        goal_theta = float(group["goal_theta_rad"])
    else:
        goal_base_xz = np.asarray(canonical_summary["goal"]["base_xy"], dtype=np.float32)
        operating_xz = np.asarray(canonical_summary["goal"]["camera_xz"], dtype=np.float32)
        goal_theta = float(canonical_summary["goal"]["theta_rad"])
    return {
        "scene": scene,
        "group": group,
        "episode": episode,
        "family": family,
        "variant_index": variant_index,
        "identity": identity,
        "output_dir": output_dir,
        "result_path": result_path,
        "summary_path": summary_path,
        "canonical_summary_path": canonical_summary_path,
        "task": str(group["task"]),
        "base_xz": base_xz,
        "theta": theta,
        "rgb": rgb,
        "goal_base_xz": goal_base_xz,
        "operating_xz": operating_xz,
        "goal_theta": goal_theta,
        "start_audit": start_audit,
        "calls": [],
        "frames": [],
        "near_zero_streak": 0,
        # GR00T N1.7 consumes the current navigation velocity state.  The
        # simulator starts at rest; the policy adapter returns the last
        # commanded velocity for the next chunk boundary.
        "navigation_state": np.zeros(2, dtype=np.float32),
        "blocked_waypoints": 0,
        "reference_chunks": int(group.get(
            "reference_chunk_count",
            math.ceil(int(group["canonical"]["frame_count"]) / args.action_horizon),
        )),
        "stop_reason": "budget_exhausted",
        "started": time.monotonic(),
        "call_index": 0,
    }


def finalize_state(state: dict, args: argparse.Namespace) -> dict:
    base_xz = state["base_xz"]
    theta = state["theta"]
    final_camera_xz = bench.camera_xz_from_base(base_xz, theta)
    operating_distance = float(np.linalg.norm(final_camera_xz - state["operating_xz"]))
    base_position_error = float(np.linalg.norm(base_xz - state["goal_base_xz"]))
    heading_error = bench.angular_error_rad(theta, state["goal_theta"])
    distance_success = operating_distance <= args.success_position_m
    heading_success = heading_error <= math.radians(args.success_heading_deg)
    joint_success = bool(distance_success and heading_success)
    if state["stop_reason"] == "model_stop":
        outcome = "success" if distance_success else "premature_stop"
    elif state["stop_reason"] == "out_of_navigable":
        outcome = "out_of_navigable"
    else:
        outcome = "timeout_at_goal" if distance_success else "timeout"
    result = {
        "identity": state["identity"],
        "group_id": state["group"]["group_id"],
        "split": state["group"].get("split"),
        "observation_id": state["group"].get("observation_id"),
        "source_variant_id": state["group"].get("variant_id"),
        "family": state["family"],
        "variant_index": state["variant_index"],
        "target_uid": state["group"]["target_uid"],
        "target_name": state["group"]["target_name"],
        "task": state["task"],
        "policy_checkpoint": (getattr(args, "policy_metadata", None) or {}).get(
            "checkpoint"
        ),
        "policy_preprocess_profile": (
            getattr(args, "policy_metadata", None) or {}
        ).get("preprocess_profile"),
        "episode_summary": str(state["summary_path"]),
        "canonical_summary": str(state["canonical_summary_path"]),
        "reference_motion_steps": int(
            state["group"].get(
                "reference_motion_steps",
                state["reference_chunks"] * args.action_horizon,
            )
        ),
        "reference_chunk_count": state["reference_chunks"],
        "max_prediction_chunk_count": 2 * state["reference_chunks"],
        "prediction_chunk_count": len(state["calls"]),
        "stop_reason": state["stop_reason"],
        "outcome": outcome,
        "terminal_error": state.get("terminal_error"),
        "geometry_success": joint_success,
        "distance_success": bool(distance_success),
        "heading_success": bool(heading_success),
        "joint_success": joint_success,
        "strict_model_stop_success": bool(distance_success and state["stop_reason"] == "model_stop"),
        "final_base_xz": base_xz.astype(float).tolist(),
        "goal_base_xz": state["goal_base_xz"].astype(float).tolist(),
        "final_camera_xz": final_camera_xz.astype(float).tolist(),
        "standard_operating_point_xz": state["operating_xz"].astype(float).tolist(),
        "final_theta_rad": float(theta),
        "goal_theta_rad": float(state["goal_theta"]),
        "final_operating_point_distance_m": operating_distance,
        "final_base_position_error_m": base_position_error,
        "final_heading_deviation_rad": float(heading_error),
        "final_heading_deviation_deg": math.degrees(heading_error),
        "blocked_waypoints": state["blocked_waypoints"],
        "start_audit": state["start_audit"],
        "calls": state["calls"],
        "elapsed_s": float(time.monotonic() - state["started"]),
        "output_dir": str(state["output_dir"]),
    }
    if args.write_video:
        video_path = state["output_dir"] / "closedloop.mp4"
        bench.write_video(video_path, state["frames"])
        result["video"] = str(video_path)
    atomic_write_json(state["result_path"], result)
    return result


def process_batch(states: list[dict], client: Any, args: argparse.Namespace) -> list[dict]:
    observations = []
    seeds = []
    for state in states:
        # The official contract is per-episode call_index.  Batch size,
        # variant identity, and shard placement must not perturb this seed.
        seed = args.base_seed + state["call_index"]
        if not args.fixed_diffusion_seed:
            pass
        else:
            seed = args.base_seed
        seeds.append(seed)
        observations.append({
            "image": state["rgb"],
            "task": state["task"],
            "navigation_state": state["navigation_state"].astype(np.float32),
            "diffusion_seed": int(seed),
            "return_debug": True,
        })
    try:
        response = client.infer({"batch": observations})
    finally:
        if args.policy_one_shot:
            close_policy_client(client)
    replies = response.get("batch") if isinstance(response, dict) else None
    if not isinstance(replies, list) or len(replies) != len(states):
        raise RuntimeError(
            "Policy did not return a batch with matching length: "
            f"expected={len(states)} got={type(replies).__name__}/{len(replies) if isinstance(replies, list) else 'n/a'}"
        )
    finished = []
    for state, reply, seed in zip(states, replies, seeds):
        actions = np.asarray(reply.get("relative_se2_actions"), dtype=np.float32)
        if actions.shape != (args.action_horizon, 3) or not np.all(np.isfinite(actions)):
            raise RuntimeError(f"Invalid policy action shape/content: {actions.shape}")
        is_zero, zero_ratio = bench.near_zero_chunk(actions, args)
        state["near_zero_streak"] = state["near_zero_streak"] + 1 if is_zero else 0
        state["call_index"] += 1
        row = {
            "call_index": state["call_index"],
            "max_prediction_chunks": 2 * state["reference_chunks"],
            "diffusion_seed": int(seed),
            "base_xz_before": state["base_xz"].astype(float).tolist(),
            "theta_before": float(state["theta"]),
            "navigation_state_before": state["navigation_state"].astype(float).tolist(),
            "position_error_before_m": float(np.linalg.norm(
                bench.camera_xz_from_base(state["base_xz"], state["theta"]) - state["operating_xz"]
            )),
            "heading_error_before_deg": math.degrees(
                bench.angular_error_rad(state["theta"], state["goal_theta"])
            ),
            "endpoint": actions[-1].astype(float).tolist(),
            "relative_se2_actions": actions.astype(float).tolist(),
            "near_zero_ratio": zero_ratio,
            "near_zero_chunk": is_zero,
            "near_zero_streak": state["near_zero_streak"],
            "near_zero_required": args.near_zero_consecutive,
            "policy_timing": reply.get("policy_timing", {}),
        }
        raw_velocity = np.asarray(reply.get("raw_velocity_actions", []), dtype=np.float32)
        if raw_velocity.shape == (args.action_horizon, 2) and np.all(np.isfinite(raw_velocity)):
            row["raw_velocity_actions"] = raw_velocity.astype(float).tolist()
        full_actions = np.asarray(reply.get("actions", actions), dtype=np.float32)
        if full_actions.shape == (args.action_horizon, 9):
            row["bbox_endpoint"] = full_actions[-1, 3:7].astype(float).tolist()
            row["navigable_endpoint"] = full_actions[-1, 7:9].astype(float).tolist()
            row["bbox_prediction"] = full_actions[:, 3:7].astype(float).tolist()
            row["navigable_prediction"] = full_actions[:, 7:9].astype(float).tolist()
        state["calls"].append(row)
        if args.write_video:
            state["frames"].append(bench.draw_panel(state["rgb"], row))
        if state["near_zero_streak"] >= args.near_zero_consecutive:
            state["stop_reason"] = "model_stop"
            finished.append(state)
            continue
        state["base_xz"], state["theta"], blocked, motion_ratio = bench.execute_chunk(
            state["scene"], state["base_xz"], state["theta"], actions
        )
        next_state = np.asarray(reply.get("navigation_state_after", state["navigation_state"]), dtype=np.float32)
        if next_state.shape != (2,) or not np.all(np.isfinite(next_state)):
            raise RuntimeError(f"Invalid navigation_state_after shape/content: {next_state.shape}")
        state["navigation_state"] = next_state
        row["navigation_state_after"] = next_state.astype(float).tolist()
        state["blocked_waypoints"] += blocked
        row["blocked_waypoints"] = blocked
        row["accepted_motion_ratio"] = motion_ratio
        try:
            state["rgb"], _, _ = state["scene"].render(state["base_xz"], state["theta"])
        except RuntimeError as exc:
            # A policy can leave the navigable mesh.  End only that trajectory;
            # do not convert the other members of the active policy batch into
            # errors because one render failed.
            if "Cannot snap camera" not in str(exc):
                raise
            state["stop_reason"] = "out_of_navigable"
            state["terminal_error"] = f"{type(exc).__name__}: {exc}"
            row["terminal_event"] = "out_of_navigable"
            row["terminal_error"] = state["terminal_error"]
            finished.append(state)
            continue
        if state["call_index"] >= 2 * state["reference_chunks"]:
            finished.append(state)
    return finished


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
    }


def summarize(results: list[dict], errors: list[dict], args: argparse.Namespace, elapsed: float) -> dict:
    distances = [float(row["final_operating_point_distance_m"]) for row in results]
    headings = [float(row["final_heading_deviation_deg"]) for row in results]
    return {
        "benchmark": str(args.benchmark.resolve()),
        "benchmark_sha256": args.benchmark_sha256,
        "policy_endpoint": f"{args.policy_host}:{args.policy_port}",
        "policy_metadata": getattr(args, "policy_metadata", None),
        "batch_size": args.batch_size,
        "policy_retries": args.policy_retries,
        "policy_one_shot": args.policy_one_shot,
        "render_lock_path": (
            str(args.render_lock_path.resolve()) if args.render_lock_path else None
        ),
        "source_prefetch_workers": args.source_prefetch_workers,
        "group_offset": args.group_offset,
        "group_limit": args.group_limit,
        "target_shard": getattr(args, "target_shard_plan", None),
        "dynamic_batch_refill": True,
        "variant_mode": args.variant_mode,
        "action_horizon": args.action_horizon,
        "camera_wh": [args.camera_width, args.camera_height],
        "base_seed": args.base_seed,
        "fixed_diffusion_seed": args.fixed_diffusion_seed,
        "near_zero": {
            "ratio": args.near_zero_ratio,
            "position_m": args.near_zero_position_m,
            "heading_rad": args.near_zero_heading_rad,
            "consecutive": args.near_zero_consecutive,
        },
        "success_thresholds": {
            "position_m": args.success_position_m,
            "heading_deg": args.success_heading_deg,
        },
        "expected_episode_count": int(args.expected_episode_count),
        "resumed_episode_count": int(args.resumed_episode_count),
        "episode_count": len(results),
        "completed_episode_count": len(results) + len(errors),
        "error_count": len(errors),
        "outcomes": {
            outcome: sum(row["outcome"] == outcome for row in results)
            for outcome in sorted({row["outcome"] for row in results})
        },
        "stop_reasons": {
            reason: sum(row["stop_reason"] == reason for row in results)
            for reason in sorted({row["stop_reason"] for row in results})
        },
        "distance_success_rate": float(np.mean([r["distance_success"] for r in results])) if results else 0.0,
        "heading_success_rate": float(np.mean([r["heading_success"] for r in results])) if results else 0.0,
        "joint_success_rate": float(np.mean([r["joint_success"] for r in results])) if results else 0.0,
        "strict_model_stop_success_rate": float(np.mean([r["strict_model_stop_success"] for r in results])) if results else 0.0,
        "premature_stop_rate": float(np.mean([r["outcome"] == "premature_stop" for r in results])) if results else 0.0,
        "timeout_rate": float(np.mean([r["outcome"] == "timeout" for r in results])) if results else 0.0,
        "mean_final_operating_point_distance_m": float(np.mean(distances)) if results else None,
        "mean_final_heading_deviation_deg": float(np.mean(headings)) if results else None,
        "final_operating_point_distance_m": distribution(distances),
        "final_heading_deviation_deg": distribution(headings),
        "mean_prediction_chunks": float(np.mean([r["prediction_chunk_count"] for r in results])) if results else None,
        "blocked_waypoints_total": int(sum(r["blocked_waypoints"] for r in results)),
        "elapsed_s": elapsed,
        "errors": errors,
    }


def main() -> int:
    args = parse_args()
    if args.data_root is not None:
        bench.configure_data_root(args.data_root)
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.policy_retries < 0:
        raise ValueError("policy-retries must be non-negative")
    if args.source_prefetch_workers < 0:
        raise ValueError("source-prefetch-workers must be non-negative")
    if args.near_zero_consecutive <= 0:
        raise ValueError("near-zero-consecutive must be positive")
    if args.target_shard_count > 1 and (args.group_offset or args.group_limit):
        raise ValueError(
            "target sharding cannot be combined with group-offset/group-limit"
        )
    if not args.benchmark.is_file():
        raise FileNotFoundError(args.benchmark)
    args.benchmark_sha256 = hashlib.sha256(args.benchmark.read_bytes()).hexdigest()
    bench.CAMERA_WIDTH = args.camera_width
    bench.CAMERA_HEIGHT = args.camera_height
    benchmark = load_json(args.benchmark)
    selected_groups, args.target_shard_plan = target_balanced_groups(
        benchmark["groups"], args.target_shard_count, args.target_shard_index
    )
    flat = flatten_episodes({"groups": selected_groups}, args.variant_mode)
    if args.group_offset:
        flat = flat[args.group_offset:]
    if args.group_limit > 0:
        flat = flat[:args.group_limit]
    args.expected_episode_count = len(flat)
    args.policy_metadata = None
    args.render_lock_file = None
    if args.render_lock_path is not None:
        args.render_lock_path.parent.mkdir(parents=True, exist_ok=True)
        args.render_lock_file = args.render_lock_path.open("a+")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    if args.resume:
        pending = []
        for item in flat:
            result_path = episode_result_path(args.output_dir, item)
            if not result_path.is_file():
                pending.append(item)
                continue
            try:
                result = load_json(result_path)
                if result.get("group_id") != item[0]["group_id"]:
                    raise ValueError("group_id mismatch")
                results.append(result)
            except Exception:
                pending.append(item)
        flat = pending
    args.resumed_episode_count = len(results)
    if args.resumed_episode_count:
        print(
            f"resume={args.resumed_episode_count}/{args.expected_episode_count} "
            f"pending={len(flat)}",
            flush=True,
        )
    errors: list[dict] = []
    started = time.monotonic()
    if not flat:
        summary = summarize(results, errors, args, time.monotonic() - started)
        atomic_write_json(args.output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
        return 0
    modules = bench.load_render_modules()
    source_prefetcher = None
    if args.source_prefetch_workers:
        source_prefetcher = SourceFramePrefetcher(
            bench.decode_first_rgb,
            args.source_prefetch_workers,
        )
        bench.decode_first_rgb = source_prefetcher.decode
    client = load_policy_client(args.policy_host, args.policy_port)
    if hasattr(client, "get_server_metadata"):
        args.policy_metadata = client.get_server_metadata()
    if args.policy_one_shot:
        close_policy_client(client)
        client = None
    by_target: dict[str, list[tuple[dict, str, int, dict]]] = {}
    for item in flat:
        by_target.setdefault(str(item[0]["target_uid"]), []).append(item)

    request_index = 0
    for target_number, (target_uid, target_items) in enumerate(by_target.items(), start=1):
        if source_prefetcher is not None:
            source_prefetcher.prepare(target_items)
        # Keep one Habitat SceneContext per target. Each active episode still
        # carries independent pose, stop, and navigation state.
        target_scene = None
        try:
            with render_guard(args):
                target_scene = bench.SceneContext(str(target_uid), modules)
            install_render_guard(target_scene, args)
        except Exception as exc:
            for group, family, variant_index, _episode in target_items:
                error = {
                    "stage": "target_scene_initialization",
                    "request_index": request_index,
                    "group_id": group["group_id"],
                    "target_uid": group["target_uid"],
                    "family": family,
                    "variant_index": variant_index,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                errors.append(error)
                print(json.dumps(error), flush=True)
            if args.fail_fast:
                raise
            print(
                f"target_complete={target_number}/{len(by_target)} target_uid={target_uid} "
                f"completed={len(results) + len(errors)}/{args.expected_episode_count}",
                flush=True,
            )
            continue

        item_cursor = 0
        active: list[dict] = []
        while active or item_cursor < len(target_items):
            # Keep the policy batch full.  A completed episode is replaced on
            # the very next request instead of waiting for the slowest member
            # of a fixed batch.
            while len(active) < args.batch_size and item_cursor < len(target_items):
                group, family, variant_index, episode = target_items[item_cursor]
                item_cursor += 1
                try:
                    state = make_state(
                        target_scene, group, family, variant_index, episode, args
                    )
                    active.append(state)
                except Exception as exc:
                    error = {
                        "stage": "episode_initialization",
                        "request_index": request_index,
                        "group_id": group["group_id"],
                        "target_uid": group["target_uid"],
                        "family": family,
                        "variant_index": variant_index,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    errors.append(error)
                    print(json.dumps(error), flush=True)
                    if args.fail_fast:
                        raise
            if not active:
                continue

            request_index += 1
            finished = None
            policy_error = None
            for attempt in range(args.policy_retries + 1):
                try:
                    if args.policy_one_shot:
                        client = load_policy_client(args.policy_host, args.policy_port)
                    finished = process_batch(active, client, args)
                    policy_error = None
                    break
                except Exception as exc:
                    policy_error = exc
                    is_disconnect = type(exc).__name__.startswith("ConnectionClosed")
                    if not is_disconnect or attempt >= args.policy_retries:
                        break
                    print(
                        f"policy_reconnect request={request_index} "
                        f"attempt={attempt + 1}/{args.policy_retries} error={exc}",
                        flush=True,
                    )
                    if not args.policy_one_shot:
                        close_policy_client(client)
                        client = load_policy_client(args.policy_host, args.policy_port)

            if policy_error is not None:
                exc = policy_error
                for state in active:
                    error = {
                        "stage": "policy_or_execution",
                        "request_index": request_index,
                        "group_id": state["group"]["group_id"],
                        "target_uid": state["group"]["target_uid"],
                        "family": state["family"],
                        "variant_index": state["variant_index"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    errors.append(error)
                    print(json.dumps(error), flush=True)
                active = []
                if args.fail_fast:
                    raise
                continue

            assert finished is not None

            for state in finished:
                result = finalize_state(state, args)
                results.append(result)
                print(
                    f"progress={len(results)}/{args.expected_episode_count} "
                    f"request={request_index} target={state['group']['target_name']} "
                    f"outcome={result['outcome']} "
                    f"op_dist={result['final_operating_point_distance_m']:.3f} "
                    f"heading={result['final_heading_deviation_deg']:.1f}",
                    flush=True,
                )
            finished_ids = {id(state) for state in finished}
            active = [state for state in active if id(state) not in finished_ids]

        print(
            f"target_complete={target_number}/{len(by_target)} target_uid={target_uid} "
            f"completed={len(results) + len(errors)}/{args.expected_episode_count}",
            flush=True,
        )
        atomic_write_json(
            args.output_dir / "summary_live.json",
            summarize(results, errors, args, time.monotonic() - started),
        )
        try:
            with render_guard(args):
                target_scene.close()
        except Exception as exc:
            error = {
                "stage": "target_scene_close",
                "target_uid": target_uid,
                "error": f"{type(exc).__name__}: {exc}",
            }
            errors.append(error)
            print(json.dumps(error), flush=True)
            if args.fail_fast:
                raise
    summary = summarize(results, errors, args, time.monotonic() - started)
    atomic_write_json(args.output_dir / "summary.json", summary)
    if source_prefetcher is not None:
        source_prefetcher.close()
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
