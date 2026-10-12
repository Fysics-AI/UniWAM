#!/usr/bin/env python3
"""Closed-loop HM3D benchmark for cumulative relative-SE(2) policies."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
RELEASE_ROOT = HERE.parents[2]
DATA_ROOT = Path(os.environ.get("MAP_BENCH_DATA_ROOT", str(RELEASE_ROOT))).resolve()
SCENE_CONTEXT_DIR = DATA_ROOT / "evaluation" / "scene_contexts"
SCRIPTS_DIR = HERE.parent / "habitat_scripts"
HM3D_SCENE_ROOT_VALUE = os.environ.get("HM3D_SCENE_ROOT")
HM3D_SCENE_ROOT = Path(HM3D_SCENE_ROOT_VALUE) if HM3D_SCENE_ROOT_VALUE else None
DEFAULT_BENCHMARK = DATA_ROOT / "evaluation" / "benchmark_15k.json"
DEFAULT_OUTPUT = Path("outputs/closedloop")
CAMERA_WIDTH = 848
CAMERA_HEIGHT = 480
CAMERA_HEIGHT_M = 1.22
CAMERA_FORWARD_M = 0.15
CAMERA_PITCH_DEG = 22.5
CAMERA_HFOV_DEG = 89.451536
DEFAULT_ACTION_HORIZON = 32


def configure_data_root(data_root: str | Path) -> None:
    """Point the evaluator at a separate data checkout or Hugging Face clone."""
    global DATA_ROOT, SCENE_CONTEXT_DIR
    DATA_ROOT = Path(data_root).expanduser().resolve()
    SCENE_CONTEXT_DIR = DATA_ROOT / "evaluation" / "scene_contexts"


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
    parser.add_argument("--group-limit", type=int, default=0)
    parser.add_argument("--group-offset", type=int, default=0)
    parser.add_argument("--variant-mode", choices=("canonical", "all"), default="all")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--shard-mode",
        choices=("index", "target_uid", "target_uid_balanced"),
        default="index",
        help=(
            "Keep all episodes for one target on one shard with target_uid; "
            "target_uid_balanced additionally balances expected prediction chunks."
        ),
    )
    parser.add_argument("--write-video", action="store_true")
    parser.add_argument("--save-start-audit", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--connection-retries", type=int, default=3)
    parser.add_argument("--near-zero-ratio", type=float, default=0.80)
    parser.add_argument("--near-zero-position-m", type=float, default=0.05)
    parser.add_argument("--near-zero-heading-rad", type=float, default=0.05)
    parser.add_argument("--near-zero-consecutive", type=int, default=1)
    parser.add_argument("--success-position-m", type=float, default=0.50)
    parser.add_argument("--success-heading-deg", type=float, default=15.0)
    parser.add_argument("--start-camera-tolerance-m", type=float, default=0.003)
    parser.add_argument("--start-yaw-tolerance-deg", type=float, default=0.02)
    parser.add_argument("--start-pixel-mae-limit", type=float, default=25.0)
    parser.add_argument("--base-seed", type=int, default=20260807)
    parser.add_argument("--fixed-diffusion-seed", action="store_true")
    parser.add_argument("--action-horizon", type=int, choices=(16, 32), default=DEFAULT_ACTION_HORIZON)
    parser.add_argument("--camera-width", type=int, default=CAMERA_WIDTH)
    parser.add_argument("--camera-height", type=int, default=CAMERA_HEIGHT)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def balanced_target_assignment(groups: list[dict], shard_count: int) -> dict[str, int]:
    target_weights: dict[str, int] = {}
    for group in groups:
        target_uid = str(group["target_uid"])
        target_weights[target_uid] = target_weights.get(target_uid, 0) + int(
            group.get("max_prediction_chunk_count", 1)
        )
    shard_weights = [0] * shard_count
    assignment = {}
    ordered = sorted(
        target_weights,
        key=lambda target_uid: (
            -target_weights[target_uid],
            hashlib.sha256(target_uid.encode("utf-8")).digest(),
        ),
    )
    for target_uid in ordered:
        shard_index = min(
            range(shard_count), key=lambda index: (shard_weights[index], index)
        )
        assignment[target_uid] = shard_index
        shard_weights[shard_index] += target_weights[target_uid]
    return assignment


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_render_modules() -> dict[str, Any]:
    curve = load_module(HERE / "render_observer_to_operating_curve_video.py", "_se2_eval_curve")
    base_views = curve.load_module(HERE / "generate_accepted_base_views.py", "_se2_eval_base_views")
    render_fp = curve.load_module(HERE / "render_fp_from_observer_candidates.py", "_se2_eval_render_fp")
    base_mod = base_views.load_module(
        SCRIPTS_DIR / "render_local_topdown.py",
        "_se2_eval_base_mod",
    )
    prod = load_module(HERE / "hm3d_navigation_geometry.py", "_se2_eval_prod")
    return {
        "curve": curve,
        "base_views": base_views,
        "render_fp": render_fp,
        "base_mod": base_mod,
        "prod": prod,
    }


def load_policy_client(host: str, port: int) -> Any:
    for path in (
        HERE.parent / "openpi_client_src",
        HERE / ".policy_ws_deps_nodeps",
    ):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    return WebsocketClientPolicy(host=host, port=port)


def wrap_rad(value: float) -> float:
    return float((float(value) + math.pi) % (2.0 * math.pi) - math.pi)


def angular_error_rad(value: float, target: float) -> float:
    return abs(wrap_rad(float(value) - float(target)))


def camera_xz_from_base(base_xz: np.ndarray, theta: float) -> np.ndarray:
    return np.asarray(base_xz, dtype=np.float32) + CAMERA_FORWARD_M * np.asarray(
        [math.cos(theta), math.sin(theta)], dtype=np.float32
    )


def safe_name(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    stem = "".join(character if character.isalnum() or character in "._-" else "_" for character in value)
    return f"{stem[:120]}__{digest}"


def episode_summary(episode: dict) -> tuple[Path, dict]:
    path = Path(episode["annotation_path"]).parent / "summary.json"
    return path, load_json(path)


def first_annotation(episode: dict) -> dict:
    path = Path(episode["annotation_path"])
    with path.open("r", encoding="utf-8") as stream:
        line = stream.readline()
    if not line:
        raise ValueError(f"Empty annotation file: {path}")
    return json.loads(line)


def decode_first_rgb(video_path: str) -> np.ndarray:
    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        check=True,
        stdout=subprocess.PIPE,
    )
    expected_bytes = CAMERA_WIDTH * CAMERA_HEIGHT * 3
    if len(completed.stdout) != expected_bytes:
        raise RuntimeError(
            f"Invalid decoded first frame size for {video_path}: "
            f"expected {expected_bytes} bytes, got {len(completed.stdout)}"
        )
    return np.frombuffer(completed.stdout, dtype=np.uint8).reshape(
        CAMERA_HEIGHT, CAMERA_WIDTH, 3
    ).copy()


class SceneContext:
    def __init__(self, target_uid: str, modules: dict[str, Any]) -> None:
        self.target_uid = target_uid
        self.modules = modules
        if HM3D_SCENE_ROOT is None:
            raise RuntimeError("Set HM3D_SCENE_ROOT to HM3D v0.2 scenes_train_val")
        context = load_json(SCENE_CONTEXT_DIR / f"{target_uid}.json")
        trajectory = dict(context["trajectory"])
        self.nav_y = float(context["nav_y"])
        self.semantic_ids = [int(value) for value in context.get("semantic_ids", []) if int(value) > 0]
        scene_source = str(HM3D_SCENE_ROOT / trajectory["scene_id"])
        trajectory["scene_path"] = scene_source
        trajectory["dataset_config"] = str(
            HM3D_SCENE_ROOT.parent / "hm3d_annotated_basis.scene_dataset_config.json"
        )
        self.sim = modules["base_views"].build_sim_for_target(
            base_mod=modules["base_mod"],
            traj=trajectory,
            scene_source=scene_source,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            sensor_height_m=CAMERA_HEIGHT_M,
            hfov_deg=CAMERA_HFOV_DEG,
        )

    def close(self) -> None:
        self.sim.close()

    def render(self, base_xz: np.ndarray, theta: float) -> tuple[np.ndarray, np.ndarray, float]:
        forward = np.asarray([math.cos(theta), math.sin(theta)], dtype=np.float32)
        requested_camera = np.asarray(base_xz, dtype=np.float32) + CAMERA_FORWARD_M * forward
        snapped_camera = self.modules["render_fp"].precise_snap(
            self.sim.pathfinder,
            float(requested_camera[0]),
            float(requested_camera[1]),
            self.nav_y,
        )
        if snapped_camera is None:
            raise RuntimeError(f"Cannot snap camera at {requested_camera.tolist()}")
        snapped_camera = np.asarray(snapped_camera, dtype=np.float32)
        yaw_deg = self.modules["prod"].planar_theta_to_hm3d_yaw_deg(theta)
        observation = self.modules["curve"].render_first_person_observation_eye_locked(
            base_mod=self.modules["base_mod"],
            sim=self.sim,
            nominal_base_world_xyz=np.asarray(
                [float(snapped_camera[0]), self.nav_y, float(snapped_camera[2])], dtype=np.float32
            ),
            yaw_deg=float(yaw_deg),
            pitch_down_deg=CAMERA_PITCH_DEG,
            sensor_height_m=CAMERA_HEIGHT_M,
        )
        rgb = np.asarray(observation["rgb"], dtype=np.uint8)
        if rgb.ndim == 3 and rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        if rgb.shape != (CAMERA_HEIGHT, CAMERA_WIDTH, 3):
            raise ValueError(f"Unexpected rendered RGB shape {rgb.shape}")
        return rgb, snapped_camera[[0, 2]], float(yaw_deg)

    def collision_filter(self, current_xz: np.ndarray, desired_xz: np.ndarray) -> tuple[np.ndarray, bool]:
        start = np.asarray([float(current_xz[0]), self.nav_y, float(current_xz[1])], dtype=np.float32)
        end = np.asarray([float(desired_xz[0]), self.nav_y, float(desired_xz[1])], dtype=np.float32)
        filtered = np.asarray(self.sim.pathfinder.try_step_no_sliding(start, end), dtype=np.float32)
        if filtered.shape != (3,) or not np.all(np.isfinite(filtered)):
            raise RuntimeError(f"Pathfinder returned invalid point {filtered}")
        requested = float(np.linalg.norm(end[[0, 2]] - start[[0, 2]]))
        accepted = float(np.linalg.norm(filtered[[0, 2]] - start[[0, 2]]))
        blocked = bool(requested > 1e-5 and accepted < 0.5 * requested)
        return filtered[[0, 2]].astype(np.float32), blocked


def validate_camera_contract(summary: dict) -> None:
    model = summary.get("camera_model", {})
    expected = {
        "width": CAMERA_WIDTH,
        "height": CAMERA_HEIGHT,
        "height_m": CAMERA_HEIGHT_M,
        "forward_offset_m": CAMERA_FORWARD_M,
        "pitch_down_deg": CAMERA_PITCH_DEG,
        "hfov_used_deg": CAMERA_HFOV_DEG,
    }
    for key, value in expected.items():
        actual = model.get(key)
        if actual is None or abs(float(actual) - float(value)) > 1e-5:
            raise ValueError(f"Camera contract mismatch {key}: expected {value}, got {actual}")


def start_pose_audit(
    scene: SceneContext,
    episode: dict,
    summary: dict,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[np.ndarray, float, np.ndarray, dict]:
    if "start_base_xz" in episode:
        base_xz = np.asarray(episode["start_base_xz"], dtype=np.float32)
        theta = float(episode["start_theta_rad"])
        expected_camera = np.asarray(episode["start_camera_xz"], dtype=np.float32)
        computed_camera = camera_xz_from_base(base_xz, theta)
        rendered, render_camera, render_yaw = scene.render(base_xz, theta)
        source_path = str(episode.get("video_path", ""))
        pixel_mae = None
        source = None
        if source_path:
            source = decode_first_rgb(source_path)
            pixel_mae = float(
                np.mean(np.abs(source.astype(np.float32) - rendered.astype(np.float32)))
            )
        audit = {
            "mode": "source_video_and_geometry" if source is not None else "geometry_only",
            "computed_vs_expected_camera_m": float(np.linalg.norm(computed_camera - expected_camera)),
            "render_vs_expected_camera_m": float(np.linalg.norm(render_camera - expected_camera)),
            "source_render_pixel_mae": pixel_mae,
            "render_yaw_deg": float(render_yaw),
        }
        if max(
            audit["computed_vs_expected_camera_m"],
            audit["render_vs_expected_camera_m"],
        ) > args.start_camera_tolerance_m:
            raise RuntimeError(f"Start camera mismatch: {audit}")
        if pixel_mae is not None and pixel_mae > args.start_pixel_mae_limit:
            raise RuntimeError(f"Start render does not match source video: {audit}")
        if args.save_start_audit:
            output_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(output_dir / "start_render.png"), cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
            )
            if source is not None:
                cv2.imwrite(
                    str(output_dir / "start_source.png"), cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
                )
                difference = np.clip(
                    np.abs(source.astype(np.int16) - rendered.astype(np.int16)) * 4, 0, 255
                ).astype(np.uint8)
                cv2.imwrite(
                    str(output_dir / "start_diff_x4.png"),
                    cv2.cvtColor(difference, cv2.COLOR_RGB2BGR),
                )
        return base_xz, theta, rendered, audit

    validate_camera_contract(summary)
    base_xz = np.asarray(summary["start"]["base_xy"], dtype=np.float32)
    theta = float(summary["start"]["theta_rad"])
    expected_camera = np.asarray(summary["start"]["camera_xz"], dtype=np.float32)
    annotation = first_annotation(episode)
    annotation_camera = np.asarray(annotation["camera_world_xz"], dtype=np.float32)
    expected_yaw = float(summary["start"]["yaw_deg"])
    annotation_yaw = float(annotation["hm3d_yaw_deg"])
    computed_camera = base_xz + CAMERA_FORWARD_M * np.asarray(
        [math.cos(theta), math.sin(theta)], dtype=np.float32
    )
    rendered, render_camera, render_yaw = scene.render(base_xz, theta)
    source = decode_first_rgb(episode["video_path"])
    if source.shape != rendered.shape:
        raise ValueError(f"Source/render shape mismatch {source.shape} != {rendered.shape}")
    pixel_mae = float(np.mean(np.abs(source.astype(np.float32) - rendered.astype(np.float32))))
    audit = {
        "computed_vs_summary_camera_m": float(np.linalg.norm(computed_camera - expected_camera)),
        "annotation_vs_summary_camera_m": float(np.linalg.norm(annotation_camera - expected_camera)),
        "render_vs_summary_camera_m": float(np.linalg.norm(render_camera - expected_camera)),
        "annotation_vs_summary_yaw_deg": float(abs(wrap_rad(math.radians(annotation_yaw - expected_yaw))) * 180.0 / math.pi),
        "render_vs_summary_yaw_deg": float(abs(wrap_rad(math.radians(render_yaw - expected_yaw))) * 180.0 / math.pi),
        "source_render_pixel_mae": pixel_mae,
    }
    camera_error = max(
        audit["computed_vs_summary_camera_m"],
        audit["annotation_vs_summary_camera_m"],
        audit["render_vs_summary_camera_m"],
    )
    yaw_error = max(audit["annotation_vs_summary_yaw_deg"], audit["render_vs_summary_yaw_deg"])
    if camera_error > args.start_camera_tolerance_m:
        raise RuntimeError(f"Start camera mismatch: {audit}")
    if yaw_error > args.start_yaw_tolerance_deg:
        raise RuntimeError(f"Start yaw mismatch: {audit}")
    if pixel_mae > args.start_pixel_mae_limit:
        raise RuntimeError(f"Start render does not match source video: {audit}")
    if args.save_start_audit:
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / "start_source.png"), cv2.cvtColor(source, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output_dir / "start_render.png"), cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
        difference = np.clip(np.abs(source.astype(np.int16) - rendered.astype(np.int16)) * 4, 0, 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / "start_diff_x4.png"), cv2.cvtColor(difference, cv2.COLOR_RGB2BGR))
    return base_xz, theta, rendered, audit


def near_zero_chunk(actions: np.ndarray, args: argparse.Namespace) -> tuple[bool, float]:
    position = np.linalg.norm(actions[:, :2], axis=1)
    heading = np.abs(actions[:, 2])
    mask = (position <= args.near_zero_position_m) & (heading <= args.near_zero_heading_rad)
    ratio = float(np.mean(mask))
    endpoint_zero = bool(mask[-1])
    return bool(ratio >= args.near_zero_ratio and endpoint_zero), ratio


def execute_chunk(
    scene: SceneContext,
    origin_xz: np.ndarray,
    origin_theta: float,
    actions: np.ndarray,
) -> tuple[np.ndarray, float, int, float]:
    forward = np.asarray([math.cos(origin_theta), math.sin(origin_theta)], dtype=np.float32)
    left = np.asarray([-math.sin(origin_theta), math.cos(origin_theta)], dtype=np.float32)
    current = np.asarray(origin_xz, dtype=np.float32).copy()
    blocked_count = 0
    requested_distance = 0.0
    accepted_distance = 0.0
    for waypoint in actions:
        desired = origin_xz + forward * float(waypoint[0]) + left * float(waypoint[1])
        requested_distance += float(np.linalg.norm(desired - current))
        filtered, blocked = scene.collision_filter(current, desired)
        accepted_distance += float(np.linalg.norm(filtered - current))
        blocked_count += int(blocked)
        current = filtered
    theta = wrap_rad(origin_theta + float(actions[-1, 2]))
    motion_ratio = accepted_distance / max(requested_distance, 1e-8)
    return current, theta, blocked_count, float(motion_ratio)


def draw_panel(rgb: np.ndarray, row: dict) -> np.ndarray:
    panel = np.asarray(rgb, dtype=np.uint8).copy()
    lines = [
        f"call {row['call_index']}/{row['max_prediction_chunks']}",
        f"endpoint x={row['endpoint'][0]:+.3f} y={row['endpoint'][1]:+.3f} h={row['endpoint'][2]:+.3f}",
        f"zero_ratio={row['near_zero_ratio']:.2f} stop={row['near_zero_chunk']}",
        f"goal pos={row['position_error_before_m']:.3f}m yaw={row['heading_error_before_deg']:.1f}deg",
    ]
    y = 28
    for line in lines:
        cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        y += 28
    return panel


def write_video(path: Path, frames: list[np.ndarray]) -> None:
    if not frames:
        return
    temporary = path.with_name(path.stem + "_tmp_mp4v.mp4")
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (CAMERA_WIDTH, CAMERA_HEIGHT)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer {temporary}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(temporary),
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
        ],
        check=True,
    )
    temporary.unlink(missing_ok=True)


def run_episode(
    scene: SceneContext,
    client: Any,
    group: dict,
    episode: dict,
    family: str,
    variant_index: int,
    args: argparse.Namespace,
) -> dict:
    identity = f"{group['group_id']}::{family}_{variant_index:02d}"
    output_dir = args.output_dir / "episodes" / safe_name(identity)
    result_path = output_dir / "result.json"
    if args.resume and result_path.is_file():
        return load_json(result_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    synthetic = "start_base_xz" in episode
    if synthetic:
        summary_path = Path(group.get("plan_summary_json") or args.benchmark)
        canonical_summary_path = summary_path
        summary = canonical_summary = {}
    else:
        summary_path, summary = episode_summary(episode)
        canonical_summary_path, canonical_summary = episode_summary(group["canonical"])
    base_xz, theta, rgb, start_audit = start_pose_audit(scene, episode, summary, output_dir, args)
    if synthetic:
        goal_base_xz = np.asarray(group["goal_base_xz"], dtype=np.float32)
        operating_xz = np.asarray(group["operating_camera_xz"], dtype=np.float32)
        goal_theta = float(group["goal_theta_rad"])
    else:
        goal_base_xz = np.asarray(canonical_summary["goal"]["base_xy"], dtype=np.float32)
        operating_xz = np.asarray(canonical_summary["goal"]["camera_xz"], dtype=np.float32)
        goal_theta = float(canonical_summary["goal"]["theta_rad"])
    if args.audit_only:
        result = {
            "identity": identity,
            "group_id": group["group_id"],
            "family": family,
            "variant_index": variant_index,
            "target_uid": group["target_uid"],
            "target_name": group["target_name"],
            "task": str(group["task"]),
            "episode_summary": str(summary_path),
            "canonical_summary": str(canonical_summary_path),
            "prediction_chunk_count": 0,
            "outcome": "audit_only",
            "geometry_success": False,
            "distance_success": False,
            "heading_success": False,
            "joint_success": False,
            "strict_model_stop_success": False,
            "final_operating_point_distance_m": float("nan"),
            "final_heading_deviation_deg": float("nan"),
            "start_audit": start_audit,
            "output_dir": str(output_dir),
        }
        temporary = result_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        temporary.replace(result_path)
        return result
    reference_chunks = int(group.get(
        "reference_chunk_count",
        math.ceil(int(group["canonical"]["frame_count"]) / args.action_horizon),
    ))
    max_chunks = 2 * reference_chunks
    task = str(group["task"])
    calls = []
    frames = []
    stop_reason = "budget_exhausted"
    blocked_waypoints = 0
    near_zero_streak = 0
    started = time.monotonic()

    for call_index in range(max_chunks):
        camera_before = camera_xz_from_base(base_xz, theta)
        position_before = float(np.linalg.norm(camera_before - operating_xz))
        heading_before = angular_error_rad(theta, goal_theta)
        seed = int(
            args.base_seed
            + variant_index * 1000
            + (0 if args.fixed_diffusion_seed else call_index)
        )
        reply = client.infer({
            "image": rgb,
            "task": task,
            "diffusion_seed": seed,
            "return_debug": True,
        })
        actions = np.asarray(reply.get("relative_se2_actions"), dtype=np.float32)
        if actions.shape != (args.action_horizon, 3) or not np.all(np.isfinite(actions)):
            raise RuntimeError(f"Invalid policy action shape/content: {actions.shape}")
        is_zero, zero_ratio = near_zero_chunk(actions, args)
        near_zero_streak = near_zero_streak + 1 if is_zero else 0
        row = {
            "call_index": call_index + 1,
            "max_prediction_chunks": max_chunks,
            "diffusion_seed": seed,
            "base_xz_before": base_xz.astype(float).tolist(),
            "theta_before": float(theta),
            "position_error_before_m": position_before,
            "heading_error_before_deg": math.degrees(heading_before),
            "endpoint": actions[-1].astype(float).tolist(),
            "near_zero_ratio": zero_ratio,
            "near_zero_chunk": is_zero,
            "near_zero_streak": near_zero_streak,
            "near_zero_required": args.near_zero_consecutive,
            "policy_timing": reply.get("policy_timing", {}),
        }
        full_actions = np.asarray(reply.get("actions", actions), dtype=np.float32)
        if full_actions.shape == (args.action_horizon, 9):
            row["bbox_endpoint"] = full_actions[-1, 3:7].astype(float).tolist()
            row["navigable_endpoint"] = full_actions[-1, 7:9].astype(float).tolist()
            row["bbox_prediction"] = full_actions[:, 3:7].astype(float).tolist()
            row["navigable_prediction"] = full_actions[:, 7:9].astype(float).tolist()
        calls.append(row)
        if args.write_video:
            frames.append(draw_panel(rgb, row))
        if near_zero_streak >= args.near_zero_consecutive:
            stop_reason = "model_stop"
            break
        base_xz, theta, blocked, motion_ratio = execute_chunk(scene, base_xz, theta, actions)
        blocked_waypoints += blocked
        row["blocked_waypoints"] = blocked
        row["accepted_motion_ratio"] = motion_ratio
        rgb, _, _ = scene.render(base_xz, theta)

    final_camera_xz = camera_xz_from_base(base_xz, theta)
    operating_distance = float(np.linalg.norm(final_camera_xz - operating_xz))
    base_position_error = float(np.linalg.norm(base_xz - goal_base_xz))
    heading_error = angular_error_rad(theta, goal_theta)
    distance_success = bool(operating_distance <= args.success_position_m)
    heading_success = bool(heading_error <= math.radians(args.success_heading_deg))
    joint_success = bool(distance_success and heading_success)
    if stop_reason == "model_stop":
        # The primary navigation criterion is reaching the operating point.
        # Heading remains an independent quality metric and is also reported
        # through joint_success for analyses that require both.
        outcome = "success" if distance_success else "premature_stop"
    else:
        outcome = "timeout_at_goal" if distance_success else "timeout"
    result = {
        "identity": identity,
        "group_id": group["group_id"],
        "family": family,
        "variant_index": variant_index,
        "target_uid": group["target_uid"],
        "target_name": group["target_name"],
        "task": task,
        "episode_summary": str(summary_path),
        "canonical_summary": str(canonical_summary_path),
        "reference_chunk_count": reference_chunks,
        "max_prediction_chunk_count": max_chunks,
        "prediction_chunk_count": len(calls),
        "stop_reason": stop_reason,
        "outcome": outcome,
        "geometry_success": joint_success,
        "distance_success": distance_success,
        "heading_success": heading_success,
        "joint_success": joint_success,
        "strict_model_stop_success": bool(distance_success and stop_reason == "model_stop"),
        "final_base_xz": base_xz.astype(float).tolist(),
        "goal_base_xz": goal_base_xz.astype(float).tolist(),
        "final_camera_xz": final_camera_xz.astype(float).tolist(),
        "standard_operating_point_xz": operating_xz.astype(float).tolist(),
        "final_theta_rad": float(theta),
        "goal_theta_rad": float(goal_theta),
        "final_operating_point_distance_m": operating_distance,
        "final_base_position_error_m": base_position_error,
        "final_heading_deviation_rad": heading_error,
        "final_heading_deviation_deg": math.degrees(heading_error),
        "blocked_waypoints": blocked_waypoints,
        "start_audit": start_audit,
        "calls": calls,
        "elapsed_s": float(time.monotonic() - started),
        "output_dir": str(output_dir),
    }
    if args.write_video:
        video_path = output_dir / "closedloop.mp4"
        write_video(video_path, frames)
        result["video"] = str(video_path)
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(result_path)
    return result


def aggregate(results: list[dict], errors: list[dict], args: argparse.Namespace) -> dict:
    outcomes: dict[str, int] = {}
    for result in results:
        outcomes[result["outcome"]] = outcomes.get(result["outcome"], 0) + 1
    count = len(results)
    return {
        "benchmark": str(args.benchmark.resolve()),
        "policy_endpoint": f"{args.policy_host}:{args.policy_port}",
        "action_horizon": args.action_horizon,
        "shard": {"index": args.shard_index, "count": args.shard_count},
        "shard_mode": args.shard_mode,
        "episode_count": count,
        "error_count": len(errors),
        "outcomes": outcomes,
        "distance_success_rate": float(
            np.mean([row["distance_success"] for row in results])
        ) if results else 0.0,
        "heading_success_rate": float(
            np.mean([row["heading_success"] for row in results])
        ) if results else 0.0,
        "joint_success_rate": float(
            np.mean([row["joint_success"] for row in results])
        ) if results else 0.0,
        "geometry_success_rate": float(np.mean([row["geometry_success"] for row in results])) if results else 0.0,
        "strict_model_stop_success_rate": float(
            np.mean([row["strict_model_stop_success"] for row in results])
        ) if results else 0.0,
        "mean_final_operating_point_distance_m": float(
            np.mean([row["final_operating_point_distance_m"] for row in results])
        ) if results else None,
        "mean_final_heading_deviation_deg": float(
            np.mean([row["final_heading_deviation_deg"] for row in results])
        ) if results else None,
        "mean_prediction_chunks": float(
            np.mean([row["prediction_chunk_count"] for row in results])
        ) if results else None,
        "errors": errors,
    }


def main() -> int:
    global CAMERA_WIDTH, CAMERA_HEIGHT
    args = parse_args()
    if args.data_root is not None:
        configure_data_root(args.data_root)
    CAMERA_WIDTH = int(args.camera_width)
    CAMERA_HEIGHT = int(args.camera_height)
    if CAMERA_WIDTH <= 0 or CAMERA_HEIGHT <= 0:
        raise ValueError("Camera dimensions must be positive")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid shard index/count")
    if args.near_zero_consecutive <= 0:
        raise ValueError("near-zero-consecutive must be positive")
    if args.connection_retries < 0:
        raise ValueError("connection-retries must be non-negative")
    benchmark = load_json(args.benchmark)
    if args.shard_mode == "index":
        groups = [
            group for index, group in enumerate(benchmark["groups"])
            if index % args.shard_count == args.shard_index
        ]
    elif args.shard_mode == "target_uid":
        groups = [
            group for group in benchmark["groups"]
            if int.from_bytes(
                hashlib.sha256(str(group["target_uid"]).encode("utf-8")).digest()[:8],
                "big",
            ) % args.shard_count == args.shard_index
        ]
    else:
        assignment = balanced_target_assignment(benchmark["groups"], args.shard_count)
        groups = [
            group for group in benchmark["groups"]
            if assignment[str(group["target_uid"])] == args.shard_index
        ]
    if args.group_offset < 0:
        raise ValueError("group-offset must be non-negative")
    if args.group_offset:
        groups = groups[args.group_offset :]
    if args.group_limit > 0:
        groups = groups[: args.group_limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    modules = load_render_modules()
    client = load_policy_client(args.policy_host, args.policy_port)
    results = []
    errors = []
    scene: SceneContext | None = None
    current_uid = None
    started = time.monotonic()
    try:
        for group_index, group in enumerate(groups):
            target_uid = str(group["target_uid"])
            if target_uid != current_uid:
                if scene is not None:
                    scene.close()
                scene = SceneContext(target_uid, modules)
                current_uid = target_uid
            episodes = [("canonical", 0, group["canonical"])]
            if args.variant_mode == "all":
                episodes.extend(
                    ("initial_yaw", index, episode)
                    for index, episode in enumerate(group["initial_yaw"], start=1)
                )
            for family, variant_index, episode in episodes:
                for connection_attempt in range(args.connection_retries + 1):
                    try:
                        result = run_episode(
                            scene, client, group, episode, family, variant_index, args
                        )
                        results.append(result)
                        print(
                            f"progress={len(results)} group={group_index + 1}/{len(groups)} "
                            f"target={group['target_name']} family={family} outcome={result['outcome']} "
                            f"op_dist={result['final_operating_point_distance_m']:.3f} "
                            f"heading={result['final_heading_deviation_deg']:.1f}",
                            flush=True,
                        )
                        break
                    except Exception as exc:
                        connection_closed = "ConnectionClosed" in type(exc).__name__
                        if connection_closed and connection_attempt < args.connection_retries:
                            print(json.dumps({
                                "group_id": group["group_id"],
                                "family": family,
                                "connection_retry": connection_attempt + 1,
                                "error": f"{type(exc).__name__}: {exc}",
                            }), flush=True)
                            time.sleep(1.0)
                            client = load_policy_client(args.policy_host, args.policy_port)
                            continue
                        error = {
                            "group_id": group["group_id"],
                            "family": family,
                            "variant_index": variant_index,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        errors.append(error)
                        print(json.dumps(error), flush=True)
                        if args.fail_fast:
                            raise
                        break
            summary = aggregate(results, errors, args)
            summary["elapsed_s"] = float(time.monotonic() - started)
            (args.output_dir / f"summary_shard{args.shard_index:02d}.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
    finally:
        if scene is not None:
            scene.close()
    summary = aggregate(results, errors, args)
    summary["elapsed_s"] = float(time.monotonic() - started)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
