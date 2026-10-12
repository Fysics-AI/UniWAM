"""ROS 2 edge executor for the nav3/manip26 EEF-XY+visibility async-prefix12 policy.

The edge sends queued absolute EEF14+base3 targets. The cloud rebases them on
the newest 23D SE(3)/rot6D state, generates only the suffix, and returns
suffix IK targets while the matching prefix continues to execute.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import socket
import sys
import threading
import time
from collections import deque
from pathlib import Path
from queue import Full, Queue
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image as RosImage
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

from .async_queue import plan_head_prefix_splice
from uniwam_piper_common.camera_frame import validate_se3
from uniwam_piper_common.rotation6d import matrix_to_rpy_xyz, rotation_6d_to_matrix

from .rot6d_state import compose_state23, state23_from_rpy, validate_state23
from .protocol import recv_message, send_message


STATE_DIM = 23
EEF_DIM = 14
MODEL_ACTION_DIM = 17
JOINT_DIM = 14
BASE_DIM = 3
NAV_ACTION_DIM = 3
NAV_AUX_DIM = 0
BBOX_DIM = 6  # Historical wire name; [xy4, visibility2], not a bbox.
MANIP_ACTION_DIM = 26
INFERENCE_MODES = {"paired", "manip_only", "nav_only"}


def ros_image_to_rgb(msg: RosImage) -> np.ndarray:
    if msg.encoding not in {"rgb8", "bgr8", "rgba8", "bgra8"}:
        raise ValueError(f"Unsupported image encoding {msg.encoding!r}; expected rgb8/bgr8/rgba8/bgra8.")
    channels = 4 if "a8" in msg.encoding else 3
    array = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)
    if msg.encoding in {"rgb8", "rgba8"}:
        return array[:, :, :3].copy()
    return array[:, :, [2, 1, 0]].copy()


def canonicalize_quaternion_xyzw(quaternion: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if np.any(~np.isfinite(norm)) or np.any(norm < eps):
        raise ValueError("Quaternion is non-finite or has near-zero norm.")
    value = value / norm
    sign = np.where(value[..., 3] < -eps, -1.0, 1.0)
    unresolved = np.abs(value[..., 3]) <= eps
    for component in (2, 1, 0):
        negative = unresolved & (value[..., component] < -eps)
        positive = unresolved & (value[..., component] > eps)
        sign = np.where(negative, -1.0, sign)
        unresolved &= ~(negative | positive)
    return (value * sign[..., None]).astype(np.float32)


def rpy_to_quaternion_xyzw(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64).reshape(3)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return canonicalize_quaternion_xyzw(
        np.asarray(
            [
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
                cr * cp * cy + sr * sp * sy,
            ],
            dtype=np.float64,
        )
    )


def pose_msg_to_eef7(msg: PoseStamped) -> np.ndarray:
    position = msg.pose.position
    orientation = msg.pose.orientation
    result = np.asarray(
        [position.x, position.y, position.z, orientation.x, orientation.y, orientation.z, orientation.w],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("PoseStamped contains NaN or Inf.")
    result[3:7] = canonicalize_quaternion_xyzw(result[3:7])
    return result


def joint_msg_to_position7(msg: JointState) -> np.ndarray:
    position = np.asarray(msg.position, dtype=np.float32)
    if position.shape[0] < 7:
        raise ValueError(f"Expected at least 7 joint positions, got {position.shape[0]}.")
    if not np.all(np.isfinite(position[:7])):
        raise ValueError("JointState contains NaN or Inf.")
    return position[:7].copy()


def resize_rgb(
    image: np.ndarray,
    size_hw: tuple[int, int] | None,
    *,
    preserve_aspect: bool = True,
) -> np.ndarray:
    if size_hw is None:
        return image
    height, width = size_hw
    if image.shape[:2] == (height, width):
        return image
    source = Image.fromarray(image)
    if preserve_aspect:
        source_width, source_height = source.size
        target_aspect = float(width) / float(height)
        source_aspect = float(source_width) / float(source_height)
        if source_aspect > target_aspect:
            crop_width = max(1, int(round(source_height * target_aspect)))
            left = (source_width - crop_width) // 2
            source = source.crop((left, 0, left + crop_width, source_height))
        elif source_aspect < target_aspect:
            crop_height = max(1, int(round(source_width / target_aspect)))
            top = (source_height - crop_height) // 2
            source = source.crop((0, top, source_width, top + crop_height))
    return np.asarray(source.resize((width, height), resample=Image.Resampling.BILINEAR), dtype=np.uint8)


def encode_image(image: np.ndarray, encoding: str, quality: int) -> bytes:
    buffer = io.BytesIO()
    normalized = str(encoding).strip().lower()
    if normalized in {"jpg", "jpeg"}:
        Image.fromarray(image).save(buffer, format="JPEG", quality=int(quality))
    elif normalized == "png":
        Image.fromarray(image).save(buffer, format="PNG")
    else:
        raise ValueError(f"Unsupported image encoding {encoding!r}; use jpeg or png.")
    return buffer.getvalue()


def parse_resize_before_send(value: Any) -> dict[str, tuple[int, int]]:
    if value in (None, "null"):
        return {}
    plain = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value

    def parse_size(size: Any) -> tuple[int, int]:
        if not isinstance(size, (list, tuple)) or len(size) != 2:
            raise ValueError(f"resize_before_send item must be [height,width], got {size!r}")
        return int(size[0]), int(size[1])

    keys = ("cam_nav", "cam_manip_high", "cam_left_wrist", "cam_right_wrist")
    if isinstance(plain, (list, tuple)):
        size = parse_size(plain)
        return {key: size for key in keys}
    if not isinstance(plain, dict):
        raise ValueError("resize_before_send must be null, [H,W], or a camera-key map.")
    return {str(key): parse_size(size) for key, size in plain.items() if size not in (None, "null")}


def parse_camera_extrinsic(name: str, value: Any) -> np.ndarray:
    plain = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    return validate_se3(name, plain)


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return value


def array_summary(value: np.ndarray | None) -> dict[str, Any] | None:
    if value is None:
        return None
    array = np.asarray(value)
    summary: dict[str, Any] = {"shape": list(array.shape)}
    if not array.size:
        return summary
    rows = array if array.ndim >= 2 else array.reshape(1, -1)
    summary.update(
        {
            "first": rows[0].tolist(),
            "last": rows[-1].tolist(),
            "min": np.nanmin(array, axis=0).tolist() if array.ndim >= 2 else float(np.nanmin(array)),
            "max": np.nanmax(array, axis=0).tolist() if array.ndim >= 2 else float(np.nanmax(array)),
        }
    )
    return summary


class InferenceRecorder:
    """Persist enough information to replay and visualize model behavior offline."""

    def __init__(self, cfg: Any) -> None:
        record_cfg = cfg.get("record", {})
        self.enabled = bool(record_cfg.get("enabled", False))
        self.root: Path | None = None
        self.meta_file = None
        self.image_write_queue: Queue[tuple[Path, np.ndarray] | None] | None = None
        self.image_write_thread: threading.Thread | None = None
        self.save_images = bool(record_cfg.get("save_images", True))
        self.save_publish_images = bool(record_cfg.get("save_publish_images", True))
        self.publish_image_jpeg_quality = int(record_cfg.get("publish_image_jpeg_quality", 90))
        self.image_every_n = max(1, int(record_cfg.get("image_every_n", 1)))
        self.publish_every_n = max(1, int(record_cfg.get("publish_every_n", 1)))
        if not self.enabled:
            return
        root = Path(str(record_cfg.get("output_dir", "~/uniwam_async_prefix12_rot6d_records"))).expanduser()
        if bool(record_cfg.get("session_subdir", True)):
            root = root / time.strftime("session_%Y%m%d_%H%M%S")
        (root / "chunks").mkdir(parents=True, exist_ok=True)
        (root / "images").mkdir(parents=True, exist_ok=True)
        self.root = root
        self.meta_file = (root / "events.jsonl").open("a", buffering=1, encoding="utf-8")
        if self.save_publish_images:
            self.image_write_queue = Queue(maxsize=max(64, int(record_cfg.get("image_write_queue", 256))))
            self.image_write_thread = threading.Thread(
                target=self._image_writer,
                name="uniwam-record-image-writer",
                daemon=True,
            )
            self.image_write_thread.start()

    def _image_writer(self) -> None:
        assert self.image_write_queue is not None
        while True:
            item = self.image_write_queue.get()
            try:
                if item is None:
                    return
                path, image = item
                try:
                    Image.fromarray(image).save(
                        path,
                        format="JPEG",
                        quality=self.publish_image_jpeg_quality,
                    )
                except Exception:
                    # A missing diagnostic frame is handled by renderer fallback.
                    pass
            finally:
                self.image_write_queue.task_done()

    def close(self) -> None:
        if self.image_write_queue is not None:
            self.image_write_queue.put(None)
            self.image_write_queue.join()
        if self.image_write_thread is not None:
            self.image_write_thread.join(timeout=5.0)
            self.image_write_thread = None
            self.image_write_queue = None
        if self.meta_file is not None:
            self.meta_file.close()
            self.meta_file = None

    def _write_event(self, event: str, payload: dict[str, Any]) -> None:
        if not self.enabled or self.meta_file is None:
            return
        row = {"event": event, "wall_time": time.time(), **payload}
        self.meta_file.write(json.dumps(jsonable(row), ensure_ascii=True) + "\n")

    def record_observation(self, packet: dict[str, Any]) -> None:
        if not self.enabled:
            return
        request_id = int(packet["request_id"])
        image_paths: dict[str, str] = {}
        if self.save_images and self.root is not None and request_id % self.image_every_n == 0:
            extension = "jpg" if str(packet["image_encoding"]).lower() in {"jpg", "jpeg"} else "png"
            for key, data in packet["images"].items():
                rel = Path("images") / f"request_{request_id:06d}_{key}.{extension}"
                (self.root / rel).write_bytes(bytes(data))
                image_paths[key] = str(rel)
        self._write_event(
            "observation",
            {
                "request_id": request_id,
                "instruction": packet.get("instruction", ""),
                "state23_rot6d": packet["eef_state"],
                "joint_state": packet.get("joint_state"),
                "prefix_eef_actions": packet.get("prefix_eef_actions"),
                "prefix_joint_actions": packet.get("prefix_joint_actions"),
                "prefix_bbox_actions": packet.get("prefix_bbox_actions"),
                "prefix_nav_aux_actions": packet.get("prefix_nav_aux_actions"),
                "image_encoding": packet["image_encoding"],
                "image_paths": image_paths,
                "image_bytes": {key: len(bytes(data)) for key, data in packet["images"].items()},
            },
        )

    def record_chunk(
        self,
        *,
        request_id: int,
        response: dict[str, Any],
        joint_actions: np.ndarray,
        eef_actions: np.ndarray,
        model_actions: np.ndarray,
        base_actions: np.ndarray,
        nav_actions: np.ndarray,
        nav_aux_actions: np.ndarray,
        manip_actions: np.ndarray,
        bbox_actions: np.ndarray,
        queue_len: int,
    ) -> None:
        if not self.enabled or self.root is None:
            return
        arrays = {
            "joint_actions": np.asarray(joint_actions, dtype=np.float32),
            "eef_actions": np.asarray(eef_actions, dtype=np.float32),
            "model_actions": np.asarray(model_actions, dtype=np.float32),
            "base_actions": np.asarray(base_actions, dtype=np.float32),
            "nav_actions": np.asarray(nav_actions, dtype=np.float32),
            "nav_aux_actions": np.asarray(nav_aux_actions, dtype=np.float32),
            "manip_actions": np.asarray(manip_actions, dtype=np.float32),
            "bbox_actions": np.asarray(bbox_actions, dtype=np.float32),
        }
        rel = Path("chunks") / f"chunk_{request_id:06d}.npz"
        np.savez_compressed(self.root / rel, **arrays)
        self._write_event(
            "chunk",
            {
                "request_id": int(request_id),
                "chunk_npz": str(rel),
                "server_latency_s": response.get("server_latency_s"),
                "request_model_infer_s": response.get("request_model_infer_s"),
                "latency_breakdown_s": response.get("latency_breakdown_s"),
                "queue_len": int(queue_len),
                **{key: array_summary(value) for key, value in arrays.items()},
            },
        )

    def record_publish(
        self,
        *,
        action_index: int,
        joint_command: np.ndarray | None,
        base_raw: np.ndarray | None,
        base_command: np.ndarray | None,
        queue_len_after_pop: int,
        source_request_id: int | None,
        source_chunk_step: int | None,
        bbox_action: np.ndarray | None,
        nav_aux_action: np.ndarray | None,
        cam_manip_high: np.ndarray | None,
        cam_nav: np.ndarray | None,
    ) -> None:
        if not self.enabled or action_index % self.publish_every_n != 0:
            return
        image_paths: dict[str, str | None] = {"cam_manip_high": None, "cam_nav": None}
        if self.save_publish_images and self.root is not None and self.image_write_queue is not None:
            for key, value in (("cam_manip_high", cam_manip_high), ("cam_nav", cam_nav)):
                if value is None:
                    continue
                try:
                    image = np.asarray(value, dtype=np.uint8)
                    if image.ndim != 3 or image.shape[2] != 3:
                        raise ValueError(f"Executed {key} must be HxWx3 RGB, got {image.shape}")
                    rel = Path("images") / f"publish_{action_index:06d}_{key}.jpg"
                    self.image_write_queue.put_nowait((self.root / rel, image.copy()))
                    image_paths[key] = str(rel)
                except (Exception, Full):
                    # Recording diagnostics must never interrupt the 30 Hz control loop.
                    image_paths[key] = None
        self._write_event(
            "publish",
            {
                "action_index": int(action_index),
                "joint_command": joint_command,
                "base_raw": base_raw,
                "base_command": base_command,
                "queue_len_after_pop": int(queue_len_after_pop),
                "source_request_id": source_request_id,
                "source_chunk_step": source_chunk_step,
                "bbox_action": bbox_action,
                "nav_aux_action": nav_aux_action,
                "cam_manip_high_path": image_paths["cam_manip_high"],
                "cam_nav_path": image_paths["cam_nav"],
            },
        )

    def wants_publish_image(self, action_index: int) -> bool:
        return bool(
            self.enabled
            and self.save_publish_images
            and action_index % self.publish_every_n == 0
        )


class UniWAMPiperAsyncPrefix12Rot6DClient(Node):
    def __init__(self, cfg: Any) -> None:
        super().__init__("uniwam_piper_client")
        self.cfg = cfg
        self.control_hz = float(cfg.control_hz)
        if self.control_hz <= 0.0:
            raise ValueError("control_hz must be positive.")
        self.control_dt = 1.0 / self.control_hz
        self.chunk_low_watermark = int(cfg.get("chunk_low_watermark", 0))
        self.max_queue_steps = int(cfg.get("max_queue_steps", 32))
        self.prefix_steps = int(cfg.get("prefix_steps", 12))
        self.min_prefix_steps = int(cfg.get("min_prefix_steps", 6))
        adaptive_cfg = cfg.get("adaptive_prefix", {})
        self.adaptive_prefix_enabled = bool(adaptive_cfg.get("enabled", True))
        self.adaptive_prefix_margin_steps = max(
            0, int(adaptive_cfg.get("margin_steps", 1))
        )
        self.adaptive_prefix_alpha = float(adaptive_cfg.get("ewma_alpha", 0.25))
        self.adaptive_prefix_initial_latency_s = float(
            adaptive_cfg.get("initial_latency_s", 0.20)
        )
        if not 0.0 < self.adaptive_prefix_alpha <= 1.0:
            raise ValueError("adaptive_prefix.ewma_alpha must be in (0,1].")
        if self.adaptive_prefix_initial_latency_s <= 0.0:
            raise ValueError("adaptive_prefix.initial_latency_s must be positive.")
        self.replan_while_queue_nonempty = bool(
            cfg.get("replan_while_queue_nonempty", True)
        )
        self.request_retry_interval_s = float(cfg.get("request_retry_interval_s", 1.0))
        if self.prefix_steps < 0 or self.prefix_steps >= self.max_queue_steps:
            raise ValueError("prefix_steps must be in [0, max_queue_steps).")
        if self.prefix_steps == 0:
            if self.min_prefix_steps != 0:
                raise ValueError("min_prefix_steps must be 0 when prefix_steps=0.")
        elif not 0 < self.min_prefix_steps <= self.prefix_steps:
            raise ValueError("min_prefix_steps must be in [1, prefix_steps].")
        if self.prefix_steps > 0 and not self.replan_while_queue_nonempty:
            raise ValueError("async-prefix12 client requires replan_while_queue_nonempty=true.")
        if self.chunk_low_watermark < self.prefix_steps:
            raise ValueError("chunk_low_watermark must be >= prefix_steps for head-prefix replanning.")
        if self.max_queue_steps <= 0:
            raise ValueError("max_queue_steps must be positive.")
        if self.request_retry_interval_s <= 0.0:
            raise ValueError("request_retry_interval_s must be positive.")

        self.state_source = str(cfg.state.get("source", "piper_pose_topics")).strip().lower()
        if self.state_source not in {"piper_pose_topics", "eef_topic"}:
            raise ValueError("state.source must be piper_pose_topics or eef_topic.")
        self.jpeg_quality = int(cfg.image.get("jpeg_quality", 85))
        self.image_encoding = str(cfg.image.get("encoding", "jpeg")).strip().lower()
        self.resize_before_send = parse_resize_before_send(cfg.image.get("resize_before_send", None))
        self.preserve_image_aspect = bool(cfg.image.get("preserve_aspect", True))
        self.task_prompt = str(cfg.get("task_prompt", "")).strip()
        self.task_prompt_topic = str(cfg.get("task_prompt_topic", "/pi05/task_prompt")).strip()
        self.inference_mode = str(cfg.get("inference_mode", "paired")).strip().lower()
        if self.inference_mode not in INFERENCE_MODES:
            raise ValueError(
                f"inference_mode must be one of {sorted(INFERENCE_MODES)}, got {self.inference_mode!r}."
            )
        camera_cfg = cfg.get("camera_frame", {})
        self.extrinsics_profile = str(
            camera_cfg.get("extrinsics_profile", "agilex_fixed_reference")
        ).strip()
        self.camera_from_left_base = parse_camera_extrinsic(
            "camera_frame.camera_from_left_base", camera_cfg.get("camera_from_left_base")
        )
        self.camera_from_right_base = parse_camera_extrinsic(
            "camera_frame.camera_from_right_base", camera_cfg.get("camera_from_right_base")
        )
        self.joint_names = list(cfg.state.joint_names)

        control_cfg = cfg.get("control", {})
        self.dry_run = bool(control_cfg.get("dry_run", True))
        self.hold_when_feedback_missing = bool(control_cfg.get("hold_when_feedback_missing", True))
        self.max_publish_joint_delta_rad = float(control_cfg.get("max_publish_joint_delta_rad", 0.0))
        self.piper_speed_percent = float(control_cfg.get("piper_speed_percent", 100.0))
        self.gripper_effort = float(control_cfg.get("gripper_effort", 2.0))
        self.binary_gripper = bool(control_cfg.get("binary_gripper", False))
        self.clamp_gripper_commands = bool(control_cfg.get("clamp_gripper_commands", False))
        self.gripper_min_m = float(control_cfg.get("gripper_min_m", 0.0))
        self.gripper_max_m = float(control_cfg.get("gripper_max_m", 0.105))
        self.gripper_closed_m = float(control_cfg.get("gripper_closed_m", 0.0))
        self.gripper_open_m = float(control_cfg.get("gripper_open_m", 0.105))
        self.gripper_binary_threshold_m = float(control_cfg.get("gripper_binary_threshold_m", 0.060))
        self.left_gripper_threshold_m = float(control_cfg.get("left_gripper_binary_threshold_m", self.gripper_binary_threshold_m))
        self.right_gripper_threshold_m = float(control_cfg.get("right_gripper_binary_threshold_m", self.gripper_binary_threshold_m))
        self.log_published_actions = bool(control_cfg.get("log_published_actions", True))
        self.debug_action_deltas = bool(control_cfg.get("debug_action_deltas", False))
        self.manual_hold = {
            "left": bool(control_cfg.get("manual_hold_left", False)),
            "right": bool(control_cfg.get("manual_hold_right", False)),
        }
        self.manual_hold_joint: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self.manual_hold_eef: dict[str, np.ndarray | None] = {"left": None, "right": None}

        base_cfg = cfg.get("base", {})
        self.base_enabled = bool(base_cfg.get("enabled", True))
        self.base_cmd_vel_topic = str(base_cfg.get("cmd_vel_topic", "/xw/cmd_vel_direct_can")).strip()
        self.base_dry_run = bool(base_cfg.get("dry_run", self.dry_run))
        self.base_state_follows_dry_run_command = bool(base_cfg.get("state_follows_dry_run_command", False))
        self.base_clamp_commands = bool(base_cfg.get("clamp_commands", False))
        self.base_min_vx_mps = float(base_cfg.get("min_vx_mps", -0.12))
        self.base_max_vx_mps = float(base_cfg.get("max_vx_mps", 0.12))
        self.base_max_vy_mps = float(base_cfg.get("max_vy_mps", 0.0))
        self.base_max_wz_radps = float(base_cfg.get("max_wz_radps", 0.18))
        self.base_max_accel_mps2 = float(base_cfg.get("max_accel_mps2", 0.0))
        self.base_max_yaw_accel_radps2 = float(base_cfg.get("max_yaw_accel_radps2", 0.0))
        self.base_vx_scale = float(base_cfg.get("vx_scale", 1.0))
        self.base_vy_scale = float(base_cfg.get("vy_scale", 1.0))
        self.base_wz_scale = float(base_cfg.get("wz_scale", 1.0))
        self.base_vx_deadband_mps = float(base_cfg.get("vx_deadband_mps", 0.0))
        self.base_vy_deadband_mps = float(base_cfg.get("vy_deadband_mps", 0.0))
        self.base_wz_deadband_radps = float(base_cfg.get("wz_deadband_radps", 0.0))
        self.base_command_timeout_s = float(base_cfg.get("command_timeout_s", 0.25))
        self.base_stop_on_empty_queue = bool(base_cfg.get("stop_on_empty_queue", True))
        self.base_zero_on_shutdown = bool(base_cfg.get("zero_on_shutdown", True))
        if self.base_min_vx_mps > self.base_max_vx_mps:
            raise ValueError("base.min_vx_mps must not exceed base.max_vx_mps.")

        self.recorder = InferenceRecorder(cfg)
        self.latest_images: dict[str, RosImage] = {}
        self.latest_pose_eef7: dict[str, np.ndarray] = {}
        self.latest_joints: dict[str, np.ndarray] = {}
        self.latest_eef_topic_state: np.ndarray | None = None
        self.joint_queue: deque[np.ndarray] = deque()
        self.model_queue: deque[np.ndarray] = deque()
        self.base_queue: deque[np.ndarray] = deque()
        self.nav_aux_queue: deque[np.ndarray] = deque()
        self.bbox_queue: deque[np.ndarray] = deque()
        self.source_request_queue: deque[int] = deque()
        self.source_chunk_step_queue: deque[int] = deque()
        self.observation_lock = threading.Lock()
        self.queue_lock = threading.Lock()
        self.request_lock = threading.Lock()
        self.queue_epoch = 0
        self.request_inflight = False
        self.request_id = 0
        self.sock: socket.socket | None = None
        self.current_base_command = np.zeros(BASE_DIM, dtype=np.float32)
        self.base_ramp_command = np.zeros(BASE_DIM, dtype=np.float32)
        self.last_base_update_mono = time.monotonic()
        self.last_base_publish_mono = 0.0
        self.last_readiness_log_mono = 0.0
        self.last_request_error_log_mono = 0.0
        self.next_request_attempt_mono = 0.0
        self.request_latency_ewma_s = self.adaptive_prefix_initial_latency_s
        self.last_request_rtt_s: float | None = None
        self.published_action_count = 0
        self.published_base_count = 0
        self.last_published_action: np.ndarray | None = None
        self.binary_gripper_state: dict[str, bool | None] = {"left": None, "right": None}
        self.warned_vy = False

        self.left_pub = self.create_publisher(JointState, str(cfg.ros.left_joint_topic), 1)
        self.right_pub = self.create_publisher(JointState, str(cfg.ros.right_joint_topic), 1)
        self.base_pub = self.create_publisher(Twist, self.base_cmd_vel_topic, 1) if self.base_enabled else None
        self.create_subscription(RosImage, str(cfg.ros.cam_nav_topic), self._image_callback("cam_nav"), qos_profile_sensor_data)
        self.create_subscription(RosImage, str(cfg.ros.cam_manip_high_topic), self._image_callback("cam_manip_high"), qos_profile_sensor_data)
        self.create_subscription(RosImage, str(cfg.ros.cam_left_wrist_topic), self._image_callback("cam_left_wrist"), qos_profile_sensor_data)
        self.create_subscription(RosImage, str(cfg.ros.cam_right_wrist_topic), self._image_callback("cam_right_wrist"), qos_profile_sensor_data)
        if self.state_source == "piper_pose_topics":
            self.create_subscription(PoseStamped, str(cfg.state.left_pose_topic), self._pose_callback("left"), qos_profile_sensor_data)
            self.create_subscription(PoseStamped, str(cfg.state.right_pose_topic), self._pose_callback("right"), qos_profile_sensor_data)
            self.create_subscription(JointState, str(cfg.ros.left_feedback_topic), self._joint_callback("left"), qos_profile_sensor_data)
            self.create_subscription(JointState, str(cfg.ros.right_feedback_topic), self._joint_callback("right"), qos_profile_sensor_data)
        else:
            self.create_subscription(Float64MultiArray, str(cfg.state.eef_topic), self._eef_state_callback, qos_profile_sensor_data)
        if self.task_prompt_topic:
            self.create_subscription(String, self.task_prompt_topic, self._task_prompt_callback, 1)
        self.timer = self.create_timer(self.control_dt, self.control_tick)
        self.get_logger().info(
            "UniWAM async-prefix12 rot6D client ready. "
            f"server={cfg.server_host}:{cfg.server_port} state=23D_rot6d action=17D "
            f"control_hz={self.control_hz:.1f} async_head_prefix="
            f"{self.min_prefix_steps}..{self.prefix_steps} adaptive={self.adaptive_prefix_enabled} "
            f"chunk_low_watermark={self.chunk_low_watermark} dry_run={self.dry_run} "
            f"inference_mode={self.inference_mode} "
            f"base_enabled={self.base_enabled} base_dry_run={self.base_dry_run} "
            f"model_clamps=edge_joint_delta:{self.max_publish_joint_delta_rad:.4f} "
            f"gripper:{self.clamp_gripper_commands} base:{self.base_clamp_commands} "
            f"record_dir={str(self.recorder.root) if self.recorder.enabled else 'disabled'}"
        )
        self._start_manual_hold_console()
        self.get_logger().info(
            "Cloud connection is lazy and will retry after observations are ready; "
            "the client can therefore start before its SSH tunnel or cloud server."
        )

    def connect(self) -> None:
        if self.sock is not None:
            return
        sock = socket.create_connection((str(self.cfg.server_host), int(self.cfg.server_port)), timeout=float(self.cfg.request_timeout_s))
        sock.settimeout(float(self.cfg.request_timeout_s))
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        # Keep small control packets from waiting behind the image payload.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2 * 1024 * 1024)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
        self.sock = sock

    def destroy_node(self) -> bool:
        if self.base_enabled and self.base_zero_on_shutdown:
            self.publish_zero_base("shutdown")
        if self.sock is not None:
            self.sock.close()
            self.sock = None
        self.recorder.close()
        return super().destroy_node()

    def _image_callback(self, key: str):
        def callback(message: RosImage) -> None:
            with self.observation_lock:
                self.latest_images[key] = message
        return callback

    def _pose_callback(self, arm: str):
        def callback(message: PoseStamped) -> None:
            try:
                value = pose_msg_to_eef7(message)
                with self.observation_lock:
                    self.latest_pose_eef7[arm] = value
            except Exception as exc:
                self.get_logger().warn(f"Failed to parse {arm} pose: {type(exc).__name__}: {exc}")
        return callback

    def _joint_callback(self, arm: str):
        def callback(message: JointState) -> None:
            try:
                value = joint_msg_to_position7(message)
                with self.observation_lock:
                    self.latest_joints[arm] = value
            except Exception as exc:
                self.get_logger().warn(f"Failed to parse {arm} joint feedback: {type(exc).__name__}: {exc}")
        return callback

    def _eef_state_callback(self, message: Float64MultiArray) -> None:
        values = np.asarray(message.data, dtype=np.float32)
        try:
            if values.shape == (STATE_DIM,):
                state = validate_state23(values)
            elif values.shape == (17,):
                state = state23_from_rpy(values[:3], values[3:])
            elif values.shape == (14,):
                state = state23_from_rpy(np.zeros(3, dtype=np.float32), values)
            else:
                raise ValueError(f"expected 14D, 17D, or 23D rot6D state, got {values.shape}")
            with self.observation_lock:
                self.latest_eef_topic_state = state
        except Exception as exc:
            self.get_logger().warn(f"Invalid EEF state topic: {type(exc).__name__}: {exc}")

    def _task_prompt_callback(self, message: String) -> None:
        prompt = str(message.data).strip()
        if prompt:
            self.task_prompt = prompt
            self.get_logger().info(f"Updated task prompt: {prompt!r}")

    def _current_model_state_locked(self) -> np.ndarray | None:
        if self.state_source == "eef_topic":
            return None if self.latest_eef_topic_state is None else self.latest_eef_topic_state.copy()
        if "left" not in self.latest_pose_eef7 or "right" not in self.latest_pose_eef7:
            return None
        if "left" not in self.latest_joints or "right" not in self.latest_joints:
            return None
        base = self.current_base_command if self.base_enabled else np.zeros(BASE_DIM, dtype=np.float32)
        return compose_state23(
            base,
            self.latest_pose_eef7["left"],
            self.latest_joints["left"],
            self.latest_pose_eef7["right"],
            self.latest_joints["right"],
        )

    def current_model_state(self) -> np.ndarray | None:
        with self.observation_lock:
            return self._current_model_state_locked()

    def _current_joint_state_locked(self) -> np.ndarray | None:
        if "left" not in self.latest_joints or "right" not in self.latest_joints:
            return None
        return np.concatenate((self.latest_joints["left"], self.latest_joints["right"])).astype(np.float32)

    def _held_eef14_from_state_locked(self, arm: str) -> np.ndarray | None:
        state = self._current_model_state_locked()
        if state is None:
            return None
        if arm == "left":
            position, rotation, gripper = state[3:6], state[6:12], state[12]
        else:
            position, rotation, gripper = state[13:16], state[16:22], state[22]
        result = np.empty(7, dtype=np.float32)
        result[:3] = position
        result[3:6] = matrix_to_rpy_xyz(rotation_6d_to_matrix(rotation))
        result[6] = gripper
        return result

    def _start_manual_hold_console(self) -> None:
        if not sys.stdin.isatty():
            return

        def reader() -> None:
            self.get_logger().info(
                "Manual arm hold console: enter 1 to toggle LEFT hold, "
                "2 to toggle RIGHT hold, 0 to release both."
            )
            for line in sys.stdin:
                command = line.strip().lower()
                if command == "1":
                    self._set_manual_hold("left", not self.manual_hold["left"])
                elif command == "2":
                    self._set_manual_hold("right", not self.manual_hold["right"])
                elif command == "0":
                    self._set_manual_hold("left", False)
                    self._set_manual_hold("right", False)
                elif command in {"q", "quit", "exit"}:
                    return

        threading.Thread(target=reader, name="uniwam-manual-arm-hold", daemon=True).start()

    def _set_manual_hold(self, arm: str, enabled: bool) -> None:
        if arm not in {"left", "right"}:
            raise ValueError(f"Unknown arm {arm!r}")
        with self.queue_lock:
            if enabled:
                with self.observation_lock:
                    joint = self.latest_joints.get(arm)
                    eef = self._held_eef14_from_state_locked(arm)
                if joint is None or eef is None:
                    self.get_logger().warning(
                        f"Cannot hold {arm} arm yet: live joint/EEF feedback is unavailable."
                    )
                    return
                self.manual_hold_joint[arm] = np.asarray(joint, dtype=np.float32).copy()
                self.manual_hold_eef[arm] = np.asarray(eef, dtype=np.float32).copy()
            else:
                self.manual_hold_joint[arm] = None
                self.manual_hold_eef[arm] = None
            self.manual_hold[arm] = bool(enabled)
            # Discard stale actions and invalidate an in-flight response. The
            # next request is a fresh prefix=0 snapshot under the new policy.
            self._clear_queues_locked()
        self.get_logger().warning(
            f"Manual {'enabled' if enabled else 'released'} hold for {arm} arm; "
            f"holds={{left:{self.manual_hold['left']},right:{self.manual_hold['right']}}}"
        )

    def _desired_prefix_steps_locked(self) -> int:
        if not self.adaptive_prefix_enabled:
            return self.prefix_steps
        estimated = int(
            np.ceil(self.request_latency_ewma_s * self.control_hz)
        ) + self.adaptive_prefix_margin_steps
        return int(np.clip(estimated, self.min_prefix_steps, self.prefix_steps))

    def _update_request_latency(self, rtt_s: float) -> None:
        value = float(rtt_s)
        if not np.isfinite(value) or value <= 0.0:
            return
        self.last_request_rtt_s = value
        self.request_latency_ewma_s = (
            self.adaptive_prefix_alpha * value
            + (1.0 - self.adaptive_prefix_alpha) * self.request_latency_ewma_s
        )

    def current_joint_state(self) -> np.ndarray | None:
        with self.observation_lock:
            return self._current_joint_state_locked()

    def base_executor_ready(self) -> bool:
        if not self.base_enabled or self.base_dry_run:
            return True
        return self.base_pub is not None and self.base_pub.get_subscription_count() > 0

    def missing_observations(self) -> list[str]:
        missing: list[str] = []
        topics = {
            "cam_nav": self.cfg.ros.cam_nav_topic,
            "cam_manip_high": self.cfg.ros.cam_manip_high_topic,
            "cam_left_wrist": self.cfg.ros.cam_left_wrist_topic,
            "cam_right_wrist": self.cfg.ros.cam_right_wrist_topic,
        }
        with self.observation_lock:
            missing.extend(
                f"{key}({topic})"
                for key, topic in topics.items()
                if key not in self.latest_images
            )
            if self.state_source == "piper_pose_topics":
                for arm in ("left", "right"):
                    if arm not in self.latest_pose_eef7:
                        missing.append(f"{arm}_pose({self.cfg.state[f'{arm}_pose_topic']})")
                    if arm not in self.latest_joints:
                        missing.append(f"{arm}_joint({self.cfg.ros[f'{arm}_feedback_topic']})")
            elif self.latest_eef_topic_state is None:
                missing.append(f"eef_state({self.cfg.state.eef_topic})")
        if not self.base_executor_ready():
            missing.append(f"base_executor_subscriber({self.base_cmd_vel_topic})")
        return missing

    def observations_ready(self) -> bool:
        return self.current_model_state() is not None and not self.missing_observations()

    def build_observation_packet(self) -> dict[str, Any]:
        image_keys = (
            "cam_nav",
            "cam_manip_high",
            "cam_left_wrist",
            "cam_right_wrist",
        )
        with self.queue_lock:
            with self.observation_lock:
                state = self._current_model_state_locked()
                if state is None:
                    raise RuntimeError("23D rot6D model state is unavailable.")
                joint_state = self._current_joint_state_locked()
                image_messages = {key: self.latest_images[key] for key in image_keys}
                snapshot_wall_time = time.time()
            queue_len_at_request = len(self.joint_queue)
            queue_epoch_at_request = self.queue_epoch
            desired_prefix_steps = self._desired_prefix_steps_locked()
            prefix_length = min(
                desired_prefix_steps,
                len(self.joint_queue),
                len(self.model_queue),
                len(self.nav_aux_queue),
                len(self.bbox_queue),
            )
            if prefix_length:
                prefix_eef = np.stack(list(self.model_queue)[:prefix_length]).astype(np.float32)
                prefix_joint = np.stack(list(self.joint_queue)[:prefix_length]).astype(np.float32)
                prefix_nav_aux = np.stack(list(self.nav_aux_queue)[:prefix_length]).astype(np.float32)
                prefix_bbox = np.stack(list(self.bbox_queue)[:prefix_length]).astype(np.float32)
            else:
                prefix_eef = None
                prefix_joint = None
                prefix_nav_aux = None
                prefix_bbox = None
        if 0 < prefix_length < self.min_prefix_steps:
            raise RuntimeError(
                f"Refusing untrained prefix length {prefix_length}; "
                f"expected 0 or [{self.min_prefix_steps}, {self.prefix_steps}]."
            )
        images: dict[str, bytes] = {}
        for key, message in image_messages.items():
            rgb = resize_rgb(
                ros_image_to_rgb(message),
                self.resize_before_send.get(key),
                preserve_aspect=self.preserve_image_aspect,
            )
            images[key] = encode_image(rgb, self.image_encoding, self.jpeg_quality)
        self.request_id += 1
        packet = {
            "type": "observation",
            "robot_type": "piper",
            "request_id": self.request_id,
            "stamp": snapshot_wall_time,
            "instruction": self.task_prompt,
            "inference_mode": self.inference_mode,
            "extrinsics_profile": self.extrinsics_profile,
            "camera_from_left_base": self.camera_from_left_base,
            "camera_from_right_base": self.camera_from_right_base,
            "eef_state": state,
            "joint_state": joint_state,
            "images": images,
            "image_encoding": self.image_encoding,
            "prefix_mode": "head",
            "prefix_eef_actions": prefix_eef,
            "prefix_joint_actions": prefix_joint,
                # Historical wire key; for manip26 this is the queued model
                # EEF-XY+visibility prefix [L,6], not a manipulation bbox.
                "prefix_bbox_actions": prefix_bbox,
            "prefix_nav_aux_actions": prefix_nav_aux,
            "queue_len_at_request": queue_len_at_request,
            "queue_epoch_at_request": queue_epoch_at_request,
            "adaptive_prefix_target": desired_prefix_steps,
        }
        self.recorder.record_observation(packet)
        return packet

    def _assert_queue_alignment_locked(self) -> None:
        expected = len(self.joint_queue)
        lengths = {
            "joint": expected,
            "model": len(self.model_queue),
            "base": len(self.base_queue),
            "nav_aux": len(self.nav_aux_queue),
            "bbox": len(self.bbox_queue),
            "request": len(self.source_request_queue),
            "chunk_step": len(self.source_chunk_step_queue),
        }
        if any(length != expected for length in lengths.values()):
            raise RuntimeError(f"Queue alignment failure: {lengths}")

    def _clear_queues_locked(self) -> None:
        self.queue_epoch += 1
        self.joint_queue.clear()
        self.model_queue.clear()
        self.base_queue.clear()
        self.nav_aux_queue.clear()
        self.bbox_queue.clear()
        self.source_request_queue.clear()
        self.source_chunk_step_queue.clear()

    def maybe_request_chunk(self) -> None:
        with self.queue_lock:
            queue_length = len(self.joint_queue)
            should_request = queue_length == 0 or (
                self.replan_while_queue_nonempty
                and self.min_prefix_steps
                <= queue_length
                <= self.chunk_low_watermark
            )
        if not should_request or self.request_inflight:
            return
        if time.monotonic() < self.next_request_attempt_mono:
            return
        if not self.observations_ready():
            now = time.monotonic()
            if now - self.last_readiness_log_mono >= 2.0:
                self.get_logger().warn("Waiting for observations: " + ", ".join(self.missing_observations()))
                self.last_readiness_log_mono = now
            return
        self.request_inflight = True
        threading.Thread(target=self.request_chunk_worker, daemon=True).start()

    def _apply_manual_holds_to_chunk(
        self,
        joint: np.ndarray,
        eef: np.ndarray,
        model: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        joint = np.asarray(joint, dtype=np.float32).copy()
        eef = np.asarray(eef, dtype=np.float32).copy()
        model = np.asarray(model, dtype=np.float32).copy()
        for arm, joint_start, eef_start in (
            ("left", 0, 0),
            ("right", 7, 7),
        ):
            if not self.manual_hold[arm]:
                continue
            held_joint = self.manual_hold_joint[arm]
            held_eef = self.manual_hold_eef[arm]
            if held_joint is None or held_eef is None:
                with self.observation_lock:
                    live_joint = self.latest_joints.get(arm)
                    live_eef = self._held_eef14_from_state_locked(arm)
                if live_joint is not None and live_eef is not None:
                    held_joint = np.asarray(live_joint, dtype=np.float32).copy()
                    held_eef = np.asarray(live_eef, dtype=np.float32).copy()
                    self.manual_hold_joint[arm] = held_joint
                    self.manual_hold_eef[arm] = held_eef
            if held_joint is not None:
                joint[:, joint_start : joint_start + 7] = held_joint
            if held_eef is not None:
                eef[:, eef_start : eef_start + 7] = held_eef
                model[:, eef_start : eef_start + 7] = held_eef
        return joint, eef, model

    def request_chunk_worker(self) -> None:
        try:
            with self.request_lock:
                request_started_mono = time.monotonic()
                self.connect()
                assert self.sock is not None
                packet = self.build_observation_packet()
                send_message(self.sock, packet)
                response = recv_message(self.sock)
                request_rtt_s = time.monotonic() - request_started_mono
            if not response.get("ok", False):
                raise RuntimeError(f"Cloud error: {response.get('error')}")
            response_extrinsics = response.get("camera_extrinsics", {})
            if str(response_extrinsics.get("profile", "")) != self.extrinsics_profile:
                raise RuntimeError(
                    "Cloud camera extrinsics profile does not match edge request: "
                    f"{response_extrinsics!r} vs {self.extrinsics_profile!r}"
                )
            response_mode = str(response.get("inference_mode", "")).strip().lower()
            if response_mode != self.inference_mode:
                raise RuntimeError(
                    "Cloud inference_mode="
                    f"{response_mode!r} does not match edge request={self.inference_mode!r}."
                )
            joint = np.asarray(response["joint_actions"], dtype=np.float32)
            eef = np.asarray(response["eef_actions"], dtype=np.float32)
            model = np.asarray(response["model_actions"], dtype=np.float32)
            base = np.asarray(response["base_actions"], dtype=np.float32)
            nav = np.asarray(response["nav_actions"], dtype=np.float32)
            nav_aux = np.asarray(response["nav_aux_actions"], dtype=np.float32)
            manip = np.asarray(response["manip_actions"], dtype=np.float32)
            bbox = np.asarray(response.get("eef_xy_actions", response["bbox_actions"]), dtype=np.float32)
            with self.queue_lock:
                joint, eef, model = self._apply_manual_holds_to_chunk(joint, eef, model)
                self._update_request_latency(request_rtt_s)
            rows = int(joint.shape[0]) if joint.ndim == 2 else 0
            expected = {
                "joint_actions": (rows, JOINT_DIM),
                "eef_actions": (rows, EEF_DIM),
                "model_actions": (rows, MODEL_ACTION_DIM),
                "base_actions": (rows, BASE_DIM),
                "nav_actions": (rows, NAV_ACTION_DIM),
                "nav_aux_actions": (rows, NAV_AUX_DIM),
                "manip_actions": (rows, MANIP_ACTION_DIM),
                "bbox_actions": (rows, BBOX_DIM),
            }
            for name, values in {
                "joint_actions": joint,
                "eef_actions": eef,
                "model_actions": model,
                "base_actions": base,
                "nav_actions": nav,
                "nav_aux_actions": nav_aux,
                "manip_actions": manip,
                "bbox_actions": bbox,
            }.items():
                if tuple(values.shape) != expected[name] or not np.all(np.isfinite(values)):
                    raise RuntimeError(f"Bad {name}: shape={values.shape}, expected={expected[name]}")
            if rows <= 0 or rows > self.max_queue_steps:
                raise RuntimeError(f"Refusing chunk length {rows}; max_queue_steps={self.max_queue_steps}.")
            prefix_length = int(response.get("prefix_length", -1))
            model_horizon = int(response.get("model_horizon", rows))
            execute_horizon = int(response.get("execute_horizon", rows))
            if execute_horizon != rows:
                raise RuntimeError(
                    f"Cloud execute_horizon={execute_horizon} does not match returned rows={rows}."
                )
            if model_horizon <= 0 or prefix_length < 0 or prefix_length + rows > model_horizon:
                raise RuntimeError(
                    "Invalid cloud horizon metadata: "
                    f"model_horizon={model_horizon}, prefix={prefix_length}, rows={rows}."
                )
            requested_prefix = packet.get("prefix_eef_actions")
            expected_prefix_length = (
                0 if requested_prefix is None else int(np.asarray(requested_prefix).shape[0])
            )
            if prefix_length != expected_prefix_length:
                raise RuntimeError(
                    f"Cloud prefix_length={prefix_length} does not match request={expected_prefix_length}."
                )
            if not np.allclose(model[:, 14:17], base, atol=1e-5, rtol=1e-5):
                raise RuntimeError("Cloud base_actions disagree with model_actions[:,14:17].")
            if tuple(bbox.shape) != (rows, BBOX_DIM):
                raise RuntimeError(f"manip26 cloud must return EEF-XY+visibility bbox_actions [rows,6], got {bbox.shape}")
            if not np.allclose(nav[:, BASE_DIM:], nav_aux, atol=1e-6, rtol=1e-6):
                raise RuntimeError("Cloud nav_aux_actions disagree with nav_actions[:,3:9].")

            with self.queue_lock:
                queue_len_at_request = int(packet["queue_len_at_request"])
                splice = plan_head_prefix_splice(
                    queue_len_at_request=queue_len_at_request,
                    queue_len_at_response=len(self.joint_queue),
                    prefix_length=prefix_length,
                    suffix_length=rows,
                    queue_epoch_at_request=int(packet["queue_epoch_at_request"]),
                    queue_epoch_at_response=self.queue_epoch,
                )
                executed_during_request = splice.executed_during_request
                keep_old = splice.keep_old
                drop_new = splice.drop_new

                # A response whose requested prefix has already been fully
                # consumed is stale.  Splicing its suffix onto the live queue
                # creates an unconditioned seam (and can produce a large IK
                # jump). Keep the actions already queued and request again
                # from the current observation instead.
                if executed_during_request > prefix_length and self.joint_queue:
                    queue_len = len(self.joint_queue)
                    self.next_request_attempt_mono = 0.0
                    self.get_logger().warning(
                        "discarding stale async chunk: "
                        f"request={int(packet['request_id'])} "
                        f"prefix={prefix_length} executed={executed_during_request} "
                        f"queue={queue_len}"
                    )
                    self.recorder.record_chunk(
                        request_id=int(packet["request_id"]),
                        response=response,
                        joint_actions=joint,
                        eef_actions=eef,
                        model_actions=model,
                        base_actions=base,
                        nav_actions=nav,
                        nav_aux_actions=nav_aux,
                        manip_actions=manip,
                        bbox_actions=bbox,
                        queue_len=queue_len,
                    )
                    return
                seam_reference = (
                    self.joint_queue[keep_old - 1].copy()
                    if keep_old
                    else None
                    if self.last_published_action is None
                    else self.last_published_action.copy()
                )
                while len(self.joint_queue) > keep_old:
                    self.joint_queue.pop()
                    self.model_queue.pop()
                    self.base_queue.pop()
                    self.nav_aux_queue.pop()
                    self.bbox_queue.pop()
                    self.source_request_queue.pop()
                    self.source_chunk_step_queue.pop()
                for index in range(drop_new, rows):
                    self.joint_queue.append(joint[index])
                    self.model_queue.append(model[index])
                    self.base_queue.append(base[index])
                    self.nav_aux_queue.append(nav_aux[index])
                    self.bbox_queue.append(bbox[index])
                    self.source_request_queue.append(int(packet["request_id"]))
                    self.source_chunk_step_queue.append(index)
                self._assert_queue_alignment_locked()
                queue_len = len(self.joint_queue)
                seam_target = (
                    self.joint_queue[keep_old].copy()
                    if len(self.joint_queue) > keep_old
                    else None
                )
            if self.debug_action_deltas and seam_reference is not None and seam_target is not None:
                seam_delta = np.abs(seam_target - seam_reference)
                self.get_logger().info(
                    "chunk splice delta "
                    f"request={int(packet['request_id'])} prefix={prefix_length} "
                    f"executed={executed_during_request} drop_new={drop_new} "
                    f"left_joint_max={float(np.max(seam_delta[:6])):.5f} "
                    f"right_joint_max={float(np.max(seam_delta[7:13])):.5f} "
                    f"gripper=[{float(seam_delta[6]):.5f},{float(seam_delta[13]):.5f}]"
                )
            self.recorder.record_chunk(
                request_id=int(packet["request_id"]),
                response=response,
                joint_actions=joint,
                eef_actions=eef,
                model_actions=model,
                base_actions=base,
                nav_actions=nav,
                nav_aux_actions=nav_aux,
                manip_actions=manip,
                bbox_actions=bbox,
                queue_len=queue_len,
            )
            self.next_request_attempt_mono = 0.0
            self.get_logger().info(
                f"received async-prefix12 model_T={model_horizon} execute_T={rows} "
                f"prefix={prefix_length} inference_mode={response_mode} "
                f"arm_hold={bool(response.get('arm_hold', False))} "
                f"executed={executed_during_request} keep_old={keep_old} "
                f"drop_new={drop_new} queue={queue_len} "
                    f"server_latency={float(response.get('server_latency_s', -1.0)):.3f}s "
                    f"model_infer={float(response.get('request_model_infer_s', -1.0)):.3f}s "
                    f"rtt={request_rtt_s:.3f}s adaptive_prefix={self._desired_prefix_steps_locked()} "
                f"base_first={base[0].tolist()} base_last={base[-1].tolist()}"
            )
        except Exception as exc:
            now = time.monotonic()
            self.next_request_attempt_mono = now + self.request_retry_interval_s
            if now - self.last_request_error_log_mono >= self.request_retry_interval_s:
                self.get_logger().error(
                    f"request_chunk failed: {type(exc).__name__}: {exc}; "
                    f"retrying in {self.request_retry_interval_s:.1f}s"
                )
                self.last_request_error_log_mono = now
            if self.sock is not None:
                try:
                    self.sock.close()
                finally:
                    self.sock = None
        finally:
            self.request_inflight = False

    def control_tick(self) -> None:
        self.maybe_request_chunk()
        if not self.base_executor_ready():
            with self.queue_lock:
                self._clear_queues_locked()
            self.publish_zero_base("base_executor_offline")
            return
        with self.queue_lock:
            if not self.joint_queue:
                item = None
            else:
                item = (
                    self.joint_queue.popleft(),
                    self.base_queue.popleft(),
                    self.nav_aux_queue.popleft(),
                    self.bbox_queue.popleft(),
                    self.source_request_queue.popleft(),
                    self.source_chunk_step_queue.popleft(),
                    len(self.joint_queue),
                )
                self.model_queue.popleft()
                self._assert_queue_alignment_locked()
        if item is None:
            self.maybe_stop_base_on_idle()
            return
        joint_action, base_action, nav_aux_action, bbox_action, request_id, chunk_step, queue_len = item
        joint_command = self.publish_joint_action(joint_action)
        if joint_command is None:
            base_command = self.publish_zero_base("joint_hold")
        else:
            base_command = self.publish_base_action(base_action)
        publish_image = None
        if self.recorder.wants_publish_image(self.published_action_count):
            try:
                with self.observation_lock:
                    image_message = self.latest_images.get("cam_manip_high")
                if image_message is not None:
                    publish_image = ros_image_to_rgb(image_message)
            except Exception as exc:
                self.get_logger().warn(
                    "Failed to capture executed cam_manip_high frame: "
                    f"{type(exc).__name__}: {exc}"
                )
        nav_image = None
        if self.recorder.wants_publish_image(self.published_action_count):
            try:
                with self.observation_lock:
                    image_message = self.latest_images.get("cam_nav")
                if image_message is not None:
                    nav_image = ros_image_to_rgb(image_message)
            except Exception as exc:
                self.get_logger().warn(
                    "Failed to capture executed cam_nav frame: "
                    f"{type(exc).__name__}: {exc}"
                )
        self.recorder.record_publish(
            action_index=self.published_action_count,
            joint_command=joint_command,
            base_raw=base_action,
            base_command=base_command,
            queue_len_after_pop=queue_len,
            source_request_id=request_id,
            source_chunk_step=chunk_step,
            bbox_action=bbox_action,
            nav_aux_action=nav_aux_action,
            cam_manip_high=publish_image,
            cam_nav=nav_image,
        )

    def gripper_command(self, arm: str, raw_value: float) -> float:
        if self.binary_gripper:
            threshold = self.left_gripper_threshold_m if arm == "left" else self.right_gripper_threshold_m
            previous = self.binary_gripper_state[arm]
            is_open = raw_value >= threshold if previous is None else raw_value >= threshold
            self.binary_gripper_state[arm] = is_open
            return self.gripper_open_m if is_open else self.gripper_closed_m
        if self.clamp_gripper_commands:
            return float(np.clip(raw_value, self.gripper_min_m, self.gripper_max_m))
        return float(raw_value)

    def publish_joint_action(self, action: np.ndarray) -> np.ndarray | None:
        command = np.asarray(action, dtype=np.float32).reshape(JOINT_DIM).copy()
        with self.queue_lock:
            for arm, start in (("left", 0), ("right", 7)):
                held = self.manual_hold_joint[arm]
                if self.manual_hold[arm] and held is not None:
                    command[start : start + 7] = held
        raw_command = command.copy()
        reference = self.last_published_action if self.dry_run else self.current_joint_state()
        if reference is None:
            reference = self.current_joint_state()
        if reference is None and self.hold_when_feedback_missing:
            self.get_logger().warn("Holding joint action because feedback is unavailable.")
            return None
        if reference is not None and self.max_publish_joint_delta_rad > 0.0:
            reference = np.asarray(reference, dtype=np.float32).reshape(JOINT_DIM)
            for start in (0, 7):
                command[start : start + 6] = reference[start : start + 6] + np.clip(
                    command[start : start + 6] - reference[start : start + 6],
                    -self.max_publish_joint_delta_rad,
                    self.max_publish_joint_delta_rad,
                )
        left_gripper_raw, right_gripper_raw = float(command[6]), float(command[13])
        command[6] = self.gripper_command("left", left_gripper_raw)
        command[13] = self.gripper_command("right", right_gripper_raw)
        self.published_action_count += 1
        self.last_published_action = command.copy()
        should_log = self.published_action_count <= 5 or self.published_action_count % 30 == 0
        if self.dry_run:
            if self.log_published_actions and should_log:
                self.get_logger().info(
                    f"dry-run joint #{self.published_action_count} "
                    f"max_edge_adjustment={float(np.max(np.abs(command - raw_command))):.4f} "
                    f"gripper_raw=[{left_gripper_raw:.4f},{right_gripper_raw:.4f}] "
                    f"gripper_cmd=[{command[6]:.4f},{command[13]:.4f}]"
                )
            return command
        stamp = self.get_clock().now().to_msg()
        left = JointState()
        left.header.stamp = stamp
        left.name = self.joint_names
        left.position = [float(value) for value in command[:7]]
        left.velocity = [0.0] * 6 + [self.piper_speed_percent]
        left.effort = [0.0] * 6 + [self.gripper_effort]
        right = JointState()
        right.header.stamp = stamp
        right.name = self.joint_names
        right.position = [float(value) for value in command[7:14]]
        right.velocity = [0.0] * 6 + [self.piper_speed_percent]
        right.effort = [0.0] * 6 + [self.gripper_effort]
        self.left_pub.publish(left)
        self.right_pub.publish(right)
        if self.log_published_actions and should_log:
            self.get_logger().info(
                f"published joint #{self.published_action_count} "
                f"gripper_raw=[{left_gripper_raw:.4f},{right_gripper_raw:.4f}] "
                f"gripper_cmd=[{command[6]:.4f},{command[13]:.4f}]"
            )
        return command

    @staticmethod
    def _deadband(value: float, threshold: float) -> float:
        return 0.0 if abs(value) <= max(0.0, threshold) else value

    def clamp_base_target(self, raw_action: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw_action, dtype=np.float32).reshape(BASE_DIM)
        target = np.asarray(
            [raw[0] * self.base_vx_scale, raw[1] * self.base_vy_scale, raw[2] * self.base_wz_scale],
            dtype=np.float32,
        )
        target[0] = self._deadband(float(target[0]), self.base_vx_deadband_mps)
        target[1] = self._deadband(float(target[1]), self.base_vy_deadband_mps)
        target[2] = self._deadband(float(target[2]), self.base_wz_deadband_radps)
        if self.base_clamp_commands:
            target[0] = np.clip(target[0], self.base_min_vx_mps, self.base_max_vx_mps)
            target[1] = np.clip(target[1], -self.base_max_vy_mps, self.base_max_vy_mps)
            target[2] = np.clip(target[2], -self.base_max_wz_radps, self.base_max_wz_radps)
        return target.astype(np.float32)

    def ramp_base_target(self, target: np.ndarray) -> np.ndarray:
        now = time.monotonic()
        dt = max(1e-3, min(0.5, now - self.last_base_update_mono))
        result = np.asarray(target, dtype=np.float32).copy()
        previous = self.base_ramp_command
        if self.base_max_accel_mps2 > 0.0:
            delta = self.base_max_accel_mps2 * dt
            result[:2] = previous[:2] + np.clip(result[:2] - previous[:2], -delta, delta)
        if self.base_max_yaw_accel_radps2 > 0.0:
            delta = self.base_max_yaw_accel_radps2 * dt
            result[2] = previous[2] + np.clip(result[2] - previous[2], -delta, delta)
        self.last_base_update_mono = now
        self.base_ramp_command = result.copy()
        return result

    def publish_base_action(self, base_action: np.ndarray) -> np.ndarray | None:
        if not self.base_enabled:
            return None
        raw = np.asarray(base_action, dtype=np.float32).reshape(BASE_DIM)
        command = self.ramp_base_target(self.clamp_base_target(raw))
        if not self.base_dry_run or self.base_state_follows_dry_run_command:
            self.current_base_command = command.copy()
        self.publish_base_twist(command, raw=raw, reason="action")
        return command

    def publish_zero_base(self, reason: str) -> np.ndarray | None:
        if not self.base_enabled:
            return None
        zero = np.zeros(BASE_DIM, dtype=np.float32)
        self.current_base_command = zero.copy()
        self.base_ramp_command = zero.copy()
        self.last_base_update_mono = time.monotonic()
        self.publish_base_twist(zero, raw=zero, reason=reason)
        return zero

    def publish_base_twist(self, command: np.ndarray, *, raw: np.ndarray, reason: str) -> None:
        if not self.base_enabled or self.base_pub is None:
            return
        command = np.asarray(command, dtype=np.float32).reshape(BASE_DIM)
        self.published_base_count += 1
        self.last_base_publish_mono = time.monotonic()
        should_log = self.published_base_count <= 5 or self.published_base_count % 30 == 0 or reason != "action"
        if abs(float(command[1])) > 1e-5 and not self.warned_vy:
            self.warned_vy = True
            self.get_logger().warn(
                "Publishing vy on Twist.linear.y. The bundled direct_can_cmdvel_bridge transmits only vx/wz; "
                "a holonomic base bridge is required for physical vy execution."
            )
        if self.base_dry_run:
            if should_log:
                self.get_logger().info(
                    f"dry-run base #{self.published_base_count} reason={reason} "
                    f"raw={raw.tolist()} cmd={command.tolist()}"
                )
            return
        twist = Twist()
        twist.linear.x, twist.linear.y, twist.linear.z = float(command[0]), float(command[1]), 0.0
        twist.angular.x, twist.angular.y, twist.angular.z = 0.0, 0.0, float(command[2])
        self.base_pub.publish(twist)
        if should_log:
            self.get_logger().info(
                f"published base #{self.published_base_count} reason={reason} cmd={command.tolist()}"
            )

    def maybe_stop_base_on_idle(self) -> None:
        if not self.base_enabled or not self.base_stop_on_empty_queue:
            return
        if self.last_base_publish_mono <= 0.0:
            return
        if time.monotonic() - self.last_base_publish_mono < self.base_command_timeout_s:
            return
        if float(np.max(np.abs(self.base_ramp_command))) > 1e-6:
            self.publish_zero_base("empty_queue_timeout")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    rclpy.init()
    node = UniWAMPiperAsyncPrefix12Rot6DClient(cfg)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
