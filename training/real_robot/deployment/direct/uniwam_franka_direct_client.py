"""Direct Franka/RealSense client for the UniWAM camera-frame H32 policy.

The robot-side runtime is deliberately inherited from ``openpi_robot_client``:
Polymetis controls the two arms and pyrealsense2 supplies the three cameras.
Only the policy transport and the camera-frame H32 state/action adapter live
here. No ROS2 node, Piper IK, or mobile-base interface is used.
"""

from __future__ import annotations

import argparse
import io
import socket
import struct
import threading
import time
import json
from collections import deque
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from openpi_robot_client import (
    DEFAULT_GRIPPER_MAX_OPEN,
    DEFAULT_LEFT_GRIPPER_PORT,
    DEFAULT_LEFT_ROBOT_HOST,
    DEFAULT_LEFT_ROBOT_PORT,
    DEFAULT_CAMERA_NAMES,
    DEFAULT_CAMERA_SERIALS,
    ArmCommandState,
    ArmTarget,
    FrankaArm,
    Rate,
    RealSenseCameras,
    as_numpy,
    command_arm_to_target,
    limit_rotation_step,
    limit_vector_step,
    normalize_quat_xyzw,
)


HEADER = struct.Struct("!I")
MAX_FRAME_BYTES = 256 * 1024 * 1024
EEF_DIM = 14
MODEL_ACTION_DIM = 17
STATE_DIM = 23
MODEL_GRIPPER_MAX_WIDTH_M = 0.09

# Exact Franka hand FK constants copied from the supplied direct reference
# runtime. The training source uses this franka_hand TCP, not a Polymetis
# panda_link8 pose plus an approximate post-hoc offset.
_FRANKA_ORIGIN_XYZ = np.asarray(
    (
        (0.0, 0.0, 0.333),
        (0.0, 0.0, 0.0),
        (0.0, -0.316, 0.0),
        (0.0825, 0.0, 0.0),
        (-0.0825, 0.384, 0.0),
        (0.0, 0.0, 0.0),
        (0.088, 0.0, 0.0),
    ), dtype=np.float64
)
_SQRT_HALF = np.sqrt(0.5)
_FRANKA_ORIGIN_WXYZ = np.asarray(
    (
        (1.0, 0.0, 0.0, 0.0),
        (_SQRT_HALF, -_SQRT_HALF, 0.0, 0.0),
        (_SQRT_HALF, _SQRT_HALF, 0.0, 0.0),
        (_SQRT_HALF, _SQRT_HALF, 0.0, 0.0),
        (_SQRT_HALF, -_SQRT_HALF, 0.0, 0.0),
        (_SQRT_HALF, _SQRT_HALF, 0.0, 0.0),
        (_SQRT_HALF, _SQRT_HALF, 0.0, 0.0),
    ), dtype=np.float64
)
_FRANKA_HAND_XYZ = np.asarray((0.0, 0.0, 0.2104), dtype=np.float64)
_FRANKA_HAND_WXYZ = np.asarray(
    (np.cos(-np.pi / 8.0), 0.0, 0.0, np.sin(-np.pi / 8.0)), dtype=np.float64
)
FRANKA_PROMPT_PREFIX = (
    "Two white Franka arms stand across the table from a fixed D455 camera. "
    "Express end-effector states and actions in this fixed camera frame."
)
FRANKA_BASE_PROMPT_PREFIX = (
    "Two white Franka arms stand across the table from a fixed external D455 camera, "
    "with a D435 camera on each wrist. Express each arm's end-effector state and action "
    "in that arm's own fixed base frame."
)
FRANKA_CAMERA_FROM_LEFT_BASE = np.asarray(
    [[0.14730180, 0.98371190, 0.10301948, 0.19956264],
     [0.87091390, -0.07962561, -0.48494180, -0.43527493],
     [-0.46884006, 0.16115388, -0.86845730, 1.33748340],
     [0.0, 0.0, 0.0, 1.0]], dtype=np.float32
)
FRANKA_CAMERA_FROM_RIGHT_BASE = np.asarray(
    [[0.07129902, 0.99705476, -0.02825694, -0.39148414],
     [0.86807780, -0.07597923, -0.49057943, -0.39173418],
     [-0.49128145, 0.01044863, -0.87093840, 1.27074170],
     [0.0, 0.0, 0.0, 1.0]], dtype=np.float32
)


def _encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        buffer = io.BytesIO()
        np.save(buffer, value, allow_pickle=False)
        return {"__ndarray__": True, "payload": buffer.getvalue()}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported msgpack value: {type(value)!r}")


def _decode(value: Any) -> Any:
    if isinstance(value, dict) and value.get("__ndarray__"):
        return np.load(io.BytesIO(value["payload"]), allow_pickle=False)
    return value


def send_message(sock: socket.socket, payload: dict[str, Any]) -> None:
    body = msgpack.packb(payload, default=_encode, use_bin_type=True)
    sock.sendall(HEADER.pack(len(body)) + body)


def recv_exact(sock: socket.socket, size: int) -> bytes:
    parts = []
    while size:
        part = sock.recv(size)
        if not part:
            raise EOFError("Cloud socket closed while receiving a response.")
        parts.append(part)
        size -= len(part)
    return b"".join(parts)


def recv_message(sock: socket.socket) -> dict[str, Any]:
    (size,) = HEADER.unpack(recv_exact(sock, HEADER.size))
    if size <= 0 or size > MAX_FRAME_BYTES:
        raise RuntimeError(f"Invalid cloud frame size: {size}")
    payload = msgpack.unpackb(recv_exact(sock, size), raw=False, object_hook=_decode)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Cloud response is not a dict: {type(payload)!r}")
    return payload


def resize_aspect_pad_rgb(
    image_rgb: np.ndarray,
    *,
    target_width: int = 424,
    target_height: int = 240,
) -> np.ndarray:
    """Match the training input: preserve 4:3 content, edge-pad to 424x240."""
    image = np.asarray(image_rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB HWC image, got {image.shape}")
    height, width = image.shape[:2]
    scale = min(float(target_width) / width, float(target_height) / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    bilinear = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
    resized_rgb = np.asarray(
        Image.fromarray(image, mode="RGB").resize(
            (resized_width, resized_height), resample=bilinear
        ),
        dtype=np.uint8,
    )
    canvas_rgb = np.empty((target_height, target_width, 3), dtype=np.uint8)
    pad_top = (target_height - resized_height) // 2
    pad_left = (target_width - resized_width) // 2
    canvas_rgb[...] = resized_rgb[-1:, -1:, :]
    canvas_rgb[pad_top : pad_top + resized_height, pad_left : pad_left + resized_width] = resized_rgb
    if pad_top:
        canvas_rgb[:pad_top, pad_left : pad_left + resized_width] = resized_rgb[:1]
        canvas_rgb[pad_top + resized_height :, pad_left : pad_left + resized_width] = resized_rgb[-1:]
    if pad_left:
        canvas_rgb[:, :pad_left] = canvas_rgb[:, pad_left : pad_left + 1]
        canvas_rgb[:, pad_left + resized_width :] = canvas_rgb[:, pad_left + resized_width - 1 : pad_left + resized_width]
    return canvas_rgb


def encode_rgb_jpeg(image_rgb: np.ndarray, quality: int = 85) -> bytes:
    # The supplied RealSenseCameras.read() already returns RGB arrays.
    image = resize_aspect_pad_rgb(image_rgb, target_width=424, target_height=240)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB HWC image, got {image.shape}")
    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="JPEG", quality=int(quality))
    return buffer.getvalue()


def rotation6d_from_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    matrix = Rotation.from_quat(normalize_quat_xyzw(quat_xyzw)).as_matrix()
    return matrix[:2, :].reshape(6).astype(np.float32)


def _quat_wxyz_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ), axis=-1
    )


def _quat_wxyz_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    v = np.asarray(vector, dtype=np.float64)
    qv = q[..., 1:]
    cross = np.cross(qv, v)
    return v + 2.0 * (q[..., :1] * cross + np.cross(qv, cross))


def franka_hand_pose_from_joints(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact reference FK: 7 joints -> franka_hand xyz and xyzw quaternion."""
    joints = np.asarray(joints, dtype=np.float64).reshape(7)
    if not np.all(np.isfinite(joints)):
        raise ValueError("Franka joints contain NaN or Inf.")
    position = np.zeros(3, dtype=np.float64)
    quaternion = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
    for index in range(7):
        position += _quat_wxyz_rotate(quaternion, _FRANKA_ORIGIN_XYZ[index])
        quaternion = _quat_wxyz_multiply(quaternion, _FRANKA_ORIGIN_WXYZ[index])
        half_angle = joints[index] / 2.0
        joint_quaternion = np.asarray(
            (np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)), dtype=np.float64
        )
        quaternion = _quat_wxyz_multiply(quaternion, joint_quaternion)
    position += _quat_wxyz_rotate(quaternion, _FRANKA_HAND_XYZ)
    quaternion = _quat_wxyz_multiply(quaternion, _FRANKA_HAND_WXYZ)
    quaternion /= np.linalg.norm(quaternion)
    return position, np.asarray(
        (quaternion[1], quaternion[2], quaternion[3], quaternion[0]), dtype=np.float64
    )


def polymetis_width_to_model_gripper(width_m: float) -> float:
    """Convert live Polymetis metres to the training normalized gripper state."""
    return float(np.clip(1.0 - float(width_m) / MODEL_GRIPPER_MAX_WIDTH_M, 0.0, 1.0))


def state23_from_arms(left: FrankaArm, right: FrankaArm) -> np.ndarray:
    left_position, left_quat = franka_hand_pose_from_joints(
        as_numpy(left.robot.get_joint_positions()).reshape(7)
    )
    right_position, right_quat = franka_hand_pose_from_joints(
        as_numpy(right.robot.get_joint_positions()).reshape(7)
    )
    state = np.zeros(STATE_DIM, dtype=np.float32)
    state[3:6] = np.asarray(left_position, dtype=np.float32)
    state[6:12] = rotation6d_from_xyzw(left_quat)
    state[12] = polymetis_width_to_model_gripper(left.get_gripper_width())
    state[13:16] = np.asarray(right_position, dtype=np.float32)
    state[16:22] = rotation6d_from_xyzw(right_quat)
    state[22] = polymetis_width_to_model_gripper(right.get_gripper_width())
    if not np.all(np.isfinite(state)):
        raise ValueError("Franka state contains NaN or Inf.")
    return state


def polymetis_ee_to_model_tcp(pose: ArmCommandState) -> ArmCommandState:
    """Convert live panda_link8 feedback to the training franka_hand TCP."""
    ee_rotation = Rotation.from_quat(normalize_quat_xyzw(pose.quat_xyzw))
    tcp_from_ee = Rotation.from_euler("z", -np.pi / 4.0)
    tcp_rotation = ee_rotation * tcp_from_ee
    tcp_position = np.asarray(pose.position, dtype=np.float64) + tcp_rotation.apply(
        [0.0, 0.0, 0.2104 - 0.107]
    )
    return ArmCommandState(
        position=tcp_position.astype(np.float64),
        quat_xyzw=normalize_quat_xyzw(tcp_rotation.as_quat()),
    )


def model_tcp_target_to_polymetis_ee_target(
    tcp_position: np.ndarray, tcp_rpy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert fixed-camera model output (after cloud inverse transform) to link8."""
    tcp_position = np.asarray(tcp_position, dtype=np.float64).reshape(3)
    tcp_rotation = Rotation.from_euler("xyz", np.asarray(tcp_rpy, dtype=np.float64).reshape(3))
    tcp_from_ee = Rotation.from_euler("z", -np.pi / 4.0)
    ee_rotation = tcp_rotation * tcp_from_ee.inv()
    ee_position = tcp_position + tcp_rotation.apply([0.0, 0.0, -(0.2104 - 0.107)])
    return ee_position.astype(np.float64), normalize_quat_xyzw(ee_rotation.as_quat())


def gripper_norm_to_width(
    value: float,
    maximum: float,
    threshold: float,
    enabled: bool,
    binary: bool = False,
    binary_threshold: float = 0.5,
    binary_open_threshold: float = 0.2,
    binary_state: bool | None = None,
    single_threshold: float | None = None,
) -> tuple[float, bool] | float:
    # Franka training labels use the reference normalized convention: 0=open,
    # 1=closed. Polymetis receives the converted physical width in metres.
    normalized = float(np.clip(value, 0.0, 1.0))
    width = (1.0 - normalized) * float(maximum)
    if single_threshold is not None:
        closed = normalized >= float(single_threshold)
        return (0.0 if closed else float(maximum)), closed
    if binary:
        if binary_state is None:
            binary_state = normalized >= float(binary_threshold)
        elif binary_state and normalized <= float(binary_open_threshold):
            binary_state = False
        elif not binary_state and normalized >= float(binary_threshold):
            binary_state = True
        return (0.0 if binary_state else float(maximum)), binary_state
    return 0.0 if enabled and width <= threshold else width


def parse_extrinsic(value: str | None, default: Any) -> np.ndarray:
    if value in (None, ""):
        return np.asarray(default, dtype=np.float32).reshape(4, 4)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = [float(item) for item in value.split(",") if item.strip()]
    matrix = np.asarray(parsed, dtype=np.float32)
    if matrix.size != 16:
        raise ValueError("An explicit Franka extrinsic must contain 16 row-major values.")
    return matrix.reshape(4, 4)


class UniWAMPolicyClient:
    def __init__(
        self,
        host: str,
        port: int,
        profile: str,
        left_extrinsic: Any | None,
        right_extrinsic: Any | None,
        prompt_prefix: str = FRANKA_PROMPT_PREFIX,
    ):
        self.host, self.port = host, int(port)
        self.profile = profile
        self.prompt_prefix = str(prompt_prefix).strip()
        if not self.prompt_prefix:
            raise ValueError("The Franka prompt prefix must not be empty.")
        if (left_extrinsic is None) != (right_extrinsic is None):
            raise ValueError("Explicit Franka extrinsics require both left and right matrices.")
        self.left_extrinsic = (
            None if left_extrinsic is None else np.asarray(left_extrinsic, dtype=np.float32).reshape(4, 4)
        )
        self.right_extrinsic = (
            None if right_extrinsic is None else np.asarray(right_extrinsic, dtype=np.float32).reshape(4, 4)
        )
        self.sock: socket.socket | None = None
        self.request_id = 0

    def connect(self) -> None:
        if self.sock is None:
            self.sock = socket.create_connection((self.host, self.port), timeout=120.0)
            self.sock.settimeout(120.0)
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2 * 1024 * 1024)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)

    def infer(
        self,
        *,
        state: np.ndarray,
        images: dict[str, bytes],
        prompt: str,
        prefix: np.ndarray | None,
        prefix_eef_xy: np.ndarray | None,
        queue_len: int,
    ) -> dict[str, Any]:
        self.connect()
        self.request_id += 1
        full_prompt = str(prompt).strip()
        if not full_prompt.startswith(self.prompt_prefix):
            full_prompt = f"{self.prompt_prefix} {full_prompt}"
        packet = {
            "type": "observation",
            "robot_type": "franka",
            "request_id": self.request_id,
            "stamp": time.time(),
            "instruction": full_prompt,
            "inference_mode": "manip_only",
            "task_profile": self.profile,
            "extrinsics_profile": self.profile,
            "camera_from_left_base": self.left_extrinsic,
            "camera_from_right_base": self.right_extrinsic,
            "eef_state": state,
            "joint_state": None,
            "images": images,
            "image_encoding": "jpeg",
            "prefix_mode": "head",
            "prefix_eef_actions": prefix,
            "prefix_bbox_actions": prefix_eef_xy,
            "prefix_joint_actions": None,
            "queue_len_at_request": int(queue_len),
            "queue_epoch_at_request": 0,
        }
        try:
            send_message(self.sock, packet)
            response = recv_message(self.sock)
        except Exception:
            if self.sock is not None:
                self.sock.close()
                self.sock = None
            raise
        if not response.get("ok", False):
            raise RuntimeError(response.get("error", "unknown cloud error"))
        if response.get("actuation_backend") != "eef_passthrough":
            raise RuntimeError("The cloud is not using the Franka eef_passthrough backend.")
        return response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=18025)
    parser.add_argument("--task-profile", default="auto", help="auto, cup_pyramid, or insert_tube")
    parser.add_argument("--prompt-prefix", default=FRANKA_PROMPT_PREFIX)
    parser.add_argument("--left-extrinsic", default=None, help="JSON 4x4 or 16 comma-separated camera_from_left_base values")
    parser.add_argument("--right-extrinsic", default=None, help="JSON 4x4 or 16 comma-separated camera_from_right_base values")
    parser.add_argument("--prompt", default="Build a stable pyramid from six colored cups gathered from both sides: three cups on the bottom, two in the middle, and one on top.")
    parser.add_argument(
        "--camera-serials",
        nargs=3,
        default=("425122300063", "408322073690", "408322073870"),
    )
    parser.add_argument("--camera-names", nargs=3, default=("external", "left_wrist", "right_wrist"))
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-warmup-frames", type=int, default=10)
    parser.add_argument("--left-robot-host", default=DEFAULT_LEFT_ROBOT_HOST)
    parser.add_argument("--left-robot-port", type=int, default=DEFAULT_LEFT_ROBOT_PORT)
    parser.add_argument("--left-gripper-port", type=int, default=DEFAULT_LEFT_GRIPPER_PORT)
    parser.add_argument("--right-robot-host", default="192.168.1.101")
    parser.add_argument("--right-robot-port", type=int, default=50051)
    parser.add_argument("--right-gripper-port", type=int, default=50053)
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=0,
        help="Number of executed action steps; 0 means run until Ctrl-C.",
    )
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--chunk-low-watermark", type=int, default=8)
    parser.add_argument("--prefix-steps", type=int, default=12)
    parser.add_argument("--min-prefix-steps", type=int, default=6)
    parser.add_argument("--max-action-pos-delta", type=float, default=0.06)
    parser.add_argument("--max-action-rot-delta", type=float, default=0.35)
    parser.add_argument("--max-pos-step", type=float, default=0.003)
    parser.add_argument("--max-rot-step", type=float, default=0.03)
    parser.add_argument("--gripper-max-open", type=float, default=DEFAULT_GRIPPER_MAX_OPEN)
    parser.add_argument("--gripper-speed", type=float, default=1.0)
    parser.add_argument("--gripper-force", type=float, default=1.0)
    parser.add_argument("--gripper-close-threshold", type=float, default=0.02)
    parser.add_argument("--enable-gripper-close-threshold", action="store_true")
    parser.add_argument("--enable-gripper-binary", action="store_true")
    parser.add_argument(
        "--gripper-binary-close-threshold",
        type=float,
        default=0.5,
        help="Close when normalized UniWAM gripper output reaches this value.",
    )
    parser.add_argument(
        "--gripper-binary-open-threshold",
        type=float,
        default=0.2,
        help="Re-open a latched gripper only when normalized output drops to this value.",
    )
    parser.add_argument(
        "--gripper-binary-threshold",
        type=float,
        default=None,
        help="Single-threshold, non-latched mode: raw >= threshold closes, otherwise opens.",
    )
    parser.add_argument(
        "--gripper-binary-mode",
        choices=("single", "hysteresis"),
        default=None,
        help="Select one-threshold switching or two-threshold latched switching.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    left_extrinsic = parse_extrinsic(args.left_extrinsic, FRANKA_CAMERA_FROM_LEFT_BASE)
    right_extrinsic = parse_extrinsic(args.right_extrinsic, FRANKA_CAMERA_FROM_RIGHT_BASE)
    policy = UniWAMPolicyClient(
        args.policy_host,
        args.policy_port,
        args.task_profile.strip().lower() or "auto",
        left_extrinsic,
        right_extrinsic,
        args.prompt_prefix,
    )
    left = FrankaArm(name="left", robot_host=args.left_robot_host, robot_port=args.left_robot_port, gripper_host=args.left_robot_host, gripper_port=args.left_gripper_port, require_gripper=True, gripper_max_open=args.gripper_max_open)
    right = FrankaArm(name="right", robot_host=args.right_robot_host, robot_port=args.right_robot_port, gripper_host=args.right_robot_host, gripper_port=args.right_gripper_port, require_gripper=True, gripper_max_open=args.gripper_max_open)
    cameras = RealSenseCameras(camera_names=args.camera_names, camera_serials=args.camera_serials, width=args.camera_width, height=args.camera_height, fps=args.camera_fps, warmup_frames=args.camera_warmup_frames)
    if not 0.0 <= args.gripper_binary_open_threshold < args.gripper_binary_close_threshold <= 1.0:
        raise ValueError("Binary gripper thresholds must satisfy 0 <= open < close <= 1.")
    if args.gripper_binary_threshold is not None and not 0.0 <= args.gripper_binary_threshold <= 1.0:
        raise ValueError("--gripper-binary-threshold must be in [0,1].")
    if args.gripper_binary_mode == "single" and args.gripper_binary_threshold is None:
        raise ValueError("--gripper-binary-mode single requires --gripper-binary-threshold.")
    if args.gripper_binary_mode == "hysteresis" and args.gripper_binary_threshold is not None:
        raise ValueError("Do not combine hysteresis mode with --gripper-binary-threshold.")
    binary_mode = args.gripper_binary_mode
    if binary_mode is None:
        if args.gripper_binary_threshold is not None:
            binary_mode = "single"
        elif args.enable_gripper_binary:
            binary_mode = "hysteresis"
    if args.execute and not args.yes:
        if input("Type EXECUTE_FRANKA to continue: ").strip() != "EXECUTE_FRANKA":
            raise SystemExit("Aborted")
    if args.execute:
        left.start_cartesian_impedance(); right.start_cartesian_impedance()
    queue: deque[np.ndarray] = deque()
    eef_xy_queue: deque[np.ndarray] = deque()
    inflight = False
    lock = threading.Lock()
    last_left = left.get_command_state(); last_right = right.get_command_state()
    binary_gripper_state: dict[str, bool | None] = {"left": None, "right": None}
    errors = []

    def request_worker():
        nonlocal inflight
        try:
            with lock:
                prefix = np.stack(list(queue)[:args.prefix_steps]).astype(np.float32) if len(queue) >= args.min_prefix_steps else None
                # EEF auxiliary prefix must have exactly the same length as
                # the action prefix sent in this request.
                prefix_eef_xy = (
                    np.stack(list(eef_xy_queue)[:len(prefix)]).astype(np.float32)
                    if prefix is not None and len(eef_xy_queue) >= len(prefix)
                    else None
                )
                queue_len = len(queue)
            # Match the supplied Franka direct runtime: snapshot robot state
            # first, then acquire the three camera frames for this request.
            state_snapshot = state23_from_arms(left, right)
            frames = cameras.read()
            images = {"cam_nav": encode_rgb_jpeg(frames["external"]), "cam_manip_high": encode_rgb_jpeg(frames["external"]), "cam_left_wrist": encode_rgb_jpeg(frames["left_wrist"]), "cam_right_wrist": encode_rgb_jpeg(frames["right_wrist"])}
            response = policy.infer(state=state_snapshot, images=images, prompt=args.prompt, prefix=prefix, prefix_eef_xy=prefix_eef_xy, queue_len=queue_len)
            actions = np.asarray(response["eef_actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != EEF_DIM:
                raise RuntimeError(f"Bad cloud EEF action shape: {actions.shape}")
            model_actions = np.concatenate([actions, np.zeros((len(actions), 3), dtype=np.float32)], axis=1)
            eef_xy_actions = np.asarray(response.get("eef_xy_actions", response.get("bbox_actions")), dtype=np.float32)
            if eef_xy_actions.shape != (len(actions), 6) or not np.all(np.isfinite(eef_xy_actions)):
                raise RuntimeError(f"Bad cloud EEF-XY/visibility shape: {eef_xy_actions.shape}")
            prefix_len = int(response.get("prefix_length", 0))
            with lock:
                keep = min(prefix_len, len(queue))
                while len(queue) > keep: queue.pop()
                while len(eef_xy_queue) > keep: eef_xy_queue.pop()
                queue.extend(model_actions)
                eef_xy_queue.extend(eef_xy_actions)
                while len(queue) > 32: queue.pop()
                while len(eef_xy_queue) > 32: eef_xy_queue.pop()
            print(f"received H32 prefix={prefix_len} execute={len(actions)} queue={len(queue)} model_infer={float(response.get('request_model_infer_s', -1)):.3f}s", flush=True)
        except Exception as exc:
            with lock:
                queue.clear()
            errors.append(exc)
            print(f"inference request failed: {type(exc).__name__}: {exc}", flush=True)
        finally:
            inflight = False

    rate = Rate(args.control_hz)
    try:
        executed_steps = 0
        while args.rollout_steps <= 0 or executed_steps < args.rollout_steps:
            if errors:
                raise RuntimeError(
                    "Stopping direct execution after an inference failure: "
                    f"{type(errors[-1]).__name__}: {errors[-1]}"
                )
            with lock:
                need = len(queue) == 0 or len(queue) <= args.chunk_low_watermark
                action = queue.popleft()[:EEF_DIM] if queue else None
            if need and not inflight:
                inflight = True
                threading.Thread(target=request_worker, daemon=True).start()
            if action is not None:
                targets = []
                for arm_name, arm_action, previous in (("left", action[:7], last_left), ("right", action[7:], last_right)):
                    pos, quat = model_tcp_target_to_polymetis_ee_target(arm_action[:3], arm_action[3:6])
                    gripper_result = gripper_norm_to_width(
                        arm_action[6],
                        args.gripper_max_open,
                        args.gripper_close_threshold,
                        args.enable_gripper_close_threshold,
                        binary_mode is not None,
                        args.gripper_binary_close_threshold,
                        args.gripper_binary_open_threshold,
                        binary_gripper_state[arm_name],
                        args.gripper_binary_threshold if binary_mode == "single" else None,
                    )
                    if binary_mode is not None:
                        gripper_width, binary_gripper_state[arm_name] = gripper_result
                    else:
                        gripper_width = gripper_result
                    target = ArmTarget(position=limit_vector_step(previous.position, pos, args.max_action_pos_delta), quat_xyzw=limit_rotation_step(previous.quat_xyzw, quat, args.max_action_rot_delta), gripper_width=gripper_width)
                    targets.append(target)
                if args.execute:
                    last_left = command_arm_to_target(arm=left, target=targets[0], last_command=last_left, max_pos_step=args.max_pos_step, max_rot_step=args.max_rot_step, gripper_speed=args.gripper_speed, gripper_force=args.gripper_force, gripper_max_open=args.gripper_max_open)
                    last_right = command_arm_to_target(arm=right, target=targets[1], last_command=last_right, max_pos_step=args.max_pos_step, max_rot_step=args.max_rot_step, gripper_speed=args.gripper_speed, gripper_force=args.gripper_force, gripper_max_open=args.gripper_max_open)
                else:
                    last_left = ArmCommandState(targets[0].position, targets[0].quat_xyzw); last_right = ArmCommandState(targets[1].position, targets[1].quat_xyzw)
                if executed_steps % 10 == 0:
                    print(
                        f"step={executed_steps} dry_run={not args.execute} "
                        f"left={targets[0].position.tolist()} "
                        f"right={targets[1].position.tolist()} "
                        f"gripper_raw=[{float(action[6]):.3f},{float(action[13]):.3f}] "
                        f"gripper_width=[{targets[0].gripper_width:.4f},{targets[1].gripper_width:.4f}] "
                        f"gripper_closed={binary_gripper_state}",
                        flush=True,
                    )
                executed_steps += 1
            rate.sleep()
    finally:
        cameras.close()
        if args.execute:
            left.terminate_policy(); right.terminate_policy()


if __name__ == "__main__":
    main()
