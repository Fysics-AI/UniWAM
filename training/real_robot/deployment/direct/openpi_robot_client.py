#!/usr/bin/env python3
"""Shared robot/camera runtime used by the local cup-pyramid client.

Default policy layout for ``--action-mode absolute-quat``:
    observation.images.external
    observation.images.left_wrist
    observation.state = [x, y, z, qw, qx, qy, qz, gripper_width]
    prompt

Default policy action:
    [x, y, z, qw, qx, qy, qz, gripper_target]

The absolute-quat action is used directly as the target TCP pose. Quaternions
are represented as wxyz in the dataset/model space and converted to xyzw only at
the Polymetis update_desired_ee_pose boundary.

``--action-mode absolute-rotvec`` observation:
    [x, y, z, rx, ry, rz, gripper_width]

``absolute-rotvec`` action:
    [x, y, z, rx, ry, rz, gripper_target]

This is for rotvec configs whose inference output transform
has already converted the model output back to absolute xyz+rotvec actions.
The client converts the absolute rotvec orientation to a quaternion before
sending the target TCP pose to Polymetis.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation


OPENPI_CLIENT_SRC = "/home/pnp/openpi-main/packages/openpi-client/src"
DEFAULT_POLICY_HOST = "127.0.0.1"
DEFAULT_POLICY_PORT = 8000
DEFAULT_PROMPT = "Hold the pink cup by its sides and place it inside the blue cup."
DEFAULT_CAMERA_NAMES = ("left_wrist", "external")
DEFAULT_CAMERA_SERIALS = ("408322073690", "425122300063")
DEFAULT_LEFT_ROBOT_HOST = "192.168.1.100"
DEFAULT_LEFT_ROBOT_PORT = 50052
DEFAULT_LEFT_GRIPPER_PORT = 50054
DEFAULT_GRIPPER_MAX_OPEN = 0.085
DEFAULT_CONTROL_HZ = 30.0
DEFAULT_MODEL_ACTION_HZ = 30.0
DEFAULT_EXPECTED_ACTION_HORIZON = 50
DEFAULT_OPEN_LOOP_HORIZON = 50
DEFAULT_ACTION_MODE = "absolute-quat"
ABSOLUTE_QUAT_DIM = 8
ABSOLUTE_ROTVEC_DIM = 7
MAX_LOG_ACTION_DIM = 8

PREDICT_CSV_FIELDS = (
    "step",
    "time",
    "source_index",
    "interp_alpha",
    "model_cursor_after",
    "model_step_per_tick",
    "expected_action_horizon",
    "open_loop_horizon",
    "inference_sec",
    "execute",
    "action_mode",
    "reference_pose_source",
    "gripper_close_threshold_enabled",
    "gripper_close_threshold",
    "action_0",
    "action_1",
    "action_2",
    "action_3",
    "action_4",
    "action_5",
    "action_6",
    "action_7",
    "rel_dx",
    "rel_dy",
    "rel_dz",
    "rel_drx",
    "rel_dry",
    "rel_drz",
    "gripper",
    "target_x",
    "target_y",
    "target_z",
    "target_qw",
    "target_qx",
    "target_qy",
    "target_qz",
)


def as_numpy(value: Any, dtype: np.dtype = np.float64) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype).copy()


def normalize_quat_xyzw(quat_xyzw: Sequence[float]) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quat)
    if not np.isfinite(norm) or norm < 1e-9:
        raise ValueError(f"Invalid quaternion: {quat}")
    return quat / norm


def xyzw_to_wxyz(quat_xyzw: Sequence[float]) -> np.ndarray:
    x, y, z, w = normalize_quat_xyzw(quat_xyzw)
    return np.asarray([w, x, y, z], dtype=np.float64)


def wxyz_to_xyzw(quat_wxyz: Sequence[float]) -> np.ndarray:
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    return normalize_quat_xyzw([x, y, z, w])


def quat_xyzw_to_rotvec(quat_xyzw: Sequence[float]) -> np.ndarray:
    return Rotation.from_quat(normalize_quat_xyzw(quat_xyzw)).as_rotvec()


def action_dim_for_mode(action_mode: str) -> int:
    if action_mode == "absolute-quat":
        return ABSOLUTE_QUAT_DIM
    if action_mode == "absolute-rotvec":
        return ABSOLUTE_ROTVEC_DIM
    raise ValueError(f"Unsupported action mode: {action_mode}")


def limit_vector_step(start: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
    if max_step <= 0:
        return target
    delta = target - start
    distance = float(np.linalg.norm(delta))
    if distance <= max_step or distance < 1e-12:
        return target
    return start + delta * (max_step / distance)


def limit_rotation_step(
    start_quat_xyzw: Sequence[float],
    target_quat_xyzw: Sequence[float],
    max_angle: float,
) -> np.ndarray:
    if max_angle <= 0:
        return normalize_quat_xyzw(target_quat_xyzw)
    start = Rotation.from_quat(normalize_quat_xyzw(start_quat_xyzw))
    target = Rotation.from_quat(normalize_quat_xyzw(target_quat_xyzw))
    delta = target * start.inv()
    angle = float(delta.magnitude())
    if angle <= max_angle or angle < 1e-12:
        return target.as_quat()
    limited = Rotation.from_rotvec(delta.as_rotvec() * (max_angle / angle)) * start
    return normalize_quat_xyzw(limited.as_quat())


class Rate:
    def __init__(self, hz: float) -> None:
        if hz <= 0:
            raise ValueError("--control-hz must be positive")
        self.period = 1.0 / hz
        self.next_tick = time.monotonic()
        self.overruns = 0

    def sleep(self) -> None:
        self.next_tick += self.period
        remaining = self.next_tick - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        else:
            self.overruns += 1
            self.next_tick = time.monotonic()


@contextlib.contextmanager
def defer_keyboard_interrupt():
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum: int, frame: Any) -> None:
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


@dataclass
class ArmCommandState:
    position: np.ndarray
    quat_xyzw: np.ndarray

    @property
    def quat_wxyz(self) -> np.ndarray:
        return xyzw_to_wxyz(self.quat_xyzw)


@dataclass
class ArmTarget:
    position: np.ndarray
    quat_xyzw: np.ndarray
    gripper_width: float

    @property
    def quat_wxyz(self) -> np.ndarray:
        return xyzw_to_wxyz(self.quat_xyzw)


class FrankaArm:
    def __init__(
        self,
        *,
        name: str,
        robot_host: str,
        robot_port: int,
        gripper_host: str,
        gripper_port: int,
        require_gripper: bool,
        gripper_max_open: float,
    ) -> None:
        from polymetis import GripperInterface, RobotInterface

        self.name = name
        self.gripper_max_open = float(gripper_max_open)
        try:
            self.robot = RobotInterface(
                ip_address=robot_host,
                port=robot_port,
                enforce_version=False,
            )
        except TypeError:
            self.robot = RobotInterface(ip_address=robot_host, port=robot_port)

        self.gripper = None
        try:
            self.gripper = GripperInterface(ip_address=gripper_host, port=gripper_port)
        except Exception:
            if require_gripper:
                raise
            print(
                f"[warn] {name} gripper unavailable at {gripper_host}:{gripper_port}; "
                "using --gripper-max-open as observed width.",
                flush=True,
            )

    def get_command_state(self) -> ArmCommandState:
        position, quat_xyzw = self.robot.get_ee_pose()
        return ArmCommandState(
            position=as_numpy(position),
            quat_xyzw=normalize_quat_xyzw(as_numpy(quat_xyzw)),
        )

    def get_gripper_width(self) -> float:
        if self.gripper is not None:
            return float(self.gripper.get_state().width)
        return self.gripper_max_open

    def get_state(self, action_mode: str) -> np.ndarray:
        pose = self.get_command_state()
        gripper_width = self.get_gripper_width()
        if action_mode == "absolute-quat":
            return np.concatenate([pose.position, pose.quat_wxyz, [gripper_width]]).astype(np.float32)
        if action_mode == "absolute-rotvec":
            rotvec = quat_xyzw_to_rotvec(pose.quat_xyzw)
            return np.concatenate([pose.position, rotvec, [gripper_width]]).astype(np.float32)
        raise ValueError(f"Unsupported action mode: {action_mode}")

    def start_cartesian_impedance(self) -> None:
        self.robot.start_cartesian_impedance()

    def update_pose(self, position: np.ndarray, quat_xyzw: np.ndarray) -> None:
        self.robot.update_desired_ee_pose(
            position=torch.as_tensor(position, dtype=torch.float32),
            orientation=torch.as_tensor(quat_xyzw, dtype=torch.float32),
        )

    def goto_gripper(self, width_m: float, speed: float, force: float) -> None:
        if self.gripper is None:
            return
        width_m = float(np.clip(width_m, 0.0, self.gripper_max_open))
        try:
            self.gripper.goto(width=width_m, speed=speed, force=force, blocking=False)
        except TypeError:
            self.gripper.goto(width=width_m, speed=speed, force=force)

    def terminate_policy(self) -> None:
        try:
            self.robot.terminate_current_policy()
        except Exception as exc:
            print(f"[warn] failed to terminate {self.name} policy: {exc}", flush=True)


class RealSenseCameras:
    def __init__(
        self,
        *,
        camera_names: Sequence[str],
        camera_serials: Optional[Sequence[str]],
        width: int,
        height: int,
        fps: int,
        warmup_frames: int,
    ) -> None:
        import pyrealsense2 as rs

        self.rs = rs
        self.pipelines: Dict[str, Any] = {}
        self.camera_names = tuple(camera_names)
        serials = list(camera_serials) if camera_serials else self.connected_serials()
        if len(serials) != len(self.camera_names):
            raise RuntimeError(
                f"Expected {len(self.camera_names)} camera serials, got {len(serials)}: {serials}"
            )
        if camera_serials is None:
            print(
                "[warn] --camera-serials was not set; using sorted RealSense serials. "
                "Verify physical camera order before executing on hardware.",
                flush=True,
            )
        for name, serial in zip(self.camera_names, serials):
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            pipeline.start(config)
            self.pipelines[name] = pipeline
            print(f"camera {name}: serial={serial}", flush=True)
        for _ in range(max(0, int(warmup_frames))):
            self.read()

    def connected_serials(self) -> List[str]:
        context = self.rs.context()
        return sorted(
            dev.get_info(self.rs.camera_info.serial_number)
            for dev in context.query_devices()
        )

    def read(self) -> Dict[str, np.ndarray]:
        images: Dict[str, np.ndarray] = {}
        for name, pipeline in self.pipelines.items():
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                raise RuntimeError(f"Camera {name} returned no color frame")
            bgr = np.asanyarray(color_frame.get_data()).copy()
            images[name] = bgr[:, :, ::-1].copy()
        return images

    def close(self) -> None:
        for pipeline in self.pipelines.values():
            try:
                pipeline.stop()
            except Exception:
                pass


class ImageFileSource:
    def __init__(self, camera_names: Sequence[str], image_files: Sequence[str]) -> None:
        from PIL import Image

        if len(image_files) != len(camera_names):
            raise ValueError(
                f"--image-files must provide {len(camera_names)} files in --camera-names order"
            )
        self.images: Dict[str, np.ndarray] = {}
        for name, path in zip(camera_names, image_files):
            image = Image.open(path).convert("RGB")
            self.images[name] = np.asarray(image).copy()

    def read(self) -> Dict[str, np.ndarray]:
        return {name: image.copy() for name, image in self.images.items()}

    def close(self) -> None:
        return None


def ensure_openpi_client_path(path: str) -> None:
    if path and path not in sys.path:
        sys.path.insert(0, path)


class OpenPIClient:
    def __init__(
        self,
        *,
        host: str,
        port: Optional[int],
        api_key: Optional[str],
        client_src: str,
        resize_images: bool,
    ) -> None:
        ensure_openpi_client_path(client_src)
        from openpi_client import image_tools
        from openpi_client import websocket_client_policy

        self.image_tools = image_tools
        self.policy = websocket_client_policy.WebsocketClientPolicy(
            host=host,
            port=port,
            api_key=api_key,
        )
        self.resize_images = bool(resize_images)

    def metadata(self) -> Mapping[str, Any]:
        return self.policy.get_server_metadata()

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if self.resize_images:
            image = self.image_tools.resize_with_pad(image, 224, 224)
            image = self.image_tools.convert_to_uint8(image)
        return image

    def predict_action_chunk(
        self,
        *,
        external_image: np.ndarray,
        left_wrist_image: np.ndarray,
        right_wrist_image: np.ndarray | None = None,
        state: np.ndarray,
        prompt: str,
        action_dim: int,
    ) -> Tuple[np.ndarray, Mapping[str, Any]]:
        external = self._prepare_image(external_image)
        left_wrist = self._prepare_image(left_wrist_image)
        state = np.asarray(state, dtype=np.float32).reshape(action_dim)
        obs = {
            # Keys used by the local cup-pyramid training repack transform.
            "observation.images.external": external,
            "observation.images.left_wrist": left_wrist,
            "observation.state": state,
            # Keys consumed by the default OpenPI websocket inference transform.
            "observation/image": external,
            "observation/wrist_image": left_wrist,
            "observation/state": state,
            "prompt": str(prompt),
        }
        if right_wrist_image is not None:
            right_wrist = self._prepare_image(right_wrist_image)
            obs.update(
                {
                    "observation.images.right_wrist": right_wrist,
                    "observation/right_wrist_image": right_wrist,
                }
            )
        result = self.policy.infer(obs)
        if "actions" not in result:
            raise RuntimeError(f"OpenPI response missing actions: {result}")
        actions = np.asarray(result["actions"], dtype=np.float64)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.ndim != 2 or actions.shape[1] < action_dim:
            raise RuntimeError(f"Expected action chunk shape (T, >={action_dim}), got {actions.shape}")
        return actions[:, :action_dim].copy(), result


def choose_images(
    image_by_name: Mapping[str, np.ndarray],
    external_name: str,
    left_wrist_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    missing = [name for name in (external_name, left_wrist_name) if name not in image_by_name]
    if missing:
        raise KeyError(f"Missing camera images: {missing}")
    return image_by_name[external_name], image_by_name[left_wrist_name]


def validate_action_chunk(
    action_chunk: np.ndarray,
    action_dim: int,
    expected_horizon: Optional[int] = None,
) -> None:
    if action_chunk.ndim != 2 or action_chunk.shape[1] != action_dim:
        raise RuntimeError(f"Expected action chunk shape (T, {action_dim}), got {action_chunk.shape}")
    if expected_horizon is not None and expected_horizon > 0 and action_chunk.shape[0] != expected_horizon:
        raise RuntimeError(
            f"Expected action horizon {expected_horizon}, got {action_chunk.shape[0]}"
        )


def sample_action(action_chunk: np.ndarray, model_index: float) -> Tuple[np.ndarray, int, float]:
    max_index = len(action_chunk) - 1
    lo = int(np.floor(model_index))
    lo = int(np.clip(lo, 0, max_index))
    hi = min(lo + 1, max_index)
    alpha = 0.0 if hi == lo else float(np.clip(model_index - lo, 0.0, 1.0))
    if alpha <= 1e-9:
        return action_chunk[lo].copy(), lo, 0.0
    return ((1.0 - alpha) * action_chunk[lo] + alpha * action_chunk[hi]), lo, alpha


def maybe_apply_gripper_close_threshold(
    action_values: Sequence[float],
    *,
    enabled: bool,
    threshold: float,
) -> np.ndarray:
    action = np.asarray(action_values, dtype=np.float64).copy()
    if enabled and action[-1] <= float(threshold):
        action[-1] = 0.0
    return action


def target_from_absolute_quat_action(absolute_action8: Sequence[float]) -> ArmTarget:
    action = np.asarray(absolute_action8, dtype=np.float64).reshape(ABSOLUTE_QUAT_DIM)
    return ArmTarget(
        position=action[:3].copy(),
        quat_xyzw=wxyz_to_xyzw(action[3:7]),
        gripper_width=float(action[7]),
    )


def target_from_absolute_rotvec_action(absolute_action7: Sequence[float]) -> ArmTarget:
    action = np.asarray(absolute_action7, dtype=np.float64).reshape(ABSOLUTE_ROTVEC_DIM)
    target_rot = Rotation.from_rotvec(action[3:6])
    return ArmTarget(
        position=action[:3].copy(),
        quat_xyzw=normalize_quat_xyzw(target_rot.as_quat()),
        gripper_width=float(action[6]),
    )


def target_from_action(
    *,
    action_mode: str,
    action_values: Sequence[float],
) -> ArmTarget:
    if action_mode == "absolute-quat":
        return target_from_absolute_quat_action(action_values)
    if action_mode == "absolute-rotvec":
        return target_from_absolute_rotvec_action(action_values)
    raise ValueError(f"Unsupported action mode: {action_mode}")


def command_arm_to_target(
    *,
    arm: FrankaArm,
    target: ArmTarget,
    last_command: ArmCommandState,
    max_pos_step: float,
    max_rot_step: float,
    gripper_speed: float,
    gripper_force: float,
    gripper_max_open: float,
) -> ArmCommandState:
    limited_pos = limit_vector_step(last_command.position, target.position, max_pos_step)
    limited_quat_xyzw = limit_rotation_step(last_command.quat_xyzw, target.quat_xyzw, max_rot_step)
    arm.update_pose(limited_pos, limited_quat_xyzw)
    arm.goto_gripper(
        width_m=float(np.clip(target.gripper_width, 0.0, gripper_max_open)),
        speed=gripper_speed,
        force=gripper_force,
    )
    return ArmCommandState(position=limited_pos, quat_xyzw=limited_quat_xyzw)


def maybe_confirm_execute(args: argparse.Namespace) -> None:
    if not args.execute or args.yes:
        return
    print()
    print(f"About to start Cartesian impedance and send OpenPI {args.action_mode} actions.")
    print(f"left robot: {args.left_robot_host}:{args.left_robot_port}")
    print("execute left arm: yes")
    print("execute right arm: no (not connected by this script)")
    print(f"policy server: {args.policy_host}:{args.policy_port}")
    print(f"prompt: {args.prompt!r}")
    print(f"expected action horizon: {args.expected_action_horizon}")
    print(f"open-loop horizon: {args.open_loop_horizon}")
    print(f"reference pose source: unused for {args.action_mode}")
    print(
        "gripper close threshold: "
        f"{'enabled' if args.enable_gripper_close_threshold else 'disabled'} "
        f"(threshold={args.gripper_close_threshold:.4f} m)"
    )
    response = input("Type EXECUTE to continue: ").strip()
    if response != "EXECUTE":
        raise SystemExit("Aborted before sending robot commands.")


def print_action_summary(
    index: int,
    action_values: np.ndarray,
    action_mode: str,
    target: ArmTarget,
    inference_sec: float,
) -> None:
    if action_mode == "absolute-quat":
        action_text = (
            f"abs_xyz={np.array2string(action_values[:3], precision=4)} "
            f"abs_quat_wxyz={np.array2string(target.quat_wxyz, precision=4)} "
            f"grip={float(action_values[7]):.4f}"
        )
    else:
        action_text = (
            f"abs_xyz={np.array2string(action_values[:3], precision=4)} "
            f"abs_rotvec={np.array2string(action_values[3:6], precision=4)} "
            f"grip={float(action_values[6]):.4f}"
        )
    print(
        f"step={index:04d} "
        f"{action_text} "
        f"target_pos={np.array2string(target.position, precision=4)} "
        f"target_quat_wxyz={np.array2string(target.quat_wxyz, precision=4)} "
        f"inference_sec={inference_sec:.3f}",
        flush=True,
    )


def predict_csv_action_values(action_values: np.ndarray, action_mode: str) -> Dict[str, Any]:
    values: Dict[str, Any] = {f"action_{i}": "" for i in range(MAX_LOG_ACTION_DIM)}
    for i, value in enumerate(np.asarray(action_values, dtype=np.float64).ravel()[:MAX_LOG_ACTION_DIM]):
        values[f"action_{i}"] = float(value)
    rel_fields = {
        "rel_dx": "",
        "rel_dy": "",
        "rel_dz": "",
        "rel_drx": "",
        "rel_dry": "",
        "rel_drz": "",
        "gripper": float(action_values[-1]),
    }
    values.update(rel_fields)
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-host", default=DEFAULT_POLICY_HOST)
    parser.add_argument("--policy-port", type=int, default=DEFAULT_POLICY_PORT)
    parser.add_argument("--policy-api-key", default=None)
    parser.add_argument("--openpi-client-src", default=OPENPI_CLIENT_SRC)
    parser.add_argument("--prompt", "--task", dest="prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--resize-images", dest="resize_images", action="store_true", default=True)
    parser.add_argument("--no-resize-images", dest="resize_images", action="store_false")

    parser.add_argument("--left-robot-host", default=DEFAULT_LEFT_ROBOT_HOST)
    parser.add_argument("--left-robot-port", type=int, default=DEFAULT_LEFT_ROBOT_PORT)
    parser.add_argument("--left-gripper-host", default=None)
    parser.add_argument("--left-gripper-port", type=int, default=DEFAULT_LEFT_GRIPPER_PORT)
    parser.add_argument("--allow-missing-gripper", action="store_true")
    parser.add_argument("--gripper-max-open", type=float, default=DEFAULT_GRIPPER_MAX_OPEN)
    parser.add_argument("--gripper-speed", type=float, default=1.0)
    parser.add_argument("--gripper-force", type=float, default=1.0)
    parser.add_argument(
        "--enable-gripper-close-threshold",
        action="store_true",
        help="Force gripper targets <= --gripper-close-threshold to 0.0 after OpenPI unnormalization.",
    )
    parser.add_argument(
        "--gripper-close-threshold",
        type=float,
        default=0.02,
        help="Meters. Only used when --enable-gripper-close-threshold is set.",
    )

    parser.add_argument("--camera-names", nargs=2, default=DEFAULT_CAMERA_NAMES)
    parser.add_argument("--camera-serials", nargs=2, default=DEFAULT_CAMERA_SERIALS)
    parser.add_argument("--external-camera-name", default="external")
    parser.add_argument("--left-wrist-camera-name", default="left_wrist")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-warmup-frames", type=int, default=10)
    parser.add_argument(
        "--image-files",
        nargs=2,
        default=None,
        help="Optional RGB files in --camera-names order; useful for policy dry-runs without RealSense.",
    )

    parser.add_argument("--rollout-steps", type=int, default=300)
    parser.add_argument("--expected-action-horizon", type=int, default=DEFAULT_EXPECTED_ACTION_HORIZON)
    parser.add_argument("--open-loop-horizon", type=int, default=DEFAULT_OPEN_LOOP_HORIZON)
    parser.add_argument("--model-action-hz", type=float, default=DEFAULT_MODEL_ACTION_HZ)
    parser.add_argument("--control-hz", type=float, default=DEFAULT_CONTROL_HZ)
    parser.add_argument(
        "--action-mode",
        choices=("absolute-quat", "absolute-rotvec"),
        default=DEFAULT_ACTION_MODE,
        help=(
            "absolute-quat: policy state/action is xyz+quat_wxyz+gripper. "
            "absolute-rotvec: policy state/action is xyz+rotvec+gripper."
        ),
    )
    parser.add_argument(
        "--max-pos-step",
        type=float,
        default=0.005,
        help="Meters per control step; <=0 disables.",
    )
    parser.add_argument(
        "--max-rot-step",
        type=float,
        default=0.05,
        help="Radians per control step; <=0 disables.",
    )
    parser.add_argument("--execute", action="store_true", help="Actually command the robot. Default is dry-run.")
    parser.add_argument("--yes", action="store_true", help="Skip the EXECUTE confirmation prompt.")
    parser.add_argument("--leave-policy-running", action="store_true")
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--log-jsonl", default=None)
    parser.add_argument("--predict-csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rollout_steps < 1:
        raise ValueError("--rollout-steps must be >= 1")
    if args.expected_action_horizon < 1:
        raise ValueError("--expected-action-horizon must be >= 1")
    if args.open_loop_horizon < 1:
        raise ValueError("--open-loop-horizon must be >= 1")
    if args.open_loop_horizon > args.expected_action_horizon:
        raise ValueError("--open-loop-horizon cannot exceed --expected-action-horizon")
    if args.model_action_hz <= 0:
        raise ValueError("--model-action-hz must be positive")
    if args.control_hz < args.model_action_hz:
        raise ValueError("--control-hz should be >= --model-action-hz for interpolation")
    if args.gripper_close_threshold < 0:
        raise ValueError("--gripper-close-threshold must be >= 0")
    action_dim = action_dim_for_mode(args.action_mode)

    print(
        f"rollout: control_hz={args.control_hz}, model_action_hz={args.model_action_hz}, "
        f"action_mode={args.action_mode}, action_dim={action_dim}, "
        f"expected_action_horizon={args.expected_action_horizon}, "
        f"open_loop_horizon={args.open_loop_horizon}, "
        "reference_pose_source=unused, "
        f"gripper_close_threshold_enabled={args.enable_gripper_close_threshold}, "
        f"gripper_close_threshold={args.gripper_close_threshold}",
        flush=True,
    )

    print("Connecting to OpenPI websocket policy server...", flush=True)
    policy_client = OpenPIClient(
        host=args.policy_host,
        port=args.policy_port,
        api_key=args.policy_api_key,
        client_src=args.openpi_client_src,
        resize_images=args.resize_images,
    )
    print(f"policy_server_metadata={policy_client.metadata()}", flush=True)

    left_gripper_host = args.left_gripper_host or args.left_robot_host
    print("Connecting to left Franka interface...", flush=True)
    left_arm = FrankaArm(
        name="left",
        robot_host=args.left_robot_host,
        robot_port=args.left_robot_port,
        gripper_host=left_gripper_host,
        gripper_port=args.left_gripper_port,
        require_gripper=not args.allow_missing_gripper,
        gripper_max_open=args.gripper_max_open,
    )

    print("Starting image source...", flush=True)
    image_source: Any
    if args.image_files:
        image_source = ImageFileSource(args.camera_names, args.image_files)
    else:
        image_source = RealSenseCameras(
            camera_names=args.camera_names,
            camera_serials=args.camera_serials,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            warmup_frames=args.camera_warmup_frames,
        )

    log_handle = None
    predict_csv_handle = None
    predict_writer = None
    policy_started = False

    try:
        maybe_confirm_execute(args)
        if args.execute:
            left_arm.start_cartesian_impedance()
            policy_started = True

        left_last = left_arm.get_command_state()
        chunk_reference = left_last
        rate = Rate(args.control_hz)
        executed = 0
        action_chunk: Optional[np.ndarray] = None
        model_cursor = 0.0
        model_step_per_tick = float(args.model_action_hz) / float(args.control_hz)
        last_state: Optional[np.ndarray] = None
        last_infer_sec = 0.0
        last_policy_timing: Mapping[str, Any] = {}

        if args.log_jsonl:
            log_handle = open(args.log_jsonl, "a", encoding="utf-8")
        if args.predict_csv:
            predict_csv_path = Path(args.predict_csv)
            predict_csv_path.parent.mkdir(parents=True, exist_ok=True)
            predict_csv_handle = predict_csv_path.open("a", newline="", encoding="utf-8")
            predict_writer = csv.DictWriter(predict_csv_handle, fieldnames=PREDICT_CSV_FIELDS)
            if predict_csv_handle.tell() == 0:
                predict_writer.writeheader()
                predict_csv_handle.flush()

        while executed < args.rollout_steps:
            step_start = time.monotonic()
            need_replan = (
                action_chunk is None
                or model_cursor >= len(action_chunk) - 1e-9
                or model_cursor >= args.open_loop_horizon - 1e-9
            )
            if need_replan:
                chunk_reference = left_arm.get_command_state()
                gripper_width = left_arm.get_gripper_width()
                if args.action_mode == "absolute-quat":
                    last_state = np.concatenate(
                        [chunk_reference.position, chunk_reference.quat_wxyz, [gripper_width]]
                    ).astype(np.float32)
                else:
                    last_state = np.concatenate(
                        [
                            chunk_reference.position,
                            quat_xyzw_to_rotvec(chunk_reference.quat_xyzw),
                            [gripper_width],
                        ]
                    ).astype(np.float32)

                image_by_name = image_source.read()
                external_image, left_wrist_image = choose_images(
                    image_by_name,
                    args.external_camera_name,
                    args.left_wrist_camera_name,
                )
                infer_start = time.monotonic()
                with defer_keyboard_interrupt():
                    action_chunk, policy_result = policy_client.predict_action_chunk(
                        external_image=external_image,
                        left_wrist_image=left_wrist_image,
                        state=last_state,
                        prompt=args.prompt,
                        action_dim=action_dim,
                    )
                validate_action_chunk(action_chunk, action_dim, args.expected_action_horizon)
                last_infer_sec = time.monotonic() - infer_start
                last_policy_timing = dict(policy_result.get("policy_timing", {}))
                model_cursor = 0.0

            action_values, source_index, interp_alpha = sample_action(action_chunk, model_cursor)
            action_values = maybe_apply_gripper_close_threshold(
                action_values,
                enabled=args.enable_gripper_close_threshold,
                threshold=args.gripper_close_threshold,
            )
            model_cursor += model_step_per_tick

            target = target_from_action(
                action_mode=args.action_mode,
                action_values=action_values,
            )

            if args.print_every > 0 and executed % args.print_every == 0:
                print_action_summary(executed, action_values, args.action_mode, target, last_infer_sec)
                print(
                    f"model_cursor={model_cursor:.2f}/{args.open_loop_horizon} "
                    f"source_index={source_index} interp_alpha={interp_alpha:.2f} "
                    f"policy_timing={last_policy_timing}",
                    flush=True,
                )

            record = {
                "step": executed,
                "time": time.time(),
                "state": None if last_state is None else last_state.tolist(),
                "action_mode": args.action_mode,
                "policy_action": action_values.tolist(),
                "relative_action": None,
                "absolute_action": action_values.tolist(),
                "target_position": target.position.tolist(),
                "target_quat_wxyz": target.quat_wxyz.tolist(),
                "source_index": int(source_index),
                "interp_alpha": float(interp_alpha),
                "model_cursor_after": float(model_cursor),
                "model_step_per_tick": float(model_step_per_tick),
                "expected_action_horizon": int(args.expected_action_horizon),
                "open_loop_horizon": int(args.open_loop_horizon),
                "inference_sec": float(last_infer_sec),
                "policy_timing": dict(last_policy_timing),
                "execute": bool(args.execute),
                "reference_pose_source": "unused",
                "gripper_close_threshold_enabled": bool(args.enable_gripper_close_threshold),
                "gripper_close_threshold": float(args.gripper_close_threshold),
            }
            if log_handle is not None:
                log_handle.write(json.dumps(record) + "\n")
                log_handle.flush()
            if predict_writer is not None:
                quat_wxyz = target.quat_wxyz
                predict_writer.writerow(
                    {
                        "step": executed,
                        "time": time.time(),
                        "source_index": int(source_index),
                        "interp_alpha": float(interp_alpha),
                        "model_cursor_after": float(model_cursor),
                        "model_step_per_tick": float(model_step_per_tick),
                        "expected_action_horizon": int(args.expected_action_horizon),
                        "open_loop_horizon": int(args.open_loop_horizon),
                        "inference_sec": float(last_infer_sec),
                        "execute": bool(args.execute),
                        "action_mode": args.action_mode,
                        "reference_pose_source": "unused",
                        "gripper_close_threshold_enabled": bool(args.enable_gripper_close_threshold),
                        "gripper_close_threshold": float(args.gripper_close_threshold),
                        **predict_csv_action_values(action_values, args.action_mode),
                        "target_x": float(target.position[0]),
                        "target_y": float(target.position[1]),
                        "target_z": float(target.position[2]),
                        "target_qw": float(quat_wxyz[0]),
                        "target_qx": float(quat_wxyz[1]),
                        "target_qy": float(quat_wxyz[2]),
                        "target_qz": float(quat_wxyz[3]),
                    }
                )
                predict_csv_handle.flush()

            if args.execute:
                left_last = command_arm_to_target(
                    arm=left_arm,
                    target=target,
                    last_command=left_last,
                    max_pos_step=args.max_pos_step,
                    max_rot_step=args.max_rot_step,
                    gripper_speed=args.gripper_speed,
                    gripper_force=args.gripper_force,
                    gripper_max_open=args.gripper_max_open,
                )
            else:
                left_last = ArmCommandState(position=target.position, quat_xyzw=target.quat_xyzw)

            executed += 1
            elapsed = time.monotonic() - step_start
            if elapsed > rate.period:
                print(
                    f"[warn] step {executed} took {elapsed:.3f}s > {rate.period:.3f}s control period",
                    flush=True,
                )
            rate.sleep()

        if "rate" in locals() and rate.overruns:
            print(f"[warn] control loop overruns: {rate.overruns}", flush=True)
    finally:
        if log_handle is not None:
            log_handle.close()
        if predict_csv_handle is not None:
            predict_csv_handle.close()
        image_source.close()
        if policy_started and not args.leave_policy_running:
            left_arm.terminate_policy()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
