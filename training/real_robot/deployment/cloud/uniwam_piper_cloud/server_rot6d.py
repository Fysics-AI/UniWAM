"""Socket server for AgileX manip26 EEF-XY+visibility async-prefix12 inference and Piper IK."""

from __future__ import annotations

import argparse
import socket
import time
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from uniwam_piper_common.camera_frame import (
    DualArmCameraExtrinsics,
    action17_base_to_camera,
    action17_camera_to_base,
    state23_base_to_camera,
)
from uniwam_piper_common.rotation6d import (
    matrix_to_rpy_xyz,
    rotation_6d_to_matrix,
    rpy_xyz_to_matrix,
)

from .piper_ik import DualPiperIK
from .policy_rot6d import LiveUniWAMAsyncPrefix12Rot6DPolicy
from .protocol_rot6d import parse_observation, recv_message, send_message


def wrap_angle_delta(delta: np.ndarray) -> np.ndarray:
    return (np.asarray(delta) + np.pi) % (2.0 * np.pi) - np.pi


def canonical_eef_state(state_model: np.ndarray) -> np.ndarray:
    """Convert an exact 23D policy state into Piper IK's EEF14 RPY state."""
    state = np.asarray(state_model, dtype=np.float32).reshape(-1)
    if state.shape == (23,):
        result = np.empty(14, dtype=np.float32)
        result[0:3] = state[3:6]
        result[3:6] = matrix_to_rpy_xyz(rotation_6d_to_matrix(state[6:12]))
        result[6] = state[12]
        result[7:10] = state[13:16]
        result[10:13] = matrix_to_rpy_xyz(rotation_6d_to_matrix(state[16:22]))
        result[13] = state[22]
        return result
    if state.shape == (17,):
        return state[3:17].copy()
    if state.shape == (14,):
        return state.copy()
    raise ValueError(f"Unsupported state shape {state.shape}; expected 14, 17, or 23.")


def clamp_eef_action(action: np.ndarray, reference: np.ndarray, safety: Any) -> np.ndarray:
    """Optional operator-selected clamp; disabled by the shipped configuration."""
    out = np.asarray(action, dtype=np.float32).copy()
    reference = np.asarray(reference, dtype=np.float32).reshape(14)
    max_position = float(safety.get("max_eef_delta_m", 0.08))
    max_rotation = float(safety.get("max_eef_delta_rad", 0.45))
    gripper_min, gripper_max = safety.get("clamp_gripper_m", [0.0, 0.105])
    for start in (0, 7):
        out[start : start + 3] = reference[start : start + 3] + np.clip(
            out[start : start + 3] - reference[start : start + 3], -max_position, max_position
        )
        rotation = np.clip(
            wrap_angle_delta(out[start + 3 : start + 6] - reference[start + 3 : start + 6]),
            -max_rotation,
            max_rotation,
        )
        out[start + 3 : start + 6] = reference[start + 3 : start + 6] + rotation
        out[start + 6] = np.clip(out[start + 6], float(gripper_min), float(gripper_max))
    return out


def clamp_model_action_sequence(actions: np.ndarray, state_eef: np.ndarray, safety: Any) -> np.ndarray:
    """Optional explicit clamp for [EEF14, vx, vy, wz] actions."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 17:
        raise ValueError(f"Expected model actions [T,17], got {actions.shape}.")
    rolling = str(safety.get("eef_delta_reference", "rolling")).lower() == "rolling"
    reference = np.asarray(state_eef, dtype=np.float32).reshape(14).copy()
    result = actions.copy()
    for index in range(result.shape[0]):
        eef_reference = reference if rolling else state_eef
        result[index, :14] = clamp_eef_action(result[index, :14], eef_reference, safety)
        reference = result[index, :14]
    result[:, 14] = np.clip(
        result[:, 14],
        float(safety.get("min_base_vx_mps", -0.12)),
        float(safety.get("max_base_vx_mps", 0.12)),
    )
    result[:, 15] = np.clip(
        result[:, 15],
        -float(safety.get("max_base_vy_mps", 0.0)),
        float(safety.get("max_base_vy_mps", 0.0)),
    )
    result[:, 16] = np.clip(
        result[:, 16],
        -float(safety.get("max_base_wz_radps", 0.18)),
        float(safety.get("max_base_wz_radps", 0.18)),
    )
    return result.astype(np.float32)


def clamp_joint_delta(joints: np.ndarray, seed: np.ndarray | None, max_delta: float) -> np.ndarray:
    if seed is None or max_delta <= 0.0:
        return np.asarray(joints, dtype=np.float32)
    out = np.asarray(joints, dtype=np.float32).copy()
    previous = np.asarray(seed, dtype=np.float32).reshape(14)
    for row in out:
        for start in (0, 7):
            row[start : start + 6] = previous[start : start + 6] + np.clip(
                row[start : start + 6] - previous[start : start + 6], -max_delta, max_delta
            )
        previous = row
    return out


def repeat_hold_joint_actions(seed: np.ndarray | None, steps: int) -> np.ndarray:
    """Build a joint chunk that holds a known physical joint target exactly."""
    if seed is None:
        raise RuntimeError(
            "nav_only inference requires live joint feedback, a queued joint prefix, "
            "or a prior valid joint command to hold the arms."
        )
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}.")
    joints = np.asarray(seed, dtype=np.float32).reshape(14)
    if not np.all(np.isfinite(joints)):
        raise ValueError("Held joint seed contains NaN or Inf.")
    return np.repeat(joints.reshape(1, 14), int(steps), axis=0)


def _fmt_vector(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(value):+.4f}" for value in np.asarray(values).reshape(-1)) + "]"


def _fmt_actions(actions: np.ndarray, state_eef: np.ndarray | None = None) -> str:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 17 or not actions.shape[0]:
        return f"bad_shape={actions.shape}"
    pieces = [
        f"T={actions.shape[0]}",
        f"first_left={_fmt_vector(actions[0, :7])}",
        f"first_right={_fmt_vector(actions[0, 7:14])}",
        f"base_first={_fmt_vector(actions[0, 14:17])}",
        f"base_last={_fmt_vector(actions[-1, 14:17])}",
    ]
    if state_eef is not None:
        pieces.append(f"first_minus_state={_fmt_vector(actions[0, :14] - state_eef)}")
    return " ".join(pieces)


def _rotation_step_angles(rpy: np.ndarray) -> np.ndarray:
    matrices = rpy_xyz_to_matrix(np.asarray(rpy, dtype=np.float32))
    if matrices.shape[0] < 2:
        return np.empty(0, dtype=np.float32)
    relative = np.matmul(np.swapaxes(matrices[:-1], -1, -2), matrices[1:])
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    return np.arccos(cosine).astype(np.float32)


def _fmt_jump_diagnostics(
    eef_actions: np.ndarray,
    state_eef: np.ndarray,
    joints: np.ndarray,
    seed: np.ndarray | None,
    manip_actions: np.ndarray,
) -> str:
    eef = np.asarray(eef_actions, dtype=np.float32)
    state = np.asarray(state_eef, dtype=np.float32).reshape(1, 14)
    eef_with_state = np.concatenate((state, eef), axis=0)
    pieces: list[str] = []
    for name, start in (("L", 0), ("R", 7)):
        xyz_steps = np.linalg.norm(np.diff(eef_with_state[:, start : start + 3], axis=0), axis=1)
        rot_steps = _rotation_step_angles(eef_with_state[:, start + 3 : start + 6])
        xyz_index = int(np.argmax(xyz_steps))
        rot_index = int(np.argmax(rot_steps))
        pieces.extend(
            (
                f"{name}_xyz_first={xyz_steps[0]:.5f}",
                f"{name}_xyz_max={xyz_steps[xyz_index]:.5f}@{xyz_index}",
                f"{name}_rot_first={rot_steps[0]:.5f}",
                f"{name}_rot_max={rot_steps[rot_index]:.5f}@{rot_index}",
            )
        )

    joint_values = np.asarray(joints, dtype=np.float32)
    if seed is not None:
        joint_values = np.concatenate(
            (np.asarray(seed, dtype=np.float32).reshape(1, 14), joint_values), axis=0
        )
    if joint_values.shape[0] >= 2:
        for name, start in (("L", 0), ("R", 7)):
            joint_steps = np.max(np.abs(np.diff(joint_values[:, start : start + 6], axis=0)), axis=1)
            index = int(np.argmax(joint_steps))
            pieces.append(f"{name}_joint_max={joint_steps[index]:.5f}@{index}")

    manip = np.asarray(manip_actions, dtype=np.float32)
    for name, start in (("L", 3), ("R", 13)):
        rot6d = manip[:, start : start + 6]
        row1 = rot6d[:, :3]
        row2 = rot6d[:, 3:6]
        row1_norm = np.linalg.norm(row1, axis=1)
        row1_unit = row1 / np.maximum(row1_norm[:, None], 1.0e-12)
        orth_norm = np.linalg.norm(
            row2 - np.sum(row1_unit * row2, axis=1, keepdims=True) * row1_unit,
            axis=1,
        )
        pieces.extend(
            (
                f"{name}_rot6d_row1_min={np.min(row1_norm):.5f}",
                f"{name}_rot6d_orth_min={np.min(orth_norm):.5f}",
            )
        )
    pieces.append(
        f"gripper_range=[{np.min(eef[:, [6, 13]]):.4f},{np.max(eef[:, [6, 13]]):.4f}]"
    )
    return " ".join(pieces)


class UniWAMPiperAsyncPrefix12Rot6DServer:
    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.policy = LiveUniWAMAsyncPrefix12Rot6DPolicy(cfg)
        urdf_path = cfg.get("urdf_path", None)
        if urdf_path in (None, "", "null"):
            raise FileNotFoundError("Set urdf_path to a locally licensed Piper URDF.")
        self.ik = DualPiperIK(
            urdf_path=urdf_path,
            left_config=cfg.left_ik,
            right_config=cfg.right_ik,
            enable_collision=True,
            parallel_backend=str(cfg.get("ik_parallel_backend", "process")),
        )
        self.clamp_model_outputs = bool(cfg.safety.get("clamp_model_outputs", False))
        self.clamp_joint_outputs = bool(cfg.safety.get("clamp_joint_outputs", False))
        self.last_joints: np.ndarray | None = None
        print(
            "[uniwam-piper-cloud][ROT6D_SAFETY] "
            f"model_output_clamp={self.clamp_model_outputs} "
            f"joint_output_clamp={self.clamp_joint_outputs} "
            "IK URDF joint limits and self-collision checking remain intrinsic to the solver.",
            flush=True,
        )

    def close(self) -> None:
        self.policy.close()
        self.ik.close()

    def handle_observation(self, payload: dict[str, Any]) -> dict[str, Any]:
        observation = parse_observation(payload)
        robot_type = observation.robot_type
        if robot_type not in {"piper", "franka"}:
            raise ValueError(f"robot_type must be piper or franka, got {robot_type!r}")
        franka_passthrough = robot_type == "franka"
        if observation.camera_from_left_base is None or observation.camera_from_right_base is None:
            raise ValueError("camera-frame inference requires explicit dual-arm camera extrinsics")
        extrinsics = DualArmCameraExtrinsics.from_values(
            profile=observation.extrinsics_profile or "explicit",
            camera_from_left_base=observation.camera_from_left_base,
            camera_from_right_base=observation.camera_from_right_base,
            source="edge_explicit",
        )
        prefix_eef = observation.prefix_eef_actions
        prefix_joint = observation.prefix_joint_actions
        prefix_bbox = observation.prefix_bbox_actions
        prefix_nav_aux = observation.prefix_nav_aux_actions
        prefix_length = 0 if prefix_eef is None else int(prefix_eef.shape[0])
        if prefix_length:
            if prefix_eef.ndim != 2 or prefix_eef.shape[1] != 17:
                raise ValueError(f"prefix_eef_actions must be [L,17], got {prefix_eef.shape}.")
            if not franka_passthrough and (prefix_joint is None or prefix_joint.shape != (prefix_length, 14)):
                raise ValueError(
                    "A head EEF prefix requires aligned prefix_joint_actions [L,14]; "
                    f"got {None if prefix_joint is None else prefix_joint.shape}."
                )
            if prefix_bbox is None or prefix_bbox.shape != (prefix_length, 6):
                raise ValueError(
                    "manip26 deployment requires aligned prefix EEF-XY+visibility [L,6]; "
                    f"got {None if prefix_bbox is None else prefix_bbox.shape}."
                )
            if prefix_nav_aux is not None and prefix_nav_aux.shape != (prefix_length, 0):
                raise ValueError(
                    "A head prefix requires aligned prefix_nav_aux_actions [L,0]; "
                    f"got {prefix_nav_aux.shape}."
                )
        elif prefix_joint is not None and prefix_joint.size:
            raise ValueError("prefix_joint_actions were provided without an EEF prefix.")
        elif prefix_bbox is not None and prefix_bbox.size:
            raise ValueError("prefix_bbox_actions were provided without an EEF prefix.")
        elif prefix_nav_aux is not None and prefix_nav_aux.size:
            raise ValueError("prefix_nav_aux_actions were provided without an EEF prefix.")

        started = time.perf_counter()
        base_state = self.policy.observation_builder.canonical_state(observation.eef_state)
        camera_state = state23_base_to_camera(base_state, extrinsics)
        prefix_eef_base = prefix_eef
        prefix_eef = (
            None if prefix_eef_base is None
            else action17_base_to_camera(prefix_eef_base, extrinsics)
        )
        state_eef = canonical_eef_state(base_state)
        sample = self.policy.make_sample(
            images=observation.images,
            image_encoding=observation.image_encoding,
            eef_state=camera_state,
            instruction=payload.get("instruction", None),
            inference_mode=observation.inference_mode,
            robot_type=robot_type,
        )
        sample_done = time.perf_counter()
        policy_output = self.policy.request_suffix(
            sample,
            prefix_eef_actions=prefix_eef,
            prefix_bbox_actions=prefix_bbox,
            prefix_nav_aux_actions=prefix_nav_aux,
            inference_mode=observation.inference_mode,
        )
        policy_done = time.perf_counter()

        camera_model_actions = np.asarray(policy_output["model_actions"], dtype=np.float32)
        if camera_model_actions.ndim != 2 or camera_model_actions.shape[1] != 17:
            raise RuntimeError(f"Bad camera-frame model action shape: {camera_model_actions.shape}")
        model_actions = action17_camera_to_base(camera_model_actions, extrinsics)
        if int(policy_output.get("prefix_length", -1)) != prefix_length:
            raise RuntimeError(
                "Policy prefix length does not match the request: "
                f"{policy_output.get('prefix_length')} vs {prefix_length}."
            )
        if bool(self.cfg.get("debug_eef", False)):
            print("[uniwam-piper-cloud][MODEL_RAW_ACTION] " + _fmt_actions(model_actions, state_eef), flush=True)

        arm_hold = bool(policy_output.get("arm_hold", False))
        safe_actions = (
            clamp_model_action_sequence(model_actions, state_eef, self.cfg.safety)
            if self.clamp_model_outputs
            else model_actions.copy()
        )
        seed = (
            prefix_joint[-1]
            if prefix_length and prefix_joint is not None
            else observation.joint_state
            if observation.joint_state is not None
            else self.last_joints
        )
        ik_started = time.perf_counter()
        if franka_passthrough:
            joints = np.zeros((safe_actions.shape[0], 14), dtype=np.float32)
            ik_infos = [{"eef_passthrough": True, "ik_skipped": True} for _ in range(safe_actions.shape[0])]
            ik_done = ik_started
        elif arm_hold:
            joints = repeat_hold_joint_actions(seed, safe_actions.shape[0])
            ik_infos = [
                {"arm_hold": True, "ik_skipped": True}
                for _ in range(safe_actions.shape[0])
            ]
            ik_done = ik_started
        else:
            joints, ik_infos = self.ik.solve_chunk(
                safe_actions[:, :14],
                require_success=bool(self.cfg.safety.get("require_successful_ik", True)),
                seed_joints=seed,
            )
            ik_done = time.perf_counter()
            if self.clamp_joint_outputs:
                joints = clamp_joint_delta(
                    joints,
                    seed,
                    float(self.cfg.safety.get("max_joint_delta_rad", 0.35)),
                )
        if joints.shape[0]:
            self.last_joints = joints[-1].copy()

        elapsed = time.perf_counter() - started
        model_infer_s = float(policy_output.get("request_model_infer_s", 0.0))
        latency = {
            "observation_build_s": float(sample_done - started),
            "policy_total_s": float(policy_done - sample_done),
            "model_infer_s": model_infer_s,
            "policy_postprocess_s": float(max(0.0, policy_done - sample_done - model_infer_s)),
            "pre_ik_s": float(ik_started - policy_done),
            "ik_s": float(ik_done - ik_started),
            "post_ik_s": float(max(0.0, elapsed - (ik_done - started))),
            "total_s": float(elapsed),
        }
        if bool(self.cfg.get("debug_latency", False)):
            print(
                "[uniwam-piper-cloud][LATENCY] "
                + " ".join(f"{key}={value:.4f}" for key, value in latency.items()),
                flush=True,
            )
        if bool(self.cfg.get("debug_eef", False)):
            print("[uniwam-piper-cloud][IK_INPUT_ACTION] " + _fmt_actions(safe_actions, state_eef), flush=True)
        if bool(self.cfg.get("debug_action_deltas", False)):
            normalized_state = np.asarray(sample["proprio"].detach().cpu(), dtype=np.float32)
            print(
                "[uniwam-piper-cloud][ACTION_DELTAS] "
                f"state_norm_abs_max={float(np.max(np.abs(normalized_state))):.4f} "
                f"state_norm_oob={int(np.count_nonzero(np.abs(normalized_state) > 1.0))} "
                + _fmt_jump_diagnostics(
                    safe_actions[:, :14],
                    state_eef,
                    joints,
                    seed,
                    np.asarray(policy_output["manip_actions"], dtype=np.float32),
                ),
                flush=True,
            )

        return {
            "type": "action_chunk",
            "request_id": observation.request_id,
            "ok": True,
            "control_hz": float(self.cfg.control_hz),
            "joint_actions": joints.astype(np.float32),
            "eef_actions": safe_actions[:, :14].astype(np.float32),
            "model_actions": safe_actions.astype(np.float32),
            "base_actions": safe_actions[:, 14:17].astype(np.float32),
            "nav_actions": np.asarray(policy_output["nav_actions"], dtype=np.float32),
            "nav_aux_actions": np.asarray(policy_output["nav_aux_actions"], dtype=np.float32),
            "manip_actions": np.asarray(policy_output["manip_actions"], dtype=np.float32),
            "bbox_actions": np.asarray(policy_output["bbox_actions"], dtype=np.float32),
            "eef_xy_actions": np.asarray(policy_output["bbox_actions"], dtype=np.float32),
            "policy_step_count": int(policy_output["step_count"]),
            "prefix_length": prefix_length,
            "prefix_mode": "head",
            "inference_mode": str(policy_output["inference_mode"]),
            "arm_hold": arm_hold,
            "model_horizon": int(policy_output.get("model_horizon", model_actions.shape[0])),
            "execute_horizon": int(policy_output.get("execute_horizon", model_actions.shape[0])),
            "output_clamp_config": {
                "model_outputs": self.clamp_model_outputs,
                "joint_outputs": self.clamp_joint_outputs,
            },
            "server_latency_s": float(elapsed),
            "request_model_infer_s": model_infer_s,
            "model_profile_ms": policy_output.get("model_profile_ms", {}),
            "latency_breakdown_s": latency,
            "policy_stats": policy_output["stats"],
            "ik": ik_infos,
            "camera_extrinsics": {
                "profile": extrinsics.profile,
                "source": extrinsics.source,
            },
            "actuation_backend": "eef_passthrough" if franka_passthrough else "piper_ik",
        }

    def serve_forever(self) -> None:
        host = str(self.cfg.listen_host)
        port = int(self.cfg.listen_port)
        timeout_s = float(self.cfg.socket_timeout_s)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((host, port))
            server_socket.listen(1)
            print(f"[uniwam-piper-cloud] listening on {host}:{port}", flush=True)
            while True:
                connection, address = server_socket.accept()
                with connection:
                    connection.settimeout(timeout_s)
                    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    connection.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                    connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2 * 1024 * 1024)
                    connection.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
                    print(f"[uniwam-piper-cloud] client connected: {address}", flush=True)
                    while True:
                        try:
                            request = recv_message(connection)
                            if request.get("type") == "ping":
                                send_message(connection, {"type": "pong", "time": time.time()})
                                continue
                            response = self.handle_observation(request)
                        except EOFError:
                            print("[uniwam-piper-cloud] client disconnected", flush=True)
                            break
                        except Exception as exc:
                            response = {"type": "error", "ok": False, "error": f"{type(exc).__name__}: {exc}"}
                        send_message(connection, response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    server = UniWAMPiperAsyncPrefix12Rot6DServer(cfg)
    try:
        server.serve_forever()
    finally:
        server.close()


if __name__ == "__main__":
    main()
