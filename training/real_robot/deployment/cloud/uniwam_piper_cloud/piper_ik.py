from __future__ import annotations

import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import casadi
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin


def wrap_angle(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def matrix_to_rpy(rotation: np.ndarray) -> np.ndarray:
    sy = math.sqrt(rotation[0, 0] * rotation[0, 0] + rotation[1, 0] * rotation[1, 0])
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float64)


def pose_vec_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (6,):
        raise ValueError(f"Expected pose shape (6,), got {pose.shape}")
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rpy_to_matrix(float(pose[3]), float(pose[4]), float(pose[5]))
    out[:3, 3] = pose[:3]
    return out


def matrix_to_pose_vec(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    out = np.zeros(6, dtype=np.float64)
    out[:3] = matrix[:3, 3]
    out[3:6] = matrix_to_rpy(matrix[:3, :3])
    return out


def transform_pose(target_in_world: np.ndarray, base_in_world: np.ndarray) -> np.ndarray:
    return np.linalg.inv(base_in_world) @ target_in_world


def config_pose_to_matrix(config: Any | None) -> np.ndarray:
    if config is None:
        return np.eye(4, dtype=np.float64)
    xyz = np.asarray(config.get("xyz", [0.0, 0.0, 0.0]), dtype=np.float64)
    rpy = np.asarray(config.get("rpy", [0.0, 0.0, 0.0]), dtype=np.float64)
    return pose_vec_to_matrix(np.concatenate([xyz, rpy], axis=0))


@dataclass
class IKResult:
    q: np.ndarray | None
    success: bool
    collision_free: bool
    error: str = ""


class PiperArmIK:
    """Pinocchio/CasADi IK adapted from agilexrobotics/piper_ros.

    The official script lives under:
    `src/piper/scripts/piper_pinocchio/piper_pinocchio.py`.
    This class keeps the same reduced-robot setup and optimization objective,
    but removes ROS and Meshcat so it can run inside the cloud inference server.
    """

    def __init__(
        self,
        *,
        urdf_path: str | Path,
        enable_collision: bool = True,
        max_iter: int = 50,
        tol: float = 1e-4,
        max_jump_reset_rad: float = math.radians(30.0),
    ) -> None:
        self.urdf_path = str(urdf_path)
        self.enable_collision = bool(enable_collision)
        self.max_jump_reset_rad = float(max_jump_reset_rad)

        # IK only needs the kinematic tree. RobotWrapper.BuildFromURDF also
        # tries to load package:// meshes, which are not required on the cloud.
        self.model = pin.buildModelFromUrdf(self.urdf_path)
        if not self.model.existJointName("joint6"):
            raise ValueError("URDF does not contain joint6")
        locked_joint_ids = [
            self.model.getJointId(name)
            for name in ("joint7", "joint8")
            if self.model.existJointName(name)
        ]
        self.reduced_model = pin.buildReducedModel(
            self.model,
            locked_joint_ids,
            np.zeros(self.model.nq, dtype=np.float64),
        )
        self.data = self.model.createData()
        self.reduced_data = self.reduced_model.createData()
        if self.reduced_model.getFrameId("ee") >= self.reduced_model.nframes:
            self.reduced_model.addFrame(
                pin.Frame(
                    "ee",
                    self.reduced_model.getJointId("joint6"),
                    pin.SE3(np.eye(3), np.zeros(3)),
                    pin.FrameType.OP_FRAME,
                )
            )

        self.geom_model = None
        self.geometry_data = None
        if self.enable_collision:
            try:
                self.geom_model = pin.buildGeomFromUrdf(self.model, self.urdf_path, pin.GeometryType.COLLISION)
                max_geom = len(self.geom_model.geometryObjects)
                for i in range(4, min(10, max_geom)):
                    for j in range(0, min(3, max_geom)):
                        self.geom_model.addCollisionPair(pin.CollisionPair(i, j))
                self.geometry_data = pin.GeometryData(self.geom_model)
            except Exception:
                self.geom_model = None
                self.geometry_data = None

        self.init_q = np.zeros(self.reduced_model.nq, dtype=np.float64)
        self.last_q = np.zeros(self.reduced_model.nq, dtype=np.float64)

        self.cmodel = cpin.Model(self.reduced_model)
        self.cdata = self.cmodel.createData()
        self.cq = casadi.SX.sym("q", self.reduced_model.nq, 1)
        self.cTf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)
        self.ee_frame_id = self.reduced_model.getFrameId("ee")
        self.error = casadi.Function(
            "error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    cpin.log6(self.cdata.oMf[self.ee_frame_id].inverse() * cpin.SE3(self.cTf)).vector
                )
            ],
        )

        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_model.nq)
        self.param_tf = self.opti.parameter(4, 4)
        self.param_q_ref = self.opti.parameter(self.reduced_model.nq)
        error_vec = self.error(self.var_q, self.param_tf)
        pos_error = error_vec[:3]
        ori_error = error_vec[3:]
        total_cost = casadi.sumsqr(pos_error) + casadi.sumsqr(0.1 * ori_error)
        regularization = casadi.sumsqr(self.var_q)
        smooth_cost = casadi.sumsqr(self.var_q - self.param_q_ref)
        self.opti.subject_to(
            self.opti.bounded(
                self.reduced_model.lowerPositionLimit,
                self.var_q,
                self.reduced_model.upperPositionLimit,
            )
        )
        self.opti.minimize(20.0 * total_cost + 0.01 * regularization + 0.05 * smooth_cost)
        self.opti.solver(
            "ipopt",
            {
                "ipopt": {
                    "print_level": 0,
                    "max_iter": int(max_iter),
                    "tol": float(tol),
                },
                "print_time": False,
            },
        )

    def solve_pose_matrix(
        self,
        target_pose: np.ndarray,
        gripper: float = 0.0,
        seed_q: np.ndarray | None = None,
    ) -> IKResult:
        target_pose = np.asarray(target_pose, dtype=np.float64)
        if target_pose.shape != (4, 4):
            raise ValueError(f"Expected target_pose shape (4,4), got {target_pose.shape}")
        try:
            if seed_q is None:
                seed = self.init_q
            else:
                seed = np.asarray(seed_q, dtype=np.float64).reshape(-1)[: self.reduced_model.nq]
                seed = np.clip(seed, self.reduced_model.lowerPositionLimit, self.reduced_model.upperPositionLimit)
            self.opti.set_initial(self.var_q, seed)
            self.opti.set_value(self.param_tf, target_pose)
            self.opti.set_value(self.param_q_ref, seed)
            sol = self.opti.solve_limited()
            q = np.asarray(self.opti.value(self.var_q), dtype=np.float64).reshape(-1)
            jump = float(np.max(np.abs(q - self.last_q))) if self.last_q is not None else 0.0
            self.last_q = q.copy()
            self.init_q = q.copy() if jump <= self.max_jump_reset_rad else np.zeros_like(q)
            collision_free = not self.check_self_collision(q, gripper)
            return IKResult(q=q, success=True, collision_free=collision_free)
        except Exception as exc:
            return IKResult(q=None, success=False, collision_free=False, error=str(exc))

    def solve_pose_vec(self, pose: np.ndarray, gripper: float = 0.0, seed_q: np.ndarray | None = None) -> IKResult:
        return self.solve_pose_matrix(pose_vec_to_matrix(pose), gripper=gripper, seed_q=seed_q)

    def check_self_collision(self, q: np.ndarray, gripper: float = 0.0) -> bool:
        if self.geom_model is None or self.geometry_data is None:
            return False
        gripper_pair = np.array([gripper / 2.0, -gripper / 2.0], dtype=np.float64)
        pin.forwardKinematics(self.model, self.data, np.concatenate([q, gripper_pair], axis=0))
        pin.updateGeometryPlacements(self.model, self.data, self.geom_model, self.geometry_data)
        return bool(pin.computeCollisions(self.geom_model, self.geometry_data, False))


_PROCESS_ARM: PiperArmIK | None = None


def _init_process_arm(urdf_path: str, enable_collision: bool) -> None:
    global _PROCESS_ARM
    _PROCESS_ARM = PiperArmIK(urdf_path=urdf_path, enable_collision=enable_collision)


def _process_worker_ready(delay_s: float = 0.1) -> int:
    if _PROCESS_ARM is None:
        raise RuntimeError("Piper IK worker was not initialized.")
    time.sleep(float(delay_s))
    return os.getpid()


def _solve_process_arm(payload: tuple) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if _PROCESS_ARM is None:
        raise RuntimeError("Piper IK worker was not initialized.")
    poses, grippers, base_in_world, initial_seed, require_success, label = payload
    poses = np.asarray(poses, dtype=np.float64)
    grippers = np.asarray(grippers, dtype=np.float64)
    base_in_world = np.asarray(base_in_world, dtype=np.float64)
    current_seed = None if initial_seed is None else np.asarray(initial_seed, dtype=np.float64)
    joints = []
    infos = []
    for pose, gripper in zip(poses, grippers):
        target_world = pose_vec_to_matrix(pose)
        target_base = transform_pose(target_world, base_in_world)
        result = _PROCESS_ARM.solve_pose_matrix(
            target_base,
            gripper=float(gripper),
            seed_q=current_seed,
        )
        info = {
            f"{label}_success": result.success,
            f"{label}_collision_free": result.collision_free,
            f"{label}_error": result.error,
        }
        if require_success and (not result.success or not result.collision_free):
            raise RuntimeError(
                f"{label.capitalize()} IK failed: success={result.success} "
                f"collision_free={result.collision_free} {result.error}"
            )
        if result.q is None:
            if current_seed is None:
                raise RuntimeError(f"{label.capitalize()} IK returned no solution and no fallback seed.")
            joint = current_seed.copy()
        else:
            joint = np.asarray(result.q, dtype=np.float64).copy()
        joints.append(joint)
        infos.append(info)
        current_seed = joint
    return np.stack(joints, axis=0), infos


class DualPiperIK:
    def __init__(
        self,
        *,
        urdf_path: str | Path,
        left_config: Any,
        right_config: Any,
        enable_collision: bool = True,
        parallel_backend: str = "process",
    ) -> None:
        self.left_enabled = bool(left_config.get("enabled", True))
        self.right_enabled = bool(right_config.get("enabled", True))
        self.left_base_in_world = config_pose_to_matrix(left_config.get("base_transform", None))
        self.right_base_in_world = config_pose_to_matrix(right_config.get("base_transform", None))
        self.urdf_path = str(urdf_path)
        self.enable_collision = bool(enable_collision)
        self.parallel_backend = str(parallel_backend).strip().lower()
        if self.parallel_backend not in {"process", "thread", "sequential"}:
            raise ValueError("parallel_backend must be process, thread, or sequential.")
        use_processes = self.parallel_backend == "process" and self.left_enabled and self.right_enabled
        self.left = (
            None
            if use_processes or not self.left_enabled
            else PiperArmIK(urdf_path=urdf_path, enable_collision=enable_collision)
        )
        self.right = (
            None
            if use_processes or not self.right_enabled
            else PiperArmIK(urdf_path=urdf_path, enable_collision=enable_collision)
        )
        self._process_executor: ProcessPoolExecutor | None = None
        self._thread_executor: ThreadPoolExecutor | None = None
        if use_processes:
            self._process_executor = ProcessPoolExecutor(
                max_workers=2,
                mp_context=mp.get_context("spawn"),
                initializer=_init_process_arm,
                initargs=(self.urdf_path, self.enable_collision),
            )
            # Force both expensive Pinocchio/IPOPT initializers to finish before
            # the server starts listening, so request 1 has no worker cold start.
            ready = [self._process_executor.submit(_process_worker_ready, 0.2) for _ in range(2)]
            worker_pids = {future.result() for future in ready}
            if len(worker_pids) != 2:
                raise RuntimeError(f"Expected two distinct IK worker processes, got {worker_pids}.")
        elif self.parallel_backend == "thread" and self.left is not None and self.right is not None:
            self._thread_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="piper-ik")

    def close(self) -> None:
        if self._thread_executor is not None:
            self._thread_executor.shutdown(wait=True, cancel_futures=True)
            self._thread_executor = None
        if self._process_executor is not None:
            self._process_executor.shutdown(wait=True, cancel_futures=True)
            self._process_executor = None

    def solve_eef_action(
        self,
        action: np.ndarray,
        require_success: bool = True,
        seed_joints: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (14,):
            raise ValueError(f"Expected action shape (14,), got {action.shape}")
        if self._process_executor is not None:
            joints, infos = self.solve_chunk(
                action.reshape(1, 14),
                require_success=require_success,
                seed_joints=seed_joints,
            )
            return joints[0], infos[0]
        out = np.zeros(14, dtype=np.float32)
        info: dict[str, Any] = {}
        seed = None if seed_joints is None else np.asarray(seed_joints, dtype=np.float64).reshape(14)

        if self.left is not None:
            left_target_world = pose_vec_to_matrix(action[0:6])
            left_target_base = transform_pose(left_target_world, self.left_base_in_world)
            left_seed = None if seed is None else seed[0:6]
            res = self.left.solve_pose_matrix(left_target_base, gripper=float(action[6]), seed_q=left_seed)
            info["left_success"] = res.success
            info["left_collision_free"] = res.collision_free
            info["left_error"] = res.error
            if require_success and (not res.success or not res.collision_free):
                raise RuntimeError(f"Left IK failed: success={res.success} collision_free={res.collision_free} {res.error}")
            if res.q is not None:
                out[0:6] = res.q.astype(np.float32)
            out[6] = float(action[6])

        if self.right is not None:
            right_target_world = pose_vec_to_matrix(action[7:13])
            right_target_base = transform_pose(right_target_world, self.right_base_in_world)
            right_seed = None if seed is None else seed[7:13]
            res = self.right.solve_pose_matrix(right_target_base, gripper=float(action[13]), seed_q=right_seed)
            info["right_success"] = res.success
            info["right_collision_free"] = res.collision_free
            info["right_error"] = res.error
            if require_success and (not res.success or not res.collision_free):
                raise RuntimeError(f"Right IK failed: success={res.success} collision_free={res.collision_free} {res.error}")
            if res.q is not None:
                out[7:13] = res.q.astype(np.float32)
            out[13] = float(action[13])

        return out, info

    def solve_chunk(
        self,
        actions: np.ndarray,
        require_success: bool = True,
        seed_joints: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise ValueError(f"Expected actions shape [T,14], got {actions.shape}")
        seed = None if seed_joints is None else np.asarray(seed_joints, dtype=np.float64).reshape(14)

        if self._process_executor is not None:
            left_future = self._process_executor.submit(
                _solve_process_arm,
                (
                    actions[:, 0:6],
                    actions[:, 6],
                    self.left_base_in_world,
                    None if seed is None else seed[0:6],
                    require_success,
                    "left",
                ),
            )
            right_future = self._process_executor.submit(
                _solve_process_arm,
                (
                    actions[:, 7:13],
                    actions[:, 13],
                    self.right_base_in_world,
                    None if seed is None else seed[7:13],
                    require_success,
                    "right",
                ),
            )
            left_result = left_future.result()
            right_result = right_future.result()
            output = np.zeros((actions.shape[0], 14), dtype=np.float32)
            output[:, 0:6] = left_result[0].astype(np.float32)
            output[:, 6] = actions[:, 6].astype(np.float32)
            output[:, 7:13] = right_result[0].astype(np.float32)
            output[:, 13] = actions[:, 13].astype(np.float32)
            infos = [
                {**left_info, **right_info}
                for left_info, right_info in zip(left_result[1], right_result[1])
            ]
            return output, infos

        def solve_arm_sequence(
            arm: PiperArmIK,
            pose_slice: slice,
            gripper_index: int,
            base_in_world: np.ndarray,
            initial_seed: np.ndarray | None,
            label: str,
        ) -> tuple[np.ndarray, list[dict[str, Any]]]:
            arm_joints = []
            arm_infos = []
            current_seed = initial_seed
            for row in actions:
                target_world = pose_vec_to_matrix(row[pose_slice])
                target_base = transform_pose(target_world, base_in_world)
                result = arm.solve_pose_matrix(
                    target_base,
                    gripper=float(row[gripper_index]),
                    seed_q=current_seed,
                )
                info = {
                    f"{label}_success": result.success,
                    f"{label}_collision_free": result.collision_free,
                    f"{label}_error": result.error,
                }
                if require_success and (not result.success or not result.collision_free):
                    raise RuntimeError(
                        f"{label.capitalize()} IK failed: success={result.success} "
                        f"collision_free={result.collision_free} {result.error}"
                    )
                if result.q is None:
                    if current_seed is None:
                        raise RuntimeError(f"{label.capitalize()} IK returned no solution and no fallback seed.")
                    joint = np.asarray(current_seed, dtype=np.float64).copy()
                else:
                    joint = np.asarray(result.q, dtype=np.float64).copy()
                arm_joints.append(joint)
                arm_infos.append(info)
                current_seed = joint
            return np.stack(arm_joints, axis=0), arm_infos

        left_seed = None if seed is None else seed[0:6]
        right_seed = None if seed is None else seed[7:13]
        left_result = None
        right_result = None
        if self._thread_executor is not None:
            assert self.left is not None and self.right is not None
            left_future = self._thread_executor.submit(
                solve_arm_sequence,
                self.left,
                slice(0, 6),
                6,
                self.left_base_in_world,
                left_seed,
                "left",
            )
            right_future = self._thread_executor.submit(
                solve_arm_sequence,
                self.right,
                slice(7, 13),
                13,
                self.right_base_in_world,
                right_seed,
                "right",
            )
            left_result = left_future.result()
            right_result = right_future.result()
        else:
            if self.left is not None:
                left_result = solve_arm_sequence(
                    self.left, slice(0, 6), 6, self.left_base_in_world, left_seed, "left"
                )
            if self.right is not None:
                right_result = solve_arm_sequence(
                    self.right, slice(7, 13), 13, self.right_base_in_world, right_seed, "right"
                )

        output = np.zeros((actions.shape[0], 14), dtype=np.float32)
        output[:, 6] = actions[:, 6].astype(np.float32)
        output[:, 13] = actions[:, 13].astype(np.float32)
        infos: list[dict[str, Any]] = [dict() for _ in range(actions.shape[0])]
        if left_result is not None:
            output[:, 0:6] = left_result[0].astype(np.float32)
            for index, info in enumerate(left_result[1]):
                infos[index].update(info)
        if right_result is not None:
            output[:, 7:13] = right_result[0].astype(np.float32)
            for index, info in enumerate(right_result[1]):
                infos[index].update(info)
        return output, infos
