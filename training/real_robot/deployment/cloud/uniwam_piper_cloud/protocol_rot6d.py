"""Length-prefixed msgpack protocol for nav3/manip26 EEF-XY+visibility inference."""

from __future__ import annotations

import io
import socket
import struct
from dataclasses import dataclass
from typing import Any

import msgpack
import numpy as np


HEADER = struct.Struct("!I")
MAX_FRAME_BYTES = 256 * 1024 * 1024
REQUIRED_IMAGE_KEYS = {
    "cam_nav",
    "cam_manip_high",
    "cam_left_wrist",
    "cam_right_wrist",
}
INFERENCE_MODES = {"paired", "manip_only", "nav_only"}


class ProtocolError(RuntimeError):
    pass


@dataclass
class ObservationPacket:
    request_id: int
    stamp: float
    eef_state: np.ndarray
    images: dict[str, bytes]
    image_encoding: str
    joint_state: np.ndarray | None
    prefix_eef_actions: np.ndarray | None
    prefix_joint_actions: np.ndarray | None
    prefix_bbox_actions: np.ndarray | None
    prefix_nav_aux_actions: np.ndarray | None
    prefix_mode: str
    # None retains compatibility with pre-update edge clients. The cloud then
    # falls back to its configured default.
    inference_mode: str | None
    extrinsics_profile: str
    camera_from_left_base: np.ndarray | None
    camera_from_right_base: np.ndarray | None
    robot_type: str


def _encode_obj(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        buffer = io.BytesIO()
        np.save(buffer, obj, allow_pickle=False)
        return {"__ndarray__": True, "payload": buffer.getvalue()}
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Cannot msgpack-encode object of type {type(obj)!r}")


def _decode_obj(obj: Any) -> Any:
    if isinstance(obj, dict) and obj.get("__ndarray__"):
        return np.load(io.BytesIO(obj["payload"]), allow_pickle=False)
    return obj


def pack_message(payload: dict[str, Any]) -> bytes:
    body = msgpack.packb(payload, default=_encode_obj, use_bin_type=True)
    return HEADER.pack(len(body)) + body


def recv_exact(sock: socket.socket, size: int) -> bytes:
    parts: list[bytes] = []
    remaining = int(size)
    while remaining:
        part = sock.recv(remaining)
        if not part:
            raise EOFError("socket closed while receiving frame")
        parts.append(part)
        remaining -= len(part)
    return b"".join(parts)


def recv_message(sock: socket.socket) -> dict[str, Any]:
    header = recv_exact(sock, HEADER.size)
    (size,) = HEADER.unpack(header)
    if size <= 0 or size > MAX_FRAME_BYTES:
        raise ProtocolError(f"Invalid frame size: {size}")
    payload = msgpack.unpackb(recv_exact(sock, size), raw=False, object_hook=_decode_obj)
    if not isinstance(payload, dict):
        raise ProtocolError(f"Expected message dict, got {type(payload)!r}")
    return payload


def send_message(sock: socket.socket, payload: dict[str, Any]) -> None:
    sock.sendall(pack_message(payload))


def _finite_array(name: str, value: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if shape is not None and tuple(array.shape) != shape:
        raise ProtocolError(f"`{name}` must have shape {shape}, got {tuple(array.shape)}")
    if not np.all(np.isfinite(array)):
        raise ProtocolError(f"`{name}` contains NaN or Inf")
    return array


def _optional_se3(name: str, value: Any) -> np.ndarray | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ProtocolError(f"`{name}` must be a finite 4x4 matrix, got {matrix.shape}")
    return matrix


def parse_observation(payload: dict[str, Any]) -> ObservationPacket:
    if payload.get("type") != "observation":
        raise ProtocolError(f"Expected observation packet, got {payload.get('type')!r}")

    state = _finite_array("eef_state", payload.get("eef_state"))
    # 14D/17D forms are deliberately retained only for replay tools. The live
    # rot6D ROS client always sends the exact 23D train-time state.
    if tuple(state.shape) not in {(14,), (17,), (23,)}:
        raise ProtocolError(
            "`eef_state` must be eef14, [base3,eef14]17, or rot6D23; "
            f"got {tuple(state.shape)}"
        )

    images = payload.get("images")
    if not isinstance(images, dict):
        raise ProtocolError("`images` must be a dict")
    missing = REQUIRED_IMAGE_KEYS - set(images)
    if missing:
        raise ProtocolError(f"Missing image keys: {sorted(missing)}")
    image_bytes: dict[str, bytes] = {}
    for key in REQUIRED_IMAGE_KEYS:
        try:
            image_bytes[key] = bytes(images[key])
        except Exception as exc:
            raise ProtocolError(f"Image payload {key!r} is not bytes-like") from exc
        if not image_bytes[key]:
            raise ProtocolError(f"Image payload {key!r} is empty")

    joint_state = payload.get("joint_state")
    if joint_state is not None:
        joint_state = _finite_array("joint_state", joint_state, (14,))

    prefix_mode = str(payload.get("prefix_mode", "head")).strip().lower()
    if prefix_mode != "head":
        raise ProtocolError(f"`prefix_mode` must be `head`, got {prefix_mode!r}")

    raw_inference_mode = payload.get("inference_mode", None)
    if raw_inference_mode is None or str(raw_inference_mode).strip() == "":
        inference_mode = None
    else:
        inference_mode = str(raw_inference_mode).strip().lower()
        if inference_mode not in INFERENCE_MODES:
            raise ProtocolError(
                "`inference_mode` must be one of "
                f"{sorted(INFERENCE_MODES)}, got {inference_mode!r}"
            )

    prefix_joint = payload.get("prefix_joint_actions")
    if prefix_joint is not None:
        prefix_joint = _finite_array("prefix_joint_actions", prefix_joint)
        if prefix_joint.ndim != 2 or prefix_joint.shape[1] != 14:
            raise ProtocolError(
                f"`prefix_joint_actions` must have shape [T,14], got {tuple(prefix_joint.shape)}"
            )

    prefix_bbox = payload.get("prefix_bbox_actions")
    if prefix_bbox is not None:
        prefix_bbox = _finite_array("prefix_bbox_actions", prefix_bbox)
        if prefix_bbox.ndim != 2 or prefix_bbox.shape[1] != 6:
            raise ProtocolError(
                f"`prefix_bbox_actions` must have shape [T,6], got {tuple(prefix_bbox.shape)}"
            )

    prefix_eef = payload.get("prefix_eef_actions")
    if prefix_eef is not None:
        prefix_eef = _finite_array("prefix_eef_actions", prefix_eef)
        if prefix_eef.ndim != 2 or prefix_eef.shape[1] != 17:
            raise ProtocolError(
                "`prefix_eef_actions` must have shape [T,17] as EEF14-RPY plus base3, "
                f"got {tuple(prefix_eef.shape)}"
            )

    prefix_nav_aux = payload.get("prefix_nav_aux_actions")
    if prefix_nav_aux is not None:
        prefix_nav_aux = _finite_array("prefix_nav_aux_actions", prefix_nav_aux)
        if prefix_nav_aux.ndim != 2 or prefix_nav_aux.shape[1] != 0:
            raise ProtocolError(
                "`prefix_nav_aux_actions` must have shape [T,0] for this nav3 checkpoint, "
                f"got {tuple(prefix_nav_aux.shape)}"
            )
        if prefix_eef is None:
            raise ProtocolError("`prefix_nav_aux_actions` requires `prefix_eef_actions`.")
        if prefix_nav_aux.shape[0] != prefix_eef.shape[0]:
            raise ProtocolError(
                "`prefix_nav_aux_actions` must be time-aligned with `prefix_eef_actions`: "
                f"got {prefix_nav_aux.shape[0]} versus {prefix_eef.shape[0]}"
            )

    left_extrinsic = _optional_se3("camera_from_left_base", payload.get("camera_from_left_base"))
    right_extrinsic = _optional_se3("camera_from_right_base", payload.get("camera_from_right_base"))
    if (left_extrinsic is None) != (right_extrinsic is None):
        raise ProtocolError("Both camera extrinsics must be provided together.")

    return ObservationPacket(
        request_id=int(payload.get("request_id", 0)),
        stamp=float(payload.get("stamp", 0.0)),
        eef_state=state,
        images=image_bytes,
        image_encoding=str(payload.get("image_encoding", "jpeg")),
        joint_state=joint_state,
        prefix_eef_actions=prefix_eef,
        prefix_joint_actions=prefix_joint,
        prefix_bbox_actions=prefix_bbox,
        prefix_nav_aux_actions=prefix_nav_aux,
        prefix_mode=prefix_mode,
        inference_mode=inference_mode,
        extrinsics_profile=str(payload.get("extrinsics_profile", "explicit")).strip(),
        camera_from_left_base=left_extrinsic,
        camera_from_right_base=right_extrinsic,
        robot_type=str(payload.get("robot_type", "piper")).strip().lower(),
    )
