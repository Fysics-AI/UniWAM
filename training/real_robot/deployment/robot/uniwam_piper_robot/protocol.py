from __future__ import annotations

import io
import socket
import struct
from typing import Any

import msgpack
import numpy as np


HEADER = struct.Struct("!I")
MAX_FRAME_BYTES = 256 * 1024 * 1024


def _encode_obj(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {
            "__ndarray__": True,
            "payload": buf.getvalue(),
        }
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Cannot msgpack-encode object of type {type(obj)!r}")


def _decode_obj(obj: Any) -> Any:
    if isinstance(obj, dict) and obj.get("__ndarray__"):
        return np.load(io.BytesIO(obj["payload"]), allow_pickle=False)
    return obj


def recv_exact(sock: socket.socket, size: int) -> bytes:
    out: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("socket closed while receiving frame")
        out.append(chunk)
        remaining -= len(chunk)
    return b"".join(out)


def recv_message(sock: socket.socket) -> dict[str, Any]:
    header = recv_exact(sock, HEADER.size)
    (size,) = HEADER.unpack(header)
    if size <= 0 or size > MAX_FRAME_BYTES:
        raise RuntimeError(f"Invalid frame size: {size}")
    body = recv_exact(sock, size)
    return msgpack.unpackb(body, raw=False, object_hook=_decode_obj)


def send_message(sock: socket.socket, payload: dict[str, Any]) -> None:
    body = msgpack.packb(payload, default=_encode_obj, use_bin_type=True)
    sock.sendall(HEADER.pack(len(body)) + body)
