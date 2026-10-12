#!/usr/bin/env python3
"""Serve the MAP-Bench π0.5 navigation checkpoint."""

from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
import logging
import math
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from openpi.models import model as model_lib
from pi05_model import create_navigation_policy
from openpi.serving import websocket_policy_server
from openpi_client import base_policy


SOURCE_WIDTH = 566
SOURCE_HEIGHT = 320
SOURCE_RATE_HZ = 15.0
MODEL_ACTION_HORIZON = 50
EXECUTION_HORIZON = 32
MODEL_ACTION_DIM = 32
VX_SLOT = 14
VYAW_SLOT = 16


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_rgb(observation: dict[str, Any]) -> np.ndarray:
    jpeg = observation.get("image_jpeg")
    if jpeg is not None:
        if not isinstance(jpeg, (bytes, bytearray, memoryview)):
            raise ValueError("image_jpeg must contain JPEG bytes")
        with Image.open(BytesIO(jpeg)) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        value = observation.get("image")
        if value is None:
            raise ValueError("Each request must contain image or image_jpeg")
        rgb = np.asarray(value)
        if rgb.ndim != 3:
            raise ValueError(f"image must be HWC or CHW, got {rgb.shape}")
        if rgb.shape[0] == 3 and rgb.shape[-1] != 3:
            rgb = np.moveaxis(rgb, 0, -1)
        if rgb.shape[-1] != 3:
            raise ValueError(f"image must have three RGB channels, got {rgb.shape}")
        if np.issubdtype(rgb.dtype, np.floating):
            rgb = rgb * (255.0 if float(np.nanmax(rgb)) <= 1.01 else 1.0)
        rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    if rgb.shape != (SOURCE_HEIGHT, SOURCE_WIDTH, 3):
        raise ValueError(
            f"Expected native HM3D RGB {(SOURCE_HEIGHT, SOURCE_WIDTH, 3)}, got {rgb.shape}"
        )
    return np.ascontiguousarray(rgb)


def unnormalize_nav_slots(
    normalized_actions: np.ndarray,
    q01: np.ndarray,
    q99: np.ndarray,
) -> np.ndarray:
    normalized = np.asarray(normalized_actions, dtype=np.float32)
    if normalized.shape[-1] != 2:
        raise ValueError(f"Expected normalized [vx,vyaw], got {normalized.shape}")
    return (normalized + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


def velocity_to_relative_se2(
    velocity_actions: np.ndarray,
    *,
    rate_hz: float = SOURCE_RATE_HZ,
) -> np.ndarray:
    """Integrate [vx, vyaw] with midpoint heading into cumulative body-frame SE(2)."""
    velocity = np.asarray(velocity_actions, dtype=np.float32)
    if velocity.ndim != 2 or velocity.shape[1] != 2 or not np.all(np.isfinite(velocity)):
        raise ValueError(f"velocity_actions must be finite [H,2], got {velocity.shape}")
    if rate_hz <= 0:
        raise ValueError(f"rate_hz must be positive, got {rate_hz}")
    dt = 1.0 / float(rate_hz)
    output = np.empty((velocity.shape[0], 3), dtype=np.float32)
    x_body = 0.0
    y_body = 0.0
    heading = 0.0
    for index, (vx_mps, vyaw_radps) in enumerate(velocity):
        delta_heading = float(vyaw_radps) * dt
        midpoint_heading = heading + 0.5 * delta_heading
        distance = float(vx_mps) * dt
        x_body += distance * math.cos(midpoint_heading)
        y_body += distance * math.sin(midpoint_heading)
        heading = (heading + delta_heading + math.pi) % (2.0 * math.pi) - math.pi
        output[index] = (x_body, y_body, heading)
    return output


class Pi05NavigationPolicy(base_policy.BasePolicy):
    def __init__(self, args: argparse.Namespace) -> None:
        self.checkpoint = Path(args.checkpoint).resolve()
        self.fixed_batch_size = int(args.fixed_batch_size)
        self.rate_hz = float(args.rate_hz)
        if self.fixed_batch_size <= 0:
            raise ValueError("fixed_batch_size must be positive")

        self.policy = create_navigation_policy(self.checkpoint)
        stats_path = self.checkpoint / "normalization.json"
        stats = json.loads(stats_path.read_text(encoding="utf-8"))["norm_stats"]
        self.q01 = np.asarray(stats["actions"]["q01"], dtype=np.float32)
        self.q99 = np.asarray(stats["actions"]["q99"], dtype=np.float32)
        if self.q01.shape != (2,) or self.q99.shape != (2,):
            raise ValueError(f"Expected 2D action norm stats, got {self.q01.shape}/{self.q99.shape}")

        params_metadata = self.checkpoint / "params" / "_METADATA"
        self._metadata = {
            "policy_name": "pi05_map_bench",
            "checkpoint": "pi05",
            "params_metadata_sha256": sha256_file(params_metadata),
            "norm_stats_sha256": sha256_file(stats_path),
            "preprocess_profile": "openpi_hm3d_oneview_resize_with_pad_224x224",
            "preprocess_owner": "policy_service_only",
            "source_rgb_shape": [SOURCE_HEIGHT, SOURCE_WIDTH, 3],
            "source_rate_hz": self.rate_hz,
            "model_action_horizon": MODEL_ACTION_HORIZON,
            "execution_action_horizon": EXECUTION_HORIZON,
            "model_action_slots": {str(VX_SLOT): "vx_mps", str(VYAW_SLOT): "vyaw_radps"},
            "action_unnormalization": "checkpoint 2D q01/q99 after selecting slots 14/16",
            "integration": "midpoint_heading_unicycle_dt_1_over_15",
            "action_reference": "each [x,y,heading] row is cumulative from request pose",
            "fixed_policy_batch_size": self.fixed_batch_size,
            "seed_semantics": "NumPy PCG64 diffusion noise per episode call; padding cannot perturb real items",
        }
        logging.info("Loaded OpenPI checkpoint %s", self.checkpoint)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def _transform_observation(self, item: dict[str, Any]) -> dict[str, Any]:
        task = str(item.get("task", "")).strip()
        if not task:
            raise ValueError("Every request must contain a non-empty benchmark task")
        state = np.asarray(item.get("navigation_state", np.zeros(2)), dtype=np.float32)
        if state.shape != (2,) or not np.all(np.isfinite(state)):
            raise ValueError(f"navigation_state must be one finite [vx,vyaw] vector, got {state.shape}")
        raw = {
            "images": {"cam_high": extract_rgb(item)},
            "state": state,
            "prompt": task,
        }
        return self.policy._input_transform(raw)

    def infer_batch(self, observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not observations:
            raise ValueError("batch must contain at least one observation")
        if len(observations) > self.fixed_batch_size:
            raise ValueError(
                f"Request batch {len(observations)} exceeds fixed batch {self.fixed_batch_size}"
            )
        started = time.monotonic()
        real_count = len(observations)
        transformed = [self._transform_observation(item) for item in observations]
        transformed.extend([transformed[-1]] * (self.fixed_batch_size - real_count))
        inputs = jax.tree.map(lambda *values: jnp.asarray(np.stack(values)), *transformed)

        seeds = [int(item.get("diffusion_seed", 20260807)) for item in observations]
        seeds.extend([0] * (self.fixed_batch_size - real_count))
        noise = np.stack(
            [
                np.random.default_rng(seed).standard_normal(
                    (MODEL_ACTION_HORIZON, MODEL_ACTION_DIM), dtype=np.float32
                )
                for seed in seeds
            ]
        )
        model_observation = model_lib.Observation.from_dict(inputs)
        model_started = time.monotonic()
        model_actions = self.policy._sample_actions(
            jax.random.key(0), model_observation, noise=jnp.asarray(noise)
        )
        model_actions.block_until_ready()
        batch_model_s = float(time.monotonic() - model_started)
        model_actions = np.asarray(model_actions[:real_count], dtype=np.float32)
        expected = (real_count, MODEL_ACTION_HORIZON, MODEL_ACTION_DIM)
        if model_actions.shape != expected or not np.all(np.isfinite(model_actions)):
            raise ValueError(f"Invalid model action output: expected={expected}, got={model_actions.shape}")

        normalized_nav = model_actions[:, :EXECUTION_HORIZON, [VX_SLOT, VYAW_SLOT]]
        velocity = unnormalize_nav_slots(normalized_nav, self.q01, self.q99)
        if not np.all(np.isfinite(velocity)):
            raise ValueError("Unnormalized velocity output contains non-finite values")
        replies = []
        for index, (commands, seed) in enumerate(zip(velocity, seeds[:real_count])):
            replies.append(
                {
                    "relative_se2_actions": velocity_to_relative_se2(commands, rate_hz=self.rate_hz),
                    "raw_velocity_actions": commands,
                    "navigation_state_after": commands[-1].copy(),
                    "diffusion_seed": seed,
                    "batch_index": index,
                    "policy_timing": {
                        "request_batch_size": real_count,
                        "fixed_model_batch_size": self.fixed_batch_size,
                        "batch_model_s": batch_model_s,
                        "per_real_sample_model_s": batch_model_s / real_count,
                        "infer_s": float(time.monotonic() - started),
                    },
                }
            )
        return replies

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        batch = observation.get("batch") if isinstance(observation, dict) else None
        if batch is not None:
            if not isinstance(batch, list) or any(not isinstance(item, dict) for item in batch):
                raise ValueError("batch must be a list of observation dictionaries")
            return {"batch": self.infer_batch(batch)}
        return self.infer_batch([observation])[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--fixed-batch-size", type=int, default=32)
    parser.add_argument("--rate-hz", type=float, default=SOURCE_RATE_HZ)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = Pi05NavigationPolicy(args)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
