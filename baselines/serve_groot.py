#!/usr/bin/env python3
"""Serve the MAP-Bench GR00T N1.7 navigation checkpoint."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from io import BytesIO
import hashlib
import logging
import math
import os
import sys
from pathlib import Path
import time
from typing import Any, Iterator
from unittest import mock

import numpy as np
from PIL import Image
import torch

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
# The benchmark WebSocket interface is bundled with this repository.
CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "runtime/openpi_server_src"))
sys.path.insert(0, str(CODE_ROOT / "runtime/openpi_client_src"))
from openpi.serving import websocket_policy_server
from openpi_client import base_policy as _base_policy


SOURCE_WIDTH = 566
SOURCE_HEIGHT = 320
SOURCE_RATE_HZ = 15.0
EXECUTION_HORIZON = 32
MODEL_ACTION_HORIZON = 40
VELOCITY_ORDER = ("vx_mps", "vyaw_radps")


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
            "MAP-Bench evaluation requires native RGB shape "
            f"{(SOURCE_HEIGHT, SOURCE_WIDTH, 3)}, got {rgb.shape}"
        )
    return np.ascontiguousarray(rgb)


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


@contextmanager
def seeded_action_noise(
    *,
    seeds: list[int],
    action_horizon: int,
    action_dim: int,
) -> Iterator[None]:
    """Replace only GR00T's initial action-noise draw with per-item seeded draws."""
    original_randn = torch.randn
    replacement_count = 0
    expected_shape = (len(seeds), action_horizon, action_dim)

    def deterministic_randn(*args: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal replacement_count
        size = kwargs.get("size")
        if size is None:
            size = args[0] if len(args) == 1 and isinstance(args[0], (tuple, list)) else args
        shape = tuple(int(value) for value in size)
        if shape != expected_shape:
            return original_randn(*args, **kwargs)
        if replacement_count:
            raise RuntimeError("GR00T requested its initial action noise more than once")
        replacement_count += 1
        device = torch.device(kwargs.get("device", "cpu"))
        dtype = kwargs.get("dtype")
        return torch.cat(
            [
                original_randn(
                    (1, action_horizon, action_dim),
                    dtype=dtype,
                    device=device,
                    generator=torch.Generator(device=device).manual_seed(int(seed)),
                )
                for seed in seeds
            ],
            dim=0,
        )

    with mock.patch.object(torch, "randn", deterministic_randn):
        yield
    if replacement_count != 1:
        raise RuntimeError(
            "Did not intercept GR00T's initial action-noise draw; deterministic "
            "per-episode seeding is not active"
        )


class GrootNavigationPolicy(_base_policy.BasePolicy):
    def __init__(self, args: argparse.Namespace) -> None:
        self.model_path = Path(args.model_path).resolve()
        self.embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)
        self.device = str(args.device)
        self.execution_horizon = int(args.execution_horizon)
        self.rate_hz = float(args.rate_hz)
        if self.execution_horizon != EXECUTION_HORIZON:
            raise ValueError(
                f"MAP-Bench evaluation requires H{EXECUTION_HORIZON}, "
                f"got H{self.execution_horizon}"
            )
        # Resolve the bundled backbone from the checkpoint directory during loading.
        # Restore the caller's directory before serving any requests.
        previous_directory = Path.cwd()
        try:
            os.chdir(self.model_path)
            self.policy = Gr00tPolicy(
                embodiment_tag=self.embodiment_tag,
                model_path=str(self.model_path), device=self.device, strict=True,
            )
        finally:
            os.chdir(previous_directory)
        modality = self.policy.modality_configs
        expected = {
            "video": ("ego_view",),
            "state": ("navigation_state",),
            "action": ("navigate_command",),
            "language": ("task",),
        }
        for name, keys in expected.items():
            actual = tuple(modality[name].modality_keys)
            if actual != keys:
                raise ValueError(f"Unexpected {name} keys: expected={keys}, got={actual}")
        model_horizon = len(modality["action"].delta_indices)
        if model_horizon != MODEL_ACTION_HORIZON:
            raise ValueError(
                f"Expected trained H{MODEL_ACTION_HORIZON} action output, got H{model_horizon}"
            )
        self.model_action_horizon = model_horizon
        self.action_dim = int(self.policy.model.action_head.action_dim)
        self.video_key = expected["video"][0]
        self.state_key = expected["state"][0]
        self.action_key = expected["action"][0]
        self.language_key = self.policy.language_key
        index_path = self.model_path / "model.safetensors.index.json"
        processor_path = self.model_path / "processor_config.json"
        stats_path = self.model_path / "statistics.json"
        self._metadata = {
            "policy_name": "groot_n17_map_bench",
            "checkpoint": "groot_n17",
            "checkpoint_index_sha256": sha256_file(index_path),
            "processor_config_sha256": sha256_file(processor_path),
            "statistics_sha256": sha256_file(stats_path),
            "embodiment_tag": self.embodiment_tag.value,
            "preprocess_profile": "checkpoint_gr00t_processor_224x224",
            "preprocess_owner": "policy_service_only",
            "source_rgb_shape": [SOURCE_HEIGHT, SOURCE_WIDTH, 3],
            "source_rate_hz": self.rate_hz,
            "model_action_horizon": self.model_action_horizon,
            "execution_action_horizon": self.execution_horizon,
            "raw_action_names": VELOCITY_ORDER,
            "raw_action_units": ["m/s", "rad/s"],
            "state_semantics": "previous_executed_velocity_command",
            "integration": "midpoint_heading_unicycle_dt_1_over_15",
            "action_reference": "each [x,y,heading] row is cumulative from request pose",
            "supports_batch": True,
            "batch_request": {"batch": "list[observation]"},
            "batch_response": {"batch": "list[reply]"},
            "seed_semantics": "one diffusion noise seed per episode call, independent of batch position",
        }
        logging.info("Loaded GR00T checkpoint %s on %s", self.model_path, self.device)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def infer_batch(self, observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not observations:
            raise ValueError("batch must contain at least one observation")
        started = time.monotonic()
        images = [extract_rgb(item) for item in observations]
        tasks = [str(item.get("task", "")).strip() for item in observations]
        if any(not task for task in tasks):
            raise ValueError("Every request must contain a non-empty benchmark task")
        navigation_states = np.stack(
            [
                np.asarray(item.get("navigation_state", np.zeros(2)), dtype=np.float32)
                for item in observations
            ],
            axis=0,
        )
        if navigation_states.shape != (len(observations), 2) or not np.all(
            np.isfinite(navigation_states)
        ):
            raise ValueError(
                "navigation_state must be one finite [vx,vyaw] vector per request, "
                f"got {navigation_states.shape}"
            )
        seeds = [int(item.get("diffusion_seed", 20260807)) for item in observations]
        model_observation = {
            "video": {
                self.video_key: np.stack(images, axis=0)[:, None, ...],
            },
            "state": {
                self.state_key: navigation_states[:, None, :],
            },
            "language": {
                self.language_key: [[task] for task in tasks],
            },
        }
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            torch.cuda.synchronize(torch.device(self.device))
        model_started = time.monotonic()
        with seeded_action_noise(
            seeds=seeds,
            action_horizon=self.model_action_horizon,
            action_dim=self.action_dim,
        ):
            action, _ = self.policy.get_action(model_observation)
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            torch.cuda.synchronize(torch.device(self.device))
        batch_model_s = float(time.monotonic() - model_started)
        velocity = np.asarray(action[self.action_key], dtype=np.float32)
        expected_shape = (len(observations), self.model_action_horizon, 2)
        if velocity.shape != expected_shape or not np.all(np.isfinite(velocity)):
            raise ValueError(
                f"Invalid decoded {self.action_key} output: "
                f"expected={expected_shape}, got={velocity.shape}"
            )
        replies: list[dict[str, Any]] = []
        for index, (commands, task, seed) in enumerate(zip(velocity, tasks, seeds)):
            executed = commands[: self.execution_horizon].copy()
            relative_se2 = velocity_to_relative_se2(executed, rate_hz=self.rate_hz)
            replies.append(
                {
                    "relative_se2_actions": relative_se2,
                    "raw_velocity_actions": executed,
                    "navigation_state_after": executed[-1].copy(),
                    "task": task,
                    "diffusion_seed": seed,
                    "batch_index": index,
                    "policy_timing": {
                        "batch_size": len(observations),
                        "batch_model_s": batch_model_s,
                        "per_sample_model_s": batch_model_s / len(observations),
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
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--execution-horizon", type=int, default=EXECUTION_HORIZON)
    parser.add_argument("--rate-hz", type=float, default=SOURCE_RATE_HZ)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = GrootNavigationPolicy(args)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
