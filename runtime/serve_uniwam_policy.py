#!/usr/bin/env python3
"""Serve the UniWAM navigation policy."""

from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf


SNAPSHOT_ROOT = Path(__file__).resolve().parent
for path in (
    SNAPSHOT_ROOT / "uniwam_src",
    SNAPSHOT_ROOT / "openpi_server_src",
    SNAPSHOT_ROOT / "openpi_client_src",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

PROMPT_TEMPLATE = "A video recorded from a robot's point of view executing the following instruction: {task}"
from fastwam.utils.config_resolvers import register_default_resolvers

try:
    from openpi.serving import websocket_policy_server
    from openpi_client import base_policy as _base_policy
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This service needs the OpenPI WebSocket modules.") from exc


ACTION_ORDER = (
    "x_body_m", "y_body_m", "heading_rad", "bbox_x1", "bbox_y1",
    "bbox_x2", "bbox_y2", "navigable_x", "navigable_y",
)
PREPROCESS_PROFILES = {
    "base": {
        "source_wh": (566, 320),
        "resize_wh": (453, 256),
        "canvas_wh": (480, 256),
        "pad_ltrb": (5, 0, 22, 0),
    },
    "scaled": {
        "source_wh": (566, 320),
        "resize_wh": (453, 256),
        "canvas_wh": (480, 256),
        "pad_ltrb": (13, 0, 14, 0),
    },
}


def model_dtype(value: str) -> torch.dtype:
    values = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}
    try:
        return values[str(value).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported mixed precision: {value}") from exc


def extract_image(obs: dict[str, Any]) -> np.ndarray:
    jpeg = obs.get("image_jpeg")
    if jpeg is not None:
        if not isinstance(jpeg, (bytes, bytearray, memoryview)):
            raise ValueError("image_jpeg must be JPEG bytes")
        with Image.open(BytesIO(jpeg)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    value = obs.get("image")
    if value is None:
        raise ValueError("Send image or image_jpeg")
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"image must be HWC or CHW, got {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError(f"image must have three channels, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = image * (255.0 if float(np.nanmax(image)) <= 1.01 else 1.0)
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def preprocess_image(
    rgb: np.ndarray,
    profile: dict[str, tuple[int, ...]],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    source_wh = tuple(profile["source_wh"])
    if rgb.shape[:2] != (source_wh[1], source_wh[0]):
        raise ValueError(
            f"MAP-Bench evaluation requires source RGB {source_wh}, got "
            f"{(rgb.shape[1], rgb.shape[0])}"
        )
    resize_w, resize_h = profile["resize_wh"]
    canvas_w, canvas_h = profile["canvas_wh"]
    left, top, right, bottom = profile["pad_ltrb"]
    if resize_w + left + right != canvas_w or resize_h + top + bottom != canvas_h:
        raise ValueError("Invalid resize/padding profile")
    frame = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).float()
    resized = F.interpolate(
        frame, size=(resize_h, resize_w), mode="bilinear",
        align_corners=False, antialias=True,
    )
    canvas = torch.zeros((1, 3, canvas_h, canvas_w), dtype=torch.float32)
    canvas[:, :, top : top + resize_h, left : left + resize_w] = resized
    return canvas.mul_(2.0 / 255.0).sub_(1.0).to(device=device, dtype=dtype)


class UniWAMPolicy(_base_policy.BasePolicy):
    def __init__(self, args: argparse.Namespace) -> None:
        register_default_resolvers()
        self.cfg = OmegaConf.load(args.config_path)
        self.cfg.model.load_text_encoder = False
        self.model = instantiate(
            self.cfg.model,
            model_dtype=model_dtype(self.cfg.mixed_precision),
            device=args.device,
        )
        self.model.load_checkpoint(str(args.checkpoint_path))
        self.model = self.model.to(args.device).eval()
        self.device = self.model.device
        self.action_horizon = 32
        self.action_dim = 9
        self.profile_name = str(args.preprocess_profile)
        self.profile = PREPROCESS_PROFILES[self.profile_name]
        self.num_inference_steps = int(args.num_inference_steps)
        self.seed = int(args.seed)
        self.rand_device = str(args.rand_device)
        self.default_task = str(args.default_task)

        stats = json.loads(Path(args.stats_path).read_text(encoding="utf-8"))
        if tuple(stats.get("action_order", ())) != ACTION_ORDER:
            raise ValueError(f"Invalid action order in {args.stats_path}")
        self.q01 = np.asarray(stats["q01"], dtype=np.float32)
        self.q99 = np.asarray(stats["q99"], dtype=np.float32)
        if self.q01.shape != (9,) or self.q99.shape != (9,) or not np.all(self.q99 > self.q01):
            raise ValueError(f"Invalid action q01/q99 in {args.stats_path}")
        self.scale = 2.0 / (self.q99 - self.q01)
        self.offset = -1.0 - self.scale * self.q01

        self.cache_dirs = tuple(args.text_cache_dir)
        self.context_len = 128
        self.context_cache: dict[str, tuple[torch.Tensor, torch.Tensor, Path]] = {}
        self.checkpoint_path = Path(args.checkpoint_path)
        self.stats_path = Path(args.stats_path)
        logging.info(
            "Loaded checkpoint=%s preprocess=%s q01=%s q99=%s",
            self.checkpoint_path, self.profile_name,
            self.q01.tolist(), self.q99.tolist(),
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "policy_name": "UniWAM",
            "checkpoint": self.checkpoint_path.name,
            "stats": self.stats_path.name,
            "preprocess_profile": self.profile_name,
            "preprocess": self.profile,
            "source_rate_hz": 15.0,
            "action_horizon": self.action_horizon,
            "action_names": ACTION_ORDER,
            "action_reference": "each [x,y,heading] row is cumulative from request pose",
            "supports_batch": True,
            "batch_request": {"batch": "list[observation]"},
            "batch_response": {"batch": "list[reply]"},
        }

    def context(self, task: str) -> tuple[torch.Tensor, torch.Tensor, Path]:
        prompt = PROMPT_TEMPLATE.format(task=task)
        cached = self.context_cache.get(prompt)
        if cached is not None:
            return cached
        filename = hashlib.sha256(prompt.encode("utf-8")).hexdigest() + ".t5_len128.wan22ti2v5b.pt"
        path = next((root / filename for root in self.cache_dirs if (root / filename).is_file()), None)
        if path is None:
            raise FileNotFoundError(f"No T5 cache for task {task!r}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        context = payload["context"].float().contiguous().clone()
        source_mask = payload["mask"].bool().contiguous()
        if context.shape[0] != self.context_len or source_mask.shape != (self.context_len,):
            raise ValueError(f"Invalid T5 cache shape in {path}")
        context[~source_mask] = 0.0
        value = (context, torch.ones_like(source_mask), path)
        self.context_cache[prompt] = value
        return value

    def _format_reply(
        self,
        actions: np.ndarray,
        task: str,
        cache_path: Path,
        started: float,
        model_started: float,
    ) -> dict[str, Any]:
        if actions.shape != (self.action_horizon, self.action_dim) or not np.all(np.isfinite(actions)):
            raise ValueError(f"Invalid denormalized action output {actions.shape}")
        motion = actions[:, :3].astype(np.float32)
        near_zero = (
            (np.linalg.norm(motion[:, :2], axis=1) <= 0.05)
            & (np.abs(motion[:, 2]) <= 0.05)
        )
        return {
            "actions": actions.astype(np.float32),
            "relative_se2_actions": motion,
            "bbox_actions": actions[:, 3:7].astype(np.float32),
            "navigable_actions": actions[:, 7:9].astype(np.float32),
            "near_zero_chunk": bool(float(np.mean(near_zero)) >= 0.8 and bool(near_zero[-1])),
            "near_zero_ratio": float(np.mean(near_zero)),
            "task": task,
            "text_cache_file": cache_path.name,
            "policy_timing": {
                "model_s": float(time.monotonic() - model_started),
                "infer_s": float(time.monotonic() - started),
            },
        }

    def _infer_action_batch(
        self,
        model_input: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        seeds: list[int],
    ) -> torch.Tensor:
        """Run the causal ActionDiT with one cached video prefix per sample."""
        batch_size = int(model_input.shape[0])
        if batch_size != len(seeds):
            raise ValueError(
                f"seed list must have length {batch_size}, got {len(seeds)}"
            )

        model = self.model
        model_input = model_input.to(device=self.device, dtype=model.torch_dtype)
        context = context.to(device=self.device, dtype=model.torch_dtype)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool)
        first_frame_latents = model._encode_input_image_latents_tensor(
            input_image=model_input,
            tiled=False,
        )

        samples = []
        for seed in seeds:
            generator = torch.Generator(device=self.rand_device).manual_seed(int(seed))
            samples.append(
                torch.randn(
                    (self.action_horizon, model.action_expert.action_dim),
                    generator=generator,
                    device=self.rand_device,
                    dtype=torch.float32,
                )
            )
        latents_action = torch.stack(samples, dim=0).to(
            device=self.device,
            dtype=model.torch_dtype,
        )

        timestep_video = torch.zeros(
            (batch_size,),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = model.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=bool(
                getattr(model.video_expert, "fuse_vae_embedding_in_latents", False)
            ),
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = model._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=self.action_horizon,
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = model.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        timesteps, deltas = model.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=self.num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
        )
        for step_t, step_delta in zip(timesteps, deltas):
            timestep_action = step_t.expand(batch_size).to(
                device=self.device,
                dtype=latents_action.dtype,
            )
            pred_action = model._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            latents_action = model.infer_action_scheduler.step(
                pred_action,
                step_delta,
                latents_action,
            )
        return latents_action.detach().to(device="cpu", dtype=torch.float32)

    def infer_batch(self, observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not observations:
            raise ValueError("batch must contain at least one observation")
        started = time.monotonic()
        rgbs = [extract_image(obs) for obs in observations]
        tasks = [str(obs.get("task") or self.default_task).strip() for obs in observations]
        contexts = [self.context(task) for task in tasks]
        model_input = torch.cat([
            preprocess_image(rgb, self.profile, self.device, self.model.torch_dtype)
            for rgb in rgbs
        ], dim=0)
        context = torch.stack([value[0] for value in contexts], dim=0)
        context_mask = torch.stack([value[1] for value in contexts], dim=0)
        seeds = [int(obs.get("diffusion_seed", self.seed)) for obs in observations]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        model_started = time.monotonic()
        with torch.inference_mode():
            normalized = self._infer_action_batch(
                model_input=model_input,
                context=context,
                context_mask=context_mask,
                seeds=seeds,
            ).numpy()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        # infer_action preserves the legacy [H,D] return for a single sample;
        # the batch protocol always keeps [B,H,D], including B=1.
        if len(observations) == 1 and normalized.ndim == 2:
            normalized = normalized[None, ...]
        if normalized.shape != (len(observations), self.action_horizon, self.action_dim):
            raise ValueError(f"Invalid normalized batch output {normalized.shape}")
        actions = (normalized - self.offset[None, None, :]) / self.scale[None, None, :]
        batch_model_s = float(time.monotonic() - model_started)
        replies = []
        for index, (action, task, cache) in enumerate(zip(actions, tasks, contexts)):
            reply = self._format_reply(
                action.astype(np.float32), task, cache[2], started, model_started
            )
            reply["batch_index"] = index
            reply["policy_timing"]["batch_size"] = len(observations)
            reply["policy_timing"]["batch_model_s"] = batch_model_s
            reply["policy_timing"]["per_sample_model_s"] = batch_model_s / len(observations)
            replies.append(reply)
        return replies

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        batch = obs.get("batch") if isinstance(obs, dict) else None
        if batch is not None:
            if not isinstance(batch, list) or any(not isinstance(item, dict) for item in batch):
                raise ValueError("batch must be a list of observation dictionaries")
            return {"batch": self.infer_batch(batch)}

        started = time.monotonic()
        rgb = extract_image(obs)
        task = str(obs.get("task") or self.default_task).strip()
        context, context_mask, cache_path = self.context(task)
        model_input = preprocess_image(
            rgb, self.profile, self.device, self.model.torch_dtype
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        model_started = time.monotonic()
        with torch.inference_mode():
            normalized = self.model.infer_action(
                prompt=None,
                input_image=model_input,
                action_horizon=self.action_horizon,
                context=context,
                context_mask=context_mask,
                num_inference_steps=self.num_inference_steps,
                seed=self.seed,
                rand_device=self.rand_device,
                tiled=False,
            )["action"].float().cpu().numpy()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        if normalized.shape != (32, 9) or not np.all(np.isfinite(normalized)):
            raise ValueError(f"Invalid normalized action output {normalized.shape}")
        actions = (normalized - self.offset[None, :]) / self.scale[None, :]
        motion = actions[:, :3].astype(np.float32)
        near_zero = (
            (np.linalg.norm(motion[:, :2], axis=1) <= 0.05)
            & (np.abs(motion[:, 2]) <= 0.05)
        )
        timing = {
            "model_s": float(time.monotonic() - model_started),
            "infer_s": float(time.monotonic() - started),
        }
        return self._format_reply(actions, task, cache_path, started, model_started)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--text-cache-dir", type=Path, action="append", default=[])
    parser.add_argument("--preprocess-profile", choices=tuple(PREPROCESS_PROFILES), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=18018)
    parser.add_argument("--default-task", default="Move to the chair.")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rand-device", default="cpu")
    args = parser.parse_args()
    policy = UniWAMPolicy(args)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port, metadata=policy.metadata
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
