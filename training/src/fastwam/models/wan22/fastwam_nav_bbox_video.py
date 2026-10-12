"""Standard FastWAM with an auxiliary future-target bbox video loss.

The bbox is never model input.  It is used only as a spatial supervision mask
for the generated future video, so the ActionDiT remains the normal 2D
``[vx, vyaw]`` FastWAM ActionDiT.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .fastwam import FastWAM


class FastWAMNavBBoxVideo(FastWAM):
    """Single-view navigation FastWAM with bbox-local video supervision."""

    def __init__(self, *args, loss_lambda_video_bbox: float = 0.5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_lambda_video_bbox = float(loss_lambda_video_bbox)

    def build_inputs(self, sample, tiled: bool = False):
        inputs = super().build_inputs(sample, tiled=tiled)
        video_bbox = sample.get("video_bbox")
        video_bbox_visible = sample.get("video_bbox_visible")
        if video_bbox is None or video_bbox_visible is None:
            raise ValueError("FastWAMNavBBoxVideo requires video_bbox and video_bbox_visible.")

        expected_shape = (inputs["input_latents"].shape[0], sample["video"].shape[2])
        if video_bbox.ndim != 3 or tuple(video_bbox.shape) != (*expected_shape, 4):
            raise ValueError(
                "video_bbox must be [B,T,4], got "
                f"{tuple(video_bbox.shape)} for expected B,T={expected_shape}."
            )
        if video_bbox_visible.ndim != 2 or tuple(video_bbox_visible.shape) != expected_shape:
            raise ValueError(
                "video_bbox_visible must be [B,T], got "
                f"{tuple(video_bbox_visible.shape)} for expected B,T={expected_shape}."
            )
        inputs["video_bbox"] = video_bbox.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        inputs["video_bbox_visible"] = video_bbox_visible.to(
            device=self.device, dtype=torch.bool, non_blocking=True
        )
        return inputs

    @staticmethod
    def _bbox_mask(
        bbox: torch.Tensor,
        visible: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Rasterize normalized boxes into a [B,T,H,W] loss mask."""
        x1 = torch.floor(bbox[..., 0].clamp(0.0, 1.0) * width).clamp(0, width - 1)
        y1 = torch.floor(bbox[..., 1].clamp(0.0, 1.0) * height).clamp(0, height - 1)
        x2 = torch.ceil(bbox[..., 2].clamp(0.0, 1.0) * width).clamp(1, width)
        y2 = torch.ceil(bbox[..., 3].clamp(0.0, 1.0) * height).clamp(1, height)
        x2 = torch.maximum(x2, x1 + 1)
        y2 = torch.maximum(y2, y1 + 1)
        x_coords = torch.arange(width, device=bbox.device).view(1, 1, 1, width)
        y_coords = torch.arange(height, device=bbox.device).view(1, 1, height, 1)
        inside = (
            (x_coords >= x1[..., None, None])
            & (x_coords < x2[..., None, None])
            & (y_coords >= y1[..., None, None])
            & (y_coords < y2[..., None, None])
            & visible[..., None, None]
        )
        return inside.to(dtype=bbox.dtype)

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=input_latents.dtype
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)
        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=action.dtype
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            context_all={
                "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
            },
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
        )
        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        include_initial_video_step = inputs["first_frame_latents"] is None
        if not include_initial_video_step:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        video_global_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        bbox_mask = self._bbox_mask(
            inputs["video_bbox"],
            inputs["video_bbox_visible"],
            height=int(sample["video"].shape[-2]),
            width=int(sample["video"].shape[-1]),
        )
        video_bbox_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
            video_loss_mask=bbox_mask,
        )
        # The known first frame is not a video prediction.  A sample without
        # any bbox on its predicted horizon contributes no auxiliary term.
        bbox_valid = inputs["video_bbox_visible"][:, 1:].any(dim=1)
        bbox_valid_float = bbox_valid.to(dtype=video_bbox_per_sample.dtype)
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            video_global_per_sample.device, dtype=video_global_per_sample.dtype
        )
        loss_video_global = (video_global_per_sample * video_weight).mean()
        # Keep the same flow-matching importance weight as global video loss,
        # while excluding samples without any future visible bbox.
        loss_video_bbox = (video_bbox_per_sample * video_weight * bbox_valid_float).sum() / bbox_valid_float.sum().clamp(min=1.0)

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
        valid_action = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
        action_loss_per_sample = (action_loss_token * valid_action).sum(dim=1) / valid_action.sum(dim=1).clamp(min=1.0)
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_video = loss_video_global + self.loss_lambda_video_bbox * loss_video_bbox
        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        return loss_total, {
            "loss_video_global": float((self.loss_lambda_video * loss_video_global).detach().item()),
            "loss_video_bbox": float(
                (self.loss_lambda_video * self.loss_lambda_video_bbox * loss_video_bbox).detach().item()
            ),
            "loss_video": float((self.loss_lambda_video * loss_video).detach().item()),
            "loss_action": float((self.loss_lambda_action * loss_action).detach().item()),
            "bbox_video_valid_fraction": float(bbox_valid_float.detach().mean().item()),
        }
