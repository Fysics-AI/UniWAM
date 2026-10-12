"""Single-view HM3D navigation with causal 9D ActionDiT supervision."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .fastwam_nav_bbox_video import FastWAMNavBBoxVideo


class FastWAMNavCausalAux9(FastWAMNavBBoxVideo):
    """Standard FastWAM video stream plus a prefix-free causal H32 ActionDiT.

    Action order is ``[x, y, heading, bbox_xyxy, navigable_xy]``. Raw image-space
    coordinates are in [0, 1], absent annotations use -0.1, and all nine action
    dimensions are q01/q99-normalized before entering the model.
    """

    def __init__(
        self,
        *args,
        action_causal_local_window: int = 32,
        loss_lambda_motion: float = 1.5,
        loss_lambda_bbox: float = 1.0,
        loss_lambda_navigable: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.action_causal_local_window = int(action_causal_local_window)
        if self.action_causal_local_window <= 0:
            raise ValueError("action_causal_local_window must be positive")
        self.loss_lambda_motion = float(loss_lambda_motion)
        self.loss_lambda_bbox = float(loss_lambda_bbox)
        self.loss_lambda_navigable = float(loss_lambda_navigable)

    def build_inputs(self, sample, tiled: bool = False):
        inputs = super().build_inputs(sample, tiled=tiled)
        action = inputs["action"]
        if action.shape[-1] != 9:
            raise ValueError(f"FastWAMNavCausalAux9 requires 9D actions, got {action.shape[-1]}.")
        return inputs

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = super()._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=action_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        indices = torch.arange(action_seq_len, device=device)
        query = indices[:, None]
        key = indices[None, :]
        causal = (key <= query) & ((query - key) <= self.action_causal_local_window)
        mask[video_seq_len:, video_seq_len:] = causal
        return mask

    @staticmethod
    def _group_per_sample_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
        dims: slice,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        per_step = F.mse_loss(
            prediction[:, :, dims].float(), target[:, :, dims].float(), reduction="none"
        ).mean(dim=2)
        weights = valid.to(device=per_step.device, dtype=per_step.dtype)
        return (per_step * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

    def _weighted_action_per_sample_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        dimension_weights = prediction.new_tensor(
            [
                self.loss_lambda_motion,
                self.loss_lambda_motion,
                self.loss_lambda_motion,
                self.loss_lambda_bbox,
                self.loss_lambda_bbox,
                self.loss_lambda_bbox,
                self.loss_lambda_bbox,
                self.loss_lambda_navigable,
                self.loss_lambda_navigable,
            ],
            dtype=torch.float32,
        )
        per_step = (
            F.mse_loss(prediction.float(), target.float(), reduction="none")
            * dimension_weights.view(1, 1, -1)
        ).mean(dim=2)
        weights = valid.to(device=per_step.device, dtype=per_step.dtype)
        return (per_step * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

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
        target_video = self.train_video_scheduler.training_target(
            input_latents, noise_video, timestep_video
        )
        if inputs["first_frame_latents"] is not None:
            latents[:, :, :1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=action.dtype
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(
            action, noise_action, timestep_action
        )

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

        include_initial = inputs["first_frame_latents"] is None
        if not include_initial:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]
        video_global_per_sample = self._compute_video_loss_per_sample(
            pred_video, target_video, image_is_pad, include_initial
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            video_global_per_sample.device, dtype=video_global_per_sample.dtype
        )
        loss_video_global = (video_global_per_sample * video_weight).mean()
        if self.loss_lambda_video_bbox > 0.0:
            bbox_mask = self._bbox_mask(
                inputs["video_bbox"],
                inputs["video_bbox_visible"],
                height=int(sample["video"].shape[-2]),
                width=int(sample["video"].shape[-1]),
            )
            video_bbox_per_sample = self._compute_video_loss_per_sample(
                pred_video, target_video, image_is_pad, include_initial, bbox_mask
            )
            bbox_video_valid = inputs["video_bbox_visible"][:, 1:].any(dim=1).to(
                video_bbox_per_sample.dtype
            )
            loss_video_bbox = (
                video_bbox_per_sample * video_weight * bbox_video_valid
            ).sum() / bbox_video_valid.sum().clamp(min=1.0)
            bbox_video_valid_fraction = float(bbox_video_valid.mean().detach().item())
        else:
            loss_video_bbox = loss_video_global.new_zeros(())
            bbox_video_valid_fraction = 0.0
        loss_video = loss_video_global + self.loss_lambda_video_bbox * loss_video_bbox

        valid = ~action_is_pad
        motion_per_sample = self._group_per_sample_loss(
            pred_action, target_action, slice(0, 3), valid
        )
        bbox_per_sample = self._group_per_sample_loss(
            pred_action, target_action, slice(3, 7), valid
        )
        navigable_per_sample = self._group_per_sample_loss(
            pred_action, target_action, slice(7, 9), valid
        )
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            motion_per_sample.device, dtype=motion_per_sample.dtype
        )
        action_per_sample = self._weighted_action_per_sample_loss(
            pred_action, target_action, valid
        )
        loss_action = (action_per_sample * action_weight).mean()
        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        return loss_total, {
            "loss_video_global": float(
                (self.loss_lambda_video * loss_video_global).detach().item()
            ),
            "loss_video_bbox": float(
                (
                    self.loss_lambda_video
                    * self.loss_lambda_video_bbox
                    * loss_video_bbox
                ).detach().item()
            ),
            "loss_video": float((self.loss_lambda_video * loss_video).detach().item()),
            "loss_action": float((self.loss_lambda_action * loss_action).detach().item()),
            "loss_motion": float(motion_per_sample.mean().detach().item()),
            "loss_bbox": float(bbox_per_sample.mean().detach().item()),
            "loss_navigable": float(navigable_per_sample.mean().detach().item()),
            "bbox_video_valid_fraction": bbox_video_valid_fraction,
        }
