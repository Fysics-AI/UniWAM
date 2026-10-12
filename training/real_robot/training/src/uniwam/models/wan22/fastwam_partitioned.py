from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from uniwam.utils.logging_config import get_logger

from .fastwam import FastWAM
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .partitioned_action_dit import PartitionedActionDiT

logger = get_logger(__name__)


class FastWAMPartitioned(FastWAM):
    """Two independent video streams conditioned through one partitioned ActionDiT."""

    def __init__(
        self,
        video_expert,
        action_expert: PartitionedActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_nav_action: float = 1.0,
        loss_lambda_manip_action: float = 1.0,
    ):
        super().__init__(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_dim=text_dim,
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=1.0,
        )
        self.loss_lambda_nav_action = float(loss_lambda_nav_action)
        self.loss_lambda_manip_action = float(loss_lambda_manip_action)

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_nav_action: float = 1.0,
        loss_lambda_manip_action: float = 1.0,
    ):
        if video_dit_config is None or action_dit_config is None:
            raise ValueError("video_dit_config and action_dit_config are required.")
        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )
        action_expert = PartitionedActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        video_expert = components.dit
        if action_expert.num_heads != video_expert.num_heads:
            raise ValueError("Action/video num_heads mismatch.")
        if action_expert.attn_head_dim != video_expert.attn_head_dim:
            raise ValueError("Action/video attn_head_dim mismatch.")
        if len(action_expert.blocks) != len(video_expert.blocks):
            raise ValueError("Action/video layer count mismatch.")
        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )
        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_nav_action=loss_lambda_nav_action,
            loss_lambda_manip_action=loss_lambda_manip_action,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model

    @staticmethod
    def _validate_video(name: str, video: torch.Tensor) -> tuple[int, int]:
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"{name} must be [B,3,T,H,W], got {tuple(video.shape)}.")
        if video.shape[2] % 4 != 1 or video.shape[3] % 32 or video.shape[4] % 32:
            raise ValueError(f"Invalid Wan video shape for {name}: {tuple(video.shape)}.")
        return int(video.shape[0]), int(video.shape[2])

    @staticmethod
    def _split_video_pre(pre_state: dict, batch_size: int) -> tuple[dict, dict]:
        batch_keys = {"tokens", "t", "t_mod", "context", "context_mask"}

        def take(start: int, end: int) -> dict:
            result = {}
            for key, value in pre_state.items():
                if key == "meta":
                    result[key] = dict(value, batch_size=end - start)
                elif key in batch_keys:
                    result[key] = value[start:end]
                else:
                    result[key] = value
            return result

        return take(0, batch_size), take(batch_size, 2 * batch_size)

    def build_inputs(self, sample, tiled: bool = False):
        if "nav_video" not in sample or "manip_video" not in sample:
            raise ValueError("FastWAMPartitioned requires separate nav_video and manip_video tensors.")
        nav_video = sample["nav_video"]
        manip_video = sample["manip_video"]
        batch_size, num_frames = self._validate_video("nav_video", nav_video)
        manip_batch, manip_frames = self._validate_video("manip_video", manip_video)
        if manip_batch != batch_size or manip_frames != num_frames or nav_video.shape != manip_video.shape:
            raise ValueError(
                "Matched 2B mode requires identical nav/manip video shapes, got "
                f"{tuple(nav_video.shape)} and {tuple(manip_video.shape)}."
            )
        manip_action = sample["manip_action"]
        nav_action = sample["nav_action"]
        if tuple(manip_action.shape[1:]) != (
            self.action_expert.manip_horizon,
            self.action_expert.manip_action_dim,
        ):
            raise ValueError(f"manip_action shape mismatch: {tuple(manip_action.shape)}.")
        if tuple(nav_action.shape[1:]) != (
            self.action_expert.nav_horizon,
            self.action_expert.nav_action_dim,
        ):
            raise ValueError(f"nav_action shape mismatch: {tuple(nav_action.shape)}.")

        paired_video = torch.cat((nav_video, manip_video), dim=0).to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        paired_latents = self._encode_video_latents(paired_video, tiled=tiled)
        nav_input_latents, manip_input_latents = paired_latents.chunk(2, dim=0)
        fuse = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
        nav_first = nav_input_latents[:, :, :1] if fuse else None
        manip_first = manip_input_latents[:, :, :1] if fuse else None

        context = sample["context"].to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        context_mask = sample["context_mask"].to(
            device=self.device, dtype=torch.bool, non_blocking=True
        )
        if self.proprio_encoder is not None:
            proprio = sample.get("proprio")
            if proprio is None or proprio.ndim != 3 or proprio.shape[-1] != self.proprio_dim:
                raise ValueError(f"Expected proprio [B,T,{self.proprio_dim}].")
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio[:, 0].to(device=self.device, dtype=self.torch_dtype),
            )
        return {
            "nav_input_latents": nav_input_latents,
            "manip_input_latents": manip_input_latents,
            "nav_first_frame_latents": nav_first,
            "manip_first_frame_latents": manip_first,
            "fuse_vae_embedding_in_latents": fuse,
            "context": context,
            "context_mask": context_mask,
            "nav_action": nav_action.to(self.device, self.torch_dtype, non_blocking=True),
            "manip_action": manip_action.to(self.device, self.torch_dtype, non_blocking=True),
            "action_is_pad": sample.get("action_is_pad"),
            "nav_action_is_pad": sample.get("nav_action_is_pad"),
            "image_is_pad": sample.get("image_is_pad"),
            "nav_loss_valid": sample.get("nav_loss_valid"),
            "manip_loss_valid": sample.get("manip_loss_valid"),
        }

    def _partitioned_mot_forward(
        self,
        nav_video_pre: dict,
        manip_video_pre: dict,
        action_pre: dict,
    ) -> dict[str, torch.Tensor]:
        nav_video_tokens = nav_video_pre["tokens"]
        manip_video_tokens = manip_video_pre["tokens"]
        action_tokens = action_pre["tokens"]
        if nav_video_tokens.shape != manip_video_tokens.shape:
            raise ValueError("The two video token streams must have identical shapes.")
        video_seq_len = int(nav_video_tokens.shape[1])
        tokens_per_frame = int(nav_video_pre["meta"]["tokens_per_frame"])
        if tokens_per_frame != int(manip_video_pre["meta"]["tokens_per_frame"]):
            raise ValueError("The two video streams must have matching first-frame token counts.")
        first_frame_tokens = min(video_seq_len, tokens_per_frame)
        manip_horizon = self.action_expert.manip_horizon
        nav_horizon = self.action_expert.nav_horizon
        if manip_horizon != nav_horizon:
            raise ValueError(
                "Batch-packed shared ActionDiT requires equal manipulation/navigation horizons, "
                f"got {manip_horizon} and {nav_horizon}."
            )
        device = action_tokens.device

        video_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=tokens_per_frame,
            device=device,
        )
        # The two logically block-diagonal branches have equal lengths, so pack
        # them along batch instead of materializing a sparse 64-token mask. This
        # is mathematically equivalent and keeps the attention kernel dense.
        branch_action_mask = torch.ones(
            (manip_horizon, first_frame_tokens + manip_horizon),
            dtype=torch.bool,
            device=device,
        )

        for layer_idx in range(self.mot.num_layers):
            video_block = self.video_expert.blocks[layer_idx]
            video_batched = torch.cat((nav_video_tokens, manip_video_tokens), dim=0)
            video_io = self.mot._build_expert_attention_io(
                expert=self.video_expert,
                block=video_block,
                x=video_batched,
                freqs=nav_video_pre["freqs"],
                t_mod=torch.cat((nav_video_pre["t_mod"], manip_video_pre["t_mod"]), dim=0),
            )
            (
                q_video,
                k_video,
                v_video,
                residual_video,
                gate_msa_video,
                shift_mlp_video,
                scale_mlp_video,
                gate_mlp_video,
                video_checkpoint,
            ) = video_io
            mixed_video = self.mot._mixed_attention(q_video, k_video, v_video, video_mask)
            video_batched = self.mot._apply_post_with_optional_checkpoint(
                block=video_block,
                residual_x=residual_video,
                gate_msa=gate_msa_video,
                shift_mlp=shift_mlp_video,
                scale_mlp=scale_mlp_video,
                gate_mlp=gate_mlp_video,
                use_gradient_checkpointing=video_checkpoint,
                mixed_slice=mixed_video,
                context_payload={
                    "context": torch.cat((nav_video_pre["context"], manip_video_pre["context"]), dim=0),
                    "mask": torch.cat((nav_video_pre["context_mask"], manip_video_pre["context_mask"]), dim=0),
                },
            )
            nav_video_tokens, manip_video_tokens = video_batched.chunk(2, dim=0)
            k_nav_video, k_manip_video = k_video.chunk(2, dim=0)
            v_nav_video, v_manip_video = v_video.chunk(2, dim=0)

            action_block = self.action_expert.blocks[layer_idx]
            manip_action_tokens = action_tokens[:, :manip_horizon]
            nav_action_tokens = action_tokens[:, manip_horizon:]
            action_batched = torch.cat((manip_action_tokens, nav_action_tokens), dim=0)
            action_t_mod = torch.cat(
                (
                    action_pre["t_mod"][:, :manip_horizon],
                    action_pre["t_mod"][:, manip_horizon:],
                ),
                dim=0,
            )
            action_io = self.mot._build_expert_attention_io(
                expert=self.action_expert,
                block=action_block,
                x=action_batched,
                freqs=action_pre["freqs"][:manip_horizon],
                t_mod=action_t_mod,
            )
            (
                q_action,
                k_action,
                v_action,
                residual_action,
                gate_msa_action,
                shift_mlp_action,
                scale_mlp_action,
                gate_mlp_action,
                action_checkpoint,
            ) = action_io
            action_video_k = torch.cat(
                (
                    k_manip_video[:, :first_frame_tokens],
                    k_nav_video[:, :first_frame_tokens],
                ),
                dim=0,
            )
            action_video_v = torch.cat(
                (
                    v_manip_video[:, :first_frame_tokens],
                    v_nav_video[:, :first_frame_tokens],
                ),
                dim=0,
            )
            mixed_action = self.mot._mixed_attention(
                q_action,
                torch.cat((action_video_k, k_action), dim=1),
                torch.cat((action_video_v, v_action), dim=1),
                branch_action_mask,
            )
            action_batched = self.mot._apply_post_with_optional_checkpoint(
                block=action_block,
                residual_x=residual_action,
                gate_msa=gate_msa_action,
                shift_mlp=shift_mlp_action,
                scale_mlp=scale_mlp_action,
                gate_mlp=gate_mlp_action,
                use_gradient_checkpointing=action_checkpoint,
                mixed_slice=mixed_action,
                context_payload={
                    "context": torch.cat((action_pre["context"], action_pre["context"]), dim=0),
                    "mask": torch.cat(
                        (
                            action_pre["context_mask"][:, :manip_horizon],
                            action_pre["context_mask"][:, manip_horizon:],
                        ),
                        dim=0,
                    ),
                },
            )
            manip_action_tokens, nav_action_tokens = action_batched.chunk(2, dim=0)
            action_tokens = torch.cat((manip_action_tokens, nav_action_tokens), dim=1)
        return {
            "nav_video": nav_video_tokens,
            "manip_video": manip_video_tokens,
            "action": action_tokens,
            "action_attention_mask": branch_action_mask,
        }

    @staticmethod
    def build_partitioned_action_mask(
        *,
        first_frame_tokens: int,
        manip_horizon: int,
        nav_horizon: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        """Build Q(action) x K(manip video, nav video, action) block mask."""
        if min(first_frame_tokens, manip_horizon, nav_horizon) <= 0:
            raise ValueError("Mask dimensions must all be positive.")
        mask = torch.zeros(
            (
                manip_horizon + nav_horizon,
                2 * first_frame_tokens + manip_horizon + nav_horizon,
            ),
            dtype=torch.bool,
            device=device,
        )
        action_key_start = 2 * first_frame_tokens
        mask[:manip_horizon, :first_frame_tokens] = True
        mask[:manip_horizon, action_key_start : action_key_start + manip_horizon] = True
        mask[manip_horizon:, first_frame_tokens : 2 * first_frame_tokens] = True
        mask[manip_horizon:, action_key_start + manip_horizon :] = True
        return mask

    def _action_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        is_pad: Optional[torch.Tensor],
        loss_valid: Optional[torch.Tensor],
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        token_loss = F.mse_loss(prediction.float(), target.float(), reduction="none").mean(dim=-1)
        valid = torch.ones_like(token_loss) if is_pad is None else (~is_pad.to(self.device)).to(token_loss.dtype)
        active = (
            torch.ones((token_loss.shape[0],), device=self.device, dtype=token_loss.dtype)
            if loss_valid is None
            else loss_valid.to(self.device, dtype=torch.bool).to(token_loss.dtype)
        )
        per_sample = (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        weight = self.train_action_scheduler.training_weight(timestep).to(per_sample)
        return (per_sample * weight * active).sum() / active.sum().clamp(min=1.0)

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        nav_input = inputs["nav_input_latents"]
        manip_input = inputs["manip_input_latents"]
        batch_size = int(nav_input.shape[0])
        context = inputs["context"]
        context_mask = inputs["context_mask"]

        def noisy_video(source: torch.Tensor):
            noise = torch.randn_like(source)
            timestep = self.train_video_scheduler.sample_training_t(
                batch_size=batch_size, device=self.device, dtype=source.dtype
            )
            noisy = self.train_video_scheduler.add_noise(source, noise, timestep)
            target = self.train_video_scheduler.training_target(source, noise, timestep)
            return noisy, target, timestep

        nav_latents, target_nav_video, timestep_nav_video = noisy_video(nav_input)
        manip_latents, target_manip_video, timestep_manip_video = noisy_video(manip_input)
        if inputs["nav_first_frame_latents"] is not None:
            nav_latents[:, :, :1] = inputs["nav_first_frame_latents"]
            manip_latents[:, :, :1] = inputs["manip_first_frame_latents"]

        def noisy_action(source: torch.Tensor):
            noise = torch.randn_like(source)
            timestep = self.train_action_scheduler.sample_training_t(
                batch_size=batch_size, device=self.device, dtype=source.dtype
            )
            noisy = self.train_action_scheduler.add_noise(source, noise, timestep)
            target = self.train_action_scheduler.training_target(source, noise, timestep)
            return noisy, target, timestep

        noisy_manip, target_manip, timestep_manip = noisy_action(inputs["manip_action"])
        noisy_nav, target_nav, timestep_nav = noisy_action(inputs["nav_action"])

        paired_video_pre = self.video_expert.pre_dit(
            x=torch.cat((nav_latents, manip_latents), dim=0),
            timestep=torch.cat((timestep_nav_video, timestep_manip_video), dim=0),
            context=torch.cat((context, context), dim=0),
            context_mask=torch.cat((context_mask, context_mask), dim=0),
            action=None,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        nav_video_pre, manip_video_pre = self._split_video_pre(paired_video_pre, batch_size)
        action_pre = self.action_expert.pre_dit(
            manip_action=noisy_manip,
            nav_action=noisy_nav,
            manip_timestep=timestep_manip,
            nav_timestep=timestep_nav,
            context=context,
            context_mask=context_mask,
        )
        output = self._partitioned_mot_forward(nav_video_pre, manip_video_pre, action_pre)
        pred_nav_video = self.video_expert.post_dit(output["nav_video"], nav_video_pre)
        pred_manip_video = self.video_expert.post_dit(output["manip_video"], manip_video_pre)
        action_output = self.action_expert.post_dit(output["action"], action_pre)

        include_initial = inputs["nav_first_frame_latents"] is None
        if not include_initial:
            pred_nav_video = pred_nav_video[:, :, 1:]
            pred_manip_video = pred_manip_video[:, :, 1:]
            target_nav_video = target_nav_video[:, :, 1:]
            target_manip_video = target_manip_video[:, :, 1:]
        image_is_pad = inputs["image_is_pad"]
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(self.device, dtype=torch.bool)
        nav_video_per_sample = self._compute_video_loss_per_sample(
            pred_nav_video, target_nav_video, image_is_pad, include_initial
        )
        manip_video_per_sample = self._compute_video_loss_per_sample(
            pred_manip_video, target_manip_video, image_is_pad, include_initial
        )
        loss_video_nav = (
            nav_video_per_sample
            * self.train_video_scheduler.training_weight(timestep_nav_video).to(nav_video_per_sample)
        ).mean()
        loss_video_manip = (
            manip_video_per_sample
            * self.train_video_scheduler.training_weight(timestep_manip_video).to(manip_video_per_sample)
        ).mean()
        loss_video = 0.5 * (loss_video_nav + loss_video_manip)
        loss_manip = self._action_loss(
            action_output["manip"],
            target_manip,
            inputs["action_is_pad"],
            inputs["manip_loss_valid"],
            timestep_manip,
        )
        loss_nav = self._action_loss(
            action_output["nav"],
            target_nav,
            inputs["nav_action_is_pad"],
            inputs["nav_loss_valid"],
            timestep_nav,
        )
        loss = (
            self.loss_lambda_video * loss_video
            + self.loss_lambda_manip_action * loss_manip
            + self.loss_lambda_nav_action * loss_nav
        )
        return loss, {
            "loss_video": self.loss_lambda_video * float(loss_video.detach()),
            "loss_video_nav": self.loss_lambda_video * float(loss_video_nav.detach()),
            "loss_video_manip": self.loss_lambda_video * float(loss_video_manip.detach()),
            "loss_action_manip": self.loss_lambda_manip_action * float(loss_manip.detach()),
            "loss_action_nav": self.loss_lambda_nav_action * float(loss_nav.detach()),
        }

    @torch.no_grad()
    def _predict_partitioned_action_with_cache(
        self,
        *,
        latents_manip: torch.Tensor,
        latents_nav: torch.Tensor,
        timestep_manip: torch.Tensor,
        timestep_nav: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        paired_video_cache: list[dict[str, torch.Tensor]],
        first_frame_tokens: int,
    ) -> dict[str, torch.Tensor]:
        """Run one shared ActionDiT pass with isolated manip/nav batch branches."""
        action_pre = self.action_expert.pre_dit(
            manip_action=latents_manip,
            nav_action=latents_nav,
            manip_timestep=timestep_manip,
            nav_timestep=timestep_nav,
            context=context,
            context_mask=context_mask,
        )
        batch_size = int(latents_manip.shape[0])
        horizon = int(self.action_expert.manip_horizon)
        if self.action_expert.nav_horizon != horizon:
            raise ValueError("Cached partitioned inference currently requires equal action horizons.")
        if len(paired_video_cache) != self.mot.num_layers:
            raise ValueError(
                f"Expected {self.mot.num_layers} video cache layers, got {len(paired_video_cache)}."
            )
        if first_frame_tokens <= 0:
            raise ValueError(f"first_frame_tokens must be positive, got {first_frame_tokens}.")

        action_tokens = action_pre["tokens"]
        manip_tokens = action_tokens[:, :horizon]
        nav_tokens = action_tokens[:, horizon:]
        action_batched = torch.cat((manip_tokens, nav_tokens), dim=0)
        action_t_mod = torch.cat(
            (action_pre["t_mod"][:, :horizon], action_pre["t_mod"][:, horizon:]),
            dim=0,
        )
        action_context = torch.cat((action_pre["context"], action_pre["context"]), dim=0)
        action_context_mask = torch.cat(
            (
                action_pre["context_mask"][:, :horizon],
                action_pre["context_mask"][:, horizon:],
            ),
            dim=0,
        )
        attention_mask = torch.ones(
            (horizon, first_frame_tokens + horizon),
            dtype=torch.bool,
            device=action_batched.device,
        )

        for layer_idx in range(self.mot.num_layers):
            cache = paired_video_cache[layer_idx]
            k_video = cache["k"]
            v_video = cache["v"]
            if k_video.shape[0] != 2 * batch_size or v_video.shape[0] != 2 * batch_size:
                raise ValueError(
                    f"Video cache batch mismatch at layer {layer_idx}: "
                    f"k={tuple(k_video.shape)} v={tuple(v_video.shape)} expected batch={2 * batch_size}."
                )
            if k_video.shape[1] < first_frame_tokens or v_video.shape[1] < first_frame_tokens:
                raise ValueError(f"Video cache is shorter than one frame at layer {layer_idx}.")

            # Video cache order is [nav, manip], while training packs action as
            # [manip, nav]. Reorder it to preserve the exact training contract.
            k_nav, k_manip = k_video.chunk(2, dim=0)
            v_nav, v_manip = v_video.chunk(2, dim=0)
            action_video_k = torch.cat(
                (k_manip[:, :first_frame_tokens], k_nav[:, :first_frame_tokens]), dim=0
            )
            action_video_v = torch.cat(
                (v_manip[:, :first_frame_tokens], v_nav[:, :first_frame_tokens]), dim=0
            )

            block = self.action_expert.blocks[layer_idx]
            (
                q_action,
                k_action,
                v_action,
                residual_action,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self.mot._build_expert_attention_io(
                expert=self.action_expert,
                block=block,
                x=action_batched,
                freqs=action_pre["freqs"][:horizon],
                t_mod=action_t_mod,
            )
            mixed_action = self.mot._mixed_attention(
                q_action,
                torch.cat((action_video_k, k_action), dim=1),
                torch.cat((action_video_v, v_action), dim=1),
                attention_mask,
            )
            action_batched = self.mot._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_action,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed_action,
                context_payload={"context": action_context, "mask": action_context_mask},
            )

        manip_tokens, nav_tokens = action_batched.chunk(2, dim=0)
        packed_tokens = torch.cat((manip_tokens, nav_tokens), dim=1)
        return self.action_expert.post_dit(packed_tokens, action_pre)

    @torch.no_grad()
    def infer_action(
        self,
        *,
        action_horizon: int = 32,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        nav_input_image: Optional[torch.Tensor] = None,
        manip_input_image: Optional[torch.Tensor] = None,
        num_inference_steps: int = 10,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        """Infer isolated navigation and manipulation chunks with one shared ActionDiT."""
        self.eval()
        profile_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}

        def profile_start() -> Optional[torch.cuda.Event]:
            if not torch.cuda.is_available() or not str(self.device).startswith("cuda"):
                return None
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event

        def profile_end(name: str, start: Optional[torch.cuda.Event]) -> None:
            if start is None:
                return
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            profile_events.setdefault(name, []).append((start, end))
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError("infer_action requires video_attention_mask_mode='first_frame_causal'.")
        if action_horizon != self.action_expert.manip_horizon or action_horizon != self.action_expert.nav_horizon:
            raise ValueError(
                f"action_horizon={action_horizon} does not match trained horizons "
                f"({self.action_expert.manip_horizon}, {self.action_expert.nav_horizon})."
            )
        if nav_input_image is None or manip_input_image is None:
            raise ValueError("Both nav_input_image and manip_input_image are required.")

        def prepare_image(name: str, image: torch.Tensor) -> torch.Tensor:
            if image.ndim == 3:
                image = image.unsqueeze(0)
            if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
                raise ValueError(f"{name} must be [3,H,W] or [1,3,H,W], got {tuple(image.shape)}.")
            if image.shape[-2] % 32 or image.shape[-1] % 32:
                raise ValueError(f"{name} H/W must be divisible by 32, got {tuple(image.shape[-2:])}.")
            return image.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        nav_image = prepare_image("nav_input_image", nav_input_image)
        manip_image = prepare_image("manip_input_image", manip_input_image)
        if nav_image.shape != manip_image.shape:
            raise ValueError(
                f"Current matched 2B checkpoint requires equal image shapes, got "
                f"{tuple(nav_image.shape)} and {tuple(manip_image.shape)}."
            )
        if context is None or context_mask is None:
            raise ValueError("Precomputed context and context_mask are required.")
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        if context.ndim != 3 or context_mask.ndim != 2 or context.shape[:2] != context_mask.shape:
            raise ValueError(
                f"Bad context shapes: context={tuple(context.shape)} mask={tuple(context_mask.shape)}."
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 2 or proprio.shape != (1, self.proprio_dim):
                raise ValueError(f"proprio must be [1,{self.proprio_dim}], got {tuple(proprio.shape)}.")
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
            )

        stage_start = profile_start()
        paired_latents = self._encode_video_latents(
            torch.cat((nav_image, manip_image), dim=0).unsqueeze(2), tiled=tiled
        )
        profile_end("vae_encode_2b", stage_start)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
        stage_start = profile_start()
        paired_video_pre = self.video_expert.pre_dit(
            x=paired_latents,
            timestep=torch.zeros((2,), dtype=paired_latents.dtype, device=self.device),
            context=torch.cat((context, context), dim=0),
            context_mask=torch.cat((context_mask, context_mask), dim=0),
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        profile_end("video_pre_dit_2b", stage_start)
        video_seq_len = int(paired_video_pre["tokens"].shape[1])
        first_frame_tokens = int(paired_video_pre["meta"]["tokens_per_frame"])
        if video_seq_len != first_frame_tokens:
            raise ValueError(
                f"Image-only prefill must contain exactly one latent frame, got "
                f"seq={video_seq_len}, frame={first_frame_tokens}."
            )
        video_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=first_frame_tokens,
            device=paired_video_pre["tokens"].device,
        )
        stage_start = profile_start()
        paired_video_cache = self.mot.prefill_video_cache(
            video_tokens=paired_video_pre["tokens"],
            video_freqs=paired_video_pre["freqs"],
            video_t_mod=paired_video_pre["t_mod"],
            video_context_payload={
                "context": paired_video_pre["context"],
                "mask": paired_video_pre["context_mask"],
            },
            video_attention_mask=video_mask,
        )
        profile_end("wan_video_prefill_2b", stage_start)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_manip = torch.randn(
            (1, action_horizon, self.action_expert.manip_action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_nav = torch.randn(
            (1, action_horizon, self.action_expert.nav_action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        timesteps, deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_manip.dtype,
            shift_override=sigma_shift,
        )
        denoise_start = profile_start()
        for step_t, step_delta in zip(timesteps, deltas):
            timestep = step_t.unsqueeze(0).to(device=self.device, dtype=latents_manip.dtype)
            step_start = profile_start()
            prediction = self._predict_partitioned_action_with_cache(
                latents_manip=latents_manip,
                latents_nav=latents_nav,
                timestep_manip=timestep,
                timestep_nav=timestep,
                context=context,
                context_mask=context_mask,
                paired_video_cache=paired_video_cache,
                first_frame_tokens=first_frame_tokens,
            )
            profile_end("shared_action_dit_steps", step_start)
            latents_manip = self.infer_action_scheduler.step(
                prediction["manip"], step_delta, latents_manip
            )
            latents_nav = self.infer_action_scheduler.step(prediction["nav"], step_delta, latents_nav)
        profile_end("action_denoise_total", denoise_start)

        profile_ms: dict[str, float] = {}
        if profile_events:
            torch.cuda.synchronize(device=self.device)
            profile_ms = {
                name: float(sum(start.elapsed_time(end) for start, end in pairs))
                for name, pairs in profile_events.items()
            }
            profile_ms["shared_action_dit_step_mean"] = (
                profile_ms["shared_action_dit_steps"] / max(1, int(num_inference_steps))
            )

        return {
            "manip_action": latents_manip[0].detach().to(device="cpu", dtype=torch.float32),
            "nav_action": latents_nav[0].detach().to(device="cpu", dtype=torch.float32),
            "profile_ms": profile_ms,
        }

    def forward(self, sample, tiled: bool = False, **kwargs):
        del kwargs
        return self.training_loss(sample, tiled=tiled)
