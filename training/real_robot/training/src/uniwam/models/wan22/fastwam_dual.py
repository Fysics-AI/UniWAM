from __future__ import annotations

import os
from typing import Any, Optional

import torch
import torch.nn.functional as F

from uniwam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .fastwam import FastWAM
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT

logger = get_logger(__name__)


class FastWAMDual(FastWAM):
    """Shared Wan video expert with separate navigation and manipulation ActionDiTs."""

    def __init__(
        self,
        video_expert,
        nav_action_expert: ActionDiT,
        manip_action_expert: ActionDiT,
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
        attention_partition_mode: str = "robotwin_top_nav",
    ):
        super().__init__(
            video_expert=video_expert,
            action_expert=nav_action_expert,
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
            loss_lambda_action=loss_lambda_nav_action,
        )
        self.nav_action_expert = nav_action_expert
        self.manip_action_expert = manip_action_expert
        self.loss_lambda_nav_action = float(loss_lambda_nav_action)
        self.loss_lambda_manip_action = float(loss_lambda_manip_action)
        self.attention_partition_mode = str(attention_partition_mode)

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
        nav_action_dit_config: dict[str, Any] | None = None,
        manip_action_dit_config: dict[str, Any] | None = None,
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
        attention_partition_mode: str = "robotwin_top_nav",
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for FastWAMDual.")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for FastWAMDual.")
        if nav_action_dit_config is None or manip_action_dit_config is None:
            raise ValueError("Both nav_action_dit_config and manip_action_dit_config are required.")

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
        video_expert = components.dit
        nav_action_expert = ActionDiT.from_pretrained(
            action_dit_config=nav_action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        manip_action_expert = ActionDiT.from_pretrained(
            action_dit_config=manip_action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        for name, expert in (("nav", nav_action_expert), ("manip", manip_action_expert)):
            if int(expert.num_heads) != int(video_expert.num_heads):
                raise ValueError(f"{name} ActionDiT `num_heads` must match video expert.")
            if int(expert.attn_head_dim) != int(video_expert.attn_head_dim):
                raise ValueError(f"{name} ActionDiT `attn_head_dim` must match video expert.")
            if int(len(expert.blocks)) != int(len(video_expert.blocks)):
                raise ValueError(f"{name} ActionDiT `num_layers` must match video expert.")

        mot = MoT(
            mixtures={"video": video_expert, "action": nav_action_expert, "manip": manip_action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )
        model = cls(
            video_expert=video_expert,
            nav_action_expert=nav_action_expert,
            manip_action_expert=manip_action_expert,
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
            attention_partition_mode=attention_partition_mode,
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

    @torch.no_grad()
    def _build_dual_mot_attention_mask(
        self,
        video_seq_len: int,
        nav_seq_len: int,
        manip_seq_len: int,
        video_tokens_per_frame: int,
        video_grid_size: tuple[int, int, int],
        device: torch.device,
    ) -> torch.Tensor:
        if video_seq_len <= 0:
            raise ValueError(f"`video_seq_len` must be positive, got {video_seq_len}")
        if video_tokens_per_frame <= 0:
            raise ValueError(f"`video_tokens_per_frame` must be positive, got {video_tokens_per_frame}")
        if video_seq_len % video_tokens_per_frame != 0:
            raise ValueError(
                "`video_seq_len` must be divisible by `video_tokens_per_frame`, "
                f"got video_seq_len={video_seq_len}, video_tokens_per_frame={video_tokens_per_frame}"
            )

        total = video_seq_len + nav_seq_len + manip_seq_len
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)

        video_end = video_seq_len
        nav_start = video_end
        nav_end = nav_start + nav_seq_len
        manip_start = nav_end
        manip_end = manip_start + manip_seq_len

        mask[:video_end, :video_end] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        # Video tokens are frame-major, then row-major within each frame.
        if len(video_grid_size) != 3:
            raise ValueError(f"`video_grid_size` must be (frames, height, width), got {video_grid_size}")
        _, token_h, token_w = [int(x) for x in video_grid_size]
        if token_h <= 0 or token_w <= 0 or token_h * token_w != video_tokens_per_frame:
            raise ValueError(
                "Unsupported first-frame token grid for partitioned attention: "
                f"video_grid_size={video_grid_size}, video_tokens_per_frame={video_tokens_per_frame}"
            )

        partition_mode = getattr(self, "attention_partition_mode", "robotwin_top_nav")
        nav_first_frame_mask = torch.zeros(first_frame_tokens, dtype=torch.bool, device=device)
        manip_first_frame_mask = torch.zeros(first_frame_tokens, dtype=torch.bool, device=device)
        if partition_mode == "robotwin_top_nav":
            if token_h % 3 != 0:
                raise ValueError(
                    "Robotwin partition requires token grid height divisible by 3, "
                    f"got token_h={token_h}."
                )
            nav_token_h = (token_h * 2) // 3
            nav_first_frame_mask[: nav_token_h * token_w] = True
            manip_first_frame_mask[:] = True
        elif partition_mode == "dual_canvas_nav_left_manip_right":
            if token_w % 2 != 0:
                raise ValueError(
                    "Dual-canvas partition requires token grid width divisible by 2, "
                    f"got token_w={token_w}."
                )
            mid_w = token_w // 2
            grid = torch.zeros((token_h, token_w), dtype=torch.bool, device=device)
            nav_grid = grid.clone()
            manip_grid = grid.clone()
            nav_grid[:, :mid_w] = True
            manip_grid[:, mid_w:] = True
            nav_first_frame_mask[:] = nav_grid.reshape(-1)
            manip_first_frame_mask[:] = manip_grid.reshape(-1)
        else:
            raise ValueError(f"Unsupported attention_partition_mode: {partition_mode}")

        mask[nav_start:nav_end, :first_frame_tokens] = nav_first_frame_mask.unsqueeze(0).expand(
            nav_seq_len, -1
        )
        mask[nav_start:nav_end, nav_start:nav_end] = True
        mask[manip_start:manip_end, :first_frame_tokens] = manip_first_frame_mask.unsqueeze(0).expand(
            manip_seq_len, -1
        )
        mask[manip_start:manip_end, manip_start:manip_end] = True
        return mask


    @torch.no_grad()
    def _build_parallel_mot_attention_mask(
        self,
        nav_video_seq_len: int,
        manip_video_seq_len: int,
        nav_seq_len: int,
        manip_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        if nav_video_seq_len <= 0 or manip_video_seq_len <= 0:
            raise ValueError(
                f"Video seq lens must be positive, got nav={nav_video_seq_len}, manip={manip_video_seq_len}"
            )
        if nav_video_seq_len != manip_video_seq_len:
            raise ValueError(
                f"Parallel streams must have same video seq len, got {nav_video_seq_len} and {manip_video_seq_len}."
            )
        if video_tokens_per_frame <= 0:
            raise ValueError(f"`video_tokens_per_frame` must be positive, got {video_tokens_per_frame}")
        total = nav_video_seq_len + manip_video_seq_len + nav_seq_len + manip_seq_len
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)

        nav_video_start = 0
        nav_video_end = nav_video_seq_len
        manip_video_start = nav_video_end
        manip_video_end = manip_video_start + manip_video_seq_len
        nav_start = manip_video_end
        nav_end = nav_start + nav_seq_len
        manip_start = nav_end
        manip_end = manip_start + manip_seq_len

        video_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=nav_video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        mask[nav_video_start:nav_video_end, nav_video_start:nav_video_end] = video_mask
        mask[manip_video_start:manip_video_end, manip_video_start:manip_video_end] = video_mask

        first_frame_tokens = min(video_tokens_per_frame, nav_video_seq_len)
        mask[nav_start:nav_end, nav_video_start : nav_video_start + first_frame_tokens] = True
        mask[nav_start:nav_end, nav_start:nav_end] = True
        mask[manip_start:manip_end, manip_video_start : manip_video_start + first_frame_tokens] = True
        mask[manip_start:manip_end, manip_start:manip_end] = True
        return mask

    def _validate_video_tensor(self, name: str, video: torch.Tensor) -> tuple[int, int, int, int, int]:
        if video.ndim != 5:
            raise ValueError(f"`sample[{name}]` must be 5D [B,3,T,H,W], got {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`sample[{name}]` channel dimension must be 3, got {tuple(video.shape)}")
        batch_size, channels, num_frames, height, width = video.shape
        del channels
        if height % 32 != 0 or width % 32 != 0:
            raise ValueError(f"Video spatial dims must be multiples of 32, got H={height}, W={width}")
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        return batch_size, 3, num_frames, height, width

    @staticmethod
    def _split_batched_video_pre(pre_state: dict, batch_size: int) -> tuple[dict, dict]:
        batch_keys = {"tokens", "t", "t_mod", "context", "context_mask"}

        def split(start: int, end: int) -> dict:
            result = {}
            for key, value in pre_state.items():
                if key == "meta":
                    meta = dict(value)
                    meta["batch_size"] = end - start
                    result[key] = meta
                elif key in batch_keys:
                    result[key] = value[start:end]
                else:
                    result[key] = value
            return result

        return split(0, batch_size), split(batch_size, batch_size * 2)

    def build_inputs(self, sample, tiled: bool = False):
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)
        nav_action = sample["nav_action"]
        manip_action = sample["manip_action"]

        parallel_video = "nav_video" in sample and "manip_video" in sample
        if parallel_video:
            nav_video = sample["nav_video"]
            manip_video = sample["manip_video"]
            batch_size, _, num_frames, _, _ = self._validate_video_tensor("nav_video", nav_video)
            manip_shape = self._validate_video_tensor("manip_video", manip_video)
            if manip_shape[0] != batch_size or manip_shape[2] != num_frames:
                raise ValueError(
                    "Parallel video streams must have identical batch and time dimensions, got "
                    f"nav={tuple(nav_video.shape)} and manip={tuple(manip_video.shape)}"
                )
        else:
            video = sample["video"]
            batch_size, _, num_frames, height, width = self._validate_video_tensor("video", video)

        if nav_action.ndim != 3 or nav_action.shape[-1] != self.nav_action_expert.action_dim:
            raise ValueError(f"`nav_action` shape mismatch: {tuple(nav_action.shape)}")
        if manip_action.ndim != 3 or manip_action.shape[-1] != self.manip_action_expert.action_dim:
            raise ValueError(f"`manip_action` shape mismatch: {tuple(manip_action.shape)}")
        if nav_action.shape[1] % (num_frames - 1) != 0:
            raise ValueError("nav_action temporal dimension must be divisible by video transitions.")
        if manip_action.shape[1] != nav_action.shape[1]:
            raise ValueError("nav_action and manip_action must have same horizon.")

        if parallel_video:
            nav_video = nav_video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            manip_video = manip_video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            if nav_video.shape == manip_video.shape:
                paired_latents = self._encode_video_latents(
                    torch.cat([nav_video, manip_video], dim=0),
                    tiled=tiled,
                )
                nav_input_latents, manip_input_latents = paired_latents.chunk(2, dim=0)
            else:
                nav_input_latents = self._encode_video_latents(nav_video, tiled=tiled)
                manip_input_latents = self._encode_video_latents(manip_video, tiled=tiled)
            first_frame_latents = None
            nav_first_frame_latents = None
            manip_first_frame_latents = None
            fuse_flag = False
            if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
                nav_first_frame_latents = nav_input_latents[:, :, 0:1]
                manip_first_frame_latents = manip_input_latents[:, :, 0:1]
                fuse_flag = True
        else:
            input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            input_latents = self._encode_video_latents(input_video, tiled=tiled)
            first_frame_latents = None
            fuse_flag = False
            if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
                first_frame_latents = input_latents[:, :, 0:1]
                fuse_flag = True

        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample[proprio]` is required when proprio_dim is enabled.")
            if proprio.ndim != 3 or proprio.shape[2] != self.proprio_dim:
                raise ValueError(f"`proprio` shape mismatch: {tuple(proprio.shape)}")
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio[:, 0, :].to(device=self.device, dtype=self.torch_dtype),
            )

        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        image_is_pad = sample.get("image_is_pad")
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        out = {
            "parallel_video": parallel_video,
            "context": context,
            "context_mask": context_mask,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "nav_action": nav_action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
            "manip_action": manip_action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
            "nav_loss_valid": sample.get("nav_loss_valid"),
            "manip_loss_valid": sample.get("manip_loss_valid"),
        }
        if parallel_video:
            out.update(
                {
                    "nav_input_latents": nav_input_latents,
                    "manip_input_latents": manip_input_latents,
                    "nav_first_frame_latents": nav_first_frame_latents,
                    "manip_first_frame_latents": manip_first_frame_latents,
                }
            )
        else:
            out.update(
                {
                    "input_latents": input_latents,
                    "first_frame_latents": first_frame_latents,
                }
            )
        return out

    def _compute_action_loss(
        self,
        *,
        pred_action: torch.Tensor,
        target_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        loss_valid: Optional[torch.Tensor],
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        token_loss = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
        if action_is_pad is None:
            valid_time = torch.ones_like(token_loss)
        else:
            valid_time = (~action_is_pad).to(device=token_loss.device, dtype=token_loss.dtype)

        if loss_valid is None:
            active = torch.ones((token_loss.shape[0],), device=token_loss.device, dtype=token_loss.dtype)
        else:
            active = loss_valid.to(device=token_loss.device, dtype=torch.bool).to(dtype=token_loss.dtype)

        valid_sum = valid_time.sum(dim=1).clamp(min=1.0)
        per_sample = (token_loss * valid_time).sum(dim=1) / valid_sum
        weight = self.train_action_scheduler.training_weight(timestep).to(
            device=per_sample.device,
            dtype=per_sample.dtype,
        )
        weighted = per_sample * weight * active
        return weighted.sum() / active.sum().clamp(min=1.0)


    def _parallel_mot_forward(
        self,
        *,
        nav_video_pre: dict,
        manip_video_pre: dict,
        nav_pre: dict,
        manip_pre: dict,
    ) -> dict[str, torch.Tensor]:
        """Run the dual-stream MoT without dense cross-stream masked attention.

        This is mathematically equivalent to the block-diagonal parallel mask: nav
        video and manip video never attend to each other, and each action expert
        attends only to its own first-frame video tokens plus its own action tokens.
        The two video streams are batched together to keep the shared Wan expert
        GPU-friendly.
        """
        nav_video_tokens = nav_video_pre["tokens"]
        manip_video_tokens = manip_video_pre["tokens"]
        nav_action_tokens = nav_pre["tokens"]
        manip_action_tokens = manip_pre["tokens"]

        if nav_video_tokens.shape[0] != manip_video_tokens.shape[0]:
            raise ValueError(
                f"Parallel video batch mismatch: nav={tuple(nav_video_tokens.shape)} "
                f"manip={tuple(manip_video_tokens.shape)}"
            )
        nav_video_seq_len = int(nav_video_tokens.shape[1])
        manip_video_seq_len = int(manip_video_tokens.shape[1])
        nav_video_tokens_per_frame = int(nav_video_pre["meta"]["tokens_per_frame"])
        manip_video_tokens_per_frame = int(manip_video_pre["meta"]["tokens_per_frame"])
        nav_first_frame_tokens = min(nav_video_tokens_per_frame, nav_video_seq_len)
        manip_first_frame_tokens = min(manip_video_tokens_per_frame, manip_video_seq_len)
        device = nav_video_tokens.device

        nav_video_attention_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=nav_video_seq_len,
            video_tokens_per_frame=nav_video_tokens_per_frame,
            device=device,
        )
        manip_video_attention_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=manip_video_seq_len,
            video_tokens_per_frame=manip_video_tokens_per_frame,
            device=device,
        )
        nav_action_attention_mask = torch.ones(
            (nav_action_tokens.shape[1], nav_first_frame_tokens + nav_action_tokens.shape[1]),
            dtype=torch.bool,
            device=device,
        )
        manip_action_attention_mask = torch.ones(
            (manip_action_tokens.shape[1], manip_first_frame_tokens + manip_action_tokens.shape[1]),
            dtype=torch.bool,
            device=device,
        )

        nav_video_context_payload = {
            "context": nav_video_pre["context"],
            "mask": nav_video_pre["context_mask"],
        }
        manip_video_context_payload = {
            "context": manip_video_pre["context"],
            "mask": manip_video_pre["context_mask"],
        }
        nav_context_payload = {"context": nav_pre["context"], "mask": nav_pre["context_mask"]}
        manip_context_payload = {"context": manip_pre["context"], "mask": manip_pre["context_mask"]}
        batch_video_streams = (
            nav_video_tokens.shape == manip_video_tokens.shape
            and nav_video_tokens_per_frame == manip_video_tokens_per_frame
        )

        for layer_idx in range(self.mot.num_layers):
            video_block = self.video_expert.blocks[layer_idx]
            if batch_video_streams:
                video_tokens_batched = torch.cat([nav_video_tokens, manip_video_tokens], dim=0)
                (
                    q_video,
                    k_video,
                    v_video,
                    residual_video,
                    gate_msa_video,
                    shift_mlp_video,
                    scale_mlp_video,
                    gate_mlp_video,
                    video_use_checkpoint,
                ) = self.mot._build_expert_attention_io(
                    expert=self.video_expert,
                    block=video_block,
                    x=video_tokens_batched,
                    freqs=nav_video_pre["freqs"],
                    t_mod=torch.cat([nav_video_pre["t_mod"], manip_video_pre["t_mod"]], dim=0),
                )
                mixed_video = self.mot._mixed_attention(
                    q_cat=q_video,
                    k_cat=k_video,
                    v_cat=v_video,
                    attention_mask=nav_video_attention_mask,
                )
                video_tokens_batched = self.mot._apply_post_with_optional_checkpoint(
                    block=video_block,
                    residual_x=residual_video,
                    gate_msa=gate_msa_video,
                    shift_mlp=shift_mlp_video,
                    scale_mlp=scale_mlp_video,
                    gate_mlp=gate_mlp_video,
                    use_gradient_checkpointing=video_use_checkpoint,
                    mixed_slice=mixed_video,
                    context_payload={
                        "context": torch.cat(
                            [nav_video_pre["context"], manip_video_pre["context"]], dim=0
                        ),
                        "mask": torch.cat(
                            [nav_video_pre["context_mask"], manip_video_pre["context_mask"]], dim=0
                        ),
                    },
                )
                nav_video_tokens, manip_video_tokens = video_tokens_batched.chunk(2, dim=0)
                k_nav_video, k_manip_video = k_video.chunk(2, dim=0)
                v_nav_video, v_manip_video = v_video.chunk(2, dim=0)
            else:
                nav_video_io = self.mot._build_expert_attention_io(
                    expert=self.video_expert,
                    block=video_block,
                    x=nav_video_tokens,
                    freqs=nav_video_pre["freqs"],
                    t_mod=nav_video_pre["t_mod"],
                )
                manip_video_io = self.mot._build_expert_attention_io(
                    expert=self.video_expert,
                    block=video_block,
                    x=manip_video_tokens,
                    freqs=manip_video_pre["freqs"],
                    t_mod=manip_video_pre["t_mod"],
                )
                (
                    q_nav_video,
                    k_nav_video,
                    v_nav_video,
                    residual_nav_video,
                    gate_msa_nav_video,
                    shift_mlp_nav_video,
                    scale_mlp_nav_video,
                    gate_mlp_nav_video,
                    nav_video_use_checkpoint,
                ) = nav_video_io
                (
                    q_manip_video,
                    k_manip_video,
                    v_manip_video,
                    residual_manip_video,
                    gate_msa_manip_video,
                    shift_mlp_manip_video,
                    scale_mlp_manip_video,
                    gate_mlp_manip_video,
                    manip_video_use_checkpoint,
                ) = manip_video_io
                mixed_nav_video = self.mot._mixed_attention(
                    q_cat=q_nav_video,
                    k_cat=k_nav_video,
                    v_cat=v_nav_video,
                    attention_mask=nav_video_attention_mask,
                )
                mixed_manip_video = self.mot._mixed_attention(
                    q_cat=q_manip_video,
                    k_cat=k_manip_video,
                    v_cat=v_manip_video,
                    attention_mask=manip_video_attention_mask,
                )
                nav_video_tokens = self.mot._apply_post_with_optional_checkpoint(
                    block=video_block,
                    residual_x=residual_nav_video,
                    gate_msa=gate_msa_nav_video,
                    shift_mlp=shift_mlp_nav_video,
                    scale_mlp=scale_mlp_nav_video,
                    gate_mlp=gate_mlp_nav_video,
                    use_gradient_checkpointing=nav_video_use_checkpoint,
                    mixed_slice=mixed_nav_video,
                    context_payload=nav_video_context_payload,
                )
                manip_video_tokens = self.mot._apply_post_with_optional_checkpoint(
                    block=video_block,
                    residual_x=residual_manip_video,
                    gate_msa=gate_msa_manip_video,
                    shift_mlp=shift_mlp_manip_video,
                    scale_mlp=scale_mlp_manip_video,
                    gate_mlp=gate_mlp_manip_video,
                    use_gradient_checkpointing=manip_video_use_checkpoint,
                    mixed_slice=mixed_manip_video,
                    context_payload=manip_video_context_payload,
                )

            nav_block = self.nav_action_expert.blocks[layer_idx]
            (
                q_nav,
                k_nav,
                v_nav,
                residual_nav,
                gate_msa_nav,
                shift_mlp_nav,
                scale_mlp_nav,
                gate_mlp_nav,
                nav_use_checkpoint,
            ) = self.mot._build_expert_attention_io(
                expert=self.nav_action_expert,
                block=nav_block,
                x=nav_action_tokens,
                freqs=nav_pre["freqs"],
                t_mod=nav_pre["t_mod"],
            )
            mixed_nav = self.mot._mixed_attention(
                q_cat=q_nav,
                k_cat=torch.cat([k_nav_video[:, :nav_first_frame_tokens], k_nav], dim=1),
                v_cat=torch.cat([v_nav_video[:, :nav_first_frame_tokens], v_nav], dim=1),
                attention_mask=nav_action_attention_mask,
            )
            nav_action_tokens = self.mot._apply_post_with_optional_checkpoint(
                block=nav_block,
                residual_x=residual_nav,
                gate_msa=gate_msa_nav,
                shift_mlp=shift_mlp_nav,
                scale_mlp=scale_mlp_nav,
                gate_mlp=gate_mlp_nav,
                use_gradient_checkpointing=nav_use_checkpoint,
                mixed_slice=mixed_nav,
                context_payload=nav_context_payload,
            )

            manip_block = self.manip_action_expert.blocks[layer_idx]
            (
                q_manip,
                k_manip,
                v_manip,
                residual_manip,
                gate_msa_manip,
                shift_mlp_manip,
                scale_mlp_manip,
                gate_mlp_manip,
                manip_use_checkpoint,
            ) = self.mot._build_expert_attention_io(
                expert=self.manip_action_expert,
                block=manip_block,
                x=manip_action_tokens,
                freqs=manip_pre["freqs"],
                t_mod=manip_pre["t_mod"],
            )
            mixed_manip = self.mot._mixed_attention(
                q_cat=q_manip,
                k_cat=torch.cat([k_manip_video[:, :manip_first_frame_tokens], k_manip], dim=1),
                v_cat=torch.cat([v_manip_video[:, :manip_first_frame_tokens], v_manip], dim=1),
                attention_mask=manip_action_attention_mask,
            )
            manip_action_tokens = self.mot._apply_post_with_optional_checkpoint(
                block=manip_block,
                residual_x=residual_manip,
                gate_msa=gate_msa_manip,
                shift_mlp=shift_mlp_manip,
                scale_mlp=scale_mlp_manip,
                gate_mlp=gate_mlp_manip,
                use_gradient_checkpointing=manip_use_checkpoint,
                mixed_slice=mixed_manip,
                context_payload=manip_context_payload,
            )

        return {
            "video": torch.cat([nav_video_tokens, manip_video_tokens], dim=1),
            "action": nav_action_tokens,
            "manip": manip_action_tokens,
        }

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        if not inputs.get("parallel_video", False):
            input_latents = inputs["input_latents"]
            batch_size = input_latents.shape[0]
            context = inputs["context"]
            context_mask = inputs["context_mask"]
            nav_action = inputs["nav_action"]
            manip_action = inputs["manip_action"]

            noise_video = torch.randn_like(input_latents)
            timestep_video = self.train_video_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=input_latents.dtype,
            )
            latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
            target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)
            if inputs["first_frame_latents"] is not None:
                latents[:, :, 0:1] = inputs["first_frame_latents"]

            noise_nav = torch.randn_like(nav_action)
            timestep_nav = self.train_action_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=nav_action.dtype,
            )
            noisy_nav = self.train_action_scheduler.add_noise(nav_action, noise_nav, timestep_nav)
            target_nav = self.train_action_scheduler.training_target(nav_action, noise_nav, timestep_nav)

            noise_manip = torch.randn_like(manip_action)
            timestep_manip = self.train_action_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=manip_action.dtype,
            )
            noisy_manip = self.train_action_scheduler.add_noise(manip_action, noise_manip, timestep_manip)
            target_manip = self.train_action_scheduler.training_target(manip_action, noise_manip, timestep_manip)

            video_pre = self.video_expert.pre_dit(
                x=latents,
                timestep=timestep_video,
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
            )
            nav_pre = self.nav_action_expert.pre_dit(
                action_tokens=noisy_nav,
                timestep=timestep_nav,
                context=context,
                context_mask=context_mask,
            )
            manip_pre = self.manip_action_expert.pre_dit(
                action_tokens=noisy_manip,
                timestep=timestep_manip,
                context=context,
                context_mask=context_mask,
            )

            attention_mask = self._build_dual_mot_attention_mask(
                video_seq_len=video_pre["tokens"].shape[1],
                nav_seq_len=nav_pre["tokens"].shape[1],
                manip_seq_len=manip_pre["tokens"].shape[1],
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                video_grid_size=video_pre["meta"]["grid_size"],
                device=video_pre["tokens"].device,
            )
            tokens_out = self.mot(
                embeds_all={
                    "video": video_pre["tokens"],
                    "action": nav_pre["tokens"],
                    "manip": manip_pre["tokens"],
                },
                attention_mask=attention_mask,
                freqs_all={
                    "video": video_pre["freqs"],
                    "action": nav_pre["freqs"],
                    "manip": manip_pre["freqs"],
                },
                context_all={
                    "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                    "action": {"context": nav_pre["context"], "mask": nav_pre["context_mask"]},
                    "manip": {"context": manip_pre["context"], "mask": manip_pre["context_mask"]},
                },
                t_mod_all={
                    "video": video_pre["t_mod"],
                    "action": nav_pre["t_mod"],
                    "manip": manip_pre["t_mod"],
                },
            )

            pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
            pred_nav = self.nav_action_expert.post_dit(tokens_out["action"], nav_pre)
            pred_manip = self.manip_action_expert.post_dit(tokens_out["manip"], manip_pre)

            include_initial_video_step = inputs["first_frame_latents"] is None
            if inputs["first_frame_latents"] is not None:
                pred_video = pred_video[:, :, 1:]
                target_video = target_video[:, :, 1:]
            loss_video_per_sample = self._compute_video_loss_per_sample(
                pred_video=pred_video,
                target_video=target_video,
                image_is_pad=inputs["image_is_pad"],
                include_initial_video_step=include_initial_video_step,
            )
            video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
                loss_video_per_sample.device,
                dtype=loss_video_per_sample.dtype,
            )
            loss_video = (loss_video_per_sample * video_weight).mean()
            loss_video_nav = None
            loss_video_manip = None
        else:
            nav_input_latents = inputs["nav_input_latents"]
            manip_input_latents = inputs["manip_input_latents"]
            batch_size = nav_input_latents.shape[0]
            context = inputs["context"]
            context_mask = inputs["context_mask"]
            nav_action = inputs["nav_action"]
            manip_action = inputs["manip_action"]

            noise_nav_video = torch.randn_like(nav_input_latents)
            noise_manip_video = torch.randn_like(manip_input_latents)
            timestep_nav_video = self.train_video_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=nav_input_latents.dtype,
            )
            timestep_manip_video = self.train_video_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=manip_input_latents.dtype,
            )
            nav_latents = self.train_video_scheduler.add_noise(
                nav_input_latents,
                noise_nav_video,
                timestep_nav_video,
            )
            manip_latents = self.train_video_scheduler.add_noise(
                manip_input_latents,
                noise_manip_video,
                timestep_manip_video,
            )
            target_nav_video = self.train_video_scheduler.training_target(
                nav_input_latents,
                noise_nav_video,
                timestep_nav_video,
            )
            target_manip_video = self.train_video_scheduler.training_target(
                manip_input_latents,
                noise_manip_video,
                timestep_manip_video,
            )
            if inputs["nav_first_frame_latents"] is not None:
                nav_latents[:, :, 0:1] = inputs["nav_first_frame_latents"]
            if inputs["manip_first_frame_latents"] is not None:
                manip_latents[:, :, 0:1] = inputs["manip_first_frame_latents"]

            noise_nav = torch.randn_like(nav_action)
            timestep_nav = self.train_action_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=nav_action.dtype,
            )
            noisy_nav = self.train_action_scheduler.add_noise(nav_action, noise_nav, timestep_nav)
            target_nav = self.train_action_scheduler.training_target(nav_action, noise_nav, timestep_nav)

            noise_manip = torch.randn_like(manip_action)
            timestep_manip = self.train_action_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=manip_action.dtype,
            )
            noisy_manip = self.train_action_scheduler.add_noise(manip_action, noise_manip, timestep_manip)
            target_manip = self.train_action_scheduler.training_target(manip_action, noise_manip, timestep_manip)

            if nav_latents.shape == manip_latents.shape:
                paired_video_pre = self.video_expert.pre_dit(
                    x=torch.cat([nav_latents, manip_latents], dim=0),
                    timestep=torch.cat([timestep_nav_video, timestep_manip_video], dim=0),
                    context=torch.cat([context, context], dim=0),
                    context_mask=torch.cat([context_mask, context_mask], dim=0),
                    action=None,
                    fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
                )
                nav_video_pre, manip_video_pre = self._split_batched_video_pre(
                    paired_video_pre,
                    batch_size,
                )
            else:
                nav_video_pre = self.video_expert.pre_dit(
                    x=nav_latents,
                    timestep=timestep_nav_video,
                    context=context,
                    context_mask=context_mask,
                    action=None,
                    fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
                )
                manip_video_pre = self.video_expert.pre_dit(
                    x=manip_latents,
                    timestep=timestep_manip_video,
                    context=context,
                    context_mask=context_mask,
                    action=None,
                    fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
                )

            nav_pre = self.nav_action_expert.pre_dit(
                action_tokens=noisy_nav,
                timestep=timestep_nav,
                context=context,
                context_mask=context_mask,
            )
            manip_pre = self.manip_action_expert.pre_dit(
                action_tokens=noisy_manip,
                timestep=timestep_manip,
                context=context,
                context_mask=context_mask,
            )

            tokens_out = self._parallel_mot_forward(
                nav_video_pre=nav_video_pre,
                manip_video_pre=manip_video_pre,
                nav_pre=nav_pre,
                manip_pre=manip_pre,
            )

            nav_video_seq_len = nav_video_pre["tokens"].shape[1]
            pred_nav_video = self.video_expert.post_dit(tokens_out["video"][:, :nav_video_seq_len], nav_video_pre)
            pred_manip_video = self.video_expert.post_dit(tokens_out["video"][:, nav_video_seq_len:], manip_video_pre)
            pred_nav = self.nav_action_expert.post_dit(tokens_out["action"], nav_pre)
            pred_manip = self.manip_action_expert.post_dit(tokens_out["manip"], manip_pre)

            include_initial_video_step = inputs["nav_first_frame_latents"] is None
            if inputs["nav_first_frame_latents"] is not None:
                pred_nav_video = pred_nav_video[:, :, 1:]
                target_nav_video = target_nav_video[:, :, 1:]
            if inputs["manip_first_frame_latents"] is not None:
                pred_manip_video = pred_manip_video[:, :, 1:]
                target_manip_video = target_manip_video[:, :, 1:]
            loss_nav_video_per_sample = self._compute_video_loss_per_sample(
                pred_video=pred_nav_video,
                target_video=target_nav_video,
                image_is_pad=inputs["image_is_pad"],
                include_initial_video_step=include_initial_video_step,
            )
            loss_manip_video_per_sample = self._compute_video_loss_per_sample(
                pred_video=pred_manip_video,
                target_video=target_manip_video,
                image_is_pad=inputs["image_is_pad"],
                include_initial_video_step=include_initial_video_step,
            )
            nav_video_weight = self.train_video_scheduler.training_weight(timestep_nav_video).to(
                loss_nav_video_per_sample.device,
                dtype=loss_nav_video_per_sample.dtype,
            )
            manip_video_weight = self.train_video_scheduler.training_weight(timestep_manip_video).to(
                loss_manip_video_per_sample.device,
                dtype=loss_manip_video_per_sample.dtype,
            )
            loss_video_nav = (loss_nav_video_per_sample * nav_video_weight).mean()
            loss_video_manip = (loss_manip_video_per_sample * manip_video_weight).mean()
            loss_video = 0.5 * (loss_video_nav + loss_video_manip)

        loss_nav = self._compute_action_loss(
            pred_action=pred_nav,
            target_action=target_nav,
            action_is_pad=inputs["action_is_pad"],
            loss_valid=inputs["nav_loss_valid"],
            timestep=timestep_nav,
        )
        loss_manip = self._compute_action_loss(
            pred_action=pred_manip,
            target_action=target_manip,
            action_is_pad=inputs["action_is_pad"],
            loss_valid=inputs["manip_loss_valid"],
            timestep=timestep_manip,
        )
        loss_total = (
            self.loss_lambda_video * loss_video
            + self.loss_lambda_nav_action * loss_nav
            + self.loss_lambda_manip_action * loss_manip
        )
        metrics = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action_nav": self.loss_lambda_nav_action * float(loss_nav.detach().item()),
            "loss_action_manip": self.loss_lambda_manip_action * float(loss_manip.detach().item()),
        }
        if loss_video_nav is not None and loss_video_manip is not None:
            metrics["loss_video_nav"] = self.loss_lambda_video * float(loss_video_nav.detach().item())
            metrics["loss_video_manip"] = self.loss_lambda_video * float(loss_video_manip.detach().item())
        return loss_total, metrics

    @torch.no_grad()
    def _predict_dual_action_noise_with_cache(
        self,
        *,
        expert: ActionDiT,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        action_pre = expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        if len(video_kv_cache) != self.mot.num_layers:
            raise ValueError(
                f"`video_kv_cache` must contain {self.mot.num_layers} layers, got {len(video_kv_cache)}."
            )
        action_tokens = action_pre["tokens"]
        action_seq_len = int(action_tokens.shape[1])
        total_seq_len = int(video_seq_len) + action_seq_len
        if attention_mask.ndim != 2 or attention_mask.shape[0] != total_seq_len:
            raise ValueError(
                "Dual action attention mask shape mismatch: "
                f"mask={tuple(attention_mask.shape)} expected=({total_seq_len},{total_seq_len})"
            )
        action_attention_mask = attention_mask[video_seq_len:total_seq_len, :total_seq_len]
        action_context_payload = {"context": action_pre["context"], "mask": action_pre["context_mask"]}

        x = action_tokens
        for layer_idx in range(self.mot.num_layers):
            block = expert.blocks[layer_idx]
            (
                q_action,
                k_action,
                v_action,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self.mot._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=action_pre["freqs"],
                t_mod=action_pre["t_mod"],
            )
            k_video = video_kv_cache[layer_idx]["k"]
            v_video = video_kv_cache[layer_idx]["v"]
            if k_video.shape[1] != video_seq_len or v_video.shape[1] != video_seq_len:
                raise ValueError(f"video cache seq len mismatch at layer {layer_idx}.")
            mixed = self.mot._mixed_attention(
                q_cat=q_action,
                k_cat=torch.cat([k_video, k_action], dim=1),
                v_cat=torch.cat([v_video, v_action], dim=1),
                attention_mask=action_attention_mask,
            )
            x = self.mot._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=action_context_payload,
            )
        return expert.post_dit(x, action_pre)

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str] = None,
        input_image: Optional[torch.Tensor] = None,
        action_horizon: int = 32,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        nav_input_image: Optional[torch.Tensor] = None,
        manip_input_image: Optional[torch.Tensor] = None,
        action_prefix_nav: Optional[torch.Tensor] = None,
        action_prefix_manip: Optional[torch.Tensor] = None,
        prefix_length: int = 0,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        _ = negative_prompt, text_cfg_scale
        self.eval()
        profile_enabled = (
            os.environ.get("FASTWAM_PROFILE_DUAL_INFER", "0") == "1"
            and torch.cuda.is_available()
            and str(self.device).startswith("cuda")
        )
        profile_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}

        def _profile_start(name: str) -> Optional[torch.cuda.Event]:
            if not profile_enabled:
                return None
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event

        def _profile_end(name: str, start: Optional[torch.cuda.Event]) -> None:
            if start is None:
                return
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            profile_events.setdefault(name, []).append((start, end))

        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError("`infer_action` requires `video_attention_mask_mode='first_frame_causal'`.")
        if nav_input_image is None:
            nav_input_image = input_image
        if manip_input_image is None:
            manip_input_image = input_image
        if nav_input_image is None or manip_input_image is None:
            raise ValueError("Dual inference requires `nav_input_image` and `manip_input_image`.")

        def _prep_image(name: str, image: torch.Tensor) -> torch.Tensor:
            if image.ndim == 3:
                image = image.unsqueeze(0)
            if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
                raise ValueError(f"`{name}` must be [1,3,H,W] or [3,H,W], got {tuple(image.shape)}")
            _, _, height, width = image.shape
            if height % 32 != 0 or width % 32 != 0:
                raise ValueError(f"`{name}` H/W must be multiples of 32, got {height}x{width}")
            return image.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        nav_input_image = _prep_image("nav_input_image", nav_input_image)
        manip_input_image = _prep_image("manip_input_image", manip_input_image)

        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None`.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")
        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(context=context, context_mask=context_mask, proprio=proprio)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_nav = torch.randn(
            (1, action_horizon, self.nav_action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_manip = torch.randn(
            (1, action_horizon, self.manip_action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        prefix_length = int(max(0, min(prefix_length, action_horizon)))

        def _prep_prefix(prefix: Optional[torch.Tensor], target: torch.Tensor, name: str) -> Optional[torch.Tensor]:
            if prefix is None or prefix_length <= 0:
                return None
            if prefix.ndim == 2:
                prefix = prefix.unsqueeze(0)
            if prefix.ndim != 3 or prefix.shape[0] != 1 or prefix.shape[-1] != target.shape[-1]:
                raise ValueError(f"`{name}` must be [P,D] or [1,P,D], got {tuple(prefix.shape)}")
            if prefix.shape[1] < prefix_length:
                raise ValueError(f"`{name}` has {prefix.shape[1]} steps but prefix_length={prefix_length}")
            return prefix[:, :prefix_length].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        prefix_nav = _prep_prefix(action_prefix_nav, latents_nav, "action_prefix_nav")
        prefix_manip = _prep_prefix(action_prefix_manip, latents_manip, "action_prefix_manip")
        if prefix_nav is not None:
            latents_nav[:, :prefix_length] = prefix_nav
        if prefix_manip is not None:
            latents_manip[:, :prefix_length] = prefix_manip

        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
        profiled_total_start = _profile_start("profiled_total")
        if nav_input_image.shape == manip_input_image.shape:
            stage_start = _profile_start("vae_encode_2b")
            paired_video_latents = self._encode_video_latents(
                torch.cat([nav_input_image, manip_input_image], dim=0).unsqueeze(2),
                tiled=tiled,
            )
            _profile_end("vae_encode_2b", stage_start)
            stage_start = _profile_start("video_pre_dit_2b")
            paired_video_pre = self.video_expert.pre_dit(
                x=paired_video_latents,
                timestep=torch.zeros((2,), dtype=paired_video_latents.dtype, device=self.device),
                context=torch.cat([context, context], dim=0),
                context_mask=torch.cat([context_mask, context_mask], dim=0),
                action=None,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            _profile_end("video_pre_dit_2b", stage_start)
            nav_video_pre, manip_video_pre = self._split_batched_video_pre(paired_video_pre, 1)
        else:
            stage_start = _profile_start("vae_encode_nav")
            nav_video_latents = self._encode_video_latents(nav_input_image.unsqueeze(2), tiled=tiled)
            _profile_end("vae_encode_nav", stage_start)
            stage_start = _profile_start("vae_encode_manip")
            manip_video_latents = self._encode_video_latents(manip_input_image.unsqueeze(2), tiled=tiled)
            _profile_end("vae_encode_manip", stage_start)
            stage_start = _profile_start("video_pre_dit_nav")
            nav_video_pre = self.video_expert.pre_dit(
                x=nav_video_latents,
                timestep=torch.zeros((1,), dtype=nav_video_latents.dtype, device=self.device),
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            _profile_end("video_pre_dit_nav", stage_start)
            stage_start = _profile_start("video_pre_dit_manip")
            manip_video_pre = self.video_expert.pre_dit(
                x=manip_video_latents,
                timestep=torch.zeros((1,), dtype=manip_video_latents.dtype, device=self.device),
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            _profile_end("video_pre_dit_manip", stage_start)
        nav_video_seq_len = int(nav_video_pre["tokens"].shape[1])
        manip_video_seq_len = int(manip_video_pre["tokens"].shape[1])
        nav_tokens_per_frame = int(nav_video_pre["meta"]["tokens_per_frame"])
        manip_tokens_per_frame = int(manip_video_pre["meta"]["tokens_per_frame"])
        nav_video_attention_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=nav_video_seq_len,
            video_tokens_per_frame=nav_tokens_per_frame,
            device=nav_video_pre["tokens"].device,
        )
        manip_video_attention_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=manip_video_seq_len,
            video_tokens_per_frame=manip_tokens_per_frame,
            device=manip_video_pre["tokens"].device,
        )
        if nav_video_pre["tokens"].shape == manip_video_pre["tokens"].shape:
            stage_start = _profile_start("wan_video_prefill_2b")
            paired_video_cache = self.mot.prefill_video_cache(
                video_tokens=torch.cat([nav_video_pre["tokens"], manip_video_pre["tokens"]], dim=0),
                video_freqs=nav_video_pre["freqs"],
                video_t_mod=torch.cat([nav_video_pre["t_mod"], manip_video_pre["t_mod"]], dim=0),
                video_context_payload={
                    "context": torch.cat([nav_video_pre["context"], manip_video_pre["context"]], dim=0),
                    "mask": torch.cat(
                        [nav_video_pre["context_mask"], manip_video_pre["context_mask"]], dim=0
                    ),
                },
                video_attention_mask=nav_video_attention_mask,
            )
            _profile_end("wan_video_prefill_2b", stage_start)
            nav_video_cache = [
                {"k": layer["k"][:1], "v": layer["v"][:1]} for layer in paired_video_cache
            ]
            manip_video_cache = [
                {"k": layer["k"][1:], "v": layer["v"][1:]} for layer in paired_video_cache
            ]
        else:
            stage_start = _profile_start("wan_video_prefill_nav")
            nav_video_cache = self.mot.prefill_video_cache(
                video_tokens=nav_video_pre["tokens"],
                video_freqs=nav_video_pre["freqs"],
                video_t_mod=nav_video_pre["t_mod"],
                video_context_payload={
                    "context": nav_video_pre["context"],
                    "mask": nav_video_pre["context_mask"],
                },
                video_attention_mask=nav_video_attention_mask,
            )
            _profile_end("wan_video_prefill_nav", stage_start)
            stage_start = _profile_start("wan_video_prefill_manip")
            manip_video_cache = self.mot.prefill_video_cache(
                video_tokens=manip_video_pre["tokens"],
                video_freqs=manip_video_pre["freqs"],
                video_t_mod=manip_video_pre["t_mod"],
                video_context_payload={
                    "context": manip_video_pre["context"],
                    "mask": manip_video_pre["context_mask"],
                },
                video_attention_mask=manip_video_attention_mask,
            )
            _profile_end("wan_video_prefill_manip", stage_start)
        nav_attention_mask = self._build_mot_attention_mask(
            video_seq_len=nav_video_seq_len,
            action_seq_len=latents_nav.shape[1],
            video_tokens_per_frame=nav_tokens_per_frame,
            device=nav_video_pre["tokens"].device,
        )
        manip_attention_mask = self._build_mot_attention_mask(
            video_seq_len=manip_video_seq_len,
            action_seq_len=latents_manip.shape[1],
            video_tokens_per_frame=manip_tokens_per_frame,
            device=manip_video_pre["tokens"].device,
        )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_nav.dtype,
            shift_override=sigma_shift,
        )
        parallel_action_experts = (
            os.environ.get("FASTWAM_DUAL_ACTION_PARALLEL", "0") == "1"
            and torch.cuda.is_available()
            and str(self.device).startswith("cuda")
        )
        if parallel_action_experts:
            action_nav_stream = torch.cuda.Stream(device=self.device)
            action_manip_stream = torch.cuda.Stream(device=self.device)
            action_parent_stream = torch.cuda.current_stream(device=self.device)
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_nav.dtype, device=self.device)
            if parallel_action_experts:
                action_nav_stream.wait_stream(action_parent_stream)
                action_manip_stream.wait_stream(action_parent_stream)
                with torch.cuda.stream(action_nav_stream):
                    stage_start = _profile_start("nav_action_dit")
                    pred_nav = self._predict_dual_action_noise_with_cache(
                        expert=self.nav_action_expert,
                        latents_action=latents_nav,
                        timestep_action=timestep_action,
                        context=context,
                        context_mask=context_mask,
                        video_kv_cache=nav_video_cache,
                        attention_mask=nav_attention_mask,
                        video_seq_len=nav_video_seq_len,
                    )
                    _profile_end("nav_action_dit", stage_start)
                with torch.cuda.stream(action_manip_stream):
                    stage_start = _profile_start("manip_action_dit")
                    pred_manip = self._predict_dual_action_noise_with_cache(
                        expert=self.manip_action_expert,
                        latents_action=latents_manip,
                        timestep_action=timestep_action,
                        context=context,
                        context_mask=context_mask,
                        video_kv_cache=manip_video_cache,
                        attention_mask=manip_attention_mask,
                        video_seq_len=manip_video_seq_len,
                    )
                    _profile_end("manip_action_dit", stage_start)
                action_parent_stream.wait_stream(action_nav_stream)
                action_parent_stream.wait_stream(action_manip_stream)
            else:
                stage_start = _profile_start("nav_action_dit")
                pred_nav = self._predict_dual_action_noise_with_cache(
                    expert=self.nav_action_expert,
                    latents_action=latents_nav,
                    timestep_action=timestep_action,
                    context=context,
                    context_mask=context_mask,
                    video_kv_cache=nav_video_cache,
                    attention_mask=nav_attention_mask,
                    video_seq_len=nav_video_seq_len,
                )
                _profile_end("nav_action_dit", stage_start)
                stage_start = _profile_start("manip_action_dit")
                pred_manip = self._predict_dual_action_noise_with_cache(
                    expert=self.manip_action_expert,
                    latents_action=latents_manip,
                    timestep_action=timestep_action,
                    context=context,
                    context_mask=context_mask,
                    video_kv_cache=manip_video_cache,
                    attention_mask=manip_attention_mask,
                    video_seq_len=manip_video_seq_len,
                )
                _profile_end("manip_action_dit", stage_start)
            latents_nav = self.infer_action_scheduler.step(pred_nav, step_delta_action, latents_nav)
            latents_manip = self.infer_action_scheduler.step(pred_manip, step_delta_action, latents_manip)
            if prefix_nav is not None:
                latents_nav[:, :prefix_length] = prefix_nav
            if prefix_manip is not None:
                latents_manip[:, :prefix_length] = prefix_manip

        _profile_end("profiled_total", profiled_total_start)
        profile_ms: dict[str, float] = {}
        if profile_enabled:
            torch.cuda.synchronize(device=self.device)
            profile_ms = {
                name: float(sum(start.elapsed_time(end) for start, end in pairs))
                for name, pairs in profile_events.items()
            }
            logger.info("[DUAL_INFER_PROFILE_MS] %s", profile_ms)

        nav_out = latents_nav[0].detach().to(device="cpu", dtype=torch.float32)
        manip_out = latents_manip[0].detach().to(device="cpu", dtype=torch.float32)
        return {
            "nav_action": nav_out,
            "manip_action": manip_out,
            "action": torch.cat([nav_out, manip_out], dim=-1),
            "profile_ms": profile_ms,
            "parallel_action_experts": parallel_action_experts,
        }

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
