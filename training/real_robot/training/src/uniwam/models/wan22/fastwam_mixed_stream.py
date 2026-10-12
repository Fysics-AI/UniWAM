from __future__ import annotations

import os
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from uniwam.utils.logging_config import get_logger

from .fastwam_partitioned import FastWAMPartitioned
from .helpers.loader import load_wan22_ti2v_5b_components
from .mixed_stream_action_dit import MixedStreamActionDiT
from .mot import MoT


MANIP_BRANCH = 0
NAV_BRANCH = 1
logger = get_logger(__name__)


class _FiniteZeroAnchor(torch.autograd.Function):
    """Keep conditional parameters in the graph without propagating NaN through zero."""

    @staticmethod
    def forward(ctx, *values: torch.Tensor) -> torch.Tensor:
        if not values:
            raise ValueError("_FiniteZeroAnchor requires at least one scalar tensor.")
        ctx.input_count = len(values)
        return values[0].new_zeros(())

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        zero = torch.zeros_like(grad_output)
        return (zero,) * ctx.input_count


class FastWAMMixedStream(FastWAMPartitioned):
    """Dynamically pack active nav/manip streams through shared Wan and ActionDiT."""

    def __init__(
        self,
        *args,
        view_type_token_init_std: float = 0.02,
        view_type_token_count: int = 2,
        async_prefix_train: bool = False,
        async_prefix_prob: float = 0.5,
        async_prefix_min_length: int = 1,
        async_prefix_max_length: int = 6,
        async_prefix_lambda_attention: bool = False,
        async_prefix_local_window: int = 4,
        async_prefix_random_prefix_mask: bool = False,
        async_prefix_mask_prob: float = 0.5,
        async_prefix_keep_last_k: int = 2,
        async_prefix_rope_offset: int = 10,
        async_prefix_dynamic_loss_weight: bool = False,
        async_prefix_dynamic_loss_steps: int = 5,
        async_prefix_dynamic_loss_min: float = 0.5,
        async_prefix_dynamic_loss_max: float = 5.0,
        manip_robot_action_dim: Optional[int] = None,
        disable_manip_aux_training: bool = False,
        loss_lambda_manip_aux_action: float = 1.0,
        shared_future_state_training: bool = False,
        shared_future_state_max_delay_steps: int = 12,
        shared_future_state_delay_offsets: Optional[list[int]] = None,
        compile_training_denoise: bool = False,
        compile_action_infer: bool = False,
        compile_vae_infer: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not isinstance(self.action_expert, MixedStreamActionDiT):
            raise TypeError("FastWAMMixedStream requires MixedStreamActionDiT.")
        self.manip_robot_action_dim = int(
            self.action_expert.manip_action_dim
            if manip_robot_action_dim is None
            else manip_robot_action_dim
        )
        if not 0 < self.manip_robot_action_dim <= self.action_expert.manip_action_dim:
            raise ValueError("manip_robot_action_dim must be within the manipulation action width")
        self.manip_aux_action_dim = (
            self.action_expert.manip_action_dim - self.manip_robot_action_dim
        )
        self.disable_manip_aux_training = bool(disable_manip_aux_training)
        if self.manip_aux_action_dim not in (0, 4, 6, 8):
            raise ValueError(
                "The manipulation auxiliary action must be absent, 4D EEF-XY, "
                "6D EEF-XY plus visibility, or "
                f"8D bbox, got {self.manip_aux_action_dim}D."
            )
        self.loss_lambda_manip_aux_action = float(loss_lambda_manip_aux_action)
        if self.loss_lambda_manip_aux_action < 0.0:
            raise ValueError("loss_lambda_manip_aux_action must be non-negative")
        self.view_type_token_init_std = float(view_type_token_init_std)
        self.view_type_token_count = int(view_type_token_count)
        self.compile_training_denoise = bool(compile_training_denoise)
        self.compile_action_infer = bool(compile_action_infer)
        self.compile_vae_infer = bool(compile_vae_infer)
        self.mot.compile_training_layers = self.compile_training_denoise
        if self.compile_training_denoise and self.mot.mot_checkpoint_mixed_attn:
            raise ValueError(
                "compile_training_denoise requires mot_checkpoint_mixed_attn=false."
            )
        checkpointed_experts = [
            name
            for name, expert in (("video", self.video_expert), ("action", self.action_expert))
            if bool(getattr(expert, "use_gradient_checkpointing", False))
        ]
        if self.compile_training_denoise and checkpointed_experts:
            raise ValueError(
                "compile_training_denoise requires expert gradient checkpointing off: "
                f"{checkpointed_experts}."
            )
        if self.view_type_token_count <= 0:
            raise ValueError("view_type_token_count must be positive.")
        view_tokens = nn.ParameterDict(
            {
                "manip": nn.Parameter(
                    torch.empty(
                        1,
                        self.view_type_token_count,
                        self.video_expert.hidden_dim,
                        device=self.device,
                        dtype=self.torch_dtype,
                    )
                ),
                "nav": nn.Parameter(
                    torch.empty(
                        1,
                        self.view_type_token_count,
                        self.video_expert.hidden_dim,
                        device=self.device,
                        dtype=self.torch_dtype,
                    )
                ),
            }
        )
        nn.init.normal_(view_tokens["manip"], std=self.view_type_token_init_std)
        nn.init.normal_(view_tokens["nav"], std=self.view_type_token_init_std)
        # Keep type tokens inside model.dit so optimizer/checkpoint/freeze paths include them.
        self.mot.add_module("view_type_tokens", view_tokens)
        self.async_prefix_train = bool(async_prefix_train)
        self.async_prefix_prob = float(async_prefix_prob)
        self.async_prefix_min_length = int(async_prefix_min_length)
        self.async_prefix_max_length = int(async_prefix_max_length)
        self.async_prefix_lambda_attention = bool(async_prefix_lambda_attention)
        self.async_prefix_local_window = int(async_prefix_local_window)
        self.async_prefix_random_prefix_mask = bool(async_prefix_random_prefix_mask)
        self.async_prefix_mask_prob = float(async_prefix_mask_prob)
        self.async_prefix_keep_last_k = int(async_prefix_keep_last_k)
        self.async_prefix_rope_offset = int(async_prefix_rope_offset)
        self.async_prefix_dynamic_loss_weight = bool(async_prefix_dynamic_loss_weight)
        self.async_prefix_dynamic_loss_steps = int(async_prefix_dynamic_loss_steps)
        self.async_prefix_dynamic_loss_min = float(async_prefix_dynamic_loss_min)
        self.async_prefix_dynamic_loss_max = float(async_prefix_dynamic_loss_max)
        self.shared_future_state_training = bool(shared_future_state_training)
        self.shared_future_state_max_delay_steps = int(shared_future_state_max_delay_steps)
        if self.shared_future_state_max_delay_steps < 0:
            raise ValueError("shared_future_state_max_delay_steps must be non-negative.")
        if shared_future_state_delay_offsets is None:
            delay_offsets = tuple(range(self.shared_future_state_max_delay_steps + 1))
        else:
            delay_offsets = tuple(int(value) for value in shared_future_state_delay_offsets)
        if not delay_offsets or delay_offsets[0] != 0 or tuple(sorted(set(delay_offsets))) != delay_offsets:
            raise ValueError(
                "shared_future_state_delay_offsets must be sorted, unique, non-negative, "
                f"and start at zero; got {delay_offsets}."
            )
        self.shared_future_state_delay_offsets = delay_offsets
        if not 0.0 <= self.async_prefix_prob <= 1.0:
            raise ValueError("async_prefix_prob must be in [0, 1].")
        if self.async_prefix_min_length < 0:
            raise ValueError("async_prefix_min_length must be non-negative.")
        if self.async_prefix_max_length < self.async_prefix_min_length:
            raise ValueError("async_prefix_max_length must be >= async_prefix_min_length.")
        if self.async_prefix_local_window <= 0:
            raise ValueError("async_prefix_local_window must be positive.")
        if not 0.0 <= self.async_prefix_mask_prob <= 1.0:
            raise ValueError("async_prefix_mask_prob must be in [0, 1].")
        if self.async_prefix_keep_last_k < 0:
            raise ValueError("async_prefix_keep_last_k must be non-negative.")
        if self.async_prefix_rope_offset < 0:
            raise ValueError("async_prefix_rope_offset must be non-negative.")
        if self.async_prefix_dynamic_loss_steps <= 0:
            raise ValueError("async_prefix_dynamic_loss_steps must be positive.")
        if self.async_prefix_dynamic_loss_min <= 0.0:
            raise ValueError("async_prefix_dynamic_loss_min must be positive.")
        if self.async_prefix_dynamic_loss_max < self.async_prefix_dynamic_loss_min:
            raise ValueError(
                "async_prefix_dynamic_loss_max must be >= async_prefix_dynamic_loss_min."
            )

    @property
    def view_type_tokens(self) -> nn.ParameterDict:
        return self.mot.view_type_tokens

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
        loss_lambda_manip_aux_action: float = 1.0,
        manip_robot_action_dim: Optional[int] = None,
        disable_manip_aux_training: bool = False,
        view_type_token_init_std: float = 0.02,
        view_type_token_count: int = 2,
        async_prefix_train: bool = False,
        async_prefix_prob: float = 0.5,
        async_prefix_min_length: int = 1,
        async_prefix_max_length: int = 6,
        async_prefix_lambda_attention: bool = False,
        async_prefix_local_window: int = 4,
        async_prefix_random_prefix_mask: bool = False,
        async_prefix_mask_prob: float = 0.5,
        async_prefix_keep_last_k: int = 2,
        async_prefix_rope_offset: int = 10,
        async_prefix_dynamic_loss_weight: bool = False,
        async_prefix_dynamic_loss_steps: int = 5,
        async_prefix_dynamic_loss_min: float = 0.5,
        async_prefix_dynamic_loss_max: float = 5.0,
        shared_future_state_training: bool = False,
        shared_future_state_max_delay_steps: int = 12,
        shared_future_state_delay_offsets: Optional[list[int]] = None,
        compile_training_denoise: bool = False,
        compile_action_infer: bool = False,
        compile_vae_infer: bool = False,
    ) -> "FastWAMMixedStream":
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
        action_expert = MixedStreamActionDiT.from_pretrained(
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
            loss_lambda_manip_aux_action=loss_lambda_manip_aux_action,
            manip_robot_action_dim=manip_robot_action_dim,
            disable_manip_aux_training=disable_manip_aux_training,
            view_type_token_init_std=view_type_token_init_std,
            view_type_token_count=view_type_token_count,
            async_prefix_train=async_prefix_train,
            async_prefix_prob=async_prefix_prob,
            async_prefix_min_length=async_prefix_min_length,
            async_prefix_max_length=async_prefix_max_length,
            async_prefix_lambda_attention=async_prefix_lambda_attention,
            async_prefix_local_window=async_prefix_local_window,
            async_prefix_random_prefix_mask=async_prefix_random_prefix_mask,
            async_prefix_mask_prob=async_prefix_mask_prob,
            async_prefix_keep_last_k=async_prefix_keep_last_k,
            async_prefix_rope_offset=async_prefix_rope_offset,
            async_prefix_dynamic_loss_weight=async_prefix_dynamic_loss_weight,
            async_prefix_dynamic_loss_steps=async_prefix_dynamic_loss_steps,
            async_prefix_dynamic_loss_min=async_prefix_dynamic_loss_min,
            async_prefix_dynamic_loss_max=async_prefix_dynamic_loss_max,
            shared_future_state_training=shared_future_state_training,
            shared_future_state_max_delay_steps=shared_future_state_max_delay_steps,
            shared_future_state_delay_offsets=shared_future_state_delay_offsets,
            compile_training_denoise=compile_training_denoise,
            compile_action_infer=compile_action_infer,
            compile_vae_infer=compile_vae_infer,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN"
                if skip_dit_load_from_pretrain
                else action_dit_pretrained_path
            ),
        }
        return model

    @staticmethod
    def _as_branch_mask(sample: dict[str, Any], key: str, batch_size: int) -> torch.Tensor:
        value = sample.get(key)
        if value is None:
            return torch.ones(batch_size, dtype=torch.bool)
        value = torch.as_tensor(value, dtype=torch.bool).reshape(-1)
        if value.shape != (batch_size,):
            raise ValueError(f"{key} must be [B], got {tuple(value.shape)}.")
        return value

    @staticmethod
    def _branch_indices(
        sample: dict[str, Any],
        branch: str,
        branch_mask: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        key = f"{branch}_owner_indices"
        value = sample.get(key)
        expected = torch.nonzero(branch_mask, as_tuple=False).flatten()
        if value is None:
            return expected
        value = torch.as_tensor(value, dtype=torch.long).reshape(-1)
        if value.numel() and (int(value.min()) < 0 or int(value.max()) >= batch_size):
            raise ValueError(f"{key} contains an index outside [0, {batch_size}).")
        if value.unique().numel() != value.numel():
            raise ValueError(f"{key} must not contain duplicate sample indices.")
        if not torch.equal(value.cpu(), expected.cpu()):
            raise ValueError(
                f"{key} must match {branch}_branch_valid in ascending sample order: "
                f"got {value.tolist()}, expected {expected.tolist()}."
            )
        return value

    @staticmethod
    def _select_branch_tensor(
        sample: dict[str, Any],
        key: str,
        indices: torch.Tensor,
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        value = sample.get(key)
        if value is None:
            return None
        if not isinstance(value, torch.Tensor) or value.ndim == 0:
            raise TypeError(f"{key} must be a batched tensor, got {type(value)}.")
        branch_size = int(indices.numel())
        if value.shape[0] == branch_size:
            return value
        if value.shape[0] == batch_size:
            return value.index_select(0, indices)
        raise ValueError(
            f"{key} leading dimension must be packed ({branch_size}) or full batch "
            f"({batch_size}), got {value.shape[0]}."
        )

    @classmethod
    def _branch_tensor_or_default(
        cls,
        sample: dict[str, Any],
        key: str,
        indices: torch.Tensor,
        batch_size: int,
        *,
        tail_shape: tuple[int, ...],
        dtype: torch.dtype,
        default: bool,
    ) -> torch.Tensor:
        value = cls._select_branch_tensor(sample, key, indices, batch_size)
        branch_size = int(indices.numel())
        if value is None:
            return torch.full((branch_size, *tail_shape), default, dtype=dtype)
        if tuple(value.shape[1:]) != tail_shape:
            raise ValueError(
                f"{key} must end in {tail_shape}, got {tuple(value.shape)}."
            )
        return value.to(dtype=dtype)

    def _base_context(self, sample: dict[str, Any], batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        context = sample.get("context")
        context_mask = sample.get("context_mask")
        if context is None or context_mask is None:
            raise ValueError("Mixed-stream training requires context and context_mask.")
        if context.ndim != 3 or context.shape[0] != batch_size:
            raise ValueError(f"context must be [B,L,D], got {tuple(context.shape)}.")
        if tuple(context_mask.shape) != tuple(context.shape[:2]):
            raise ValueError(f"context_mask mismatch: {tuple(context_mask.shape)}.")
        context = context.to(self.device, self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(self.device, torch.bool, non_blocking=True)
        if self.proprio_encoder is not None:
            proprio = sample.get("proprio")
            if proprio is None or proprio.ndim != 3 or proprio.shape[-1] != self.proprio_dim:
                raise ValueError(f"Expected proprio [B,T,{self.proprio_dim}].")
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio[:, 0].to(self.device, self.torch_dtype, non_blocking=True),
            )
            proprio_valid = sample.get("proprio_valid")
            if proprio_valid is not None:
                proprio_valid = torch.as_tensor(
                    proprio_valid, dtype=torch.bool, device=self.device
                ).reshape(-1)
                if proprio_valid.shape != (batch_size,):
                    raise ValueError(
                        "proprio_valid must be [B] when proprio conditioning is enabled, "
                        f"got {tuple(proprio_valid.shape)}."
                    )
                # The appended token is present for tensor-shape stability, but
                # nav-only samples must not expose a synthetic zero arm state to
                # cross-attention.
                context_mask = context_mask.clone()
                context_mask[:, -1] &= proprio_valid
        return context, context_mask

    def build_inputs(self, sample: dict[str, Any], tiled: bool = False) -> dict[str, Any]:
        context_value = sample.get("context")
        if context_value is None or context_value.ndim != 3:
            raise ValueError("Cannot infer batch size from sample context.")
        batch_size = int(context_value.shape[0])
        shared_offsets = sample.get("offset_mask")
        shared_mode = shared_offsets is not None
        shared_training_enabled = bool(getattr(self, "shared_future_state_training", False))
        if shared_mode != shared_training_enabled:
            raise ValueError(
                "Shared-future-state dataset/model contract mismatch: "
                f"sample={shared_mode}, model={shared_training_enabled}."
            )
        num_offsets = 1
        if shared_mode:
            if shared_offsets.ndim != 2 or shared_offsets.shape[0] != batch_size:
                raise ValueError(f"offset_mask must be [B,N], got {tuple(shared_offsets.shape)}.")
            num_offsets = int(shared_offsets.shape[1])
            expected_offsets = tuple(
                getattr(
                    self,
                    "shared_future_state_delay_offsets",
                    range(int(getattr(self, "shared_future_state_max_delay_steps", 12)) + 1),
                )
            )
            expected = len(expected_offsets)
            if num_offsets != expected:
                raise ValueError(f"Expected {expected} shared offsets, got {num_offsets}.")
            delay_offsets = sample.get("delay_offsets")
            if delay_offsets is None or tuple(delay_offsets.shape) != (batch_size, num_offsets):
                raise ValueError(
                    f"delay_offsets must be [B,N]={batch_size, num_offsets}, got "
                    f"{getattr(delay_offsets, 'shape', None)}."
                )
            expected_tensor = torch.tensor(expected_offsets, device=delay_offsets.device)
            if not torch.equal(delay_offsets, expected_tensor.expand_as(delay_offsets)):
                raise ValueError(
                    f"Expected shared delay offsets {expected_offsets}, got "
                    f"{delay_offsets[0].tolist()}."
                )
        manip_valid = self._as_branch_mask(sample, "manip_branch_valid", batch_size)
        nav_valid = self._as_branch_mask(sample, "nav_branch_valid", batch_size)
        if not bool(manip_valid.any() or nav_valid.any()):
            raise ValueError("A mixed-stream batch must contain at least one active branch.")
        manip_indices = self._branch_indices(
            sample, "manip", manip_valid, batch_size
        )
        nav_indices = self._branch_indices(sample, "nav", nav_valid, batch_size)
        manip_count = int(manip_indices.numel())
        nav_count = int(nav_indices.numel())

        packed_videos: list[torch.Tensor] = []
        cached_latents: list[torch.Tensor] = []
        video_destinations: list[torch.Tensor] = []
        latent_destinations: list[torch.Tensor] = []
        video_shapes: list[tuple[int, ...]] = []
        branch_offset = 0
        for branch, indices in (("manip", manip_indices), ("nav", nav_indices)):
            if indices.numel() == 0:
                continue
            video_key = f"{branch}_video"
            latent_key = f"{branch}_latents"
            video = sample.get(video_key)
            latent = sample.get(latent_key)
            if video is None and latent is None:
                raise ValueError(
                    f"Active {branch} samples require {video_key} and/or {latent_key}."
                )
            if video is not None:
                if not isinstance(video, torch.Tensor):
                    raise TypeError(f"{video_key} must be a tensor.")
                self._validate_video(video_key, video)
                positions = sample.get(f"{branch}_video_positions")
                if positions is None:
                    if int(video.shape[0]) != int(indices.numel()):
                        raise ValueError(f"{video_key} requires explicit packed positions.")
                    positions = torch.arange(indices.numel(), dtype=torch.long)
                positions = torch.as_tensor(positions, dtype=torch.long)
                if positions.shape != (video.shape[0],):
                    raise ValueError(f"{branch}_video_positions shape mismatch.")
                packed_videos.append(video)
                video_shapes.append(tuple(video.shape[1:]))
                video_destinations.append(positions + branch_offset)
            if latent is not None:
                if not isinstance(latent, torch.Tensor):
                    raise TypeError(f"{latent_key} must be a tensor.")
                if (
                    latent.ndim != 5
                    or int(latent.shape[1]) != 48
                    or int(latent.shape[2]) <= 0
                    or tuple(latent.shape[3:]) != (20, 24)
                ):
                    raise ValueError(f"{latent_key} shape mismatch: {tuple(latent.shape)}.")
                positions = sample.get(f"{branch}_latent_positions")
                if positions is None:
                    if int(latent.shape[0]) != int(indices.numel()):
                        raise ValueError(f"{latent_key} requires explicit packed positions.")
                    positions = torch.arange(indices.numel(), dtype=torch.long)
                positions = torch.as_tensor(positions, dtype=torch.long)
                if positions.shape != (latent.shape[0],):
                    raise ValueError(f"{branch}_latent_positions shape mismatch.")
                cached_latents.append(latent)
                latent_destinations.append(positions + branch_offset)
            all_positions = torch.cat(
                ([video_destinations[-1]] if video is not None else [])
                + ([latent_destinations[-1]] if latent is not None else [])
            ) - branch_offset
            if (
                all_positions.numel() != indices.numel()
                or not torch.equal(
                    torch.sort(all_positions).values.cpu(),
                    torch.arange(indices.numel(), dtype=torch.long),
                )
            ):
                raise ValueError(
                    f"Raw/cached {branch} visual positions must partition its active streams."
                )
            branch_offset += int(indices.numel())

        encoded_latents = None
        if packed_videos:
            if len(set(video_shapes)) != 1:
                raise ValueError(f"All active video streams must share shape, got {video_shapes}.")
            packed_video = torch.cat(packed_videos, dim=0).to(
                self.device, self.torch_dtype, non_blocking=True
            )
            encoded_latents = self._encode_video_latents(packed_video, tiled=tiled)
            num_video_frames = int(packed_video.shape[2])
        if cached_latents:
            latent_shapes = {tuple(latent.shape[1:]) for latent in cached_latents}
            if len(latent_shapes) != 1:
                raise ValueError(
                    f"All active cached streams must share shape, got {sorted(latent_shapes)}."
                )
            cached_latents_tensor = torch.cat(cached_latents, dim=0).to(
                self.device, self.torch_dtype, non_blocking=True
            )
            num_video_frames = (int(cached_latents_tensor.shape[2]) - 1) * int(
                self.vae.temporal_downsample_factor
            ) + 1
        else:
            cached_latents_tensor = None

        stream_count = manip_count + nav_count
        latent_template = encoded_latents if encoded_latents is not None else cached_latents_tensor
        assert latent_template is not None
        packed_latents = torch.empty(
            (stream_count, *latent_template.shape[1:]),
            device=self.device,
            dtype=self.torch_dtype,
        )
        if encoded_latents is not None:
            packed_latents.index_copy_(
                0,
                torch.cat(video_destinations).to(self.device),
                encoded_latents,
            )
        if cached_latents_tensor is not None:
            packed_latents.index_copy_(
                0,
                torch.cat(latent_destinations).to(self.device),
                cached_latents_tensor,
            )

        context, context_mask = self._base_context(sample, batch_size)
        owner_indices = torch.cat((manip_indices, nav_indices), dim=0).to(self.device)
        stream_kind = torch.cat(
            (
                torch.full((manip_count,), MANIP_BRANCH, dtype=torch.long),
                torch.full((nav_count,), NAV_BRANCH, dtype=torch.long),
            ),
            dim=0,
        ).to(self.device)
        packed_context = context.index_select(0, owner_indices)
        packed_context_mask = context_mask.index_select(0, owner_indices)

        horizon = self.action_expert.manip_horizon
        if self.action_expert.nav_horizon != horizon:
            raise ValueError("Mixed-stream mode requires equal nav/manip horizons.")
        manip_action = None
        nav_action = None
        manip_feature_mask = None
        nav_feature_mask = None
        manip_is_pad = None
        nav_is_pad = None
        manip_loss_valid = None
        nav_loss_valid = None
        if manip_count:
            value = self._select_branch_tensor(
                sample, "manip_action", manip_indices, batch_size
            )
            expected_tail = (
                (num_offsets, horizon, self.action_expert.manip_action_dim)
                if shared_mode
                else (horizon, self.action_expert.manip_action_dim)
            )
            if value is None or tuple(value.shape[1:]) != expected_tail:
                raise ValueError(f"manip_action shape mismatch: {getattr(value, 'shape', None)}")
            manip_action = value.to(self.device, self.torch_dtype, non_blocking=True)
            manip_feature_mask = self._branch_tensor_or_default(
                sample,
                "manip_action_feature_mask",
                manip_indices,
                batch_size,
                tail_shape=expected_tail,
                dtype=torch.bool,
                default=True,
            ).to(self.device)
            manip_feature_mask = self._apply_manip_aux_training_mask(
                manip_feature_mask
            )
            manip_is_pad = self._branch_tensor_or_default(
                sample,
                "action_is_pad",
                manip_indices,
                batch_size,
                tail_shape=((num_offsets, horizon) if shared_mode else (horizon,)),
                dtype=torch.bool,
                default=False,
            ).to(self.device)
            manip_loss_valid_base = self._branch_tensor_or_default(
                sample,
                "manip_loss_valid",
                manip_indices,
                batch_size,
                tail_shape=(),
                dtype=torch.bool,
                default=True,
            ).to(self.device)
            if shared_mode:
                offset_valid = shared_offsets.index_select(0, manip_indices).to(self.device)
                manip_loss_valid = (
                    manip_loss_valid_base[:, None] & offset_valid
                ).reshape(-1)
                manip_action = manip_action.reshape(-1, horizon, self.action_expert.manip_action_dim)
                manip_feature_mask = manip_feature_mask.reshape_as(manip_action)
                manip_is_pad = manip_is_pad.reshape(-1, horizon)
            else:
                manip_loss_valid = manip_loss_valid_base
        if nav_count:
            value = self._select_branch_tensor(
                sample, "nav_action", nav_indices, batch_size
            )
            expected_tail = (
                (num_offsets, horizon, self.action_expert.nav_action_dim)
                if shared_mode
                else (horizon, self.action_expert.nav_action_dim)
            )
            if value is None or tuple(value.shape[1:]) != expected_tail:
                raise ValueError(f"nav_action shape mismatch: {getattr(value, 'shape', None)}")
            nav_action = value.to(
                self.device, self.torch_dtype, non_blocking=True
            )
            nav_feature_mask = self._branch_tensor_or_default(
                sample,
                "nav_action_feature_mask",
                nav_indices,
                batch_size,
                tail_shape=expected_tail,
                dtype=torch.bool,
                default=True,
            ).to(self.device)
            nav_is_pad = self._branch_tensor_or_default(
                sample,
                "nav_action_is_pad",
                nav_indices,
                batch_size,
                tail_shape=((num_offsets, horizon) if shared_mode else (horizon,)),
                dtype=torch.bool,
                default=False,
            ).to(self.device)
            nav_loss_valid_base = self._branch_tensor_or_default(
                sample,
                "nav_loss_valid",
                nav_indices,
                batch_size,
                tail_shape=(),
                dtype=torch.bool,
                default=True,
            ).to(self.device)
            if shared_mode:
                offset_valid = shared_offsets.index_select(0, nav_indices).to(self.device)
                nav_loss_valid = (nav_loss_valid_base[:, None] & offset_valid).reshape(-1)
                nav_action = nav_action.reshape(-1, horizon, self.action_expert.nav_action_dim)
                nav_feature_mask = nav_feature_mask.reshape_as(nav_action)
                nav_is_pad = nav_is_pad.reshape(-1, horizon)
            else:
                nav_loss_valid = nav_loss_valid_base

        action_context = packed_context
        action_context_mask = packed_context_mask
        if shared_mode:
            proprio = sample.get("proprio")
            if proprio is None or tuple(proprio.shape[:2]) != (batch_size, num_offsets):
                raise ValueError(
                    f"Shared mode requires proprio [B,N,{self.proprio_dim}], got "
                    f"{getattr(proprio, 'shape', None)}."
                )
            raw_context = context_value.to(self.device, self.torch_dtype, non_blocking=True)
            raw_mask = sample["context_mask"].to(self.device, torch.bool, non_blocking=True)
            action_context = raw_context.index_select(0, owner_indices).repeat_interleave(
                num_offsets, dim=0
            )
            action_context_mask = raw_mask.index_select(0, owner_indices).repeat_interleave(
                num_offsets, dim=0
            )
            future_proprio = proprio.index_select(0, owner_indices).reshape(
                -1, self.proprio_dim
            ).to(self.device, self.torch_dtype, non_blocking=True)
            action_context, action_context_mask = self._append_proprio_to_context(
                context=action_context,
                context_mask=action_context_mask,
                proprio=future_proprio,
            )

        image_masks = []
        for branch, indices in (("manip", manip_indices), ("nav", nav_indices)):
            if indices.numel() == 0:
                continue
            key = f"{branch}_image_is_pad"
            if key not in sample:
                key = "image_is_pad"
            image_masks.append(
                self._branch_tensor_or_default(
                    sample,
                    key,
                    indices,
                    batch_size,
                    tail_shape=(num_video_frames,),
                    dtype=torch.bool,
                    default=False,
                )
            )

        fuse = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
        return {
            "packed_input_latents": packed_latents,
            "packed_first_frame_latents": packed_latents[:, :, :1] if fuse else None,
            "fuse_vae_embedding_in_latents": fuse,
            "packed_context": packed_context,
            "packed_context_mask": packed_context_mask,
            "action_context": action_context,
            "action_context_mask": action_context_mask,
            "stream_kind": stream_kind,
            "owner_indices": owner_indices,
            "batch_size": batch_size,
            "manip_count": manip_count,
            "nav_count": nav_count,
            "manip_action": manip_action,
            "nav_action": nav_action,
            "manip_feature_mask": manip_feature_mask,
            "nav_feature_mask": nav_feature_mask,
            "manip_is_pad": manip_is_pad,
            "nav_is_pad": nav_is_pad,
            "manip_loss_valid": manip_loss_valid,
            "nav_loss_valid": nav_loss_valid,
            "shared_future_state": shared_mode,
            "num_offsets": num_offsets,
            "image_is_pad": torch.cat(image_masks, dim=0).to(self.device),
        }

    def _prepend_view_type_token(
        self,
        pre_state: dict[str, Any],
        stream_kind: torch.Tensor,
    ) -> dict[str, Any]:
        tokens = pre_state["tokens"]
        if stream_kind.shape != (tokens.shape[0],):
            raise ValueError("stream_kind must align with packed video batch.")
        token_bank = torch.cat(
            (self.view_type_tokens["manip"], self.view_type_tokens["nav"]), dim=0
        ).to(device=tokens.device, dtype=tokens.dtype)
        prefix = token_bank.index_select(0, stream_kind)
        type_token_count = int(prefix.shape[1])
        result = dict(pre_state)
        result["tokens"] = torch.cat((prefix, tokens), dim=1)
        result["freqs"] = torch.cat(
            (
                torch.ones_like(pre_state["freqs"][:1]).expand(
                    type_token_count, -1, -1
                ),
                pre_state["freqs"],
            ),
            dim=0,
        )
        result["t_mod"] = torch.cat(
            (
                pre_state["t_mod"][:, :1].expand(-1, type_token_count, -1, -1),
                pre_state["t_mod"],
            ),
            dim=1,
        )
        result["context_mask"] = torch.cat(
            (
                pre_state["context_mask"][:, :1].expand(
                    -1, type_token_count, -1
                ),
                pre_state["context_mask"],
            ),
            dim=1,
        )
        result["meta"] = dict(
            pre_state["meta"], type_token_count=type_token_count
        )
        return result

    def _video_mask_with_type(self, pre_state: dict[str, Any]) -> torch.Tensor:
        seq_len = int(pre_state["tokens"].shape[1])
        type_token_count = int(pre_state["meta"]["type_token_count"])
        tokens_per_frame = int(pre_state["meta"]["tokens_per_frame"])
        base_len = seq_len - type_token_count
        base = self.video_expert.build_video_to_video_mask(
            video_seq_len=base_len,
            video_tokens_per_frame=tokens_per_frame,
            device=pre_state["tokens"].device,
        )
        mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=base.device)
        mask[type_token_count:, type_token_count:] = base
        mask[:, :type_token_count] = True
        mask[
            :type_token_count,
            type_token_count : type_token_count + min(tokens_per_frame, base_len),
        ] = True
        return mask

    def _sample_prefix_length(self, horizon: int) -> int:
        if not self.async_prefix_train or horizon <= 1:
            return 0
        if float(torch.rand((), device=self.device).item()) >= self.async_prefix_prob:
            return 0
        max_prefix = min(self.async_prefix_max_length, horizon - 1)
        min_prefix = min(self.async_prefix_min_length, max_prefix)
        if min_prefix <= 0 or max_prefix <= 0:
            return 0
        return int(
            torch.randint(
                min_prefix,
                max_prefix + 1,
                (),
                device=self.device,
            ).item()
        )

    def _apply_prefix_rope(
        self,
        action_pre: dict[str, Any],
        prefix_length: int,
    ) -> dict[str, Any]:
        """Offset suffix RoPE positions without counting branch type tokens as steps."""
        prefix_length = int(prefix_length)
        if (
            prefix_length <= 0
            or not self.async_prefix_lambda_attention
            or self.async_prefix_rope_offset <= 0
        ):
            return action_pre
        meta = action_pre["meta"]
        horizon = int(meta["horizon"])
        type_count = int(meta["type_token_count"])
        if not 0 <= prefix_length < horizon:
            raise ValueError(f"prefix_length must be in [0, {horizon}), got {prefix_length}.")
        positions = torch.arange(horizon, device=action_pre["tokens"].device)
        positions[prefix_length:] += self.async_prefix_rope_offset
        if int(positions.max()) >= int(self.action_expert.freqs.shape[0]):
            raise ValueError("Prefix RoPE offset exceeds the ActionDiT frequency cache.")
        temporal_freqs = self.action_expert.freqs.to(positions.device)[positions].view(
            horizon, 1, -1
        )
        type_freqs = torch.ones_like(temporal_freqs[:1]).expand(type_count, -1, -1)
        result = dict(action_pre)
        result["freqs"] = torch.cat((type_freqs, temporal_freqs), dim=0)
        return result

    def _build_action_attention_mask(
        self,
        action_pre: dict[str, Any],
        prefix_length: int,
        *,
        random_prefix_mask: bool = False,
    ) -> torch.Tensor:
        """Build a leak-free mask for [type tokens][temporal action tokens]."""
        seq_len = int(action_pre["tokens"].shape[1])
        type_count = int(action_pre["meta"]["type_token_count"])
        horizon = int(action_pre["meta"]["horizon"])
        if seq_len != type_count + horizon:
            raise ValueError("Action token metadata does not match the packed sequence.")
        prefix_length = int(prefix_length)
        if not 0 <= prefix_length < horizon:
            raise ValueError(f"prefix_length must be in [0, {horizon}), got {prefix_length}.")
        # The first asynchronous request has no executed prefix, but it must
        # still use the same causal local policy as subsequent requests.  A
        # full bidirectional special case would train a different action model
        # exactly when the edge starts a fresh queue.
        if not self.async_prefix_lambda_attention:
            return torch.ones((seq_len, seq_len), dtype=torch.bool, device=self.device)

        mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=self.device)
        # Type queries may only read type keys; otherwise they become a future-token side channel.
        mask[:type_count, :type_count] = True
        mask[type_count:, :type_count] = True
        indices = torch.arange(horizon, device=self.device)
        query = indices[:, None]
        key = indices[None, :]
        temporal = (key <= query) & ((query - key) <= self.async_prefix_local_window)
        if random_prefix_mask and prefix_length > self.async_prefix_keep_last_k:
            maskable = prefix_length - self.async_prefix_keep_last_k
            dropped = torch.rand(maskable, device=self.device) < self.async_prefix_mask_prob
            if bool(dropped.any()):
                suffix_rows = indices >= prefix_length
                temporal[suffix_rows, :maskable] &= ~dropped.unsqueeze(0)
        mask[type_count:, type_count:] = temporal
        return mask

    def _packed_mot_forward(
        self,
        video_pre: dict[str, Any],
        action_pre: dict[str, Any],
        prefix_length: int = 0,
        random_prefix_mask: bool = False,
    ) -> dict[str, torch.Tensor]:
        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]
        if video_tokens.shape[0] != action_tokens.shape[0]:
            raise ValueError("Packed video/action stream batches must align.")
        first_frame_tokens = int(video_pre["meta"]["tokens_per_frame"])
        video_type_count = int(video_pre["meta"]["type_token_count"])
        video_prefix_len = video_type_count + min(
            first_frame_tokens, video_tokens.shape[1] - video_type_count
        )
        video_mask = self._video_mask_with_type(video_pre)
        action_self_mask = self._build_action_attention_mask(
            action_pre,
            prefix_length,
            random_prefix_mask=random_prefix_mask,
        )
        action_mask = torch.cat(
            (
                torch.ones(
                    (action_tokens.shape[1], video_prefix_len),
                    dtype=torch.bool,
                    device=action_tokens.device,
                ),
                action_self_mask,
            ),
            dim=1,
        )

        if self.compile_training_denoise:
            video_tokens, action_tokens = self.mot.forward_packed_core_tensor(
                video_tokens=video_tokens,
                action_tokens=action_tokens,
                video_freqs=video_pre["freqs"],
                action_freqs=action_pre["freqs"],
                video_t_mod=video_pre["t_mod"],
                action_t_mod=action_pre["t_mod"],
                video_context=video_pre["context"],
                video_context_mask=video_pre["context_mask"],
                action_context=action_pre["context"],
                action_context_mask=action_pre["context_mask"],
                video_attention_mask=video_mask,
                action_attention_mask=action_mask,
                video_prefix_len=video_prefix_len,
            )
            return {"video": video_tokens, "action": action_tokens}

        for layer_idx in range(self.mot.num_layers):
            video_block = self.video_expert.blocks[layer_idx]
            video_io = self.mot._build_expert_attention_io(
                expert=self.video_expert,
                block=video_block,
                x=video_tokens,
                freqs=video_pre["freqs"],
                t_mod=video_pre["t_mod"],
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
            video_tokens = self.mot._apply_post_with_optional_checkpoint(
                block=video_block,
                residual_x=residual_video,
                gate_msa=gate_msa_video,
                shift_mlp=shift_mlp_video,
                scale_mlp=scale_mlp_video,
                gate_mlp=gate_mlp_video,
                use_gradient_checkpointing=video_checkpoint,
                mixed_slice=mixed_video,
                context_payload={
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
            )

            action_block = self.action_expert.blocks[layer_idx]
            action_io = self.mot._build_expert_attention_io(
                expert=self.action_expert,
                block=action_block,
                x=action_tokens,
                freqs=action_pre["freqs"],
                t_mod=action_pre["t_mod"],
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
            mixed_action = self.mot._mixed_attention(
                q_action,
                torch.cat((k_video[:, :video_prefix_len], k_action), dim=1),
                torch.cat((v_video[:, :video_prefix_len], v_action), dim=1),
                action_mask,
            )
            action_tokens = self.mot._apply_post_with_optional_checkpoint(
                block=action_block,
                residual_x=residual_action,
                gate_msa=gate_msa_action,
                shift_mlp=shift_mlp_action,
                scale_mlp=scale_mlp_action,
                gate_mlp=gate_mlp_action,
                use_gradient_checkpointing=action_checkpoint,
                mixed_slice=mixed_action,
                context_payload={
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            )
        return {"video": video_tokens, "action": action_tokens}

    def _packed_mot_forward_shared_observation(
        self,
        video_pre: dict[str, Any],
        action_pre: dict[str, Any],
        num_offsets: int,
    ) -> dict[str, torch.Tensor]:
        """VLASH shared observation: one video prefix, isolated offset suffix rows."""
        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]
        num_offsets = int(num_offsets)
        if action_tokens.shape[0] != video_tokens.shape[0] * num_offsets:
            raise ValueError(
                "Shared observation batch mismatch: "
                f"video={video_tokens.shape[0]}, offsets={num_offsets}, action={action_tokens.shape[0]}."
            )
        first_frame_tokens = int(video_pre["meta"]["tokens_per_frame"])
        video_type_count = int(video_pre["meta"]["type_token_count"])
        video_prefix_len = video_type_count + min(
            first_frame_tokens, video_tokens.shape[1] - video_type_count
        )
        video_mask = self._video_mask_with_type(video_pre)
        action_self_mask = self._build_action_attention_mask(action_pre, prefix_length=0)
        action_mask = torch.cat(
            (
                torch.ones(
                    (action_tokens.shape[1], video_prefix_len),
                    dtype=torch.bool,
                    device=action_tokens.device,
                ),
                action_self_mask,
            ),
            dim=1,
        )
        for layer_idx in range(self.mot.num_layers):
            video_block = self.video_expert.blocks[layer_idx]
            video_io = self.mot._build_expert_attention_io(
                expert=self.video_expert,
                block=video_block,
                x=video_tokens,
                freqs=video_pre["freqs"],
                t_mod=video_pre["t_mod"],
            )
            (
                q_video, k_video, v_video, residual_video, gate_msa_video,
                shift_mlp_video, scale_mlp_video, gate_mlp_video, video_checkpoint,
            ) = video_io
            mixed_video = self.mot._mixed_attention(q_video, k_video, v_video, video_mask)
            video_tokens = self.mot._apply_post_with_optional_checkpoint(
                block=video_block,
                residual_x=residual_video,
                gate_msa=gate_msa_video,
                shift_mlp=shift_mlp_video,
                scale_mlp=scale_mlp_video,
                gate_mlp=gate_mlp_video,
                use_gradient_checkpointing=video_checkpoint,
                mixed_slice=mixed_video,
                context_payload={"context": video_pre["context"], "mask": video_pre["context_mask"]},
            )

            action_block = self.action_expert.blocks[layer_idx]
            action_io = self.mot._build_expert_attention_io(
                expert=self.action_expert,
                block=action_block,
                x=action_tokens,
                freqs=action_pre["freqs"],
                t_mod=action_pre["t_mod"],
            )
            (
                q_action, k_action, v_action, residual_action, gate_msa_action,
                shift_mlp_action, scale_mlp_action, gate_mlp_action, action_checkpoint,
            ) = action_io
            k_prefix = k_video[:, :video_prefix_len].repeat_interleave(num_offsets, dim=0)
            v_prefix = v_video[:, :video_prefix_len].repeat_interleave(num_offsets, dim=0)
            mixed_action = self.mot._mixed_attention(
                q_action,
                torch.cat((k_prefix, k_action), dim=1),
                torch.cat((v_prefix, v_action), dim=1),
                action_mask,
            )
            action_tokens = self.mot._apply_post_with_optional_checkpoint(
                block=action_block,
                residual_x=residual_action,
                gate_msa=gate_msa_action,
                shift_mlp=shift_mlp_action,
                scale_mlp=scale_mlp_action,
                gate_mlp=gate_mlp_action,
                use_gradient_checkpointing=action_checkpoint,
                mixed_slice=mixed_action,
                context_payload={"context": action_pre["context"], "mask": action_pre["context_mask"]},
            )
        return {"video": video_tokens, "action": action_tokens}

    @staticmethod
    def _global_mean(local_sum: torch.Tensor, local_count: torch.Tensor) -> torch.Tensor:
        # Every rank must enter the collective with the same dtype. In particular,
        # an empty conditional branch can provide a bf16 parameter anchor while a
        # populated rank computes its loss/count in fp32. Also clone the detached
        # count because all_reduce is in-place and `.to(...)` may otherwise alias
        # the zero anchor's storage.
        numerator = local_sum.float()
        count = local_count.detach().to(device=local_sum.device, dtype=torch.float32).clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
            world_size = dist.get_world_size()
            if float(count.item()) <= 0.0:
                return numerator * 0.0
            return numerator * float(world_size) / count
        if float(count.item()) <= 0.0:
            return numerator * 0.0
        return numerator / count

    @staticmethod
    def _empty_global_loss(zero_anchor: torch.Tensor) -> torch.Tensor:
        zero_count = torch.zeros((), device=zero_anchor.device, dtype=torch.float32)
        return FastWAMMixedStream._global_mean(zero_anchor.float(), zero_count)

    def _split_manip_loss_masks(
        self, feature_mask: Optional[torch.Tensor]
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if feature_mask is None or not self.manip_aux_action_dim:
            return feature_mask, None
        if feature_mask.shape[-1] != self.action_expert.manip_action_dim:
            raise ValueError(
                "Manipulation feature mask width disagrees with ActionDiT: "
                f"{feature_mask.shape[-1]} vs {self.action_expert.manip_action_dim}."
            )
        control = feature_mask.clone()
        control[..., self.manip_robot_action_dim :] = False
        auxiliary = feature_mask.clone()
        auxiliary[..., : self.manip_robot_action_dim] = False
        return control, auxiliary

    def _apply_manip_aux_training_mask(
        self, feature_mask: torch.Tensor
    ) -> torch.Tensor:
        """Remove auxiliary dimensions from clean inputs, noise, targets, and loss."""
        if not self.disable_manip_aux_training or not self.manip_aux_action_dim:
            return feature_mask
        if feature_mask.shape[-1] != self.action_expert.manip_action_dim:
            raise ValueError(
                "Manipulation feature mask width disagrees with ActionDiT: "
                f"{feature_mask.shape[-1]} vs {self.action_expert.manip_action_dim}."
            )
        result = feature_mask.clone()
        result[..., self.manip_robot_action_dim :] = False
        return result

    def _masked_action_loss(
        self,
        prediction: Optional[torch.Tensor],
        target: Optional[torch.Tensor],
        feature_mask: Optional[torch.Tensor],
        is_pad: Optional[torch.Tensor],
        loss_valid: Optional[torch.Tensor],
        timestep: Optional[torch.Tensor],
        zero_anchor: torch.Tensor,
        prefix_length: int = 0,
        dynamic_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if prediction is None:
            return self._empty_global_loss(zero_anchor)
        assert target is not None and feature_mask is not None and timestep is not None
        element_loss = F.mse_loss(prediction.float(), target.float(), reduction="none")
        if dynamic_weight is not None:
            if dynamic_weight.shape != element_loss.shape:
                raise ValueError(
                    f"dynamic_weight shape {dynamic_weight.shape} != loss shape {element_loss.shape}."
                )
            element_loss = element_loss * dynamic_weight.to(element_loss)
        valid_feature = feature_mask.to(self.device, dtype=element_loss.dtype)
        feature_count = valid_feature.sum(dim=-1)
        token_loss = (element_loss * valid_feature).sum(dim=-1) / feature_count.clamp(min=1.0)
        token_valid = feature_count > 0
        if is_pad is not None:
            token_valid &= ~is_pad.to(self.device, dtype=torch.bool)
        prefix_length = int(prefix_length)
        if not 0 <= prefix_length < token_valid.shape[1]:
            raise ValueError(
                f"prefix_length must be in [0, {token_valid.shape[1]}), got {prefix_length}."
            )
        if prefix_length:
            token_valid = token_valid.clone()
            token_valid[:, :prefix_length] = False
        valid = token_valid.to(token_loss.dtype)
        per_sample = (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        active = torch.ones_like(per_sample, dtype=torch.bool)
        if loss_valid is not None:
            active &= loss_valid.to(self.device, dtype=torch.bool)
        active &= valid.sum(dim=1) > 0
        weight = self.train_action_scheduler.training_weight(timestep).to(per_sample)
        local_sum = (per_sample * weight * active.to(per_sample.dtype)).sum() + zero_anchor
        return self._global_mean(local_sum, active.sum().to(per_sample.dtype))

    def _masked_action_loss_exact(
        self,
        prediction: Optional[torch.Tensor],
        target: Optional[torch.Tensor],
        feature_mask: Optional[torch.Tensor],
        is_pad: Optional[torch.Tensor],
        loss_valid: Optional[torch.Tensor],
        timestep: Optional[torch.Tensor],
        zero_anchor: torch.Tensor,
    ) -> torch.Tensor:
        """Mean over exactly the supervised VLASH offset/token/feature elements."""
        if prediction is None:
            return self._empty_global_loss(zero_anchor)
        assert target is not None and feature_mask is not None and timestep is not None
        valid = feature_mask.to(self.device, dtype=torch.bool).clone()
        if is_pad is not None:
            valid &= ~is_pad.to(self.device, dtype=torch.bool).unsqueeze(-1)
        if loss_valid is not None:
            valid &= loss_valid.to(self.device, dtype=torch.bool)[:, None, None]
        element = F.mse_loss(prediction.float(), target.float(), reduction="none")
        diffusion_weight = self.train_action_scheduler.training_weight(timestep).to(element)
        weighted = element * diffusion_weight[:, None, None]
        local_sum = (weighted * valid.to(weighted)).sum() + zero_anchor
        return self._global_mean(local_sum, valid.sum().to(weighted))

    @torch.no_grad()
    def _dynamic_prefix_weights(
        self,
        *,
        inputs: dict[str, Any],
        initial_manip: Optional[torch.Tensor],
        initial_nav: Optional[torch.Tensor],
        prefix_length: int,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Estimate suffix difficulty with a short, cached action-only rollout."""
        first_frame = inputs["packed_first_frame_latents"]
        if first_frame is None or prefix_length <= 0:
            return None, None
        latents_manip = (
            None
            if initial_manip is None
            else torch.where(
                inputs["manip_feature_mask"],
                initial_manip,
                torch.zeros_like(initial_manip),
            )
        )
        latents_nav = (
            None
            if initial_nav is None
            else torch.where(
                inputs["nav_feature_mask"],
                initial_nav,
                torch.zeros_like(initial_nav),
            )
        )
        clean_manip = inputs["manip_action"]
        clean_nav = inputs["nav_action"]
        if latents_manip is not None:
            clean_manip = torch.where(
                inputs["manip_feature_mask"], clean_manip, torch.zeros_like(clean_manip)
            )
            latents_manip[:, :prefix_length] = clean_manip[:, :prefix_length]
        if latents_nav is not None:
            clean_nav = torch.where(
                inputs["nav_feature_mask"], clean_nav, torch.zeros_like(clean_nav)
            )
            latents_nav[:, :prefix_length] = clean_nav[:, :prefix_length]

        stream_count = int(first_frame.shape[0])
        video_pre_base = self.video_expert.pre_dit(
            x=first_frame,
            timestep=torch.zeros(
                stream_count, device=self.device, dtype=first_frame.dtype
            ),
            context=inputs["packed_context"],
            context_mask=inputs["packed_context_mask"],
            action=None,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        video_pre = self._prepend_view_type_token(
            video_pre_base, inputs["stream_kind"]
        )
        video_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=self._video_mask_with_type(video_pre),
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        video_prefix_len = int(video_pre["meta"]["type_token_count"]) + int(
            video_pre["meta"]["tokens_per_frame"]
        )
        dtype = latents_manip.dtype if latents_manip is not None else latents_nav.dtype
        timesteps, deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=self.async_prefix_dynamic_loss_steps,
            device=self.device,
            dtype=dtype,
        )
        for step_t, step_delta in zip(timesteps, deltas):
            prediction = self._predict_action_with_cache(
                latents_manip=latents_manip,
                latents_nav=latents_nav,
                timestep=step_t.reshape(1).to(self.device, dtype),
                context=inputs["packed_context"],
                context_mask=inputs["packed_context_mask"],
                video_cache=video_cache,
                video_seq_len=video_seq_len,
                video_prefix_len=video_prefix_len,
                prefix_length=prefix_length,
            )
            if latents_manip is not None:
                latents_manip = self.infer_action_scheduler.step(
                    prediction["manip"], step_delta, latents_manip
                )
                latents_manip[:, :prefix_length] = clean_manip[:, :prefix_length]
            if latents_nav is not None:
                latents_nav = self.infer_action_scheduler.step(
                    prediction["nav"], step_delta, latents_nav
                )
                latents_nav[:, :prefix_length] = clean_nav[:, :prefix_length]

        def make_weight(
            generated: Optional[torch.Tensor],
            clean: Optional[torch.Tensor],
            feature_mask: Optional[torch.Tensor],
            is_pad: Optional[torch.Tensor],
            loss_valid: Optional[torch.Tensor],
        ) -> Optional[torch.Tensor]:
            if generated is None:
                return None
            error = (generated.float() - clean.float()).abs()
            valid = feature_mask.to(self.device, dtype=torch.bool).clone()
            valid[:, :prefix_length] = False
            if is_pad is not None:
                valid &= ~is_pad.to(self.device, dtype=torch.bool).unsqueeze(-1)
            if loss_valid is not None:
                valid &= loss_valid.to(self.device, dtype=torch.bool)[:, None, None]
            denominator = valid.sum().clamp(min=1).to(error)
            mean_error = (error * valid.to(error)).sum() / denominator
            weight = (error / mean_error.clamp(min=1.0e-6)).clamp(
                self.async_prefix_dynamic_loss_min,
                self.async_prefix_dynamic_loss_max,
            )
            return torch.where(valid, weight, torch.ones_like(weight))

        return (
            make_weight(
                latents_manip,
                clean_manip,
                inputs["manip_feature_mask"],
                inputs["manip_is_pad"],
                inputs["manip_loss_valid"],
            ),
            make_weight(
                latents_nav,
                clean_nav,
                inputs["nav_feature_mask"],
                inputs["nav_is_pad"],
                inputs["nav_loss_valid"],
            ),
        )

    def _parameter_anchor(self) -> torch.Tensor:
        conditional_modules = (
            self.action_expert.manip_action_encoder,
            self.action_expert.nav_action_encoder,
            self.action_expert.manip_head,
            self.action_expert.nav_head,
        )
        parameters = [
            parameter
            for module in conditional_modules
            for parameter in module.parameters()
        ]
        parameters.extend(
            (
                self.action_expert.manip_type_token,
                self.action_expert.nav_type_token,
                self.view_type_tokens["manip"],
                self.view_type_tokens["nav"],
            )
        )
        return _FiniteZeroAnchor.apply(
            *(parameter.reshape(-1)[0] for parameter in parameters)
        )

    def training_loss(self, sample: dict[str, Any], tiled: bool = False):
        profile_enabled = (
            os.environ.get("FASTWAM_PROFILE_TRAIN", "0") == "1"
            and self.device.type == "cuda"
        )
        profile_events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}

        def profile_start() -> Optional[torch.cuda.Event]:
            if not profile_enabled:
                return None
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event

        def profile_end(name: str, start: Optional[torch.cuda.Event]) -> None:
            if start is None:
                return
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            profile_events[name] = (start, end)

        stage_start = profile_start()
        inputs = self.build_inputs(sample, tiled=tiled)
        profile_end("vae_and_pack", stage_start)
        source_video = inputs["packed_input_latents"]
        active_streams = int(source_video.shape[0])
        prefix_length = (
            0
            if inputs["shared_future_state"]
            else self._sample_prefix_length(self.action_expert.manip_horizon)
        )
        noise_video = torch.randn_like(source_video)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=active_streams,
            device=self.device,
            dtype=source_video.dtype,
        )
        noisy_video = self.train_video_scheduler.add_noise(
            source_video, noise_video, timestep_video
        )
        target_video = self.train_video_scheduler.training_target(
            source_video, noise_video, timestep_video
        )
        if inputs["packed_first_frame_latents"] is not None:
            noisy_video[:, :, :1] = inputs["packed_first_frame_latents"]

        def noisy_action(source: Optional[torch.Tensor], mask: Optional[torch.Tensor]):
            if source is None:
                return None, None, None, None
            assert mask is not None
            source = torch.where(mask, source, torch.zeros_like(source))
            noise = torch.randn_like(source)
            timestep = self.train_action_scheduler.sample_training_t(
                batch_size=source.shape[0], device=self.device, dtype=source.dtype
            )
            noisy = self.train_action_scheduler.add_noise(source, noise, timestep)
            target = self.train_action_scheduler.training_target(source, noise, timestep)
            if prefix_length:
                noisy[:, :prefix_length] = source[:, :prefix_length]
            noisy = torch.where(mask, noisy, torch.zeros_like(noisy))
            target = torch.where(mask, target, torch.zeros_like(target))
            return noisy, target, timestep, noise

        noisy_manip, target_manip, timestep_manip, initial_manip = noisy_action(
            inputs["manip_action"], inputs["manip_feature_mask"]
        )
        noisy_nav, target_nav, timestep_nav, initial_nav = noisy_action(
            inputs["nav_action"], inputs["nav_feature_mask"]
        )

        stage_start = profile_start()
        video_pre_base = self.video_expert.pre_dit(
            x=noisy_video,
            timestep=timestep_video,
            context=inputs["packed_context"],
            context_mask=inputs["packed_context_mask"],
            action=None,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        video_pre = self._prepend_view_type_token(video_pre_base, inputs["stream_kind"])
        profile_end("video_pre", stage_start)
        manip_count = int(inputs["manip_count"])
        nav_count = int(inputs["nav_count"])
        num_offsets = int(inputs["num_offsets"])
        action_manip_count = manip_count * num_offsets
        action_nav_count = nav_count * num_offsets
        stage_start = profile_start()
        action_pre = self.action_expert.pre_dit_packed(
            manip_action=noisy_manip,
            nav_action=noisy_nav,
            manip_timestep=timestep_manip,
            nav_timestep=timestep_nav,
            manip_context=(inputs["action_context"][:action_manip_count] if manip_count else None),
            nav_context=(inputs["action_context"][action_manip_count:] if nav_count else None),
            manip_context_mask=(
                inputs["action_context_mask"][:action_manip_count] if manip_count else None
            ),
            nav_context_mask=(
                inputs["action_context_mask"][action_manip_count:] if nav_count else None
            ),
        )
        action_pre = self._apply_prefix_rope(action_pre, prefix_length)
        profile_end("action_pre", stage_start)
        stage_start = profile_start()
        if inputs["shared_future_state"]:
            output = self._packed_mot_forward_shared_observation(
                video_pre, action_pre, num_offsets=num_offsets
            )
        else:
            output = self._packed_mot_forward(
                video_pre,
                action_pre,
                prefix_length=prefix_length,
                random_prefix_mask=(
                    self.async_prefix_random_prefix_mask
                    and self.async_prefix_lambda_attention
                    and prefix_length > 0
                ),
            )
        profile_end("mot_30_layers", stage_start)
        stage_start = profile_start()
        video_type_count = int(video_pre["meta"]["type_token_count"])
        pred_video = self.video_expert.post_dit(
            output["video"][:, video_type_count:], video_pre_base
        )
        action_output = self.action_expert.post_dit_packed(output["action"], action_pre)

        include_initial = inputs["packed_first_frame_latents"] is None
        if not include_initial:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]
        video_per_sample = self._compute_video_loss_per_sample(
            pred_video,
            target_video,
            inputs["image_is_pad"],
            include_initial,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            video_per_sample
        )
        weighted_video = video_per_sample * video_weight
        loss_video = self._global_mean(
            weighted_video.sum(),
            torch.tensor(active_streams, device=self.device, dtype=weighted_video.dtype),
        )
        stream_kind = inputs["stream_kind"]
        manip_video_mask = stream_kind == MANIP_BRANCH
        nav_video_mask = stream_kind == NAV_BRANCH
        loss_video_manip = self._global_mean(
            weighted_video[manip_video_mask].sum(), manip_video_mask.sum().to(weighted_video)
        )
        loss_video_nav = self._global_mean(
            weighted_video[nav_video_mask].sum(), nav_video_mask.sum().to(weighted_video)
        )

        anchor = self._parameter_anchor()
        dynamic_manip = None
        dynamic_nav = None
        if self.async_prefix_dynamic_loss_weight and prefix_length > 0:
            dynamic_manip, dynamic_nav = self._dynamic_prefix_weights(
                inputs=inputs,
                initial_manip=initial_manip,
                initial_nav=initial_nav,
                prefix_length=prefix_length,
            )
        action_loss_fn = (
            self._masked_action_loss_exact
            if inputs["shared_future_state"]
            else self._masked_action_loss
        )
        manip_control_mask, manip_aux_mask = self._split_manip_loss_masks(
            inputs["manip_feature_mask"]
        )
        loss_manip = action_loss_fn(
            action_output["manip"],
            target_manip,
            manip_control_mask,
            inputs["manip_is_pad"],
            inputs["manip_loss_valid"],
            timestep_manip,
            anchor,
            *(() if inputs["shared_future_state"] else (prefix_length, dynamic_manip)),
        )
        loss_manip_aux = action_loss_fn(
            action_output["manip"],
            target_manip,
            manip_aux_mask,
            inputs["manip_is_pad"],
            inputs["manip_loss_valid"],
            timestep_manip,
            anchor,
            *(() if inputs["shared_future_state"] else (prefix_length, dynamic_manip)),
        ) if self.manip_aux_action_dim else anchor
        loss_nav = action_loss_fn(
            action_output["nav"],
            target_nav,
            inputs["nav_feature_mask"],
            inputs["nav_is_pad"],
            inputs["nav_loss_valid"],
            timestep_nav,
            anchor,
            *(() if inputs["shared_future_state"] else (prefix_length, dynamic_nav)),
        )
        loss = (
            self.loss_lambda_video * loss_video
            + self.loss_lambda_manip_action * loss_manip
            + self.loss_lambda_manip_aux_action * loss_manip_aux
            + self.loss_lambda_nav_action * loss_nav
            + anchor
        )
        profile_end("post_and_loss", stage_start)
        if profile_enabled:
            torch.cuda.synchronize(self.device)
            if not dist.is_initialized() or dist.get_rank() == 0:
                timings = " ".join(
                    f"{name}={start.elapsed_time(end):.1f}ms"
                    for name, (start, end) in profile_events.items()
                )
                logger.info(
                    "[train-profile:model] streams=%d manip=%d nav=%d %s",
                    active_streams,
                    manip_count,
                    nav_count,
                    timings,
                )
        return loss, {
            "loss_video": self.loss_lambda_video * float(loss_video.detach()),
            "loss_video_nav": self.loss_lambda_video * float(loss_video_nav.detach()),
            "loss_video_manip": self.loss_lambda_video * float(loss_video_manip.detach()),
            "loss_action_manip": self.loss_lambda_manip_action * float(loss_manip.detach()),
            "loss_action_manip_aux": self.loss_lambda_manip_aux_action
            * float(loss_manip_aux.detach()),
            "loss_action_nav": self.loss_lambda_nav_action * float(loss_nav.detach()),
            "active_streams_per_sample": float(active_streams)
            / float(inputs["batch_size"]),
            "active_manip_streams": float(manip_count),
            "active_nav_streams": float(nav_count),
            "async_prefix_length": float(prefix_length),
            "async_prefix_active": float(prefix_length > 0),
            "shared_observation_offsets": float(num_offsets),
            "async_dynamic_weight_manip": float(
                0.0 if dynamic_manip is None else dynamic_manip.mean().item()
            ),
            "async_dynamic_weight_nav": float(
                0.0 if dynamic_nav is None else dynamic_nav.mean().item()
            ),
        }

    def _predict_action_with_cache(
        self,
        *,
        latents_manip: Optional[torch.Tensor],
        latents_nav: Optional[torch.Tensor],
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_cache: list[dict[str, torch.Tensor]]
        | tuple[list[torch.Tensor], list[torch.Tensor]],
        video_seq_len: int,
        video_prefix_len: int,
        prefix_length: int = 0,
        compiled: bool = False,
    ) -> dict[str, Optional[torch.Tensor]]:
        manip_count = 0 if latents_manip is None else int(latents_manip.shape[0])
        nav_count = 0 if latents_nav is None else int(latents_nav.shape[0])
        action_pre = self.action_expert.pre_dit_packed(
            manip_action=latents_manip,
            nav_action=latents_nav,
            manip_timestep=(timestep.expand(manip_count) if manip_count else None),
            nav_timestep=(timestep.expand(nav_count) if nav_count else None),
            manip_context=(context[:manip_count] if manip_count else None),
            nav_context=(context[manip_count:] if nav_count else None),
            manip_context_mask=(context_mask[:manip_count] if manip_count else None),
            nav_context_mask=(context_mask[manip_count:] if nav_count else None),
        )
        action_pre = self._apply_prefix_rope(action_pre, prefix_length)
        action_seq_len = int(action_pre["tokens"].shape[1])
        joint_mask = torch.zeros(
            (video_seq_len + action_seq_len, video_seq_len + action_seq_len),
            dtype=torch.bool,
            device=self.device,
        )
        joint_mask[:video_seq_len, :video_seq_len] = True
        joint_mask[video_seq_len:, :video_prefix_len] = True
        joint_mask[video_seq_len:, video_seq_len:] = self._build_action_attention_mask(
            action_pre,
            prefix_length,
        )
        if compiled:
            if not isinstance(video_cache, tuple) or len(video_cache) != 2:
                raise TypeError("Compiled inference requires tensor-form (K, V) video cache.")
            if not hasattr(self, "_action_with_video_cache_compiled"):
                self._action_with_video_cache_compiled = torch.compile(
                    self.mot.forward_action_with_video_cache_tensor,
                    mode="reduce-overhead",
                    fullgraph=True,
                )
            torch.compiler.cudagraph_mark_step_begin()
            tokens = self._action_with_video_cache_compiled(
                action_tokens=action_pre["tokens"],
                action_freqs=action_pre["freqs"],
                action_t_mod=action_pre["t_mod"],
                action_context=action_pre["context"],
                action_context_mask=action_pre["context_mask"],
                video_cache_k=video_cache[0],
                video_cache_v=video_cache[1],
                action_attention_mask=joint_mask[video_seq_len:, :],
            )
        else:
            tokens = self.mot.forward_action_with_video_cache(
                action_tokens=action_pre["tokens"],
                action_freqs=action_pre["freqs"],
                action_t_mod=action_pre["t_mod"],
                action_context_payload={
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
                video_kv_cache=video_cache,
                attention_mask=joint_mask,
                video_seq_len=video_seq_len,
            )
        return self.action_expert.post_dit_packed(tokens, action_pre)

    @torch.no_grad()
    def infer_action(
        self,
        *,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        future_proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        nav_input_image: Optional[torch.Tensor] = None,
        manip_input_image: Optional[torch.Tensor] = None,
        num_inference_steps: int = 10,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        manip_action_prefix: Optional[torch.Tensor] = None,
        nav_action_prefix: Optional[torch.Tensor] = None,
        prefix_length: int = 0,
        compile_action_infer: Optional[bool] = None,
    ) -> dict[str, Any]:
        self.eval()
        if action_horizon != self.action_expert.manip_horizon:
            raise ValueError("action_horizon does not match trained manipulation horizon.")
        if action_horizon != self.action_expert.nav_horizon:
            raise ValueError("action_horizon does not match trained navigation horizon.")
        if manip_input_image is None and nav_input_image is None:
            raise ValueError("At least one manipulation or navigation image is required.")
        if context is None or context_mask is None:
            raise ValueError("Precomputed context and context_mask are required.")
        prefix_length = int(prefix_length)
        if self.shared_future_state_training and prefix_length != 0:
            raise ValueError("VLASH shared-future-state inference does not accept an action prefix.")
        if not 0 <= prefix_length < action_horizon:
            raise ValueError(
                f"prefix_length must be in [0, {action_horizon}), got {prefix_length}."
            )
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        context = context.to(self.device, self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(self.device, torch.bool, non_blocking=True)
        action_context = context
        action_context_mask = context_mask
        if proprio is not None:
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(self.device, self.torch_dtype, non_blocking=True),
            )
        if self.shared_future_state_training:
            if future_proprio is None:
                raise ValueError("Shared-future-state inference requires future_proprio.")
            if future_proprio.ndim == 1:
                future_proprio = future_proprio.unsqueeze(0)
            action_context, action_context_mask = self._append_proprio_to_context(
                context=action_context,
                context_mask=action_context_mask,
                proprio=future_proprio.to(self.device, self.torch_dtype, non_blocking=True),
            )
        elif future_proprio is not None:
            raise ValueError("future_proprio is only valid for shared-future-state checkpoints.")
        else:
            action_context, action_context_mask = context, context_mask

        def prepare_image(name: str, image: torch.Tensor) -> torch.Tensor:
            if image.ndim == 3:
                image = image.unsqueeze(0)
            if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
                raise ValueError(f"{name} must be [1,3,H,W], got {tuple(image.shape)}.")
            if image.shape[-2] % 32 or image.shape[-1] % 32:
                raise ValueError(f"{name} spatial dimensions must be divisible by 32.")
            return image.to(self.device, self.torch_dtype, non_blocking=True)

        images = []
        kinds = []
        if manip_input_image is not None:
            images.append(prepare_image("manip_input_image", manip_input_image))
            kinds.append(MANIP_BRANCH)
        if nav_input_image is not None:
            images.append(prepare_image("nav_input_image", nav_input_image))
            kinds.append(NAV_BRANCH)
        if len({tuple(image.shape[1:]) for image in images}) != 1:
            raise ValueError("All active inference images must have identical shapes.")
        stream_count = len(images)
        packed_images = torch.cat(images, dim=0)
        packed_context = context.expand(stream_count, -1, -1)
        packed_context_mask = context_mask.expand(stream_count, -1)
        packed_action_context = action_context.expand(stream_count, -1, -1)
        packed_action_context_mask = action_context_mask.expand(stream_count, -1)
        use_compiled_action = (
            self.compile_action_infer
            if compile_action_infer is None
            else bool(compile_action_infer)
        )
        if self.compile_vae_infer:
            image_latents = self._encode_video_latents_compiled(
                packed_images.unsqueeze(2), tiled=tiled
            )
        else:
            image_latents = self._encode_video_latents(
                packed_images.unsqueeze(2), tiled=tiled
            )
        video_pre_base = self.video_expert.pre_dit(
            x=image_latents,
            timestep=torch.zeros(stream_count, device=self.device, dtype=image_latents.dtype),
            context=packed_context,
            context_mask=packed_context_mask,
            action=None,
            fuse_vae_embedding_in_latents=bool(
                getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)
            ),
        )
        kind_tensor = torch.tensor(kinds, device=self.device, dtype=torch.long)
        video_pre = self._prepend_view_type_token(video_pre_base, kind_tensor)
        video_mask = self._video_mask_with_type(video_pre)
        if use_compiled_action:
            if not hasattr(self, "_prefill_video_cache_compiled"):
                self._prefill_video_cache_compiled = torch.compile(
                    self.mot.prefill_video_cache_tensor,
                    mode="reduce-overhead",
                    fullgraph=True,
                )
            torch.compiler.cudagraph_mark_step_begin()
            video_cache_k, video_cache_v = self._prefill_video_cache_compiled(
                video_tokens=video_pre["tokens"],
                video_freqs=video_pre["freqs"],
                video_t_mod=video_pre["t_mod"],
                video_context=video_pre["context"],
                video_context_mask=video_pre["context_mask"],
                video_attention_mask=video_mask,
            )
            # Inductor reduce-overhead outputs can alias graph-owned replay buffers.
            video_cache = (
                [cache.clone() for cache in video_cache_k],
                [cache.clone() for cache in video_cache_v],
            )
        else:
            video_cache = self.mot.prefill_video_cache(
                video_tokens=video_pre["tokens"],
                video_freqs=video_pre["freqs"],
                video_t_mod=video_pre["t_mod"],
                video_context_payload={
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                video_attention_mask=video_mask,
            )
        video_seq_len = int(video_pre["tokens"].shape[1])
        video_prefix_len = int(video_pre["meta"]["type_token_count"]) + int(
            video_pre["meta"]["tokens_per_frame"]
        )

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_manip = None
        latents_nav = None
        if manip_input_image is not None:
            latents_manip = torch.randn(
                (1, action_horizon, self.action_expert.manip_action_dim),
                generator=generator,
                device=rand_device,
                dtype=torch.float32,
            ).to(self.device, self.torch_dtype)
        if nav_input_image is not None:
            latents_nav = torch.randn(
                (1, action_horizon, self.action_expert.nav_action_dim),
                generator=generator,
                device=rand_device,
                dtype=torch.float32,
            ).to(self.device, self.torch_dtype)

        def prepare_prefix(
            name: str,
            value: Optional[torch.Tensor],
            *,
            active: bool,
            action_dim: int,
        ) -> Optional[torch.Tensor]:
            if not active:
                if value is not None and value.numel():
                    raise ValueError(f"{name} was provided for an inactive branch.")
                return None
            if prefix_length == 0:
                if value is not None and value.numel():
                    raise ValueError(f"{name} must be empty when prefix_length=0.")
                return None
            if value is None:
                raise ValueError(f"{name} is required when prefix_length={prefix_length}.")
            if value.ndim == 2:
                value = value.unsqueeze(0)
            expected = (1, prefix_length, action_dim)
            if tuple(value.shape) != expected:
                raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}.")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} contains non-finite values.")
            return value.to(self.device, self.torch_dtype, non_blocking=True)

        clean_manip_prefix = prepare_prefix(
            "manip_action_prefix",
            manip_action_prefix,
            active=latents_manip is not None,
            action_dim=self.action_expert.manip_action_dim,
        )
        clean_nav_prefix = prepare_prefix(
            "nav_action_prefix",
            nav_action_prefix,
            active=latents_nav is not None,
            action_dim=self.action_expert.nav_action_dim,
        )
        if clean_manip_prefix is not None:
            latents_manip[:, :prefix_length] = clean_manip_prefix
        if clean_nav_prefix is not None:
            latents_nav[:, :prefix_length] = clean_nav_prefix
        schedule_dtype = (
            latents_manip.dtype if latents_manip is not None else latents_nav.dtype
        )
        timesteps, deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=schedule_dtype,
            shift_override=sigma_shift,
        )
        for step_t, step_delta in zip(timesteps, deltas):
            prediction = self._predict_action_with_cache(
                latents_manip=latents_manip,
                latents_nav=latents_nav,
                timestep=step_t.reshape(1).to(self.device, schedule_dtype),
                context=packed_action_context,
                context_mask=packed_action_context_mask,
                video_cache=video_cache,
                video_seq_len=video_seq_len,
                video_prefix_len=video_prefix_len,
                prefix_length=prefix_length,
                compiled=use_compiled_action,
            )
            if latents_manip is not None:
                assert prediction["manip"] is not None
                latents_manip = self.infer_action_scheduler.step(
                    prediction["manip"], step_delta, latents_manip
                )
                if clean_manip_prefix is not None:
                    latents_manip[:, :prefix_length] = clean_manip_prefix
            if latents_nav is not None:
                assert prediction["nav"] is not None
                latents_nav = self.infer_action_scheduler.step(
                    prediction["nav"], step_delta, latents_nav
                )
                if clean_nav_prefix is not None:
                    latents_nav[:, :prefix_length] = clean_nav_prefix
        result: dict[str, Any] = {"profile_ms": {}, "prefix_length": prefix_length}
        if latents_manip is not None:
            result["manip_action"] = latents_manip[0].float().cpu()
        if latents_nav is not None:
            result["nav_action"] = latents_nav[0].float().cpu()
        return result

    def forward(self, sample, tiled: bool = False, **kwargs):
        del kwargs
        return self.training_loss(sample, tiled=tiled)
