from __future__ import annotations

from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger

from .fastwam import FastWAM
from .helpers.loader import load_wan22_ti2v_5b_components
from .uniwam_action_dit import UniWAMActionDiT
from .mot import MoT
from .uniwam_initialization import initialize_action_from_video


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


class UniWAM(FastWAM):
    """Dynamically pack active nav/manip streams through shared Wan and ActionDiT."""

    def __init__(
        self,
        *args,
        view_type_token_init_std: float = 0.02,
        view_type_token_count: int = 2,
        prefix_probability: float = 0.5,
        prefix_min: int = 1,
        prefix_max: int = 8,
        loss_lambda_nav_action: float = 1.0,
        loss_lambda_manip_action: float = 1.0,
        loss_lambda_nav_aux: float = 1.0,
        loss_lambda_manip_aux: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_lambda_nav_action = float(loss_lambda_nav_action)
        self.loss_lambda_manip_action = float(loss_lambda_manip_action)
        self.loss_lambda_nav_aux = float(loss_lambda_nav_aux)
        self.loss_lambda_manip_aux = float(loss_lambda_manip_aux)
        self.prefix_probability = float(prefix_probability)
        self.prefix_min, self.prefix_max = int(prefix_min), int(prefix_max)
        if not 0 <= self.prefix_probability <= 1:
            raise ValueError("prefix_probability must be in [0,1].")
        if not 0 <= self.prefix_min <= self.prefix_max < self.action_expert.manip_horizon:
            raise ValueError("Require 0 <= prefix_min <= prefix_max < horizon.")
        if (self.action_expert.manip_action_dim, self.action_expert.nav_action_dim) != (26, 9):
            raise ValueError("UniWAM expects 20+6 manipulation and 3+6 navigation targets.")
        if not isinstance(self.action_expert, UniWAMActionDiT):
            raise TypeError("UniWAM requires UniWAMActionDiT.")
        self.view_type_token_init_std = float(view_type_token_init_std)
        self.view_type_token_count = int(view_type_token_count)
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
        view_type_token_init_std: float = 0.02,
        view_type_token_count: int = 2,
        prefix_probability: float = 0.5,
        prefix_min: int = 1,
        prefix_max: int = 8,
        loss_lambda_nav_aux: float = 1.0,
        loss_lambda_manip_aux: float = 1.0,
    ) -> "UniWAM":
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
        action_expert = UniWAMActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        video_expert = components.dit
        if not skip_dit_load_from_pretrain and action_dit_pretrained_path is None:
            initialize_action_from_video(action_expert, video_expert)
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
            view_type_token_init_std=view_type_token_init_std,
            view_type_token_count=view_type_token_count,
            prefix_probability=prefix_probability, prefix_min=prefix_min, prefix_max=prefix_max,
            loss_lambda_nav_aux=loss_lambda_nav_aux,
            loss_lambda_manip_aux=loss_lambda_manip_aux,
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
    def _validate_video(name, video):
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"{name} must be [B,3,T,H,W].")
        if video.shape[2] % 4 != 1 or video.shape[3] % 32 or video.shape[4] % 32:
            raise ValueError(f"Invalid Wan video shape: {tuple(video.shape)}")
        return video.shape[0], video.shape[2]

    @staticmethod
    def action_attention_mask(horizon, type_count, video_prefix_len, device):
        mask = torch.zeros((type_count + horizon, video_prefix_len + type_count + horizon),
                           dtype=torch.bool, device=device)
        mask[:, :video_prefix_len + type_count] = True
        mask[type_count:, video_prefix_len + type_count:] = torch.ones(
            (horizon, horizon), dtype=torch.bool, device=device).tril()
        return mask

    def _prefix_lengths(self, batch_size, horizon, supplied=None):
        if supplied is not None:
            lengths = torch.as_tensor(supplied, dtype=torch.long, device=self.device)
            if lengths.shape != (batch_size,) or (lengths < 0).any() or (lengths >= horizon).any():
                raise ValueError("Prefix lengths must be [B], between zero and H-1.")
            return lengths
        lengths = torch.randint(self.prefix_min, self.prefix_max + 1, (batch_size,), device=self.device)
        use = torch.rand(batch_size, device=self.device) < self.prefix_probability
        return lengths * use

    @staticmethod
    def _token_timesteps(timestep, horizon, lengths):
        t = timestep[:, None].expand(-1, horizon).clone()
        if lengths is not None:
            t.masked_fill_(torch.arange(horizon, device=t.device)[None] < lengths[:, None], 0)
        return t

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
                    proprio_valid, device=self.device, dtype=torch.bool
                ).reshape(-1)
                if proprio_valid.shape != (batch_size,):
                    raise ValueError("proprio_valid must be [B].")
                context_mask[:, -1] &= proprio_valid
        return context, context_mask

    def build_inputs(self, sample: dict[str, Any], tiled: bool = False) -> dict[str, Any]:
        context_value = sample.get("context")
        if context_value is None or context_value.ndim != 3:
            raise ValueError("Cannot infer batch size from sample context.")
        batch_size = int(context_value.shape[0])
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
        video_shapes: list[tuple[int, ...]] = []
        for branch, indices in (("manip", manip_indices), ("nav", nav_indices)):
            if indices.numel() == 0:
                continue
            key = f"{branch}_video"
            video = self._select_branch_tensor(sample, key, indices, batch_size)
            if video is None:
                raise ValueError(f"Active {branch} samples require {key}.")
            self._validate_video(key, video)
            packed_videos.append(video)
            video_shapes.append(tuple(video.shape[1:]))
        if len(set(video_shapes)) != 1:
            raise ValueError(f"All active video streams must share shape, got {video_shapes}.")
        packed_video = torch.cat(packed_videos, dim=0)
        if packed_video.dtype == torch.uint8:
            packed_video = packed_video.to(self.device, torch.float32, non_blocking=True)
            packed_video.mul_(2 / 255).sub_(1)
            packed_video = packed_video.to(self.torch_dtype)
        else:
            packed_video = packed_video.to(self.device, self.torch_dtype, non_blocking=True)
        packed_latents = self._encode_video_latents(packed_video, tiled=tiled)

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
            if value is None or tuple(value.shape[1:]) != (
                horizon,
                self.action_expert.manip_action_dim,
            ):
                raise ValueError(f"manip_action shape mismatch: {getattr(value, 'shape', None)}")
            manip_action = value.to(
                self.device, self.torch_dtype, non_blocking=True
            )
            manip_feature_mask = self._branch_tensor_or_default(
                sample,
                "manip_action_feature_mask",
                manip_indices,
                batch_size,
                tail_shape=(horizon, self.action_expert.manip_action_dim),
                dtype=torch.bool,
                default=True,
            ).to(self.device)
            manip_is_pad = self._branch_tensor_or_default(
                sample,
                "action_is_pad",
                manip_indices,
                batch_size,
                tail_shape=(horizon,),
                dtype=torch.bool,
                default=False,
            ).to(self.device)
            manip_loss_valid = self._branch_tensor_or_default(
                sample,
                "manip_loss_valid",
                manip_indices,
                batch_size,
                tail_shape=(),
                dtype=torch.bool,
                default=True,
            ).to(self.device)
        if nav_count:
            value = self._select_branch_tensor(
                sample, "nav_action", nav_indices, batch_size
            )
            if value is None or tuple(value.shape[1:]) != (
                horizon,
                self.action_expert.nav_action_dim,
            ):
                raise ValueError(f"nav_action shape mismatch: {getattr(value, 'shape', None)}")
            nav_action = value.to(
                self.device, self.torch_dtype, non_blocking=True
            )
            nav_feature_mask = self._branch_tensor_or_default(
                sample,
                "nav_action_feature_mask",
                nav_indices,
                batch_size,
                tail_shape=(horizon, self.action_expert.nav_action_dim),
                dtype=torch.bool,
                default=True,
            ).to(self.device)
            nav_is_pad = self._branch_tensor_or_default(
                sample,
                "nav_action_is_pad",
                nav_indices,
                batch_size,
                tail_shape=(horizon,),
                dtype=torch.bool,
                default=False,
            ).to(self.device)
            nav_loss_valid = self._branch_tensor_or_default(
                sample,
                "nav_loss_valid",
                nav_indices,
                batch_size,
                tail_shape=(),
                dtype=torch.bool,
                default=True,
            ).to(self.device)

        image_masks = []
        video_loss_valid = []
        num_video_frames = int(packed_video.shape[2])
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
            video_loss_valid.append(
                self._branch_tensor_or_default(
                    sample, f"{branch}_video_loss_valid", indices, batch_size,
                    tail_shape=(), dtype=torch.bool, default=True,
                )
            )

        fuse = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
        return {
            "packed_input_latents": packed_latents,
            "packed_first_frame_latents": packed_latents[:, :, :1] if fuse else None,
            "fuse_vae_embedding_in_latents": fuse,
            "packed_context": packed_context,
            "packed_context_mask": packed_context_mask,
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
            "image_is_pad": torch.cat(image_masks, dim=0).to(self.device),
            "video_loss_valid": torch.cat(video_loss_valid, dim=0).to(self.device),
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

    def _packed_mot_forward(
        self,
        video_pre: dict[str, Any],
        action_pre: dict[str, Any],
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
        action_mask = self.action_attention_mask(
            self.action_expert.manip_horizon, self.action_expert.type_token_count,
            video_prefix_len, action_tokens.device)

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

    @staticmethod
    def _global_means(
        local_sums: tuple[torch.Tensor, ...], local_counts: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        counts = torch.stack([
            count.detach().to(device=local_sums[0].device, dtype=local_sums[0].dtype)
            for count in local_counts
        ])
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            world_size = dist.get_world_size()
        else:
            world_size = 1
        scales = (counts > 0).to(counts) * (world_size / counts.clamp_min(1))
        return tuple(local_sum * scales[index] for index, local_sum in enumerate(local_sums))

    def _masked_action_sum_count(
        self,
        prediction: Optional[torch.Tensor],
        target: Optional[torch.Tensor],
        feature_mask: Optional[torch.Tensor],
        is_pad: Optional[torch.Tensor],
        loss_valid: Optional[torch.Tensor],
        timestep: Optional[torch.Tensor],
        zero_anchor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prediction is None:
            return zero_anchor, zero_anchor.detach()
        assert target is not None and feature_mask is not None and timestep is not None
        element_loss = F.mse_loss(prediction.float(), target.float(), reduction="none")
        valid_feature = feature_mask.to(self.device, dtype=element_loss.dtype)
        feature_count = valid_feature.sum(dim=-1)
        token_loss = (element_loss * valid_feature).sum(dim=-1) / feature_count.clamp(min=1.0)
        token_valid = feature_count > 0
        if is_pad is not None:
            token_valid &= ~is_pad.to(self.device, dtype=torch.bool)
        valid = token_valid.to(token_loss.dtype)
        per_sample = (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        active = torch.ones_like(per_sample, dtype=torch.bool)
        if loss_valid is not None:
            active &= loss_valid.to(self.device, dtype=torch.bool)
        active &= valid.sum(dim=1) > 0
        weight = self.train_action_scheduler.training_weight(timestep).to(per_sample)
        local_sum = (per_sample * weight * active.to(per_sample.dtype)).sum() + zero_anchor
        return local_sum, active.sum().to(per_sample.dtype)

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
        inputs = self.build_inputs(sample, tiled=tiled)
        source_video = inputs["packed_input_latents"]
        active_streams = int(source_video.shape[0])
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
                return None, None, None
            assert mask is not None
            source = torch.where(mask, source, torch.zeros_like(source))
            noise = torch.randn_like(source)
            timestep = self.train_action_scheduler.sample_training_t(
                batch_size=source.shape[0], device=self.device, dtype=source.dtype
            )
            noisy = self.train_action_scheduler.add_noise(source, noise, timestep)
            target = self.train_action_scheduler.training_target(source, noise, timestep)
            noisy = torch.where(mask, noisy, torch.zeros_like(noisy))
            target = torch.where(mask, target, torch.zeros_like(target))
            return noisy, target, timestep

        noisy_manip, target_manip, timestep_manip = noisy_action(
            inputs["manip_action"], inputs["manip_feature_mask"]
        )
        noisy_nav, target_nav, timestep_nav = noisy_action(
            inputs["nav_action"], inputs["nav_feature_mask"]
        )

        token_times = {}
        for branch, noisy, timestep in (("manip", noisy_manip, timestep_manip), ("nav", noisy_nav, timestep_nav)):
            if noisy is None:
                token_times[branch] = None
                continue
            lengths = self._prefix_lengths(noisy.shape[0], noisy.shape[1], sample.get(f"{branch}_prefix_lengths"))
            prefix = torch.arange(noisy.shape[1], device=self.device)[None] < lengths[:, None]
            clean = torch.where(inputs[f"{branch}_feature_mask"], inputs[f"{branch}_action"], 0)
            noisy.copy_(torch.where(prefix[..., None], clean, noisy))
            inputs[f"{branch}_is_pad"] = inputs[f"{branch}_is_pad"] | prefix
            token_times[branch] = self._token_timesteps(timestep, noisy.shape[1], lengths)

        video_pre_base = self.video_expert.pre_dit(
            x=noisy_video,
            timestep=timestep_video,
            context=inputs["packed_context"],
            context_mask=inputs["packed_context_mask"],
            action=None,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        video_pre = self._prepend_view_type_token(video_pre_base, inputs["stream_kind"])
        manip_count = int(inputs["manip_count"])
        nav_count = int(inputs["nav_count"])
        action_pre = self.action_expert.pre_dit_packed(
            manip_action=noisy_manip,
            nav_action=noisy_nav,
            manip_timestep=token_times["manip"],
            nav_timestep=token_times["nav"],
            manip_context=(inputs["packed_context"][:manip_count] if manip_count else None),
            nav_context=(inputs["packed_context"][manip_count:] if nav_count else None),
            manip_context_mask=(
                inputs["packed_context_mask"][:manip_count] if manip_count else None
            ),
            nav_context_mask=(
                inputs["packed_context_mask"][manip_count:] if nav_count else None
            ),
        )
        output = self._packed_mot_forward(video_pre, action_pre)
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
        video_valid = inputs["video_loss_valid"] & ~inputs["image_is_pad"][:, 1:].all(dim=1)
        weighted_video = video_per_sample * video_weight * video_valid.to(video_per_sample)
        stream_kind = inputs["stream_kind"]
        manip_video_mask = (stream_kind == MANIP_BRANCH) & video_valid
        nav_video_mask = (stream_kind == NAV_BRANCH) & video_valid

        anchor = self._parameter_anchor()
        sums, counts, names, coefficients = [weighted_video.sum()], [video_valid.sum()], ["video"], [self.loss_lambda_video]
        for branch, target, timestep, physical_dim in (("manip", target_manip, timestep_manip, 20), ("nav", target_nav, timestep_nav, 3)):
            prediction = action_output[branch]
            for group, interval in (("action", slice(0, physical_dim)), ("aux", slice(physical_dim, None))):
                feature_mask = inputs[f"{branch}_feature_mask"]
                total, count = self._masked_action_sum_count(
                    None if prediction is None else prediction[..., interval],
                    None if target is None else target[..., interval],
                    None if feature_mask is None else feature_mask[..., interval],
                    inputs[f"{branch}_is_pad"], inputs[f"{branch}_loss_valid"], timestep, anchor)
                sums.append(total); counts.append(count)
                names.append(f"{branch}_{group}")
                coefficients.append(getattr(self, f"loss_lambda_{branch}_{group}"))
        means = self._global_means(tuple(sums), tuple(counts))
        loss = sum(weight * mean for weight, mean in zip(coefficients, means)) + anchor
        return loss, {f"loss_{name}": value.detach() for name, value in zip(names, means)}

    def _predict_action_with_cache(
        self,
        *,
        latents_manip: Optional[torch.Tensor],
        latents_nav: Optional[torch.Tensor],
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_cache: list[dict[str, torch.Tensor]],
        video_seq_len: int,
        video_prefix_len: int,
        prefix_lengths: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, Optional[torch.Tensor]]:
        manip_count = 0 if latents_manip is None else int(latents_manip.shape[0])
        nav_count = 0 if latents_nav is None else int(latents_nav.shape[0])
        action_pre = self.action_expert.pre_dit_packed(
            manip_action=latents_manip,
            nav_action=latents_nav,
            manip_timestep=(self._token_timesteps(timestep.expand(manip_count), self.action_expert.manip_horizon, (prefix_lengths or {}).get("manip")) if manip_count else None),
            nav_timestep=(self._token_timesteps(timestep.expand(nav_count), self.action_expert.nav_horizon, (prefix_lengths or {}).get("nav")) if nav_count else None),
            manip_context=(context[:manip_count] if manip_count else None),
            nav_context=(context[manip_count:] if nav_count else None),
            manip_context_mask=(context_mask[:manip_count] if manip_count else None),
            nav_context_mask=(context_mask[manip_count:] if nav_count else None),
        )
        action_seq_len = int(action_pre["tokens"].shape[1])
        joint_mask = torch.zeros(
            (video_seq_len + action_seq_len, video_seq_len + action_seq_len),
            dtype=torch.bool,
            device=self.device,
        )
        joint_mask[:video_seq_len, :video_seq_len] = True
        local_mask = self.action_attention_mask(
            self.action_expert.manip_horizon, self.action_expert.type_token_count,
            video_prefix_len, self.device)
        joint_mask[video_seq_len:, :video_prefix_len] = local_mask[:, :video_prefix_len]
        joint_mask[video_seq_len:, video_seq_len:] = local_mask[:, video_prefix_len:]
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
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        nav_input_image: Optional[torch.Tensor] = None,
        manip_input_image: Optional[torch.Tensor] = None,
        num_inference_steps: int = 10,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        committed_prefix: Optional[dict[str, torch.Tensor]] = None,
        action_feature_mask: Optional[dict[str, torch.Tensor]] = None,
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
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        context = context.to(self.device, self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(self.device, torch.bool, non_blocking=True)
        if proprio is not None:
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(self.device, self.torch_dtype, non_blocking=True),
            )

        def prepare_image(name: str, image: torch.Tensor) -> torch.Tensor:
            if image.ndim == 3:
                image = image.unsqueeze(0)
            if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
                raise ValueError(f"{name} must be [1,3,H,W], got {tuple(image.shape)}.")
            if image.shape[-2] % 32 or image.shape[-1] % 32:
                raise ValueError(f"{name} spatial dimensions must be divisible by 32.")
            if image.dtype == torch.uint8:
                image = image.float().mul(2 / 255).sub(1)
            elif not torch.isfinite(image).all() or image.min() < -1 or image.max() > 1:
                raise ValueError("Float input images must be finite and in [-1,1].")
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
        image_latents = self._encode_video_latents(packed_images.unsqueeze(2), tiled=tiled)
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
        prefixes, masks, prefix_lengths = {}, {}, {}
        for branch, latent in (("manip", latents_manip), ("nav", latents_nav)):
            if latent is None:
                continue
            dim = latent.shape[-1]
            mask = (action_feature_mask or {}).get(branch, torch.ones(action_horizon, dim))
            mask = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
            if mask.shape != (action_horizon, dim):
                raise ValueError("Inference feature masks must be [H,D].")
            masks[branch] = mask[None]
            prefix = (committed_prefix or {}).get(branch, torch.empty(0, dim))
            prefix = torch.as_tensor(prefix, device=self.device, dtype=self.torch_dtype)
            if prefix.ndim != 2 or prefix.shape[1] != dim or prefix.shape[0] >= action_horizon or not torch.isfinite(prefix).all():
                raise ValueError("Committed prefix must be finite normalized [P,D], with P < H.")
            prefixes[branch] = torch.where(mask[:len(prefix)], prefix, 0)
            prefix_lengths[branch] = torch.tensor([len(prefix)], device=self.device)
            latent.masked_fill_(~masks[branch], 0)
            latent[:, :len(prefix)] = prefixes[branch]

        def clamp_branch(branch, latent):
            if latent is not None:
                latent.masked_fill_(~masks[branch], 0)
                latent[:, :len(prefixes[branch])] = prefixes[branch]
            return latent

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
                context=packed_context,
                context_mask=packed_context_mask,
                video_cache=video_cache,
                video_seq_len=video_seq_len,
                video_prefix_len=video_prefix_len,
                prefix_lengths=prefix_lengths,
            )
            if latents_manip is not None:
                assert prediction["manip"] is not None
                latents_manip = self.infer_action_scheduler.step(
                    prediction["manip"], step_delta, latents_manip
                )
            if latents_nav is not None:
                assert prediction["nav"] is not None
                latents_nav = self.infer_action_scheduler.step(
                    prediction["nav"], step_delta, latents_nav
                )
            latents_manip = clamp_branch("manip", latents_manip)
            latents_nav = clamp_branch("nav", latents_nav)
        result: dict[str, Any] = {}
        if latents_manip is not None:
            result["manip_action"] = latents_manip[0].float().cpu()
        if latents_nav is not None:
            result["nav_action"] = latents_nav[0].float().cpu()
        return result

    def forward(self, sample, tiled: bool = False, **kwargs):
        del kwargs
        return self.training_loss(sample, tiled=tiled)
