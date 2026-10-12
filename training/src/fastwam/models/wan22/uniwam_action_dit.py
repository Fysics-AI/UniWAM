from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from fastwam.utils.logging_config import get_logger

from .wan_video_dit import DiTBlock, precompute_freqs_cis, sinusoidal_embedding_1d

logger = get_logger(__name__)


class _StreamBackbone(nn.Module):
    """One ActionDiT backbone with heterogeneous manipulation/navigation tokens."""

    BACKBONE_SKIP_PREFIXES = (
        "manip_action_encoder.",
        "nav_action_encoder.",
        "manip_head.",
        "nav_head.",
    )
    BACKBONE_META_KEYS = (
        "hidden_dim",
        "ffn_dim",
        "num_layers",
        "num_heads",
        "attn_head_dim",
        "text_dim",
        "freq_dim",
        "eps",
    )

    def __init__(
        self,
        hidden_dim: int,
        manip_action_dim: int,
        nav_action_dim: int,
        manip_horizon: int,
        nav_horizon: int,
        ffn_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.manip_action_dim = int(manip_action_dim)
        self.nav_action_dim = int(nav_action_dim)
        self.manip_horizon = int(manip_horizon)
        self.nav_horizon = int(nav_horizon)
        self.ffn_dim = int(ffn_dim)
        self.text_dim = int(text_dim)
        self.freq_dim = int(freq_dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        if min(self.manip_horizon, self.nav_horizon) <= 0:
            raise ValueError("Both action horizons must be positive.")
        if self.attn_head_dim % 2 != 0:
            raise ValueError("attn_head_dim must be even for RoPE.")

        self.manip_action_encoder = nn.Linear(self.manip_action_dim, self.hidden_dim)
        self.nav_action_encoder = nn.Linear(self.nav_action_dim, self.hidden_dim)
        self.text_embedding = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim * 6),
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=self.hidden_dim,
                    attn_head_dim=self.attn_head_dim,
                    num_heads=self.num_heads,
                    ffn_dim=self.ffn_dim,
                    eps=eps,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.manip_head = nn.Linear(self.hidden_dim, self.manip_action_dim)
        self.nav_head = nn.Linear(self.hidden_dim, self.nav_action_dim)
        self.freqs = precompute_freqs_cis(self.attn_head_dim, end=1024)

    @property
    def total_horizon(self) -> int:
        return self.manip_horizon + self.nav_horizon

    @classmethod
    def from_pretrained(
        cls,
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "_StreamBackbone":
        expert = cls(**dict(action_dit_config)).to(device=device, dtype=torch_dtype)
        if skip_dit_load_from_pretrain:
            logger.info("Skipping partitioned ActionDiT pretrained backbone load.")
            return expert
        if not action_dit_pretrained_path:
            logger.info("No ActionDiT backbone path; partitioned ActionDiT is randomly initialized.")
            return expert

        path = Path(action_dit_pretrained_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("backbone_state_dict"), dict):
            raise ValueError(f"Invalid ActionDiT backbone payload: {path}")

        config = dict(action_dit_config)
        meta = payload.get("meta", {})
        expected_meta = {
            "hidden_dim": int(config["hidden_dim"]),
            "ffn_dim": int(config["ffn_dim"]),
            "num_layers": int(config["num_layers"]),
            "num_heads": int(config["num_heads"]),
            "attn_head_dim": int(config["attn_head_dim"]),
            "text_dim": int(config["text_dim"]),
            "freq_dim": int(config["freq_dim"]),
            "eps": float(config["eps"]),
        }
        for key in cls.BACKBONE_META_KEYS:
            if key not in meta:
                raise ValueError(f"meta.{key} missing in {path}")
            if key == "eps":
                matches = abs(float(meta[key]) - float(expected_meta[key])) <= 1e-12
            else:
                matches = int(meta[key]) == int(expected_meta[key])
            if not matches:
                raise ValueError(
                    f"meta.{key} mismatch: expected {expected_meta[key]}, got {meta[key]}"
                )

        state = expert.state_dict()
        expected_keys = {
            key
            for key in state
            if not any(key.startswith(prefix) for prefix in cls.BACKBONE_SKIP_PREFIXES)
        }
        provided = payload["backbone_state_dict"]
        if expected_keys != set(provided):
            missing = sorted(expected_keys - set(provided))
            unexpected = sorted(set(provided) - expected_keys)
            raise ValueError(
                f"ActionDiT backbone keys mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}"
            )
        for key in expected_keys:
            source = provided[key]
            target = state[key]
            if tuple(source.shape) != tuple(target.shape):
                raise ValueError(f"Backbone shape mismatch for {key}: {source.shape} vs {target.shape}")
            state[key] = source.to(device=target.device, dtype=target.dtype)
        expert.load_state_dict(state, strict=True)
        logger.info(
            "Loaded one shared partitioned ActionDiT backbone from %s (keys=%d).",
            path,
            len(expected_keys),
        )
        return expert

    @staticmethod
    def _validate_timestep(timestep: torch.Tensor, batch_size: int, name: str) -> torch.Tensor:
        if timestep.ndim != 1 or timestep.shape[0] not in (1, batch_size):
            raise ValueError(f"{name} must be [1] or [B], got {tuple(timestep.shape)}.")
        if timestep.shape[0] == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)
        return timestep



class UniWAMActionDiT(_StreamBackbone):
    """Shared ActionDiT with dynamically packed manipulation/navigation streams."""

    BACKBONE_SKIP_PREFIXES = _StreamBackbone.BACKBONE_SKIP_PREFIXES + (
        "manip_type_token",
        "nav_type_token",
    )

    def __init__(
        self,
        *args,
        type_token_init_std: float = 0.02,
        type_token_count: int = 2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.type_token_init_std = float(type_token_init_std)
        self.type_token_count = int(type_token_count)
        if self.type_token_count <= 0:
            raise ValueError("type_token_count must be positive.")
        self.manip_type_token = nn.Parameter(
            torch.empty(1, self.type_token_count, self.hidden_dim)
        )
        self.nav_type_token = nn.Parameter(
            torch.empty(1, self.type_token_count, self.hidden_dim)
        )
        nn.init.normal_(self.manip_type_token, std=self.type_token_init_std)
        nn.init.normal_(self.nav_type_token, std=self.type_token_init_std)

    def _branch_pre_dit(
        self,
        *,
        branch: str,
        action: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if branch == "manip":
            horizon = self.manip_horizon
            action_dim = self.manip_action_dim
            encoder = self.manip_action_encoder
            type_token = self.manip_type_token
        elif branch == "nav":
            horizon = self.nav_horizon
            action_dim = self.nav_action_dim
            encoder = self.nav_action_encoder
            type_token = self.nav_type_token
        else:
            raise ValueError(f"Unsupported action branch: {branch!r}")

        if action.ndim != 3 or tuple(action.shape[1:]) != (horizon, action_dim):
            raise ValueError(
                f"{branch}_action must be [B,{horizon},{action_dim}], got {tuple(action.shape)}."
            )
        batch_size = int(action.shape[0])
        if timestep.ndim == 1:
            timestep = self._validate_timestep(timestep, batch_size, f"{branch}_timestep")
            timestep = timestep[:, None].expand(-1, horizon)
        if timestep.shape != (batch_size, horizon):
            raise ValueError("Action timesteps must be [B] or [B,H].")
        if context.ndim != 3 or context.shape[0] != batch_size:
            raise ValueError(f"{branch}_context batch mismatch: {tuple(context.shape)}")
        if tuple(context_mask.shape) != tuple(context.shape[:2]):
            raise ValueError(
                f"{branch}_context_mask must match [B,L], got {tuple(context_mask.shape)}."
            )

        action_tokens = encoder(action)
        prefix = type_token.to(device=action_tokens.device, dtype=action_tokens.dtype).expand(
            batch_size, -1, -1
        )
        tokens = torch.cat((prefix, action_tokens), dim=1)

        # Type tokens are clean; prefix action positions also receive t=0.
        token_t = torch.cat((timestep.new_zeros(batch_size, self.type_token_count), timestep), dim=1)
        time = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, token_t.flatten()))
        time = time.view(batch_size, -1, self.hidden_dim)
        modulation = self.time_projection(time).unflatten(-1, (6, self.hidden_dim))
        return {
            "tokens": tokens,
            "t": time,
            "t_mod": modulation,
            "context": context,
            "context_mask": context_mask,
        }

    def pre_dit_packed(
        self,
        *,
        manip_action: Optional[torch.Tensor],
        nav_action: Optional[torch.Tensor],
        manip_timestep: Optional[torch.Tensor],
        nav_timestep: Optional[torch.Tensor],
        manip_context: Optional[torch.Tensor],
        nav_context: Optional[torch.Tensor],
        manip_context_mask: Optional[torch.Tensor],
        nav_context_mask: Optional[torch.Tensor],
    ) -> dict[str, Any]:
        """Pack active branches along batch; branch order is always manip then nav."""

        chunks: list[dict[str, torch.Tensor]] = []
        branch_sizes: dict[str, int] = {"manip": 0, "nav": 0}
        branch_inputs = (
            ("manip", manip_action, manip_timestep, manip_context, manip_context_mask),
            ("nav", nav_action, nav_timestep, nav_context, nav_context_mask),
        )
        for branch, action, timestep, context, context_mask in branch_inputs:
            values = (action, timestep, context, context_mask)
            if all(value is None for value in values):
                continue
            if any(value is None for value in values):
                raise ValueError(f"Active {branch} branch requires action/timestep/context/context_mask.")
            assert action is not None
            assert timestep is not None
            assert context is not None
            assert context_mask is not None
            if action.shape[0] == 0:
                continue
            chunk = self._branch_pre_dit(
                branch=branch,
                action=action,
                timestep=timestep,
                context=context,
                context_mask=context_mask,
            )
            chunks.append(chunk)
            branch_sizes[branch] = int(action.shape[0])

        if not chunks:
            raise ValueError("At least one manipulation or navigation stream must be active.")
        if self.manip_horizon != self.nav_horizon:
            raise ValueError("Packed mixed-stream ActionDiT currently requires equal branch horizons.")

        horizon = self.manip_horizon
        temporal_freqs = self.freqs[:horizon].view(horizon, 1, -1)
        type_freq = torch.ones_like(temporal_freqs[:1]).expand(
            self.type_token_count, -1, -1
        )
        freqs = torch.cat((type_freq, temporal_freqs), dim=0).to(chunks[0]["tokens"].device)
        context = torch.cat([chunk["context"] for chunk in chunks], dim=0)
        context_mask_base = torch.cat([chunk["context_mask"] for chunk in chunks], dim=0)
        return {
            "tokens": torch.cat([chunk["tokens"] for chunk in chunks], dim=0),
            "freqs": freqs,
            "t": torch.cat([chunk["t"] for chunk in chunks], dim=0),
            "t_mod": torch.cat([chunk["t_mod"] for chunk in chunks], dim=0),
            "context": self.text_embedding(context),
            "context_mask": context_mask_base.unsqueeze(1).expand(
                -1, horizon + self.type_token_count, -1
            ),
            "meta": {
                "batch_size": sum(branch_sizes.values()),
                "seq_len": horizon + self.type_token_count,
                "horizon": horizon,
                "type_token_count": self.type_token_count,
                "branch_sizes": branch_sizes,
                "branch_order": ("manip", "nav"),
            },
        }

    def post_dit_packed(
        self,
        tokens: torch.Tensor,
        pre_state: dict[str, Any],
    ) -> dict[str, Optional[torch.Tensor]]:
        meta = pre_state["meta"]
        horizon = int(meta["horizon"])
        type_token_count = int(meta["type_token_count"])
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (
            horizon + type_token_count,
            self.hidden_dim,
        ):
            raise ValueError(f"Packed action token shape mismatch: {tuple(tokens.shape)}")
        branch_sizes = dict(meta["branch_sizes"])
        manip_size = int(branch_sizes["manip"])
        nav_size = int(branch_sizes["nav"])
        if manip_size + nav_size != tokens.shape[0]:
            raise ValueError("Packed action branch sizes do not match token batch.")

        manip_output = None
        nav_output = None
        cursor = 0
        if manip_size:
            manip_output = self.manip_head(
                tokens[cursor : cursor + manip_size, type_token_count:]
            )
            cursor += manip_size
        if nav_size:
            nav_output = self.nav_head(tokens[cursor : cursor + nav_size, type_token_count:])
        return {"manip": manip_output, "nav": nav_output}
