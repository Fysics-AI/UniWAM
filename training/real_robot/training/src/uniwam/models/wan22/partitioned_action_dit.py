from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from uniwam.utils.logging_config import get_logger

from .wan_video_dit import DiTBlock, precompute_freqs_cis, sinusoidal_embedding_1d

logger = get_logger(__name__)


class PartitionedActionDiT(nn.Module):
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

    def _apply(self, fn):
        result = super()._apply(fn)
        device = next(self.parameters()).device
        self.freqs = self.freqs.to(device=device)
        return result

    def get_freqs(self, seq_len: int) -> torch.Tensor:
        return self.freqs[:seq_len].view(seq_len, 1, -1)

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
    ) -> "PartitionedActionDiT":
        expert = cls(**dict(action_dit_config)).to(device=device, dtype=torch_dtype)
        if skip_dit_load_from_pretrain:
            logger.info("Skipping partitioned ActionDiT pretrained backbone load.")
            return expert
        if not action_dit_pretrained_path:
            logger.info("No ActionDiT backbone path; partitioned ActionDiT is randomly initialized.")
            return expert

        path = Path(action_dit_pretrained_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[4] / path
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu")
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

    def pre_dit(
        self,
        manip_action: torch.Tensor,
        nav_action: torch.Tensor,
        manip_timestep: torch.Tensor,
        nav_timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        if manip_action.ndim != 3 or tuple(manip_action.shape[1:]) != (
            self.manip_horizon,
            self.manip_action_dim,
        ):
            raise ValueError(
                f"manip_action must be [B,{self.manip_horizon},{self.manip_action_dim}], "
                f"got {tuple(manip_action.shape)}."
            )
        if nav_action.ndim != 3 or tuple(nav_action.shape[1:]) != (
            self.nav_horizon,
            self.nav_action_dim,
        ):
            raise ValueError(
                f"nav_action must be [B,{self.nav_horizon},{self.nav_action_dim}], "
                f"got {tuple(nav_action.shape)}."
            )
        batch_size = int(manip_action.shape[0])
        if nav_action.shape[0] != batch_size or context.ndim != 3 or context.shape[0] != batch_size:
            raise ValueError("Batch mismatch across partitioned action inputs and text context.")
        manip_timestep = self._validate_timestep(manip_timestep, batch_size, "manip_timestep")
        nav_timestep = self._validate_timestep(nav_timestep, batch_size, "nav_timestep")
        if context_mask is None:
            context_mask = torch.ones(
                (batch_size, context.shape[1]), dtype=torch.bool, device=context.device
            )
        if tuple(context_mask.shape) != tuple(context.shape[:2]):
            raise ValueError("context_mask must match context [B,L].")

        manip_tokens = self.manip_action_encoder(manip_action)
        nav_tokens = self.nav_action_encoder(nav_action)
        tokens = torch.cat((manip_tokens, nav_tokens), dim=1)

        def time_state(timestep: torch.Tensor, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
            time = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
            modulation = self.time_projection(time).unflatten(1, (6, self.hidden_dim))
            return time, modulation.unsqueeze(1).expand(-1, horizon, -1, -1)

        manip_time, manip_modulation = time_state(manip_timestep, self.manip_horizon)
        nav_time, nav_modulation = time_state(nav_timestep, self.nav_horizon)
        t_mod = torch.cat((manip_modulation, nav_modulation), dim=1)

        context_emb = self.text_embedding(context)
        context_attn_mask = context_mask.unsqueeze(1).expand(-1, self.total_horizon, -1)
        freqs = torch.cat(
            (self.freqs[: self.manip_horizon], self.freqs[: self.nav_horizon]), dim=0
        ).view(self.total_horizon, 1, -1).to(tokens.device)
        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": {"manip": manip_time, "nav": nav_time},
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": context_attn_mask,
            "meta": {
                "batch_size": batch_size,
                "seq_len": self.total_horizon,
                "manip_horizon": self.manip_horizon,
                "nav_horizon": self.nav_horizon,
            },
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: dict[str, Any]) -> dict[str, torch.Tensor]:
        if tuple(tokens.shape[1:]) != (self.total_horizon, self.hidden_dim):
            raise ValueError(f"Partitioned token output shape mismatch: {tuple(tokens.shape)}.")
        manip_tokens = tokens[:, : self.manip_horizon]
        nav_tokens = tokens[:, self.manip_horizon :]
        return {
            "manip": self.manip_head(manip_tokens),
            "nav": self.nav_head(nav_tokens),
        }
