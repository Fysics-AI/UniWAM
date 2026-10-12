from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from .partitioned_action_dit import PartitionedActionDiT
from .wan_video_dit import sinusoidal_embedding_1d


class MixedStreamActionDiT(PartitionedActionDiT):
    """Shared ActionDiT with dynamically packed manipulation/navigation streams."""

    BACKBONE_SKIP_PREFIXES = PartitionedActionDiT.BACKBONE_SKIP_PREFIXES + (
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
        timestep = self._validate_timestep(timestep, batch_size, f"{branch}_timestep")
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

        time = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        modulation = self.time_projection(time).unflatten(1, (6, self.hidden_dim))
        modulation = modulation.unsqueeze(1).expand(
            -1, horizon + self.type_token_count, -1, -1
        )
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
        temporal_freqs = self.get_freqs(horizon)
        type_freq = torch.ones_like(temporal_freqs[:1]).expand(
            self.type_token_count, -1, -1
        )
        freqs = torch.cat((type_freq, temporal_freqs), dim=0)
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
