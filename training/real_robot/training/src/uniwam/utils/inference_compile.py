"""Select compiled inference compatible with the active CUDA allocator."""

from __future__ import annotations

import os
import re

import torch

from .logging_config import get_logger

logger = get_logger(__name__)


def _cuda_graphs_compatible(device: torch.device | str) -> bool:
    if torch.device(device).type != "cuda":
        return False
    try:
        settings = torch.cuda.memory._snapshot()["allocator_settings"]
        if "expandable_segments" in settings:
            return not bool(settings["expandable_segments"])
    except (AttributeError, KeyError, RuntimeError, TypeError, AssertionError):
        pass

    # Older PyTorch builds may not expose active allocator settings. A declared
    # expandable allocator is enough to select the conservative compiled path.
    for variable in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        values = re.findall(
            r"(?:^|,)\s*expandable_segments\s*:\s*(true|false)\s*(?=,|$)",
            os.environ.get(variable, ""),
            flags=re.IGNORECASE,
        )
        if values and values[-1].lower() == "true":
            return False
    return True


def compile_inference(function, *, device: torch.device | str, vae: bool = False):
    if _cuda_graphs_compatible(device):
        if vae:
            return torch.compile(function, backend="cudagraphs", fullgraph=True)
        return torch.compile(function, mode="reduce-overhead", fullgraph=True)
    if torch.device(device).type == "cuda":
        logger.warning(
            "Using TorchInductor inference without CUDA Graph capture because "
            "expandable CUDA allocator segments are enabled."
        )
    return torch.compile(
        function,
        backend="inductor",
        fullgraph=True,
        options={"triton.cudagraphs": False},
    )
