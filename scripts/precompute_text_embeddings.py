#!/usr/bin/env python3
"""Generate or verify the exact T5 cache format consumed by the UniWAM server."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

PROMPT_TEMPLATE = "A video recorded from a robot's point of view executing the following instruction: {task}"
SUFFIX = ".t5_len128.wan22ti2v5b.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--instructions", type=Path, help="UTF-8 file, one instruction per line")
    source.add_argument("--benchmark", type=Path, help="MAP-Bench evaluation/benchmark_15k.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--component-root", type=Path, default=Path("../UniWAM-weights/components/wan2.2"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.benchmark:
        data = json.loads(args.benchmark.read_text())
        tasks = sorted({group["task"] for group in data["groups"]})
    else:
        tasks = list(dict.fromkeys(line for line in args.instructions.read_text().splitlines() if line.strip()))
    if not tasks:
        parser.error("No instructions found")
    entries = [(PROMPT_TEMPLATE.format(task=task), task) for task in tasks]
    paths = [(prompt, args.output_dir / (hashlib.sha256(prompt.encode()).hexdigest() + SUFFIX)) for prompt, _ in entries]
    import torch
    if args.check_only:
        missing = []
        for prompt, path in paths:
            if not path.is_file():
                missing.append(path.name)
                continue
            payload = torch.load(path, map_location="cpu", weights_only=True)
            context, mask = payload["context"], payload["mask"]
            if context.shape != (128, 4096) or mask.shape != (128,) or not torch.isfinite(context).all():
                raise ValueError(f"Invalid feature shape or values: {path.name}")
            if context.dtype != torch.bfloat16 or mask.dtype != torch.bool:
                raise ValueError(f"Invalid feature dtype: {path.name}")
        if missing:
            raise SystemExit(f"Missing {len(missing)}/{len(paths)} features: " + ", ".join(missing[:5]))
        print(f"Verified {len(paths)} distinct instruction features")
        return
    pending = [(prompt, path) for prompt, path in paths if args.overwrite or not path.is_file()]
    if not pending:
        print(f"All {len(paths)} features already exist; use --check-only to validate or --overwrite to regenerate")
        return
    code_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(code_root / "runtime/uniwam_src"))
    from fastwam.models.wan22.helpers.loader import _load_registered_model
    from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
    upstream = args.component_root
    weights = upstream / "text_encoder.safetensors"
    tokenizer_dir = upstream / "tokenizer"
    if not weights.is_file() or not tokenizer_dir.is_dir():
        raise SystemExit("Missing T5 assets. Run: python scripts/download_assets.py --text-encoder")
    tokenizer = HuggingfaceTokenizer(str(tokenizer_dir), seq_len=128, clean="whitespace", local_files_only=True)
    encoder = _load_registered_model(str(weights), "wan_video_text_encoder", torch.bfloat16, args.device).eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            ids, mask = tokenizer([item[0] for item in batch], return_mask=True, add_special_tokens=True)
            ids, mask = ids.to(args.device), mask.to(args.device, dtype=torch.bool)
            context = encoder(ids, mask).to(dtype=torch.bfloat16)
            context = context.masked_fill(~mask[..., None], 0)
            for index, (_, path) in enumerate(batch):
                # Clone each slice so torch.save does not serialize other prompts' storage.
                payload = {"context": context[index].cpu().clone(), "mask": mask[index].cpu().clone()}
                temp = path.with_suffix(".tmp")
                torch.save(payload, temp)
                os.replace(temp, path)
            print(f"Encoded {min(start + args.batch_size, len(pending))}/{len(pending)}", flush=True)


if __name__ == "__main__":
    main()
