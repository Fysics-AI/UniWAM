#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from uniwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from uniwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
from uniwam.utils.config_resolvers import register_default_resolvers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Prompt cache encoding requires CUDA")
    register_default_resolvers()
    prompts = [json.loads(line)["prompt"] for line in args.manifest.read_text().splitlines() if line.strip()]
    args.output.mkdir(parents=True, exist_ok=True)
    model_id = "Wan-AI/Wan2.2-TI2V-5B"
    tokenizer_model_id = "Wan-AI/Wan2.1-T2V-1.3B"
    _, text_config, _, tokenizer_config = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=True,
    )
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()
    device = "cuda"
    encoder = _load_registered_model(text_config.path, "wan_video_text_encoder", torch_dtype=torch.bfloat16, device=device).eval()
    tokenizer = HuggingfaceTokenizer(name=tokenizer_config.path, seq_len=128, clean="whitespace")
    encoded = 0
    skipped = 0
    with torch.no_grad():
        for start in range(0, len(prompts), args.batch_size):
            batch = prompts[start:start + args.batch_size]
            pending = []
            for prompt in batch:
                digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                path = args.output / f"{digest}.t5_len128.wan22ti2v5b.pt"
                if path.exists() and not args.overwrite:
                    skipped += 1
                else:
                    pending.append((prompt, path))
            if not pending:
                continue
            ids, mask = tokenizer([item[0] for item in pending], return_mask=True, add_special_tokens=True)
            context = encoder(ids.to(device), mask.to(device=device, dtype=torch.bool))
            for index, (_, path) in enumerate(pending):
                payload = {
                    "context": context[index].detach().to("cpu", dtype=torch.bfloat16).contiguous(),
                    "mask": mask[index].detach().to("cpu", dtype=torch.bool).contiguous(),
                }
                temporary = path.with_suffix(path.suffix + ".tmp")
                torch.save(payload, temporary)
                os.replace(temporary, path)
                encoded += 1
    report = {"status": "PASS", "prompt_count": len(prompts), "encoded": encoded, "skipped": skipped, "output": str(args.output)}
    (args.output / "cache_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
