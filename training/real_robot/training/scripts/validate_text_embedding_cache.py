#!/usr/bin/env python3
"""Verify that a task's complete prompt cache is present and well formed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, ListConfig

from uniwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def cache_prompt_pairs(node, pairs: dict[Path, set[str]]) -> None:
    if isinstance(node, DictConfig):
        dirs = node.get("dataset_dirs")
        if dirs is not None:
            cache = node.get("text_embedding_cache_dir")
            if cache is None:
                raise ValueError("Dataset node is missing text_embedding_cache_dir")
            path = Path(str(cache))
            prompts = pairs.setdefault(path, set())
            for dataset_dir in dirs:
                tasks_path = Path(str(dataset_dir)) / "meta/tasks.jsonl"
                for line in tasks_path.read_text(encoding="utf-8").splitlines():
                    if line:
                        task = str(json.loads(line)["task"])
                        prefix = str(node.get("instruction_prefix") or "").strip()
                        if prefix:
                            task = f"{prefix} {task}"
                        prompts.add(DEFAULT_PROMPT.format(task=task))
            variants_path = node.get("instruction_variants_path")
            if variants_path is not None:
                path = Path(str(variants_path))
                if not path.is_file():
                    raise FileNotFoundError(path)
                for line_number, line in enumerate(
                    path.read_text(encoding="utf-8").splitlines(), start=1
                ):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    for field in ("detailed_prompt", "simple_prompt"):
                        task = str(record[field]).strip()
                        if not task:
                            raise ValueError(f"Empty {field} at {path}:{line_number}")
                        prompts.add(DEFAULT_PROMPT.format(task=task))
        for _, value in node.items():
            cache_prompt_pairs(value, pairs)
    elif isinstance(node, (ListConfig, list, tuple)):
        for value in node:
            cache_prompt_pairs(value, pairs)


def main() -> int:
    args = parse_args()
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=[f"task={args.task}"])
    pairs: dict[Path, set[str]] = {}
    cache_prompt_pairs(cfg.data.train, pairs)
    if not pairs or not any(pairs.values()):
        raise RuntimeError("The task must expose at least one dataset cache and one prompt.")
    model_name = str(cfg.model.model_id).split("/")[-1].lower()
    encoder_id = "".join(character for character in model_name if character.isalnum())
    rows = []
    for cache_dir, prompts in pairs.items():
        for prompt in sorted(prompts):
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            name = f"{digest}.t5_len{int(cfg.data.train.datasets[0].dataset.context_len)}.{encoder_id}.pt"
            path = cache_dir / name
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = torch.load(path, map_location="cpu")
            context = torch.as_tensor(payload["context"])
            mask = torch.as_tensor(payload["mask"])
            if context.ndim != 2 or context.shape[0] != 128 or context.shape[1] != 4096:
                raise ValueError(f"Unexpected context shape in {path}: {tuple(context.shape)}")
            if mask.shape != (128,) or mask.dtype != torch.bool:
                raise ValueError(f"Unexpected mask in {path}: {tuple(mask.shape)} {mask.dtype}")
            if not torch.isfinite(context.float()).all():
                raise ValueError(f"Non-finite context in {path}")
            rows.append({"prompt": prompt, "cache": str(path), "context_shape": list(context.shape)})
    report = {
        "status": "PASS",
        "cache_dirs": [str(path) for path in pairs],
        "prompt_counts_by_cache": {str(path): len(prompts) for path, prompts in pairs.items()},
        "entries": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
