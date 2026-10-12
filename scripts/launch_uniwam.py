"""Serve a released UniWAM checkpoint from a local model checkout."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("base", "scaled"), required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--ports", default="18144,18145")
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/policies"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    code_root = Path(__file__).resolve().parents[1]
    model_root = args.model_root.expanduser().resolve()
    manifest = json.loads((model_root / "weights/weights_manifest.json").read_text())[args.model]
    gpus, ports = args.gpus.split(","), args.ports.split(",")
    if len(gpus) != len(ports) or not gpus:
        raise SystemExit("--gpus and --ports need the same nonzero count")
    for key in ("checkpoint", "config", "normalization"):
        if not (model_root / manifest[key]).is_file():
            raise SystemExit(f"Missing {key}: {model_root / manifest[key]}")
    text_cache = model_root / "text_embeddings"
    if not text_cache.is_dir():
        raise SystemExit(f"Missing text embeddings: {text_cache}")
    components = model_root / "components/wan2.2"
    if not (components / "vae.safetensors").is_file():
        raise SystemExit("Missing bundled VAE. Run scripts/download_assets.py --vae")
    for gpu, port in zip(gpus, ports):
        command = [
            sys.executable, str(code_root / "runtime/serve_uniwam_policy.py"),
            "--config-path", str(model_root / manifest["config"]),
            "--checkpoint-path", str(model_root / manifest["checkpoint"]),
            "--stats-path", str(model_root / manifest["normalization"]),
            "--text-cache-dir", str(text_cache),
            "--preprocess-profile", manifest["preprocess_profile"],
            "--device", "cuda:0", "--port", port,
            "--num-inference-steps", "10", "--seed", "42", "--rand-device", "cpu",
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["UNIWAM_COMPONENTS_ROOT"] = str(components)
        env.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(code_root / "checkpoints"))
        env.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "huggingface")
        print(f"CUDA_VISIBLE_DEVICES={gpu}", " ".join(command), flush=True)
        if not args.dry_run:
            args.log_dir.mkdir(parents=True, exist_ok=True)
            log = args.log_dir / f"{args.model}_{port}.log"
            with log.open("ab") as stream:
                child = subprocess.Popen(command, env=env, cwd=code_root, start_new_session=True, stdout=stream, stderr=subprocess.STDOUT)
            (args.log_dir / f"{args.model}_{port}.pid").write_text(str(child.pid) + "\n")
            print(f"Started PID {child.pid}; log: {log}")


if __name__ == "__main__":
    main()
