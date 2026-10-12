#!/usr/bin/env python3
"""Download UniWAM/MAP-Bench and pinned upstream assets into a relative workspace."""
from __future__ import annotations

import argparse
import hashlib
import fnmatch
import json
from pathlib import Path
import shutil
import tarfile



def extract_archives(root: Path, archive_dir: str, destination: str) -> None:
    import zstandard

    sums = root / archive_dir / "SHA256SUMS"
    for line in sums.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        if Path(name).name != name or not name.endswith(".tar.zst"):
            raise ValueError(f"Invalid archive name: {name}")
        archive = sums.parent / name
        h = hashlib.sha256()
        with archive.open("rb") as f:
            for block in iter(lambda: f.read(8 << 20), b""):
                h.update(block)
        if h.hexdigest() != digest:
            raise ValueError(f"SHA256 mismatch: {archive}")
        with archive.open("rb") as f, zstandard.ZstdDecompressor().stream_reader(f) as stream:
            with tarfile.open(fileobj=stream, mode="r|") as tar:
                for member in tar:
                    target = (root / member.name).resolve()
                    allowed_root = (root / destination).resolve()
                    if allowed_root not in target.parents and target != allowed_root:
                        raise ValueError(f"Unexpected archive path: {member.name}")
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    elif member.isfile():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with tar.extractfile(member) as source, target.open("wb") as output:
                            shutil.copyfileobj(source, output)
                    else:
                        raise ValueError(f"Unsupported archive member: {member.name}")
        print(f"Verified and extracted {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("base", "scaled", "all"))
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--training", action="store_true", help="Download, verify, and extract the MAP-Bench training split")
    parser.add_argument("--traces", action="store_true", help="Download, verify, and extract five trace archives")
    parser.add_argument("--text-encoder", action="store_true", help="Download Wan UMT5 and tokenizer for new instructions")
    parser.add_argument("--vae", action="store_true", help="Download VAE independently of model checkpoints")
    parser.add_argument("--baseline", choices=("pi05", "groot_n17", "all"))
    parser.add_argument("--model-repo", default="Fysics-AI/UniWAM")
    parser.add_argument("--benchmark-repo", default="Fysics-AI/MAP-Bench")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--benchmark-revision", default="main")
    parser.add_argument("--model-root", type=Path, default=Path("../UniWAM-weights"))
    parser.add_argument("--data-root", type=Path, default=Path("../MAP-Bench"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not any((args.model, args.benchmark, args.training, args.traces, args.text_encoder, args.vae, args.baseline)):
        parser.error("Select --model, --benchmark, --training, --traces, --text-encoder, --vae, or --baseline")
    plan = []
    if args.model or args.baseline or args.vae or args.text_encoder:
        patterns = ["README.md", "LICENSE*", "THIRD_PARTY*", "licenses/**", "FILES.json"]
        if args.model:
            models = ("base", "scaled") if args.model == "all" else (args.model,)
            patterns += [f"weights/{model}/*" for model in models]
            patterns += ["weights/weights_manifest.json", "text_embeddings/*", "components/**"]
        if args.vae:
            patterns += ["components/wan2.2/vae.safetensors", "components/wan2.2/README.md", "components/wan2.2/LICENSE", "components/wan2.2/SHA256SUMS"]
        if args.text_encoder:
            patterns += ["components/wan2.2/text_encoder.safetensors", "components/wan2.2/tokenizer/*", "components/wan2.2/README.md", "components/wan2.2/LICENSE", "components/wan2.2/SHA256SUMS"]
        if args.baseline:
            models = ("pi05", "groot_n17") if args.baseline == "all" else (args.baseline,)
            patterns += [f"baselines/{model}/**" for model in models]
        plan.append((args.model_repo, "model", args.model_revision, args.model_root, patterns))
    if args.benchmark or args.traces or args.training:
        patterns = ["README.md", "LICENSE*", "*.json"]
        if args.benchmark or args.traces:
            patterns += ["evaluation/**", "results/*.csv"]
        if args.training:
            patterns += ["training/**"]
        if args.traces:
            patterns += ["archives/*"]
        plan.append((args.benchmark_repo, "dataset", args.benchmark_revision, args.data_root, patterns))
    for repo, kind, revision, dest, patterns in plan:
        print(f"{kind}: {repo}@{revision}\n  to: {dest}\n  files: {', '.join(patterns)}", flush=True)
        if not args.dry_run:
            from huggingface_hub import HfApi, snapshot_download
            if kind == "model":
                api = HfApi()
                info = api.repo_info(repo, repo_type=kind, revision=revision, files_metadata=True)
                available = {entry.rfilename: entry.size for entry in info.siblings}
                if "FILES.json" not in available:
                    raise SystemExit("Model package manifest FILES.json is missing from this repository revision.")
                from huggingface_hub import hf_hub_download
                manifest_path = hf_hub_download(repo, "FILES.json", revision=info.sha)
                manifest = json.loads(Path(manifest_path).read_text())["files"]
                required = {name: size for name, size in manifest.items() if any(fnmatch.fnmatch(name, pattern) for pattern in patterns)}
                missing = [name for name, size in required.items() if available.get(name) != size]
                if missing:
                    raise SystemExit("Missing or incomplete model files: " + ", ".join(missing[:12]))
                revision = info.sha
            snapshot_download(repo, repo_type=kind, revision=revision, local_dir=dest, allow_patterns=patterns)
    if args.traces and not args.dry_run:
        extract_archives(args.data_root, "archives", "results/raw_traces")
    if args.training and not args.dry_run:
        extract_archives(args.data_root, "training/archives", "training/episodes")


if __name__ == "__main__":
    main()
