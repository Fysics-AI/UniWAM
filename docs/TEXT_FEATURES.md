# T5 text features

The policy uses the Wan2.2 UMT5-XXL encoder with the Wan tokenizer, length 128,
and the following prompt template:

```text
A video recorded from a robot's point of view executing the following instruction: {task}
```

Each cache filename is the SHA256 of this exact UTF-8 prompt followed by
`.t5_len128.wan22ti2v5b.pt`. Each file contains `context` (128 × 4096 bfloat16)
and `mask` (128 boolean values). The server zeroes padding tokens and uses the
same attention convention as the released training/inference code.

Precomputed features avoid loading an 11 GB encoder at inference time and are
also supported in upstream FastWAM. They represent the benchmark instructions;
new instructions need their own features.

```bash
python scripts/download_assets.py --text-encoder
python scripts/precompute_text_embeddings.py   --instructions instructions.txt --output-dir ../UniWAM-weights/text_embeddings
# Or regenerate from the benchmark's task strings:
python scripts/precompute_text_embeddings.py   --benchmark ../MAP-Bench/evaluation/benchmark_15k.json   --output-dir ../UniWAM-weights/text_embeddings
# Check coverage without loading the encoder:
python scripts/precompute_text_embeddings.py   --benchmark ../MAP-Bench/evaluation/benchmark_15k.json   --output-dir ../UniWAM-weights/text_embeddings --check-only
```

Use the policy environment, with enough memory for UMT5-XXL. Default encoding
uses CUDA and batch size 1; `--device cpu` is available for machines with
sufficient host RAM. Existing files are reused; pass `--overwrite` to regenerate.

Regenerating bfloat16 features can introduce small numerical differences with
hardware and batching. Use the supplied benchmark caches when reproducing the
released evaluation settings. The generator has been checked on one instruction.
