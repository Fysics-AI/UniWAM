# Installation and downloads

Follow the environment commands in the [README](../README.md). Download
models from [UniWAM](https://huggingface.co/Fysics-AI/UniWAM) and training and
evaluation data from [MAP-Bench](https://huggingface.co/datasets/Fysics-AI/MAP-Bench).

## Workspace

```text
uniwam-workspace/
├── UniWAM/                        # code
├── UniWAM-weights/               # one Hugging Face model package
│   ├── weights/{base,scaled}/
│   ├── weights/weights_manifest.json
│   ├── components/wan2.2/
│   │   ├── vae.safetensors
│   │   ├── text_encoder.safetensors
│   │   ├── tokenizer/
│   │   ├── SHA256SUMS
│   │   └── LICENSE
│   ├── text_embeddings/
│   └── baselines/pi05/
├── MAP-Bench/
│   ├── training/                       # 35,454 training trajectories
│   │   ├── index/
│   │   ├── episode_index/
│   │   ├── stats_train_q01q99.json
│   │   ├── text_embeddings/
│   │   ├── archives/
│   │   └── episodes/
│   ├── evaluation/benchmark_15k.json
│   ├── evaluation/scene_contexts/
│   ├── evaluation/trajectory_ids.csv
│   ├── results/summary.csv
│   └── archives/                         # five raw-trace archives
├── MAP-Data/                             # optional additional HM3D subset
└── hm3d-v0.2/scenes_train_val/            # obtain from HM3D
```

Run download commands from `UniWAM/`. `scripts/download_assets.py --help` lists
model/dataset repository overrides for mirrors or ownership transfers.
`--dry-run` prints the complete download plan without network requests.
`--traces` verifies archive SHA256 values and safely extracts them to
`../MAP-Bench/results/raw_traces/`.
`--training` downloads the training split and verifies and extracts its eight
archives into `../MAP-Bench/training/episodes/`.

## Bundled VAE and T5

`--model base`, `--model scaled`, and `--model all` each download the shared
VAE, T5 encoder, tokenizer, and text caches together with the chosen checkpoint.
The VAE and encoder are the exact safetensors files used in our evaluation.
They are stored once under `UniWAM-weights/components/wan2.2/`, together with
SHA256 checksums and upstream license notices. The encoder is about 11.36 GB;
it is loaded only when generating new instruction features.

The launcher resolves bundled components automatically and does not need an
additional Wan download. For manual policy startup or training, set:

```bash
export UNIWAM_COMPONENTS_ROOT="$(realpath ../UniWAM-weights/components/wan2.2)"
```

`--vae` and `--text-encoder` can retrieve individual components when needed.
HM3D downloads
require the publisher's access process: follow the [HM3D instructions](https://aihabitat.org/datasets/hm3d/).
Keep scene files and their relative directories intact under `scenes_train_val`.

## Traces without checkpoints

You can inspect the complete benchmark and reproduce its result table without
a GPU, HM3D assets, or model weights:

```bash
python -m pip install huggingface_hub zstandard
python scripts/download_assets.py --benchmark --traces
python scripts/verify_evaluation.py --data-root ../MAP-Bench
python scripts/compute_table4.py --data-root ../MAP-Bench
```
