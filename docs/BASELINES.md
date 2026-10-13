# Comparison baselines

The model package includes UniWAM-Base, UniWAM-Scaled, and the MAP-Bench navigation
adaptation of π₀.₅. The benchmark package contains 15,000 evaluation traces each
for GR00T N1.7, π₀.₅, FastWAM, UniWAM-Base, and UniWAM-Scaled. The GR00T
navigation-adapted checkpoint is not released; its traces support result
recomputation, while reproducing its rollouts requires a compatible checkpoint.

## Official pretrained sources

- **π₀.₅:** [OpenPI base models](https://github.com/Physical-Intelligence/openpi#base-models),
  with the official checkpoint at `gs://openpi-assets/checkpoints/pi05_base`.
- **GR00T N1.7:** [NVIDIA's pretrained model](https://huggingface.co/nvidia/GR00T-N1.7-3B).

These are the upstream starting points. The MAP-Bench results use navigation
adaptations with matching normalization and action decoding.

## Download the π₀.₅ checkpoint

```bash
python scripts/download_assets.py --baseline pi05
```

The π₀.₅ directory contains Orbax parameters, normalization, and the PaliGemma
tokenizer. Both adapters use one forward RGB camera; π₀.₅'s two wrist inputs are
masked placeholders.

## π₀.₅ environment and serving

Install [OpenPI](https://github.com/Physical-Intelligence/openpi) in its own Python
3.11 environment using its official installation instructions. Use this source
revision as the reference version:

```bash
git clone https://github.com/Physical-Intelligence/openpi.git third_party/openpi
cd third_party/openpi
git checkout 215abfb217dbac7d5f1273282331b9b1866c0479
GIT_LFS_SKIP_SMUDGE=1 uv sync
cd ../..
```

From `UniWAM/`, using the OpenPI environment:

```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.40
CUDA_VISIBLE_DEVICES=0 third_party/openpi/.venv/bin/python baselines/serve_pi05.py \
  --checkpoint ../UniWAM-weights/baselines/pi05 \
  --fixed-batch-size 32 --port 18144
```

The wrapper uses the public model API and the supplied inference configuration.
The model predicts 50 steps; the evaluator executes 32. Base velocity and yaw
rate are decoded from slots 14 and 16, unnormalized using the checkpoint's
quantiles, then integrated at 15 Hz into cumulative relative SE(2) poses.

## GR00T evaluation adapter

The released adapter documents the evaluation interface. Running it requires
an independently obtained, compatible navigation-adapted GR00T checkpoint;
NVIDIA's upstream pretrained checkpoint alone does not reproduce the released
MAP-Bench traces.

Use a separate Python 3.12 environment and follow the
[Isaac GR00T installation instructions](https://github.com/NVIDIA/Isaac-GR00T).
Reference source revision:

```bash
git clone https://github.com/NVIDIA/Isaac-GR00T.git third_party/Isaac-GR00T
cd third_party/Isaac-GR00T
git checkout d2b7e75b937e3ec9aa5dbc798f08b89692c49734
git apply ../../baselines/patches/Isaac-GR00T-local-backbone.patch
uv sync
cd ../..
# WebSocket dependencies for the benchmark adapter:
uv pip install --python third_party/Isaac-GR00T/.venv/bin/python \
  'websockets>=15' msgpack typing_extensions
```

The patch lets this pinned upstream revision recognize a checkpoint's bundled
`./backbone` path. It changes no model parameters.

```bash
CUDA_VISIBLE_DEVICES=0 third_party/Isaac-GR00T/.venv/bin/python baselines/serve_groot.py \
  --model-path /path/to/compatible/navigation-checkpoint \
  --embodiment-tag NEW_EMBODIMENT --port 18144
```

The model predicts 40 velocity steps; the adapter executes 32 and integrates
commands at 15 Hz. A compatible processor must contain the navigation
embodiment. Original model tensor names are retained for upstream compatibility.

## Evaluation

Run the same `scripts/launch_external_eval.py` command as for UniWAM, with the
baseline server's port and a separate output directory. The evaluator batch
size must not exceed π₀.₅'s `--fixed-batch-size`.

## Licenses

π₀.₅ retains OpenPI and Gemma terms. GR00T and Cosmos retain their upstream
terms if separately obtained. These are separate from UniWAM's code license.
