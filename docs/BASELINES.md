# π₀.₅ comparison checkpoint

The model package includes UniWAM-Base, UniWAM-Scaled, and the MAP-Bench
navigation adaptation of π₀.₅. FastWAM is represented by released evaluation
traces in the benchmark package and does not require a checkpoint here.

## Official pretrained sources

- **π₀.₅:** [OpenPI base models](https://github.com/Physical-Intelligence/openpi#base-models),
  with the official checkpoint at `gs://openpi-assets/checkpoints/pi05_base`.

These are the upstream starting points. The MAP-Bench results use the navigation
adaptations below, with their matching normalization and action decoding.

## Download navigation checkpoints

```bash
python scripts/download_assets.py --baseline pi05
```

The π₀.₅ directory contains Orbax parameters, normalization, and the PaliGemma
tokenizer. The adapter uses one forward RGB camera; π₀.₅'s two wrist inputs are
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

## Evaluation

Run the same `scripts/launch_external_eval.py` command as for UniWAM, with the
baseline server's port and a separate output directory. The evaluator batch
size must not exceed π₀.₅'s `--fixed-batch-size`.

## Licenses

π₀.₅ retains OpenPI and Gemma terms. These terms accompany the checkpoint
directory and are separate from UniWAM's code license.
