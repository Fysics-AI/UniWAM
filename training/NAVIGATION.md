# UniWAM-Base MAP-Bench training

This directory provides the UniWAM-Base navigation expert and its training
configuration, built on [FastWAM](https://github.com/yuantianyuan01/FastWAM).
Training uses the **35,454-trajectory MAP-Bench training split** with a single
forward RGB camera. Download the training data and shared components from the
repository root:

```bash
python scripts/download_assets.py --vae --training
```

This preset covers the navigation expert used with MAP-Bench. For the full mobile manipulation architecture and real-robot Base configuration,
see the [robot training guide](real_robot/README.md).

## Scope of this preset

This preset reproduces the MAP-Bench navigation setting: a single navigation
expert, one forward RGB camera, and 9D outputs consisting of base waypoints,
target-object boxes, and image-plane MAP coordinates. It uses the 35,454-example
training split and is evaluated on 15,000 evaluation trajectories with HM3D scenes.
The full unified UniWAM architecture is documented separately in
[real_robot/README.md](real_robot/README.md); it has both streams and is initialized from Wan2.2 for
real-robot demonstrations.

MAP-Bench evaluates navigation only. Its released checkpoints use one navigation
expert, one forward RGB view, and synchronous closed-loop execution. Training
samples five frames at offsets `[0, 8, 16, 24, 32]` on a `480 × 256` canvas.
The navigation model retains causal action attention and both bbox and image-plane
MAP supervision. Unavailable auxiliary coordinates use the `-0.1` missing-label
convention in this navigation implementation.

The full mobile manipulation architecture uses the nine-frame, multi-view
`384 × 320` layout described in the report. Its clean committed-prefix mechanism
supports asynchronous robot execution. These features are not used by the released
MAP-Bench navigation checkpoints; the MAP-Bench ablations evaluate pose targets,
causal attention, and the two auxiliary targets.

## Contents


- `scripts/train.py`: Hydra training entry point.
- `scripts/train.sh`: multi-process launcher using Accelerate.
- `configs/`: the public UniWAM-Base model, data, and optimization presets.
- `src/fastwam/`: the model, dataset, and trainer code used by the preset.
- `requirements-training.txt`: training-only dependencies.

## Data layout

Set `UNIWAM_DATA_ROOT` to the downloaded `MAP-Bench/training` directory.
The Base preset expects the following relative files or directories:

```text
MAP-Bench/training/
  index/
  episode_index/
    manifest.json
    episode_ids.npy
    targets.jsonl
  stats_train_q01q99.json
  text_embeddings/
  episodes/
    000000/
      rgb.mp4
      frame_annotations.jsonl
      actions_vx_vyaw.npy
```

The package includes 7,184,098 RGB frames at 15 Hz, navigation commands,
target boxes, MAP annotations, and instructions. The download helper verifies
and extracts the training archives into `episodes/`. All index paths resolve
relative to the index files, so the dataset can be moved with its folder.
Some episodes store commands in the frame annotations instead of a separate
NumPy file. HM3D scene assets are needed only for closed-loop evaluation.

## Install

Use the policy environment from the [installation guide](../README.md#installation).
From the repository root, install the training dependencies and resolve the
shared model components before entering `training/`:

```bash
conda activate uniwam-policy
python -m pip install -r training/requirements-training.txt
export UNIWAM_COMPONENTS_ROOT="$(realpath ../UniWAM-weights/components/wan2.2)"
```

The preset uses the training split's cached T5 features (`load_text_encoder: false`).
See [text features](../docs/TEXT_FEATURES.md) to generate additional prompts.

The default initializes the video DiT from
[Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) and the action
backbone through FastWAM's interpolation of the video DiT. The action input
encoder and output head are newly initialized. Wan weights download on first
use into `../UniWAM-weights/upstream/` relative to the repository root, shared
with the unified model. To download them in advance, run from the repository root:

```bash
hf download Wan-AI/Wan2.2-TI2V-5B \
  --include 'diffusion_pytorch_model*.safetensors' \
  --local-dir ../UniWAM-weights/upstream/Wan-AI/Wan2.2-TI2V-5B
```

To adapt the released Base checkpoint instead, download it with
`python scripts/download_assets.py --model base`, then set
`UNIWAM_INIT_CHECKPOINT` to its path before training. A supplied checkpoint
skips Wan initialization and starts a fresh optimizer and schedule.
For an explicit random-initialization experiment, set both `resume=null` and
`model.skip_dit_load_from_pretrain=true`.

## Single-GPU smoke test

Use a small step count to verify imports, configuration resolution, and data
access before a long run:

```bash
cd training
export UNIWAM_DATA_ROOT="$(realpath ../../MAP-Bench/training)"
export UNIWAM_TEXT_CACHE="$UNIWAM_DATA_ROOT/text_embeddings"
PYTHONPATH=src python scripts/train.py \
  --config-name task/uniwam_base_hm3d \
  max_steps=1 batch_size=1 num_workers=0 persistent_workers=false \
  output_dir=./runs/smoke
```

This command still needs one valid training sample and a CUDA-capable model
environment; it is a short functional check, not a quality benchmark.

## Training

For one GPU, invoke the same entry point with the released Base preset:

```bash
cd training
export UNIWAM_DATA_ROOT="$(realpath ../../MAP-Bench/training)"
export UNIWAM_TEXT_CACHE="$UNIWAM_DATA_ROOT/text_embeddings"
PYTHONPATH=src python scripts/train.py \
  --config-name task/uniwam_base_hm3d
```

For distributed training, pass the number of processes to the launcher:

```bash
cd training
export UNIWAM_DATA_ROOT="$(realpath ../../MAP-Bench/training)"
export UNIWAM_TEXT_CACHE="$UNIWAM_DATA_ROOT/text_embeddings"
bash scripts/train.sh 16
```

The report's MAP-Bench runs use 16 GPUs. For a local smoke test or a smaller
replication, replace `16` with the number of GPUs available on your machine.

Hydra overrides can be appended to either command, for example
`max_steps=1000 output_dir=./runs/base_trial`. The default output is a new
timestamped directory under `training/runs/uniwam_base/`, containing model
weights under `checkpoints/weights/`. Training state saving is disabled by
default; enable it only when you deliberately need a private resume artifact.

## Public Base preset

The checked-in preset follows the report where it specifies the benchmark
recipe: 40,000 steps, AdamW with initial learning rate `1e-5`, cosine decay,
bf16 mixed precision, action horizon 32, action dimension 9, and ten Euler
inference steps. The code also provides portable release defaults for data
loading, a local sampling window of 640, a short warmup, gradient clipping,
normalization, and flow-matching schedules; these are implementation defaults
where the report does not state a value. W&B logging is disabled by default.
Edit the YAML through Hydra overrides when adapting it to your hardware or
data preparation.

The optional HM3D subset of MAP-Data provides additional navigation data.
It is separate from the MAP-Bench training split used by this Base preset.
