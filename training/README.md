# UniWAM training and inference

The code is built on [FastWAM](https://github.com/yuantianyuan01/FastWAM).
Choose the interface matching your task:

| Task | Architecture and initialization | Guide |
| :--- | :--- | :--- |
| Real-robot navigation and manipulation | Full shared-backbone model, separate streams; initialize from Wan2.2 | [Train and infer](ROBOT.md) |
| MAP-Bench navigation | Single expert and forward RGB view; initialize from Wan2.2 | [Train and evaluate](NAVIGATION.md) |

The default full-model preset corresponds to **real-robot UniWAM-Base**:
training on task demonstrations, with the paper's architecture and optimization
settings. The guide distinguishes paper-specified settings from configurable
release defaults. Task-trained real-robot checkpoints are not included.

Start from the repository root:

```bash
python -m pip install -r training/requirements-training.txt
python scripts/download_assets.py --vae --text-encoder
python training/scripts/train_robot.py --data-root ../UniWAM-robot-data --data-only
bash training/scripts/train_robot.sh 32 --data-root ../UniWAM-robot-data
```

## Two reproducible paths

### Unified UniWAM

[ROBOT.md](ROBOT.md) implements the technical-report architecture: independent
navigation/manipulation sampling, separate camera and action interfaces, shared
video/action DiTs, image-plane auxiliary targets, and prefix-conditioned causal
inference. Its default is real-robot UniWAM-Base with Wan2.2 initialization and
your own demonstrations. We do not provide task-trained mobile-manipulation
weights.

### MAP-Bench navigation

[NAVIGATION.md](NAVIGATION.md) is a separate single-stream recipe. It uses one
forward RGB view, one navigation expert, the 35,454-trajectory training split,
and Wan2.2 initialization with FastWAM action-backbone interpolation. MAP-Bench evaluation uses the
15,000-trajectory held-out split and HM3D scenes; it does not instantiate the
manipulation stream. This separation makes the public benchmark run directly
reproducible while keeping the full unified training path available.

Prepare your demonstrations using the [robot data contract](ROBOT.md#prepare-demonstrations).
The shared Wan backbone downloads automatically on the first initialization;
VAE, T5 and tokenizer are shared with the model package.
