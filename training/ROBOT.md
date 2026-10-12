# UniWAM real-robot training and inference

This release provides the full navigation/manipulation architecture and scripts
for training it on your own demonstrations. The default preset follows the
**real-robot UniWAM-Base** setting: task demonstrations, shared Wan initialization,
and both streams in one model. Task-trained real-robot checkpoints are not part
of this release. The downloadable Base/Scaled checkpoints are for MAP-Bench;
use the [navigation guide](NAVIGATION.md) with those checkpoints.

## Architecture

| Paper component | Implementation |
| :--- | :--- |
| Shared video DiT and action DiT; separate action encoders, heads, and learnable stream tokens | [UniWAM](src/fastwam/models/wan22/uniwam.py), [Action DiT](src/fastwam/models/wan22/uniwam_action_dit.py) |
| Independently sampled navigation/manipulation examples, packed along the batch dimension | [Dataset, collator, and sampler](src/fastwam/datasets/robot_windows.py) |
| Clean current observation; action access to current visual context and earlier actions only | `action_attention_mask`, `_video_mask_with_type`, `_packed_mot_forward` |
| Clean committed action prefix, excluded from loss and clamped during inference | `training_loss`, `infer_action` |
| Separate physical-action and auxiliary flow losses, averaged over valid features/positions/examples with global worker counts | `_masked_action_sum_count`, `_global_means` |
| Camera-frame poses, column-based 6D rotations, single-arm padding, visibility masks | [Geometry helpers](src/fastwam/datasets/robot_geometry.py) |
| Cached observation KV and concurrent two-stream prediction | `infer_action`, [MoT](src/fastwam/models/wan22/mot.py) |

Navigation outputs **9 values** per action: chunk-relative base `(dx, dy, dtheta)`,
a normalized target box `(x1, y1, x2, y2)`, and a normalized MAP point `(u, v)`.
Manipulation outputs **26 values**: 20 physical values followed by 6 auxiliary values:

```text
0:10    left  [position xyz, rotation column 0 xyz, rotation column 1 xyz, gripper]
10:20   right [position xyz, rotation column 0 xyz, rotation column 1 xyz, gripper]
20:23   left  [u, v, visible]
23:26   right [u, v, visible]
```

Manipulation positions (meters) and orientations use the **main camera frame at
the start of the action chunk**. Navigation angles use radians. Gripper values
use the robot adapter's consistent scalar convention. The absent arm is zero
padded and loss masked, including its auxiliary outputs. Only visible end-effector
coordinates are supervised; a valid occluded point still supervises visibility as
zero. Invalid labels supervise neither coordinates nor visibility.

The main navigation camera supplies the navigation row. The manipulation main
camera supplies the manipulation row. Wrist views are included in manipulation
and, for mobile manipulation tasks, also in navigation. `camera_canvas` provides
a common 384 × 320 canvas (width × height); its default layout places a 256-pixel-high main view above two
64-pixel-high wrist panels. This panel split is a release default; the paper
specifies the overall canvas size. Missing wrist views are black. Streams share transformer weights
and remain separate batch rows throughout attention.

## Installation and initialization

From the repository root, use the policy environment described in the
[main installation guide](../README.md#installation):

```bash
python -m pip install -r training/requirements-training.txt
python scripts/download_assets.py --vae --text-encoder
```

The shared VAE, UMT5-XXL encoder, and tokenizer live in
`../UniWAM-weights/components/wan2.2/`. The training loader uses this directory
when present; `UNIWAM_COMPONENTS_ROOT` can override it.

The first model initialization downloads the video DiT from
[Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B), under
`../UniWAM-weights/upstream/`. To download it explicitly in advance:

```bash
hf download Wan-AI/Wan2.2-TI2V-5B \
  --include 'diffusion_pytorch_model*.safetensors' \
  --local-dir ../UniWAM-weights/upstream/Wan-AI/Wan2.2-TI2V-5B
```

The Action DiT backbone is initialized from that video DiT using
[FastWAM's interpolation procedure](https://github.com/yuantianyuan01/FastWAM#model-preparation),
implemented in [uniwam_initialization.py](src/fastwam/models/wan22/uniwam_initialization.py).
Stream encoders, output heads, type tokens, and the state projection are newly
initialized. No task-trained mobile checkpoint is needed for initialization.
The VAE and text encoder remain frozen.

## Prepare demonstrations

Real-robot demonstrations and calibration are supplied by the user. Convert each
recording into independent navigation and manipulation windows. A mobile episode
can contribute windows to both streams; the two rows need not share a start frame.
Mark active/idle windows using your robot's motion thresholds. Camera calibration,
arm ordering, gripper convention, and appearance-based visibility labels must
match your recordings.

For the manipulation stream's image-plane auxiliary labels, use the
[end-effector point-tracking pipeline](../robot-eef-point-tracking/README.md)
to project calibrated end-effector points into the main camera and assign
visibility and quality masks. Its README documents geometry generation,
DINOv2 visibility filtering, validation, and adaptation to other robots.
Apply the same crop and resize to its `track_xy_geom` pixel coordinates as to
the main-view training video, then divide by the resulting image width and
height to obtain normalized `uv`. Pass its `quality` values to
`manipulation_targets`: 0 means invalid, 1 visible, and 2 in-frame but
occluded or uncertain. The helper masks `uv` loss except where quality is 1,
while supervising visibility for both quality 1 and 2. Use the geometric
coordinates with that mask; `track_xy` contains NaNs for invisible points.

```text
UniWAM-robot-data/
  windows.jsonl
  normalization.json
  windows/
    nav_0000.npz
    manip_0000.npz
  text_embeddings/
    <prompt-hash>.t5_len128.wan22ti2v5b.pt
```

Each manifest line contains only paths relative to this data directory:

```json
{"source":"task_demos","episode":"episode_0001","start":0,"stream":"manip","active":true,"path":"windows/manip_0000.npz","text":"text_embeddings/<prompt-hash>.t5_len128.wan22ti2v5b.pt"}
```

Each NumPy window contains the following arrays (`allow_pickle=False`):

| Key | Shape / dtype | Meaning |
| :--- | :--- | :--- |
| `video` | `[3,9,320,384]`, uint8 | Canvas frames at offsets 0,4,…,32 of a 33-frame window |
| `action` | `[32,9]` or `[32,26]`, float32 | Unnormalized targets for future steps 1,…,32 |
| `feature_mask` | Same shape as action, bool | Valid physical and auxiliary features |
| `action_is_pad` | `[32]`, bool | Padded future action positions |
| `image_is_pad` | `[9]`, bool | Padded video positions; frame zero must be valid |
| `proprio` | `[20]`, float32 | Current two-arm state in the same camera-frame pose/gripper representation |
| `proprio_mask` | `[20]`, bool | Available state dimensions; absent arms are masked |

Use `poses_in_camera`, `manipulation_targets`, and `navigation_targets` to build
these arrays. Current camera extrinsics stay fixed across the physical action
chunk. Image auxiliary labels are the per-frame main-view projections described
in the paper. A gripper visibility point can have a calibrated offset from the
end-effector frame used by physical actions. The `quality` argument is 0 for
invalid, 1 for visible, and 2 for valid but occluded. Supply appearance-checked
visibility labels; the geometry helper does not infer occlusion from RGB.

Generate language features with the bundled encoder and the documented prompt:

```bash
python scripts/precompute_text_embeddings.py \
  --instructions instructions.txt \
  --output-dir ../UniWAM-robot-data/text_embeddings
```

See [T5 feature generation](../docs/TEXT_FEATURES.md). A feature file contains
`context` `[128,4096]` and `mask` `[128]`; many windows can reference the same file.

`normalization.json` contains `nav`, `manip`, and `proprio` objects, each with
`offset` and positive `scale` arrays of lengths 9, 26, and 20 respectively.
The loader applies `(value - offset) / scale` and then zeros invalid features.
Compute these values on **training demonstrations only**, per valid dimension.
For quantile normalization, use `(q01 + q99) / 2` as offset and
`max((q99 - q01) / 2, epsilon)` as scale. Image coordinates/visibility can retain
identity normalization. Use the same file for training and inference.

## Base defaults

The defaults are in [uniwam_robot_base.yaml](configs/uniwam_robot_base.yaml)
and its [model configuration](configs/model/uniwam_robot_base.yaml).

| Setting | Default | Basis |
| :--- | :--- | :--- |
| Video / action hidden width | 3072 / 1024 | Paper |
| Transformer layers | 30 in each DiT | Paper |
| Video window / stride / encoded frames | 33 / 4 / 9 → 3 VAE latents | Paper |
| Canvas | 384 × 320 | Paper |
| Optimizer / initial learning rate | AdamW / 1e−5, cosine decay | Paper |
| Training steps | 20,000 | Real-robot UniWAM-Base in the paper |
| Data | User task demonstrations | Base scope; no extra pretraining data |
| Idle-window weight, manipulation / navigation | 0.20 / 0.15 | Paper |
| Flow shift / inference steps / CFG | 5 / 10 Euler steps / none | Paper |
| Action horizon | 32 | Configurable release default |
| Prefix probability / lengths | 0.5 / uniform 1–8 | Configurable release defaults; tune to execution latency |
| Stream type tokens / proprioceptive state | 4 tokens / 20D camera-frame state | Configurable release defaults |
| Batch size / weight decay / gradient clip | 2 per rank / 0.01 / 1.0 | Configurable release defaults |
| Five loss coefficients | 1 each | Configurable release defaults |

The paper reports 32 GPUs for the Base experiment. The launcher accepts your
available GPU count; batch size and hardware choice affect the training run.
The paper does not specify every configuration value. The release defaults in
those rows are explicit choices, not claims of recovered experiment settings.

Source weights use keys such as `task_demos/nav` and `task_demos/manip`.
Each source/stream mass is normalized independently of its number of windows;
active windows have weight 1 and idle windows use the factors above. The sampler
rejects duplicate and nearby windows in the **global batch across ranks**. Its
minimum start separation is 33 frames by default. Reduce batch size if your
small dataset cannot satisfy this constraint.

## Train

Paths inside the configuration resolve relative to the configuration file;
CLI path overrides resolve relative to your current working directory.
From the repository root:

```bash
# Validate actual data and a complete batch without allocating the model.
python training/scripts/train_robot.py \
  --data-root ../UniWAM-robot-data --data-only

# One update to check your data and GPU setup.
bash training/scripts/train_robot.sh 1 \
  --data-root ../UniWAM-robot-data --batch-size 1 --steps 1

# Base training on your demonstrations (the paper uses 32 GPUs).
bash training/scripts/train_robot.sh 32 --data-root ../UniWAM-robot-data
```

For multiple nodes set `NNODES`, `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT`
consistently for `torchrun`. The sampler already shards global batches; do not
add a second distributed sampler. This entry point uses DDP and keeps one full
model per GPU. Adjust per-rank batch size for available memory.

The output directory contains `model.pt`, `model_config.json`, and
`normalization.json`. `model.pt` contains model parameters. To adapt your own
trained model with a fresh optimizer, use `--initialize-from path/to/model.pt`.
It is not an optimizer-state resume command.

## Infer with your trained model

Create a tensor-only observation `.pt` containing:

- `nav_image` and/or `manip_image`: uint8 `[3,320,384]` canvases. Supplying both runs both streams in one batch.
- `context` `[128,4096]`, `context_mask` `[128]`: one instruction's T5 features.
- `proprio` `[20]`, `proprio_mask` `[20]`: current unnormalized robot state and its validity.
- Optional `committed_prefix`: dictionary with `nav` `[P,9]` / `manip` `[P,26]` **normalized** committed actions, with `P < 32`.
- Optional `action_feature_mask`: dictionary with `nav` `[32,9]` / `manip` `[32,26]` masks. Supply the absent-arm mask for a single-arm robot.

```bash
python training/scripts/infer_robot.py \
  --checkpoint runs/uniwam_robot_base/model.pt \
  --normalization runs/uniwam_robot_base/normalization.json \
  --observation observation.pt --output prediction.pt
```

Use the same model configuration as training (`--config` if you changed it).
The output includes normalized and decoded chunks. Convert 6D orientations with
`rotation_from_6d`, transform camera-frame poses into your controller's frame,
and feed physical actions to your robot adapter. Auxiliary dimensions are for
supervision/inspection. Predicted visibility is a continuous output; threshold
it for display as appropriate.

For asynchronous execution, your controller retains the unexecuted committed
prefix, transforms it into the new chunk's reference frame, normalizes it, and
supplies it for the next call. The model clamps that prefix at every denoising
step. Robot transport, inverse kinematics, motion limits, and execution scheduling
belong to the deployment adapter.

## Validation

```bash
python -m pip install pytest
python -m pytest tests/test_robot_model.py -q
torchrun --standalone --nproc_per_node=2 tests/robot_distributed_check.py
```

The tests exercise real DiT blocks at small sizes: separate stream interfaces,
forward/backward, causality, no cross-row attention, prefix clamping, single-arm
and visibility masks, camera-frame rotations, and distributed sampling.
The public training interface is intended for user demonstrations; paper results
are reported separately from these implementation checks.

The full 30-layer model has also passed a synthetic two-stream forward/backward
check, a one-update training-script run, checkpoint save/reload, and ten-step
inference from Wan initialization. These checks validate execution and interfaces;
they do not reproduce the paper's real-robot success rates or deployment latency.
