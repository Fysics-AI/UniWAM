# UniWAM

UniWAM is a camera-frame world-action model for dual-arm mobile manipulation.
This repository provides the training, data preparation, cloud inference, and
robot-client code needed to adapt the model to your task data.

Training and deployment require external datasets, videos, model weights, text
embedding caches, VAE latent caches, camera calibration, and robot description
files. Set their paths explicitly in your environment.

## Repository layout

- `training/src/uniwam`: model, dataset, normalization, and training code.
- `training/data_pipeline`: conversion from LeRobot episodes to camera-frame
  data, H32 windows, auxiliary labels, statistics, and latent-cache indexes.
- `training/configs`: Hydra training and data contracts.
- `deployment`: cloud server, prompt preparation, and robot-client code.
- `docs`: data, action-slot, fine-tuning, and deployment instructions.

For the full data-to-deployment sequence, see
[Fine-tuning and Deployment](docs/FINETUNING.md).

## Installation

Clone this repository, then install Conda or micromamba and check that
`nvidia-smi` detects the training
GPU. From the repository root, create a Python 3.10 environment, install the
PyTorch wheel matching the installed CUDA driver, then install the pinned
dependencies. For a driver that supports CUDA 12.8:

```bash
conda create -n uniwam python=3.10 -y
conda activate uniwam
conda install -c conda-forge ffmpeg git ninja -y
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r training/requirements-training.txt
python -m pip install -r deployment/requirements-cloud.txt
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda)'
```

Use the appropriate PyTorch wheel index if the driver does not support CUDA
12.8. The cloud server uses this environment. For CUDA selection, FFmpeg,
optional ROS 2 robot-side setup, and smoke tests, see
[Installation](docs/INSTALLATION.md).

Edit the path block in the environment script and source it in the same shell
before preparing data or training:

```bash
source scripts/activate_uniwam_env.sh
```

The template is an ordinary shell file. It does not guess paths, download
private assets, or modify the repository. Edit the path block once for your
machine, then source it before running the commands below. The
script checks the inputs needed at this stage. Prepared data, window indexes,
and caches can be generated afterward.

## Data contract

The default input is a dual-arm, fixed-camera dataset in LeRobot format. Each
episode contains at least 33 contiguous frames, the three manipulation camera
streams, `meta/episodes.jsonl`, and `meta/tasks.jsonl`. The task sentence in
`tasks.jsonl` is the prompt used by training and inference.

The model state contains 20 values:

`left XYZ + left row-major Rot6D + left gripper + right XYZ + right row-major Rot6D + right gripper`.

The manipulation action head contains 26 values: 20 robot-control values plus
six optional image-plane tracking values. Navigation supports a base 3-value
action `(dx, dy, dtheta)` and a trained 9-value action `3 + 4 + 2`, appending
target-object bbox `(xmin, ymin, xmax, ymax)` and operating-point `(x, y)`. The
included fine-tuning example defaults to 3D; select matching 9D model/data/
sidecar settings for the extended contract. See
[Action Slots and Feature Masks](docs/ACTION_SLOTS.md) before changing a
dimension. A 3D checkpoint and a 9D checkpoint are separate contracts; enabling
the 9D variant requires coordinated
changes to the sidecar, dataset, model head, normalization, decoder, and tests.

### Image sizes

The supplied Piper configuration reads three `424x240` source images (`W x H`):
`cam_manip_high`, `cam_left_wrist`, and `cam_right_wrist`. For each video frame,
it resizes the main view to `384x216` and each wrist view to `192x108`, using
bilinear interpolation with antialiasing. It places the main view across the
top and the wrist views side by side below it:

```text
             x = 0                       383
             +-----------------------------+
  y = 0      |        main 384 x 216       |
             +--------------+--------------+  y = 216
             | left 192 x 108|right 192 x 108|
             +--------------+--------------+  y = 324
```

The intermediate mosaic is `384x324` (`W x H`). Rows 0-1 and 322-323 are
cropped, leaving the `384x320` model input (`video_size: [320, 384]`, `H x W`).
In final-canvas coordinates, the main view occupies rows 0-213, and the wrist
row occupies rows 214-319. Navigation uses the same layout with `cam_nav` in
the top position. The edge client sends each camera image at `424x240`; the
cloud builds the same mosaic used by training. EEF tracking coordinates must
refer to the final `384x320` manipulation canvas, not to an individual source
image. For an EEF point `(x, y)` in the `424x240` main image, the sidecar uses
pixel-center mapping `x' = (x + 0.5) * 384/424 - 0.5` and
`y' = (y + 0.5) * 216/240 - 0.5 - 2`, then normalizes `(x', y')` on the final
canvas to `[-1, 1]`. Points outside the cropped canvas are masked. See
[Fine-tuning and Deployment](docs/FINETUNING.md) for the camera keys and
calibration contract.

### Missing auxiliary labels

If point-tracking labels are unavailable, append finite placeholder values
and set their six feature-mask entries to `false`. For the included fine-tuning
recipe, use `append_missing_eef_xy_slots=true` together with
`disable_manip_aux_training=true` (or an auxiliary loss weight of zero). Masked
features are removed from the loss numerator and denominator. Never mark a
missing value valid just to satisfy a tensor shape.

For the 9D navigation extension, return a value tensor and boolean mask for
each of the six appended bbox/operating-point slots. Missing bbox coordinates
and operating points are masked independently.

## Prepare data

Measure the rigid transforms from each arm base to the fixed camera and save
them as `camera_from_left_base` and `camera_from_right_base` in a JSON file.
Do not reuse calibration from another robot or camera rig.

```bash
python "$REPO/training/data_pipeline/prepare_customer_piper_views.py" \
  --source-root /absolute/path/to/action_current_lerobot \
  --output-root "$UNIWAM_CUSTOM_DATA" \
  --calibration /absolute/path/to/customer_camera_extrinsics.json

python "$REPO/training/data_pipeline/prepare_customer_piper_h32.py" \
  --dataset-root "$UNIWAM_CUSTOM_DATA" \
  --calibration /absolute/path/to/customer_camera_extrinsics.json \
  --output "$UNIWAM_CUSTOM_WINDOW_INDEX"
```

The converter validates rigid transforms, row-major Rot6D geometry, gripper
units, frame continuity, and the H32 boundary. For a dataset already expressed
in the fixed camera frame, use the documented camera-frame option. For a legacy
23-value observation with a three-value base prefix, use the documented
`base3_arm20` option; the model still receives only the 20 arm values.

You may train with one dataset or combine multiple datasets. For a
multi-dataset experiment, add one dataset entry per prepared dataset in a copy
of the Hydra data config, assign explicit sampling weights, and give
each entry its own window index, latent source, prompt set, and statistics
contract. The number of sources and their sampling weights are determined by
the experiment.

## Build caches and train

Prepare the exact task prompts, then generate text embeddings and VAE latents:

```bash
cd "$UNIWAM_TRAINING_ROOT"
python scripts/precompute_text_embeds.py task=uniwam_customer_piper_manip_sft

TASK=uniwam_customer_piper_manip_sft \
OUTPUT_ROOT="$UNIWAM_CUSTOM_LATENT_ROOT" \
NPROC_PER_NODE=1 \
bash data_pipeline/launch_precompute_vae.sh

python scripts/preflight_customer_piper.py
```

The preflight must report `CUSTOMER_SFT_INPUTS_OK` before training. Start a
single-host run with:

```bash
bash scripts/train_zero1.sh 1 task=uniwam_customer_piper_manip_sft
```

The recipe loads the supplied initialization checkpoint as weights, creates a
new optimizer, and writes periodic checkpoints. Increase the first argument to
the number of GPUs on one host. For multiple hosts, use the standard
`NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, and `NNODES` environment variables
with the same task configuration on every host.

Set the run length, per-GPU batch size, and checkpoint interval with Hydra
overrides, for example:

```bash
bash scripts/train_zero1.sh 8 task=uniwam_customer_piper_manip_sft \
  max_steps=50000 batch_size=8 save_every=5000
```

The supplied task config is a single-source Piper-compatible example. For a
different embodiment or action representation, adapt the data and deployment
contracts together before training.

## Cloud inference and robot execution

Asynchronous chunked inference is enabled in the supplied cloud and robot
configs. The client starts a new cloud request while it is still executing the
previous action queue; the prefix length adapts to round-trip latency. The
first request has `prefix=0` because no queued actions exist yet. See
[Asynchronous inference](docs/FINETUNING.md#asynchronous-inference) for the
prefix contract, cross-machine SSH forwarding, and expected log fields.

Use the exact fine-tuned checkpoint and the exact task sentence from the
experiment's `tasks.jsonl`:

```bash
export CHECKPOINT=/absolute/path/to/run/checkpoints/weights/step_xxxxxx.pt
export TASK_PROMPT='The exact task sentence used for this run'
export UNIWAM_PIPER_URDF=/absolute/path/to/robot_description.urdf

export UNIWAM_CLOUD_TASK=uniwam_customer_piper_manip_sft
export UNIWAM_CLOUD_SOURCE_INDEX=0
export UNIWAM_CLOUD_AUX_WEIGHT=0.0

CONFIG="$REPO/deployment/configs/uniwam_cloud_customer_async.yaml" \
bash "$REPO/deployment/scripts/precompute_inference_prompt_async_prefix12_rot6d.sh"

CONFIG="$REPO/deployment/configs/uniwam_cloud_customer_async.yaml" \
bash "$REPO/deployment/scripts/launch_uniwam_cloud_server.sh"
```

Generate the robot-client configuration from the same calibration JSON used by
data preparation, then begin with a dry run:

```bash
python "$REPO/deployment/scripts/make_customer_piper_robot_config.py" \
  --dataset-root "$UNIWAM_CUSTOM_DATA" \
  --calibration /absolute/path/to/customer_camera_extrinsics.json \
  --output /absolute/path/to/customer_robot.yaml

CONFIG=/absolute/path/to/customer_robot.yaml \
bash "$REPO/deployment/scripts/launch_uniwam_piper_client.sh" \
  server_host=127.0.0.1 server_port=8031 \
  inference_mode=manip_only task_prompt="$TASK_PROMPT" \
  prefix_steps=12 min_prefix_steps=6 \
  adaptive_prefix.enabled=true replan_while_queue_nonempty=true \
  control.dry_run=true base.dry_run=true
```

Confirm camera topics, frame transforms, action signs, units, and IK outputs in
the dry-run recording before enabling physical motion. Network forwarding and
robot safety procedures are deployment-specific.

## License

UniWAM contributions are released under the [MIT License](LICENSE). Bundled
third-party code retains its original terms; see
[Third-Party Notices](THIRD_PARTY_NOTICES.md).
