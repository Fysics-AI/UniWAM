# Fine-tune UniWAM on your task

For the complete slot order and masking contract, read
[Action Slots and Feature Masks](ACTION_SLOTS.md) before changing dimensions.

This path adapts a compatible 20D-state/26D-action UniWAM checkpoint to a
manipulation dataset. You supply task text, videos, robot calibration, and
LeRobot episodes. The supplied conversion and robot
client target the camera-frame dual-arm contract documented below; a different
robot kinematic or camera contract requires a corresponding data/configuration
extension.

## Supported input contract

The model state is **20D dual-arm only**:
`[left EEF xyz/row-major Rot6D/gripper, right EEF xyz/row-major Rot6D/gripper]`.
By default, the action-current LeRobot input has this same 20D layout in
`observation.state` and `action.manip`, expressed in the two arm-base frames.
Use `--input-frame camera` only when both columns already use the fixed main
camera frame. A legacy recording with a three-value base prefix in
`observation.state` can use `--raw-state-layout base3_arm20`; that prefix is
never part of the model state. XYZ and gripper widths are in meters; rotations
are the first two **rows** of a rotation matrix. The
manipulation camera keys are `cam_manip_high`, `cam_left_wrist`, and
`cam_right_wrist` at 424x240. Each LeRobot episode needs at least 33 frames,
contiguous `frame_index`, videos, `meta/episodes.jsonl`, and `meta/tasks.jsonl`.
The `task` field in `tasks.jsonl` is the task sentence. Keep the
source dataset and its video directory available after conversion; the
camera-frame dataset links to those videos.

Here `424x240` is each source frame's `W x H`, not the model input size. The
main image is resized to `384x216`; the left and right wrist images are resized
to `192x108` each and concatenated left-to-right below the main image. This
forms a `384x324` mosaic. Cropping its first two and last two rows yields the
`384x320` model input (`W x H`). The main view occupies final rows 0-213 and
the wrist views occupy rows 214-319. The navigation stream substitutes
`cam_nav` for the top image and uses the same two wrist views below. Resize is
bilinear with antialiasing. Auxiliary image coordinates must use the coordinate
frame declared by their sidecar; the supplied EEF sidecar uses the final
`384x320` manipulation canvas.

The Piper ROS client may transmit a 23D observation packet containing base 3D
plus arm 20D for navigation/transport compatibility. The cloud takes its
last 20 values as model proprioception; this does not make 23D a training
state requirement.

Use normalization statistics that match the robot embodiment, state/action
units, and checkpoint. Keep the same statistics for fine-tuning and deployment;
do not substitute statistics from another embodiment or representation. The
manipulation action head is 26D: the first 20 positions are
dual-arm control, and positions 20-25 are six point-tracking auxiliary values.
Navigation supports a 3D `(dx, dy, dtheta)` head and a trained 9D variant
`3 + 4 + 2`, adding target bbox `xyxy` and operating-point `xy`. The included
fine-tuning configuration defaults to 3D; select matching 9D model/data/sidecar
settings when using the extended contract. A
If point-tracking labels are unavailable, do **not** fabricate them:
`append_missing_eef_xy_slots=true` appends six zeros with a false feature mask;
`disable_manip_aux_training=true` excludes those slots from clean inputs,
noise, targets, and loss; `lambda_manip_aux_action=0` disables the auxiliary
loss. The control loss divides by the number of valid control features (20),
not 26. The cloud action decoder uses only the first 20 values for arm control.

## Environment and assets

Follow [Installation](INSTALLATION.md) for the Conda environment, CUDA-matched
PyTorch, pinned training/cloud dependencies, FFmpeg, and robot-side ROS Python
environment. Complete those steps before preparing data.

```bash
export REPO=/absolute/path/to/UniWAM
source "$REPO/scripts/activate_uniwam_env.sh"
```

The Wan/ActionDiT weights, an initialization checkpoint with its adjacent
`config.yaml`, and the robot description are external assets. They are not
licensed or distributed by this source repository.

Edit the paths in `scripts/activate_uniwam_env.sh`, then source it.
For newly generated latent caches, `UNIWAM_CUSTOM_SOURCE_NAME` must equal the
final camera-frame dataset directory name. For an existing cache, set it to
the directory name inside `UNIWAM_CUSTOM_LATENT_ROOT`. All paths must be
readable on every training rank.

## Prepare data

Measure `camera_from_left_base` and `camera_from_right_base` for **this robot
and camera rig**. Store the two rigid 4x4 matrices in a JSON file. Never reuse
the repository's example/historical extrinsics without measuring your rig.

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

The default conversion applies the measured base-to-camera transforms to a
20D dual-arm input. For an already camera-frame 20D dataset, add
`--input-frame camera` to the first command; that mode copies the arm values
without transforming them, so the user must certify the stated frame and
calibration. For old 23D recordings, add `--raw-state-layout base3_arm20`;
the first three values are discarded before converting the two arm poses.
Neither option changes the 20D model input.

The second command checks the base-to-camera transform for base-frame input,
or exact value preservation for declared camera-frame input, plus Rot6D
geometry, gripper units, and H32 boundaries. It refuses to overwrite
an existing index. Use `prepare_agilex_camera_views.py` only when a dataset
explicitly follows its documented input contract; it is not a replacement for
the conversion command above.

Encode the exact task sentences and the 33-frame visual windows:

```bash
cd "$UNIWAM_TRAINING_ROOT"
python scripts/precompute_text_embeds.py task=uniwam_customer_piper_manip_sft
TASK=uniwam_customer_piper_manip_sft \
OUTPUT_ROOT="$UNIWAM_CUSTOM_LATENT_ROOT" \
NPROC_PER_NODE=1 \
bash data_pipeline/launch_precompute_vae.sh
python scripts/preflight_customer_piper.py
```

Raise `NPROC_PER_NODE` only to the number of visible GPUs. The preflight
loads a real sample and verifies the checkpoint dimensions, normalization path,
latent/text caches, 26D action shape, and zero mask on all six auxiliary
features. It must
print `CUSTOMER_SFT_INPUTS_OK` before training.

## Fine-tune

```bash
cd "$UNIWAM_TRAINING_ROOT"
bash scripts/train_zero1.sh 1 task=uniwam_customer_piper_manip_sft
```

The template loads **weights only** from `UNIWAM_INITIALIZATION_CHECKPOINT`, starts a
fresh optimizer at step 0, uses a fresh learning-rate schedule, and defaults
to 20,000 steps with one checkpoint every 2,000 steps. Its per-GPU batch size
is 16. Increase the first launcher argument for more GPUs on one host. Check
the first loss and later loss/throughput in the run log before relying on the
result; a successful Hydra compose or data preflight is not a training run.

## Cloud and Piper edge

### Asynchronous inference

The included async-prefix configuration is enabled without an additional
switch. Training uses `model.async_prefix_training.enabled=true`, with a 70%
chance of sampling a prefix of 6--12 steps; the remaining training samples
use prefix 0. Cloud inference has `contract.async_enabled=true`, a 32-step
action horizon, and a 20-step executable suffix. The client runs at 30 Hz,
starts another request before the current queue is empty, and chooses a prefix
between 6 and 12 steps from a moving estimate of request round-trip time.
The initial request necessarily uses prefix 0 because the queue is empty.

The robot config generated below inherits `prefix_steps=12`,
`min_prefix_steps=6`, `adaptive_prefix.enabled=true`,
`replan_while_queue_nonempty=true`, `chunk_low_watermark=13`, and
`max_queue_steps=32`. Keep these settings consistent with the cloud contract.
`inference_mode=manip_only` chooses the manipulation branch; it does **not**
turn asynchronous execution off. Use `paired` only with a checkpoint and data
contract that include the navigation branch.

Use the fine-tuned checkpoint, **not** the initialization checkpoint, with the
selected task/data contract. Keep the training root and the data
paths accessible to the cloud process because the saved checkpoint config
contains those paths. The cloud service validates the checkpoint config,
Piper stats, camera mosaic, action representation, and prompt prefix before
listening.

```bash
export UNIWAM_PIPER_URDF=/absolute/path/to/piper_description.urdf
export CHECKPOINT=/absolute/path/to/sft-run/checkpoints/weights/step_020000.pt
export TASK_PROMPT='Your exact task sentence from meta/tasks.jsonl'

export UNIWAM_CLOUD_TASK=uniwam_customer_piper_manip_sft
export UNIWAM_CLOUD_SOURCE_INDEX=0
export UNIWAM_CLOUD_AUX_WEIGHT=0.0

CONFIG="$REPO/deployment/configs/uniwam_cloud_customer_async.yaml" \
bash "$REPO/deployment/scripts/precompute_inference_prompt_async_prefix12_rot6d.sh"

PYTHONPATH="$REPO/deployment/cloud:$REPO/deployment/common:$UNIWAM_TRAINING_ROOT/src" \
python "$REPO/deployment/scripts/preflight_cloud_contract.py" \
  --config "$REPO/deployment/configs/uniwam_cloud_customer_async.yaml" \
  --checkpoint "$CHECKPOINT"

CONFIG="$REPO/deployment/configs/uniwam_cloud_customer_async.yaml" \
bash "$REPO/deployment/scripts/launch_uniwam_cloud_server.sh"
```

Run prompt precomputation separately for each task sentence used at inference.
On the Piper machine, generate a config from the **same** calibration JSON
used for training; the generator verifies its matrices against the dataset:

```bash
python "$REPO/deployment/scripts/make_customer_piper_robot_config.py" \
  --dataset-root "$UNIWAM_CUSTOM_DATA" \
  --calibration /absolute/path/to/customer_camera_extrinsics.json \
  --output /absolute/path/to/customer_robot.yaml

CONFIG=/absolute/path/to/customer_robot.yaml \
bash "$REPO/deployment/scripts/launch_uniwam_piper_client.sh" \
  server_host=127.0.0.1 server_port=8031 inference_mode=manip_only \
  prefix_steps=12 min_prefix_steps=6 \
  adaptive_prefix.enabled=true replan_while_queue_nonempty=true \
  task_prompt="$TASK_PROMPT" control.dry_run=true base.dry_run=true
```

If the cloud server runs on a different host reachable by SSH, forward its
port from the robot machine, then change only the client's local port:

```bash
ssh -N -L 18031:127.0.0.1:8031 user@cloud-host
# In another terminal, launch the client with server_port=18031.
```

Keep the SSH forward open while the client runs. The cloud server must first
pass its checkpoint contract check and listen on port 8031. In a healthy client
log, the first `received async-prefix12` line may show `prefix=0`; subsequent
replans should show a prefix in the trained 6--12 range when enough queued
actions remain. `execute_T=20`, `queue`, `rtt`, and `adaptive_prefix` show the
suffix length, queue state, request latency, and next target prefix. A repeated
connection error means the server or tunnel is unreachable, not that the
prefix feature is disabled. Begin with `control.dry_run=true` and
`base.dry_run=true`; physical execution requires independent robot checks.

Activate the environment through `scripts/activate_uniwam_env.sh` after
replacing every placeholder. One dataset or many may be used; define the
corresponding dataset entries and sampling weights in the Hydra data
configuration.

If cloud and edge are on different machines, establish an authenticated SSH
forward for the cloud listen port and set `server_port` to the chosen local
forward. Verify camera topics, observation pose, camera transforms, IK, action
signs, and gripper units in dry-run recordings before any physical motion.
Physical execution is specific to the robot safety procedure and
is not validated by this repository's automated tests.
