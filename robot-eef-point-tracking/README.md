# Calibrated Robot End-Effector Point Tracking

Generate per-frame 2D end-effector labels from robot poses and a calibrated
camera. Geometry determines the pixel coordinate; DINOv2 decides whether that
point is visible. The same approach works for a single arm or multiple arms,
including other embodiments such as Franka when their pose and camera
calibration are provided.

This UniWAM directory includes a ready-to-run LeRobot adapter, a reusable
visibility stage, preview rendering, and label validation. Datasets, videos,
DINOv2 code, and model weights are not included. The output supplies the
manipulation stream's image-plane auxiliary labels; see the
[UniWAM training guide](../training/ROBOT.md#prepare-demonstrations).

## Install

You need Linux, Python 3.10-3.12, a CUDA-capable GPU for DINOv2, and
`ffmpeg`/`ffprobe` for preview videos.

```bash
git clone https://github.com/Fysics-AI/UniWAM.git
cd UniWAM/robot-eef-point-tracking
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
```

Install CUDA-enabled PyTorch and torchvision for your driver using
https://pytorch.org/get-started/locally/, then install the remaining packages:

```bash
python3 -m pip install -r requirements.txt
```

Provide a local checkout of
[DINOv2](https://github.com/facebookresearch/dinov2) and a matching
`dinov2_vitb14` checkpoint. The loader accepts a PyTorch state dictionary or
a checkpoint with a `model` entry.

## Run The Included LeRobot Adapter

Set `EEF_DATA_ROOT` to the directory containing the LeRobot datasets. The
included geometry launcher uses the dataset names and calibration profiles in
`src/agx_calibration_registry.py` and `scripts/run_geometry.sh`. Keep the
original episode ordering, state representation, source metadata, and videos.

```bash
export EEF_DATA_ROOT=/absolute/path/to/lerobot
export WORKERS=16
bash scripts/run_geometry.sh
```

This produces `outputs/flange_geometry/geometry_manifest.jsonl` and one
geometry NPZ per episode under `outputs/flange_geometry/`. The launcher
projects the selected robot point into the main manipulation view; navigation
and wrist cameras are not used.

The supplied `configs/dino_prototypes.json` contains positive and negative
image points on videos from the included datasets. Check that those videos are
present under `EEF_DATA_ROOT` before running visibility inference.

```bash
python3 scripts/run_visibility.py \
  --data-root "$EEF_DATA_ROOT" \
  --manifest outputs/flange_geometry/geometry_manifest.jsonl \
  --geometry-root outputs/flange_geometry \
  --output-root outputs/eef_tracks \
  --dinov2-repo /absolute/path/to/dinov2 \
  --checkpoint /absolute/path/to/dinov2_vitb14_pretrain.pth \
  --devices 0 \
  --workers-per-device 1 \
  --batch-size 16
```

Use a comma-separated `--devices` list for multiple GPUs. Increase
`--workers-per-device` or `--batch-size` only if memory permits. Interrupted
runs resume after verifying their inputs and outputs.

Validate the completed included dataset:

```bash
bash scripts/run_validation.sh
```

This launcher checks the complete included dataset contract. For a subset,
call `src/validate_labels.py` directly with the generated geometry manifest
and without `--require-complete`. A filtered geometry run may name its
manifest `geometry_manifest.partial.jsonl`.

## Read The Labels

Visibility outputs are written to
`outputs/eef_tracks/DATASET/chunk-000/episode_000000.npz`. The main arrays
have `T` frames and `A` tracked arms:

| Array | Shape | Meaning |
|---|---|---|
| `track_xy_geom` | `[T, A, 2]` | Calibrated `(x, y)` coordinates in original video pixels |
| `operation_track_geometric_in_fov` | `[T, A]` | Point has positive depth, lies in the image, and is in the selected operation phase |
| `track_visible` | `[T, A]` | DINOv2 confirms the projected point's appearance |
| `track_xy` | `[T, A, 2]` | Visible-only coordinates; invisible entries are NaN |
| `quality` | `[T, A]` | 0: invalid; 1: visible; 2: in-frame but uncertain or occluded |

```python
import numpy as np

with np.load("episode_000000.npz", allow_pickle=False) as labels:
    xy = labels["track_xy_geom"].astype(np.float32)
    geometric_mask = labels["operation_track_geometric_in_fov"].astype(bool)
    visible_mask = labels["track_visible"].astype(bool)

# Use geometric_mask for in-frame targets, or geometric_mask & visible_mask
# when training only on appearance-confirmed points.
```

If you resize or crop the video, apply the same image transform to the pixel
coordinates. Do not train directly on `track_xy` without masking its NaNs.

Render a review video in H.264/yuv420p:

```bash
bash scripts/render_preview.sh \
  /absolute/path/to/main_view.mp4 \
  outputs/eef_tracks/DATASET/chunk-000/episode_000000.npz \
  /tmp/eef_preview.mp4 \
  2
```

## Use Another Robot

The projection stage is where an embodiment is defined. For a new robot or
camera, provide the following in a geometry adapter:

1. A time-aligned end-effector pose for each frame. If the data contains only
   joint angles, compute the pose with the robot's forward kinematics.
2. A 3D point in the end-effector frame, such as a flange center or gripper
   center. Name the point explicitly and keep the same definition throughout
   the dataset.
3. Camera intrinsics `K` and a transform from the robot reference frame to
   the camera. For multiple bases, also provide each base-to-reference
   transform. Project the 3D point only when depth is positive; mark points
   outside the image invalid.

For each frame and arm, the projection is
`p_camera = T_camera_base @ T_base_eef @ [point_eef, 1]`, followed by
`pixel = (K @ p_camera[:3])[:2] / p_camera[2]`. Use the original video width
and height for the image-bound check.

Write one geometry NPZ with `algorithm_version`, `track_xy_geom` of shape
`[T, A, 2]`, `track_geometric_in_fov` of shape `[T, A]`, and
`operation_mask` of shape `[T]`. Write a JSONL manifest inside the geometry
root with one entry per episode:

```json
{"dataset":"my_robot_task","episode_index":0,"geometry_npz":"/absolute/path/to/episode_000000.npz","video":"/absolute/path/to/episode_000000.mp4"}
```

For a new camera or robot appearance, provide your own positive and negative
prototype points in original-video pixels. The `video` path can be absolute
or relative to `--data-root`:

```json
{
  "positive": [{"video": "/absolute/path/to/video.mp4", "frame": 120, "x": 320, "y": 240}],
  "negative": [{"video": "/absolute/path/to/video.mp4", "frame": 10, "x": 320, "y": 240}]
}
```

Run visibility on the new geometry:

```bash
python3 scripts/run_visibility.py \
  --data-root /absolute/path/to/data \
  --manifest /absolute/path/to/geometry/geometry_manifest.jsonl \
  --geometry-root /absolute/path/to/geometry \
  --output-root /absolute/path/to/eef_tracks \
  --prototypes /absolute/path/to/prototypes.json \
  --dinov2-repo /absolute/path/to/dinov2 \
  --checkpoint /absolute/path/to/dinov2_vitb14_pretrain.pth \
  --devices 0 --workers-per-device 1 --batch-size 16
```

Its output uses the same label arrays described above. The included
full-dataset validation launcher checks the included adapter's metadata;
add adapter-specific checks for a new robot.

## Troubleshooting

- Points are consistently offset: check the camera-transform direction,
  intrinsic resolution, end-effector-frame offset, and pose convention.
- A point is missing at the start of an episode: inspect `operation_mask`,
  positive depth, image bounds, and the visibility prototype points.
- Prototype video not found: correct `--data-root` or use an explicit
  `video` path in the prototype JSON.
- CUDA out of memory: reduce `--batch-size` or `--workers-per-device`.

## License

The repository is MIT-licensed. DINOv2, PyTorch, datasets, and model weights
retain their own licenses and are not redistributed here.
