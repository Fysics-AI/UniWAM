# Data Preparation

The data path starts from LeRobot episode tables and produces
camera-frame state/action columns, H32 windows, text embeddings, and VAE
latents. It does not require any bundled dataset or a fixed number of sources.

For a Piper-compatible dual-arm dataset, first edit and source
`scripts/activate_uniwam_env.sh` from the repository root. The dataset
directory, window index, text cache, and latent directory are output paths;
they may not exist when the environment is first loaded.

## Camera-frame conversion

Record the two measured rigid transforms from arm base to fixed camera in a
JSON file. The input LeRobot tables must have 20D dual-arm state and action
columns, with each arm ordered as XYZ, row-major Rot6D, and gripper width.
For an older observation with a 3D base prefix, use
`--raw-state-layout base3_arm20`. Do not copy another robot's calibration.

```bash
python "$REPO/training/data_pipeline/prepare_customer_piper_views.py" \
  --source-root /absolute/path/to/customer_lerobot \
  --output-root "$UNIWAM_CUSTOM_DATA" \
  --calibration /absolute/path/to/customer_camera_extrinsics.json

python "$REPO/training/data_pipeline/prepare_customer_piper_h32.py" \
  --dataset-root "$UNIWAM_CUSTOM_DATA" \
  --calibration /absolute/path/to/customer_camera_extrinsics.json \
  --output "$UNIWAM_CUSTOM_WINDOW_INDEX"
```

The converter refuses to overwrite existing output and validates Rot6D
orthogonality. Source frames in the supplied config are `424x240` pixels
(`W x H`). The loader puts a `384x216` main image above two `192x108` wrist
images, then crops two rows from both the top and bottom of the resulting
`384x324` mosaic. The model sees `384x320` pixels. See the layout diagram in
the repository README. Adjust `shape_meta` and camera preprocessing together
if the recording resolution differs.

## Text and latent caches

```bash
cd "$UNIWAM_TRAINING_ROOT"
python scripts/precompute_text_embeds.py task=uniwam_customer_piper_manip_sft
TASK=uniwam_customer_piper_manip_sft \
OUTPUT_ROOT="$UNIWAM_CUSTOM_LATENT_ROOT" \
NPROC_PER_NODE=1 \
bash data_pipeline/launch_precompute_vae.sh
python scripts/preflight_customer_piper.py
```

The prompt cache uses the exact task text from `meta/tasks.jsonl`. The latent
source name must match the prepared dataset
directory name. The preflight loads a real sample and checks the action shape,
normalization range, and missing-label mask before training.

## More datasets and action slots

For several datasets, add one entry per source to a copy of the Hydra
data config. Set `source_sampling_weights` to the desired mixture and give each
entry its own dataset root, H32 index, latent source, prompt set, and compatible
statistics. The number of entries and weights are experiment parameters, not
requirements imposed by UniWAM.

The default manipulation output is 20 robot values plus six optional
point-tracking values. A dataset without tracking annotations should append
six zero placeholders with false feature masks and disable auxiliary training.
Navigation output may be 3D or 9D. The 9D form appends a 4D target bbox and 2D
operating point, with matching sidecar, mask, model head, and decoder. See
`docs/ACTION_SLOTS.md` for slot order and masking.

The path above is self-contained once the experiment's data,
calibration, model weights, and caches are available.
