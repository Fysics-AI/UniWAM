# Action Slots and Feature Masks

This document describes the public action contract. Slot order is part of the
checkpoint contract: changing it requires a new model configuration and a new
checkpoint.

## Manipulation action

The public manipulation head is 26-dimensional:

| Slots | Meaning | Required |
| --- | --- | --- |
| 0--2 | left end-effector XYZ in the fixed camera frame | yes |
| 3--8 | left end-effector row-major Rot6D | yes |
| 9 | left gripper width | yes |
| 10--12 | right end-effector XYZ in the fixed camera frame | yes |
| 13--18 | right end-effector row-major Rot6D | yes |
| 19 | right gripper width | yes |
| 20--25 | left X/Y, right X/Y, left visibility, right visibility | optional auxiliary labels |

The first 20 values are the robot control action. The final six values are
image-space end-effector tracking auxiliaries and are never sent to the robot
controller. A source without tracking labels may append six zero values with a
false feature mask, or disable the auxiliary head during fine-tuning. It must
not invent labels.

The included fine-tuning recipe uses `manip_action_dim: 26` and
`robot_manip_action_dim: 20`. The data loader validates the suffix width and
the training loss uses the feature mask when computing its denominator. Thus an
invalid auxiliary value contributes neither a numerator nor a denominator.

## Navigation action

The base navigation action currently used by the released recipe is
3-dimensional:

| Slots | Meaning |
| --- | --- |
| 0--2 | planar target delta `(dx, dy, dtheta)` |

The intended extended navigation contract is 9-dimensional, written as
`3 + 4 + 2`:

| Slots | Meaning |
| --- | --- |
| 0--2 | `(dx, dy, dtheta)` |
| 3--6 | target-object bounding box `(xmin, ymin, xmax, ymax)` |
| 7--8 | target operating point `(x, y)` |

The six auxiliary values use normalized camera-image coordinates. Their order,
alignment, and coordinate frame must be recorded in the sidecar manifest. The
included fine-tuning example keeps `nav_action_dim: 3`, while the repository
also supports the trained 9D navigation contract. Select the matching 9D model,
dataset, sidecar, normalization, cloud decoder, and tests together; changing
only one config value is insufficient. Existing 3D and 9D checkpoints are
separate contracts and cannot consume each other's action head.

## Extending the slots

To train a new slot group:

1. Define a fixed slot order, units, coordinate frame, sentinel value, and
   frame alignment in a sidecar manifest.
2. Add a reader that returns both `[T, D]` values and a boolean `[T, D]`
   feature mask. Missing values must be finite placeholders and must be false
   in the mask.
3. Add the suffix to the dataset output dimension and update the model
   `action_dit_config.*_action_dim`, the dataset `*_action_dim`, and the
   collator target dimension together.
4. Add normalization/statistics for the new slots without mixing them with
   robot-control statistics.
5. Add a unit test for slot order, alignment, missing labels, and loss
   denominator accounting before training.

For the exact 9D configuration changes, follow the next section. The six
auxiliary slots must be appended in the order shown above.

### Enabling the 9D contract

The data loader treats the robot navigation command and the six image labels as
separate parts. Therefore the raw robot navigation dimension remains 3 in the
dataset config; the model output dimension becomes 9 after the sidecar is
attached. A concrete conversion from the included 3D recipe is:

```yaml
# training/configs/model/<your_camera_frame_model>.yaml
action_dit_config:
  nav_action_dim: 9

# for every navigation dataset entry
dataset:
  nav_action_dim: 3          # raw (dx, dy, dtheta)
  robot_nav_action_dim: 3
  nav_aux_index_root: /abs/path/to/navigation_sidecars
  nav_aux_source_name: your_source_name
```

The sidecar directory must contain `manifest.json`,
`nav_aux_xyxy_op.npy` with shape `[total_frames, 6]`, and `feature_mask.npy`
with the same shape. `manifest.json` must declare the slot order
`bbox_x1,bbox_y1,bbox_x2,bbox_y2,op_x,op_y`, the `next_observation`
alignment, and the `cam_nav_original_424x240` coordinate frame. The loader
maps valid sidecar values from `[0,1]` to `[-1,1]`, concatenates them after the
raw 3D command, and the collator produces `[T,9]` actions for the 9D ActionDiT
head. Do not set the dataset's raw `nav_action_dim` to 9: the sidecar path
intentionally requires raw/model robot navigation dimensions of 3.

Regenerate normalization/statistics and text/VAE caches for the selected data,
then run the data preflight before training a fresh 9D checkpoint. A 3D
checkpoint cannot be resized after the fact. For cloud deployment, use a
contract whose runtime model has `model.action_dit_config.nav_action_dim: 9`
and whose checkpoint contract also expects 9D navigation; the cloud decoder
validates this dimension. Use `paired` when the navigation stream is executed,
or `manip_only` when only the manipulation stream is requested.

## Masking rules

- Robot manipulation slots 0--19 are valid whenever the corresponding robot
  action is present.
- Manipulation auxiliary slots 20--25 are valid only when the sidecar marks
  the individual value valid. The `feature_mask` is passed to the masked loss.
- Navigation slots 0--2 are valid for a labeled navigation sample.
- Navigation auxiliary slots 3--8 are valid independently per value. Missing
  bbox or operating-point values are false in the mask and are excluded from
  the loss denominator.
- A missing entire branch uses the branch-level validity flag and contributes
  no branch loss.
- A missing point-tracking source should use
  `append_missing_eef_xy_slots=true` together with
  `disable_manip_aux_training=true` (or `lambda_manip_aux_action=0`).
  Do not set a false mask to true merely to make tensor shapes match.

The implementation is split between the sidecar readers, the dataset/collator,
and the masked action-loss path under `training/src/uniwam`.
