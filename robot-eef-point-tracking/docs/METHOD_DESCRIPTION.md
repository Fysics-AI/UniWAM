# Algorithm Description

The proposed labeling pipeline consists of three complementary procedures.
First, **Calibrated Main-View End-Effector Geometry** converts the robot's
Cartesian end-effector state into physically defined 3D points and projects
them into the main camera using source-episode camera intrinsics, camera
extrinsics, and the dual-arm base transform. This stage produces the canonical
J6 reference, the visible flange candidate, and a hard geometric validity mask;
it is deterministic and does not depend on temporal visual tracking. Second,
**Appearance-Gated Visibility for the Geometric End-Effector Track** evaluates
the DINOv2 feature at the exact projected image token and compares it with
reviewed positive and negative prototype banks. A positive-minus-negative
cosine margin, hysteresis thresholds, segment-local interpolation, and temporal
morphology produce a visibility mask and an uncertainty-aware quality label,
while the geometric coordinate remains unchanged. Third, **End-to-End
Geometry-Guided Visibility-Aware Label Generation** applies the two procedures
independently to all episodes in parallel, writes each NPZ/report pair
atomically, records calibration and input identities, and performs a complete
manifest-level validation before publishing the dataset. This separation makes
the coordinate target physically reproducible while allowing appearance-based
occlusion handling without the drift commonly introduced by free-running point
trackers.

## Symbol Notes

The paper algorithm uses explicit transform directions:

- $E_{c\leftarrow\mathrm{ref}}$ maps a chosen robot reference frame to camera coordinates.
- $M^i_{\mathrm{ref}\leftarrow b_i}$ maps arm $i$'s native base frame to that reference
  frame. This notation covers both fixed dual-arm calibration and mobile AGX
  arm-mount translations without conflating their coordinate systems.
- $h(\cdot)$ lifts a 3D point to homogeneous coordinates.
- $\operatorname{xyz}(\cdot)$ drops the homogeneous coordinate after a rigid
  transform.
- $\delta_{\mathrm{J6}}$ is the canonical J6 offset; $\delta_{\mathrm{vis}}$
  is the separately named visible flange offset.
- $g_t^i$ is hard geometric validity for arm $i$ at frame $t$.
- $z_t^i$ is appearance-confirmed visibility; $q_t^i\in\{0,1,2\}$ denotes
  invalid, visible, and geometrically in-frame but uncertain/occluded.
