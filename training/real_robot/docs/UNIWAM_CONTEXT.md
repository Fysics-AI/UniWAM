# UniWAM Architecture Overview

UniWAM combines camera-frame manipulation and mobile navigation in a shared
video/action model. The public interfaces are organized around three ideas:

1. **Mixed streams.** Manipulation and navigation samples are packed and routed
   through their respective visual and action interfaces while sharing the
   backbone.
2. **Camera-frame action targets.** Dual-arm end-effector state and action use
   a fixed camera frame, row-major Rot6D, and explicit embodiment-matched
   normalization. Optional image-plane labels are carried by feature masks.
3. **Asynchronous execution.** The cloud policy returns action chunks while
   the robot client executes the previous chunk, with configurable prefix and
   replanning behavior.

The repository is intended to be adapted to new tasks and datasets.
The data pipeline requires calibration, episode tables,
task prompts, latent caches, and statistics that match the selected embodiment
and action representation.
