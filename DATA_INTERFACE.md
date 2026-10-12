# Data interface

MAP-Bench contains 35,454 training trajectories and 15,000 evaluation episodes.
For the training split and relative index layout, see the
[training guide](training/NAVIGATION.md#data-layout).

## Evaluation data

The evaluator is decoupled from the data repository through `--data-root` or
`MAP_BENCH_DATA_ROOT`.

Required paths:

- `evaluation/benchmark_15k.json`
- `evaluation/scene_contexts/<target_uid>.json`

Required benchmark fields:

- `group_id`, `target_uid`, `task`, `scene_id`
- `canonical.start_base_xz`, `canonical.start_theta_rad`, `canonical.start_camera_xz`
- `canonical.frame_count`
- `goal_base_xz`, `operating_camera_xz`, `goal_theta_rad`
- `reference_chunk_count`, `max_prediction_chunk_count`

Required scene-context fields:

- `target_uid`, `nav_y`, `semantic_ids`
- `trajectory.scene_id`

No source video, frame annotation, reference action, or reference summary is
required by the geometry-only benchmark.
