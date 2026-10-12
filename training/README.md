# UniWAM training and inference

This directory contains two separate training paths. Choose the one matching
the checkpoint and action contract you intend to use; they are not interchangeable.

| Path | Action and observation contract | Start here |
| :--- | :--- | :--- |
| Camera-frame navigation and dual-arm manipulation | Independent streams, 20D dual-arm state, 26D manipulation action, 3D navigation action in the included configuration, H32 chunks and asynchronous prefix replanning | [Real-robot code and guide](real_robot/README.md) |
| MAP-Bench navigation | One forward RGB view and a 9D navigation action with object-box and MAP-point targets | [Benchmark training and evaluation](NAVIGATION.md) |

## Camera-frame training and deployment

The complete camera-frame implementation lives in [real_robot](real_robot/README.md).
It includes `uniwam` model and dataset code, LeRobot data preparation, Hydra
configs, per-embodiment normalization, text/VAE caching, training, cloud
inference, and the ROS 2 Piper client. Follow its [installation guide](real_robot/docs/INSTALLATION.md),
then its [fine-tuning and deployment guide](real_robot/docs/FINETUNING.md).
The [action-slot contract](real_robot/docs/ACTION_SLOTS.md) explains auxiliary
labels and feature masks. A compatible initialization checkpoint and robot
calibration are external inputs; no task-trained manipulation weights are in
this repository.

Run commands from `training/real_robot`, the root of this self-contained code
path. Its Python modules and requirements are intentionally separate from the
MAP-Bench environment under this directory; do not mix their two `fastwam`
compatibility packages on one `PYTHONPATH`.

## MAP-Bench navigation

[NAVIGATION.md](NAVIGATION.md) documents the existing single-stream HM3D
benchmark, its data layout, training script, and evaluation. It does not use
the dual-arm camera-frame data contract or the ROS 2 deployment path above.
