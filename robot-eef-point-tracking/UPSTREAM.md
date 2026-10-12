# Source and integration

This directory is a standalone copy of
[KeliangLiu-Lab/robot-eef-point-tracking](https://github.com/KeliangLiu-Lab/robot-eef-point-tracking)
at commit `0ba90131f3d8f432d078102955df0e98299126c6`. Its original
[README](README.md) documents installation, label generation, validation, and
adaptation to other robots. The source files and README are unchanged in this
copy. The original [MIT license](LICENSE) applies to this directory.

For use with UniWAM's manipulation stream, see the
[real-robot training guide](../training/ROBOT.md#prepare-demonstrations). The
generator supplies image-plane training labels and visibility masks; UniWAM's
policy predicts future tracks and actions during inference.
