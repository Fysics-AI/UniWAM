# MAP-Bench evaluation

MAP-Bench provides 35,454 training trajectories and 15,000 evaluation episodes.
This guide evaluates policies on the evaluation split. The observation is a single
forward-facing RGB view and a navigation instruction. UniWAM uses only its
navigation expert for this protocol; manipulation is not evaluated here.

Short / Medium / Long are defined by reference path length at 1.5 m and 3.5 m.
Position and heading errors are evaluated at the final base pose. The overall
error is the equal-weight mean of the three level means. Position success uses
0.50 m and heading success uses 15°. A policy action chunk has 32 actions.
A chunk is near zero when at least 80% of actions have translation ≤ 0.05 m
and heading ≤ 0.05 rad, including the final action. Three consecutive near-zero
chunks trigger model stop. Episode limits and simulator implementation are in
[`scripts/source_snapshot/framework_final/`](../scripts/source_snapshot/framework_final/).

Use `scripts/launch_external_eval.py --help` for batching and GPU options;
`--dry-run` validates input paths and prints commands. Use a fresh output
folder for each method. The policy server and evaluator must have matching
ports. Policy servers started by `launch_uniwam.py` record logs and process IDs
under `outputs/policies/`.

The release provides per-trajectory metrics and raw traces for five methods,
15,000 traces per method. Paths in `results/trajectory_metrics.csv` are relative
to MAP-Bench. The CSV records each uncompressed trace's SHA256. `--traces` on the
download helper extracts the five archives after verifying archive checksums.

Refer to the [policy interface](../POLICY_INTERFACE.md) for adapters and
[data interface](../DATA_INTERFACE.md) for benchmark fields. Scene context files
are per-context assets, so their file count is not a count of distinct HM3D scenes.
