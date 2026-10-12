# UniWAM Edge Deployment

Use the UniWAM Piper client with the matching source-specific cloud contract.
Start in dry-run mode, verify camera-frame transforms, IK, gripper ranges, and
latency, then enable physical execution through the robot safety procedure.

The cloud service must be started with the matching `uniwam_source*.yaml`
contract and a checkpoint from the six-source parent run. Supply the exact
task sentence used by that source when precomputing the inference prompt.

For a new Piper/AgileX single-source fine-tune and customer-owned task prompt,
follow `docs/CUSTOMER_PIPER_FINETUNING.md` instead. It uses source index 0,
the customer SFT contract, and the fine-tuned checkpoint.
