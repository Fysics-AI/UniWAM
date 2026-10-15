# UniWAM Edge Deployment

Use the UniWAM Piper client with a matching cloud configuration.
Start in dry-run mode, verify camera-frame transforms, IK, gripper ranges, and
latency, then enable physical execution through the robot safety procedure.

Start the cloud service with `deployment/configs/uniwam_cloud_customer_async.yaml`
and the checkpoint trained for your task. Its saved training configuration,
normalization, camera layout, and action representation must match deployment.
Supply the exact task sentence used for training when preparing the inference
prompt.

Follow the [fine-tuning and deployment guide](../../docs/FINETUNING.md).
The included single-source example uses source index 0 and your fine-tuned
checkpoint; adapt the camera and action contracts together for other robots.
