# Environment

Use separate environments for the Habitat evaluator and the UniWAM policy
service. Install Habitat-Sim 0.3.3 following its official instructions for the
local CUDA version, then install the evaluator's lightweight dependencies:

```bash
python3 -m pip install -r requirements.txt
```

The evaluator setup uses Python 3.9, camera resolution 566x320, 15 Hz source
semantics, and action horizon 32. HM3D v0.2 is an external licensed dependency.

For the policy service, the inspected local environment uses Python 3.10.12,
PyTorch 2.7.0+cu128, Hydra-Core 1.3.2, OmegaConf 2.3.0, NumPy 2.2.6,
Pillow 12.2.0, WebSockets 16.0, Msgpack 1.1.0, Einops 0.8.2,
Transformers 4.53.2, Accelerate 1.5.2, Hugging Face Hub 0.36.2, and
Safetensors 0.7.0. Install the matching CUDA build of PyTorch through its
official instructions, then run `python3 -m pip install -r requirements-policy.txt`.
The upstream Wan2.2 VAE must be available to the model
loader via its Hugging Face source or cache. The two environments communicate
over the WebSocket policy interface and may use different Python versions.
