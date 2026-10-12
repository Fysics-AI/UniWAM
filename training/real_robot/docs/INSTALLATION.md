# Installation

The commands below create a reproducible Python environment for training and
cloud inference. Run them on the machine that has the GPUs and model weights.
The robot client is installed separately on the ROS 2 machine when the cloud
server and robot are not colocated.

## 1. Create a Conda environment

Conda or micromamba may be used. The tested software contract is Python 3.10.

```bash
conda create -n uniwam python=3.10 -y
conda activate uniwam
conda install -c conda-forge ffmpeg git ninja -y
python --version
ffmpeg -version
```

`ffmpeg` is used when building VAE latent caches from episode videos. `ninja`
is needed by some CUDA extensions and by optional compilation paths.

## 2. Install PyTorch for the installed CUDA driver

Check the driver first:

```bash
nvidia-smi
```

Install a PyTorch wheel whose CUDA runtime is supported by that driver. For
example, the following is a CUDA 12.8 wheel command:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
```

If the cluster uses another supported CUDA runtime, replace `cu128` with the
matching PyTorch wheel index. Do not install a CPU-only wheel for training.
Verify CUDA before continuing:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_version", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see CUDA; install a matching wheel first")
print("gpu_count", torch.cuda.device_count())
PY
```

## 3. Install UniWAM dependencies

From the repository root:

```bash
python -m pip install -r training/requirements-training.txt
python -m pip install -r deployment/requirements-cloud.txt
```

The first file installs Hydra, Accelerate, DeepSpeed, dataset/video readers,
and training utilities. The second installs the additional cloud-side IK and
robot-interface packages. The requirements files intentionally do not select
a CUDA build of PyTorch; install that first as shown above.

Check the distributed launcher and imports:

```bash
python - <<'PY'
import accelerate, deepspeed, hydra, numpy, pyarrow, torch
print("imports_ok")
print("accelerate", accelerate.__version__)
print("torch", torch.__version__)
PY

(cd training && PYTHONPATH="$PWD/src:$PWD" python scripts/train.py --help >/dev/null)
```

## 4. Configure local paths

Edit the path block in `scripts/activate_uniwam_env.sh`, then source it from
the same shell used for data preparation and training:

```bash
source scripts/activate_uniwam_env.sh
```

The script exports the training root, model-weight directory, initialization
checkpoint, dataset directory, H32 window index, text-cache directory, and VAE
latent directory. It validates the model directory and checkpoint immediately;
the dataset and cache paths may be created by later pipeline steps.

## 5. Robot-side Python environment

The ROS 2 client must use the system Python 3.10 associated with ROS 2 Humble.
On the robot machine, install ROS 2 and the robot vendor packages first. Then
run:

```bash
sudo apt-get install -y python3.10-venv
bash deployment/scripts/bootstrap_robot_python_deps.sh
```

The script creates `deployment/.robot_py310` with system-site-packages so ROS
imports remain available, and installs the client-only Python dependencies. Use
the printed interpreter path when launching the robot client.

## 6. First smoke tests

Before preparing a full dataset, verify the environment and configuration:

```bash
python -c 'import torch; print(torch.cuda.is_available())'
python training/data_pipeline/prepare_customer_piper_views.py --help
python training/data_pipeline/prepare_customer_piper_h32.py --help
PYTHONPATH="$PWD/deployment/cloud:$PWD/deployment/common:$PWD/training/src" \
  python deployment/scripts/preflight_cloud_contract.py --help
```

After the dataset, text cache, and VAE cache are ready, run the task preflight
before starting distributed training. The complete order is documented in
[Fine-tuning and Deployment](FINETUNING.md).
