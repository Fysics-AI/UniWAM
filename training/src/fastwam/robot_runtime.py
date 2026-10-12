"""Configuration and initialization shared by real-robot training and inference."""
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from fastwam.models.wan22.uniwam import UniWAM


def load_robot_config(path):
    path = Path(path).resolve()
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    model_path = path.parent / cfg['model_config']
    cfg['model'] = OmegaConf.to_container(OmegaConf.load(model_path), resolve=True)
    action_init = cfg['model'].get('action_dit_pretrained_path')
    if action_init is not None:
        cfg['model']['action_dit_pretrained_path'] = str((model_path.parent / action_init).resolve())
    for key in ('data_root', 'output_dir'):
        cfg[key] = str((path.parent / cfg[key]).resolve())
    return cfg


def make_robot_model(cfg, device, checkpoint=None):
    # Defaults resolve from the repository location, never the caller's cwd.
    repo = Path(__file__).resolve().parents[3]
    os.environ.setdefault('DIFFSYNTH_MODEL_BASE_PATH', str(repo.parent / 'UniWAM-weights' / 'upstream'))
    bundled = repo.parent / 'UniWAM-weights' / 'components' / 'wan2.2'
    if bundled.is_dir():
        os.environ.setdefault('UNIWAM_COMPONENTS_ROOT', str(bundled))
    dtype = torch.bfloat16 if cfg['mixed_precision'] == 'bf16' else torch.float32
    model_cfg = dict(cfg['model'])
    if checkpoint:
        model_cfg['skip_dit_load_from_pretrain'] = True
    model = UniWAM.from_wan22_pretrained(device=str(device), torch_dtype=dtype, **model_cfg)
    if checkpoint:
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.mot.load_state_dict(state['mot'], strict=True)
        if model.proprio_encoder is not None:
            model.proprio_encoder.load_state_dict(state['proprio_encoder'], strict=True)
    model.vae.requires_grad_(False).eval()
    if model.text_encoder is not None:
        model.text_encoder.requires_grad_(False).eval()
    return model
