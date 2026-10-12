import logging
import os
import inspect
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image
import numpy as np
from einops import repeat
from omegaconf import OmegaConf

from .trainer import Wan22Trainer
from .utils.logging_config import get_logger, setup_logging
from .utils.video_io import save_mp4
from .utils import misc

logger = get_logger(__name__)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    if not isinstance(mixed_precision, str):
        raise ValueError(f"`mixed_precision` must be str, got {type(mixed_precision)}")
    key = mixed_precision.strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16












def create_fastwam_nav_causal_aux9(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = False,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    action_causal_local_window: int = 32,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """Factory for prefix-free causal H32 HM3D navigation with 9D actions."""
    from .models.wan22.fastwam_nav_causal_aux9 import FastWAMNavCausalAux9

    repo = Path(__file__).resolve().parents[3]
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(repo.parent / "UniWAM-weights" / "upstream"))
    bundled = repo.parent / "UniWAM-weights" / "components" / "wan2.2"
    if bundled.is_dir():
        os.environ.setdefault("UNIWAM_COMPONENTS_ROOT", str(bundled))

    def as_dict(value, name: str, required: bool = False):
        if isinstance(value, DictConfig):
            value = OmegaConf.to_container(value, resolve=True)
        if value is None:
            if required:
                raise ValueError(f"{name} is required for FastWAMNavCausalAux9.")
            return {}
        if not isinstance(value, dict):
            raise ValueError(f"{name} must resolve to a dict, got {type(value)}")
        return value

    video_dit_config = as_dict(video_dit_config, "video_dit_config", required=True)
    action_dit_config = as_dict(action_dit_config, "action_dit_config", required=True)
    if int(video_dit_config.get("action_dim", -1)) != 9:
        raise ValueError("FastWAMNavCausalAux9 requires video_dit_config.action_dim=9.")
    if int(action_dit_config.get("action_dim", -1)) != 9:
        raise ValueError("FastWAMNavCausalAux9 requires action_dit_config.action_dim=9.")
    video_scheduler = as_dict(video_scheduler, "video_scheduler")
    action_scheduler = as_dict(action_scheduler, "action_scheduler", required=True)
    loss = as_dict(loss, "loss")
    required_scheduler = {"train_shift", "infer_shift", "num_train_timesteps"}
    if required_scheduler - set(action_scheduler):
        raise ValueError(
            f"action_scheduler missing keys: {sorted(required_scheduler - set(action_scheduler))}"
        )

    model = FastWAMNavCausalAux9.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=None,
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
    )
    model.action_causal_local_window = int(action_causal_local_window)
    model.loss_lambda_motion = float(loss.get("lambda_motion", 1.5))
    model.loss_lambda_bbox = float(loss.get("lambda_bbox", 1.0))
    model.loss_lambda_navigable = float(loss.get("lambda_navigable", 1.0))
    model.loss_lambda_video_bbox = float(loss.get("lambda_video_bbox", 0.5))
    return model


















def build_datasets(data_cfg: DictConfig):
    train_ds = instantiate(data_cfg.train)
    if data_cfg.get("val") is None:
        val_ds = train_ds
    else:
        train_stats_path = data_cfg.train.get("pretrained_norm_stats")
        default_stats_path = os.path.join(misc.get_work_dir(), "dataset_stats.json")
        val_stats_path = data_cfg.val.get("pretrained_norm_stats")
        pretrained_norm_stats = val_stats_path or train_stats_path or default_stats_path
        logger.info("Building val dataset with pretrained_norm_stats: %s", pretrained_norm_stats)
        val_ds = instantiate(data_cfg.val, pretrained_norm_stats=pretrained_norm_stats)
    return train_ds, val_ds


def _resolve_train_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    device_count = torch.cuda.device_count()
    if device_count <= 1:
        return "cuda:0"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank < 0 or local_rank >= device_count:
        return "cuda:0"
    return f"cuda:{local_rank}"


def run_training(cfg: DictConfig):
    setup_logging(
        log_level=logging.INFO,
        is_main_process=torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True,
    )
    misc.register_work_dir(cfg.output_dir)
    config_payload = OmegaConf.to_container(cfg, resolve=True)
    with open(Path(cfg.output_dir) / "config.yaml", "w") as f:
        OmegaConf.save(config_payload, f)

    model_device = _resolve_train_device()
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    model = instantiate(
        cfg.model, model_dtype=model_dtype, device=model_device,
        skip_dit_load_from_pretrain=bool(cfg.resume) or bool(cfg.model.skip_dit_load_from_pretrain),
    )
    train_ds, val_ds = build_datasets(cfg.data)

    trainer = Wan22Trainer(
        cfg=cfg,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
    )
    try:
        trainer.train()
    finally:
        trainer.close()
