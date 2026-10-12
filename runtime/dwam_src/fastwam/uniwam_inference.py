"""Inference-only model construction for released UniWAM checkpoints."""

from __future__ import annotations

import torch
from omegaconf import DictConfig, OmegaConf

from .models.wan22.fastwam import FastWAM


class UniWAMInference(FastWAM):
    def __init__(self, *args, action_causal_local_window: int = 32, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.action_causal_local_window = int(action_causal_local_window)

    def load_checkpoint(self, path: str) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if "mot" not in payload:
            raise ValueError("The model file does not contain UniWAM weights")
        self.mot.load_state_dict(payload["mot"], strict=False)
        if self.proprio_encoder is not None and "proprio_encoder" in payload:
            self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = super()._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=action_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        indices = torch.arange(action_seq_len, device=device)
        query = indices[:, None]
        key = indices[None, :]
        causal = (key <= query) & ((query - key) <= self.action_causal_local_window)
        mask[video_seq_len:, video_seq_len:] = causal
        return mask


def _plain(value: DictConfig | dict) -> dict:
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return dict(value)


def create_model(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config: DictConfig | dict,
    action_dit_config: DictConfig | dict,
    video_scheduler: DictConfig | dict,
    action_scheduler: DictConfig | dict,
    tokenizer_max_len: int = 128,
    load_text_encoder: bool = False,
    redirect_common_files: bool = True,
    mot_checkpoint_mixed_attn: bool = False,
    action_causal_local_window: int = 32,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> UniWAMInference:
    video = _plain(video_scheduler)
    action = _plain(action_scheduler)
    model = UniWAMInference.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=tokenizer_max_len,
        load_text_encoder=load_text_encoder,
        proprio_dim=None,
        redirect_common_files=redirect_common_files,
        video_dit_config=_plain(video_dit_config),
        action_dit_config=_plain(action_dit_config),
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
        mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        video_train_shift=5.0,
        video_infer_shift=float(video["infer_shift"]),
        video_num_train_timesteps=1000,
        action_train_shift=5.0,
        action_infer_shift=float(action["infer_shift"]),
        action_num_train_timesteps=1000,
    )
    model.action_causal_local_window = action_causal_local_window
    return model
