"""Check pretrained navigation initialization without downloading a full DiT."""
from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training/src"))

import pytest
import torch
from torch import nn

from fastwam.models.wan22 import fastwam as implementation
from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.fastwam_nav_causal_aux9 import FastWAMNavCausalAux9
from fastwam.models.wan22.wan_video_dit import WanVideoDiT


@pytest.mark.parametrize("skip_pretrained", [False, True])
def test_navigation_backbone_initialization_preserves_action_interfaces(monkeypatch, skip_pretrained):
    torch.set_num_threads(2)
    common = dict(num_heads=2, attn_head_dim=12, num_layers=2,
                  text_dim=16, freq_dim=8, eps=1e-6)
    video_config = dict(hidden_dim=24, ffn_dim=48, in_dim=4, out_dim=4,
                        has_image_input=False, patch_size=(1, 2, 2), **common)
    action_config = dict(hidden_dim=16, ffn_dim=32, action_dim=9, **common)
    video = WanVideoDiT(**video_config)
    with torch.no_grad():
        for parameter in video.parameters():
            parameter.fill_(0.25)
    components = SimpleNamespace(dit=video, vae=nn.Identity(), text_encoder=None,
                                 tokenizer=None, dit_path="wan", vae_path="vae",
                                 text_encoder_path=None, tokenizer_path=None)
    monkeypatch.setattr(implementation, "load_wan22_ti2v_5b_components", lambda **kw: components)

    torch.manual_seed(17)
    initial_action = ActionDiT(**action_config)
    torch.manual_seed(17)
    model = FastWAMNavCausalAux9.from_wan22_pretrained(
        device="cpu", torch_dtype=torch.float32, video_dit_config=video_config,
        action_dit_config=action_config, load_text_encoder=False,
        skip_dit_load_from_pretrain=skip_pretrained,
    )
    actual = model.action_expert.state_dict()
    for key, original in initial_action.state_dict().items():
        if skip_pretrained or key.startswith(("action_encoder.", "head.")):
            torch.testing.assert_close(actual[key], original)
        else:
            source = video.state_dict()[key]
            scale = (source.shape[-1] / original.shape[-1]) ** 0.5 if original.ndim >= 2 else 1.0
            torch.testing.assert_close(actual[key], torch.full_like(original, 0.25 * scale))
