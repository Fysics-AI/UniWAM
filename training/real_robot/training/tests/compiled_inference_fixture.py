"""Small mixed-stream policy fixture with real DiT blocks and a tiny VAE."""

import torch
from types import SimpleNamespace

from uniwam.models.wan22.fastwam_mixed_stream import FastWAMMixedStream
from uniwam.models.wan22.mixed_stream_action_dit import MixedStreamActionDiT
from uniwam.models.wan22.mot import MoT
from uniwam.models.wan22.wan_video_dit import WanVideoDiT


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Conv3d(3, 4, kernel_size=(1, 8, 8), stride=(1, 8, 8))

    def encode(self, video, scale):
        return self.projection(video) * scale


class TinyVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyEncoder()
        self.scale = 1.0

    def encode(self, video, device, **kwargs):
        return self.model.encode(video.to(device), self.scale)


def small_policy(device, compiled=True):
    torch.manual_seed(0)
    shared = dict(hidden_dim=24, ffn_dim=48, text_dim=16, freq_dim=16,
                  eps=1e-6, num_heads=2, attn_head_dim=12, num_layers=1)
    video = WanVideoDiT(**shared, in_dim=4, out_dim=4, patch_size=(1, 2, 2),
                       has_image_input=False, seperated_timestep=True)
    action = MixedStreamActionDiT(**shared, manip_action_dim=26, nav_action_dim=3,
                                  manip_horizon=32, nav_horizon=32)
    mot = MoT({"video": video, "action": action}, mot_checkpoint_mixed_attn=False)
    model = FastWAMMixedStream(
        video_expert=video, action_expert=action, mot=mot, vae=TinyVAE(),
        text_dim=16, device=device, torch_dtype=torch.float32,
        manip_robot_action_dim=20, async_prefix_train=True,
        async_prefix_lambda_attention=True, async_prefix_min_length=0,
        async_prefix_max_length=12, compile_action_infer=compiled,
        compile_vae_infer=compiled,
    )
    model.eval()
    return model


def inputs(device, mode, prefix_length=0):
    image = torch.linspace(-1, 1, 3 * 32 * 32, device=device).reshape(1, 3, 32, 32)
    payload = dict(
        action_horizon=32,
        context=torch.zeros(1, 4, 16, device=device),
        context_mask=torch.ones(1, 4, device=device, dtype=torch.bool),
        manip_input_image=image if mode in ("paired", "manip_only") else None,
        nav_input_image=image if mode in ("paired", "nav_only") else None,
        num_inference_steps=2,
        seed=123,
        prefix_length=prefix_length,
    )
    if prefix_length:
        if mode in ("paired", "manip_only"):
            payload["manip_action_prefix"] = torch.zeros(prefix_length, 26, device=device)
        if mode in ("paired", "nav_only"):
            payload["nav_action_prefix"] = torch.zeros(prefix_length, 3, device=device)
    return payload
