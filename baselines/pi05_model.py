"""MAP-Bench inference configuration using the public OpenPI model API."""
from pathlib import Path
import json
import jax.numpy as jnp
import numpy as np
import sentencepiece
from openpi import transforms
from openpi.models import model as model_lib
from openpi.models.pi0_config import Pi0Config
from openpi.policies.policy import Policy
from openpi.shared.normalize import NormStats


class NavigationInputs:
    def __call__(self, data):
        image = np.asarray(data["images"]["cam_high"], dtype=np.uint8)
        state = np.asarray(data["state"], dtype=np.float32)
        return {
            "image": {"base_0_rgb": image,
                      "left_wrist_0_rgb": np.zeros_like(image),
                      "right_wrist_0_rgb": np.zeros_like(image)},
            "image_mask": {"base_0_rgb": np.True_,
                           "left_wrist_0_rgb": np.False_,
                           "right_wrist_0_rgb": np.False_},
            "state": state, "prompt": data["prompt"],
        }


class NavigationSlots:
    def __call__(self, data):
        state = np.asarray(data["state"], dtype=np.float32)
        result = np.zeros_like(state)
        result[..., 14], result[..., 16] = state[..., 0], state[..., 1]
        data["state"] = result
        return data


class NavigationTokenizer:
    def __init__(self, path: Path, max_len: int):
        self.model = sentencepiece.SentencePieceProcessor(model_file=str(path))
        self.max_len = max_len

    def tokenize(self, prompt, state=None):
        if state is not None:
            raise ValueError("MAP-Bench pi0.5 uses continuous state input")
        text = prompt.strip().replace("_", " ").replace("\n", " ")
        tokens = self.model.encode(text, add_bos=True) + self.model.encode("\n")
        count = min(len(tokens), self.max_len)
        tokens = tokens[:self.max_len] + [0] * max(0, self.max_len - len(tokens))
        return np.asarray(tokens), np.arange(self.max_len) < count


def create_navigation_policy(checkpoint: Path):
    config = Pi0Config(pi05=True, action_horizon=50, action_dim=32,
                       discrete_state_input=False, pytorch_compile_mode=None)
    model = config.load(model_lib.restore_params(checkpoint / "params", dtype=jnp.bfloat16))
    stats = json.loads((checkpoint / "normalization.json").read_text())["norm_stats"]
    norm = {key: NormStats(**{k: np.asarray(v) for k, v in value.items()}) for key, value in stats.items()}
    return Policy(model, transforms=[
        NavigationInputs(), transforms.Normalize(norm, use_quantiles=True),
        transforms.ResizeImages(224, 224),
        transforms.TokenizePrompt(NavigationTokenizer(checkpoint / "tokenizer.model", config.max_token_len), discrete_state_input=False),
        transforms.PadStatesAndActions(config.action_dim), NavigationSlots(),
    ])
