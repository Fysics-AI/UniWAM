"""Cloud adapter for the nav3/manip26 EEF-XY+visibility async-prefix rot6D policy.

The deployment configuration describes the exact training contract for one
checkpoint family.  Before loading weights this module checks both the runtime
Hydra task and the immutable config stored beside the checkpoint.  That keeps
the model's action horizon, temporal VAE layout, attention mask and action
representation from drifting apart at deployment time.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from uniwam_piper_common.future_state import queued_nav_target_path
from uniwam_piper_common.nonholonomic import (
    NonholonomicPlannerConfig,
    plan_nonholonomic_commands,
)
from uniwam_piper_common.rotation6d import (
    fixed_reference_absolute_pose,
    continuous_rpy_xyz_from_matrix_sequence,
    matrix_to_rotation_6d,
    matrix_to_rpy_xyz,
    fixed_reference_relative_pose,
    rotation_6d_to_matrix,
    rpy_xyz_to_matrix,
)

from .observation_rot6d import Rot6DObservationBuilder
from .protocol_rot6d import INFERENCE_MODES


EXPECTED_MODEL_TARGET = "uniwam.runtime.create_fastwam_mixed_stream"
EXPECTED_STATE_DIM = 20
RAW_STATE_DIM = 23
EXPECTED_ROBOT_MANIP_ACTION_DIM = 20
EXPECTED_EEF_XY_ACTION_DIM = 6
EXPECTED_BBOX_ACTION_DIM = EXPECTED_EEF_XY_ACTION_DIM
EXPECTED_MANIP_ACTION_DIM = EXPECTED_ROBOT_MANIP_ACTION_DIM + EXPECTED_EEF_XY_ACTION_DIM
EXPECTED_ROBOT_NAV_ACTION_DIM = 3
EXPECTED_NAV_AUX_ACTION_DIM = 0
EXPECTED_NAV_ACTION_DIM = EXPECTED_ROBOT_NAV_ACTION_DIM + EXPECTED_NAV_AUX_ACTION_DIM
EXPECTED_VIDEO_SIZE = [320, 384]
EXPECTED_TYPE_TOKEN_COUNT = 4
EXPECTED_CONCAT_MODE = "parallel_nav_manip"
EXPECTED_NAV_IMAGE_MODE = "matched_wrist_mosaic"
EXPECTED_CONTROL_HZ = 30.0
def mixed_precision_to_dtype(value: str) -> torch.dtype:
    value = str(value).strip().lower()
    if value == "bf16":
        return torch.bfloat16
    if value == "fp16":
        return torch.float16
    if value in {"no", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported mixed_precision={value!r}.")


def load_runtime_cfg(
    project_root: Path,
    task: str,
    *,
    compile_action_infer: bool = False,
    compile_vae_infer: bool = False,
) -> Any:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    overrides = [
        f"task={task}",
        "model.load_text_encoder=false",
        "model.mot_checkpoint_mixed_attn=false",
        "model.skip_dit_load_from_pretrain=true",
        f"model.compile_action_infer={'true' if compile_action_infer else 'false'}",
        f"model.compile_vae_infer={'true' if compile_vae_infer else 'false'}",
    ]
    with initialize_config_dir(version_base="1.3", config_dir=str(project_root / "configs")):
        return compose(config_name="train", overrides=overrides)


def paired_data_cfg(cfg: Any, source_index: int = 0, required_branches: str = "both") -> Any:
    datasets = cfg.data.train.get("datasets", None)
    if datasets is None:
        raise ValueError("rot6D deployment requires a mixed-stream data config.")
    if source_index < 0 or source_index >= len(datasets):
        raise IndexError(f"paired_source_index={source_index} is outside source count {len(datasets)}.")
    source = datasets[source_index]
    data = source.get("dataset", None)
    if data is None:
        raise ValueError(f"Source {source_index} does not wrap a dataset config.")
    available = str(data.get("available_branches", ""))
    if required_branches == "manip" and available in {"manip", "both"}:
        return data
    if available != "both":
        raise ValueError(
            f"Source {source_index} must provide both inference streams, got "
            f"available_branches={available!r}."
        )
    return data


def checkpoint_train_config(checkpoint: Path) -> Path:
    if checkpoint.parent.name != "weights" or checkpoint.parent.parent.name != "checkpoints":
        raise ValueError(f"Unexpected checkpoint layout: {checkpoint}")
    config = checkpoint.parents[2] / "config.yaml"
    if not config.is_file():
        raise FileNotFoundError(f"Checkpoint training config is missing: {config}")
    return config


def resolved_value(cfg: Any, key: str) -> Any:
    value = OmegaConf.select(cfg, key)
    value = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    # Checkpoints created before the public namespace migration embed the old
    # Hydra target.  The compatibility package resolves it to the same model;
    # normalize the spelling here so old and new checkpoints share one contract.
    if key == "model._target_" and value == "fastwam.runtime.create_fastwam_mixed_stream":
        return EXPECTED_MODEL_TARGET
    return value


def _check_equal(mismatches: list[str], name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        mismatches.append(f"{name}: got={actual!r}, expected={expected!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stats_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    return {key: payload[key] for key in ("state", "action", "contract")}


def _expected_stats_contract(train_path: Path, project_root: Path) -> dict[str, Any]:
    if train_path.is_file():
        return _stats_contract(train_path)
    known = {
        "camera_frame_franka_only_q01q99.json": "franka",
        "camera_frame_piper_agx_only_q01q99.json": "piper_agx",
    }
    group = known.get(train_path.name)
    if group is None:
        raise FileNotFoundError(
            f"Checkpoint stats are unavailable and are not the known parent artifact: {train_path}"
        )
    name = train_path.name
    reference = project_root / "data_pipeline/parent_200k_artifact_reference.json"
    if not reference.is_file():
        raise FileNotFoundError(f"Parent stats reference is missing: {reference}")
    candidate = project_root / "data_indices" / name
    if not candidate.is_file():
        raise FileNotFoundError(f"Parent stats replacement is missing: {candidate}")
    expected_hash = str(json.loads(reference.read_text())["stats_sha256"][group])
    if _sha256(candidate) != expected_hash:
        raise ValueError(f"Parent stats replacement differs from the reference: {candidate}")
    return _stats_contract(candidate)


def _optional_int(value: Any) -> int | None:
    if value in (None, "null", "None", ""):
        return None
    return int(value)


@dataclass(frozen=True)
class AsyncPrefixContract:
    """Immutable train/deploy settings for one checkpoint family."""

    name: str
    action_horizon: int
    num_frames: int
    latent_temporal_frames: int
    replan_steps: int
    max_suffix_steps: int | None
    async_prob: float
    min_prefix_steps: int
    max_prefix_steps: int
    lambda_attention: bool
    local_window: int
    random_prefix_attention_mask: bool
    prefix_mask_prob: float
    keep_last_k: int
    rope_offset: int
    dynamic_loss_weighting: bool
    async_enabled: bool
    manip_aux_loss_weight: float

    @classmethod
    def from_deploy_cfg(cls, deploy_cfg: Any) -> "AsyncPrefixContract":
        raw = deploy_cfg.get("contract", None)
        if raw is None:
            raise ValueError("Deployment config must define a contract section.")
        contract = OmegaConf.to_container(raw, resolve=True) if OmegaConf.is_config(raw) else dict(raw)
        required = (
            "name",
            "action_horizon",
            "num_frames",
            "latent_temporal_frames",
            "replan_steps",
            "max_suffix_steps",
            "async_prob",
            "min_prefix_steps",
            "max_prefix_steps",
            "lambda_attention",
            "local_window",
            "random_prefix_attention_mask",
            "prefix_mask_prob",
            "keep_last_k",
            "rope_offset",
            "dynamic_loss_weighting",
            "async_enabled",
            "manip_aux_loss_weight",
        )
        missing = [key for key in required if key not in contract]
        if missing:
            raise ValueError(f"Deployment contract is missing keys: {missing}")
        result = cls(
            name=str(contract["name"]),
            action_horizon=int(contract["action_horizon"]),
            num_frames=int(contract["num_frames"]),
            latent_temporal_frames=int(contract["latent_temporal_frames"]),
            replan_steps=int(contract["replan_steps"]),
            max_suffix_steps=_optional_int(contract["max_suffix_steps"]),
            async_prob=float(contract["async_prob"]),
            min_prefix_steps=int(contract["min_prefix_steps"]),
            max_prefix_steps=int(contract["max_prefix_steps"]),
            lambda_attention=bool(contract["lambda_attention"]),
            local_window=int(contract["local_window"]),
            random_prefix_attention_mask=bool(contract["random_prefix_attention_mask"]),
            prefix_mask_prob=float(contract["prefix_mask_prob"]),
            keep_last_k=int(contract["keep_last_k"]),
            rope_offset=int(contract["rope_offset"]),
            dynamic_loss_weighting=bool(contract["dynamic_loss_weighting"]),
            async_enabled=bool(contract["async_enabled"]),
            manip_aux_loss_weight=float(contract["manip_aux_loss_weight"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.action_horizon <= 0:
            raise ValueError("contract.action_horizon must be positive.")
        if self.num_frames != self.action_horizon + 1:
            raise ValueError(
                "contract.num_frames must equal action_horizon + 1, got "
                f"{self.num_frames} for horizon {self.action_horizon}."
            )
        if self.latent_temporal_frames <= 0:
            raise ValueError("contract.latent_temporal_frames must be positive.")
        if not 0 < self.min_prefix_steps <= self.max_prefix_steps < self.action_horizon:
            raise ValueError("Invalid trained async prefix range.")
        if not 0 < self.replan_steps <= self.action_horizon:
            raise ValueError("contract.replan_steps must be in [1, action_horizon].")
        if self.max_prefix_steps + self.replan_steps > self.action_horizon:
            raise ValueError(
                "The largest prefix plus executable suffix exceeds action_horizon: "
                f"{self.max_prefix_steps}+{self.replan_steps}>{self.action_horizon}."
            )
        if self.max_suffix_steps is not None and self.max_suffix_steps != self.replan_steps:
            raise ValueError(
                "contract.max_suffix_steps must be null or equal replan_steps; "
                f"got {self.max_suffix_steps} versus {self.replan_steps}."
            )
        if not 0.0 <= self.async_prob <= 1.0:
            raise ValueError("contract.async_prob must be in [0,1].")
        if self.local_window <= 0:
            raise ValueError("contract.local_window must be positive.")


def validate_deploy_settings(deploy_cfg: Any, contract: AsyncPrefixContract) -> None:
    mismatches: list[str] = []
    _check_equal(mismatches, "deploy.action_horizon", int(deploy_cfg.action_horizon), contract.action_horizon)
    _check_equal(mismatches, "deploy.replan_steps", int(deploy_cfg.replan_steps), contract.replan_steps)
    expected_prefix_steps = contract.max_prefix_steps if contract.async_enabled else 0
    expected_min_prefix_steps = contract.min_prefix_steps if contract.async_enabled else 0
    _check_equal(mismatches, "deploy.prefix_steps", int(deploy_cfg.get("prefix_steps", 12)), expected_prefix_steps)
    _check_equal(
        mismatches,
        "deploy.min_prefix_steps",
        int(deploy_cfg.get("min_prefix_steps", 6)),
        expected_min_prefix_steps,
    )
    _check_equal(
        mismatches,
        "deploy.control_hz",
        float(deploy_cfg.get("control_hz", EXPECTED_CONTROL_HZ)),
        EXPECTED_CONTROL_HZ,
    )
    if mismatches:
        raise ValueError("Deployment settings disagree with its contract:\n  - " + "\n  - ".join(mismatches))


def validate_checkpoint_contract(
    deploy_cfg: Any,
    runtime_cfg: Any,
    checkpoint: Path,
    contract: AsyncPrefixContract,
) -> Path:
    """Fail closed when checkpoint, runtime code, or data contract diverge."""
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    min_bytes = int(deploy_cfg.get("checkpoint_min_bytes", 10_000_000_000))
    if checkpoint.stat().st_size < min_bytes:
        raise ValueError(
            f"Checkpoint is too small or incomplete: {checkpoint.stat().st_size} bytes < {min_bytes}."
        )

    train_config_path = checkpoint_train_config(checkpoint)
    train_cfg = OmegaConf.load(train_config_path)
    source_index = int(deploy_cfg.get("paired_source_index", 0))
    required_branches = "manip" if str(deploy_cfg.get("inference_mode", "paired")).strip().lower() == "manip_only" else "both"
    runtime_data = paired_data_cfg(runtime_cfg, source_index, required_branches)
    train_data = paired_data_cfg(train_cfg, source_index, required_branches)
    mismatches: list[str] = []

    required_model = {
        "model._target_": EXPECTED_MODEL_TARGET,
        "model.proprio_dim": EXPECTED_STATE_DIM,
        "model.action_dit_config.manip_action_dim": EXPECTED_MANIP_ACTION_DIM,
        "model.action_dit_config.nav_action_dim": EXPECTED_NAV_ACTION_DIM,
        "model.action_dit_config.manip_horizon": contract.action_horizon,
        "model.action_dit_config.nav_horizon": contract.action_horizon,
        "model.action_dit_config.type_token_count": EXPECTED_TYPE_TOKEN_COUNT,
        "model.view_type_token_count": EXPECTED_TYPE_TOKEN_COUNT,
        "model.async_prefix_training.enabled": contract.async_enabled,
        "model.async_prefix_training.prob": contract.async_prob,
        "model.async_prefix_training.min_length": contract.min_prefix_steps,
        "model.async_prefix_training.max_length": contract.max_prefix_steps,
        "model.async_prefix_training.lambda_attention": contract.lambda_attention,
        "model.async_prefix_training.local_window": contract.local_window,
        "model.async_prefix_training.random_prefix_attention_mask": contract.random_prefix_attention_mask,
        "model.async_prefix_training.prefix_mask_prob": contract.prefix_mask_prob,
        "model.async_prefix_training.keep_last_k": contract.keep_last_k,
        "model.async_prefix_training.rope_offset": contract.rope_offset,
        "model.async_prefix_training.dynamic_loss_weighting": contract.dynamic_loss_weighting,
        "model.async_prefix_training.max_suffix_steps": contract.max_suffix_steps,
        "model.loss.lambda_manip_aux_action": contract.manip_aux_loss_weight,
        "model.shared_future_state_training.enabled": False,
    }
    for key, expected in required_model.items():
        train_value = resolved_value(train_cfg, key)
        runtime_value = resolved_value(runtime_cfg, key)
        _check_equal(mismatches, f"checkpoint.{key}", train_value, expected)
        _check_equal(mismatches, f"runtime.{key}", runtime_value, expected)
        _check_equal(mismatches, f"runtime-vs-checkpoint.{key}", runtime_value, train_value)

    required_data = {
        "video_size": EXPECTED_VIDEO_SIZE,
        "nav_video_size": EXPECTED_VIDEO_SIZE,
        "manip_video_size": EXPECTED_VIDEO_SIZE,
        # This checkpoint is trained on the five-source manipulation recipe;
        # navigation remains the robot-only 3D branch.
        "manip_action_dim": EXPECTED_MANIP_ACTION_DIM,
        "manip_action_offset": 1,
        "manip_relative_frame": "fixed_reference",
        "proprio_dim": EXPECTED_STATE_DIM,
        "num_frames": contract.num_frames,
        "action_horizon": contract.action_horizon,
        "precomputed_latent_temporal_frames": contract.latent_temporal_frames,
        "drop_base_state": False,
        "proprio_context_steps": 1,
        "nav_action_stride": 1,
        "action_video_freq_ratio": 4,
        "norm_default_mode": "q01/q99",
        "use_stepwise_action_norm": False,
        "context_len": 128,
        "state_column": "observation.state.camera_dual_arm",
        "manip_action_column": "action.manip.camera_dual_arm",
        "action_alignment": "current",
        "manip_action_source": "recorded_action",
        "robot_manip_action_dim": EXPECTED_ROBOT_MANIP_ACTION_DIM,
    }
    if required_branches != "manip":
        required_data.update({
            "nav_image_mode": EXPECTED_NAV_IMAGE_MODE,
            "concat_multi_camera": EXPECTED_CONCAT_MODE,
            "nav_action_dim": EXPECTED_ROBOT_NAV_ACTION_DIM,
            "robot_nav_action_dim": EXPECTED_ROBOT_NAV_ACTION_DIM,
        })
    else:
        # The source's camera contract, not its position in a six-source list,
        # determines the inference mosaic for customer single-source SFTs.
        train_concat = str(train_data.get("concat_multi_camera", ""))
        if train_concat not in {"fastwam_384x320", EXPECTED_CONCAT_MODE}:
            mismatches.append(f"checkpoint.data.concat_multi_camera: unsupported {train_concat!r}")
        required_data["concat_multi_camera"] = train_concat
    for key, expected in required_data.items():
        train_value = resolved_value(train_data, key)
        runtime_value = resolved_value(runtime_data, key)
        # Dataset constructor defaults are part of the immutable contract;
        # historical configs omit these fields while retaining those defaults.
        if key == "nav_action_stride":
            train_value = 1 if train_value is None else train_value
            runtime_value = 1 if runtime_value is None else runtime_value
        elif key == "robot_nav_action_dim":
            train_value = 3 if train_value is None else train_value
            runtime_value = 3 if runtime_value is None else runtime_value
        elif key == "manip_action_offset":
            train_value = 1 if train_value is None else train_value
            runtime_value = 1 if runtime_value is None else runtime_value
        elif key == "manip_relative_frame":
            train_value = "start_eef" if train_value is None else train_value
            runtime_value = "start_eef" if runtime_value is None else runtime_value
        elif key == "norm_default_mode":
            train_value = "q01/q99" if train_value is None else train_value
            runtime_value = "q01/q99" if runtime_value is None else runtime_value
        elif key == "use_stepwise_action_norm":
            train_value = False if train_value is None else train_value
            runtime_value = False if runtime_value is None else runtime_value
        _check_equal(mismatches, f"checkpoint.data.{key}", train_value, expected)
        _check_equal(mismatches, f"runtime.data.{key}", runtime_value, expected)
        _check_equal(mismatches, f"runtime-vs-checkpoint.data.{key}", runtime_value, train_value)

    train_prefix = str(train_data.get("instruction_prefix", "") or "").strip()
    if not train_prefix:
        train_prefix = str((train_data.get("instruction_prefix_by_branch") or {}).get("manip", "") or "").strip()
    runtime_prefix = str(runtime_data.get("instruction_prefix", "") or "").strip()
    if not runtime_prefix:
        runtime_prefix = str((runtime_data.get("instruction_prefix_by_branch") or {}).get("manip", "") or "").strip()
    deploy_prefix = str(deploy_cfg.get("instruction_prefix_by_mode", {}).get("manip_only", "") or "").strip()
    _check_equal(mismatches, "runtime-vs-checkpoint.manip_instruction_prefix", runtime_prefix, train_prefix)
    _check_equal(mismatches, "deployment-vs-checkpoint.manip_instruction_prefix", deploy_prefix, train_prefix)
    if contract.manip_aux_loss_weight > 0 and runtime_data.get("manip_eef_xy_index_root", None) in (None, "", "null"):
        mismatches.append("runtime.data.manip_eef_xy_index_root: required for manip26")

    expected_stats = Path(str(train_data.pretrained_norm_stats)).resolve()
    runtime_stats = Path(str(runtime_data.pretrained_norm_stats)).resolve()
    deploy_stats = Path(str(deploy_cfg.dataset_stats)).resolve()
    _check_equal(mismatches, "runtime-vs-deployment.dataset_stats", runtime_stats, deploy_stats)
    expected_contract = _expected_stats_contract(
        expected_stats, Path(str(deploy_cfg.project_root)).resolve()
    )
    for label, path in (("runtime.data.pretrained_norm_stats", runtime_stats), ("dataset_stats", deploy_stats)):
        if not path.is_file():
            mismatches.append(f"{label}: missing {path}")
        else:
            _check_equal(mismatches, f"{label}.contract", _stats_contract(path), expected_contract)
    if mismatches:
        raise ValueError("Checkpoint/deployment contract mismatch:\n  - " + "\n  - ".join(mismatches))
    return train_config_path


class LiveUniWAMAsyncPrefix12Rot6DPolicy:
    """2B policy with state-relative arm pose labels and local SE(2) paths."""

    def __init__(self, deploy_cfg: Any) -> None:
        self.deploy_cfg = deploy_cfg
        self.contract = AsyncPrefixContract.from_deploy_cfg(deploy_cfg)
        validate_deploy_settings(deploy_cfg, self.contract)
        self.project_root = Path(str(deploy_cfg.project_root)).resolve()
        for path in (self.project_root, self.project_root / "src"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

        self.compile_action_infer = bool(deploy_cfg.get("compile_action_infer", False))
        self.compile_vae_infer = bool(deploy_cfg.get("compile_vae_infer", False))
        self.debug_action_space = bool(deploy_cfg.get("debug_action_space", False))
        self.cfg = load_runtime_cfg(
            self.project_root,
            str(deploy_cfg.task),
            compile_action_infer=self.compile_action_infer,
            compile_vae_infer=self.compile_vae_infer,
        )
        required_branches = "manip" if str(deploy_cfg.get("inference_mode", "paired")).strip().lower() == "manip_only" else "both"
        self.data_cfg = paired_data_cfg(
            self.cfg,
            int(deploy_cfg.get("paired_source_index", 0)),
            required_branches,
        )
        self.checkpoint = Path(str(deploy_cfg.checkpoint)).resolve()
        train_config = validate_checkpoint_contract(
            deploy_cfg, self.cfg, self.checkpoint, self.contract
        )

        dtype = mixed_precision_to_dtype(str(deploy_cfg.mixed_precision))
        model_cfg = OmegaConf.create(OmegaConf.to_container(self.cfg.model, resolve=True))
        self.model = instantiate(model_cfg, model_dtype=dtype, device=str(deploy_cfg.device))
        self.model.load_checkpoint(str(self.checkpoint), strict=True)
        self.model = self.model.to(str(deploy_cfg.device)).eval()

        from uniwam.datasets.lerobot.utils.normalizer import (
            LinearNormalizer,
            load_dataset_stats_from_json,
        )

        stats = load_dataset_stats_from_json(str(deploy_cfg.dataset_stats))
        normalizer_shape_meta = OmegaConf.to_container(self.data_cfg.shape_meta, resolve=True)
        # The ordered-color source is manipulation-only and therefore omits a
        # dataset nav field, while the shared model still exposes the nav head.
        # Keep nav stats available for the inactive zero branch without using
        # the source-1 video/action contract at runtime.
        action_meta = normalizer_shape_meta.setdefault("action", [])
        if not any(str(meta.get("key")) == "nav" for meta in action_meta):
            action_meta.append({"key": "nav", "raw_shape": EXPECTED_ROBOT_NAV_ACTION_DIM, "shape": EXPECTED_ROBOT_NAV_ACTION_DIM})
        self.normalizer = LinearNormalizer(
            shape_meta=normalizer_shape_meta,
            use_stepwise_action_norm=bool(
                self.data_cfg.get("use_stepwise_action_norm", False)
            ),
            default_mode=str(self.data_cfg.get("norm_default_mode", "q01/q99")),
            exception_mode=self.data_cfg.get("norm_exception_mode", None),
            stats=stats,
        )
        self.observation_builder = Rot6DObservationBuilder(
            data_cfg=self.data_cfg,
            normalizer=self.normalizer,
        )

        self.action_horizon = self.contract.action_horizon
        self.prefix_steps = self.contract.max_prefix_steps if self.contract.async_enabled else 0
        self.min_prefix_steps = self.contract.min_prefix_steps if self.contract.async_enabled else 0
        self.replan_steps = self.contract.replan_steps
        self.control_hz = float(deploy_cfg.get("control_hz", EXPECTED_CONTROL_HZ))
        self.num_inference_steps = int(deploy_cfg.num_inference_steps)
        sigma = deploy_cfg.get("sigma_shift", None)
        self.sigma_shift = None if sigma in (None, "null") else float(sigma)
        seed = deploy_cfg.get("seed", None)
        self.seed = None if seed in (None, "null") else int(seed)
        self.rand_device = str(deploy_cfg.get("rand_device", "cpu"))
        self.default_inference_mode = str(deploy_cfg.get("inference_mode", "paired")).strip().lower()
        if self.default_inference_mode not in INFERENCE_MODES:
            raise ValueError(
                "inference_mode must be one of "
                f"{sorted(INFERENCE_MODES)}, got {self.default_inference_mode!r}."
            )
        self.nav_planner_config = NonholonomicPlannerConfig.from_mapping(
            deploy_cfg.get("nav_planner", None)
        )
        self.nav_planner_config.validate()
        self.step_count = 0
        self.stats = {"num_chunks": 0, "model_infer_s": 0.0}
        self.last_model_profile_ms: dict[str, float] = {}
        print(
            "[uniwam-piper-cloud][ASYNC_ROT6D_SE2_CONTRACT_OK] "
            f"profile={self.contract.name} checkpoint={self.checkpoint} "
            f"train_config={train_config} state_dim={EXPECTED_STATE_DIM} "
            f"manip_dim={EXPECTED_MANIP_ACTION_DIM} nav_dim={EXPECTED_NAV_ACTION_DIM} "
            f"horizon={self.action_horizon} num_frames={self.contract.num_frames} "
            f"vae_temporal_frames={self.contract.latent_temporal_frames} "
            f"prefix=0|{self.min_prefix_steps}..{self.prefix_steps} "
            f"execute_steps={self.replan_steps} lambda_attention={self.contract.lambda_attention} "
            f"local_window={self.contract.local_window} causal_prefix0="
            f"{not self.contract.random_prefix_attention_mask} "
            f"default_inference_mode={self.default_inference_mode} "
            f"compile_action_infer={self.compile_action_infer} "
            f"compile_vae_infer={self.compile_vae_infer} "
            "(edge requests may override this per chunk)",
            flush=True,
        )

    def close(self) -> None:
        return None

    def resolve_inference_mode(self, requested: str | None) -> str:
        """Resolve a request-local branch mode without mutating global policy state."""
        mode = self.default_inference_mode if requested is None else str(requested).strip().lower()
        if mode not in INFERENCE_MODES:
            raise ValueError(
                f"inference_mode must be one of {sorted(INFERENCE_MODES)}, got {mode!r}."
            )
        return mode

    def make_sample(
        self,
        *,
        images: dict[str, bytes],
        image_encoding: str,
        eef_state: np.ndarray,
        instruction: str | None,
        inference_mode: str | None = None,
        robot_type: str = "piper",
    ) -> dict[str, torch.Tensor]:
        task = "" if instruction is None else str(instruction).strip()
        mode = self.resolve_inference_mode(inference_mode)
        robot_prefixes = self.deploy_cfg.get("instruction_prefix_by_robot", {})
        prefixes = self.deploy_cfg.get("instruction_prefix_by_mode", {})
        if str(robot_type).strip().lower() == "franka" and mode != "nav_only":
            prefix = str(robot_prefixes.get("franka", "")).strip()
        else:
            prefix_key = "nav_only" if mode == "nav_only" else "manip_only"
            prefix = str(prefixes.get(prefix_key, "")).strip()
        if prefix and not task.startswith(prefix):
            task = f"{prefix} {task}".strip()
        return self.observation_builder.make_sample(
            images=images,
            image_encoding=image_encoding,
            eef_state=eef_state,
            instruction=task,
        )

    def _denormalize(self, nav: torch.Tensor, manip: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        if nav.ndim != 2 or nav.shape[1] != EXPECTED_NAV_ACTION_DIM:
            raise RuntimeError(f"Normalized navigation action has bad shape {tuple(nav.shape)}.")
        if manip.ndim != 2 or manip.shape[1] != EXPECTED_MANIP_ACTION_DIM:
            raise RuntimeError(f"Normalized manipulation action has bad shape {tuple(manip.shape)}.")
        batch = {
            "action": {
                "nav": nav[:, :EXPECTED_ROBOT_NAV_ACTION_DIM].detach().cpu().float().clone(),
                "manip": manip[:, :EXPECTED_ROBOT_MANIP_ACTION_DIM].detach().cpu().float().clone(),
            },
            "state": {"default": torch.zeros((1, EXPECTED_STATE_DIM), dtype=torch.float32)},
        }
        output = self.normalizer.backward(batch)
        nav_out = np.zeros((nav.shape[0], EXPECTED_NAV_ACTION_DIM), dtype=np.float32)
        nav_out[:, :EXPECTED_ROBOT_NAV_ACTION_DIM] = output["action"]["nav"].numpy().astype(np.float32)
        # Auxiliary navigation labels are already in the model's [-1,1] space.
        nav_out[:, EXPECTED_ROBOT_NAV_ACTION_DIM:] = nav[:, EXPECTED_ROBOT_NAV_ACTION_DIM:].detach().cpu().float().numpy()
        robot_out = output["action"]["manip"].numpy().astype(np.float32)
        aux_out = manip[:, EXPECTED_ROBOT_MANIP_ACTION_DIM:].detach().cpu().float().numpy()
        if not np.isfinite(aux_out).all():
            raise RuntimeError("Model EEF-XY/visibility output contains NaN or Inf.")
        # Auxiliary image coordinates and visibility targets are both trained
        # in the bounded [-1,1] channel. The executable robot20 action is
        # intentionally left untouched.
        aux_out = np.clip(aux_out, -1.0, 1.0)
        manip_out = np.concatenate((robot_out, aux_out), axis=1).astype(np.float32)
        if nav_out.ndim != 2 or nav_out.shape[1] != EXPECTED_NAV_ACTION_DIM:
            raise RuntimeError(f"Denormalized nav action has bad shape {nav_out.shape}.")
        if manip_out.ndim != 2 or manip_out.shape[1] != EXPECTED_MANIP_ACTION_DIM:
            raise RuntimeError(f"Denormalized manipulation action has bad shape {manip_out.shape}.")
        return nav_out, manip_out

    def _normalize_prefix(
        self,
        prefix_eef_actions: np.ndarray,
        prefix_bbox_actions: np.ndarray,
        state_raw: np.ndarray,
        prefix_nav_aux_actions: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert executable EEF14+twist targets into the trained action labels."""
        prefix = np.asarray(prefix_eef_actions, dtype=np.float32)
        state = self.observation_builder.canonical_state(state_raw)
        if prefix.ndim != 2 or prefix.shape[1] != 17:
            raise ValueError(f"Head prefix must be [L,17] EEF14+base3, got {prefix.shape}.")
        if not self.min_prefix_steps <= prefix.shape[0] <= self.prefix_steps:
            raise ValueError(
                "Head prefix length must match the trained range "
                f"[{self.min_prefix_steps}, {self.prefix_steps}], got {prefix.shape[0]}."
            )
        if not np.all(np.isfinite(prefix)):
            raise ValueError("Head prefix contains NaN or Inf.")

        if prefix_nav_aux_actions is None:
            nav_aux = np.zeros((prefix.shape[0], EXPECTED_NAV_AUX_ACTION_DIM), dtype=np.float32)
        else:
            nav_aux = np.asarray(prefix_nav_aux_actions, dtype=np.float32)
            if nav_aux.shape != (prefix.shape[0], EXPECTED_NAV_AUX_ACTION_DIM):
                raise ValueError(
                    "Head navigation auxiliary prefix must be "
                    f"[L,{EXPECTED_NAV_AUX_ACTION_DIM}], got {nav_aux.shape}."
                )
            if not np.all(np.isfinite(nav_aux)):
                raise ValueError("Head navigation auxiliary prefix contains NaN or Inf.")
        manip_robot = np.empty((prefix.shape[0], EXPECTED_ROBOT_MANIP_ACTION_DIM), dtype=np.float32)
        left_xyz, left_rotation = fixed_reference_relative_pose(
            state[3:6],
            state[6:12],
            prefix[:, 0:3],
            matrix_to_rotation_6d(rpy_xyz_to_matrix(prefix[:, 3:6])),
        )
        right_xyz, right_rotation = fixed_reference_relative_pose(
            state[13:16],
            state[16:22],
            prefix[:, 7:10],
            matrix_to_rotation_6d(rpy_xyz_to_matrix(prefix[:, 10:13])),
        )
        manip_robot[:, 0:3] = left_xyz
        manip_robot[:, 3:9] = left_rotation
        manip_robot[:, 9] = prefix[:, 6]
        manip_robot[:, 10:13] = right_xyz
        manip_robot[:, 13:19] = right_rotation
        manip_robot[:, 19] = prefix[:, 13]
        nav = queued_nav_target_path(
            prefix[:, 14:17],
            control_hz=float(getattr(self, "control_hz", EXPECTED_CONTROL_HZ)),
        )
        normalized = self.normalizer.forward(
            {
                "action": {"nav": torch.from_numpy(nav), "manip": torch.from_numpy(manip_robot)},
                "state": {"default": torch.zeros((1, EXPECTED_STATE_DIM), dtype=torch.float32)},
            }
        )["action"]
        nav_normalized = torch.cat(
            (normalized["nav"].float(), torch.from_numpy(nav_aux).float()), dim=1
        ).contiguous()
        if prefix_bbox_actions is None:
            raise ValueError(
                "manip26 requires prefix_bbox_actions [L,6] containing the "
                "previously predicted EEF-XY+visibility prefix."
            )
        manip_eef_xy = np.asarray(prefix_bbox_actions, dtype=np.float32)
        if manip_eef_xy.shape != (prefix.shape[0], EXPECTED_EEF_XY_ACTION_DIM):
            raise ValueError(
                "prefix_bbox_actions must be aligned EEF-XY+visibility [L,6], got "
                f"{manip_eef_xy.shape}."
            )
        if not np.all(np.isfinite(manip_eef_xy)):
            raise ValueError("prefix EEF-XY values contain NaN or Inf.")
        # The previous model output is the next model input. Keep the image
        # auxiliary in its declared range without changing robot20 behavior.
        manip_eef_xy = np.clip(manip_eef_xy, -1.0, 1.0)
        manip_eef_xy = torch.from_numpy(manip_eef_xy)
        manip_normalized = torch.cat(
            (normalized["manip"].float(), manip_eef_xy), dim=1
        ).contiguous()
        if not torch.isfinite(nav_normalized).all() or not torch.isfinite(manip_normalized).all():
            raise ValueError("Normalized head prefix contains NaN or Inf.")
        return nav_normalized, manip_normalized

    @staticmethod
    def canonical_eef14(state_raw: np.ndarray) -> np.ndarray:
        """Convert the exact 23D train-time state to absolute EEF14 RPY targets."""
        state = Rot6DObservationBuilder.canonical_state(state_raw)
        result = np.empty(14, dtype=np.float32)
        result[0:3] = state[3:6]
        result[3:6] = matrix_to_rpy_xyz(rotation_6d_to_matrix(state[6:12]))
        result[6] = state[12]
        result[7:10] = state[13:16]
        result[10:13] = matrix_to_rpy_xyz(rotation_6d_to_matrix(state[16:22]))
        result[13] = state[22]
        return result

    @classmethod
    def hold_eef_target(
        cls,
        state_raw: np.ndarray,
        prefix_eef_actions: np.ndarray | None,
    ) -> np.ndarray:
        """Select the Cartesian target that a navigation-only suffix must hold."""
        if prefix_eef_actions is None or not np.asarray(prefix_eef_actions).size:
            return cls.canonical_eef14(state_raw)
        prefix = np.asarray(prefix_eef_actions, dtype=np.float32)
        if prefix.ndim != 2 or prefix.shape[1] != 17:
            raise ValueError(f"Head prefix must be [L,17] EEF14+base3, got {prefix.shape}.")
        return prefix[-1, :14].copy()

    @staticmethod
    def reconstruct_eef(manip_relative: np.ndarray, state_raw: np.ndarray) -> np.ndarray:
        """Undo the loader's common-start-frame ``T0^-1 @ Ttarget`` transform."""
        manip = np.asarray(manip_relative, dtype=np.float32)
        state = Rot6DObservationBuilder.canonical_state(state_raw)
        if manip.ndim != 2 or manip.shape[1] != EXPECTED_ROBOT_MANIP_ACTION_DIM:
            raise ValueError(f"Expected relative manipulation [T,20], got {manip.shape}.")
        left_xyz, left_matrix = fixed_reference_absolute_pose(
            state[3:6], state[6:12], manip[:, 0:3], manip[:, 3:9]
        )
        right_xyz, right_matrix = fixed_reference_absolute_pose(
            state[13:16], state[16:22], manip[:, 10:13], manip[:, 13:19]
        )
        left_initial_rpy = matrix_to_rpy_xyz(rotation_6d_to_matrix(state[6:12]))
        right_initial_rpy = matrix_to_rpy_xyz(rotation_6d_to_matrix(state[16:22]))
        eef = np.empty((manip.shape[0], 14), dtype=np.float32)
        eef[:, 0:3] = left_xyz
        eef[:, 3:6] = continuous_rpy_xyz_from_matrix_sequence(left_matrix, left_initial_rpy)
        eef[:, 6] = manip[:, 9]
        eef[:, 7:10] = right_xyz
        eef[:, 10:13] = continuous_rpy_xyz_from_matrix_sequence(right_matrix, right_initial_rpy)
        eef[:, 13] = manip[:, 19]
        if not np.all(np.isfinite(eef)):
            raise ValueError("Reconstructed EEF chunk contains NaN or Inf.")
        return eef

    def _suffix_stop(self, prefix_length: int) -> int:
        if not 0 <= prefix_length <= self.prefix_steps:
            raise ValueError(f"Invalid prefix length {prefix_length}.")
        stop = prefix_length + self.replan_steps
        if stop > self.action_horizon:
            raise RuntimeError(
                "Configured suffix exceeds the model horizon: "
                f"prefix={prefix_length}, execute={self.replan_steps}, horizon={self.action_horizon}."
            )
        return stop

    def _infer_chunk(
        self,
        sample: dict[str, torch.Tensor],
        prefix_eef_actions: np.ndarray | None,
        prefix_bbox_actions: np.ndarray | None,
        prefix_nav_aux_actions: np.ndarray | None,
        inference_mode: str | None = None,
    ) -> tuple[dict[str, Any], float, int]:
        mode = self.resolve_inference_mode(inference_mode)
        prefix_length = 0
        nav_prefix = None
        manip_prefix = None
        prefix_base_commands = None
        prefix_eef = None
        if prefix_eef_actions is not None and np.asarray(prefix_eef_actions).size:
            prefix_eef = np.asarray(prefix_eef_actions, dtype=np.float32)
            nav_prefix, manip_prefix = self._normalize_prefix(
                prefix_eef,
                prefix_bbox_actions,
                sample["state_raw"].numpy(),
                prefix_nav_aux_actions,
            )
            prefix_length = int(nav_prefix.shape[0])
            prefix_base_commands = prefix_eef[:, 14:17].copy()

        active_nav = mode in {"paired", "nav_only"}
        active_manip = mode in {"paired", "manip_only"}
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.infer_action(
                action_horizon=self.action_horizon,
                proprio=sample["proprio"],
                context=sample["context"],
                context_mask=sample["context_mask"],
                nav_input_image=sample["nav_image"] if active_nav else None,
                manip_input_image=sample["manip_image"] if active_manip else None,
                num_inference_steps=self.num_inference_steps,
                sigma_shift=self.sigma_shift,
                seed=self.seed,
                rand_device=self.rand_device,
                nav_action_prefix=nav_prefix if active_nav else None,
                manip_action_prefix=manip_prefix if active_manip else None,
                prefix_length=prefix_length,
            )
        if torch.cuda.is_available() and str(self.deploy_cfg.device).startswith("cuda"):
            torch.cuda.synchronize(device=str(self.deploy_cfg.device))
        infer_s = time.perf_counter() - started
        self.last_model_profile_ms = dict(output.get("profile_ms", {}))

        nav_latents = output.get("nav_action")
        manip_latents = output.get("manip_action")
        if active_nav:
            if nav_latents is None or tuple(nav_latents.shape) != (self.action_horizon, EXPECTED_NAV_ACTION_DIM):
                raise RuntimeError(f"Navigation inference returned bad action shape: {None if nav_latents is None else tuple(nav_latents.shape)}")
        elif nav_latents is not None:
            raise RuntimeError("Inactive navigation branch unexpectedly returned an action.")
        if active_manip:
            if manip_latents is None or tuple(manip_latents.shape) != (self.action_horizon, EXPECTED_MANIP_ACTION_DIM):
                raise RuntimeError(f"Manipulation inference returned bad action shape: {None if manip_latents is None else tuple(manip_latents.shape)}")
        elif manip_latents is not None:
            raise RuntimeError("Inactive manipulation branch unexpectedly returned an action.")

        if active_nav and active_manip:
            assert nav_latents is not None and manip_latents is not None
            nav, manip = self._denormalize(nav_latents, manip_latents)
        elif active_nav:
            assert nav_latents is not None
            nav, _ = self._denormalize(
                nav_latents,
                torch.zeros((self.action_horizon, EXPECTED_MANIP_ACTION_DIM)),
            )
            manip = np.zeros((self.action_horizon, EXPECTED_MANIP_ACTION_DIM), dtype=np.float32)
            manip[:, EXPECTED_ROBOT_MANIP_ACTION_DIM:] = -1.0
        else:
            assert manip_latents is not None
            _, manip = self._denormalize(
                torch.zeros((self.action_horizon, EXPECTED_NAV_ACTION_DIM)), manip_latents
            )
            nav = np.zeros((self.action_horizon, EXPECTED_NAV_ACTION_DIM), dtype=np.float32)

        state_raw = sample["state_raw"].numpy()
        if active_manip:
            if self.debug_action_space:
                debug_index = min(int(prefix_length), self.action_horizon - 1)
                normalized_first = manip_latents[0, :EXPECTED_ROBOT_MANIP_ACTION_DIM].detach().cpu().float().numpy()
                normalized_exec = manip_latents[debug_index, :EXPECTED_ROBOT_MANIP_ACTION_DIM].detach().cpu().float().numpy()
                relative_first = manip[0, :EXPECTED_ROBOT_MANIP_ACTION_DIM].copy()
                relative_exec = manip[debug_index, :EXPECTED_ROBOT_MANIP_ACTION_DIM].copy()
                state_eef = self.canonical_eef14(state_raw)
                print(
                    "[uniwam-piper-cloud][ACTION_SPACE_DEBUG] "
                    f"prefix={prefix_length} normalized0={np.array2string(normalized_first, precision=5, separator=',')} "
                    f"relative0={np.array2string(relative_first, precision=5, separator=',')} "
                    f"normalized_exec={np.array2string(normalized_exec, precision=5, separator=',')} "
                    f"relative_exec={np.array2string(relative_exec, precision=5, separator=',')} "
                    f"state_camera_eef14={np.array2string(state_eef, precision=5, separator=',')}",
                    flush=True,
                )
            eef = self.reconstruct_eef(
                manip[:, :EXPECTED_ROBOT_MANIP_ACTION_DIM], state_raw
            )
            if self.debug_action_space:
                print(
                    "[uniwam-piper-cloud][ACTION_SPACE_DEBUG_EEF] "
                    f"eef0={np.array2string(eef[0], precision=5, separator=',')} "
                    f"delta0={np.array2string(eef[0] - state_eef, precision=5, separator=',')} "
                    f"eef_exec={np.array2string(eef[debug_index], precision=5, separator=',')} "
                    f"delta_exec={np.array2string(eef[debug_index] - state_eef, precision=5, separator=',')}",
                    flush=True,
                )
        else:
            # A suffix begins after the queued head.  Holding the head's final
            # Cartesian target keeps EEF records aligned with held joint targets.
            hold_eef = self.hold_eef_target(state_raw, prefix_eef)
            eef = np.repeat(hold_eef.reshape(1, 14), self.action_horizon, axis=0).astype(np.float32)

        suffix_stop = self._suffix_stop(prefix_length)
        suffix_nav_targets = nav[prefix_length:suffix_stop].copy()
        suffix_nav_aux = suffix_nav_targets[:, EXPECTED_ROBOT_NAV_ACTION_DIM:].copy()
        suffix_eef = eef[prefix_length:suffix_stop].copy()
        suffix_manip = manip[prefix_length:suffix_stop].copy()
        suffix_bbox = suffix_manip[:, EXPECTED_ROBOT_MANIP_ACTION_DIM:].copy()
        suffix_steps = suffix_stop - prefix_length
        if suffix_steps != self.replan_steps:
            raise RuntimeError(f"Bad suffix length {suffix_steps}; expected {self.replan_steps}.")

        if active_nav:
            if prefix_length:
                assert prefix_base_commands is not None
                prefix_path = queued_nav_target_path(prefix_base_commands, control_hz=self.control_hz)
                planner_start_pose = prefix_path[-1]
                planner_seed = prefix_base_commands[-1]
            else:
                planner_start_pose = np.zeros(3, dtype=np.float32)
                planner_seed = np.asarray(state_raw[:3], dtype=np.float32)
            plan = plan_nonholonomic_commands(
                planner_start_pose,
                suffix_nav_targets[:, :EXPECTED_ROBOT_NAV_ACTION_DIM],
                control_hz=self.control_hz,
                initial_command=planner_seed,
                config=self.nav_planner_config,
            )
            suffix_base = plan.commands
            planned_path = plan.planned_path
            nav_pose_error = plan.pose_error
        else:
            suffix_base = np.zeros((suffix_steps, 3), dtype=np.float32)
            planned_path = suffix_nav_targets.copy()
            nav_pose_error = np.zeros_like(suffix_nav_targets)

        model_actions = np.concatenate((suffix_eef, suffix_base), axis=1).astype(np.float32)
        if model_actions.shape != (suffix_steps, 17):
            raise RuntimeError(f"Model action contract violation: {model_actions.shape}.")
        return {
            "eef_actions": suffix_eef,
            "model_actions": model_actions,
            "base_actions": suffix_base,
            "nav_actions": suffix_nav_targets,
            "nav_aux_actions": suffix_nav_aux,
            "nav_planned_path": planned_path,
            "nav_pose_error": nav_pose_error,
            "manip_actions": suffix_manip,
            "bbox_actions": suffix_bbox,
            "arm_hold": not active_manip,
            "inference_mode": mode,
            "model_horizon": self.action_horizon,
            "execute_horizon": suffix_steps,
        }, infer_s, prefix_length

    def request_suffix(
        self,
        sample: dict[str, torch.Tensor],
        *,
        prefix_eef_actions: np.ndarray | None,
        prefix_bbox_actions: np.ndarray | None = None,
        prefix_nav_aux_actions: np.ndarray | None = None,
        inference_mode: str | None = None,
    ) -> dict[str, Any]:
        output, infer_s, prefix_length = self._infer_chunk(
            sample,
            prefix_eef_actions,
            prefix_bbox_actions,
            prefix_nav_aux_actions,
            inference_mode=inference_mode,
        )
        steps = int(np.asarray(output["eef_actions"]).shape[0])
        self.step_count += steps
        self.stats["num_chunks"] += 1
        self.stats["model_infer_s"] += float(infer_s)
        output.update(
            {
                "prefix_length": prefix_length,
                "step_count": self.step_count,
                "request_model_infer_s": float(infer_s),
                "model_profile_ms": dict(self.last_model_profile_ms),
                "stats": dict(self.stats),
            }
        )
        return output
