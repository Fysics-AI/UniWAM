import logging
import json
import inspect
import os
import re
from math import ceil
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset

from .datasets.mixed_stream import MixedStreamCollator
from .utils.fs import ensure_dir
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)


class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.prefetch_factor = cfg.get("prefetch_factor", None)
        if self.prefetch_factor is not None:
            self.prefetch_factor = int(self.prefetch_factor)
        self.persistent_workers = bool(cfg.get("persistent_workers", self.num_workers > 0))
        self.dataloader_timeout_s = float(cfg.get("dataloader_timeout_s", 0))
        if self.dataloader_timeout_s < 0:
            raise ValueError("dataloader_timeout_s must be non-negative.")
        self.sampler_mode = str(cfg.get("sampler_mode", "random"))
        self.sampler_local_window = int(cfg.get("sampler_local_window", 0))
        self.sampler_balance_groups = list(
            cfg.get("sampler_balance_groups", ["pure_nav", "transition", "pure_manip"])
        )
        self.sampler_min_frame_gap = int(cfg.get("sampler_min_frame_gap", 0))
        self.collate_mode = str(cfg.get("collate_mode", "default")).strip().lower()
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.save_training_state = bool(cfg.get("save_training_state", True))
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        
        self.resume = cfg.resume
        # When a ZeRO state cannot be repartitioned across a changed world
        # size, a weights-only resume can still preserve the visible step
        # counter for checkpoint cadence and reporting.
        self.resume_step = int(cfg.get("resume_step", 0) or 0)
        self._weights_preloaded = False
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)
        self.finite_guard = os.environ.get("FASTWAM_FINITE_GUARD", "0") == "1"
        self.finite_guard_steps = int(os.environ.get("FASTWAM_FINITE_GUARD_STEPS", "0"))
        self._finite_guard_base_step = 0
        self._finite_guard_context = None
        self._deepspeed_optimizer_guard_installed = False
        if self.finite_guard_steps < 0:
            raise ValueError("FASTWAM_FINITE_GUARD_STEPS must be non-negative.")

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )
        
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            self.accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown"),
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        logger.info(
            "finite loss guard enabled=true full-gradient guard enabled=%s steps=%d",
            self.finite_guard,
            self.finite_guard_steps,
        )
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps DiT (+ optional proprio encoder) as trainable when ZeRO builds optimizer state.
        self._apply_dit_only_train_mode(self.model)
        trainable_params = list(self.model.dit.parameters())
        proprio_encoder = getattr(self.model, "proprio_encoder", None)
        if proprio_encoder is not None:
            trainable_params.extend(list(proprio_encoder.parameters()))
        optimizer_param_ids = [id(param) for param in trainable_params]
        if len(optimizer_param_ids) != len(set(optimizer_param_ids)):
            raise RuntimeError("Optimizer parameter list contains duplicate parameters.")
        trainable_named = {
            id(param): name
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        optimizer_param_id_set = set(optimizer_param_ids)
        missing = [
            name for param_id, name in trainable_named.items()
            if param_id not in optimizer_param_id_set
        ]
        unexpected = [
            param_id for param_id in optimizer_param_id_set
            if param_id not in trainable_named
        ]
        if missing or unexpected:
            raise RuntimeError(
                "Optimizer/trainable parameter mismatch: "
                f"missing={missing[:20]}, unexpected_count={len(unexpected)}."
            )
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        scheduler_total_steps = total_train_steps
        if self.resume and not Path(str(self.resume)).is_dir() and self.resume_step > 0:
            scheduler_total_steps = max(total_train_steps - self.resume_step, 1)
        warmup_steps = int(scheduler_total_steps * 0.05)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=scheduler_total_steps,
            warmup_steps=warmup_steps,
        )
        logger.info(
            "Training/scheduler steps: loop_stop=%d scheduler_new_steps=%d warmup_steps=%d",
            total_train_steps,
            scheduler_total_steps,
            warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        if bool(getattr(self.model, "compile_training_denoise", False)):
            logger.info(
                "Compiling mixed-stream training denoise forward/backward before DeepSpeed initialization."
            )
            compile_start = time.perf_counter()
            # Compile against the same rank-local branch shape that the first
            # prepared batch will use.  A common unsharded warmup batch can
            # compile a different mixed-stream graph on every rank at step 1.
            sampler_iter = iter(self.train_sampler)
            global_warmup_indices = [
                next(sampler_iter)
                for _ in range(self.batch_size * self.accelerator.num_processes)
            ]
            rank_start = self.accelerator.process_index * self.batch_size
            rank_indices = global_warmup_indices[rank_start : rank_start + self.batch_size]
            warmup_loader = DataLoader(
                Subset(self.train_dataset, rank_indices),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                collate_fn=self.train_loader.collate_fn,
            )
            warmup_sample = next(iter(warmup_loader))
            with self.accelerator.autocast():
                warmup_loss, _ = self.model.training_loss(warmup_sample)
            warmup_loss.backward()
            self.optimizer.zero_grad(set_to_none=True)
            self.model.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.accelerator.device)
            logger.info(
                "Finished mixed-stream denoise compile warmup in %.2f seconds.",
                time.perf_counter() - compile_start,
            )
            del warmup_sample, warmup_loss, warmup_loader
            set_global_seed(self.seed)
            self.accelerator.wait_for_everyone()

        # A weights-only checkpoint must be loaded before DeepSpeed snapshots
        # FP32 optimizer-master parameters from the model. Loading it afterward
        # leaves the master partition stale and the first optimizer step can
        # overwrite (or corrupt) the inherited BF16 model weights.
        self._preload_weight_checkpoint_if_needed()

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self._install_deepspeed_optimizer_finite_guard()
        self._assert_parameter_fingerprint_synced()
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()
        # Interpret finite_guard_steps relative to this launch, including a
        # weights-only or full-state resume at a nonzero visible step.
        self._finite_guard_base_step = self.global_step

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
            config=OmegaConf.to_container(self.cfg, resolve=True),
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    @staticmethod
    def _jsonable_batch_value(value):
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            if tensor.numel() <= 512:
                return tensor.tolist()
            finite = torch.isfinite(tensor) if tensor.is_floating_point() else None
            result = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            if finite is not None:
                result["finite"] = bool(finite.all())
                if bool(finite.any()):
                    valid = tensor[finite].float()
                    result["min"] = float(valid.min())
                    result["max"] = float(valid.max())
            return result
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, (list, tuple)):
            return [Wan22Trainer._jsonable_batch_value(item) for item in value]
        if isinstance(value, dict):
            return {
                str(key): Wan22Trainer._jsonable_batch_value(item)
                for key, item in value.items()
            }
        return repr(value)

    def _dump_nonfinite_diagnostic(
        self,
        *,
        stage: str,
        sample: dict,
        loss=None,
        loss_dict=None,
        bad_gradients=None,
        bad_parameters=None,
    ) -> Path:
        rank = int(self.accelerator.process_index)
        diagnostic_dir = Path(self.output_dir) / "diagnostics"
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
        path = diagnostic_dir / f"nonfinite_step_{self.global_step + 1:06d}_rank_{rank:02d}_{stage}.json"
        metadata_keys = (
            "episode_index",
            "frame_index",
            "window_index",
            "source_window_index",
            "sample_type",
            "prompt",
            "manip_branch_valid",
            "nav_branch_valid",
            "manip_owner_indices",
            "nav_owner_indices",
            "manip_loss_valid",
            "nav_loss_valid",
            "manip_action",
            "nav_action",
            "proprio",
        )
        payload = {
            "stage": stage,
            "step_about_to_run": self.global_step + 1,
            "rank": rank,
            "loss": self._jsonable_batch_value(loss),
            "loss_dict": self._jsonable_batch_value(loss_dict),
            "bad_gradients": self._jsonable_batch_value(bad_gradients or []),
            "bad_parameters": self._jsonable_batch_value(bad_parameters or []),
            "batch": {
                key: self._jsonable_batch_value(sample[key])
                for key in metadata_keys
                if key in sample
            },
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
        return path

    def _all_ranks_true(self, local_value: bool, *, device: torch.device) -> bool:
        flag = torch.tensor(int(local_value), dtype=torch.int32, device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    def _assert_parameter_fingerprint_synced(self) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            return
        model = self.accelerator.unwrap_model(self.model)
        selected = []
        markers = (
            "manip_action_encoder",
            "nav_action_encoder",
            "manip_head",
            "nav_head",
            "manip_type_token",
            "nav_type_token",
            "view_type_tokens",
            "proprio_encoder",
        )
        for name, parameter in model.named_parameters():
            if not any(marker in name for marker in markers):
                continue
            flat = parameter.detach().reshape(-1)
            count = min(64, flat.numel())
            indices = torch.linspace(
                0, flat.numel() - 1, count, device=flat.device
            ).round().long()
            selected.append(flat.index_select(0, indices).float())
        if not selected:
            raise RuntimeError("No conditional parameters found for distributed sync audit.")
        fingerprint = torch.cat(selected)
        gathered = [torch.empty_like(fingerprint) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, fingerprint)
        reference = gathered[0]
        max_diff = max(float((value - reference).abs().max().item()) for value in gathered)
        if max_diff != 0.0:
            raise RuntimeError(
                "Conditional model parameters differ across distributed ranks after prepare: "
                f"max_abs_diff={max_diff:.9g}."
            )
        if self.accelerator.is_main_process:
            logger.info(
                "Distributed conditional-parameter sync audit passed: values=%d max_abs_diff=0",
                fingerprint.numel(),
            )

    def _gradient_guard_active(self) -> bool:
        if not self.finite_guard:
            return False
        return self.finite_guard_steps == 0 or self.global_step < (
            self._finite_guard_base_step + self.finite_guard_steps
        )

    def _guard_diagnostic_context(self) -> tuple[dict, object, dict]:
        if self._finite_guard_context is None:
            return {}, None, {}
        return self._finite_guard_context

    def _raise_on_nonfinite_deepspeed_tensors(
        self,
        *,
        stage: str,
        tensors: list[tuple[str, torch.Tensor]],
    ) -> None:
        bad = []
        device = self.accelerator.device
        for name, tensor in tensors:
            if tensor is None or not tensor.is_floating_point():
                continue
            finite = torch.isfinite(tensor)
            if bool(finite.all()):
                continue
            entry = {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "nonfinite_count": int((~finite).sum().item()),
            }
            if bool(finite.any()):
                values = tensor.detach()[finite].float()
                entry["finite_abs_max"] = float(values.abs().max().item())
            bad.append(entry)
            if len(bad) >= 32:
                break
        if self._all_ranks_true(not bad, device=device):
            return
        sample, loss, loss_dict = self._guard_diagnostic_context()
        path = self._dump_nonfinite_diagnostic(
            stage=stage,
            sample=sample,
            loss=loss,
            loss_dict=loss_dict,
            bad_gradients=bad if "gradient" in stage else None,
            bad_parameters=bad if "gradient" not in stage else None,
        )
        raise FloatingPointError(
            f"Non-finite DeepSpeed tensor detected at {stage}; diagnostic={path}"
        )

    def _install_deepspeed_optimizer_finite_guard(self) -> None:
        if not self.finite_guard or str(self.accelerator.distributed_type) != "DistributedType.DEEPSPEED":
            return
        ds_optimizer = getattr(self.model, "optimizer", None)
        if ds_optimizer is None:
            raise RuntimeError("DeepSpeed finite guard requires model.optimizer.")
        averaged_gradients = getattr(ds_optimizer, "averaged_gradients", None)
        params_in_partition = getattr(ds_optimizer, "params_in_partition", None)
        fp32_groups = getattr(ds_optimizer, "single_partition_of_fp32_groups", None)
        original_step = getattr(ds_optimizer, "step", None)
        original_group_step = getattr(ds_optimizer, "_optimizer_step", None)
        groups_padding = getattr(ds_optimizer, "groups_padding", None)
        if (
            averaged_gradients is None
            or params_in_partition is None
            or fp32_groups is None
            or groups_padding is None
            or not callable(original_step)
            or not callable(original_group_step)
        ):
            raise RuntimeError(
                "Unsupported DeepSpeed optimizer internals for finite diagnostics."
            )

        name_by_id = {
            id(parameter): name for name, parameter in self.model.named_parameters()
        }

        def named_partition_slices(group_no, flat_tensor, suffix):
            slices = []
            flat_offset = 0
            first_param_offset = int(ds_optimizer.first_offset[group_no])
            for index, parameter in enumerate(params_in_partition[group_no]):
                parameter_offset = first_param_offset if index == 0 else 0
                available = int(parameter.numel()) - parameter_offset
                take = min(available, int(flat_tensor.numel()) - flat_offset)
                if take <= 0:
                    break
                name = name_by_id.get(id(parameter), f"group_{group_no}.parameter_{index}")
                slices.append(
                    (
                        f"{name}.{suffix}",
                        flat_tensor.narrow(0, flat_offset, take),
                    )
                )
                flat_offset += take
            return slices

        def guarded_step(closure=None):
            if self._gradient_guard_active():
                tensors = []
                for group_no, gradients in averaged_gradients.items():
                    if gradients is None:
                        continue
                    parameters = params_in_partition[group_no]
                    for index, gradient in enumerate(gradients):
                        parameter = parameters[index] if index < len(parameters) else None
                        parameter_name = name_by_id.get(
                            id(parameter), f"group_{group_no}.gradient_{index}"
                        )
                        tensors.append((parameter_name, gradient))
                self._raise_on_nonfinite_deepspeed_tensors(
                    stage="deepspeed_reduced_gradient", tensors=tensors
                )
            return original_step(closure)

        def guarded_group_step(group_no):
            padding = int(groups_padding[group_no])
            parameter = fp32_groups[group_no]
            valid_parameter = parameter[:-padding] if padding else parameter
            if self._gradient_guard_active():
                gradient = parameter.grad
                valid_gradient = gradient[:-padding] if padding else gradient
                self._raise_on_nonfinite_deepspeed_tensors(
                    stage="deepspeed_clipped_gradient",
                    tensors=[(f"fp32_partition_{group_no}.grad", valid_gradient)],
                )
                self._raise_on_nonfinite_deepspeed_tensors(
                    stage="deepspeed_pre_optimizer_parameter",
                    tensors=named_partition_slices(
                        group_no, valid_parameter, "fp32_master_before_step"
                    ),
                )
            result = original_group_step(group_no)
            if self._gradient_guard_active():
                tensors = named_partition_slices(
                    group_no, valid_parameter, "fp32_master_after_step"
                )
                state = ds_optimizer.optimizer.state.get(parameter, {})
                for key, value in state.items():
                    if not isinstance(value, torch.Tensor):
                        continue
                    valid_value = (
                        value[:-padding]
                        if padding and value.numel() == parameter.numel()
                        else value
                    )
                    tensors.append(
                        (f"fp32_partition_{group_no}.optimizer_state.{key}", valid_value)
                    )
                self._raise_on_nonfinite_deepspeed_tensors(
                    stage="deepspeed_optimizer", tensors=tensors
                )
            return result

        ds_optimizer.step = guarded_step
        ds_optimizer._optimizer_step = guarded_group_step
        self._deepspeed_optimizer_guard_installed = True
        if self.accelerator.is_main_process:
            logger.info(
                "Installed DeepSpeed finite guards at reduced-gradient, clipped-gradient, "
                "and FP32 optimizer-update stages."
            )

    def _assert_finite_loss(self, loss: torch.Tensor, loss_dict: dict, sample: dict) -> None:
        local_finite = bool(torch.isfinite(loss.detach()).all()) and all(
            np.isfinite(float(value)) for value in loss_dict.values()
        )
        if self._all_ranks_true(local_finite, device=loss.device):
            return
        path = self._dump_nonfinite_diagnostic(
            stage="forward", sample=sample, loss=loss, loss_dict=loss_dict
        )
        raise FloatingPointError(
            f"Non-finite forward loss detected globally before backward; diagnostic={path}"
        )

    def _assert_finite_gradients(self, loss: torch.Tensor, loss_dict: dict, sample: dict) -> None:
        bad = []
        for name, parameter in self.model.named_parameters():
            gradient = parameter.grad
            if gradient is None or bool(torch.isfinite(gradient).all()):
                continue
            finite = torch.isfinite(gradient)
            entry = {
                "name": name,
                "shape": list(gradient.shape),
                "nonfinite_count": int((~finite).sum().item()),
            }
            if bool(finite.any()):
                values = gradient[finite].float()
                entry["finite_abs_max"] = float(values.abs().max().item())
            bad.append(entry)
            if len(bad) >= 32:
                break
        if self._all_ranks_true(not bad, device=loss.device):
            return
        path = self._dump_nonfinite_diagnostic(
            stage="backward",
            sample=sample,
            loss=loss,
            loss_dict=loss_dict,
            bad_gradients=bad,
        )
        raise FloatingPointError(
            f"Non-finite gradient detected globally before optimizer.step; diagnostic={path}"
        )

    def _assert_finite_parameters(self, loss: torch.Tensor, loss_dict: dict, sample: dict) -> None:
        bad = []
        for name, parameter in self.model.named_parameters():
            if bool(torch.isfinite(parameter).all()):
                continue
            finite = torch.isfinite(parameter)
            entry = {
                "name": name,
                "shape": list(parameter.shape),
                "nonfinite_count": int((~finite).sum().item()),
            }
            if bool(finite.any()):
                values = parameter.detach()[finite].float()
                entry["finite_abs_max"] = float(values.abs().max().item())
            bad.append(entry)
            if len(bad) >= 32:
                break
        if self._all_ranks_true(not bad, device=loss.device):
            return
        path = self._dump_nonfinite_diagnostic(
            stage="optimizer",
            sample=sample,
            loss=loss,
            loss_dict=loss_dict,
            bad_parameters=bad,
        )
        raise FloatingPointError(
            f"Non-finite parameter detected globally after optimizer.step; diagnostic={path}"
        )

    def _global_mask_fraction(
        self,
        mask: torch.Tensor | None,
        *,
        device: torch.device,
    ) -> float:
        """Reduce variable-length rank-local masks through fixed-size statistics."""
        stats = torch.zeros(2, device=device, dtype=torch.float32)
        if mask is not None:
            local_mask = mask.to(device=device, dtype=torch.float32).reshape(-1)
            stats[0] = local_mask.sum()
            stats[1] = local_mask.numel()
        global_stats = self.accelerator.reduce(stats, reduction="sum")
        denominator = float(global_stats[1].item())
        return float(global_stats[0].item()) / denominator if denominator > 0.0 else 0.0

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def close(self):
        self._finish_wandb()
        loader = getattr(self, "train_loader", None)
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            shutdown_workers = getattr(iterator, "_shutdown_workers", None)
            if callable(shutdown_workers):
                shutdown_workers()
            loader._iterator = None
        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()

    def _build_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
            mode=self.sampler_mode,
            local_window=self.sampler_local_window,
            balanced_groups=self.sampler_balance_groups,
            min_frame_gap=self.sampler_min_frame_gap,
        )
        loader_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "sampler": self.train_sampler,
            "num_workers": self.num_workers,
            "pin_memory": torch.cuda.is_available(),
            "worker_init_fn": worker_init_fn,
        }
        if self.collate_mode == "mixed_stream":
            action_expert = self.model.action_expert
            loader_kwargs["collate_fn"] = MixedStreamCollator(
                manip_action_dim=action_expert.manip_action_dim,
                nav_action_dim=action_expert.nav_action_dim,
                manip_horizon=action_expert.manip_horizon,
                nav_horizon=action_expert.nav_horizon,
            )
        elif self.collate_mode != "default":
            raise ValueError(
                f"Unsupported collate_mode={self.collate_mode!r}; expected default or mixed_stream."
            )
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = self.persistent_workers
            loader_kwargs["timeout"] = self.dataloader_timeout_s
            if self.prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(dataset, **loader_kwargs)

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        if self._weights_preloaded:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(str(resume_path), optimizer=None)
        logger.warning("Loaded .pt weights only; optimizer/scheduler/step were not restored under ZeRO2.")
        if self.resume_step > 0:
            self.global_step = self.resume_step
            logger.warning(
                "Weights-only resume: preserving configured global step=%d; "
                "optimizer/scheduler state remains freshly initialized.",
                self.global_step,
            )

    def _preload_weight_checkpoint_if_needed(self) -> None:
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info(
            "Loading weight checkpoint before DeepSpeed optimizer-master initialization: %s",
            resume,
        )
        self.model.load_checkpoint(str(resume_path), optimizer=None)
        self._weights_preloaded = True
        logger.warning(
            "Loaded .pt weights only before distributed prepare; "
            "optimizer/scheduler state remains freshly initialized."
        )
        if self.resume_step > 0:
            self.global_step = self.resume_step
            logger.warning(
                "Weights-only resume: preserving configured global step=%d; "
                "optimizer/scheduler state remains freshly initialized.",
                self.global_step,
            )

    def _set_dit_only_train_mode(self):
        # Match DiffSynth's freeze_except("dit"): only DiT stays trainable/in-train-mode.
        logger.info("Setting DiT to train mode and freezing other model components.")
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model)

    @staticmethod
    def _apply_dit_only_train_mode(model):
        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        return {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
        }

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        # eval_index = (self.global_step + self.accelerator.process_index) % len(self.val_dataset)
        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

        # 1. training loss
        with self.accelerator.autocast():
            val_loss, _ = model.training_loss(sample)
            val_loss = val_loss.float().item()
        
        prompt = sample["prompt"][0]
        video0 = sample["video"][0] # Tensor [3, T, H, W] in (-1, 1)
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None # from [1, T, d] to [d]
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        # 2. inference and video saving
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample['action_horizon'],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        pred = model.infer(
            **infer_kwargs,
        )
        
        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        # 3. inference metrics against GT video
        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        action_l1 = None
        action_l2 = None
        if action is not None and pred_action is not None:
            if sample["proprio"] is None:
                raise ValueError("Eval sample must contain `proprio` for action denormalization.")
            proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)
            
            processor = self.val_dataset.lerobot_dataset.processor

            denorm_actions = {}
            action_meta = processor.shape_meta["action"]
            state_meta = processor.shape_meta["state"]
            for action_name, raw_action in (("pred", pred_action), ("gt", action)):
                if not isinstance(raw_action, torch.Tensor):
                    raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
                if raw_action.ndim == 2:
                    action_btd = raw_action.unsqueeze(0)
                elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                    action_btd = raw_action
                else:
                    raise ValueError(
                        f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                    )
                action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

                batch = {
                    "action": action_btd,
                    "state": proprio,
                }
                batch = processor.action_state_merger.backward(batch)
                batch = processor.normalizer.backward(batch)
                merged_batch = {
                    "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                    "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
                }
                merged_batch = processor.action_state_merger.forward(merged_batch)
                denorm_action = merged_batch["action"].unsqueeze(0)
                if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                    raise ValueError(
                        f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                    )
                denorm_actions[action_name] = denorm_action

            pred_action_denorm = denorm_actions["pred"]
            gt_action_denorm = denorm_actions["gt"]

            if pred_action_denorm.shape != gt_action_denorm.shape:
                raise ValueError(
                    "Predicted action/GT action shape mismatch after denormalization: "
                    f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                )
            action_diff = pred_action_denorm - gt_action_denorm
            action_l1 = action_diff.abs().mean().item()
            action_l2 = action_diff.pow(2).mean().item()

        # 4. VAE reconstruction metrics against GT video
        gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

        psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        stitched_video_tensor = torch.cat(
            [pred_video_tensor, vae_video_tensor, gt_video_tensor],
            dim=2,
        ).contiguous()
        stitched_frames = []
        for t in range(stitched_video_tensor.shape[1]):
            frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            stitched_frames.append(Image.fromarray(frame))

        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
        )
        save_mp4(stitched_frames, video_path, fps=8)

        local_metrics = torch.tensor(
            [
                float(val_loss),
                float(psnr_rollout_vs_gt),
                float(ssim_rollout_vs_gt),
                float(psnr_rollout_vs_decode),
                float(ssim_rollout_vs_decode),
                float(psnr_decode_vs_gt),
                float(ssim_decode_vs_gt),
                float(action_l2) if action_l2 is not None else -1.0,
                float(action_l1) if action_l1 is not None else -1.0,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        mean_metrics = gathered_metrics[:, :7].mean(dim=0)
        action_l2_mean = gathered_metrics[:, 7].mean().item() if action_l2 is not None else None
        action_l1_mean = gathered_metrics[:, 8].mean().item() if action_l1 is not None else None

        if was_dit_training:
            self._set_dit_only_train_mode()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "psnr_rg": float(mean_metrics[1].item()),
            "ssim_rg": float(mean_metrics[2].item()),
            "psnr_rd": float(mean_metrics[3].item()),
            "ssim_rd": float(mean_metrics[4].item()),
            "psnr_dg": float(mean_metrics[5].item()),
            "ssim_dg": float(mean_metrics[6].item()),
            "video_path": video_path,
        }
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        model.save_checkpoint(ckpt_path, optimizer=None, step=self.global_step)
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        if not self.save_training_state:
            return {"weights_path": ckpt_path, "state_path": None}

        state_path = os.path.join(self.state_dir, step_tag)
        ensure_dir(state_path)
        self.accelerator.save_state(output_dir=state_path)
        if self.accelerator.is_main_process:
            self._save_trainer_state(state_path)
        self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def load_training_state(self, state_dir: str):
        self.accelerator.load_state(input_dir=state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch_offset(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def train(self):
        self._set_dit_only_train_mode()

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        diagnostic_stop_step = int(
            os.environ.get("FASTWAM_DIAGNOSTIC_STOP_STEP", str(self.max_steps))
        )
        if diagnostic_stop_step <= 0:
            raise ValueError("FASTWAM_DIAGNOSTIC_STOP_STEP must be positive.")
        loop_stop_step = min(self.max_steps, diagnostic_stop_step)
        logger.info(
            "Starting training with max_steps=%d loop_stop_step=%d.",
            self.max_steps,
            loop_stop_step,
        )
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()

        while self.global_step < loop_stop_step:
            profile_step = (
                os.environ.get("FASTWAM_PROFILE_TRAIN", "0") == "1"
                and self.global_step < int(os.environ.get("FASTWAM_PROFILE_TRAIN_STEPS", "2"))
                and self.accelerator.device.type == "cuda"
            )
            if profile_step:
                torch.cuda.synchronize(self.accelerator.device)
                profile_start = time.perf_counter()
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue

            if profile_step:
                data_end = time.perf_counter()
                torch.cuda.synchronize(self.accelerator.device)
                forward_start = time.perf_counter()

            with self.accelerator.accumulate(self.model):
                train_model = self.model if hasattr(self.model, "training_loss") else self.accelerator.unwrap_model(self.model)

                with self.accelerator.autocast():
                    loss, loss_dict = train_model.training_loss(sample)
                self._assert_finite_loss(loss, loss_dict, sample)
                # DeepSpeed may reuse the scalar loss storage during backward/step.
                # Preserve the forward value before that mutation for trustworthy logs.
                loss_for_logging = loss.detach().float().clone()
                self._finite_guard_context = (sample, loss, loss_dict)
                if profile_step:
                    torch.cuda.synchronize(self.accelerator.device)
                    forward_end = time.perf_counter()
                self.accelerator.backward(loss)
                if (
                    self._gradient_guard_active()
                    and not self._deepspeed_optimizer_guard_installed
                ):
                    self._assert_finite_gradients(loss, loss_dict, sample)
                if profile_step:
                    torch.cuda.synchronize(self.accelerator.device)
                    backward_end = time.perf_counter()

                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                    if self._gradient_guard_active():
                        self._assert_finite_parameters(loss, loss_dict, sample)
                    self._finite_guard_context = None
                    self.optimizer.zero_grad(set_to_none=True)
                    if profile_step:
                        torch.cuda.synchronize(self.accelerator.device)
                        optimizer_end = time.perf_counter()
                    self.global_step += 1
                    global_loss = float(
                        self.accelerator.gather(loss_for_logging.reshape(1)).mean().item()
                    )
                    global_loss_metrics = {}
                    for key, value in loss_dict.items():
                        metric_tensor = torch.tensor(float(value), device=loss.device, dtype=torch.float32).reshape(1)
                        global_loss_metrics[key] = float(
                            self.accelerator.gather(metric_tensor).mean().item()
                        )
                    batch_mask_metrics = {}
                    for key in ("nav_loss_valid", "manip_loss_valid"):
                        batch_mask_metrics[key] = self._global_mask_fraction(
                            sample.get(key), device=loss.device
                        )
                    grad_norm_tensor = torch.tensor(grad_norm, device=loss.device, dtype=torch.float32)
                    global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())
                    if profile_step:
                        torch.cuda.synchronize(self.accelerator.device)
                        metrics_end = time.perf_counter()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[train-profile:step] data=%.3fs forward=%.3fs backward=%.3fs "
                                "optimizer=%.3fs metrics=%.3fs total=%.3fs",
                                data_end - profile_start,
                                forward_end - forward_start,
                                backward_end - forward_end,
                                optimizer_end - backward_end,
                                metrics_end - optimizer_end,
                                metrics_end - profile_start,
                            )

                    current_lr = float(self.optimizer.param_groups[0]["lr"])

                    if self.log_every > 0 and self.global_step % self.log_every == 0 and self.accelerator.is_main_process:
                        eta_str, steps_per_sec = self._estimate_eta()
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if global_loss_metrics:
                            detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                            description += detail_str + " "
                        if batch_mask_metrics:
                            detail_str = " ".join(
                                [f"{k}_frac={v:.3f}" for k, v in sorted(batch_mask_metrics.items())]
                            )
                            description += detail_str + " "
                        description += "lr=%.2e speed=%.2f step/s, %.2f samples/s eta=%s" % (
                            current_lr,
                            steps_per_sec,
                            steps_per_sec * self.batch_size * self.accelerator.num_processes,
                            eta_str,
                        )
                        active_stream_ratio = global_loss_metrics.get(
                            "active_streams_per_sample", 1.0
                        )
                        streams_per_sec = (
                            steps_per_sec
                            * self.batch_size
                            * self.accelerator.num_processes
                            * active_stream_ratio
                        )
                        description += " %.2f streams/s" % streams_per_sec
                        logger.info(description)

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": steps_per_sec * self.batch_size * self.accelerator.num_processes,
                            "performance/streams_per_sec": streams_per_sec,
                        }
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        for key, value in batch_mask_metrics.items():
                            wandb_payload[f"batch/{key}_frac"] = value
                        self._wandb_log(wandb_payload)

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            description = "[eval] step=%d val_loss=%.4f infer_psnr=%.4f infer_ssim=%.4f" % (
                                self.global_step,
                                metrics["val_loss"],
                                metrics["psnr_rd"],
                                metrics["ssim_rd"],
                            )
                            if "action_l2" in metrics:
                                description += " action_l2=%.4f" % metrics["action_l2"]
                            if "action_l1" in metrics:
                                description += " action_l1=%.4f" % metrics["action_l1"]
                            logger.info(description)
                            eval_payload = {
                                "eval/val_loss": float(metrics["val_loss"]),
                                "eval/psnr_rg": float(metrics["psnr_rg"]),
                                "eval/ssim_rg": float(metrics["ssim_rg"]),
                                "eval/psnr_rd": float(metrics["psnr_rd"]),
                                "eval/ssim_rd": float(metrics["ssim_rd"]),
                                "eval/psnr_dg": float(metrics["psnr_dg"]),
                                "eval/ssim_dg": float(metrics["ssim_dg"]),
                            }
                            if "action_l2" in metrics:
                                eval_payload["eval/action_l2"] = float(metrics["action_l2"])
                            if "action_l1" in metrics:
                                eval_payload["eval/action_l1"] = float(metrics["action_l1"])
                            self._wandb_log(eval_payload)

                    if self.save_every > 0 and self.global_step % self.save_every == 0:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )

                    if self.global_step >= self.max_steps:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[done] max_steps reached step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                        return

                    if self.global_step >= loop_stop_step:
                        self.accelerator.wait_for_everyone()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[diagnostic-done] stopped cleanly at step=%d without saving; "
                                "scheduler max_steps remained %d",
                                self.global_step,
                                self.max_steps,
                            )
                        return

        ckpt_info = self.save_checkpoint()
        if self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d weights=%s state=%s",
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )
        
