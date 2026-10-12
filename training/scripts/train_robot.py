#!/usr/bin/env python3
"""Train full UniWAM on prepared demonstrations; defaults target real-robot Base."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from fastwam.datasets.robot_windows import RobotWindows, IndependentWindowSampler, collate_robot_windows
from fastwam.robot_runtime import load_robot_config, make_robot_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/uniwam_robot_base.yaml')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--data-only', action='store_true', help='Decode and validate one complete batch without allocating a model.')
    parser.add_argument('--initialize-from', type=Path, help='Initialize from your own full-model checkpoint with a fresh optimizer.')
    args = parser.parse_args()
    cfg = load_robot_config(args.config)
    if args.data_root: cfg['data_root'] = str(args.data_root.resolve())
    if args.output_dir: cfg['output_dir'] = str(args.output_dir.resolve())
    if args.steps is not None: cfg['max_steps'] = args.steps
    if args.batch_size is not None: cfg['batch_size_per_rank'] = args.batch_size
    if cfg['mixed_precision'] not in ('bf16', 'fp32'):
        raise ValueError('mixed_precision must be bf16 or fp32.')
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    dataset = RobotWindows(cfg['data_root'], cfg['manifest'], cfg['normalization'], cfg['model']['action_dit_config']['manip_horizon'])
    sampler = IndependentWindowSampler(dataset.rows, cfg['batch_size_per_rank'], cfg['max_steps'],
                                       seed=cfg['seed'], rank=rank, world_size=world, **cfg['sampling'])
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=cfg['num_workers'], collate_fn=collate_robot_windows)
    if args.data_only:
        batch = next(iter(loader))
        print(json.dumps({k: list(v.shape) for k, v in batch.items()}))
        return
    if not torch.cuda.is_available():
        raise RuntimeError('Full model training requires a CUDA device.')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if world > 1: dist.init_process_group('nccl', device_id=device)
    torch.manual_seed(cfg['seed'])
    model = make_robot_model(cfg, device, args.initialize_from)
    model.train(); model.vae.eval()
    if model.text_encoder is not None: model.text_encoder.eval()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['max_steps'])
    wrapped = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False) if world > 1 else model
    torch.manual_seed(cfg['seed'] + rank)
    output = Path(cfg['output_dir'])
    if rank == 0: output.mkdir(parents=True, exist_ok=True)
    for step, batch in enumerate(loader, 1):
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = wrapped(batch)
        if not torch.isfinite(loss): raise FloatingPointError('Non-finite training loss.')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, cfg['max_grad_norm'], error_if_nonfinite=True)
        optimizer.step(); scheduler.step()
        if step == 1 or step % 10 == 0:
            report = torch.stack([loss.detach().float(), *[x.float() for x in metrics.values()]])
            if world > 1: dist.all_reduce(report); report /= world
            if rank == 0:
                print(json.dumps({'step': step, 'loss': report[0].item(), **{k: v.item() for k, v in zip(metrics, report[1:])}}), flush=True)
        if step % cfg['save_every'] == 0 or step == cfg['max_steps']:
            if rank == 0:
                # Portable model weights only. Task-specific weights stay with their owner.
                state = {'mot': model.mot.state_dict(), 'proprio_encoder': model.proprio_encoder.state_dict()}
                temporary = output / 'model.tmp'
                torch.save(state, temporary)
                temporary.replace(output / 'model.pt')
                (output / 'model_config.json').write_text(json.dumps(cfg['model'], indent=2) + '\n')
                (output / 'normalization.json').write_text((Path(cfg['data_root']) / cfg['normalization']).read_text())
            if world > 1: dist.barrier()
    if world > 1: dist.destroy_process_group()


if __name__ == '__main__':
    main()
