#!/usr/bin/env python3
"""Infer normalized chunks from one or both streams with your own trained model."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from fastwam.datasets.robot_windows import ActionNormalizer
from fastwam.robot_runtime import load_robot_config, make_robot_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/uniwam_robot_base.yaml')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--normalization', type=Path, required=True)
    parser.add_argument('--observation', type=Path, required=True, help='Tensor-only .pt payload described in ROBOT.md.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    cfg = load_robot_config(args.config)
    model = make_robot_model(cfg, args.device, args.checkpoint).eval()
    normalizer = ActionNormalizer(args.normalization)
    sample = torch.load(args.observation, map_location='cpu', weights_only=True)
    proprio = sample['proprio'].float()
    if proprio.shape != (20,) or not torch.isfinite(proprio).all():
        raise ValueError('proprio must be a finite [20] state in the current camera frame.')
    proprio = normalizer.encode('proprio', proprio, sample['proprio_mask'].bool())
    result = model.infer_action(
        action_horizon=cfg['model']['action_dit_config']['manip_horizon'],
        nav_input_image=sample.get('nav_image'), manip_input_image=sample.get('manip_image'),
        context=sample['context'], context_mask=sample['context_mask'], proprio=proprio,
        committed_prefix=sample.get('committed_prefix'), action_feature_mask=sample.get('action_feature_mask'),
        num_inference_steps=cfg['inference_steps'], seed=args.seed)
    output = {'normalized': result, 'decoded': {}}
    for branch in ('nav', 'manip'):
        key = f'{branch}_action'
        if key in result:
            decoded = normalizer.decode(branch, result[key])
            mask = sample.get('action_feature_mask', {}).get(branch)
            if mask is not None: decoded = torch.where(mask.bool(), decoded, 0)
            output['decoded'][key] = decoded
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(f'Saved {", ".join(result)} to {args.output}')


if __name__ == '__main__':
    main()
