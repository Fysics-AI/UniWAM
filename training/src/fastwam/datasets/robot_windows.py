"""Portable, independently sampled real-robot windows for UniWAM."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def relative_file(root, value):
    root = Path(root).resolve()
    path = Path(value)
    if path.is_absolute() or not (root / path).resolve().is_relative_to(root):
        raise ValueError("Dataset paths must be relative and stay inside data_root.")
    return root / path


class ActionNormalizer:
    """Training-set affine normalization. Invalid entries remain exactly zero."""
    def __init__(self, path):
        stats = json.loads(Path(path).read_text())
        self.stats = {}
        for branch, dim in (("nav", 9), ("manip", 26), ("proprio", 20)):
            offset = torch.tensor(stats[branch]['offset'], dtype=torch.float32)
            scale = torch.tensor(stats[branch]['scale'], dtype=torch.float32)
            if offset.shape != (dim,) or scale.shape != (dim,) or not torch.isfinite(offset).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError(f"Invalid {branch} normalization; expected {dim} finite dimensions and positive scales.")
            self.stats[branch] = offset, scale

    def encode(self, branch, values, mask):
        offset, scale = (x.to(values) for x in self.stats[branch])
        return torch.where(mask, (values - offset) / scale, 0)

    def decode(self, branch, values):
        offset, scale = (x.to(values) for x in self.stats[branch])
        return values * scale + offset


class RobotWindows(Dataset):
    def __init__(self, root, manifest='windows.jsonl', normalization='normalization.json', horizon=32):
        self.root = Path(root).resolve()
        self.rows = [json.loads(line) for line in relative_file(self.root, manifest).read_text().splitlines() if line.strip()]
        if not self.rows:
            raise ValueError('Empty robot window manifest.')
        self.normalizer = ActionNormalizer(relative_file(self.root, normalization))
        self.horizon = horizon
        for row in self.rows:
            if row['stream'] not in ('nav', 'manip') or type(row['active']) is not bool or not isinstance(row['start'], int) or row['start'] < 0:
                raise ValueError('Each row needs stream, boolean active and nonnegative integer start.')
            for key in ('source', 'episode'):
                if not isinstance(row[key], str) or not row[key]:
                    raise ValueError(f'Missing {key}.')
            for key in ('path', 'text'):
                if not relative_file(self.root, row[key]).is_file():
                    raise FileNotFoundError(row[key])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with np.load(relative_file(self.root, row['path']), allow_pickle=False) as data:
            tensors = {key: torch.from_numpy(data[key].copy()) for key in data.files}
        stream, horizon = row['stream'], self.horizon
        dim = 26 if stream == 'manip' else 9
        expected = {'video': (3, 9, 320, 384), 'action': (horizon, dim), 'feature_mask': (horizon, dim),
                    'action_is_pad': (horizon,), 'image_is_pad': (9,), 'proprio': (20,), 'proprio_mask': (20,)}
        for key, shape in expected.items():
            if key not in tensors or tuple(tensors[key].shape) != shape:
                raise ValueError(f'{key} must have shape {shape}.')
        if tensors['video'].dtype != torch.uint8:
            raise ValueError('video must be uint8.')
        for key in ('feature_mask', 'action_is_pad', 'image_is_pad', 'proprio_mask'):
            if tensors[key].dtype != torch.bool:
                raise ValueError(f'{key} must be boolean.')
        if tensors['image_is_pad'][0]:
            raise ValueError('Current observation cannot be padded.')
        action, mask = tensors['action'].float(), tensors['feature_mask']
        mask = mask & ~tensors['action_is_pad'][:, None]
        if not torch.isfinite(action[mask]).all() or not torch.isfinite(tensors['proprio'][tensors['proprio_mask']]).all():
            raise ValueError('Valid actions/state must be finite.')
        if not mask.any():
            raise ValueError('Window has no supervised actions.')
        text = torch.load(relative_file(self.root, row['text']), map_location='cpu', weights_only=True)
        context, context_mask = text['context'].float(), text['mask'].bool()
        if context.shape != (128, 4096) or context_mask.shape != (128,) or not context_mask.any() or not torch.isfinite(context).all():
            raise ValueError('Invalid T5 context or mask.')
        tensors.update(stream=stream, context=context, context_mask=context_mask,
                       action=self.normalizer.encode(stream, action, mask), feature_mask=mask,
                       proprio=self.normalizer.encode('proprio', tensors['proprio'].float(), tensors['proprio_mask']))
        return tensors


def collate_robot_windows(examples):
    """One independent window per owner; branch arrays contain only their owners."""
    batch = {key: torch.stack([x[key] for x in examples]) for key in ('context', 'context_mask', 'proprio')}
    batch['proprio'] = batch['proprio'][:, None]
    batch['proprio_valid'] = torch.tensor([x['proprio_mask'].any().item() for x in examples])
    for branch in ('manip', 'nav'):
        indices = [i for i, x in enumerate(examples) if x['stream'] == branch]
        batch[f'{branch}_branch_valid'] = torch.tensor([x['stream'] == branch for x in examples])
        if not indices:
            continue
        names = {'video': f'{branch}_video', 'action': f'{branch}_action', 'feature_mask': f'{branch}_action_feature_mask',
                 'image_is_pad': f'{branch}_image_is_pad', 'action_is_pad': 'action_is_pad' if branch == 'manip' else 'nav_action_is_pad'}
        for src, dest in names.items():
            batch[dest] = torch.stack([examples[i][src] for i in indices])
    return batch


class IndependentWindowSampler(Sampler):
    """Source-normalized masses and global-batch temporal deduplication.

    All ranks use the same seed and draw the same global batch, then take a
    disjoint slice. This sampler must not be wrapped in another sharding sampler.
    """
    def __init__(self, rows, batch_size, steps, seed=42, rank=0, world_size=1,
                 source_weights=None, idle_weights=None, min_start_separation=33):
        self.rows, self.batch_size, self.steps = rows, batch_size, steps
        self.seed, self.rank, self.world_size = seed, rank, world_size
        self.separation = min_start_separation
        if batch_size < 1 or steps < 1 or not 0 <= rank < world_size or min_start_separation < 1:
            raise ValueError('Invalid sampler configuration.')
        source_weights = source_weights or {}
        idle_weights = idle_weights or {'manip': .20, 'nav': .15}
        groups = {}
        for i, row in enumerate(rows):
            groups.setdefault((row['source'], row['stream']), []).append(i)
        probability = np.zeros(len(rows), np.float64)
        for (source, stream), indices in groups.items():
            mass = float(source_weights.get(f'{source}/{stream}', 1.0))
            weights = np.array([1.0 if rows[i]['active'] else idle_weights[stream] for i in indices], dtype=np.float64)
            if mass < 0 or not np.isfinite(mass) or (weights < 0).any() or not np.isfinite(weights).all() or weights.sum() <= 0:
                raise ValueError('Invalid source/window sampling weight.')
            probability[indices] = mass * weights / weights.sum()
        if probability.sum() <= 0:
            raise ValueError('Sampling mass must be positive.')
        self.probability = probability / probability.sum()
        self.cumulative_probability = np.cumsum(self.probability)
        self.cumulative_probability[-1] = 1.0

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        total = self.batch_size * self.world_size
        for _ in range(self.steps):
            chosen, starts = [], {}
            for attempt in range(max(1000, total * 1000)):
                index = int(np.searchsorted(self.cumulative_probability, rng.random(), side='right'))
                row = self.rows[index]
                key = row['source'], row['episode']
                if any(abs(row['start'] - start) < self.separation for start in starts.get(key, [])):
                    continue
                starts.setdefault(key, []).append(row['start'])
                chosen.append(index)
                if len(chosen) == total:
                    break
            if len(chosen) != total:
                raise ValueError('Cannot draw a separated global batch; add independent windows or reduce batch size.')
            yield chosen[self.rank * self.batch_size:(self.rank + 1) * self.batch_size]
