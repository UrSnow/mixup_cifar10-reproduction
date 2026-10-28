"""Protocol rules, kept separate from the training loop for inspection/testing."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def weights_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def stratified_split(labels, per_class, seed):
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    train, val = [], []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if not 0 < per_class < len(indices):
            raise ValueError('Each class needs both training and validation examples')
        rng.shuffle(indices)
        val.extend(indices[:per_class].tolist())
        train.extend(indices[per_class:].tolist())
    return {'train': sorted(train), 'validation': sorted(val), 'seed': seed,
            'labels_hash': digest(labels.tolist())}


def learning_rate(cfg, epoch):
    """One-based epoch: decay AFTER completed epochs 25 and 38."""
    return cfg['lr'] * 0.1 ** sum(epoch > m for m in cfg['milestones'])


def improves(error, best_error):
    # A strict comparison retains the earlier epoch on an exact tie.
    return error < best_error


def mixed_loss(logits, y_a, y_b, lam):
    return lam * F.cross_entropy(logits, y_a) + (1 - lam) * F.cross_entropy(logits, y_b)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def atomic_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    temp.replace(path)
