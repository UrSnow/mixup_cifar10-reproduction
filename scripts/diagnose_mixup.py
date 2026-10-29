# Written for this project; the authors' code is only in vendor/ (see vendor/PROVENANCE.md).
"""Diagnose the mixup-versus-ERM validation gap without loading any test data.

For each method's validation-selected weights (one seed), report:
  1. validation error with the BatchNorm statistics collected during training;
  2. error on 5,000 clean (unaugmented, unmixed) training images, i.e. how well the model fits;
  3. both again after re-estimating BatchNorm statistics on unmixed training batches.

Reading the result: if mixup's clean training error is also high, mixup is underfitting at this
budget. If its training error is low but validation error is high, it is a generalisation gap.
If re-estimating BatchNorm closes most of the gap, the statistics collected on mixed inputs do
not match clean images. Uses only the development (train/validation) split.

Run from the project root, e.g. on Colab:
    python scripts/diagnose_mixup.py --output /content/drive/MyDrive/mixup-reproduction-v1/main \
        --data /content/cifar10
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mixup_repro.experiment import (METHODS, ResNet18, choose_device, development_data,  # noqa: E402
                                    evaluate_model, loader, read_config, seed_everything)


def reestimate_batchnorm(model, data, cfg, device, batches):
    """Replace running statistics with a plain average over unmixed training batches."""
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()
            module.momentum = None  # None means a cumulative average instead of a moving one
    model.train()
    with torch.no_grad():
        for index, (x, _) in enumerate(loader(data, cfg, seed=0, shuffle=True)):
            if index == batches:
                break
            model(x.to(device))
    model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', required=True, help='run folder containing <method>_seed<seed>/')
    parser.add_argument('--data', required=True, help='CIFAR-10 root folder')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batches', type=int, default=100,
                        help='unmixed training batches used to re-estimate BatchNorm (default 100)')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()

    cfg = read_config(ROOT/'configs/proposal.json')
    device = choose_device(args.device)
    seed_everything(0)
    train, val, _ = development_data(cfg, args.data, args.output, args.download)
    # val.dataset has no augmentation, so it also serves clean training images.
    picks = np.random.default_rng(0).choice(train.indices, 5000, replace=False)
    clean_train = Subset(val.dataset, picks.tolist())

    rows = []
    for method in METHODS:
        path = Path(args.output)/f'{method}_seed{args.seed}'/'checkpoint.pt'
        state = torch.load(path, map_location='cpu', weights_only=False)
        model = ResNet18().to(device)
        model.load_state_dict(state['best_model'])
        val_before = evaluate_model(model, val, cfg, device)['error_pct']
        fit_before = evaluate_model(model, clean_train, cfg, device)['error_pct']
        reestimate_batchnorm(model, train, cfg, device, args.batches)
        val_after = evaluate_model(model, val, cfg, device)['error_pct']
        fit_after = evaluate_model(model, clean_train, cfg, device)['error_pct']
        rows.append((method, state['best_epoch'], state['best_error'], val_before, fit_before,
                     val_after, fit_after))

    print(f'Seed {args.seed}, validation-selected weights. Errors in %. No test data was loaded.')
    print('| Method | Best epoch | Val (logged) | Val | Clean train | Val, BN re-est. | Clean train, BN re-est. |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    for method, epoch, logged, *errors in rows:
        print(f'| {method} | {epoch} | {logged:.2f} | ' + ' | '.join(f'{e:.2f}' for e in errors) + ' |')
    print('"Val" should equal "Val (logged)"; a difference means the evaluation is not reproducible.')


if __name__ == '__main__':
    main()
