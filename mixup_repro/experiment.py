"""Training never constructs a test dataset. Testing requires a frozen manifest.

Run `python -m mixup_repro.experiment --help` from the project root.
"""
import argparse
import copy
import csv
import io
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from vendor.resnet import ResNet18
from .core import (atomic_checkpoint, atomic_json, digest, file_hash, improves,
                   learning_rate, mixed_loss, stratified_split, weights_hash)

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('erm', 'mixup')


def read_config(path):
    cfg = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if cfg['epochs'] < 1 or cfg['batch_size'] < 2 or cfg['alpha'] <= 0:
        raise ValueError('Invalid epoch/batch/mixup configuration')
    if not cfg['seeds'] or len(set(cfg['seeds'])) != len(cfg['seeds']):
        raise ValueError('Seeds must be nonempty and unique')
    if cfg['milestones'] != sorted(set(cfg['milestones'])):
        raise ValueError('Milestones must be ordered and unique')
    if any(not 0 < m < cfg['epochs'] for m in cfg['milestones']):
        raise ValueError('Milestones must lie inside the epoch budget')
    return cfg


def source_hash():
    files = sorted((ROOT / 'mixup_repro').glob('*.py')) + [ROOT / 'vendor/resnet.py']
    return digest({p.relative_to(ROOT).as_posix(): file_hash(p) for p in files})


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_worker(_):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def choose_device(request):
    if request == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. In Colab choose Runtime > Change runtime type > GPU. '
                           'Use --device cpu only for debugging.')
    return torch.device(request)


def environment():
    try:
        freeze = subprocess.check_output([os.sys.executable, '-m', 'pip', 'freeze'], text=True)
    except subprocess.SubprocessError:
        freeze = 'unavailable'
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'torch': torch.__version__, 'torchvision': torchvision.__version__,
            'numpy': np.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'pip_freeze': freeze}


def transform(cfg, augment=False):
    ops = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if augment else []
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(cfg['mean'], cfg['std'])])


def development_data(cfg, data_root, output, download=False):
    # The CIFAR archive contains test files, but development loads only train=True.
    raw = datasets.CIFAR10(data_root, train=True, download=download, transform=transform(cfg))
    if len(raw) != 50000:
        raise ValueError('Expected the complete CIFAR-10 training split')
    split = stratified_split(raw.targets, cfg['validation_per_class'], cfg['split_seed'])
    split_path = Path(output) / 'split.json'
    if split_path.exists():
        if json.loads(split_path.read_text()) != split:
            raise RuntimeError('Existing split differs; use a new output directory for a revised protocol')
    else:
        atomic_json(split_path, split)
    augmented = datasets.CIFAR10(data_root, train=True, download=False, transform=transform(cfg, True))
    return Subset(augmented, split['train']), Subset(raw, split['validation']), digest(split)


def loader(data, cfg, seed=0, shuffle=False):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(data, batch_size=cfg['batch_size'], shuffle=shuffle,
                      num_workers=cfg['num_workers'], generator=generator,
                      worker_init_fn=seed_worker, persistent_workers=False)


@torch.inference_mode()
def evaluate_model(model, data, cfg, device):
    model.eval()
    loss_sum, correct, count = 0., 0, 0
    for x, y in loader(data, cfg):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += nn.functional.cross_entropy(logits, y, reduction='sum').item()
        correct += (logits.argmax(1) == y).sum().item()
        count += y.numel()
    if not count:
        raise ValueError('Empty evaluation dataset')
    return {'loss': loss_sum/count, 'error_pct': 100*(count-correct)/count,
            'correct': correct, 'count': count}


def train_epoch(model, optimizer, data, cfg, seed, epoch, method, device):
    # Separate mixup randomness so paired methods keep the same shuffle/augmentation stream.
    epoch_seed = seed * 100003 + epoch
    seed_everything(epoch_seed)
    mix_rng = np.random.default_rng(epoch_seed + 700001)
    perm_rng = torch.Generator().manual_seed(epoch_seed + 900001)
    model.train()
    loss_sum, count = 0., 0
    for x, y in loader(data, cfg, epoch_seed, True):
        x, y = x.to(device), y.to(device)
        if method == 'mixup':
            lam = float(mix_rng.beta(cfg['alpha'], cfg['alpha']))
            permutation = torch.randperm(len(y), generator=perm_rng).to(device)
            inputs = lam*x + (1-lam)*x[permutation]
        else:
            inputs = x
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = mixed_loss(logits, y, y[permutation], lam) if method == 'mixup' else nn.functional.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()*len(y)
        count += len(y)
    return loss_sum/count


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}


def write_history(run, rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    path = Path(run) / f'{Path(run).name}_history.csv'
    temp = path.with_suffix('.csv.tmp')
    temp.write_text(stream.getvalue(), encoding='utf-8')
    temp.replace(path)


def train_run(cfg, method, seed, train, val, split_hash, output, device, resume=False,
              stop_after=None, model_factory=ResNet18, scope='main', save_every=1):
    """Epoch-boundary resume. stop_after is for interrupted-run verification only.

    Checkpoints are written every save_every epochs and at the final epoch. Epochs after the
    last checkpoint are rerun on resume; per-epoch reseeding makes the rerun identical.
    save_every is not part of the run metadata, so it may differ between sessions.
    """
    if method not in METHODS or seed not in cfg['seeds']:
        raise ValueError('Method/seed is outside this protocol')
    if save_every < 1:
        raise ValueError('save_every must be at least 1')
    if (Path(output)/'frozen.json').exists():
        raise RuntimeError('Protocol frozen for testing; training is now locked')
    run = Path(output)/f'{method}_seed{seed}'
    run.mkdir(parents=True, exist_ok=True)
    checkpoint = run/'checkpoint.pt'
    metadata = {'config': cfg, 'method': method, 'seed': seed, 'split_hash': split_hash,
                'source_hash': source_hash(), 'scope': scope}
    seed_everything(seed)
    model = model_factory().to(device)
    initial_hash = weights_hash(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'],
                                weight_decay=cfg['weight_decay'])
    start, history, best_error, best_epoch, best_model = 1, [], float('inf'), None, None
    env = environment()
    envs = [env]
    if checkpoint.exists():
        if not resume:
            raise FileExistsError(f'{checkpoint} exists; use --resume to continue')
        # Load only checkpoints created by this project; torch checkpoints are trusted local files.
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if state['metadata'] != metadata or state['initial_weights_sha256'] != initial_hash:
            raise RuntimeError('Resume rejected: code, config, split, initialization or scope changed')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start, history = state['epoch']+1, state['history']
        best_error, best_epoch, best_model = state['best_error'], state['best_epoch'], state['best_model']
        envs = state['environments']
        if env != envs[-1]:
            print('WARNING: runtime changed since previous segment; recorded in checkpoint', flush=True)
            envs.append(env)
    for epoch in range(start, cfg['epochs']+1):
        if stop_after is not None and epoch > stop_after:
            break
        lr = learning_rate(cfg, epoch)
        for group in optimizer.param_groups:
            group['lr'] = lr
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        begin = time.perf_counter()
        loss = train_epoch(model, optimizer, train, cfg, seed, epoch, method, device)
        result = evaluate_model(model, val, cfg, device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        seconds = time.perf_counter()-begin
        if improves(result['error_pct'], best_error):
            best_error, best_epoch, best_model = result['error_pct'], epoch, cpu_state(model)
        row = {'epoch':epoch, 'lr':lr, 'train_objective':loss, 'validation_loss':result['loss'],
               'validation_error_pct':result['error_pct'], 'compute_seconds':seconds,
               'peak_gpu_mb':torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else 0.}
        history.append(row)
        saved = epoch % save_every == 0 or epoch == cfg['epochs']
        if saved:
            state = {'metadata':metadata, 'epoch':epoch, 'model':cpu_state(model),
                     'optimizer':optimizer.state_dict(), 'best_model':best_model,
                     'best_epoch':best_epoch, 'best_error':best_error, 'history':history,
                     'initial_weights_sha256':initial_hash, 'environments':envs}
            atomic_checkpoint(checkpoint, state)
        write_history(run, history)
        print(f'{method} seed={seed} epoch={epoch}/{cfg["epochs"]} '
              f'val_error={result["error_pct"]:.2f}% lr={lr:g} compute={seconds:.1f}s'
              f'{" [checkpoint saved]" if saved else ""}', flush=True)
    if history:
        write_history(run, history)
    return checkpoint


def freeze_runs(cfg, output):
    """Verify all paired runs before permitting access to test metrics."""
    output = Path(output)
    if (output/'frozen.json').exists():
        raise FileExistsError('Already frozen; preserve the existing manifest')
    entries, splits = [], set()
    for seed in cfg['seeds']:
        initial = []
        for method in METHODS:
            path = output/f'{method}_seed{seed}/checkpoint.pt'
            state = torch.load(path, map_location='cpu', weights_only=False)
            meta = state['metadata']
            if (meta['config'] != cfg or meta['scope'] != 'main' or meta['method'] != method or
                meta['seed'] != seed or meta['source_hash'] != source_hash() or state['epoch'] != cfg['epochs']):
                raise RuntimeError(f'Incomplete or mismatched run: {path}')
            initial.append(state['initial_weights_sha256'])
            splits.add(meta['split_hash'])
            entries.append({'method':method, 'seed':seed, 'checkpoint':str(path.relative_to(output)),
                            'sha256':file_hash(path), 'best_epoch':state['best_epoch'],
                            'best_validation_error_pct':state['best_error']})
        if initial[0] != initial[1]:
            raise RuntimeError(f'Initial weights differ for paired seed {seed}')
    if len(splits) != 1 or digest(json.loads((output/'split.json').read_text())) not in splits:
        raise RuntimeError('Dataset splits differ')
    manifest = {'config':cfg,'source_hash':source_hash(),'split_hash':next(iter(splits)),'runs':entries}
    atomic_json(output/'frozen.json',manifest)
    print('Frozen all runs. Evaluation may now load the test dataset.',flush=True)


def test_frozen(output, data_root, device, download=False):
    output = Path(output)
    manifest_path = output/'frozen.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['source_hash'] != source_hash():
        raise RuntimeError('Code changed after freezing')
    for run in manifest['runs']:
        if file_hash(output/run['checkpoint']) != run['sha256']:
            raise RuntimeError('Checkpoint changed after freezing')
    result_path=output/'test_results.json'
    if result_path.exists():
        raise FileExistsError('Test results already exist; use summarize without re-evaluating')
    cfg=manifest['config']
    # This is the only production path that constructs a CIFAR-10 test dataset.
    # download only restores the archive in a fresh session, after the freeze checks above.
    test=datasets.CIFAR10(data_root, train=False, download=download, transform=transform(cfg))
    if len(test)!=10000:
        raise ValueError('Expected the complete 10,000-image test set')
    rows=[]
    for run in manifest['runs']:
        seed_everything(run['seed'])
        state=torch.load(output/run['checkpoint'],map_location='cpu',weights_only=False)
        model=ResNet18().to(device)
        model.load_state_dict(state['best_model'])
        metric=evaluate_model(model,test,cfg,device)
        rows.append({**run,**metric})
    atomic_json(result_path,{'manifest_sha256':file_hash(manifest_path),'environment':environment(),'runs':rows})
    print('Saved final test metrics. All values are from selected validation checkpoints.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['pilot','train','freeze','evaluate'])
    parser.add_argument('--config',default='configs/proposal.json')
    parser.add_argument('--data',default='data')
    parser.add_argument('--output',default='runs/main')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--method',choices=METHODS)
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--download',action='store_true')
    parser.add_argument('--save-every',type=int,default=5,
                        help='write checkpoint.pt every N epochs and at the final epoch (default 5)')
    args=parser.parse_args()
    if args.save_every<1:
        parser.error('--save-every must be at least 1')
    cfg=read_config(args.config)
    output=Path(args.output)
    if args.command=='freeze':
        freeze_runs(cfg,output)
        return
    device=choose_device(args.device)
    if args.command=='evaluate':
        test_frozen(output,args.data,device,args.download)
        return
    if args.seed not in cfg['seeds']:
        parser.error('--seed must be listed in the config')
    if args.command=='train' and args.method is None:
        parser.error('train requires --method erm or --method mixup')
    if args.command=='pilot' and output.name!='pilot':
        output=output/'pilot'
    train,val,split_hash=development_data(cfg,args.data,output,args.download)
    if args.command=='train':
        train_run(cfg,args.method,args.seed,train,val,split_hash,output,device,args.resume,
                  save_every=args.save_every)
    else:
        rng=np.random.default_rng(cfg['split_seed']+1)
        train=Subset(train,rng.choice(len(train),min(1024,len(train)),replace=False).tolist())
        val=Subset(val,rng.choice(len(val),min(256,len(val)),replace=False).tolist())
        pilot=copy.deepcopy(cfg)
        pilot.update(epochs=2,milestones=[])
        for method in METHODS:
            train_run(pilot,method,args.seed,train,val,split_hash,output,device,args.resume,scope='pilot',
                      save_every=args.save_every)
        print('Pilot only: 1,024 train / 256 validation images. These are NOT reproduction results.',flush=True)


if __name__=='__main__':
    main()
