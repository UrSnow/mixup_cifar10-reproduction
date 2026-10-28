"""Summarize measured outputs only. Never fills missing results with paper values."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

from .core import file_hash


def summarize(output):
    output=Path(output)
    manifest=json.loads((output/'frozen.json').read_text())
    results=json.loads((output/'test_results.json').read_text())
    if results['manifest_sha256'] != file_hash(output/'frozen.json'):
        raise ValueError('Test results do not match the frozen protocol')
    rows=results['runs']
    by_key={(r['seed'],r['method']):r for r in rows}
    seeds=manifest['config']['seeds']
    expected={(s,m) for s in seeds for m in ('erm','mixup')}
    if set(by_key)!=expected or len(rows)!=len(expected):
        raise ValueError('Missing, duplicate or unexpected test results')
    paired=[{'seed':s,'erm_error_pct':by_key[s,'erm']['error_pct'],
             'mixup_error_pct':by_key[s,'mixup']['error_pct'],
             'improvement_pp':by_key[s,'erm']['error_pct']-by_key[s,'mixup']['error_pct']}
            for s in seeds]
    with (output/'paired_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(paired[0])); w.writeheader(); w.writerows(paired)
    lines=['# Measured reproduction results','',f'Paired seeds: {len(seeds)}. Positive differences favour mixup.','',
           '| Method | Mean test error (%) | Sample SD (percentage points) |',
           '|---|---:|---:|']
    for m in ('erm','mixup'):
        values=[by_key[s,m]['error_pct'] for s in seeds]
        sd=f'{stdev(values):.3f}' if len(values)>1 else 'not estimable (one seed)'
        lines.append(f'| {m} | {mean(values):.3f} | {sd} |')
    differences=[p['improvement_pp'] for p in paired]
    lines+=['',f'Mean paired improvement: {mean(differences):.3f} percentage points.',
            f'Mixup wins: {sum(d>0 for d in differences)}/{len(seeds)}; ties: {sum(d==0 for d in differences)}.',
            '', 'These results apply to the frozen reduced-scale protocol. No significance claim is inferred from three seeds.']
    (output/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for seed in seeds:
        for method in ('erm','mixup'):
            path=output/f'{method}_seed{seed}'/f'{method}_seed{seed}_history.csv'
            with path.open() as f: history=list(csv.DictReader(f))
            epochs=[int(r['epoch']) for r in history]
            color='tab:blue' if method=='erm' else 'tab:orange'
            label=f'{method} seed {seed}'
            axes[0].plot(epochs,[float(r['train_objective']) for r in history],label=label,color=color,alpha=.65)
            axes[1].plot(epochs,[float(r['validation_error_pct']) for r in history],label=label,color=color,alpha=.65)
    axes[0].set(title='Training objectives differ between methods',xlabel='Epoch',ylabel='Training objective')
    axes[1].set(title='Common validation metric',xlabel='Epoch',ylabel='Validation error (%)')
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output/'learning_curves.png',dpi=180); plt.close(fig)
    print('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='runs/main')
    summarize(parser.parse_args().output)
