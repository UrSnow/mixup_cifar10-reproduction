"""Reproduce every number, table and figure in the write-up with one command.

From the committed results (CPU, a few seconds, PyTorch not needed):
    python scripts/reproduce.py --from-results results

Full rerun from a clean checkout (CUDA GPU; about 6 GPU-hours on a Colab T4):
    python scripts/reproduce.py --data /content/cifar10 --output /content/drive/MyDrive/mixup-rerun

The full rerun trains the six main runs and the small-alpha check, freezes, tests once, summarizes,
runs the seed 1 diagnostic and copies the outputs into <output>/results, laid out like results/.
Rerunning the same command after a disconnect resumes where it stopped.

Both modes then check the results against the frozen manifest and the code in this checkout, write
the paper's numbers, tables and figures to paper/generated/, and build paper/main.pdf when pdflatex
and bibtex are installed. After a full rerun, `git diff paper/generated` shows any difference from
the committed results.
"""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import mean, median, stdev

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('erm', 'mixup')
SEEDS = (1, 2, 3)
# Reference values from Zhang et al. (2018), Fig. 3(a): CIFAR-10, PreAct ResNet-18, 200 epochs.
PAPER = {'erm': 5.6, 'mixup': 4.2}
WORDS = ['One', 'Two', 'Three']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hash():
    # Same definition as mixup_repro.experiment.source_hash, without importing torch.
    files = sorted((ROOT/'mixup_repro').glob('*.py')) + [ROOT/'vendor/resnet.py']
    value = {p.relative_to(ROOT).as_posix(): sha256(p) for p in files}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_history(path):
    with open(path, newline='') as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


# ---------------------------------------------------------------- full rerun

def run(*args, stdout=None):
    print('>', 'python', *map(str, args), flush=True)
    subprocess.run([sys.executable, '-u', *map(str, args)], cwd=ROOT, check=True, stdout=stdout)


def full_rerun(data, output, save_every):
    main = output/'main'
    run('-m', 'unittest', 'discover', '-s', 'tests', '-v')
    common = ['--data', data, '--resume', '--download', '--save-every', save_every]
    if not (main/'frozen.json').exists():  # training is locked once frozen
        for seed in SEEDS:
            for method in METHODS:
                run('-m', 'mixup_repro.experiment', 'train', '--config', 'configs/proposal.json',
                    '--method', method, '--seed', seed, '--output', main, *common)
    run('-m', 'mixup_repro.experiment', 'train', '--config', 'configs/check_small_alpha.json',
        '--method', 'mixup', '--seed', 1, '--output', output/'checks'/'small_alpha', *common)
    if not (main/'frozen.json').exists():
        run('-m', 'mixup_repro.experiment', 'freeze', '--output', main)
    if not (main/'test_results.json').exists():  # the evaluator refuses to overwrite test metrics
        run('-m', 'mixup_repro.experiment', 'evaluate', '--download', '--data', data, '--output', main)
    run('-m', 'mixup_repro.summarize', '--output', main)

    results = output/'results'
    for folder in ('main', 'checks/small_alpha', 'diagnostics'):
        (results/folder).mkdir(parents=True, exist_ok=True)
    with open(results/'diagnostics'/'diagnose_seed1.txt', 'w', encoding='utf-8') as f:
        run('scripts/diagnose_mixup.py', '--output', main, '--data', data, '--seed', 1, '--download', stdout=f)
    for name in ('frozen.json', 'split.json', 'test_results.json', 'paired_results.csv',
                 'summary.md', 'learning_curves.png'):
        shutil.copy(main/name, results/'main')
    for method in METHODS:
        for seed in SEEDS:
            shutil.copy(main/f'{method}_seed{seed}'/f'{method}_seed{seed}_history.csv', results/'main')
    shutil.copy(output/'checks'/'small_alpha'/'mixup_seed1'/'mixup_seed1_history.csv',
                results/'checks'/'small_alpha')
    return results


# ---------------------------------------------------------------- checks and loading

def load(results):
    main = results/'main'
    frozen = json.loads((main/'frozen.json').read_text())
    tested = json.loads((main/'test_results.json').read_text())
    problems = []
    if tested['manifest_sha256'] != sha256(main/'frozen.json'):
        problems.append('test_results.json was not produced from this frozen.json')
    if frozen['source_hash'] != source_hash():
        problems.append('frozen.json was produced by different code than mixup_repro/ and vendor/resnet.py')
    frozen_sha = {(r['method'], r['seed']): r['sha256'] for r in frozen['runs']}
    runs, histories = {}, {}
    for r in tested['runs']:
        key = (r['method'], r['seed'])
        runs[key] = r
        histories[key] = h = read_history(main/f'{key[0]}_seed{key[1]}_history.csv')
        errors = [row['validation_error_pct'] for row in h]
        if frozen_sha.get(key) != r['sha256']:
            problems.append(f'{key}: tested checkpoint is not the frozen one')
        if [row['epoch'] for row in h] != list(range(1, frozen['config']['epochs']+1)):
            problems.append(f'{key}: history does not cover every epoch exactly once')
        if (min(errors), errors.index(min(errors))+1) != (r['best_validation_error_pct'], r['best_epoch']):
            problems.append(f'{key}: history disagrees with the selected checkpoint')
    if set(runs) != {(m, s) for m in METHODS for s in SEEDS}:
        problems.append('test results do not contain exactly the six planned runs')
    if problems:
        sys.exit('Results failed the provenance checks:\n  ' + '\n  '.join(problems))

    diagnostic = {}
    for line in (results/'diagnostics'/'diagnose_seed1.txt').read_text(encoding='utf-8').splitlines():
        cells = [c.strip() for c in line.strip('|').split('|')]
        if cells[0] in METHODS:
            diagnostic[cells[0]] = dict(zip(('epoch', 'logged', 'val', 'train', 'val_bn', 'train_bn'),
                                            map(float, cells[1:])))
    check = read_history(results/'checks'/'small_alpha'/'mixup_seed1_history.csv')
    return frozen, runs, histories, diagnostic, check


# ---------------------------------------------------------------- paper assets

def numbers_tex(frozen, runs, histories, diagnostic, check):
    test = {m: [runs[m, s]['error_pct'] for s in SEEDS] for m in METHODS}
    diffs = [e - x for e, x in zip(test['erm'], test['mixup'])]
    seconds = [row['compute_seconds'] for h in histories.values() for row in h]
    macros = {
        'ErmMean': f"{mean(test['erm']):.2f}", 'ErmSD': f"{stdev(test['erm']):.2f}",
        'MixMean': f"{mean(test['mixup']):.2f}", 'MixSD': f"{stdev(test['mixup']):.2f}",
        'DiffMean': f'{mean(diffs):.2f}', 'DiffSD': f'{stdev(diffs):.2f}',
        'MixWins': str(sum(d > 0 for d in diffs)),
        'PaperErm': f"{PAPER['erm']:.1f}", 'PaperMix': f"{PAPER['mixup']:.1f}",
        'PaperGain': f"{PAPER['erm']-PAPER['mixup']:.1f}",
        'GapToPaperErm': f"{mean(test['erm'])-PAPER['erm']:.2f}",
        'GapToPaperMix': f"{mean(test['mixup'])-PAPER['mixup']:.2f}",
        # Median epoch time x epochs: the sum would count epochs slowed by a second copy on the GPU.
        'EpochSeconds': f'{median(seconds):.0f}', 'GpuHours': f'{median(seconds)*len(seconds)/3600:.1f}',
        'PeakGpuMB': f"{max(row['peak_gpu_mb'] for h in histories.values() for row in h):.0f}",
        'SourceHash': frozen['source_hash'][:12],
        'ErmCleanTrain': f"{diagnostic['erm']['train']:.2f}", 'MixCleanTrain': f"{diagnostic['mixup']['train']:.2f}",
        'ErmValBN': f"{diagnostic['erm']['val_bn']:.2f}", 'MixValBN': f"{diagnostic['mixup']['val_bn']:.2f}",
        'MixBNGain': f"{diagnostic['mixup']['val']-diagnostic['mixup']['val_bn']:.2f}",
        'ErmBNGain': f"{diagnostic['erm']['val']-diagnostic['erm']['val_bn']:.2f}",
        'MixOneObjEpochOne': f"{histories['mixup', 1][0]['train_objective']:.2f}",
        'MixOneValEpochOne': f"{histories['mixup', 1][0]['validation_error_pct']:.2f}",
        'MixTwoValEpochOne': f"{histories['mixup', 2][0]['validation_error_pct']:.2f}",
        'MixOneValEpochSix': f"{histories['mixup', 1][5]['validation_error_pct']:.2f}",
        'ErmOneValEpochSix': f"{histories['erm', 1][5]['validation_error_pct']:.2f}",
        'CheckValEpochFive': f"{check[4]['validation_error_pct']:.2f}",
        'ErmOneValEpochFive': f"{histories['erm', 1][4]['validation_error_pct']:.2f}",
        'MixOneValEpochFive': f"{histories['mixup', 1][4]['validation_error_pct']:.2f}",
    }
    for i, s in enumerate(SEEDS):
        w = WORDS[i]
        macros[f'ErmTest{w}'] = f"{runs['erm', s]['error_pct']:.2f}"
        macros[f'MixTest{w}'] = f"{runs['mixup', s]['error_pct']:.2f}"
        macros[f'Diff{w}'] = f'{diffs[i]:+.2f}'
        macros[f'ErmVal{w}'] = f"{runs['erm', s]['best_validation_error_pct']:.2f}"
        macros[f'MixVal{w}'] = f"{runs['mixup', s]['best_validation_error_pct']:.2f}"
    lines = ['% Generated by scripts/reproduce.py. Do not edit by hand.']
    lines += [f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in macros.items()]
    return '\n'.join(lines) + '\n', test, diffs


def results_table_tex(runs, test, diffs):
    rows = [r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule',
            r'& \multicolumn{2}{c}{Test error (\%)} & ERM$-$mixup & mixup sel.\\',
            r'\cmidrule(lr){2-3}', r'Seed & ERM & mixup & (pp) & epoch\\', r'\midrule']
    for i, s in enumerate(SEEDS):
        rows.append(f"{s} & {test['erm'][i]:.2f} & {test['mixup'][i]:.2f} & ${diffs[i]:+.2f}$ & "
                    f"{runs['mixup', s]['best_epoch']}\\\\")
    rows += [r'\midrule',
             f"Mean (SD) & {mean(test['erm']):.2f} ({stdev(test['erm']):.2f}) & "
             f"{mean(test['mixup']):.2f} ({stdev(test['mixup']):.2f}) & "
             f"${mean(diffs):+.2f}$ ({stdev(diffs):.2f}) & \\\\",
             f"Paper, Fig.~3(a) & {PAPER['erm']:.1f} & {PAPER['mixup']:.1f} & "
             f"${PAPER['erm']-PAPER['mixup']:+.1f}$ & --\\\\",
             r'\bottomrule', r'\end{tabular}']
    return '\n'.join(rows) + '\n'


def diagnostic_table_tex(diagnostic):
    rows = [r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule',
            r'& \multicolumn{2}{c}{As trained} & \multicolumn{2}{c}{BN re-estimated}\\',
            r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}', r'Seed 1 & Val. & Clean train & Val. & Clean train\\',
            r'\midrule']
    for m in METHODS:
        d = diagnostic[m]
        rows.append(f"{'ERM' if m == 'erm' else 'mixup'} & {d['val']:.2f} & {d['train']:.2f} & "
                    f"{d['val_bn']:.2f} & {d['train_bn']:.2f}\\\\")
    rows += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(rows) + '\n'


def check_table_tex(histories, check):
    series = [('ERM', histories['erm', 1]), (r'mixup, $\alpha=0.001$', check),
              (r'mixup, $\alpha=1$', histories['mixup', 1])]
    rows = [r'\begin{tabular}{@{}lrrrrr@{}}', r'\toprule',
            r'Seed 1, epoch & 1 & 2 & 3 & 4 & 5\\', r'\midrule']
    for name, h in series:
        rows.append(name + ' & ' + ' & '.join(f"{row['validation_error_pct']:.2f}" for row in h[:5]) + r'\\')
    rows += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(rows) + '\n'


def figures(histories, test, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Validated two-slot categorical palette; line style and marker repeat the method for print.
    color = {'erm': '#2a78d6', 'mixup': '#eb6834'}
    style = {'erm': '-', 'mixup': '--'}
    marker = {'erm': 'o', 'mixup': 's'}
    name = {'erm': 'ERM', 'mixup': 'mixup'}
    plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.edgecolor': '#666666', 'pdf.fonttype': 42,
                         'axes.labelcolor': '#222222', 'xtick.color': '#444444', 'ytick.color': '#444444'})
    fig, (curves, dots) = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw={'width_ratios': [1.6, 1]})
    for m in METHODS:
        for s in SEEDS:
            h = histories[m, s]
            curves.plot([r['epoch'] for r in h], [r['validation_error_pct'] for r in h], style[m],
                        color=color[m], lw=1.2, alpha=.9, label=name[m] if s == 1 else None)
    # Label the mixup curves at their last epoch, nudged apart so the labels do not overlap.
    ends = sorted((histories['mixup', s][-1]['validation_error_pct'], s) for s in SEEDS)
    placed = []
    for value, s in ends:
        y = max(value, placed[-1] + 4.5) if placed else value
        placed.append(y)
        curves.annotate(f'mixup seed {s}', (50, value), xytext=(53, y), textcoords='data',
                        va='center', fontsize=7, color='#444444')
    for e in (25.5, 38.5):
        curves.axvline(e, color='#bbbbbb', lw=.8, zorder=0)
    curves.text(26, 76, 'lr$\\div$10', fontsize=7, color='#666666')
    curves.text(39, 76, 'lr$\\div$10', fontsize=7, color='#666666')
    curves.set(xlabel='Epoch', ylabel='Validation error (%)', xlim=(0, 68), ylim=(0, 85),
               xticks=[0, 10, 20, 30, 40, 50])
    curves.grid(axis='y', color='#e6e6e6', lw=.6)
    curves.legend(frameon=False, loc='center left', bbox_to_anchor=(0.5, 0.55))
    curves.set_title('(a) Validation error, all six runs', fontsize=8, loc='left')

    for i, s in enumerate(SEEDS):
        dots.plot([0, 1], [test['erm'][i], test['mixup'][i]], color='#999999', lw=.8, zorder=1)
        dots.annotate(f'seed {s}', (1, test['mixup'][i]), xytext=(6, 0), textcoords='offset points',
                      va='center', fontsize=7, color='#444444')
    for j, m in enumerate(METHODS):
        dots.scatter([j]*len(SEEDS), test[m], s=28, marker=marker[m], color=color[m],
                     edgecolor='white', linewidth=1, zorder=2)
        dots.plot([j-.25, j+.25], [PAPER[m]]*2, ':', color=color[m], lw=1.4)
        dots.annotate(f'paper {PAPER[m]}', (j-.25, PAPER[m]), xytext=(0, -9), textcoords='offset points',
                      fontsize=7, color='#444444')
    dots.set(xticks=[0, 1], xticklabels=[name[m] for m in METHODS], xlim=(-.5, 1.7), ylim=(0, 20),
             ylabel='Test error (%)')
    dots.grid(axis='y', color='#e6e6e6', lw=.6)
    dots.set_title('(b) Test error per seed vs. paper', fontsize=8, loc='left')
    fig.tight_layout()
    fig.savefig(out/'results.pdf')
    fig.savefig(out/'results.png', dpi=200)
    plt.close(fig)


def build_assets(results, out):
    frozen, runs, histories, diagnostic, check = load(results)
    out.mkdir(parents=True, exist_ok=True)
    numbers, test, diffs = numbers_tex(frozen, runs, histories, diagnostic, check)
    (out/'numbers.tex').write_text(numbers, encoding='utf-8')
    (out/'results_table.tex').write_text(results_table_tex(runs, test, diffs), encoding='utf-8')
    (out/'diagnostic_table.tex').write_text(diagnostic_table_tex(diagnostic), encoding='utf-8')
    (out/'check_table.tex').write_text(check_table_tex(histories, check), encoding='utf-8')
    figures(histories, test, out)
    print(f'Provenance checks passed. Paper assets written to {out}')
    print(f"Mean test error: ERM {mean(test['erm']):.3f}%, mixup {mean(test['mixup']):.3f}%; "
          f'mean ERM-minus-mixup {mean(diffs):+.3f} pp (SD {stdev(diffs):.3f}).')


def build_pdf(paper):
    tools = [shutil.which(t) for t in ('pdflatex', 'bibtex')]
    if not all(tools):
        print('pdflatex/bibtex not found; skipped building paper/main.pdf.')
        return
    pdflatex = [tools[0], '-interaction=nonstopmode', '-halt-on-error', 'main.tex']
    for command in (pdflatex, [tools[1], 'main'], pdflatex, pdflatex):
        done = subprocess.run(command, cwd=paper, capture_output=True, text=True)
        if done.returncode:
            sys.exit(f'{Path(command[0]).stem} failed; see paper/main.log or main.blg\n{done.stdout[-2000:]}')
    print(f'Built {paper/"main.pdf"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--from-results', type=Path, help='rebuild from a results folder (e.g. results)')
    parser.add_argument('--data', type=Path, help='CIFAR-10 folder for a full rerun')
    parser.add_argument('--output', type=Path, help='new output folder for a full rerun (use Drive on Colab)')
    parser.add_argument('--save-every', type=int, default=5)
    parser.add_argument('--assets', type=Path, default=ROOT/'paper'/'generated')
    parser.add_argument('--no-pdf', action='store_true', help='do not build paper/main.pdf')
    args = parser.parse_args()
    if args.from_results:
        results = args.from_results
    elif args.data and args.output:
        results = full_rerun(args.data, args.output, args.save_every)
    else:
        parser.error('use --from-results DIR, or --data DIR --output DIR for a full rerun')
    build_assets(results, args.assets)
    if not args.no_pdf:
        build_pdf(ROOT/'paper')


if __name__ == '__main__':
    main()
