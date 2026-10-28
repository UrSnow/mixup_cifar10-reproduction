# mixup reproduction on CIFAR-10

A reduced-scale reproduction of one claim from Zhang et al., *mixup: Beyond Empirical Risk
Minimization* (ICLR 2018): **Section 3.2, the table in Figure 3(a)**, where a PreAct ResNet-18 on
CIFAR-10 reaches 4.2% test error with mixup (alpha = 1) against 5.6% with ordinary training (ERM).

This reproduction trains for 50 epochs instead of 200, holds out 5,000 training images for
validation, uses three paired seeds, and opens the test set once, after a freeze.
The write-up is in [`paper/`](paper/).

## Result

The improvement did not reproduce at this budget.

| Seed | ERM test error (%) | mixup test error (%) | ERM minus mixup (pp) |
|---:|---:|---:|---:|
| 1 | 7.71 | 17.77 | -10.06 |
| 2 | 8.71 | 9.72 | -1.01 |
| 3 | 7.19 | 6.94 | +0.25 |
| Mean (SD) | 7.87 (0.77) | 11.48 (5.62) | -3.61 (5.62) |
| Paper | 5.6 | 4.2 | +1.4 |

mixup won 1 of 3 pairs. A diagnosis on seed 1 shows that mixup underfits at 50 epochs (15.72% error
on clean training images, against 0.46% for ERM), and a control run with alpha = 0.001 behaves like
ERM, so the gap is not an implementation bug. Details, interpretation and limits:
[`paper/main.tex`](paper/main.tex) and [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md).

## Reproduce every number and figure

Set up the environment once (conda):

```bash
conda env create -f environment.yml
conda activate mixup-repro
```

Then, from a clean checkout:

```bash
python scripts/reproduce.py --from-results results
```

This takes a few seconds on a CPU and does not need PyTorch. It:

1. checks the committed results: the test results belong to the frozen manifest, each tested
   checkpoint is the frozen one, each history agrees with its selected checkpoint, and the code hash
   in the manifest matches `mixup_repro/` and `vendor/resnet.py` in this checkout;
2. regenerates every number, table and figure in the write-up into `paper/generated/`;
3. builds `paper/main.pdf` if `pdflatex` and `bibtex` are installed (add `--no-pdf` to skip).

To rerun the whole experiment instead (a CUDA GPU; about 6 GPU-hours on a Colab T4):

```bash
python scripts/reproduce.py --data /content/cifar10 --output /content/drive/MyDrive/mixup-rerun
```

This runs the tests, the six main runs, the small-alpha check, the freeze, the single test
evaluation, the summary and the seed 1 diagnostic, and then rebuilds the paper from the new
outputs. If Colab disconnects, run the same command again and it resumes. Afterwards,
`git diff paper/generated` shows any difference from the committed results in the `.tex` files
(the figure files can also differ byte for byte with another matplotlib version). Training is
deterministic, but bit-identical results are only expected on the same GPU and software
(Tesla T4, torch 2.11.0+cu130).

## What is in the repository

| Path | Contents |
|---|---|
| `mixup_repro/` | Training, freeze, test evaluation (`experiment.py`), loss and split helpers (`core.py`), summary (`summarize.py`) |
| `vendor/` | The authors' `resnet.py`, unmodified, plus their `train.py` for reference only; see `vendor/PROVENANCE.md` |
| `configs/` | `proposal.json` (main protocol) and `check_small_alpha.json` (code-path check) |
| `scripts/` | `reproduce.py` (the command above), `diagnose_mixup.py`, `package_colab.py` |
| `notebooks/01_colab_pilot.ipynb` | The Colab notebook used for every run, with its outputs |
| `tests/` | Leakage, loss, architecture, resume and selection tests: `python -m unittest discover -s tests -v` |
| `results/main/` | Frozen manifest, test results, summary and per-epoch histories of the six runs |
| `results/checks/`, `results/diagnostics/` | The small-alpha check history and the seed 1 diagnostic output |
| `results/pilot/` | Pilot outputs (not reproduction results) |
| `docs/` | `PROTOCOL.md` (rules and deviations), `EXPERIMENT_LOG.md` (every attempt and decision), `IMPLEMENTATION_STATUS.md` |
| `paper/` | The write-up (IEEE conference format); `paper/generated/` is written by `reproduce.py` |

## Method and sources

The model is the authors' `ResNet18` (PreAct blocks) from
[facebookresearch/mixup-cifar10](https://github.com/facebookresearch/mixup-cifar10), copied
unmodified at a pinned commit. The training loop is reimplemented following the authors'
`train.py` (one lambda per minibatch, a permutation of the same minibatch, weighted cross-entropy),
with a validation split, paired randomness across methods, resumable checkpoints and provenance
records added. The protocol and every difference from the paper are in
[`docs/PROTOCOL.md`](docs/PROTOCOL.md).

## Running on Colab step by step

The notebook `notebooks/01_colab_pilot.ipynb` runs the same commands one at a time. Put
`mixup-colab-project.zip` (built with `python scripts/package_colab.py`) in the top level of
Google Drive and open the notebook from VS Code connected to Colab or in the Colab browser
interface. Training, freezing and testing are off by default and need explicit switches.

The individual commands, for a prepared environment:

```bash
python -m unittest discover -s tests -v
python -m mixup_repro.experiment train --method erm --seed 1 --data /content/cifar10 --download --resume --output /content/drive/MyDrive/mixup-reproduction-v1/main
# ...mixup seed 1, then both methods for seeds 2 and 3
python -m mixup_repro.experiment freeze --output /content/drive/MyDrive/mixup-reproduction-v1/main
python -m mixup_repro.experiment evaluate --download --data /content/cifar10 --output /content/drive/MyDrive/mixup-reproduction-v1/main
python -m mixup_repro.summarize --output /content/drive/MyDrive/mixup-reproduction-v1/main
```

Checkpoints (`checkpoint.pt`, about 135 MB) are written every 5 epochs and at the final epoch, and
`--resume` reruns any unsaved epochs identically. One run per output folder at a time. After the
freeze, training is locked, and the evaluator refuses to overwrite existing test metrics.

## Licence

`vendor/` keeps the upstream licences (CC BY-NC 4.0 and the pytorch-cifar licence). No new licence
is asserted over the upstream code.
