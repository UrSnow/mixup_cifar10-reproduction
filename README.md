# mixup reproduction on CIFAR-10

This project tests whether mixup lowers CIFAR-10 test error compared with ordinary training
under the submitted reduced-scale protocol. It implements Figure 3(a)'s PreAct ResNet-18 comparison.
It contains no claimed reproduction results yet.

Start with **`notebooks/01_colab_pilot.ipynb`**. Open it in Colab in the browser (not via VS Code)
and choose a GPU runtime. Put `mixup-colab-project.zip` in the top level of your Google Drive first.
The notebook first runs protocol checks and a small pilot. Full experiments and testing require
explicit switches, so running all cells does not silently launch six training runs or inspect test data.

## Files to read first

- `configs/proposal.json`: settings from the submitted proposal plus explicit seed choices.
- `docs/PROTOCOL.md`: experimental rules, deviations, checkpoint selection and limitations.
- `docs/EXPERIMENT_LOG.md`: keep an honest record of attempts and decisions.
- `docs/IMPLEMENTATION_STATUS.md`: checks completed and what still needs verification on Colab.
- `vendor/PROVENANCE.md`: source and license attribution.
- `mixup_repro/experiment.py`: pilot, training, freeze and evaluation.
- `tests/test_protocol.py`: leakage, loss, architecture, resume and selection checks.

## Environment

Python 3.12 is used for local checks. Colab supplies torch/torchvision with CUDA support.
Do not replace that GPU build with the CPU wheels used for local testing.
Install other missing requirements as needed; each checkpoint records actual package versions.
The broad requirements file is a compatibility declaration, not an exact environment lock.
Local verification versions are recorded separately in `docs/local-test-environment.txt`.
After a reviewed code/config change, rebuild the upload ZIP with `python scripts/package_colab.py`.
Use a new output directory for a revised protocol; do not resume old runs with changed code.

From this directory, local setup (optional, full training should use Colab):

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m unittest discover -s tests -v
```

## Commands (Colab or a prepared Python environment)

The output folder should be on Google Drive for persistence; dataset files can stay in `/content`.

```bash
# 1. Verify software on synthetic inputs (no CIFAR download).
python -m unittest discover -s tests -v

# 2. Small pilot only. Does not load/evaluate CIFAR test images.
python -m mixup_repro.experiment pilot --download --data /content/cifar10 --output /content/drive/MyDrive/mixup-reproduction-v1 --resume

# 3. After reviewing pilot feasibility, run each seed/method.
python -m mixup_repro.experiment train --method erm --seed 1 --data /content/cifar10 --output /content/drive/MyDrive/mixup-reproduction-v1/main --resume
python -m mixup_repro.experiment train --method mixup --seed 1 --data /content/cifar10 --output /content/drive/MyDrive/mixup-reproduction-v1/main --resume
# Repeat for seeds 2 and 3. Add --download after a new Colab session if the dataset is gone.

# 4. Freeze only when all six runs are complete and decisions are final.
python -m mixup_repro.experiment freeze --output /content/drive/MyDrive/mixup-reproduction-v1/main

# 5. Evaluate selected checkpoints, then summarize measured results.
python -m mixup_repro.experiment evaluate --download --data /content/cifar10 --output /content/drive/MyDrive/mixup-reproduction-v1/main
python -m mixup_repro.summarize --output /content/drive/MyDrive/mixup-reproduction-v1/main
```

The default device is CUDA and commands fail clearly if no GPU is available. `--device cpu` is
for debugging only. One run per output folder at a time. `--resume` restores an existing run or
starts it if absent. It does not extend the configured budget. After freezing, training is blocked.
`checkpoint.pt` is written every 5 epochs and at the final epoch (`--save-every N` changes this);
after a disconnect, `--resume` reruns the epochs since the last checkpoint with the same seeds.

The standard CIFAR download archive includes test files. Development commands only instantiate
the training split; downloaded test images are not loaded or evaluated until the explicit evaluator.

## Outputs

- `split.json`: fixed train/validation indices, label hash and split seed.
- `<method>_seed<seed>/checkpoint.pt`: latest full state plus validation-selected best weights.
- `<method>_seed<seed>/<method>_seed<seed>_history.csv`: training objective, validation metrics, compute time and peak GPU memory.
- `frozen.json`: completion checks and hashes of all six selected-run containers.
- `test_results.json`, `paired_results.csv`, `summary.md`, `learning_curves.png`: final measured evidence.

Training objectives differ between methods; compare validation/test errors for the performance claim.
Do not report the pilot or software-test accuracies as reproduction results.

## Version control and submission

Commit code/config/log updates as you work. Large datasets, checkpoints and local PDFs are ignored.
After final evaluation, deliberately add a compact copy of metrics/figures to a `results/` directory
for submission, keeping provenance. No public GitHub upload is performed by this setup.
