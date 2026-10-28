# Experiment log

## 2026-10-18: implementation started

- Read submitted proposal and inspected pinned upstream source.
- Implemented the proposed protocol, explicit test gate and epoch-boundary resumption.
- Chose split seed 2026 and paired training seeds 1, 2, 3 as implementation details.
- No CIFAR-10 training, pilot or test results have been produced yet.
- Software tests use synthetic inputs and are not scientific evidence for mixup's claim.
- All 8 local CPU checks passed, including model gradients, resume equivalence and evaluation safeguards.
- Windows sandbox initially blocked the PyTorch DLL; tests passed outside the sandbox in the isolated environment.
- Colab execution and pilot feasibility remain unverified; see IMPLEMENTATION_STATUS.md.

## 2026-10-18: first Colab attempt (failed before the pilot)

- Ran the notebook from VS Code connected to a Colab runtime: Tesla T4, Python 3.13.15,
  torch 2.11.0+cu130, torchvision 0.26.0+cu130 (local checks used Python 3.12 / torch 2.14.1+cpu).
- Cell 1 (environment/GPU check) succeeded. From the upload cell onwards every cell reported
  "notebook controller is DISPOSED". The likely cause is that `files.upload()` / `drive.mount()`
  need the Colab browser interface. Nothing past cell 1 ran: no tests, data, pilot or Drive output.
- No protocol change. No CIFAR-10 data was loaded.

## 2026-10-24: review fixes (before any training)

- Notebook now mounts Drive first and loads the project ZIP from `MyDrive`, falling back to upload;
  it should be run from the Colab browser interface. Failed-run outputs were cleared.
- `evaluate` now honours `--download`, so a fresh session after freezing can restore the archive.
  Test images are still only opened after the freeze and checkpoint-hash checks.
- README run folder unified with the notebook (`mixup-reproduction-v1`).
- Code changes alter the source hash; no runs exist yet, so nothing needs to be restarted.
- Watch for: Drive trash filling with replaced ~135 MB checkpoints; CUDA deterministic-mode
  errors or slowdown; full-epoch time (the pilot is only 8 batches per epoch).

## 2026-10-28: Colab pilot (feasibility only, not evidence)

- Purpose: check that the pipeline runs on a Colab GPU and measure timing and memory before
  committing to the six-run budget. No test data was loaded.
- Code: the ZIP built on 2026-10-24, identical to commit 658837e. Config `configs/proposal.json`,
  pilot override 2 epochs, no LR milestones, seed 1, 1,024 train / 256 validation images.
- Runtime: VS Code connected to a Colab runtime, Tesla T4, Python 3.13.16, torch 2.11.0+cu130,
  torchvision 0.26.0+cu130. Unlike the 2026-10-18 attempt, Drive mounting worked from VS Code.
  (Corrected: an earlier version of this entry said the Colab browser interface was used.)
- The test cell finished without an exception (`check=True`), so all 8 tests passed on the Colab
  runtime. CIFAR-10 downloaded, Drive mounted, and deterministic mode raised no CUDA error.
- Results (copies in `results/pilot/before_fixes/`; ERM identity confirmed by the notebook's
  printed history and by the later rerun):

  | Method | Epoch | Train objective | Val loss | Val error (%) | Compute (s) | Peak GPU (MB) |
  |---|---:|---:|---:|---:|---:|---:|
  | ERM | 1 | 2.483 | 2.291 | 87.50 | 4.40 | 850 |
  | ERM | 2 | 2.254 | 3.553 | 82.42 | 1.44 | 850 |
  | mixup | 1 | 2.719 | 2.352 | 88.67 | 1.58 | 853 |
  | mixup | 2 | 9.338 | 9.17e9 | 89.45 | 1.46 | 853 |

- Timing: about 1.45 s per 8-batch epoch, including validation. ERM epoch 1 includes GPU warm-up.
  Rough extrapolation is 30-60 s per full 352-batch epoch, so 25-50 min per run. This is unverified:
  the first epoch of the first main run gives the real figure.
- Memory: about 850 MB peak, a small fraction of the T4's 15 GB. Batch 128 is not memory-limited.
- Unexpected behaviour:
  - mixup epoch 2: training objective 9.34 and validation loss 9.2e9 at chance-level error.
    Likely cause: only 16 SGD steps at lr 0.1 with no warm-up (early loss spikes are common), and
    eval-mode BatchNorm running statistics that are still far from the true activation statistics
    after 16 updates, producing a few extreme logits. The mixup code matches upstream, so this
    is not believed to be a bug, but it is not yet ruled out.
  - Wall-clock time was much longer than training compute. It was dominated by the CIFAR-10
    download, four ~135 MB checkpoint writes to Drive, and start-up. Colab did not display the
    output of the subprocess commands, so progress was not visible while it ran.
  - A nested `pilot/pilot` output folder appeared on Drive. The code appends `pilot` to `--output`,
    so an output path already ending in `pilot` was nested again.
- Changes after the pilot, before any main run (commit c9f3064): history files are named per run
  (`<method>_seed<seed>_history.csv`), and `pilot` is no longer appended when the output path
  already ends in `pilot`. This changes the source hash, so the pilot is rerun with the new ZIP and
  the old pilot folder is set aside.
- Interpretation: the pipeline is feasible on a free T4. The pilot says nothing about mixup's effect.
- Check on the first main runs: after epoch 1, validation error should be well below 90% with a
  finite validation loss. If either method stays near chance with an extreme validation loss after
  a few full epochs, stop and investigate before continuing.
- Next action: rerun the pilot with the new ZIP, finalise any remaining pre-run code changes, then
  start `erm` seed 1.

## 2026-10-28: final code changes before the main runs

- Purpose: settle all code changes before `erm` seed 1 starts. After that, any change to
  `mixup_repro/` changes the source hash, which blocks resuming and freezing existing runs.
- Checkpoints are now written every 5 epochs and at the final epoch (`--save-every`, default 5)
  instead of every epoch. Reason: each checkpoint is ~135 MB, and 300 writes to Drive (~40 GB) risk
  slow sessions and, if Drive keeps replaced files in the trash, exceeding the 15 GB quota.
  A disconnect now loses up to 4 epochs. Resume reruns them with the same per-epoch seeds, so the
  result is unchanged. `--save-every` is not stored in the run metadata, so it can differ between
  sessions without blocking resume. Training, selection and evaluation rules are unchanged.
- `summary.md` now also reports a per-seed table and the sample SD of the paired ERM-minus-mixup
  differences. Pairing removes seed-level variation, so this is the relevant spread for the claim.
  Reporting only; no protocol change.
- The notebook's `run()` now streams subprocess output, so test results and epoch lines appear in
  Colab while they run. Notebook only; it is not part of the source hash.
- Considered and rejected: also testing the last-epoch weights. The submitted proposal says only
  validation-selected checkpoints are tested. Each run's selected epoch is recorded instead.
- Tests: 9 pass locally (one new test interrupts a run between checkpoints and resumes it).
- Next action: rebuild and upload the ZIP, rerun the pilot on Colab, then start `erm` seed 1.

## 2026-10-28: pilot reruns with the pre-run code

- Purpose: check the pre-run changes on Colab (tests, `--save-every`, history file names, streamed
  output) and whether results reproduce across sessions.
- Code: commit fd292f6 for `mixup_repro/`, with the notebook restored in 6c993c9. An editor save had
  overwritten the notebook changes before fd292f6 was committed; 6c993c9 restored them.
- Runtime: VS Code connected to a Colab runtime, Tesla T4, Python 3.13.16, torch 2.11.0+cu130.
- Second pilot run: completed, but its notebook outputs were cleared and its pilot folder on Drive
  was deleted before the results were inspected. Nothing from it was recorded or used.
- Third pilot run (copies in `results/pilot/after_fixes/`):
  - All 9 tests passed on Colab, with output now visible in the notebook.
  - `[checkpoint saved]` appeared only at epoch 2 of each 2-epoch run, as expected with
    `--save-every 5`. History files were written as `<method>_seed1_history.csv` in a single
    `pilot/` folder.
  - Every logged training and validation value (train objective, validation loss and error, peak
    GPU memory) is bit-for-bit identical to the first pilot, for both methods. Only compute time
    differs (ERM epoch 1: 2.09 s versus 4.40 s).
- Interpretation: training is deterministic across Colab sessions on the same GPU and software,
  so resume after a disconnect and the ERM/mixup pairing behave as designed. The mixup epoch 2 loss
  spike reproduced exactly, so it is a deterministic property of those 16 steps, not chance. The
  explanation in the first pilot entry is unchanged and still to be confirmed at full scale.
- Still open: whether replaced checkpoints accumulate in Drive's trash.
- Next action: start `erm` seed 1 and check the first full epochs (time, validation error).

## Entry template

Date / run identifier:
Question or purpose:
Config and source commit:
Hardware / runtime:
What was changed and why (before test access):
Observed outputs and saved paths:
Unexpected behaviour / failed attempts:
Interpretation and limits:
Next action:
