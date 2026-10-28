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
