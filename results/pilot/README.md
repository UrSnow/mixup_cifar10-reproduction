# Pilot outputs (not reproduction results)

Per-epoch histories from the 2026-10-28 Colab pilot: seed 1, 2 epochs, 1,024 training and 256
validation images, Tesla T4. They were downloaded from the run folders on Drive, where both files
were called `history.csv`, and renamed by method.

- `erm_history.csv`: ERM
- `mixup_history.csv`: mixup

These files only show that the pipeline runs, how long it takes and how much GPU memory it needs.
They say nothing about whether mixup lowers test error. See `docs/EXPERIMENT_LOG.md` (2026-10-28).
