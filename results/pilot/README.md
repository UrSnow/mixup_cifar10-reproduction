# Pilot outputs (not reproduction results)

Per-epoch histories from the 2026-10-28 Colab pilots: seed 1, 2 epochs, 1,024 training and 256
validation images, Tesla T4.

- `before_fixes/`: first pilot, code at commit 658837e. Downloaded from Drive, where both files were
  called `history.csv`, and renamed by method.
- `after_fixes/`: third pilot, code at commit fd292f6. File names written by the code.
  (The second pilot's outputs were discarded before being inspected.)

Apart from compute time, the two folders contain identical values, which shows that training is
deterministic across Colab sessions.

These files only show that the pipeline runs, how long it takes and how much GPU memory it needs.
They say nothing about whether mixup lowers test error. See `docs/EXPERIMENT_LOG.md` (2026-10-28).
