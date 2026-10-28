# Main experiment outputs (frozen, tested once)

Outputs of the six main runs (ERM and mixup alpha 1, seeds 1-3, 50 epochs, Tesla T4), copied from
Drive (`mixup-reproduction-v1/main/`) after the freeze and the single test evaluation. Checkpoints
(about 135 MB each) are not included.

- `frozen.json`: freeze manifest (config, source hash, split hash, and the SHA-256 of each
  checkpoint that was tested).
- `test_results.json`: test loss and error for each validation-selected checkpoint, the runtime,
  and the SHA-256 of `frozen.json`.
- `summary.md`, `paired_results.csv`, `learning_curves.png`: written by `mixup_repro.summarize`.
- `<method>_seed<seed>_history.csv`: per-epoch training objective, validation loss and error,
  compute time and peak GPU memory.
- `split.json`: the 45,000/5,000 training/validation split.

Checked on 2026-10-29: the SHA-256 of `frozen.json` matches `test_results.json`, the checkpoint
hashes match between the two, each history's best validation epoch matches the tested checkpoint,
`paired_results.csv` matches `test_results.json`, and the source hash in `frozen.json` matches the
code in `mixup_repro/` and `vendor/resnet.py` in this repository.

See `docs/EXPERIMENT_LOG.md` (2026-10-28 to 29) for the interpretation and its limits.
