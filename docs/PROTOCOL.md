# Experiment protocol

Target: Section 3.2, Figure 3(a), PreAct ResNet-18 on CIFAR-10 in Zhang et al. (ICLR 2018).
Published errors 5.6% ERM / 4.2% mixup are reference values, never measured outputs here.

The submitted proposal is the authority for this implementation. The defaults are:

| Item | Decision |
|---|---|
| Dataset | CIFAR-10, 45,000 train / 5,000 validation / 10,000 test |
| Split | Stratified, 500 validation images per class, split seed 2026 |
| Model | Official repository ResNet18 constructor (PreActBlock), unchanged |
| Augmentation | Random crop 32 with padding 4, horizontal flip, normalization from authors' code |
| Mixing | One lambda per batch from Beta(1,1), pair with a random batch permutation |
| Training | 50 epochs, batch 128, SGD, momentum 0.9, weight decay 0.0001 |
| Learning rate | 0.1 in epochs 1–25, 0.01 in 26–38, 0.001 in 39–50 |
| Repetitions | Seeds 1, 2, 3, same initial weights within a pair |
| Selection | Strictly lower validation error wins; ties keep the earlier epoch |
| Evaluation | After all six runs and protocol freeze, test only the selected checkpoints |
| Summary | Each test error, sample SD, means, paired ERM-minus-mixup error in percentage points |

## Differences from the original paper and code

- 50 rather than 200 epochs; milestones compressed and defined explicitly above.
- 45,000 training examples, reserving 5,000 for validation instead of training on all 50,000.
- Selection uses validation only. The old authors' training script evaluates test data during training.
- Three specified paired seeds; epoch-level timing and environment fingerprints are recorded.
- No CIFAR-100, other networks, corrupted labels, hyperparameter sweep or new method.
- No mixed-input accuracy is presented as ordinary training accuracy. The logged training loss uses
  different objectives across methods. Validation error is directly comparable.
- No automatic mixed precision or nondeterministic cuDNN benchmarking. A future speed change must
  be documented, applied to both conditions, and fixed before final testing.

The official model implementation is retained exactly, including its shortcut behavior.
Do not silently swap it for torchvision ResNet-18 or another PreActResNet implementation.

## Pilot and protocol changes

The pilot uses 1,024 training and 256 validation examples, for two epochs per method.
It never loads the test split. Its outputs are timing/debugging evidence, not claim evidence.
Small-sample timing excludes checkpoint I/O and may underestimate full-run cost. Inspect GPU memory
and measure a full epoch before committing to the six-run budget. Save checkpoints to Drive.

If the budget must change, create a revised config and a NEW output directory; record the reason
in EXPERIMENT_LOG.md before any test evaluation. Preserve previous runs. Do not reduce the seed count
based on observed gains. A failed reproduction at reduced scale does not alone refute the original.

## Resume and provenance

Each checkpoint atomically contains the last model, optimizer, selected best model, complete epoch
history, initial-weight hash, config, split hash, code hash and environment records. Checkpoints are
written every 5 epochs (`--save-every`) and at the final epoch, to limit ~135 MB writes to Drive;
the history CSV is still written every epoch. Resume starts after the last saved epoch. Data shuffle,
augmentation and mixup streams are reset from the paired seed and epoch, so unsaved and interrupted
epochs are rerun identically. Equality across different GPUs/software is not promised.
Runs use one process at a time per output folder. Do not edit code or config halfway through a run.

The frozen manifest verifies completion and identical pair initialization, and hashes all checkpoints.
Training is locked once frozen. Existing test metrics cannot be overwritten by the evaluator.
Only load trusted local checkpoints; checkpoint loading uses Python serialization.
