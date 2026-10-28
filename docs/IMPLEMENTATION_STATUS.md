# Implementation status

Verified locally on 2026-10-18 using Python 3.12 and an isolated CPU environment.
The exact installed packages are recorded in `local-test-environment.txt`.

## Completed checks

`python -m unittest discover -s tests -v` passed all 8 tests:

1. Learning-rate boundaries and earlier-epoch selection on validation ties.
2. Evaluation requires a frozen manifest, loads validation-selected weights, rejects changed
   checkpoints and refuses to overwrite existing test metrics.
3. Freeze rejects incomplete runs and blocks further training after completion.
4. Weighted cross-entropy loss and gradients match explicit soft targets.
5. The upstream model contains the expected PreActBlocks, initializes reproducibly, and
   supports a real 32x32 forward/backward pass with finite gradients.
6. Interrupted/resumed synthetic training reproduces uninterrupted final and selected weights;
   paired methods share initialization and changed configs cannot resume.
7. The stratified 45,000/5,000 split is reproducible, disjoint and complete.
8. Synthetic measured errors yield the expected means, sample SDs and paired differences;
   plotting creates an output image.

All 6 project/test/packaging Python files and all 9 notebook code cells passed syntax checks.
Synthetic fixtures live only in temporary test directories; their scores are not CIFAR-10 findings.
The committed notebook keeps the outputs of the 2026-10-28 Colab pilot (environment, Drive mount,
config and ERM pilot history) as a record of that run.

Re-verified on 2026-10-24 after the review fixes (evaluate `--download`, Drive-based notebook
loading): all 8 tests pass with torch 2.14.1+cpu / torchvision 0.29.1+cpu, Python 3.12.

## Verified on Colab (2026-10-28 pilot)

On a Tesla T4 with Python 3.13.16 and torch 2.11.0+cu130:

- All 8 tests passed (the test cell completed without an exception).
- CIFAR-10 download, Drive mounting and checkpoint writing to Drive worked.
- Deterministic mode ran on CUDA without error.
- Pilot timing was about 1.45 s per 8-batch epoch; peak GPU memory was about 850 MB.

See `EXPERIMENT_LOG.md` for details and the early mixup loss spike.

The commit c9f3064 changes (per-run history file names, no nested `pilot` folder) compile but
have not been run through the tests yet; the pilot rerun on Colab will do that.

## Not yet verified

- Whether replaced checkpoints accumulate in Drive's trash and consume quota.
- Full-epoch time on the complete 45,000-image training split.
- That validation error falls well below chance in the first full epochs for both methods.
- Six completed runs or any scientific reproduction result.

Next milestone: rerun the pilot with the rebuilt ZIP, then start the first main run and check
its first epochs before continuing. The pilot uses only development data.

## Local setup note

Windows sandbox application control initially blocked the PyTorch DLL. The same isolated
CPU tests passed outside the sandbox. This was an execution-environment restriction,
not a failed experiment or a change to the research protocol.
