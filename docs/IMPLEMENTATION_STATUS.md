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
The notebook contains no executed outputs. Synthetic fixtures live only in temporary test
directories; their scores are not CIFAR-10 findings.

Re-verified on 2026-10-24 after the review fixes (evaluate `--download`, Drive-based notebook
loading): all 8 tests pass with torch 2.14.1+cpu / torchvision 0.29.1+cpu, Python 3.12.

## Not yet verified

- Colab GPU execution, CUDA determinism compatibility and Google Drive checkpoint I/O.
  The first attempt (2026-10-18, via VS Code) reached a T4 but stopped at the upload cell.
- Whether replaced checkpoints accumulate in Drive's trash and consume quota.
- CIFAR-10 download and pilot timing/memory on the assigned Colab GPU.
- Full training budget feasibility, six completed runs or any scientific reproduction result.

First milestone: run the notebook's two-epoch pilot for both methods. Review its timing,
GPU memory and checkpoints before enabling a full run. The pilot uses only development data.

## Local setup note

Windows sandbox application control initially blocked the PyTorch DLL. The same isolated
CPU tests passed outside the sandbox. This was an execution-environment restriction,
not a failed experiment or a change to the research protocol.
