# Upstream code provenance

Repository: https://github.com/facebookresearch/mixup-cifar10
Pinned commit: eaff31ab397a90fbc0a4aac71fb5311144b3608b
Retrieved: 2026-10-18

- `resnet.py`: unmodified copy of upstream `models/resnet.py`. Used directly.
- `train.py`: unmodified reference copy of upstream training script. NOT executed by this project.
- `LICENSE`: upstream CC BY-NC 4.0 license, retained verbatim.
- `LICENSE-pytorch-cifar`: upstream underlying CIFAR code license, retained verbatim.

The local training loop reimplements the batching and weighted-loss method from the paper and
authors' script, adding validation separation, paired randomness, resumable state and provenance.
Upstream `ResNet18()` uses `PreActBlock`; the name alone must not be confused with torchvision's
post-activation ResNet18. The old shortcut implementation is preserved for source fidelity.

Paper: Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz.
mixup: Beyond Empirical Risk Minimization. ICLR 2018. https://arxiv.org/abs/1710.09412

Retain attribution and the included upstream licenses when sharing this coursework repository.
No new permissive license is asserted over the upstream code.
