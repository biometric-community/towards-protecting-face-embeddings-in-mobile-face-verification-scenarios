# Towards Protecting Face Embeddings in Mobile Face Verification Scenarios

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Lab twin of Hahn & Marcel (TBIOM 2022) **PolyProtect**: user-specific multivariate polynomial mapping of 128-D face embeddings, evaluated on LFW with CelebA-trained CosFace ResNet-18.

**Repository:** https://github.com/biometric-community/towards-protecting-face-embeddings-in-mobile-face-verification-scenarios

## Setup

```bash
bash scripts/setup_env.sh
bash scripts/smoke.sh
bash scripts/train_full.sh
```

See [REPORT.md](./REPORT.md) and [DEVIATIONS.md](./DEVIATIONS.md).

## License

[CC BY 4.0](./LICENSE)
