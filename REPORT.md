# REPORT — Towards Protecting Face Embeddings (PolyProtect)

**Protocol:** `full`

Lab twin: CelebA-trained CosFace ResNet-18 (128-D) + PolyProtect multivariate
polynomials (m=5, overlap=4); LFW verification,
unlinkability and irreversibility proxies.

## Training

- Epochs: 8
- Train images: 141610
- Identities: 7124
- Emb dim: 128
- Elapsed (s): 1078.2

![Training](outputs/figures/full/train_loss.svg)

## LFW verification

| Mode | Accuracy | EER | TAR@FAR1e-2 |
|------|----------|-----|-------------|
| baseline | 72.98% | 27.02% | 16.97% |
| polyprotect | 95.02% | 4.98% | 62.27% |

![Verification](outputs/figures/full/verify_metrics.svg)

## Unlinkability / irreversibility proxies

- Same-subject different-key mean cosine: -0.007931429694561908
- Random-search inversion mean recon cosine (known C,E): 0.19388081151311892

## Notes

- Paper uses FaceNet / Idiap MS-Celeb models on MOBIO; we use CelebA+LFW under the size gate.
- See `DEVIATIONS.md`.
