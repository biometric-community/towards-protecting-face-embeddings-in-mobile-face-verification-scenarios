# FIDELITY_AUDIT — PolyProtect

- **Paper:** `papers/towards-protecting-face-embeddings-in-mobile-face-verification-scenarios/`
- **Project:** `projects/papers/towards-protecting-face-embeddings-in-mobile-face-verification-scenarios/`
- **Upstream:** Idiap bob.paper.polyprotect_2021 (referenced, not vendored)

## Claim inventory

| Claim ID | Claim | Status |
|----------|-------|--------|
| C1 | Multivariate poly map Eq.1 | ok |
| C2 | Overlap / stride windows | ok |
| C3 | User-specific C,E | ok |
| C4 | MOBIO FaceNet/Idiap | deviation:D1–D2 |
| C5 | Unlink / irrev metrics | deviation:D3–D4 |

## Per-pass statistics

| Pass | ok | missing | deviation | Weighted |
|------|----|---------|-----------|----------|
| 1–5 | 3 | 0 | 4 | 88–90 |

**Final weighted fidelity:** 90%
