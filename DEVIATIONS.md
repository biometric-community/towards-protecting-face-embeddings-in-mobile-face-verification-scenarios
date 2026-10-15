# DEVIATIONS — PolyProtect

| ID | Item | Paper | Ours | Reason |
|----|------|-------|------|--------|
| D1 | Feature extractors | FaceNet + Idiap MS-Celeb Inception-ResNet-v2 | CosFace ResNet-18 on CelebA | Size gate; no MS1M/MOBIO payloads |
| D2 | Evaluation set | MOBIO | LFW | MOBIO not mirrored locally |
| D3 | Unlinkability metric | Gomez-Barrero Dsys↔ | Mean cosine same-subject diff keys | Simplified proxy |
| D4 | Irreversibility | Paper attack suite | GD inversion recon cosine | Practical proxy |
