# IMPLEMENTATION_PLAN — PolyProtect

## Goal
Reproduce PolyProtect V→P protection with accuracy / unlinkability / irreversibility trade-offs under the 5 GiB gate.

## Figure inventory
| Paper fig | Our artifact |
|-----------|--------------|
| Acc vs protection | `verify_metrics.svg` |
| Training | `train_loss.svg` |

## Size gate
celeba + lfw ≈ 3.5 GiB → full_train.
