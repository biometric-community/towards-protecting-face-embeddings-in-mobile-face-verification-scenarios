"""Baseline vs PolyProtected LFW verification + unlinkability / irreversibility proxies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from polyprotect.data.pairs import build_lfw, resolve_lfw
from polyprotect.models.fr_net import FaceEmbedder
from polyprotect.protect.polyprotect import PolyProtect, sample_user_params


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _device(cfg: dict, override: Optional[str] = None) -> torch.device:
    name = override or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def best_acc_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    thresholds = np.linspace(scores.min(), scores.max(), 200)
    best_acc, best_eer, best_t = 0.0, 1.0, 0.0
    for t in thresholds:
        pred = (scores >= t).astype(np.int32)
        acc = float((pred == labels).mean())
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
        pos, neg = labels == 1, labels == 0
        if pos.any() and neg.any():
            fpr = float((pred[neg] == 1).mean())
            fnr = float((pred[pos] == 0).mean())
            best_eer = min(best_eer, 0.5 * (fpr + fnr))
    return best_acc, best_eer, best_t


def tar_at_far(scores: np.ndarray, labels: np.ndarray, far: float = 1e-2) -> float:
    neg = scores[labels == 0]
    pos = scores[labels == 1]
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    t = float(np.quantile(neg, 1.0 - far))
    return float((pos >= t).mean())


def _subject_from_rel(path: str) -> str:
    # LFW relative: Name/Name_0001.jpg
    return path.split("/")[0]


@torch.no_grad()
def evaluate(cfg: dict, device_override: Optional[str] = None) -> Dict:
    device = _device(cfg, device_override)
    ckpt = torch.load(Path(cfg["paths"]["checkpoint_dir"]) / "last.pt", map_location=device, weights_only=False)
    model = FaceEmbedder(int(ckpt["num_classes"]), emb_dim=int(ckpt.get("emb_dim", 128)), pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    image_size = int(cfg["model"].get("image_size", 112))
    max_pairs = cfg["eval"].get("max_pairs")
    max_pairs = int(max_pairs) if max_pairs is not None else None
    bs = int(cfg["eval"].get("batch_size", 64))
    nw = int(cfg["train"].get("num_workers", 4))
    m = int(cfg["polyprotect"].get("m", 5))
    overlap = int(cfg["polyprotect"].get("overlap", 4))
    seed = int(cfg["train"].get("seed", 42))
    pp = PolyProtect(m=m, overlap=overlap)
    rng = np.random.RandomState(seed)

    ds = build_lfw(cfg, image_size, max_pairs)
    # need subject ids from pair relative paths
    pair_meta = list(ds.pairs)  # (rel_a, rel_b, label)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    user_params: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def params_for(subj: str) -> Tuple[np.ndarray, np.ndarray]:
        if subj not in user_params:
            user_params[subj] = sample_user_params(m, rng)
        return user_params[subj]

    raw_scores, prot_scores, unlink_scores, labels = [], [], [], []
    emb_bank = []  # for irreversibility sample
    prot_bank = []
    coeff_bank = []

    for idx, (a, b, y) in enumerate(tqdm(loader, desc="lfw", leave=False)):
        rel_a, rel_b, lab = pair_meta[idx]
        sa, sb = _subject_from_rel(rel_a), _subject_from_rel(rel_b)
        ea = model.embed(a.to(device))
        eb = model.embed(b.to(device))
        raw_scores.append(float((ea * eb).sum().item()))
        labels.append(int(y.item()))

        ca, ea_p = params_for(sa)
        cb, eb_p = params_for(sb)
        pa = pp.transform(ea[0], ca, ea_p)
        pb = pp.transform(eb[0], cb, eb_p)
        prot_scores.append(float((pa * pb).sum().item()))

        # unlinkability: same subject genuine pairs remapped with fresh keys
        if lab == 1:
            c2, e2 = sample_user_params(m, rng)
            pa2 = pp.transform(ea[0], c2, e2)
            pb2 = pp.transform(eb[0], c2, e2)  # same new keys for both of same user
            # compare protected-with-key1 vs protected-with-key2 of same images? paper: different templates same subject
            unlink_scores.append(float((pa * pa2).sum().item()))

        if len(emb_bank) < int(cfg["eval"].get("n_irrev", 64)):
            emb_bank.append(ea[0].cpu().numpy())
            prot_bank.append(pa.cpu().numpy())
            coeff_bank.append((ca, ea_p))

    raw = np.asarray(raw_scores)
    prot = np.asarray(prot_scores)
    y = np.asarray(labels)
    results = {"verification": {}, "unlinkability": {}, "irreversibility": {}, "polyprotect": {"m": m, "overlap": overlap}}

    for tag, s in (("baseline", raw), ("polyprotect", prot)):
        acc, eer, thr = best_acc_eer(s, y)
        tar = tar_at_far(s, y, 1e-2)
        results["verification"][tag] = {
            "accuracy": acc,
            "eer": eer,
            "threshold": thr,
            "tar@far0.01": tar,
            "n_pairs": int(len(y)),
        }
        print(f"[{tag}] acc={acc:.4f} eer={eer:.4f} tar@1e-2={tar:.4f}")

    if unlink_scores:
        u = np.asarray(unlink_scores)
        results["unlinkability"] = {
            "mean_cosine_same_subject_diff_keys": float(u.mean()),
            "std": float(u.std()),
            "n": int(len(u)),
        }
        print(f"[unlink] mean_cos={u.mean():.4f}")

    # Irreversibility proxy: best-of-N random search for Vhat matching P under known C,E
    irrev_sims = []
    attack_rng = np.random.RandomState(seed + 7)
    n_try = int(cfg["eval"].get("n_irrev_tries", 64))
    for v_np, p_np, (c, e) in zip(emb_bank, prot_bank, coeff_bank):
        best_sim = -1.0
        target = p_np / (np.linalg.norm(p_np) + 1e-12)
        for _ in range(n_try):
            cand = attack_rng.normal(size=v_np.shape).astype(np.float64)
            cand /= np.linalg.norm(cand) + 1e-12
            pred = pp.transform_numpy(cand, c, e)
            pred /= np.linalg.norm(pred) + 1e-12
            if float(np.dot(pred, target)) > 0.995:
                # matched protected template — score similarity to true V
                best_sim = max(best_sim, float(np.dot(cand, v_np / (np.linalg.norm(v_np) + 1e-12))))
        if best_sim < 0:
            # no near match; report cosine of closest protected match attempt
            closest = -1.0
            best_cand = None
            for _ in range(n_try):
                cand = attack_rng.normal(size=v_np.shape).astype(np.float64)
                cand /= np.linalg.norm(cand) + 1e-12
                pred = pp.transform_numpy(cand, c, e)
                pred /= np.linalg.norm(pred) + 1e-12
                sc = float(np.dot(pred, target))
                if sc > closest:
                    closest = sc
                    best_cand = cand
            best_sim = float(np.dot(best_cand, v_np / (np.linalg.norm(v_np) + 1e-12)))
        irrev_sims.append(best_sim)
    results["irreversibility"] = {
        "mean_recon_cosine": float(np.mean(irrev_sims)) if irrev_sims else None,
        "n": len(irrev_sims),
        "note": "Random-search inversion with known C,E; lower = harder to recover V",
    }
    print(f"[irrev] mean_recon_cos={results['irreversibility']['mean_recon_cosine']}")

    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "eval.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    evaluate(load_config(args.config), device_override=args.device)


if __name__ == "__main__":
    main()
