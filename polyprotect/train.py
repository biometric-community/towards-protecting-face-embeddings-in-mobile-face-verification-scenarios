"""Train CosFace ResNet-18 on CelebA (128-D embeddings)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from polyprotect.data.celeba import CelebAIdDataset, resolve_celeba_root
from polyprotect.models.fr_net import FaceEmbedder


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _device(cfg: dict, override: Optional[str] = None) -> torch.device:
    name = override or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def train(cfg: dict, device_override: Optional[str] = None) -> Dict:
    device = _device(cfg, device_override)
    root = resolve_celeba_root(cfg)
    seed = int(cfg["train"].get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    max_n = cfg["data"].get("max_images")
    max_n = int(max_n) if max_n is not None else None
    image_size = int(cfg["model"].get("image_size", 112))
    emb_dim = int(cfg["model"].get("emb_dim", 128))

    train_ds = CelebAIdDataset(
        root,
        split="train",
        image_size=image_size,
        train_ratio=float(cfg["data"].get("train_ratio", 0.7)),
        max_n=max_n,
        seed=seed,
        train=True,
    )
    loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"].get("batch_size", 64)),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    model = FaceEmbedder(train_ds.num_classes, emb_dim=emb_dim, pretrained=True).to(device)
    opt = torch.optim.SGD(
        model.parameters(), lr=float(cfg["train"].get("lr", 0.05)), momentum=0.9, weight_decay=5e-4
    )
    epochs = int(cfg["train"].get("epochs", 8))
    hist = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        pbar = tqdm(loader, desc=f"ep{epoch}/{epochs}", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images, labels)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            pbar.set_postfix(loss=f"{losses[-1]:.3f}")
        mean = float(np.mean(losses))
        hist.append({"epoch": epoch, "loss": mean})
        print(f"[train] epoch={epoch} loss={mean:.4f}")

    elapsed = time.time() - t0
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    log_dir = Path(cfg["paths"]["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "num_classes": train_ds.num_classes, "emb_dim": emb_dim, "config": cfg},
        ckpt_dir / "last.pt",
    )
    summary = {
        "epochs": epochs,
        "n_train": len(train_ds),
        "num_classes": train_ds.num_classes,
        "emb_dim": emb_dim,
        "elapsed_sec": elapsed,
        "history": hist,
        "protocol": cfg["train"].get("protocol", "full"),
    }
    (log_dir / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[train] done n={len(train_ds)} elapsed={elapsed:.1f}s")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    train(load_config(args.config), device_override=args.device)


if __name__ == "__main__":
    main()
