"""Embed + optionally PolyProtect a single image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from polyprotect.models.fr_net import FaceEmbedder
from polyprotect.protect.polyprotect import PolyProtect, sample_user_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="outputs/predict.json")
    ap.add_argument("--protect", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(Path(cfg["paths"]["checkpoint_dir"]) / "last.pt", map_location=device, weights_only=False)
    model = FaceEmbedder(int(ckpt["num_classes"]), emb_dim=int(ckpt.get("emb_dim", 128)), pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    image_size = int(cfg["model"].get("image_size", 112))
    tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    x = tf(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.embed(x)[0]
        payload = {"image": args.image, "embedding": emb.cpu().numpy().tolist()}
        if args.protect:
            m = int(cfg["polyprotect"].get("m", 5))
            ov = int(cfg["polyprotect"].get("overlap", 4))
            c, e = sample_user_params(m, np.random.RandomState(args.seed))
            p = PolyProtect(m, ov).transform(emb, c, e)
            payload["protected"] = p.cpu().numpy().tolist()
            payload["coeffs"] = c.tolist()
            payload["exponents"] = e.tolist()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
