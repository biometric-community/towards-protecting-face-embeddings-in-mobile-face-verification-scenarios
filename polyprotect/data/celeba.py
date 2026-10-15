"""CelebA identity training set (WebFace / MS1M substitute under size gate)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def resolve_celeba_root(cfg: dict) -> Path:
    root = Path(cfg["data"].get("celeba_root", "../../datasets/celeba/extracted/celeba"))
    if not root.is_absolute():
        root = (Path(__file__).resolve().parents[2] / root).resolve()
    if not (root / "img_align_celeba").is_dir():
        raise FileNotFoundError(root)
    if not (root / "identity_CelebA.txt").is_file():
        raise FileNotFoundError(root / "identity_CelebA.txt")
    return root


def _tf(image_size: int, train: bool) -> transforms.Compose:
    ops = [transforms.Resize((image_size, image_size))]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops += [
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
    return transforms.Compose(ops)


class CelebAIdDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str = "train",
        image_size: int = 112,
        train_ratio: float = 0.7,
        max_n: Optional[int] = None,
        seed: int = 42,
        train: bool = False,
    ) -> None:
        self.root = Path(root)
        self.img_dir = self.root / "img_align_celeba"
        self.transform = _tf(image_size, train=train)
        id_map: Dict[str, int] = {}
        with open(self.root / "identity_CelebA.txt") as f:
            for line in f:
                fn, sid = line.split()
                id_map[fn] = int(sid)
        subjects = sorted(set(id_map.values()))
        rng = np.random.RandomState(seed)
        rng.shuffle(subjects)
        n_train = int(round(len(subjects) * train_ratio))
        train_subs = set(subjects[:n_train])
        want = split == "train"
        rows = [(fn, sid) for fn, sid in id_map.items() if (sid in train_subs) == want]
        if max_n is not None:
            rows = rows[: int(max_n)]
        uniq = sorted({r[1] for r in rows})
        remap = {s: i for i, s in enumerate(uniq)}
        self.samples = [(fn, remap[sid]) for fn, sid in rows]
        self.num_classes = len(uniq)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        fn, sid = self.samples[idx]
        img = self.transform(Image.open(self.img_dir / fn).convert("RGB"))
        return img, sid
