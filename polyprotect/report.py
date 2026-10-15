"""Figures + REPORT.md for PolyProtect."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from polyprotect.plot_style import FIGSIZE, LINEWIDTH_MAIN, MULTI_SERIES_COLORS, apply_rcparams, apply_style


def plot_train(hist, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot([h["epoch"] for h in hist], [h["loss"] for h in hist], color=MULTI_SERIES_COLORS[0], linewidth=LINEWIDTH_MAIN)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CosFace CE loss")
    ax.set_title("FR backbone (CelebA → 128-D)")
    apply_style(ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def plot_verify(ver: dict, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    names = list(ver.keys())
    acc = [ver[k]["accuracy"] * 100 for k in names]
    eer = [ver[k]["eer"] * 100 for k in names]
    x = range(len(names))
    w = 0.35
    ax.bar([i - w / 2 for i in x], acc, w, label="Accuracy", color=MULTI_SERIES_COLORS[0], edgecolor="#002080")
    ax.bar([i + w / 2 for i in x], eer, w, label="EER", color=MULTI_SERIES_COLORS[1], edgecolor="#002080")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylabel("%")
    ax.set_title("LFW: unprotected vs PolyProtect")
    ax.legend()
    apply_style(ax, grid_axis="y")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def write_report(cfg, train_s, eval_res, fig_dir: Path, path: Path) -> None:
    protocol = cfg["train"].get("protocol", "full")
    ver = eval_res.get("verification", {})
    unlink = eval_res.get("unlinkability", {})
    irrev = eval_res.get("irreversibility", {})
    pp = eval_res.get("polyprotect", {})
    lines = [
        "# REPORT — Towards Protecting Face Embeddings (PolyProtect)",
        "",
        f"**Protocol:** `{protocol}`",
        "",
        "Lab twin: CelebA-trained CosFace ResNet-18 (128-D) + PolyProtect multivariate",
        f"polynomials (m={pp.get('m')}, overlap={pp.get('overlap')}); LFW verification,",
        "unlinkability and irreversibility proxies.",
        "",
        "## Training",
        "",
        f"- Epochs: {train_s.get('epochs')}",
        f"- Train images: {train_s.get('n_train')}",
        f"- Identities: {train_s.get('num_classes')}",
        f"- Emb dim: {train_s.get('emb_dim')}",
        f"- Elapsed (s): {train_s.get('elapsed_sec', 0):.1f}",
        "",
        f"![Training]({fig_dir.as_posix()}/train_loss.svg)",
        "",
        "## LFW verification",
        "",
        "| Mode | Accuracy | EER | TAR@FAR1e-2 |",
        "|------|----------|-----|-------------|",
    ]
    for k, v in ver.items():
        lines.append(
            f"| {k} | {v['accuracy']*100:.2f}% | {v['eer']*100:.2f}% | {v['tar@far0.01']*100:.2f}% |"
        )
    lines += [
        "",
        f"![Verification]({fig_dir.as_posix()}/verify_metrics.svg)",
        "",
        "## Unlinkability / irreversibility proxies",
        "",
        f"- Same-subject different-key mean cosine: {unlink.get('mean_cosine_same_subject_diff_keys')}",
        f"- GD inversion mean recon cosine (known C,E): {irrev.get('mean_recon_cosine')}",
        "",
        "## Notes",
        "",
        "- Paper uses FaceNet / Idiap MS-Celeb models on MOBIO; we use CelebA+LFW under the size gate.",
        "- See `DEVIATIONS.md`.",
        "",
    ]
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    log_dir = Path(cfg["paths"]["log_dir"])
    fig_dir = Path(cfg["paths"]["figure_dir"])
    train_s = json.loads((log_dir / "train_summary.json").read_text())
    eval_res = json.loads((log_dir / "eval.json").read_text())
    plot_train(train_s["history"], fig_dir / "train_loss.svg")
    plot_verify(eval_res["verification"], fig_dir / "verify_metrics.svg")
    write_report(cfg, train_s, eval_res, fig_dir, Path("REPORT.md"))
    print("Wrote REPORT.md")


if __name__ == "__main__":
    main()
