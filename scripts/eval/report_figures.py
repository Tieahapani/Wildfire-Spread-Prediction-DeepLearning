"""Draw the report figures: precision-recall curves and example fire maps.

Uses the seed-0 run of each model in `configs/eval/final.yaml` and thresholds-free
probabilities, so nothing here changes a reported number. Example tiles are drawn
at random from tiles with enough fire tomorrow, with a fixed seed.

Examples:
    bin/wf python scripts/eval/report_figures.py --split val
    bin/wf python scripts/eval/report_figures.py --split test
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from sklearn.metrics import precision_recall_curve

from wfspread.constants import FIRE, IGNORE_INDEX
from wfspread.data.loader import NDWSDataset
from wfspread.eval.final import (
    PERSISTENCE,
    best_checkpoint,
    build_scored_model,
    collect,
    pick_tiles,
    read_manifest,
)
from wfspread.eval.metrics import pr_auc, valid_mask

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS = {
    "persistence": "Persistence",
    "logreg": "Logistic regression",
    "mlp": "MLP 3×3",
    "unet-small": "U-Net",
}
MASK_COLORS = ListedColormap(["#bdbdbd", "#ffffff", "#d7301f"])


def main() -> None:
    """Predict the split with each model's seed-0 run, then save both figures."""
    data_root = Path(os.environ.get("WF_DATA", REPO_ROOT / "data"))
    runs_root = Path(os.environ.get("WF_RUNS", REPO_ROOT / "runs"))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "configs/eval/final.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--models", nargs="+", default=list(LABELS))
    parser.add_argument("--n-tiles", type=int, default=4)
    parser.add_argument("--min-fire", type=int, default=20, help="fire pixels tomorrow")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", type=Path, default=data_root / "processed" / "ndws-v1")
    parser.add_argument("--out", type=Path, default=runs_root / "final-eval" / "figures")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    specs = {spec.name: spec for spec in read_manifest(args.manifest, runs_root)[0]}
    dataset = NDWSDataset(args.data, args.split, preload=True)
    _, labels, prev_fire = collect(None, dataset, device)
    probs = {}
    for name in args.models:
        spec = specs[name]
        model = None
        if spec.kind != PERSISTENCE:
            model = build_scored_model(spec, best_checkpoint(spec.runs[0])).to(device)
        probs[name] = collect(model, dataset, device)[0]
        print(f"predicted {name}")

    args.out.mkdir(parents=True, exist_ok=True)
    _pr_curves(probs, labels, args.out / f"pr-curves-{args.split}.png", args.split)
    tiles = pick_tiles(labels, args.n_tiles, args.min_fire, args.seed)
    _maps(probs, labels, prev_fire, tiles, args.out / f"maps-{args.split}.png", args.split)
    print(f"tiles {tiles}; figures in {args.out}")


def _pr_curves(probs: dict, labels: torch.Tensor, path: Path, split: str) -> None:
    valid = valid_mask(labels)
    truth = (labels[valid] == FIRE).cpu().numpy()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, p in probs.items():
        precision, recall, _ = precision_recall_curve(truth, p[valid].cpu().numpy())
        # Steps, not straight lines: PR-AUC is computed as average precision over these
        # steps, and joining a yes/no model's few points with lines overstates it.
        ax.plot(
            recall,
            precision,
            drawstyle="steps-post",
            label=f"{LABELS.get(name, name)} (PR-AUC {pr_auc(p, labels):.3f})",
        )
    ax.axhline(truth.mean(), color="grey", linestyle=":", label=f"Fire rate {truth.mean():.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Precision-recall, {split} split, seed 0")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _maps(probs, labels, prev_fire, tiles, path: Path, split: str) -> None:
    columns = [
        "Fire today",
        *[LABELS.get(n, n) for n in probs if n != PERSISTENCE],
        "Fire tomorrow",
    ]
    fig, axes = plt.subplots(
        len(tiles), len(columns), figsize=(2.3 * len(columns), 2.3 * len(tiles))
    )
    axes = np.atleast_2d(axes)
    for row, tile in zip(axes, tiles, strict=True):
        images = [("mask", prev_fire[tile])]
        images += [("prob", p[tile]) for n, p in probs.items() if n != PERSISTENCE]
        images += [("mask", labels[tile])]
        for ax, (kind, image) in zip(row, images, strict=True):
            image = image.cpu().numpy()
            if kind == "mask":
                ax.imshow(
                    image, cmap=MASK_COLORS, vmin=IGNORE_INDEX, vmax=FIRE, interpolation="nearest"
                )
            else:
                shown = ax.imshow(image, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
        row[0].set_ylabel(f"tile {tile}", fontsize=8)
    for ax, title in zip(axes[0], columns, strict=True):
        ax.set_title(title, fontsize=9)
    fig.colorbar(shown, ax=axes, fraction=0.02, label="Predicted fire probability")
    fig.suptitle(f"{split} split, seed-0 runs; grey = unknown, red = fire")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
