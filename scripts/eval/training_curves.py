"""Draw training curves of the final models for the report.

For each model in `configs/eval/final.yaml` it plots every seed's training loss,
validation loss and validation PR-AUC per epoch, and the mean epoch time and peak
GPU memory, all read from the runs' `logs/metrics.csv`. Nothing is retrained.

Example:
    bin/wf python scripts/eval/training_curves.py
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

from wfspread.eval.final import read_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS = {"logreg": "Logistic regression", "mlp": "MLP 3×3", "unet-small": "U-Net"}
COLORS = {"logreg": "tab:orange", "mlp": "tab:green", "unet-small": "tab:red"}


def read_epochs(run_dir: Path) -> dict[str, list[float]]:
    """Per-epoch training loss, validation loss and PR-AUC, epoch time and peak GPU memory."""
    columns = {
        "train_loss": "train/loss_epoch",
        "val_loss": "val/loss",
        "val_pr_auc": "val/all/pr_auc",
        "epoch_s": "time/epoch_s",
        "peak_gb": "memory/peak_gpu_gb",
    }
    by_epoch: dict[int, dict[str, float]] = {}
    with (run_dir / "logs" / "metrics.csv").open() as f:
        for row in csv.DictReader(f):
            if not row["epoch"]:
                continue  # per-step diagnostics rows carry no epoch
            epoch = by_epoch.setdefault(int(float(row["epoch"])), {})
            for key, column in columns.items():
                if row.get(column):
                    epoch[key] = float(row[column])
    epochs = sorted(e for e, values in by_epoch.items() if "val_pr_auc" in values)
    out = {"epoch": [float(e) for e in epochs]}
    for key in columns:
        out[key] = [by_epoch[e].get(key, float("nan")) for e in epochs]
    return out


def main() -> None:
    """Read every run of each model and save one 2 x 2 figure."""
    runs_root = Path(os.environ.get("WF_RUNS", REPO_ROOT / "runs"))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "configs/eval/final.yaml")
    parser.add_argument("--models", nargs="+", default=list(LABELS))
    parser.add_argument("--out", type=Path, default=runs_root / "final-eval" / "figures")
    args = parser.parse_args()

    specs = {spec.name: spec for spec in read_manifest(args.manifest, runs_root)[0]}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    (ax_train, ax_val), (ax_auc, ax_cost) = axes
    costs = []
    for name in args.models:
        color, label = COLORS.get(name), LABELS.get(name, name)
        times, peaks = [], []
        for i, run in enumerate(specs[name].runs):
            log = read_epochs(run)
            first = {"label": label} if i == 0 else {}
            ax_train.plot(log["epoch"], log["train_loss"], color=color, alpha=0.7, **first)
            ax_val.plot(log["epoch"], log["val_loss"], color=color, alpha=0.7, **first)
            ax_auc.plot(log["epoch"], log["val_pr_auc"], color=color, alpha=0.7, **first)
            best = max(range(len(log["epoch"])), key=lambda k: log["val_pr_auc"][k])
            ax_auc.plot(log["epoch"][best], log["val_pr_auc"][best], "o", color=color, ms=4)
            times += log["epoch_s"]
            peaks.append(max(log["peak_gb"]))
        costs.append((label, color, statistics.median(times), max(peaks)))

    for ax, title in (
        (ax_train, "Training loss (weighted BCE)"),
        (ax_val, "Validation loss"),
        (ax_auc, "Validation PR-AUC (dot = best epoch)"),
    ):
        ax.set_xlabel("Epoch")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    ax_train.set_yscale("log")
    ax_val.set_yscale("log")

    labels = [c[0] for c in costs]
    bars = ax_cost.bar(labels, [c[2] for c in costs], color=[c[1] for c in costs])
    for bar, cost in zip(bars, costs, strict=True):
        ax_cost.annotate(
            f"{cost[2]:.1f} s\npeak {cost[3]:.2f} GB",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax_cost.set_ylabel("Median seconds per epoch")
    ax_cost.set_title("Epoch time and peak GPU memory in training", fontsize=10)
    ax_cost.margins(y=0.25)
    ax_cost.tick_params(axis="x", labelsize=8)

    fig.suptitle("Training of the final models (3 seeds each)")
    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "training-curves.png"
    fig.savefig(path, dpi=200)
    print(f"saved {path}")
    for label, _, seconds, peak in costs:
        print(f"{label}: median {seconds:.1f} s/epoch, peak {peak:.2f} GB")


if __name__ == "__main__":
    main()
