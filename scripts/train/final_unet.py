"""Train the final U-Net with the best Optuna settings over 3 seeds.

Also runs the no-fire-mask ablation (12 input channels, dropping prev_fire).
Results are saved to a CSV and printed as a summary table.

Examples:
    bin/wf python scripts/train/final_unet.py
    bin/wf python scripts/train/final_unet.py --seeds 0 1 2 --batch-size 64
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import os
from pathlib import Path

import yaml
from torch import nn
from torch.utils.data import DataLoader

from wfspread.data.loader import NDWSDataset
from wfspread.models.registry import build_model
from wfspread.models.wrappers import ChannelSubset
from wfspread.train.config import TrainConfig
from wfspread.train.results import REPORTED_METRICS, best_validation, summarize
from wfspread.train.run import fit

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = Path(os.environ.get("WF_DATA", REPO_ROOT / "data")) / "processed" / "ndws-v1"
DEFAULT_RUNS = Path(os.environ.get("WF_RUNS", REPO_ROOT / "runs")) / "final-unet"

# Best weighted-BCE Optuna trial: model and training settings.
MODEL_CONFIG = REPO_ROOT / "configs" / "model" / "unet-optuna.yaml"
TRAIN_CONFIG = REPO_ROOT / "configs" / "experiment" / "final-unet.yaml"

RUN_FIELDS = ("arm", "seed", "run_dir", "epoch", "epochs", *REPORTED_METRICS)


def make_config(seed: int) -> TrainConfig:
    """Training settings of the final runs, for one seed."""
    return dataclasses.replace(TrainConfig.from_yaml(TRAIN_CONFIG), seed=seed)


def make_model(nofire: bool = False) -> nn.Module:
    """The best Optuna U-Net; without today's fire mask, it never sees prev_fire."""
    values = yaml.safe_load(MODEL_CONFIG.read_text())
    args = dict(values["args"])
    if nofire:
        args["in_ch"] -= 1
        return ChannelSubset(build_model(values["name"], **args), ["prev_fire"])
    return build_model(values["name"], **args)


def main() -> None:
    """Run final experiments and print summary."""
    parser = argparse.ArgumentParser(description="Final U-Net: 3 seeds + no-fire ablation")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--nofire-only", action="store_true", help="Run only the no-fire ablation")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    runs_csv = args.out / "runs.csv"
    runs = _read_runs(runs_csv)
    done = {(r["arm"], int(r["seed"])) for r in runs}

    arms = []
    if not args.nofire_only:
        arms += [("final-unet", False)]
    arms += [("final-unet-nofire", True)]

    todo = [
        (arm, nofire, seed)
        for arm, nofire in arms
        for seed in args.seeds
        if (arm, seed) not in done
    ]
    print(f"{len(done)} runs already done, {len(todo)} to train")

    train_set = NDWSDataset(args.data, "train", augment=True, preload=True) if todo else None
    val_set = NDWSDataset(args.data, "val", preload=True) if todo else None

    for arm, nofire, seed in todo:
        config = make_config(seed)
        model = make_model(nofire=nofire)
        params = sum(p.numel() for p in model.parameters())
        print(f"\n=== {arm} seed={seed} ({params:,} params) ===")

        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )

        run_dir = fit(
            model,
            train_loader,
            val_loader,
            config,
            run_name=f"unet_{arm}-s{seed}",
            runs_dir=args.out,
        )
        run = {"arm": arm, "seed": seed, "run_dir": run_dir.name, **best_validation(run_dir)}
        _append_run(runs_csv, run)
        runs.append(run)
        print(f"=== {arm} seed={seed}: PR-AUC={run['val/all/pr_auc']:.4f} ===\n")

    if runs:
        rows = summarize(runs, group_by="arm")
        table = _markdown_table(rows)
        (args.out / "summary.md").write_text(table + "\n")
        print("\n" + table)


def _read_runs(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _append_run(path: Path, run: dict[str, object]) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(run)


def _markdown_table(rows: list[dict[str, object]]) -> str:
    header = ["arm", "n", *REPORTED_METRICS]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        cells = [str(row["arm"]), str(row["n"])]
        cells += [f"{row[f'{m}_mean']:.3f} ± {row[f'{m}_std']:.3f}" for m in REPORTED_METRICS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
