"""Train a model once per experiment arm and seed, then compare the arms on validation.

Each finished run is added to `runs.csv` in the output folder straight away, so an
interrupted comparison picks up where it stopped when started again.

Examples:
    bin/wf python scripts/train/compare_arms.py --arms loss-bce loss-wbce loss-focal \
        loss-combo --out "$WF_RUNS/loss-comparison"
    bin/wf python scripts/train/compare_arms.py --arms crop-32 --out "$WF_RUNS/crop-comparison"
    bin/wf python scripts/train/compare_arms.py --arms loss-wbce \
        --model configs/model/mlp.yaml --out "$WF_RUNS/final-mlp"
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import os
from pathlib import Path

import yaml
from torch.utils.data import DataLoader

from wfspread.data.loader import NDWSDataset
from wfspread.models.registry import load_model
from wfspread.train.config import TrainConfig
from wfspread.train.results import REPORTED_METRICS, best_validation, summarize
from wfspread.train.run import fit

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = REPO_ROOT / "configs" / "experiment"
RUN_FIELDS = ("arm", "seed", "run_dir", "epoch", "epochs", *REPORTED_METRICS)


def main() -> None:
    """Run every missing arm and seed, then print and save the comparison table."""
    data_root = Path(os.environ.get("WF_DATA", REPO_ROOT / "data"))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arms", nargs="+", required=True, help="configs/experiment names")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--model", type=Path, default=REPO_ROOT / "configs/model/unet.yaml")
    parser.add_argument("--data", type=Path, default=data_root / "processed" / "ndws-v1")
    parser.add_argument("--out", type=Path, required=True, help="folder for runs and summary")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    model_name = yaml.safe_load(args.model.read_text())["name"]
    args.out.mkdir(parents=True, exist_ok=True)
    runs_csv = args.out / "runs.csv"
    runs = _read_runs(runs_csv)
    done = {(run["arm"], int(run["seed"])) for run in runs}
    todo = [(arm, seed) for arm in args.arms for seed in args.seeds if (arm, seed) not in done]
    print(f"{len(done)} runs already in {runs_csv}, {len(todo)} to train")

    # Read each split once and share it across runs; loading train takes seconds.
    train_sets: dict[int | None, NDWSDataset] = {}
    val_set = NDWSDataset(args.data, "val", preload=True) if todo else None
    for arm, seed in todo:
        config = dataclasses.replace(
            TrainConfig.from_yaml(EXPERIMENT_DIR / f"{arm}.yaml"), seed=seed
        )
        if config.crop_size not in train_sets:
            train_sets[config.crop_size] = NDWSDataset(
                args.data, "train", augment=True, crop_size=config.crop_size, preload=True
            )
        run_dir = fit(
            load_model(args.model),
            _loader(train_sets[config.crop_size], args, train=True),
            _loader(val_set, args, train=False),
            config,
            run_name=f"{model_name}_{arm}-s{seed}",
            runs_dir=args.out,
        )
        run = {"arm": arm, "seed": seed, "run_dir": run_dir.name, **best_validation(run_dir)}
        _append_run(runs_csv, run)
        runs.append(run)

    rows = summarize([run for run in runs if run["arm"] in args.arms], group_by="arm")
    table = _markdown_table(rows)
    (args.out / "summary.md").write_text(table + "\n")
    print(table)


def _loader(dataset: NDWSDataset, args: argparse.Namespace, *, train: bool) -> DataLoader:
    # fit() seeds torch before training, so shuffling and augmentation follow the run seed.
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )


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
