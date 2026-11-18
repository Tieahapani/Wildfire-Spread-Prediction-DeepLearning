"""Train the U-Net baseline on cleaned NDWS data."""

from __future__ import annotations

import argparse
from pathlib import Path

from wfspread.data.loader import create_dataloader
from wfspread.models.unet import UNet
from wfspread.train.config import TrainConfig
from wfspread.train.run import fit

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = Path("/srv/sfsu_srv/wf-spread/data/processed/ndws-v1")
DEFAULT_RUNS = Path("/srv/sfsu_srv/wf-spread/runs")


def main() -> None:
    """Parse arguments, build data loaders and model, then train."""
    parser = argparse.ArgumentParser(description="Train U-Net on NDWS")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/train/default.yaml")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tag", type=str, default="baseline")
    parser.add_argument("--preload", action="store_true", help="Load data into memory")
    args = parser.parse_args()

    config = TrainConfig.from_yaml(args.config)
    print(f"Config: seed={config.seed}, lr={config.lr}, epochs={config.max_epochs}")

    train_loader = create_dataloader(
        args.data,
        split="train",
        batch_size=args.batch_size,
        num_workers=args.workers,
        seed=config.seed,
        preload=args.preload,
    )
    val_loader = create_dataloader(
        args.data,
        split="val",
        batch_size=args.batch_size,
        num_workers=args.workers,
        seed=config.seed,
        preload=args.preload,
    )
    print(f"Train: {len(train_loader.dataset)} tiles, {len(train_loader)} batches")
    print(f"Val:   {len(val_loader.dataset)} tiles, {len(val_loader)} batches")

    model = UNet()
    print(f"U-Net: {sum(p.numel() for p in model.parameters()):,} parameters")

    run_dir = fit(
        model,
        train_loader,
        val_loader,
        config,
        run_name=f"unet_{args.tag}",
        runs_dir=args.runs,
    )
    print(f"Run saved to: {run_dir}")


if __name__ == "__main__":
    main()
