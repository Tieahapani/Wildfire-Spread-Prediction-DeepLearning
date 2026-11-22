"""Optuna hyperparameter search for the U-Net.

Searches over learning rate, dropout, encoder channels and bottleneck size.
Each trial trains with early stopping and reports the best validation PR-AUC.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import optuna
import torch

from wfspread.data.loader import create_dataloader
from wfspread.models.unet import UNet
from wfspread.train.config import TrainConfig
from wfspread.train.run import fit

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = Path("/srv/sfsu_srv/wf-spread/data/processed/ndws-v1")
DEFAULT_RUNS = Path("/srv/sfsu_srv/wf-spread/runs")


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """One Optuna trial: build a U-Net with sampled hyperparameters, train, return PR-AUC."""
    lr = trial.suggest_float("lr", 5e-4, 3e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.4, step=0.1)
    n_levels = trial.suggest_int("n_levels", 2, 4)
    base_ch = trial.suggest_categorical("base_channels", [32, 48, 64])
    bottleneck_ch = trial.suggest_categorical("bottleneck_channels", [128, 256, 512])

    enc_channels = [base_ch * (2**i) for i in range(n_levels)]

    config = TrainConfig(
        seed=0,
        lr=lr,
        max_epochs=40,
        patience=10,
        loss_terms={"bce": 1.0},
        pos_weight=3.0,
        diagnostics_every_n_steps=0,
    )

    train_loader = create_dataloader(
        args.data,
        split="train",
        batch_size=args.batch_size,
        num_workers=args.workers,
        seed=config.seed,
        preload=True,
    )
    val_loader = create_dataloader(
        args.data,
        split="val",
        batch_size=args.batch_size,
        num_workers=args.workers,
        seed=config.seed,
        preload=True,
    )

    model = UNet(
        in_ch=13,
        enc_channels=enc_channels,
        bottleneck_ch=bottleneck_ch,
        dropout=dropout,
    )
    params = sum(p.numel() for p in model.parameters())
    trial.set_user_attr("params", params)
    trial.set_user_attr("enc_channels", enc_channels)
    print(
        f"\n=== Trial {trial.number}: lr={lr:.5f}, dropout={dropout:.1f}, "
        f"enc={enc_channels}, bottleneck={bottleneck_ch}, params={params:,} ==="
    )

    tag = f"search-t{trial.number:03d}"
    run_dir = fit(
        model,
        train_loader,
        val_loader,
        config,
        run_name=f"unet_{tag}",
        runs_dir=args.runs,
    )

    ckpt_dir = run_dir / "checkpoints"
    best_ckpt = sorted(ckpt_dir.glob("best-*.ckpt"))
    if not best_ckpt:
        return 0.0

    ckpt = torch.load(best_ckpt[0], map_location="cpu", weights_only=True)
    pr_auc = float(
        ckpt["callbacks"]["EarlyStopping{'monitor': 'val/all/pr_auc', 'mode': 'max'}"]["best_score"]
    )
    print(f"=== Trial {trial.number}: PR-AUC = {pr_auc:.4f} ===\n")
    return pr_auc


def main() -> None:
    """Run the Optuna search."""
    parser = argparse.ArgumentParser(description="U-Net hyperparameter search")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--n-trials", type=int, default=20)
    args = parser.parse_args()

    study = optuna.create_study(
        study_name="unet-search-wbce",
        direction="maximize",
        storage=f"sqlite:///{args.runs / 'unet_search_wbce.db'}",
        load_if_exists=True,
    )
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)

    print("\n" + "=" * 60)
    print("Best trial:")
    best = study.best_trial
    print(f"  PR-AUC: {best.value:.4f}")
    print(f"  Params: {best.params}")
    print(f"  Model size: {best.user_attrs.get('params', '?'):,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
