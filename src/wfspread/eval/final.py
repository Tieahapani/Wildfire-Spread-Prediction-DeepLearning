"""Final evaluation: score trained models on one split with seed and bootstrap spread.

Each model is a list of runs, one per seed. A run's decision threshold is always
chosen on the validation split, then applied unchanged to the scored split.
Seed variation (mean and standard deviation over runs) and sampling variation
(bootstrap over tiles) are reported separately.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from wfspread.constants import FIRE, IGNORE_INDEX, NDWS_CHANNELS
from wfspread.eval.metrics import (
    ConfidenceInterval,
    PairedComparison,
    best_f1_threshold,
    bootstrap_ci,
    evaluate,
    paired_bootstrap,
    pr_auc,
    spread_mask,
)
from wfspread.models.registry import build_model
from wfspread.models.wrappers import ChannelSubset
from wfspread.serve.predict import load_weights

PERSISTENCE = "persistence"


@dataclass(frozen=True)
class ModelSpec:
    """One model in the final evaluation.

    Attributes:
        name: Label in the results tables.
        kind: "checkpoint" for trained runs, or "persistence" (tomorrow's fire is
            today's fire, no training).
        model_config: Model YAML, for checkpoints.
        drop_channels: Input channels the model never sees (`ChannelSubset`).
        runs: Run folders, one per seed, for checkpoints.
    """

    name: str
    kind: str = "checkpoint"
    model_config: Path | None = None
    drop_channels: tuple[str, ...] = ()
    runs: tuple[Path, ...] = ()


@dataclass
class Predictions:
    """Probabilities of every run of one model on one split.

    Attributes:
        probs: float32 (B, S, H, W), S runs in the order of `ModelSpec.runs`.
        thresholds: Best-F1 threshold of each run on the validation split.
    """

    probs: Tensor
    thresholds: list[float]


def read_manifest(path: Path, runs_root: Path) -> tuple[list[ModelSpec], list[tuple[str, str]]]:
    """Models and paired comparisons from a final-evaluation YAML file.

    Model configs are relative to the file and runs relative to `runs_root`.

    Raises:
        ValueError: If a comparison names a model that is not listed.
    """
    path = Path(path)
    values = yaml.safe_load(path.read_text())
    specs = []
    for name, entry in values["models"].items():
        kind = entry.get("kind", "checkpoint")
        config = entry.get("model")
        specs.append(
            ModelSpec(
                name=name,
                kind=kind,
                model_config=(path.parent / config).resolve() if config else None,
                drop_channels=tuple(entry.get("drop_channels", ())),
                runs=tuple(runs_root / run for run in entry.get("runs", ())),
            )
        )
    names = {spec.name for spec in specs}
    comparisons = [tuple(pair) for pair in values.get("comparisons", [])]
    unknown = {name for pair in comparisons for name in pair} - names
    if unknown:
        raise ValueError(f"comparisons name unlisted models {sorted(unknown)}")
    return specs, comparisons


def build_scored_model(spec: ModelSpec, checkpoint: Path) -> nn.Module:
    """The trained model of one run, in eval mode, taking all `NDWS_CHANNELS`."""
    values = yaml.safe_load(spec.model_config.read_text())
    args = dict(values.get("args") or {})
    if spec.drop_channels:
        args["in_ch"] = len(NDWS_CHANNELS) - len(spec.drop_channels)
    model = build_model(values["name"], **args)
    if spec.drop_channels:
        model = ChannelSubset(model, spec.drop_channels)
    return load_weights(model, checkpoint)


def best_checkpoint(run_dir: Path) -> Path:
    """The best-validation checkpoint that `fit` saved in `run_dir`.

    Raises:
        ValueError: If there is not exactly one.
    """
    found = sorted(Path(run_dir).glob("checkpoints/best-*.ckpt"))
    if len(found) != 1:
        raise ValueError(f"expected one best checkpoint in {run_dir}, found {len(found)}")
    return found[0]


@torch.no_grad()
def collect(
    model: nn.Module | None, dataset: Dataset, device: torch.device, batch_size: int = 64
) -> tuple[Tensor, Tensor, Tensor]:
    """Probabilities, next-day masks and day-t masks for a whole split, in order.

    Args:
        model: Trained model, or None for persistence.
        dataset: Split in the batch format, not augmented.
        device: Where the model runs; the outputs stay there too.
        batch_size: Tiles per forward pass.

    Returns:
        probs float32 (B, H, W), next_fire int8 (B, H, W), prev_fire int8 (B, H, W).
    """
    probs, next_fire, prev_fire = [], [], []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        prev = batch["prev_fire"].to(device)
        if model is None:
            probs.append((prev == FIRE).float())
        else:
            probs.append(torch.sigmoid(model(batch["inputs"].to(device)).float()))
        next_fire.append(batch["next_fire"].to(device))
        prev_fire.append(prev)
    return torch.cat(probs), torch.cat(next_fire), torch.cat(prev_fire)


def seed_mean_pr_auc(probs: Tensor, labels: Tensor, prev_fire: Tensor) -> float:
    """Mean over runs of all-pixel PR-AUC; `probs` is (B, S, H, W)."""
    return statistics.fmean(pr_auc(probs[:, s], labels) for s in range(probs.shape[1]))


def seed_mean_spread_pr_auc(probs: Tensor, labels: Tensor, prev_fire: Tensor) -> float:
    """Mean over runs of spread-region PR-AUC; `probs` is (B, S, H, W)."""
    mask = spread_mask(labels, prev_fire)
    return statistics.fmean(pr_auc(probs[:, s], labels, mask) for s in range(probs.shape[1]))


def score_runs(
    predictions: Predictions, labels: Tensor, prev_fire: Tensor
) -> list[dict[str, float]]:
    """`evaluate` for each run, at that run's validation threshold."""
    return [
        {"threshold": threshold, **evaluate(predictions.probs[:, s], labels, prev_fire, threshold)}
        for s, threshold in enumerate(predictions.thresholds)
    ]


def validation_thresholds(probs: Tensor, labels: Tensor) -> list[float]:
    """Best-F1 threshold of each run; call on validation predictions only."""
    return [best_f1_threshold(probs[:, s], labels) for s in range(probs.shape[1])]


def confidence_intervals(
    predictions: Predictions, labels: Tensor, prev_fire: Tensor, n_resamples: int, seed: int = 0
) -> dict[str, ConfidenceInterval]:
    """Tile-bootstrap intervals of the seed-mean PR-AUC, all pixels and spread region."""
    tensors = (predictions.probs, labels, prev_fire)
    return {
        "all/pr_auc": bootstrap_ci(seed_mean_pr_auc, *tensors, n_resamples=n_resamples, seed=seed),
        "spread/pr_auc": bootstrap_ci(
            seed_mean_spread_pr_auc, *tensors, n_resamples=n_resamples, seed=seed
        ),
    }


def compare(
    a: Predictions,
    b: Predictions,
    labels: Tensor,
    prev_fire: Tensor,
    n_resamples: int,
    seed: int = 0,
) -> dict[str, PairedComparison]:
    """Paired tile bootstrap of seed-mean PR-AUC, model A minus model B."""
    return {
        name: paired_bootstrap(
            metric, a.probs, b.probs, labels, prev_fire, n_resamples=n_resamples, seed=seed
        )
        for name, metric in (
            ("all/pr_auc", seed_mean_pr_auc),
            ("spread/pr_auc", seed_mean_spread_pr_auc),
        )
    }


def stack_runs(per_run: Sequence[Tensor]) -> Tensor:
    """(B, H, W) probabilities of S runs as (B, S, H, W)."""
    return torch.stack(list(per_run), dim=1)


def without_border(labels: Tensor) -> Tensor:
    """Next-day masks (B, H, W) with the 1-pixel tile border marked unknown.

    Metrics skip unknown pixels, so this scores the interior only, as Huot et al.
    2022 did for logistic regression and persistence.
    """
    labels = labels.clone()
    labels[:, [0, -1], :] = IGNORE_INDEX
    labels[:, :, [0, -1]] = IGNORE_INDEX
    return labels


def pick_tiles(labels: Tensor, n: int, min_fire: int, seed: int = 0) -> list[int]:
    """Random tiles with at least `min_fire` fire pixels tomorrow, for example maps.

    A fixed random draw, rather than choosing by eye, keeps the figures from
    showing only the tiles where a model looks good.

    Raises:
        ValueError: If fewer than `n` tiles qualify.
    """
    fire = (labels == FIRE).flatten(1).sum(1)
    eligible = torch.nonzero(fire >= min_fire).flatten()
    if len(eligible) < n:
        raise ValueError(f"only {len(eligible)} tiles have >= {min_fire} fire pixels")
    generator = torch.Generator().manual_seed(seed)
    chosen = eligible.cpu()[torch.randperm(len(eligible), generator=generator)[:n]]
    return sorted(chosen.tolist())
