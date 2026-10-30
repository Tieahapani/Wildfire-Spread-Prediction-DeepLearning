"""Pixel-level metrics for next-day fire prediction.

Every metric takes per-pixel fire probabilities and the next-day fire mask for a
batch of tiles, and drops IGNORE_INDEX pixels before scoring. Each metric can
also be restricted to the *spread region*: pixels that were not burning on day
t. Fire that is already burning usually keeps burning, so a model can score well
over all pixels just by copying today's mask. The spread region shows whether it
predicts new fire.

Tensor shapes follow the project convention: probabilities, labels and day-t
masks are all (B, H, W).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch
from torch import Tensor

from wfspread.constants import FIRE, IGNORE_INDEX, NO_FIRE

MetricFn = Callable[..., float]


def valid_mask(labels: Tensor) -> Tensor:
    """Pixels whose next-day label is known.

    Args:
        labels: Next-day fire mask, shape (B, H, W), values in {-1, 0, 1}.

    Returns:
        Boolean mask, shape (B, H, W).
    """
    return labels != IGNORE_INDEX


def spread_mask(labels: Tensor, prev_fire: Tensor) -> Tensor:
    """Valid pixels that were not burning on day t.

    Pixels whose day-t state is uncertain are excluded too: a fire there tomorrow
    can't be told apart from fire that was already burning.

    Args:
        labels: Next-day fire mask, shape (B, H, W), values in {-1, 0, 1}.
        prev_fire: Day-t fire mask, shape (B, H, W), values in {-1, 0, 1}.

    Returns:
        Boolean mask, shape (B, H, W).
    """
    _check_same_shape(labels, prev_fire)
    return valid_mask(labels) & (prev_fire == NO_FIRE)


def pr_auc(probs: Tensor, labels: Tensor, mask: Tensor | None = None) -> float:
    """Area under the precision-recall curve, computed as average precision.

    AP = sum_n (R_n - R_{n-1}) * P_n over distinct score thresholds, i.e. the
    step-wise PR curve with no interpolation. This equals
    `sklearn.metrics.average_precision_score`. Whether it reproduces the
    published NDWS numbers is checked by the persistence and logistic-regression
    regression tests.

    Args:
        probs: Predicted fire probability, shape (B, H, W), float in [0, 1].
        labels: Next-day fire mask, shape (B, H, W), values in {-1, 0, 1}.
        mask: Optional extra boolean mask, shape (B, H, W), such as `spread_mask`.

    Returns:
        PR-AUC in [0, 1].

    Raises:
        ValueError: If the selected pixels contain no fire.
    """
    _, precision, recall = _pr_curve(*_select(probs, labels, mask))
    recall_step = torch.diff(recall, prepend=recall.new_zeros(1))
    return float((recall_step * precision).sum())


@dataclass(frozen=True)
class ThresholdScores:
    """Binary-decision scores at one probability threshold."""

    precision: float
    recall: float
    f1: float
    iou: float


def threshold_scores(
    probs: Tensor, labels: Tensor, threshold: float, mask: Tensor | None = None
) -> ThresholdScores:
    """Precision, recall, F1 and IoU of the fire class at `threshold`.

    A pixel is predicted as fire when `probs >= threshold`. A ratio whose
    denominator is zero is reported as 0.0, matching sklearn's `zero_division=0`.

    Args:
        probs: Predicted fire probability, shape (B, H, W), float in [0, 1].
        labels: Next-day fire mask, shape (B, H, W), values in {-1, 0, 1}.
        threshold: Decision threshold, chosen on the validation split only.
        mask: Optional extra boolean mask, shape (B, H, W).

    Returns:
        Scores for the fire class.
    """
    scores, targets = _select(probs, labels, mask)
    predicted = scores >= threshold
    tp = int((predicted & targets).sum())
    fp = int((predicted & ~targets).sum())
    fn = int((~predicted & targets).sum())
    return ThresholdScores(
        precision=_ratio(tp, tp + fp),
        recall=_ratio(tp, tp + fn),
        f1=_ratio(2 * tp, 2 * tp + fp + fn),
        iou=_ratio(tp, tp + fp + fn),
    )


def best_f1_threshold(probs: Tensor, labels: Tensor, mask: Tensor | None = None) -> float:
    """Threshold that maximizes F1 on the given pixels.

    Call this on the validation split only. Choosing the threshold on the test
    split inflates the reported scores (the criticism in Esparza et al. 2026).

    Args:
        probs: Predicted fire probability, shape (B, H, W), float in [0, 1].
        labels: Next-day fire mask, shape (B, H, W), values in {-1, 0, 1}.
        mask: Optional extra boolean mask, shape (B, H, W).

    Returns:
        The score threshold with the highest F1. Ties go to the higher threshold.

    Raises:
        ValueError: If the selected pixels contain no fire.
    """
    thresholds, precision, recall = _pr_curve(*_select(probs, labels, mask))
    f1 = 2 * precision * recall / (precision + recall).clamp_min(torch.finfo(torch.float64).tiny)
    return float(thresholds[torch.argmax(f1)])


def evaluate(
    probs: Tensor, labels: Tensor, prev_fire: Tensor, threshold: float
) -> dict[str, float]:
    """Full metric set over all valid pixels and over the spread region.

    Args:
        probs: Predicted fire probability, shape (B, H, W), float in [0, 1].
        labels: Next-day fire mask, shape (B, H, W), values in {-1, 0, 1}.
        prev_fire: Day-t fire mask, shape (B, H, W), values in {-1, 0, 1}.
        threshold: Decision threshold, chosen on the validation split only.

    Returns:
        Metrics keyed `"<region>/<metric>"`, where the region is `all` or
        `spread`, plus `<region>/fire_rate` (the share of selected pixels that
        burn on day t+1).
    """
    results: dict[str, float] = {}
    for region, mask in (("all", None), ("spread", spread_mask(labels, prev_fire))):
        _, targets = _select(probs, labels, mask)
        results[f"{region}/pr_auc"] = pr_auc(probs, labels, mask)
        for name, value in asdict(threshold_scores(probs, labels, threshold, mask)).items():
            results[f"{region}/{name}"] = value
        results[f"{region}/fire_rate"] = float(targets.double().mean())
    return results


@dataclass(frozen=True)
class ConfidenceInterval:
    """Point estimate with a bootstrap percentile interval."""

    estimate: float
    low: float
    high: float


@dataclass(frozen=True)
class PairedComparison:
    """Paired bootstrap comparison of model A against model B on the same tiles."""

    difference: ConfidenceInterval  # metric(A) - metric(B)
    p_value: float  # one-sided, H1: A is better than B


def bootstrap_ci(
    metric: MetricFn,
    *tensors: Tensor,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> ConfidenceInterval:
    """Bootstrap percentile confidence interval for `metric(*tensors)`.

    Whole tiles (dimension 0) are resampled, not pixels. Neighbouring pixels in a
    tile are strongly correlated, so resampling pixels would give intervals that
    are far too narrow.

    Every resample sorts all pixels again, so keep the tensors on the GPU. On the
    7.7M-pixel validation split one `pr_auc` takes about 19 ms on GB10 and about
    0.6 s on the CPU.

    Args:
        metric: Function of `tensors` returning a float, for example `pr_auc`.
        *tensors: Inputs to `metric`, all with the same first (tile) dimension.
        n_resamples: Number of bootstrap resamples.
        confidence: Two-sided coverage of the interval.
        seed: Seed for the resampling generator, so intervals are reproducible.

    Returns:
        The full-data estimate and the percentile interval.
    """
    _check_tile_counts(tensors)
    estimate = metric(*tensors)
    samples = torch.tensor(
        [
            metric(*(t[idx] for t in tensors))
            for idx in _resample_indices(tensors, n_resamples, seed)
        ],
        dtype=torch.float64,
    )
    low, high = _percentiles(samples, confidence)
    return ConfidenceInterval(estimate, low, high)


def paired_bootstrap(
    metric: MetricFn,
    probs_a: Tensor,
    probs_b: Tensor,
    *shared: Tensor,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> PairedComparison:
    """Paired bootstrap test of `metric(probs_a, *shared) - metric(probs_b, *shared)`.

    Both models are scored on the same resampled tiles. How hard each tile is
    then cancels out of the difference, which makes this test much more
    sensitive than comparing two separate intervals.

    Args:
        metric: Function `(probs, *shared) -> float`, for example `pr_auc`.
        probs_a: Model A probabilities, shape (B, H, W).
        probs_b: Model B probabilities, shape (B, H, W).
        *shared: Remaining `metric` inputs, such as labels, shape (B, H, W).
        n_resamples: Number of bootstrap resamples.
        confidence: Two-sided coverage of the interval on the difference.
        seed: Seed for the resampling generator.

    Returns:
        The difference with its interval, and the one-sided p-value
        `(1 + #{resamples with diff <= 0}) / (1 + n_resamples)`.
    """
    tensors = (probs_a, probs_b, *shared)
    _check_tile_counts(tensors)
    estimate = metric(probs_a, *shared) - metric(probs_b, *shared)
    diffs = []
    for idx in _resample_indices(tensors, n_resamples, seed):
        rest = [t[idx] for t in shared]
        diffs.append(metric(probs_a[idx], *rest) - metric(probs_b[idx], *rest))
    samples = torch.tensor(diffs, dtype=torch.float64)
    low, high = _percentiles(samples, confidence)
    p_value = (1 + int((samples <= 0).sum())) / (1 + n_resamples)
    return PairedComparison(ConfidenceInterval(estimate, low, high), p_value)


def _select(probs: Tensor, labels: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor]:
    """Flatten to the scored pixels: (scores, is_fire), both shape (N,)."""
    _check_same_shape(probs, labels)
    keep = valid_mask(labels)
    if mask is not None:
        _check_same_shape(labels, mask)
        keep &= mask
    return probs[keep].float(), labels[keep] == FIRE


def _pr_curve(scores: Tensor, targets: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Step-wise PR curve at each distinct score, highest threshold first.

    Returns:
        (thresholds, precision, recall), each shape (K,) for K distinct scores,
        with precision and recall in float64.
    """
    n_fire = int(targets.sum())
    if n_fire == 0:
        raise ValueError("no fire pixels among the selected pixels; PR metrics are undefined")

    order = torch.argsort(scores, descending=True, stable=True)
    scores, targets = scores[order], targets[order]
    # float64: on a full split the counts reach millions, beyond float32's exact range.
    true_pos = torch.cumsum(targets, dim=0, dtype=torch.float64)

    # Pixels with equal scores fall on the same side of any threshold, so each run
    # of tied scores is one point on the curve. Keep the last position of each run.
    run_end = torch.ones_like(targets)
    run_end[:-1] = scores[1:] != scores[:-1]
    predicted_pos = torch.nonzero(run_end).squeeze(1).to(torch.float64) + 1

    true_pos = true_pos[run_end]
    return scores[run_end], true_pos / predicted_pos, true_pos / n_fire


def _resample_indices(tensors: tuple[Tensor, ...], n_resamples: int, seed: int):
    """Yield `n_resamples` index vectors that draw tiles with replacement."""
    n_tiles = tensors[0].shape[0]
    generator = torch.Generator().manual_seed(seed)
    device = tensors[0].device
    for _ in range(n_resamples):
        yield torch.randint(n_tiles, (n_tiles,), generator=generator).to(device)


def _percentiles(samples: Tensor, confidence: float) -> tuple[float, float]:
    alpha = (1 - confidence) / 2
    low, high = torch.quantile(samples, torch.tensor([alpha, 1 - alpha], dtype=torch.float64))
    return float(low), float(high)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _check_tile_counts(tensors: tuple[Tensor, ...]) -> None:
    # Checked before any metric call, so a mismatch gets this clear error instead
    # of a confusing shape error from inside the metric.
    if any(t.shape[0] != tensors[0].shape[0] for t in tensors):
        raise ValueError("all inputs must have the same number of tiles (dimension 0)")


def _check_same_shape(a: Tensor, b: Tensor) -> None:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
