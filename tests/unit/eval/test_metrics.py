import pytest
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from wfspread.constants import FIRE, IGNORE_INDEX, NO_FIRE
from wfspread.eval.metrics import (
    best_f1_threshold,
    bootstrap_ci,
    evaluate,
    paired_bootstrap,
    pr_auc,
    spread_mask,
    threshold_scores,
)


def _random_batch(seed: int = 0, n_tiles: int = 8, size: int = 16, fire_rate: float = 0.1):
    """Labels with fire, no-fire and ignore pixels, plus probabilities correlated with fire."""
    g = torch.Generator().manual_seed(seed)
    labels = (torch.rand(n_tiles, size, size, generator=g) < fire_rate).long()
    labels[torch.rand(n_tiles, size, size, generator=g) < 0.05] = IGNORE_INDEX
    noise = torch.rand(n_tiles, size, size, generator=g)
    probs = (0.5 * (labels == FIRE).float() + noise).clamp(0, 1)
    return probs, labels


# --- PR-AUC -------------------------------------------------------------------


def test_pr_auc_matches_sklearn():
    probs, labels = _random_batch()
    keep = labels != IGNORE_INDEX
    expected = average_precision_score(labels[keep].numpy(), probs[keep].numpy())
    assert pr_auc(probs, labels) == pytest.approx(expected, abs=1e-12)


def test_pr_auc_matches_sklearn_with_tied_scores():
    probs, labels = _random_batch(seed=1)
    probs = (probs * 4).round() / 4  # only 5 distinct scores, so many ties
    keep = labels != IGNORE_INDEX
    expected = average_precision_score(labels[keep].numpy(), probs[keep].numpy())
    assert pr_auc(probs, labels) == pytest.approx(expected, abs=1e-12)


def test_pr_auc_perfect_ranking_is_one():
    labels = torch.tensor([[[1, 0], [0, 1]]])
    probs = torch.tensor([[[0.9, 0.1], [0.2, 0.8]]])
    assert pr_auc(probs, labels) == pytest.approx(1.0)


def test_pr_auc_constant_prediction_equals_fire_rate():
    labels = torch.tensor([[[1, 0, 0, 0], [0, 0, 1, 0]]])
    assert pr_auc(torch.full(labels.shape, 0.3), labels) == pytest.approx(2 / 8)


def test_pr_auc_binary_prediction_closed_form():
    # A hard 0/1 predictor such as persistence has two curve points, giving
    # AP = R1 * P1 + (1 - R1) * fire_rate.
    labels = torch.tensor([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]]])
    probs = torch.tensor([[[1, 0, 0, 1, 0, 0, 0, 0, 0, 0]]]).float()
    precision_1, recall_1, fire_rate = 1 / 2, 1 / 3, 3 / 10
    expected = recall_1 * precision_1 + (1 - recall_1) * fire_rate
    assert pr_auc(probs, labels) == pytest.approx(expected)


def test_ignored_pixels_do_not_affect_scores():
    probs, labels = _random_batch(seed=2)
    scrambled = probs.clone()
    ignored = labels == IGNORE_INDEX
    scrambled[ignored] = torch.rand(int(ignored.sum()))
    assert pr_auc(scrambled, labels) == pr_auc(probs, labels)


def test_pr_auc_without_fire_raises():
    labels = torch.zeros(1, 4, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="no fire"):
        pr_auc(torch.rand(1, 4, 4), labels)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        pr_auc(torch.rand(1, 4, 4), torch.zeros(1, 4, 5, dtype=torch.long))


@pytest.mark.gpu
def test_pr_auc_same_on_gpu():
    probs, labels = _random_batch(seed=3)
    assert pr_auc(probs.cuda(), labels.cuda()) == pytest.approx(pr_auc(probs, labels), abs=1e-12)


# --- Spread region -------------------------------------------------------------


def test_spread_mask_excludes_burning_and_uncertain_day_t():
    prev_fire = torch.tensor([[[FIRE, NO_FIRE], [IGNORE_INDEX, NO_FIRE]]])
    labels = torch.tensor([[[FIRE, FIRE], [FIRE, IGNORE_INDEX]]])
    expected = torch.tensor([[[False, True], [False, False]]])
    assert torch.equal(spread_mask(labels, prev_fire), expected)


def test_persistence_has_no_skill_in_spread_region():
    # Persistence predicts fire only where it burns today, which the spread
    # region removes, so it ranks all spread pixels equally.
    prev_fire = torch.tensor([[[1, 1, 0, 0], [0, 0, 0, 0]]])
    labels = torch.tensor([[[1, 1, 1, 0], [0, 0, 1, 0]]])
    mask = spread_mask(labels, prev_fire)
    assert pr_auc(prev_fire.float(), labels, mask) == pytest.approx(2 / 6)


# --- Threshold scores ----------------------------------------------------------


def test_threshold_scores_hand_built():
    # At threshold 0.5: TP = 2, FP = 1, FN = 1, so P = R = F1 = 2/3 and IoU = 1/2.
    labels = torch.tensor([[[1, 1, 1, 0], [0, 0, 0, IGNORE_INDEX]]])
    probs = torch.tensor([[[0.9, 0.6, 0.2, 0.7], [0.1, 0.3, 0.4, 0.99]]])
    scores = threshold_scores(probs, labels, threshold=0.5)
    assert scores.precision == pytest.approx(2 / 3)
    assert scores.recall == pytest.approx(2 / 3)
    assert scores.f1 == pytest.approx(2 / 3)
    assert scores.iou == pytest.approx(1 / 2)


def test_threshold_scores_match_sklearn():
    probs, labels = _random_batch(seed=4)
    keep = labels != IGNORE_INDEX
    y_true, y_pred = labels[keep].numpy(), (probs[keep] >= 0.6).numpy()
    scores = threshold_scores(probs, labels, threshold=0.6)
    assert scores.precision == pytest.approx(precision_score(y_true, y_pred))
    assert scores.recall == pytest.approx(recall_score(y_true, y_pred))
    assert scores.f1 == pytest.approx(f1_score(y_true, y_pred))


def test_threshold_scores_no_predicted_fire_gives_zero():
    labels = torch.tensor([[[1, 0]]])
    scores = threshold_scores(torch.tensor([[[0.1, 0.2]]]), labels, threshold=0.5)
    assert (scores.precision, scores.recall, scores.f1, scores.iou) == (0.0, 0.0, 0.0, 0.0)


def test_best_f1_threshold_hand_built():
    # Thresholds 0.9 / 0.6 / 0.4 / 0.1 give F1 = 0.5 / 0.8 / 0.57 / 0.75, so 0.6 is best.
    labels = torch.tensor([[[1, 1, 0, 0, 1]]])
    probs = torch.tensor([[[0.9, 0.6, 0.4, 0.4, 0.1]]])
    assert best_f1_threshold(probs, labels) == pytest.approx(0.6)


# --- Full evaluation -----------------------------------------------------------


def test_evaluate_reports_both_regions():
    probs, labels = _random_batch(seed=5)
    prev_fire = torch.zeros_like(labels)
    prev_fire[:, :4] = FIRE
    results = evaluate(probs, labels, prev_fire, threshold=0.5)
    metrics = ("pr_auc", "precision", "recall", "f1", "iou", "fire_rate")
    assert set(results) == {f"{r}/{m}" for r in ("all", "spread") for m in metrics}
    assert results["all/pr_auc"] == pr_auc(probs, labels)
    assert results["spread/pr_auc"] == pr_auc(probs, labels, spread_mask(labels, prev_fire))


# --- Bootstrap -----------------------------------------------------------------


def test_bootstrap_ci_is_reproducible_and_brackets_estimate():
    probs, labels = _random_batch(seed=6, n_tiles=20)
    ci = bootstrap_ci(pr_auc, probs, labels, n_resamples=200, seed=7)
    assert ci == bootstrap_ci(pr_auc, probs, labels, n_resamples=200, seed=7)
    assert ci.low < ci.estimate < ci.high


def test_bootstrap_rejects_mismatched_tile_counts():
    probs, labels = _random_batch(n_tiles=4)
    with pytest.raises(ValueError, match="same number of tiles"):
        bootstrap_ci(pr_auc, probs, labels[:3], n_resamples=2)


def test_paired_bootstrap_identical_models_show_no_difference():
    probs, labels = _random_batch(seed=8, n_tiles=20)
    result = paired_bootstrap(pr_auc, probs, probs, labels, n_resamples=100)
    assert result.difference == type(result.difference)(0.0, 0.0, 0.0)
    assert result.p_value == 1.0


def test_paired_bootstrap_detects_better_model():
    probs, labels = _random_batch(seed=9, n_tiles=20)
    noisy = (probs + torch.rand(probs.shape, generator=torch.Generator().manual_seed(0))).clamp(
        0, 1
    )
    result = paired_bootstrap(pr_auc, probs, noisy, labels, n_resamples=200)
    assert result.difference.estimate > 0
    assert result.difference.low > 0
    assert result.p_value < 0.01
