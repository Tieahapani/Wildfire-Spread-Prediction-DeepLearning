import math

import pytest
import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from wfspread.constants import FIRE, IGNORE_INDEX
from wfspread.train.losses import make_loss, masked_bce, masked_dice, masked_focal


def _example(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(2, 4, 4, generator=g)
    next_fire = torch.randint(0, 2, (2, 4, 4), generator=g, dtype=torch.int8)
    next_fire[0, 0, :2] = IGNORE_INDEX
    return logits, next_fire


def test_matches_bce_on_valid_pixels():
    logits, next_fire = _example()
    valid = next_fire != IGNORE_INDEX
    expected = binary_cross_entropy_with_logits(logits[valid], (next_fire[valid] == FIRE).float())
    assert masked_bce(logits, next_fire) == pytest.approx(expected.item())


def test_ignored_pixels_do_not_change_the_loss():
    logits, next_fire = _example()
    changed = logits.clone()
    changed[next_fire == IGNORE_INDEX] = 100.0
    assert masked_bce(changed, next_fire) == pytest.approx(masked_bce(logits, next_fire).item())


def test_pos_weight_raises_the_cost_of_missed_fire():
    logits = torch.full((1, 2, 2), -2.0)
    next_fire = torch.tensor([[[1, 0], [0, 0]]], dtype=torch.int8)
    assert masked_bce(logits, next_fire, pos_weight=3.0) > masked_bce(logits, next_fire)


def test_all_unknown_gives_zero_and_a_gradient():
    logits = torch.randn(1, 3, 3, requires_grad=True)
    loss = masked_bce(logits, torch.full((1, 3, 3), IGNORE_INDEX, dtype=torch.int8))
    loss.backward()
    assert loss.item() == 0.0
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_bf16_logits_give_a_float32_loss():
    logits, next_fire = _example()
    assert masked_bce(logits.bfloat16(), next_fire).dtype == torch.float32


def test_shape_mismatch_fails():
    with pytest.raises(ValueError, match="logits"):
        masked_bce(torch.zeros(1, 1, 4, 4), torch.zeros(1, 4, 4, dtype=torch.int8))


ALL_LOSSES = {
    "bce": masked_bce,
    "focal": masked_focal,
    "dice": masked_dice,
    "combo": make_loss({"bce": 0.4, "dice": 0.3, "focal": 0.3}, pos_weight=3.0),
}


@pytest.mark.parametrize("name", ALL_LOSSES)
def test_every_loss_ignores_unknown_pixels(name):
    loss_fn = ALL_LOSSES[name]
    logits, next_fire = _example()
    changed = logits.clone()
    changed[next_fire == IGNORE_INDEX] = 100.0
    assert loss_fn(changed, next_fire) == pytest.approx(loss_fn(logits, next_fire).item())


@pytest.mark.parametrize("name", ALL_LOSSES)
def test_every_loss_is_zero_with_a_gradient_when_all_unknown(name):
    logits = torch.randn(1, 3, 3, requires_grad=True)
    loss = ALL_LOSSES[name](logits, torch.full((1, 3, 3), IGNORE_INDEX, dtype=torch.int8))
    loss.backward()
    assert loss.item() == 0.0
    assert torch.equal(logits.grad, torch.zeros_like(logits))


@pytest.mark.parametrize("name", ALL_LOSSES)
def test_every_loss_is_float32_for_bf16_logits(name):
    logits, next_fire = _example()
    assert ALL_LOSSES[name](logits.bfloat16(), next_fire).dtype == torch.float32


def test_focal_without_focusing_is_half_of_bce():
    logits, next_fire = _example()
    focal = masked_focal(logits, next_fire, gamma=0.0, alpha=0.5)
    assert focal == pytest.approx(0.5 * masked_bce(logits, next_fire).item())


def test_focal_matches_a_hand_computed_pixel():
    # One fire pixel predicted at p = 0.8: BCE = -ln 0.8, scaled by 0.25 * 0.2 ** 2.
    logits = torch.tensor([[[math.log(4.0)]]])
    next_fire = torch.ones(1, 1, 1, dtype=torch.int8)
    expected = 0.25 * 0.2**2 * -math.log(0.8)
    assert masked_focal(logits, next_fire) == pytest.approx(expected, rel=1e-5)


def test_focal_shrinks_easy_pixels_more_than_hard_ones():
    next_fire = torch.tensor([[[1, 0]]], dtype=torch.int8)
    easy = torch.tensor([[[4.0, -4.0]]])
    hard = torch.tensor([[[-1.0, 1.0]]])
    easy_ratio = masked_focal(easy, next_fire) / masked_bce(easy, next_fire)
    hard_ratio = masked_focal(hard, next_fire) / masked_bce(hard, next_fire)
    assert easy_ratio < hard_ratio / 10


def test_dice_matches_a_hand_computed_batch():
    # Probabilities 0.5 everywhere; two fire pixels of four valid ones.
    logits = torch.zeros(1, 2, 3)
    next_fire = torch.tensor([[[1, 1, 0], [0, -1, -1]]], dtype=torch.int8)
    expected = 1 - (2 * 1.0 + 1) / (2.0 + 2 + 1)
    assert masked_dice(logits, next_fire) == pytest.approx(expected)


def test_dice_is_near_zero_for_a_perfect_prediction():
    next_fire = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.int8)
    logits = torch.where(next_fire == 1, 20.0, -20.0)
    assert masked_dice(logits, next_fire) < 1e-6


def test_combined_loss_is_the_weighted_sum():
    logits, next_fire = _example()
    combo = make_loss({"bce": 0.4, "dice": 0.3, "focal": 0.3}, pos_weight=3.0)
    expected = (
        0.4 * masked_bce(logits, next_fire, pos_weight=3.0)
        + 0.3 * masked_dice(logits, next_fire)
        + 0.3 * masked_focal(logits, next_fire)
    )
    assert combo(logits, next_fire) == pytest.approx(expected.item())


@pytest.mark.parametrize(
    ("terms", "message"),
    [({}, "at least one"), ({"lovasz": 1.0}, "unknown loss"), ({"bce": 0.0}, "positive")],
)
def test_make_loss_rejects_bad_terms(terms, message):
    with pytest.raises(ValueError, match=message):
        make_loss(terms)
