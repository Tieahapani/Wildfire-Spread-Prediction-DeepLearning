import pytest
import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from wfspread.constants import FIRE, IGNORE_INDEX
from wfspread.train.losses import masked_bce


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
