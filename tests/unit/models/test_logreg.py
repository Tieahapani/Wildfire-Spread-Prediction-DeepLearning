import pytest
import torch

from wfspread.models.logreg import LogisticRegression
from wfspread.train.sanity import mask_check


def test_returns_logits_for_tiles_and_crops():
    assert LogisticRegression()(torch.randn(2, 13, 32, 32)).shape == (2, 32, 32)


def test_has_one_weight_per_neighborhood_feature_plus_a_bias():
    assert sum(p.numel() for p in LogisticRegression().parameters()) == 13 * 9 + 1


def test_is_linear_in_its_inputs():
    model = LogisticRegression()
    a, b = torch.randn(1, 13, 8, 8), torch.randn(1, 13, 8, 8)
    bias = model.linear.bias
    torch.testing.assert_close(model(a + b) - bias, (model(a) - bias) + (model(b) - bias))


def test_sees_only_the_3x3_neighborhood():
    model = LogisticRegression()
    inputs = torch.randn(1, 13, 16, 16)
    changed = inputs.clone()
    changed[0, :, 8, 8] += 5.0
    diff = (model(changed) - model(inputs)).abs()[0] > 0
    expected = torch.zeros(16, 16, dtype=torch.bool)
    expected[7:10, 7:10] = True
    assert torch.equal(diff, expected)


def test_rejects_an_even_kernel():
    with pytest.raises(ValueError, match="must be odd"):
        LogisticRegression(kernel_size=2)


def test_passes_the_mask_check(fake_tiles):
    result = mask_check(LogisticRegression(), fake_tiles(4, seed=0))
    assert result.passed, result
