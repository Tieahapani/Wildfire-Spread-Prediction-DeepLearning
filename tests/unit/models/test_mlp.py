import pytest
import torch

from wfspread.models.mlp import PixelMLP
from wfspread.train.sanity import mask_check, overfit_check, shuffled_label_check


@pytest.mark.parametrize("size", [64, 32])
def test_returns_logits_for_tiles_and_crops(size):
    logits = PixelMLP().eval()(torch.randn(2, 13, size, size))
    assert logits.shape == (2, size, size)
    assert logits.dtype == torch.float32


def test_sees_only_the_3x3_neighborhood():
    model = PixelMLP().eval()
    inputs = torch.randn(1, 13, 16, 16)
    changed = inputs.clone()
    changed[0, :, 8, 8] += 5.0
    diff = (model(changed) - model(inputs)).abs()[0] > 0
    expected = torch.zeros(16, 16, dtype=torch.bool)
    expected[7:10, 7:10] = True
    assert torch.equal(diff, expected)


def test_one_pixel_version_sees_only_its_pixel():
    model = PixelMLP(kernel_size=1).eval()
    inputs = torch.randn(1, 13, 8, 8)
    changed = inputs.clone()
    changed[0, :, 4, 4] += 5.0
    diff = (model(changed) - model(inputs)).abs()[0] > 0
    assert diff.sum() == 1
    assert diff[4, 4]


def test_rejects_an_even_kernel():
    with pytest.raises(ValueError, match="odd kernel_size"):
        PixelMLP(kernel_size=2)


def test_every_parameter_gets_a_gradient():
    model = PixelMLP()
    model(torch.randn(2, 13, 16, 16)).sum().backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing


def test_passes_the_overfit_check(fake_tiles):
    result = overfit_check(PixelMLP, fake_tiles(10, seed=0), steps=100, lr=1e-3)
    assert result.passed, result


def test_passes_the_shuffled_label_check(fake_tiles):
    train = [fake_tiles(8, seed=s) for s in range(4)]
    result = shuffled_label_check(PixelMLP, train, [fake_tiles(16, seed=10)], steps=100, lr=1e-3)
    assert result.passed, result


def test_passes_the_mask_check(fake_tiles):
    result = mask_check(PixelMLP(), fake_tiles(4, seed=0))
    assert result.passed, result
