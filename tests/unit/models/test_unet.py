import pytest
import torch

from wfspread.models.unet import UNet
from wfspread.train.sanity import mask_check, overfit_check, shuffled_label_check


@pytest.mark.parametrize("size", [64, 32])
def test_returns_logits_for_tiles_and_crops(size):
    logits = UNet().eval()(torch.randn(2, 13, size, size))
    assert logits.shape == (2, size, size)
    assert logits.dtype == torch.float32


def test_has_about_two_million_parameters():
    count = sum(p.numel() for p in UNet().parameters())
    assert 1.8e6 < count < 2.1e6


def test_every_parameter_gets_a_gradient():
    model = UNet()
    model(torch.randn(2, 13, 16, 16)).sum().backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing


def test_eval_mode_is_deterministic():
    model = UNet().eval()
    inputs = torch.randn(2, 13, 16, 16)
    torch.testing.assert_close(model(inputs), model(inputs))


def test_passes_the_overfit_check(fake_tiles):
    result = overfit_check(UNet, fake_tiles(10, seed=0), steps=100, lr=1e-3)
    assert result.passed, result


def test_passes_the_shuffled_label_check(fake_tiles):
    train = [fake_tiles(8, seed=s) for s in range(4)]
    result = shuffled_label_check(UNet, train, [fake_tiles(16, seed=10)], steps=100, lr=1e-3)
    assert result.passed, result


def test_passes_the_mask_check(fake_tiles):
    result = mask_check(UNet(), fake_tiles(4, seed=0))
    assert result.passed, result
