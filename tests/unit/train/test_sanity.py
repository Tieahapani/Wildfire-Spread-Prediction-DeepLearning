import pytest
import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits

from wfspread.constants import IGNORE_INDEX
from wfspread.eval.metrics import pr_auc
from wfspread.train.losses import masked_bce
from wfspread.train.sanity import mask_check, overfit_check, shuffled_label_check


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(13, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 1, 1))

    def forward(self, inputs):
        return self.net(inputs).squeeze(1)


def test_overfit_passes_for_a_working_model(fake_tiles):
    result = overfit_check(_TinyNet, fake_tiles(10, seed=0))
    assert result.passed, result


def test_overfit_fails_when_the_model_cannot_learn(fake_tiles):
    result = overfit_check(_TinyNet, fake_tiles(10, seed=0), lr=0.0)
    assert not result.passed


def test_shuffled_labels_fall_to_about_the_fire_rate(fake_tiles):
    train = [fake_tiles(8, seed=s) for s in range(4)]
    result = shuffled_label_check(_TinyNet, train, [fake_tiles(16, seed=10)])
    assert result.passed, result


def test_real_labels_score_well_above_the_shuffled_limit(fake_tiles):
    torch.manual_seed(0)
    model = _TinyNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    train = [fake_tiles(8, seed=s) for s in range(4)]
    for step in range(300):
        batch = train[step % len(train)]
        loss = masked_bce(model(batch["inputs"]), batch["next_fire"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    val = fake_tiles(16, seed=10)
    with torch.no_grad():
        score = pr_auc(torch.sigmoid(model(val["inputs"])), val["next_fire"])
    assert score > shuffled_label_check(_TinyNet, train, [val]).limit


def test_mask_passes_with_the_project_loss(fake_tiles):
    result = mask_check(_TinyNet(), fake_tiles(4, seed=0))
    assert result.passed, result


def test_mask_fails_for_a_loss_that_counts_unknown_pixels(fake_tiles):
    def unmasked_bce(logits, next_fire):
        return binary_cross_entropy_with_logits(logits, (next_fire == 1).float())

    result = mask_check(_TinyNet(), fake_tiles(4, seed=0), loss_fn=unmasked_bce)
    assert not result.passed


def test_mask_needs_unknown_pixels(fake_tiles):
    tiles = fake_tiles(2, seed=0)
    tiles["next_fire"][tiles["next_fire"] == IGNORE_INDEX] = 0
    with pytest.raises(ValueError, match="unknown"):
        mask_check(_TinyNet(), tiles)
