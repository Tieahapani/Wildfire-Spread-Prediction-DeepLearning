import pytest
import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits

from wfspread.constants import CH, IGNORE_INDEX
from wfspread.eval.metrics import pr_auc
from wfspread.train.losses import masked_bce
from wfspread.train.sanity import mask_check, overfit_check, shuffled_label_check


def _tiles(n_tiles: int, seed: int, size: int = 16) -> dict:
    """Fake tiles where tomorrow's fire is today's fire grown one pixel to the right."""
    g = torch.Generator().manual_seed(seed)
    inputs = torch.randn(n_tiles, 13, size, size, generator=g)
    prev = torch.zeros(n_tiles, size, size, dtype=torch.int8)
    for i in range(n_tiles):
        r, c = torch.randint(2, size - 6, (2,), generator=g).tolist()
        prev[i, r : r + 3, c : c + 3] = 1
    next_fire = prev.clone()
    next_fire[:, :, 1:] |= prev[:, :, :-1]
    next_fire[:, 0, :] = IGNORE_INDEX
    inputs[:, CH["prev_fire"]] = prev.float()
    return {
        "inputs": inputs,
        "prev_fire": prev,
        "next_fire": next_fire,
        "sample_id": torch.arange(n_tiles, dtype=torch.int64),
    }


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(13, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 1, 1))

    def forward(self, inputs):
        return self.net(inputs).squeeze(1)


def test_overfit_passes_for_a_working_model():
    result = overfit_check(_TinyNet, _tiles(10, seed=0))
    assert result.passed, result


def test_overfit_fails_when_the_model_cannot_learn():
    result = overfit_check(_TinyNet, _tiles(10, seed=0), lr=0.0)
    assert not result.passed


def test_shuffled_labels_fall_to_about_the_fire_rate():
    train = [_tiles(8, seed=s) for s in range(4)]
    result = shuffled_label_check(_TinyNet, train, [_tiles(16, seed=10)])
    assert result.passed, result


def test_real_labels_score_well_above_the_shuffled_limit():
    torch.manual_seed(0)
    model = _TinyNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    train = [_tiles(8, seed=s) for s in range(4)]
    for step in range(300):
        batch = train[step % len(train)]
        loss = masked_bce(model(batch["inputs"]), batch["next_fire"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    val = _tiles(16, seed=10)
    with torch.no_grad():
        score = pr_auc(torch.sigmoid(model(val["inputs"])), val["next_fire"])
    assert score > shuffled_label_check(_TinyNet, train, [val]).limit


def test_mask_passes_with_the_project_loss():
    result = mask_check(_TinyNet(), _tiles(4, seed=0))
    assert result.passed, result


def test_mask_fails_for_a_loss_that_counts_unknown_pixels():
    def unmasked_bce(logits, next_fire):
        return binary_cross_entropy_with_logits(logits, (next_fire == 1).float())

    result = mask_check(_TinyNet(), _tiles(4, seed=0), loss_fn=unmasked_bce)
    assert not result.passed


def test_mask_needs_unknown_pixels():
    tiles = _tiles(2, seed=0)
    tiles["next_fire"][tiles["next_fire"] == IGNORE_INDEX] = 0
    with pytest.raises(ValueError, match="unknown"):
        mask_check(_TinyNet(), tiles)
