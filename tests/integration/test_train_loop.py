import csv

import pytest
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from wfspread.constants import CH, IGNORE_INDEX
from wfspread.train.config import TrainConfig
from wfspread.train.run import fit


class _FakeTiles(Dataset):
    """Tiles where tomorrow's fire is today's fire grown one pixel to the right."""

    def __init__(self, n_tiles: int, size: int = 16, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.inputs = torch.randn(n_tiles, 13, size, size, generator=g)
        prev = torch.zeros(n_tiles, size, size, dtype=torch.int8)
        for i in range(n_tiles):
            r, c = torch.randint(2, size - 6, (2,), generator=g).tolist()
            prev[i, r : r + 3, c : c + 3] = 1
        next_fire = prev.clone()
        next_fire[:, :, 1:] |= prev[:, :, :-1]
        next_fire[:, 0, :] = IGNORE_INDEX
        self.inputs[:, CH["prev_fire"]] = prev.float()
        self.prev_fire, self.next_fire = prev, next_fire

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        return {
            "inputs": self.inputs[i],
            "prev_fire": self.prev_fire[i],
            "next_fire": self.next_fire[i],
            "sample_id": torch.tensor(i, dtype=torch.int64),
        }


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(13, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 1, 1))

    def forward(self, inputs):
        return self.net(inputs).squeeze(1)


def _train(tmp_path, accelerator="cpu", max_epochs=8):
    config = TrainConfig(max_epochs=max_epochs, lr=1e-2, warmup_epochs=1, check_batches=True)
    torch.manual_seed(0)
    model = _TinyNet()
    train = DataLoader(_FakeTiles(32), batch_size=8, shuffle=True, drop_last=True)
    val = DataLoader(_FakeTiles(16, seed=1), batch_size=8)
    run_dir = fit(model, train, val, config, "tiny_test", tmp_path, accelerator=accelerator)
    rows = list(csv.DictReader((run_dir / "logs" / "metrics.csv").open()))
    return run_dir, rows


def _values(rows, key):
    return [float(r[key]) for r in rows if r.get(key)]


def _last(rows, key):
    return _values(rows, key)[-1]


def test_fit_learns_and_records_the_run(tmp_path):
    run_dir, rows = _train(tmp_path)

    pr_auc = _values(rows, "val/all/pr_auc")
    assert pr_auc[-1] > pr_auc[0]
    assert pr_auc[-1] > 2 * _last(rows, "val/all/fire_rate")
    assert _last(rows, "train/loss_epoch") < _values(rows, "train/loss_epoch")[0]
    assert _last(rows, "val/spread/pr_auc") > 0
    assert _last(rows, "time/epoch_s") > 0
    assert list((run_dir / "checkpoints").glob("best-*.ckpt"))
    info = yaml.safe_load((run_dir / "run.yaml").read_text())
    assert info["config"]["seed"] == 0
    assert info["batch_size"] == 8
    assert info["git_commit"]


def test_same_seed_gives_the_same_result(tmp_path):
    _, first = _train(tmp_path / "a", max_epochs=2)
    _, second = _train(tmp_path / "b", max_epochs=2)
    assert _last(first, "val/all/pr_auc") == _last(second, "val/all/pr_auc")


@pytest.mark.gpu
def test_fit_runs_on_the_gpu(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    _, rows = _train(tmp_path, accelerator="gpu", max_epochs=1)
    assert _last(rows, "memory/peak_gpu_gb") > 0
