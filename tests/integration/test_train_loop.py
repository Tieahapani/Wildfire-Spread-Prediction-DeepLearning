import csv

import pytest
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from wfspread.train.config import TrainConfig
from wfspread.train.run import fit


class _TileDataset(Dataset):
    def __init__(self, tiles):
        self.tiles = tiles

    def __len__(self):
        return len(self.tiles["inputs"])

    def __getitem__(self, i):
        return {key: value[i] for key, value in self.tiles.items()}


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(13, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 1, 1))

    def forward(self, inputs):
        return self.net(inputs).squeeze(1)


def _train(tmp_path, fake_tiles, accelerator="cpu", max_epochs=8):
    config = TrainConfig(
        max_epochs=max_epochs,
        lr=1e-2,
        warmup_epochs=1,
        check_batches=True,
        diagnostics_every_n_steps=2,
    )
    torch.manual_seed(0)
    model = _TinyNet()
    train = DataLoader(
        _TileDataset(fake_tiles(32, seed=0)), batch_size=8, shuffle=True, drop_last=True
    )
    val = DataLoader(_TileDataset(fake_tiles(16, seed=1)), batch_size=8)
    run_dir = fit(model, train, val, config, "tiny_test", tmp_path, accelerator=accelerator)
    rows = list(csv.DictReader((run_dir / "logs" / "metrics.csv").open()))
    return run_dir, rows


def _values(rows, key):
    return [float(r[key]) for r in rows if r.get(key)]


def _last(rows, key):
    return _values(rows, key)[-1]


def test_fit_learns_and_records_the_run(tmp_path, fake_tiles):
    run_dir, rows = _train(tmp_path, fake_tiles)

    pr_auc = _values(rows, "val/all/pr_auc")
    assert pr_auc[-1] > pr_auc[0]
    assert pr_auc[-1] > 2 * _last(rows, "val/all/fire_rate")
    assert _last(rows, "train/loss_epoch") < _values(rows, "train/loss_epoch")[0]
    assert _last(rows, "val/spread/pr_auc") > 0
    assert _last(rows, "time/epoch_s") > 0
    lr = _values(rows, "diag/lr")
    assert max(lr) == pytest.approx(1e-2, rel=0.05)
    assert lr[-1] < lr[len(lr) // 2]
    assert _last(rows, "diag/grad_norm") > 0
    assert _last(rows, "diag/update/net.0.weight") > 0
    assert list((run_dir / "checkpoints").glob("best-*.ckpt"))
    info = yaml.safe_load((run_dir / "run.yaml").read_text())
    assert info["config"]["seed"] == 0
    assert info["batch_size"] == 8
    assert info["git_commit"]


def test_same_seed_gives_the_same_result(tmp_path, fake_tiles):
    _, first = _train(tmp_path / "a", fake_tiles, max_epochs=2)
    _, second = _train(tmp_path / "b", fake_tiles, max_epochs=2)
    assert _last(first, "val/all/pr_auc") == _last(second, "val/all/pr_auc")


@pytest.mark.gpu
def test_fit_runs_on_the_gpu(tmp_path, fake_tiles):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    _, rows = _train(tmp_path, fake_tiles, accelerator="gpu", max_epochs=1)
    assert _last(rows, "memory/peak_gpu_gb") > 0
