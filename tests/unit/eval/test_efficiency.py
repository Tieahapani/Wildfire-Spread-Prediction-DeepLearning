import csv

import pytest
from torch import nn

from wfspread.eval.efficiency import (
    count_parameters,
    flops_per_tile,
    measure_inference,
    training_cost,
)


class _OneByOne(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(13, 1, 1)

    def forward(self, inputs):
        return self.conv(inputs).squeeze(1)


def test_count_parameters_skips_frozen_ones():
    model = nn.Sequential(nn.Conv2d(13, 1, 3), nn.Conv2d(1, 1, 1))
    model[1].requires_grad_(False)
    assert count_parameters(model) == 13 * 9 + 1


def test_flops_of_a_one_by_one_convolution():
    # One multiply-add (2 FLOPs) per input channel per pixel; the bias add is not counted.
    assert flops_per_tile(_OneByOne()) == 2 * 13 * 64 * 64


def test_inference_stats_are_consistent_on_cpu():
    stats = measure_inference(_OneByOne(), batch_size=4, size=16, warmup=1, repeats=3)
    assert stats.batch_size == 4
    assert stats.ms_per_tile == pytest.approx(stats.ms_per_batch / 4)
    assert stats.tiles_per_s == pytest.approx(4000 / stats.ms_per_batch)
    assert stats.peak_memory_gb is None


def _write_metrics(run_dir, rows):
    (run_dir / "logs").mkdir(parents=True)
    keys = ["epoch", "step", "time/epoch_s", "val/all/pr_auc", "memory/peak_gpu_gb"]
    with (run_dir / "logs" / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(rows)


def test_training_cost_finds_the_best_epoch_and_time_to_it(tmp_path):
    rows = []
    for epoch, (seconds, pr_auc, peak_gb) in enumerate(
        [(1800, 0.20, 1.5), (1800, 0.31, 1.6), (3600, 0.30, 1.4)]
    ):
        rows.append({"epoch": epoch, "step": epoch, "val/all/pr_auc": pr_auc})
        rows.append(
            {"epoch": epoch, "step": epoch, "time/epoch_s": seconds, "memory/peak_gpu_gb": peak_gb}
        )
    _write_metrics(tmp_path, rows)

    cost = training_cost(tmp_path)

    assert (cost.epochs, cost.best_epoch) == (3, 1)
    assert cost.train_hours == pytest.approx(2.0)
    assert cost.hours_to_best == pytest.approx(1.0)
    assert cost.peak_gpu_gb == pytest.approx(1.6)


def test_training_cost_of_a_cpu_run_has_no_gpu_memory(tmp_path):
    _write_metrics(
        tmp_path,
        [{"epoch": 0, "step": 0, "time/epoch_s": 36, "val/all/pr_auc": 0.1}],
    )
    assert training_cost(tmp_path).peak_gpu_gb is None


def test_training_cost_needs_validation_scores(tmp_path):
    _write_metrics(tmp_path, [{"epoch": 0, "step": 0, "time/epoch_s": 36}])
    with pytest.raises(ValueError, match="pr_auc"):
        training_cost(tmp_path)
