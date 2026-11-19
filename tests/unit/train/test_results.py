import csv

import pytest

from wfspread.train.results import REPORTED_METRICS, best_validation, summarize


def _write_metrics(run_dir, val_scores):
    logs = run_dir / "logs"
    logs.mkdir(parents=True)
    fields = ["epoch", "step", "train/loss_epoch", *REPORTED_METRICS]
    with (logs / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for epoch, score in enumerate(val_scores):
            writer.writerow({"epoch": epoch, "step": epoch, "train/loss_epoch": 0.5})
            writer.writerow({"epoch": epoch, "step": epoch, **{m: score for m in REPORTED_METRICS}})


def test_best_validation_picks_the_epoch_with_highest_pr_auc(tmp_path):
    _write_metrics(tmp_path, [0.1, 0.3, 0.2])

    best = best_validation(tmp_path)

    assert best["epoch"] == 1
    assert best["epochs"] == 3
    assert best["val/all/pr_auc"] == pytest.approx(0.3)


def test_best_validation_without_validation_epochs_raises(tmp_path):
    _write_metrics(tmp_path, [])

    with pytest.raises(ValueError, match="no validation epoch"):
        best_validation(tmp_path)


def test_summarize_averages_per_group_and_ranks_by_pr_auc():
    runs = [
        {"arm": "bce", "val/all/pr_auc": 0.28, "val/all/f1": 0.30},
        {"arm": "bce", "val/all/pr_auc": 0.30, "val/all/f1": 0.34},
        {"arm": "focal", "val/all/pr_auc": 0.35, "val/all/f1": 0.40},
    ]

    rows = summarize(runs, group_by="arm", metrics=("val/all/pr_auc", "val/all/f1"))

    assert [row["arm"] for row in rows] == ["focal", "bce"]
    assert rows[1]["n"] == 2
    assert rows[1]["val/all/pr_auc_mean"] == pytest.approx(0.29)
    assert rows[1]["val/all/pr_auc_std"] == pytest.approx(0.0141421, rel=1e-4)
    assert rows[0]["val/all/f1_std"] == 0.0
