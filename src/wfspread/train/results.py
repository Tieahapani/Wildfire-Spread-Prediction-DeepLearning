"""Read finished runs and compare training settings across seeds."""

from __future__ import annotations

import csv
import statistics
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from wfspread.train.module import MONITOR

# Validation metrics reported for each run, at its best epoch on MONITOR.
REPORTED_METRICS: tuple[str, ...] = (
    "val/all/pr_auc",
    "val/all/f1",
    "val/spread/pr_auc",
    "val/spread/f1",
)


def best_validation(run_dir: Path, monitor: str = MONITOR) -> dict[str, float]:
    """Validation metrics at the epoch with the highest `monitor`.

    This is the epoch the best checkpoint comes from.

    Args:
        run_dir: Run folder written by `wfspread.train.run.fit`.
        monitor: Metric that picks the epoch.

    Returns:
        `epoch`, `epochs` (validation epochs run) and every `REPORTED_METRICS` value.

    Raises:
        ValueError: If the run logged no validation epoch.
    """
    with (Path(run_dir) / "logs" / "metrics.csv").open() as f:
        rows = [row for row in csv.DictReader(f) if row.get(monitor)]
    if not rows:
        raise ValueError(f"{run_dir} has no validation epoch")
    best = max(rows, key=lambda row: float(row[monitor]))
    return {
        "epoch": int(best["epoch"]),
        "epochs": len(rows),
        **{name: float(best[name]) for name in REPORTED_METRICS},
    }


def summarize(
    runs: Iterable[Mapping[str, object]], group_by: str, metrics: Sequence[str] = REPORTED_METRICS
) -> list[dict[str, object]]:
    """Mean and sample standard deviation of each metric per group, best mean `MONITOR` first.

    Args:
        runs: One mapping per run, holding `group_by` and the metrics.
        group_by: Key that names the setting, for example the loss arm.
        metrics: Metrics to average.

    Returns:
        One row per group with `n`, `<metric>_mean` and `<metric>_std`. The standard
        deviation is 0 for a group with a single run.
    """
    groups: dict[object, list[Mapping[str, object]]] = {}
    for run in runs:
        groups.setdefault(run[group_by], []).append(run)

    rows = []
    for name, members in groups.items():
        row: dict[str, object] = {group_by: name, "n": len(members)}
        for metric in metrics:
            values = [float(member[metric]) for member in members]
            row[f"{metric}_mean"] = statistics.fmean(values)
            row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        rows.append(row)
    return sorted(rows, key=lambda row: row[f"{MONITOR}_mean"], reverse=True)
