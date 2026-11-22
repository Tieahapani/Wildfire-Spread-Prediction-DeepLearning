"""Score the final models on validation, or once on test, with seed and bootstrap spread.

The models and comparisons are listed in `configs/eval/final.yaml`. Thresholds
always come from validation. The test split is scored only with `--final-test`,
and only if no test results exist yet: run it once, at the experiment freeze.

Examples:
    bin/wf python scripts/eval/final_eval.py                       # validation
    bin/wf python scripts/eval/final_eval.py --split test --final-test
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from dataclasses import asdict
from pathlib import Path

import torch

from wfspread.data.loader import NDWSDataset
from wfspread.eval.final import (
    PERSISTENCE,
    Predictions,
    best_checkpoint,
    build_scored_model,
    collect,
    compare,
    confidence_intervals,
    read_manifest,
    score_runs,
    stack_runs,
    validation_thresholds,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TABLE_METRICS = ("all/pr_auc", "all/f1", "all/iou", "spread/pr_auc", "spread/f1")


def main() -> None:
    """Predict every run, then write per-run scores, a summary and comparisons."""
    data_root = Path(os.environ.get("WF_DATA", REPO_ROOT / "data"))
    runs_root = Path(os.environ.get("WF_RUNS", REPO_ROOT / "runs"))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "configs/eval/final.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--final-test", action="store_true", help="required to score test")
    parser.add_argument("--data", type=Path, default=data_root / "processed" / "ndws-v1")
    parser.add_argument("--out", type=Path, default=runs_root / "final-eval")
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out = args.out / args.split
    if args.split == "test":
        if not args.final_test:
            parser.error("scoring test needs --final-test; run it once, at the freeze")
        if out.exists():
            parser.error(f"{out} exists: the test split has already been scored")
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    specs, comparisons = read_manifest(args.manifest, runs_root)
    val_set = NDWSDataset(args.data, "val", preload=True)
    scored_set = val_set if args.split == "val" else NDWSDataset(args.data, "test", preload=True)

    _, val_labels, _ = collect(None, val_set, device)
    _, labels, prev_fire = collect(None, scored_set, device)
    predictions: dict[str, Predictions] = {}
    per_run_rows = []
    for spec in specs:
        val_probs, scored_probs, run_names = [], [], []
        for run in [None] if spec.kind == PERSISTENCE else spec.runs:
            model = None
            if run is not None:
                model = build_scored_model(spec, best_checkpoint(run)).to(device)
            val_probs.append(collect(model, val_set, device)[0])
            same = args.split == "val"
            scored_probs.append(val_probs[-1] if same else collect(model, scored_set, device)[0])
            run_names.append(run.name if run is not None else PERSISTENCE)
        thresholds = validation_thresholds(stack_runs(val_probs), val_labels)
        predictions[spec.name] = Predictions(stack_runs(scored_probs), thresholds)
        for run_name, scores in zip(
            run_names, score_runs(predictions[spec.name], labels, prev_fire), strict=True
        ):
            per_run_rows.append({"model": spec.name, "run": run_name, **scores})
        print(f"scored {spec.name}: {len(run_names)} run(s)")

    intervals = {
        name: confidence_intervals(p, labels, prev_fire, args.n_resamples)
        for name, p in predictions.items()
    }
    paired = {
        f"{a} vs {b}": compare(predictions[a], predictions[b], labels, prev_fire, args.n_resamples)
        for a, b in comparisons
    }

    _write_csv(out / "per_run.csv", per_run_rows)
    summary = _summary_table(per_run_rows, intervals) + "\n\n" + _comparison_table(paired)
    (out / "summary.md").write_text(f"# Final evaluation on {args.split}\n\n{summary}\n")
    (out / "results.json").write_text(
        json.dumps(
            {
                "split": args.split,
                "n_resamples": args.n_resamples,
                "per_run": per_run_rows,
                "intervals": {
                    m: {k: asdict(v) for k, v in c.items()} for m, c in intervals.items()
                },
                "comparisons": {p: {k: asdict(v) for k, v in c.items()} for p, c in paired.items()},
            },
            indent=2,
        )
    )
    print(summary)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary_table(rows: list[dict[str, object]], intervals: dict) -> str:
    header = ["model", "runs", *TABLE_METRICS, "all/pr_auc 95% CI", "spread/pr_auc 95% CI"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for model, ci in intervals.items():
        runs = [row for row in rows if row["model"] == model]
        cells = [model, str(len(runs))]
        for metric in TABLE_METRICS:
            values = [float(row[metric]) for row in runs]
            spread = statistics.stdev(values) if len(values) > 1 else 0.0
            cells.append(f"{statistics.fmean(values):.3f} ± {spread:.3f}")
        for metric in ("all/pr_auc", "spread/pr_auc"):
            cells.append(f"{ci[metric].low:.3f}–{ci[metric].high:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _comparison_table(paired: dict) -> str:
    header = ["comparison", "metric", "difference", "95% CI", "p (A better)"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for pair, results in paired.items():
        for metric, result in results.items():
            d = result.difference
            lines.append(
                f"| {pair} | {metric} | {d.estimate:+.3f} | {d.low:+.3f} to {d.high:+.3f} "
                f"| {result.p_value:.3f} |"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
