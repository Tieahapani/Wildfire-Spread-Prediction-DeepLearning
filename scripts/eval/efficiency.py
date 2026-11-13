"""Measure model size, compute and inference speed, and optionally training cost.

Examples:
    bin/wf python scripts/eval/efficiency.py --model configs/model/unet.yaml
    bin/wf python scripts/eval/efficiency.py --model configs/model/unet.yaml \
        --run "$WF_RUNS/20261120-0930_unet_final-s0" --out efficiency.json
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict
from pathlib import Path

import torch

from wfspread.eval.efficiency import (
    count_parameters,
    flops_per_tile,
    measure_inference,
    training_cost,
)
from wfspread.models.registry import load_model


def main() -> None:
    """Print a Markdown table per model and optionally save everything as JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, nargs="+", required=True, help="model configs")
    parser.add_argument("--batch-size", type=int, nargs="+", default=[1, 64])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run", type=Path, nargs="*", default=[], help="finished run folders")
    parser.add_argument("--out", type=Path, help="JSON file for the results")
    parser.add_argument(
        "--cudnn-benchmark",
        action="store_true",
        help="let cuDNN try kernels per input shape first (slow to start, often faster)",
    )
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    results = {
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "torch": str(torch.__version__),
        "cudnn_benchmark": args.cudnn_benchmark,
        "host": platform.node(),
        "models": {},
        "runs": {},
    }
    print(f"Device: {results['gpu'] or args.device}, cuDNN benchmark: {args.cudnn_benchmark}\n")
    print("| Model | Parameters | GFLOPs/tile | Batch | ms/tile | tiles/s | Peak GB |")
    print("|---|---|---|---|---|---|---|")
    for path in args.model:
        torch.manual_seed(0)
        model = load_model(path).to(args.device)
        params, flops = count_parameters(model), flops_per_tile(model)
        speeds = [measure_inference(model, size) for size in args.batch_size]
        results["models"][path.stem] = {
            "config": str(path),
            "parameters": params,
            "flops_per_tile": flops,
            "inference": [asdict(s) for s in speeds],
        }
        for s in speeds:
            peak = f"{s.peak_memory_gb:.3f}" if s.peak_memory_gb is not None else "n/a"
            print(
                f"| {path.stem} | {params:,} | {flops / 1e9:.3f} | {s.batch_size} "
                f"| {s.ms_per_tile:.3f} | {s.tiles_per_s:,.0f} | {peak} |"
            )

    if args.run:
        print("\n| Run | Epochs | GPU-hours | Best epoch | Hours to best | Peak train GB |")
        print("|---|---|---|---|---|---|")
    for run_dir in args.run:
        cost = training_cost(run_dir)
        results["runs"][run_dir.name] = asdict(cost)
        peak = f"{cost.peak_gpu_gb:.2f}" if cost.peak_gpu_gb is not None else "n/a"
        print(
            f"| {run_dir.name} | {cost.epochs} | {cost.train_hours:.2f} | {cost.best_epoch} "
            f"| {cost.hours_to_best:.2f} | {peak} |"
        )

    if args.out:
        args.out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
