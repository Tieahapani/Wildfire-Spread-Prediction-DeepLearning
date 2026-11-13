# Efficiency

The report compares models on cost as well as accuracy: size, compute, inference speed and memory, and training time. All of it comes from `scripts/eval/efficiency.py`, built on `wfspread.eval.efficiency`.

```bash
bin/wf python scripts/eval/efficiency.py --model configs/model/unet.yaml
bin/wf python scripts/eval/efficiency.py --model configs/model/*.yaml \
    --run "$WF_RUNS"/<run-id> --out "$WF_RUNS"/efficiency.json
```

## What is measured

| Number | How |
|---|---|
| Parameters | Trainable parameters |
| GFLOPs per tile | PyTorch's FLOP counter on one 64 × 64 tile; a multiply-add counts as 2 |
| ms per tile, tiles per second | Median of 50 timed forward passes in eval mode after 10 untimed ones, at each batch size (default 1 and 64); GPU work is synchronized before each clock reading |
| Peak memory | Peak allocated GPU memory during the timed passes, weights included |
| GPU-hours, best epoch, hours to best | From a finished run's `logs/metrics.csv`: summed epoch times, and the epoch with the highest validation PR-AUC |
| Peak training memory | Highest per-epoch peak GPU memory logged during training |

Inference uses random inputs: speed and memory depend on the input shape, not the values.

## Reading the numbers

- **Report the conditions with the numbers:** device, batch size and whether `--cudnn-benchmark` was on. The JSON output records all of them.
- **cuDNN benchmark:** with `--cudnn-benchmark`, cuDNN tries several kernels for each input shape first. On the GB10 this changed the U-Net's time per tile at batch 64 from 0.24 to 0.18 ms, but the first pass took over a minute. Use the same setting for every model in a comparison.
- **Shared GPUs:** the Sparks run other jobs, so time each model in the same session. Repeat a measurement if a number looks off.
