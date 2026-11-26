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

## Final models (Nov 25)

Measured on gx10-6d0c (NVIDIA GB10) in one session, cuDNN benchmark off, with nothing else on the GPU. Saved in `$WF_RUNS/final-eval/efficiency.json`.

```bash
bin/wf python scripts/eval/efficiency.py --model configs/model/logreg.yaml \
    configs/model/mlp.yaml configs/model/unet.yaml configs/model/unet-optuna.yaml \
    --run <the 12 final run folders> --out "$WF_RUNS/final-eval/efficiency.json"
```

| Model | Parameters | GFLOPs/tile | ms/tile, batch 1 | ms/tile, batch 64 | Test PR-AUC |
|---|---|---|---|---|---|
| Logistic regression | 118 | 0.001 | 0.027 | 0.002 | 0.202 |
| MLP 3 × 3 | 48,257 | 0.392 | 0.072 | 0.217 | 0.232 |
| U-Net (final) | 1,929,889 | 1.176 | 0.765 | 0.244 | 0.351 |
| U-Net, tuned | 9,107,313 | 3.158 | 0.970 | 0.366 | 0.355 |

Training, mean over the 3 seeds (batch 64, full tiles):

| Model | Epochs | GPU-hours | Hours to best epoch | Peak training GB | Machine |
|---|---|---|---|---|---|
| Logistic regression | 78 | 0.08 | 0.07 | 0.03 | gx10-6d0c |
| MLP 3 × 3 | 31 | 0.11 | 0.06 | 1.06 | gx10-6d0c |
| U-Net (final) | 31 | 0.13 | 0.06 | 0.76 | gx10-6d0c |
| U-Net, tuned | 36 | 0.24 | 0.14 | 1.25 | spark-61dd |

- **The MLP is not much faster than the U-Net at batch 64** (0.217 vs 0.244 ms per tile), although it needs a third of the FLOPs. It keeps 128 channels at the full 64 × 64 resolution in every layer, so it is limited by memory traffic. Its time per tile at batch 64 is higher than at batch 1; two repeats gave the same numbers (0.218 and 0.219 ms).
- **With `--cudnn-benchmark`**, batch 64 drops to 0.164 ms for the MLP, 0.187 for the U-Net and 0.326 for the tuned U-Net; the order does not change.
- **The tuned U-Net** costs 4.7 times the parameters, 2.7 times the FLOPs and 1.5 times the inference time of the final U-Net, for +0.005 test PR-AUC.
- Training times come from logged epoch times. The tuned U-Net trained on the other Spark, so compare its hours with care.
