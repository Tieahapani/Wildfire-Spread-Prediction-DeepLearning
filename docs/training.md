# Training

All models train through one shared loop in `src/wfspread/train/`, so every model gets the same loss, metrics, schedule, stopping rule and logging. Only the model changes.

## Running

```python
from wfspread.train.config import TrainConfig
from wfspread.train.run import fit

config = TrainConfig.from_yaml("configs/train/default.yaml")
run_dir = fit(model, train_loader, val_loader, config, run_name="unet_baseline")
```

- `model` maps `inputs` `(B, C, H, W)` to fire logits `(B, H, W)`; see [`batch-format.md`](batch-format.md).
- Loaders return batches in the batch format. The training loader shuffles and uses `drop_last=True`; the validation loader does not shuffle.
- Runs that drop a channel pass `num_channels`.

## What the loop does

| Part | Choice |
|---|---|
| Loss | Masked binary cross-entropy over valid pixels, optional fire weight (`pos_weight`) |
| Optimizer | AdamW; linear warmup, then cosine decay to `min_lr`, updated every step |
| Gradients | Norm clipped at `grad_clip` |
| Validation | Predictions for the whole split are scored once per epoch with `wfspread.eval.metrics.evaluate`. F1 and IoU use the best-F1 threshold on validation |
| Stopping | Early stopping on `val/all/pr_auc` after `patience` epochs without improvement |
| Checkpoints | Best epoch by `val/all/pr_auc`, plus the last epoch |
| Reproducibility | One seed for torch, numpy, Python and loader workers |
| Precision | float32 by default; logits and loss are always float32 |

Settings and defaults are in `configs/train/default.yaml`; unknown keys are rejected.

## Run folder

Each run writes to `$WF_RUNS/<YYYYMMDD-HHMM>_<model>_<tag>/`:

- `run.yaml`: settings, batch size, git commit, library versions and host
- `logs/metrics.csv`: training and validation metrics, epoch time, peak GPU memory and free system memory
- `checkpoints/`: best and last checkpoints
