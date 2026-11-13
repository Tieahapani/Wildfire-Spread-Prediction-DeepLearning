# Training

All models train through one shared loop in `src/wfspread/train/`, so every model gets the same loss, metrics, schedule, stopping rule and logging. Only the model changes.

## Running

```python
from wfspread.train.config import TrainConfig
from wfspread.train.run import fit

config = TrainConfig.from_yaml("configs/train/default.yaml")
run_dir = fit(model, train_loader, val_loader, config, run_name="unet_baseline")
```

- `model` maps `inputs` `(B, C, H, W)` to fire logits `(B, H, W)`; see [`batch-format.md`](batch-format.md). Build it from a config with `load_model`; see [`models.md`](models.md).
- Loaders return batches in the batch format. The training loader shuffles and uses `drop_last=True`; the validation loader does not shuffle.
- Runs that drop a channel pass `num_channels`.

## What the loop does

| Part | Choice |
|---|---|
| Loss | Weighted sum of masked BCE (optional fire weight `pos_weight`), focal and Dice, set by `loss_terms`; all skip unknown pixels. Default: plain BCE |
| Optimizer | AdamW; linear warmup, then cosine decay to `min_lr`, updated every step |
| Gradients | Norm clipped at `grad_clip` |
| Validation | Predictions for the whole split are scored once per epoch with `wfspread.eval.metrics.evaluate`. F1 and IoU use the best-F1 threshold on validation |
| Stopping | Early stopping on `val/all/pr_auc` after `patience` epochs without improvement |
| Checkpoints | Best epoch by `val/all/pr_auc`, plus the last epoch |
| Reproducibility | One seed for torch, numpy, Python and loader workers |
| Precision | float32 by default; logits and loss are always float32 |

Settings and defaults are in `configs/train/default.yaml`; unknown keys are rejected. A config can start from another with `base: <relative path>` and list only what it changes.

## Run folder

Each run writes to `$WF_RUNS/<YYYYMMDD-HHMM>_<model>_<tag>/`:

- `run.yaml`: settings, batch size, git commit, library versions and host
- `logs/metrics.csv`: training and validation metrics, diagnostics, epoch time, peak GPU memory and free system memory
- `checkpoints/`: best and last checkpoints

## Diagnostics

Every `diagnostics_every_n_steps` optimizer steps (default 50; 0 turns them off), `TrainingDiagnostics` adds these columns to `logs/metrics.csv`:

| Column | Meaning | What to look for |
|---|---|---|
| `diag/lr` | Learning rate used for that step | Follows the warmup and cosine schedule |
| `diag/grad_norm` | Gradient norm before clipping | Often above `grad_clip` means clipping is doing much of the work; spikes point to bad batches |
| `diag/update/<layer>` | Relative weight change ‖Δw‖ / ‖w‖ in one step, per weight tensor | About 1e-3 is typical for AdamW; far below means the layer has stopped learning, far above means steps are too large |

## Losses

| Term | Function | Settings |
|---|---|---|
| `bce` | `masked_bce` | `pos_weight`: weight on fire pixels |
| `focal` | `masked_focal`: BCE scaled by `alpha_t * (1 - p_t) ** gamma` (Lin et al. 2017) | `focal_gamma` (2.0), `focal_alpha` (0.25, weight on fire pixels) |
| `dice` | `masked_dice`: soft Dice over the valid pixels of the whole batch, since many tiles have no next-day fire | none |

`loss_terms` gives each term a weight, for example `{bce: 0.4, dice: 0.3, focal: 0.3}`. Because the terms are on different scales, `val/loss` is not comparable across losses; compare runs on `val/all/pr_auc`.

### Loss comparison (experiment 2)

Four arms on the U-Net (`configs/model/unet.yaml`), each in `configs/experiment/`. They start from `configs/train/default.yaml` and change only the loss:

| Config | Loss |
|---|---|
| `loss-bce.yaml` | BCE |
| `loss-wbce.yaml` | BCE, fire pixels weighted 3 |
| `loss-focal.yaml` | Focal, gamma 2, alpha 0.25 |
| `loss-combo.yaml` | 0.4 weighted BCE (fire 3) + 0.3 Dice + 0.3 focal |

The winner on validation PR-AUC becomes the loss for both models' tuning searches.

## Sanity checks

Run these on a few real batches before any search; they live in `wfspread.train.sanity`.

| Check | Passes when | A failure means |
|---|---|---|
| `overfit_check` | PR-AUC on about 10 training tiles reaches 0.9 after training on them | A bug in the model, loss or data |
| `shuffled_label_check` | With labels shuffled across pixels, validation PR-AUC stays at or below 2 × the fire rate | The score doesn't come from learning, for example scoring against the wrong mask |
| `mask_check` | Changing predictions on unknown pixels changes neither the loss nor PR-AUC | Unknown pixels leak into the loss or metrics |

Labels are shuffled pixel by pixel, not tile by tile, because fires sit near the tile center and a model could learn that position honestly.
