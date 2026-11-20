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
| Loss | Weighted sum of masked BCE (optional fire weight `pos_weight`), focal and Dice, set by `loss_terms`; all skip unknown pixels. Default: BCE with fire pixels weighted 3, the winner of the loss comparison |
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

Two U-Net runs with the same config and seed can differ by about 0.01 validation PR-AUC (GPU kernels are not bitwise deterministic), which is as large as the gaps between arms. So every arm runs with three seeds and is compared on the mean:

```bash
bin/wf python scripts/train/compare_arms.py \
    --arms loss-bce loss-wbce loss-focal loss-combo --out "$WF_RUNS/loss-comparison"
```

Seeds 0, 1 and 2 are the default. Each finished run is added to `runs.csv` (best epoch per run), and `summary.md` holds the mean ± standard deviation per arm. Starting the script again skips runs already in `runs.csv`.

**Choosing the winner:**
1. Rank the arms by mean `val/all/pr_auc`.
2. If the top arm's lead is smaller than the larger standard deviation of the two, count it as a tie and break the tie on mean `val/spread/f1`.
3. If that is also within one standard deviation, keep the simpler loss, with plain BCE as the simplest.

**Result (Nov 18, seeds 0, 1, 2, best validation epoch, mean ± standard deviation):**

| Arm | `val/all/pr_auc` | `val/all/f1` | `val/spread/pr_auc` | `val/spread/f1` |
|---|---|---|---|---|
| `loss-wbce` | 0.297 ± 0.009 | 0.367 ± 0.008 | 0.250 ± 0.011 | 0.319 ± 0.009 |
| `loss-bce` | 0.297 ± 0.004 | 0.359 ± 0.004 | 0.245 ± 0.006 | 0.308 ± 0.005 |
| `loss-combo` | 0.296 ± 0.005 | 0.360 ± 0.004 | 0.243 ± 0.004 | 0.306 ± 0.003 |
| `loss-focal` | 0.294 ± 0.002 | 0.359 ± 0.003 | 0.242 ± 0.001 | 0.309 ± 0.003 |

All four tie on PR-AUC. Weighted BCE wins the tie-break on spread F1 by 0.010, just above its standard deviation of 0.009, so the margin is small. Since Nov 19 it is the default in `configs/train/default.yaml`; runs before that date used plain BCE unless their `run.yaml` says otherwise.

### Crop experiment (experiment 3)

Does training on random 32 × 32 crops of each tile, instead of the full 64 × 64 tile, help the U-Net? Crops show the model more varied fire positions and edges per epoch, but less context around each pixel. Validation always scores full tiles.

| Config | Training input |
|---|---|
| `loss-wbce.yaml` | Full tiles; its three runs from the loss comparison are reused |
| `crop-32.yaml` | Random 32 × 32 crop per tile, re-drawn every epoch (`crop_size: 32`) |

Both use the default loss (weighted BCE) and differ only in `crop_size`. An epoch of crops covers a quarter of the training pixels, so crop runs take more, shorter epochs.

```bash
bin/wf python scripts/train/compare_arms.py --arms crop-32 --out "$WF_RUNS/crop-comparison"
```

Crops win only if their mean `val/all/pr_auc` beats full tiles by more than the larger standard deviation of the two. Otherwise full tiles stay, as the simpler setup.

## Sanity checks

Run these on a few real batches before any search; they live in `wfspread.train.sanity`.

| Check | Passes when | A failure means |
|---|---|---|
| `overfit_check` | PR-AUC on about 10 training tiles reaches 0.9 after training on them | A bug in the model, loss or data |
| `shuffled_label_check` | With labels shuffled across pixels, validation PR-AUC stays at or below 2 × the fire rate | The score doesn't come from learning, for example scoring against the wrong mask |
| `mask_check` | Changing predictions on unknown pixels changes neither the loss nor PR-AUC | Unknown pixels leak into the loss or metrics |

Labels are shuffled pixel by pixel, not tile by tile, because fires sit near the tile center and a model could learn that position honestly.
