# Batch format

Every loader returns batches in this format, and every model, loss, metric and the serving layer read it. The code is in `src/wfspread/data/batch.py` (`Batch`, `check_batch`) and the channel order in `src/wfspread/constants.py` (`NDWS_CHANNELS`, `CH`).

## Keys

| Key | Shape | Type | Values |
|---|---|---|---|
| `inputs` | `(B, C, H, W)` | float32 | Cleaned, normalized channels in `NDWS_CHANNELS` order; C = 13 |
| `prev_fire` | `(B, H, W)` | int8 | Day-t fire mask: 1 fire, 0 no fire, −1 unknown. For metrics and analysis only |
| `next_fire` | `(B, H, W)` | int8 | Day-t+1 fire mask: 1 fire, 0 no fire, −1 unknown |
| `sample_id` | `(B,)` | int64 | Tile position in its split's original TFRecord order |

- H and W are 64. Training batches in the crop experiment are 32 × 32; validation and test always use the full tile.
- At inference only `inputs` and `sample_id` are required.
- Runs that drop a channel, such as the run without today's fire mask, remove it by name, so C is 12.

## Channels

| Index | Name | NDWS source | Preparation | Z-scored |
|---|---|---|---|---|
| 0 | `elevation_m` | `elevation` | Clip | Yes |
| 1, 2 | `wind_dir_sin`, `wind_dir_cos` | `th` | Fix impossible values, augment in degrees, then sin and cos. Missing direction is 0, 0 | No |
| 3 | `wind_speed_m_s` | `vs` | Negative is missing; clip; fill | Yes |
| 4, 5 | `temp_min_k`, `temp_max_k` | `tmmn`, `tmmx` | No-data values (≤ 200 K, or implausibly high) are missing; clip; fill | Yes |
| 6 | `humidity_kg_kg` | `sph` | Negative is missing; clip; fill | Yes |
| 7 | `precip_mm` | `pr` | Small negatives become 0, large ones are missing; log1p; clip; fill | Yes |
| 8 | `drought_index` | `pdsi` | Clip | Yes |
| 9 | `ndvi` | `NDVI` | Divide by 10,000; keep real negatives; clip | Yes |
| 10 | `population_per_km2` | `population` | log1p; clip | Yes |
| 11 | `energy_release` | `erc` | Negative is missing; clip; fill | Yes |
| 12 | `prev_fire` | `PrevFireMask` | 1 where `prev_fire` is fire, otherwise 0 | No |

Missing values are marked before clipping, because NDWS stores no-data as 0 K temperatures and wind directions far outside 0–360°, and raw percentiles would keep them. Clip limits are the 0.1 and 99.9 percentiles of the training split without missing values; missing values are filled with the training median. All statistics come from the training split only and are saved to one versioned file that the loaders and the serving layer read.

## From stored tile to batch

1. Read the cleaned tile: 11 channels with wind direction in degrees, plus both masks.
2. Training only: a random flip and 90° rotation, applied to the channels and both masks. Wind direction turns with the image: left-right flip `360 − θ`; up-down flip `(180 − θ) mod 360`; counterclockwise rotation by k × 90° `(θ − 90k) mod 360`.
3. Crop experiment only: the same random 32 × 32 crop on the channels and both masks.
4. Wind direction to sin and cos, then z-scores, then the `prev_fire` channel.

## Rules

- **Models** are called as `model(batch["inputs"])` and return logits of shape `(B, H, W)`. They never receive `prev_fire` or `next_fire`, and they are fully convolutional, so one model trains on 32 × 32 and evaluates on 64 × 64.
- **Losses** skip pixels where `next_fire` is −1 and return 0 when a batch has no valid pixel. A mean over zero pixels is NaN, and some random crops contain no valid pixel.
- **Metrics** are computed once over the whole split, from all predictions collected in `sample_id` order. Per-batch PR-AUC is not averaged: it is undefined for a batch with no fire.
- **Precision:** with bf16 training, the last layer, the loss and evaluation run in float32. bf16 logits leave too few distinct probabilities and create ties in PR-AUC.
- **Evaluation** uses no augmentation, the full tile and a fixed order (`shuffle=False`).
- **Comparisons** score every model on the same pixels. The 1-pixel tile border is excluded only when checking logistic regression against Huot et al. 2022, which excluded it.
- **The test split** is loaded only for the final evaluation.

## Storage and loading

- `data/processed/ndws-v1/<split>_inputs.npy`: `(N, 11, 64, 64)` float32, wind direction in degrees, NaN where missing.
- `data/processed/ndws-v1/<split>_prev_fire.npy`, `<split>_next_fire.npy`: `(N, 64, 64)` int8.
- `data/processed/ndws-v1/<split>_sample_id.npy`: `(N,)` int64.
- `data/processed/ndws-v1/stats.json`: clip limits, medians, means and standard deviations, with a version.
- Files are memory-mapped. Training uses `shuffle=True` and `drop_last=True`, so a short last batch doesn't skew BatchNorm. Workers are seeded from the run config, and use `pin_memory=True` with `non_blocking=True` copies.
