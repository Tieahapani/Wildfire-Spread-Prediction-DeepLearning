# Data and model weights

All files are under `$WF_DATA` (`/srv/sfsu_srv/wf-spread/data`), which is not in git. `raw/` and `models/` are read-only for the team; derived files go in `processed/` and are written by scripts in `scripts/data/`.

## NDWS: Next Day Wildfire Spread (main dataset)

- **Source:** Huot et al., "Next Day Wildfire Spread", IEEE TGRS 2022. Kaggle `fantineh/next-day-wildfire-spread`. License CC BY 4.0.
- **Location:** `raw/ndws/`: 19 TFRecord files, 3.96 GB, with checksums in `raw/ndws/SHA256SUMS`. Downloaded 2026-10-29.
- **Samples:**

  | Split | Samples |
  |---|---|
  | `train` | 14,979 |
  | `eval` (our `val`) | 1,877 |
  | `test` | 1,689 |

  Checked by `tests/integration/test_environment.py`.
- **Tile format:** 64 × 64 at 1 km, with 12 input channels plus a `FireMask` target. Label values are 1 (fire), 0 (no fire) and −1 (uncertain or cloud).

## Galileo encoder weights

- **Source:** Tseng et al., Galileo (ICML 2025). Hugging Face `nasaharvest/galileo`. License MIT.
- **Location:** `models/galileo/{nano,tiny,base}/`, holding `config.json` and `encoder.pt`. SHA-256 matched the Hugging Face LFS hashes. Pretraining-only decoders were not downloaded.
- **Loading:** these are pickles, so load only with `torch.load(..., weights_only=True)`.
- **Code:** the matching model code is vendored at `third_party/galileo/` (commit `0f0b5b9`).
