# Models

Every model maps `inputs` `(B, C, H, W)` to fire logits `(B, H, W)` (see [`batch-format.md`](batch-format.md)) and trains through the shared loop in [`training.md`](training.md). A run chooses its model from a file in `configs/model/`:

```python
from wfspread.models.registry import load_model

model = load_model("configs/model/unet.yaml")
```

| Name | Code | Config | Parameters |
|---|---|---|---|
| `unet` | `src/wfspread/models/unet.py` | `configs/model/unet.yaml` | 1.93M |

## Model configs

```yaml
name: unet          # key in MODELS
args:               # constructor arguments; unknown ones are rejected
  enc_channels: [32, 64, 128]
```

## Adding a model

1. Write the model in `src/wfspread/models/<name>.py`. It must:
   - return raw logits `(B, H, W)`, not probabilities;
   - be fully convolutional, so it accepts full tiles and crops;
   - never read `prev_fire` or `next_fire` from the batch, only `inputs`.
2. Add it to `MODELS` in `src/wfspread/models/registry.py`.
3. Add `configs/model/<name>.yaml` with a short header comment.
4. Add tests in `tests/unit/models/test_<name>.py`, including the sanity checks in `wfspread.train.sanity` on fake tiles. The registry tests check that every registered model has a config and returns logits of the right shape.
