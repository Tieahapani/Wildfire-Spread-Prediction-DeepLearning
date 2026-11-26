# Serving

The serving layer turns a trained checkpoint and raw NDWS tiles into fire probability maps, for the presentation maps and for inference timing. Every NDWS tile already contains fire on day t, so this demonstrates the models; it is not an operational fire detector.

The final model is the U-Net in `configs/model/unet.yaml`; its seed-0 run's best checkpoint is:

```bash
bin/wf python scripts/serve/predict_tile.py \
    --checkpoint "$WF_RUNS"/loss-comparison/20261118-2032_unet_loss-wbce-s0/checkpoints/best-012.ckpt \
    --model configs/model/unet.yaml --split val --index 0 7 42 --out-dir maps
```

Use the best checkpoint, not `last.ckpt`: the final evaluation scores the best validation epoch.

For each tile it saves `<split>-<index>.npy` (probabilities, 64 × 64 float32) and `<split>-<index>.png`: today's fire, the predicted probability for tomorrow, and tomorrow's actual fire, with unknown pixels in grey. The test split is not offered until the one-time final evaluation.

## How a tile becomes a prediction

1. **Raw features** in source units are read from the converted split (`data/processed/ndws/<split>.h5`).
2. **Cleaning** (`wfspread.data.cleaning.clean_features`) applies the same steps as `scripts/data/clean.py`, with the saved training statistics (`ndws-v1/training_statistics.json`): missing-value rules, log1p on rain and population, clipping, median fill and z-scores. On 200 validation tiles the result is identical to the cleaned `ndws-v1/val.h5`.
3. **Channels** (`wfspread.data.channels.to_inputs`) stacks the 13 inputs in `NDWS_CHANNELS` order: wind direction becomes sine and cosine (0, 0 where missing), and only known fire counts in the `prev_fire` channel. The loader can use the same function after augmentation.
4. **Model:** the checkpoint is loaded with `weights_only=True` into the architecture from the run's model config, and the sigmoid of the logits is the probability.

## In Python

```python
from wfspread.serve.predict import FirePredictor

predictor = FirePredictor.from_files(checkpoint, "configs/model/unet.yaml", statistics)
probs = predictor.predict(raw)  # raw: NDWS feature name -> (N, 64, 64) or (64, 64)
```

`raw` needs the 11 environmental features and `PrevFireMask`; `FireMask` (tomorrow) is not used. The checkpoint can be a Lightning `.ckpt` from `wfspread.train.run.fit` or a plain state dict.
