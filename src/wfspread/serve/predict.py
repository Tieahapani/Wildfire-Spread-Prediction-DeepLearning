"""Turn raw NDWS tiles into fire probability maps with a trained checkpoint.

Raw tiles go through the same cleaning as the training data, using the saved
training statistics, so a prediction never depends on the split it came from.
Every NDWS tile contains fire on day t, so this is a demonstration of the models,
not an operational fire detector.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch import nn

from wfspread.data.channels import to_inputs
from wfspread.data.cleaning import ENVIRONMENTAL_FEATURES, clean_features
from wfspread.models.registry import load_model

# Raw NDWS features needed to predict; the day-t+1 `FireMask` is not.
INPUT_FEATURES: tuple[str, ...] = (*ENVIRONMENTAL_FEATURES, "PrevFireMask")

_LIGHTNING_PREFIX = "model."


def load_weights(model: nn.Module, checkpoint: Path) -> nn.Module:
    """Load trained weights into `model`, which must have the training architecture.

    Accepts a Lightning checkpoint from `wfspread.train.run.fit` or a plain state
    dict. `weights_only=True` refuses files that would run code when unpickled.

    Args:
        model: Untrained model built from the run's model config.
        checkpoint: `.ckpt` or `.pt` file.

    Returns:
        The same model, in eval mode.

    Raises:
        RuntimeError: If the weights do not match the model's layers.
    """
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = saved.get("state_dict", saved)
    if all(key.startswith(_LIGHTNING_PREFIX) for key in state):
        state = {key.removeprefix(_LIGHTNING_PREFIX): value for key, value in state.items()}
    model.load_state_dict(state)
    return model.eval()


class FirePredictor:
    """Predicts day-t+1 fire probability for raw NDWS tiles.

    Args:
        model: Trained model mapping inputs (B, C, H, W) to logits (B, H, W).
        statistics: Training statistics written by `scripts/data/clean.py`.
        device: Where the model runs.
    """

    def __init__(
        self,
        model: nn.Module,
        statistics: Mapping[str, Mapping[str, float]],
        device: str | torch.device = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.statistics = statistics

    @classmethod
    def from_files(
        cls,
        checkpoint: Path,
        model_config: Path,
        statistics: Path,
        device: str | torch.device = "cpu",
    ) -> FirePredictor:
        """Build a predictor from a checkpoint, its model config and the statistics file."""
        model = load_weights(load_model(model_config), checkpoint)
        return cls(model, json.loads(Path(statistics).read_text()), device)

    def prepare(self, raw: Mapping[str, np.ndarray]) -> torch.Tensor:
        """Clean raw features and stack them into model inputs.

        Args:
            raw: Raw NDWS features in source units, each (N, H, W) or (H, W).

        Returns:
            float32 inputs (N, C, H, W) on the predictor's device.

        Raises:
            ValueError: If a needed feature is missing.
        """
        missing = [name for name in INPUT_FEATURES if name not in raw]
        if missing:
            raise ValueError(f"raw tile is missing {missing}")
        features = clean_features({name: raw[name] for name in INPUT_FEATURES}, self.statistics)
        inputs = torch.from_numpy(to_inputs(features))
        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(0)
        return inputs.to(self.device)

    @torch.no_grad()
    def predict(self, raw: Mapping[str, np.ndarray]) -> np.ndarray:
        """Fire probability per pixel, float32 (N, H, W) in [0, 1]."""
        logits = self.model(self.prepare(raw)).float()
        return torch.sigmoid(logits).cpu().numpy()
