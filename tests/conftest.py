"""Shared pytest fixtures: dataset locations, fake tiles in the batch format, fake raw NDWS."""

import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor

from wfspread.constants import CH, IGNORE_INDEX
from wfspread.data.cleaning import CIRCULAR_FEATURES, ENVIRONMENTAL_FEATURES

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def data_root() -> Path:
    # bin/wf points WF_DATA at the shared copy, so personal clones see the same data.
    return Path(os.environ.get("WF_DATA", REPO_ROOT / "data"))


def _fake_tiles(n_tiles: int, seed: int, size: int = 16) -> dict[str, Tensor]:
    g = torch.Generator().manual_seed(seed)
    inputs = torch.randn(n_tiles, 13, size, size, generator=g)
    prev = torch.zeros(n_tiles, size, size, dtype=torch.int8)
    for i in range(n_tiles):
        r, c = torch.randint(2, size - 6, (2,), generator=g).tolist()
        prev[i, r : r + 3, c : c + 3] = 1
    next_fire = prev.clone()
    next_fire[:, :, 1:] |= prev[:, :, :-1]
    next_fire[:, 0, :] = IGNORE_INDEX
    inputs[:, CH["prev_fire"]] = prev.float()
    return {
        "inputs": inputs,
        "prev_fire": prev,
        "next_fire": next_fire,
        "sample_id": torch.arange(n_tiles, dtype=torch.int64),
    }


@pytest.fixture
def fake_tiles() -> Callable[..., dict[str, Tensor]]:
    """Builds fake tiles where tomorrow's fire is today's fire grown one pixel right.

    Call it as `fake_tiles(n_tiles, seed, size=16)`. The top row of `next_fire` is
    unknown (-1), so mask handling is exercised too.
    """
    return _fake_tiles


def _raw_features(n_tiles: int, seed: int, size: int = 16) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = {
        name: rng.uniform(1.0, 10.0, (n_tiles, size, size)).astype(np.float32)
        for name in ENVIRONMENTAL_FEATURES
    }
    features["tmmn"] += 270
    features["tmmx"] += 290
    features["th"] = rng.uniform(0, 360, (n_tiles, size, size)).astype(np.float32)
    fire = rng.choice([-1, 0, 1], (n_tiles, size, size), p=[0.05, 0.85, 0.1])
    features["PrevFireMask"] = fire.astype(np.float32)
    features["FireMask"] = fire.astype(np.float32)
    return features


def _statistics(features: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "lower": float(np.nanmin(features[name])),
            "upper": float(np.nanmax(features[name])),
            "median": float(np.nanmedian(features[name])),
            "mean": float(np.nanmean(features[name])),
            "std": float(np.nanstd(features[name])),
        }
        for name in ENVIRONMENTAL_FEATURES
        if name not in CIRCULAR_FEATURES
    }


@pytest.fixture
def raw_features() -> Callable[..., dict[str, np.ndarray]]:
    """Builds fake raw NDWS features in their source units, keyed by NDWS name.

    Call it as `raw_features(n_tiles, seed, size=16)`. Fire masks hold -1, 0 and 1.
    """
    return _raw_features


@pytest.fixture
def statistics_for() -> Callable[..., dict[str, dict[str, float]]]:
    """Builds cleaning statistics (clip limits, median, mean, std) from given features."""
    return _statistics
