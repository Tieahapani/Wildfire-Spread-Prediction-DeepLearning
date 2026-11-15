"""Per-pixel cleaning of raw NDWS features, shared by the cleaning script and serving.

`scripts/data/clean.py` derives the training statistics with these functions, and
the serving layer applies the same steps to new tiles with those saved statistics.
The steps and their order are described in `docs/batch-format.md`.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

ENVIRONMENTAL_FEATURES: tuple[str, ...] = (
    "NDVI",
    "elevation",
    "erc",
    "pdsi",
    "population",
    "pr",
    "sph",
    "th",
    "tmmn",
    "tmmx",
    "vs",
)

# Heavily right-skewed, so log1p before clipping and standardizing.
LOG_FEATURES = ("pr", "population")

# Circular: 359 and 1 degrees are neighbours, so no clipping or z-score.
CIRCULAR_FEATURES = ("th",)

# Small negative precipitation is resampling noise and becomes 0. Below this
# (about 0.002% of training pixels) it is treated as missing.
PR_NEGATIVE_LIMIT_MM = -1.0

# NDWS fills missing weather with 0 at the same pixels in all of these.
_SHARED_FILL_FEATURES = ("tmmn", "tmmx", "vs", "sph", "erc", "th")


def apply_missing_rules(features: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Mark impossible and no-data environmental values as NaN.

    Fire masks are copied unchanged and keep -1 for unknown labels.

    Args:
        features: NDWS feature name to array, all the same shape.

    Returns:
        float32 copies with missing values set to NaN.
    """
    cleaned = {name: np.array(array, dtype=np.float32) for name, array in features.items()}

    for name in ENVIRONMENTAL_FEATURES:
        cleaned[name][~np.isfinite(cleaned[name])] = np.nan

    shared_fill = np.logical_and.reduce([cleaned[name] == 0 for name in _SHARED_FILL_FEATURES])
    for name in _SHARED_FILL_FEATURES:
        cleaned[name][shared_fill] = np.nan

    cleaned["vs"][cleaned["vs"] < 0] = np.nan
    cleaned["sph"][cleaned["sph"] < 0] = np.nan
    cleaned["tmmn"][cleaned["tmmn"] <= 200] = np.nan
    cleaned["tmmx"][cleaned["tmmx"] <= 200] = np.nan
    cleaned["th"][(cleaned["th"] < 0) | (cleaned["th"] > 360)] = np.nan
    cleaned["erc"][cleaned["erc"] < 0] = np.nan

    pr = cleaned["pr"]
    pr[pr < PR_NEGATIVE_LIMIT_MM] = np.nan
    pr[pr < 0] = 0.0

    return cleaned


def apply_log_transforms(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Apply log1p in place to the skewed features; NaN stays NaN."""
    for name in LOG_FEATURES:
        features[name] = np.log1p(features[name])
    return features


def apply_preprocessing(
    features: Mapping[str, np.ndarray], statistics: Mapping[str, Mapping[str, float]]
) -> dict[str, np.ndarray]:
    """Clip to the training limits, fill missing values with the median, then z-score.

    Only features listed in `statistics` are changed, so wind direction (not in the
    statistics) stays in degrees with NaN where missing.

    Args:
        features: Output of `apply_log_transforms`.
        statistics: Per feature `lower`, `upper`, `median`, `mean` and `std` from the
            training split.

    Returns:
        float32 copies of all features.
    """
    processed = {name: np.array(array, dtype=np.float32) for name, array in features.items()}
    for name, stats in statistics.items():
        values = processed[name]
        valid = np.isfinite(values)
        values[valid] = np.clip(values[valid], stats["lower"], stats["upper"])
        values[~valid] = stats["median"]
        processed[name] = (values - np.float32(stats["mean"])) / np.float32(stats["std"])
    return processed


def clean_features(
    features: Mapping[str, np.ndarray], statistics: Mapping[str, Mapping[str, float]]
) -> dict[str, np.ndarray]:
    """All cleaning steps for tiles outside the training split, using its statistics."""
    return apply_preprocessing(apply_log_transforms(apply_missing_rules(features)), statistics)
