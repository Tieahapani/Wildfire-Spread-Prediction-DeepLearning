"""Stack cleaned NDWS features into the model's input channels.

This is the last step before a batch: after augmentation, which turns wind
direction in degrees, and before the model. See `docs/batch-format.md`.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from wfspread.constants import CH, FIRE, NDWS_CHANNELS

# Channels copied straight from a cleaned NDWS feature.
SOURCE_FEATURE: dict[str, str] = {
    "elevation_m": "elevation",
    "wind_speed_m_s": "vs",
    "temp_min_k": "tmmn",
    "temp_max_k": "tmmx",
    "humidity_kg_kg": "sph",
    "precip_mm": "pr",
    "drought_index": "pdsi",
    "ndvi": "NDVI",
    "population_per_km2": "population",
    "energy_release": "erc",
}


def to_inputs(features: Mapping[str, np.ndarray]) -> np.ndarray:
    """Build model inputs from cleaned features.

    Args:
        features: Cleaned NDWS features, each (N, H, W) or (H, W): z-scored
            environmental features, wind direction `th` in degrees with NaN where
            missing, and the raw `PrevFireMask` with values in {-1, 0, 1}.

    Returns:
        float32 array (N, C, H, W) in `NDWS_CHANNELS` order, or (C, H, W) for a
        single tile. Missing wind direction is (0, 0) in sin and cos, and only
        known fire counts as fire in the `prev_fire` channel.
    """
    shape = np.shape(features["PrevFireMask"])
    inputs = np.empty((len(NDWS_CHANNELS), *shape), dtype=np.float32)
    for channel, name in SOURCE_FEATURE.items():
        inputs[CH[channel]] = features[name]

    radians = np.deg2rad(np.asarray(features["th"], dtype=np.float32))
    inputs[CH["wind_dir_sin"]] = np.nan_to_num(np.sin(radians), nan=0.0)
    inputs[CH["wind_dir_cos"]] = np.nan_to_num(np.cos(radians), nan=0.0)
    inputs[CH["prev_fire"]] = np.asarray(features["PrevFireMask"]) == FIRE

    # Channels first per tile: (C, N, H, W) -> (N, C, H, W).
    return np.ascontiguousarray(np.moveaxis(inputs, 0, 1)) if len(shape) == 3 else inputs
