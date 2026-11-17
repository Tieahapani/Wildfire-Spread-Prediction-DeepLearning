"""Spatial augmentation for NDWS tiles, including wind-direction correction."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def transform_tile(
    features: Mapping[str, np.ndarray],
    *,
    horizontal_flip: bool = False,
    vertical_flip: bool = False,
    rotations_ccw: int = 0,
) -> dict[str, np.ndarray]:
    """Apply spatial transformations and correct wind direction."""
    if "th" not in features:
        raise ValueError("features must contain wind direction 'th'")

    if rotations_ccw not in (0, 1, 2, 3):
        raise ValueError("rotations_ccw must be between 0 and 3")

    arrays = {name: np.asarray(array) for name, array in features.items()}

    shape = arrays["th"].shape
    if len(shape) != 2:
        raise ValueError("features must be single tiles with shape (H, W)")

    for name, array in arrays.items():
        if array.shape != shape:
            raise ValueError(f"{name} has shape {array.shape}, expected {shape}")

    result = {}

    for name, array in arrays.items():
        transformed = array

        if horizontal_flip:
            transformed = np.flip(transformed, axis=-1)

        if vertical_flip:
            transformed = np.flip(transformed, axis=-2)

        transformed = np.rot90(transformed, k=rotations_ccw, axes=(-2, -1))
        result[name] = np.ascontiguousarray(transformed).copy()

    wind = result["th"].astype(np.float32, copy=True)

    if horizontal_flip:
        wind = 360.0 - wind

    if vertical_flip:
        wind = 180.0 - wind

    wind = (wind - 90.0 * rotations_ccw) % 360.0
    wind[~np.isfinite(result["th"])] = np.nan

    result["th"] = wind

    return result
