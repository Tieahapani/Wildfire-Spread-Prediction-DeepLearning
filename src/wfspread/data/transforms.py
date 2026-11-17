"""Spatial augmentation for NDWS tiles, including wind-direction correction.

Wind direction turns with the image so that the physical meaning of the
channel is preserved after flips and rotations.  The formulas are from
``docs/batch-format.md``.
"""

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
    """Apply spatial transforms to every feature and correct wind direction.

    Args:
        features: NDWS feature name to single-tile array, each (H, W).
        horizontal_flip: Mirror left to right.
        vertical_flip: Mirror top to bottom.
        rotations_ccw: Counter-clockwise 90° turns (0–3).

    Returns:
        Transformed copies of all features.

    Raises:
        ValueError: If ``th`` is missing, shapes disagree, or ``rotations_ccw``
            is out of range.
    """
    if "th" not in features:
        raise ValueError("features must contain wind direction 'th'")
    if rotations_ccw not in (0, 1, 2, 3):
        raise ValueError("rotations_ccw must be 0, 1, 2 or 3")

    arrays = {name: np.asarray(array) for name, array in features.items()}
    shape = arrays["th"].shape
    if len(shape) != 2:
        raise ValueError("features must be single tiles with shape (H, W)")
    for name, array in arrays.items():
        if array.shape != shape:
            raise ValueError(f"{name} has shape {array.shape}, expected {shape}")

    result: dict[str, np.ndarray] = {}
    for name, array in arrays.items():
        a = array
        if horizontal_flip:
            a = np.flip(a, axis=-1)
        if vertical_flip:
            a = np.flip(a, axis=-2)
        if rotations_ccw:
            a = np.rot90(a, k=rotations_ccw, axes=(-2, -1))
        result[name] = np.ascontiguousarray(a)

    # Wind direction follows the same spatial transforms.
    wind = result["th"].astype(np.float32, copy=True)
    valid = np.isfinite(wind)
    if horizontal_flip:
        wind[valid] = 360.0 - wind[valid]
    if vertical_flip:
        wind[valid] = (180.0 - wind[valid]) % 360.0
    if rotations_ccw:
        wind[valid] = (wind[valid] - 90.0 * rotations_ccw) % 360.0
    result["th"] = wind

    return result
