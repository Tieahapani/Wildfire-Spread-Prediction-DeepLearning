"""Tests for spatial augmentation with wind-direction correction."""

import numpy as np
import pytest

from wfspread.data.transforms import transform_tile


def _tile(th_value: float = 90.0, size: int = 4) -> dict[str, np.ndarray]:
    """Minimal tile with uniform wind direction and a ramp in elevation."""
    th = np.full((size, size), th_value, dtype=np.float32)
    elev = np.arange(size * size, dtype=np.float32).reshape(size, size)
    fire = np.zeros((size, size), dtype=np.float32)
    return {"th": th, "elevation": elev, "FireMask": fire, "PrevFireMask": fire}


class TestWindFormulas:
    """Wind direction must follow the documented formulas."""

    def test_no_transform(self):
        out = transform_tile(_tile(90.0))
        assert np.allclose(out["th"], 90.0)

    def test_horizontal_flip(self):
        out = transform_tile(_tile(90.0), horizontal_flip=True)
        np.testing.assert_allclose(out["th"], 270.0)

    def test_vertical_flip(self):
        out = transform_tile(_tile(30.0), vertical_flip=True)
        np.testing.assert_allclose(out["th"], 150.0)

    def test_vertical_flip_wraparound(self):
        out = transform_tile(_tile(350.0), vertical_flip=True)
        np.testing.assert_allclose(out["th"], (180.0 - 350.0) % 360.0)

    def test_rotation_90(self):
        out = transform_tile(_tile(0.0), rotations_ccw=1)
        np.testing.assert_allclose(out["th"], 270.0)

    def test_rotation_180(self):
        out = transform_tile(_tile(45.0), rotations_ccw=2)
        np.testing.assert_allclose(out["th"], (45.0 - 180.0) % 360.0)

    def test_combined_flip_and_rotation(self):
        out = transform_tile(_tile(90.0), horizontal_flip=True, rotations_ccw=1)
        expected = (270.0 - 90.0) % 360.0
        np.testing.assert_allclose(out["th"], expected)


class TestNanPreservation:
    """Missing wind direction must stay NaN through any transform."""

    def test_nan_stays_nan_flip(self):
        tile = _tile()
        tile["th"][0, 0] = np.nan
        out = transform_tile(tile, horizontal_flip=True, vertical_flip=True)
        assert np.sum(np.isnan(out["th"])) == 1

    def test_nan_stays_nan_rotation(self):
        tile = _tile()
        tile["th"][1, 2] = np.nan
        out = transform_tile(tile, rotations_ccw=3)
        assert np.sum(np.isnan(out["th"])) == 1


class TestSpatialConsistency:
    """All features must be transformed the same way spatially."""

    def test_flip_lr_pixels(self):
        tile = _tile()
        tile["elevation"] = np.array([[1, 2], [3, 4]], dtype=np.float32)
        tile["th"] = np.full((2, 2), 90.0, dtype=np.float32)
        tile["FireMask"] = np.zeros((2, 2), dtype=np.float32)
        tile["PrevFireMask"] = np.zeros((2, 2), dtype=np.float32)
        out = transform_tile(tile, horizontal_flip=True)
        np.testing.assert_array_equal(out["elevation"], [[2, 1], [4, 3]])


class TestValidation:
    """Bad inputs must raise."""

    def test_missing_th(self):
        with pytest.raises(ValueError, match="th"):
            transform_tile({"elevation": np.zeros((4, 4))})

    def test_bad_rotation(self):
        with pytest.raises(ValueError, match="0, 1, 2 or 3"):
            transform_tile(_tile(), rotations_ccw=5)

    def test_shape_mismatch(self):
        tile = _tile()
        tile["elevation"] = np.zeros((3, 3))
        with pytest.raises(ValueError, match="shape"):
            transform_tile(tile)
