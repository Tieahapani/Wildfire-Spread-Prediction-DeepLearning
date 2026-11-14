import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

_spec = importlib.util.spec_from_file_location("clean", REPO_ROOT / "scripts/data/clean.py")
clean = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(clean)


def _features(n_tiles=2, size=4, seed=0):
    rng = np.random.default_rng(seed)
    features = {
        name: rng.uniform(1.0, 10.0, (n_tiles, size, size)).astype(np.float32)
        for name in clean.ENVIRONMENTAL_FEATURES
    }
    features["tmmn"] += 270
    features["tmmx"] += 290
    features["th"] = rng.uniform(0, 360, (n_tiles, size, size)).astype(np.float32)
    features["FireMask"] = np.zeros((n_tiles, size, size), np.float32)
    features["PrevFireMask"] = np.zeros((n_tiles, size, size), np.float32)
    return features


def _statistics(features):
    thresholds = {
        name: {"lower": float(np.nanmin(a)), "upper": float(np.nanmax(a))}
        for name, a in features.items()
    }
    return clean.calculate_preprocessing_statistics(features, thresholds)


def test_missing_rules_zero_small_negative_rain_and_drop_bad_values():
    features = _features()
    features["pr"][0, 0, :3] = [-0.05, clean.PR_NEGATIVE_LIMIT_MM - 1, 2.0]
    features["erc"][0, 0, 0] = -3.0
    features["th"][0, 0, 0] = 500.0

    cleaned = clean.apply_missing_rules(features)

    assert cleaned["pr"][0, 0, 0] == 0.0
    assert np.isnan(cleaned["pr"][0, 0, 1])
    assert cleaned["pr"][0, 0, 2] == 2.0
    assert np.isnan(cleaned["erc"][0, 0, 0])
    assert np.isnan(cleaned["th"][0, 0, 0])


def test_log_transform_applies_to_skewed_features_only():
    features = _features()
    features["pr"][0, 0, :2] = [np.e - 1, np.nan]
    elevation = features["elevation"].copy()

    clean.apply_log_transforms(features)

    assert features["pr"][0, 0, 0] == pytest.approx(1.0)
    assert np.isnan(features["pr"][0, 0, 1])
    np.testing.assert_array_equal(features["elevation"], elevation)


def test_wind_direction_stays_in_degrees_and_others_are_standardized():
    features = _features()
    features["th"][0, 0, 0] = np.nan
    features["vs"][0, 0, 0] = np.nan
    degrees = features["th"].copy()

    statistics = _statistics(features)
    processed = clean.apply_preprocessing(features, statistics)

    assert "th" not in statistics
    assert statistics["pr"]["transform"] == "log1p"
    assert statistics["vs"]["transform"] == "none"
    np.testing.assert_array_equal(processed["th"], degrees)
    assert np.all(np.isfinite(processed["vs"]))
    clean.check_processed(processed)


def test_eval_split_uses_training_statistics_and_keeps_every_tile(tmp_path):
    train_stats = _statistics(_features(seed=1))
    val = _features(n_tiles=3, size=64, seed=2)
    val["FireMask"][:] = -1
    with h5py.File(tmp_path / "val.h5", "w") as f:
        group = f.create_group("features")
        for name in clean.FEATURE_NAMES:
            group.create_dataset(name, data=val[name])
        f.create_dataset("tile_ids", data=np.array([b"a:0", b"a:1", b"a:2"]))
        f.create_dataset("record_indices", data=np.arange(3))
        f.attrs["num_records"] = 3
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    clean.clean_eval_split(tmp_path, out_dir, "val", train_stats)

    with h5py.File(out_dir / "val.h5", "r") as f:
        assert f.attrs["split"] == "val"
        assert f.attrs["num_records"] == 3
        vs = f["features/vs"][:]
    stats = train_stats["vs"]
    expected = (np.clip(val["vs"], stats["lower"], stats["upper"]) - stats["mean"]) / stats["std"]
    np.testing.assert_allclose(vs, expected, rtol=1e-5)
