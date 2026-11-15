import numpy as np
import pytest

from wfspread.data.cleaning import (
    PR_NEGATIVE_LIMIT_MM,
    apply_log_transforms,
    apply_missing_rules,
    apply_preprocessing,
    clean_features,
)


def test_missing_rules_zero_small_negative_rain_and_drop_bad_values(raw_features):
    features = raw_features(2, seed=0)
    features["pr"][0, 0, :3] = [-0.05, PR_NEGATIVE_LIMIT_MM - 1, 2.0]
    features["erc"][0, 0, 0] = -3.0
    features["th"][0, 0, 0] = 500.0
    features["tmmn"][0, 0, 0] = 150.0

    cleaned = apply_missing_rules(features)

    assert cleaned["pr"][0, 0, 0] == 0.0
    assert np.isnan(cleaned["pr"][0, 0, 1])
    assert cleaned["pr"][0, 0, 2] == 2.0
    assert np.isnan(cleaned["erc"][0, 0, 0])
    assert np.isnan(cleaned["th"][0, 0, 0])
    assert np.isnan(cleaned["tmmn"][0, 0, 0])
    assert features["erc"][0, 0, 0] == -3.0, "input must not be changed"


def test_shared_zero_fill_is_missing_in_every_weather_feature(raw_features):
    features = raw_features(1, seed=0)
    for name in ("tmmn", "tmmx", "vs", "sph", "erc", "th"):
        features[name][0, 2, 3] = 0.0

    cleaned = apply_missing_rules(features)

    for name in ("tmmn", "tmmx", "vs", "sph", "erc", "th"):
        assert np.isnan(cleaned[name][0, 2, 3]), name
    assert not np.isnan(cleaned["elevation"][0, 2, 3])


def test_log_transform_applies_to_skewed_features_only(raw_features):
    features = raw_features(2, seed=0)
    features["pr"][0, 0, :2] = [np.e - 1, np.nan]
    elevation = features["elevation"].copy()

    apply_log_transforms(features)

    assert features["pr"][0, 0, 0] == pytest.approx(1.0)
    assert np.isnan(features["pr"][0, 0, 1])
    np.testing.assert_array_equal(features["elevation"], elevation)


def test_preprocessing_fills_and_standardizes_but_keeps_wind_in_degrees(
    raw_features, statistics_for
):
    features = raw_features(2, seed=0)
    features["th"][0, 0, 0] = np.nan
    features["vs"][0, 0, 0] = np.nan
    degrees = features["th"].copy()
    statistics = statistics_for(features)

    processed = apply_preprocessing(features, statistics)

    np.testing.assert_array_equal(processed["th"], degrees)
    assert processed["vs"][0, 0, 0] == pytest.approx(
        (statistics["vs"]["median"] - statistics["vs"]["mean"]) / statistics["vs"]["std"]
    )
    assert abs(processed["elevation"].mean()) < 1e-4


def test_clean_features_leaves_only_wind_direction_missing(raw_features, statistics_for):
    statistics = statistics_for(raw_features(4, seed=1))
    features = raw_features(2, seed=2)
    for name in ("tmmn", "tmmx", "vs", "sph", "erc", "th"):
        features[name][0, 0, 0] = 0.0

    cleaned = clean_features(features, statistics)

    for name, values in cleaned.items():
        if name == "th":
            assert np.isnan(values[0, 0, 0])
        else:
            assert np.isfinite(values).all(), name
