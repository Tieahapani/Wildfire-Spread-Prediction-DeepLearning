import numpy as np
import pytest
import torch

from wfspread.constants import CH, NDWS_CHANNELS
from wfspread.data.batch import check_batch
from wfspread.data.channels import SOURCE_FEATURE, to_inputs
from wfspread.data.cleaning import clean_features


def test_channels_follow_the_batch_format_order(raw_features):
    features = raw_features(3, seed=0)

    inputs = to_inputs(features)

    assert inputs.shape == (3, len(NDWS_CHANNELS), 16, 16)
    assert inputs.dtype == np.float32
    for channel, name in SOURCE_FEATURE.items():
        np.testing.assert_array_equal(inputs[:, CH[channel]], features[name])


def test_wind_direction_becomes_sin_and_cos_and_missing_is_zero(raw_features):
    features = raw_features(1, seed=0)
    features["th"][0, 0, :3] = [90.0, 180.0, np.nan]

    inputs = to_inputs(features)

    sin, cos = inputs[0, CH["wind_dir_sin"], 0], inputs[0, CH["wind_dir_cos"], 0]
    assert sin[:3] == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)
    assert cos[:3] == pytest.approx([0.0, -1.0, 0.0], abs=1e-6)


def test_unknown_previous_fire_is_not_fire(raw_features):
    features = raw_features(1, seed=0)
    features["PrevFireMask"][0, 0, :3] = [-1, 0, 1]

    inputs = to_inputs(features)

    assert inputs[0, CH["prev_fire"], 0, :3].tolist() == [0.0, 0.0, 1.0]


def test_single_tile_has_no_batch_dimension(raw_features):
    tile = {name: values[0] for name, values in raw_features(1, seed=0).items()}

    assert to_inputs(tile).shape == (len(NDWS_CHANNELS), 16, 16)


def test_cleaned_inputs_pass_the_batch_check(raw_features, statistics_for):
    statistics = statistics_for(raw_features(4, seed=1))
    features = raw_features(2, seed=2)
    features["th"][0, 0, 0] = 999.0

    inputs = to_inputs(clean_features(features, statistics))

    check_batch({"inputs": torch.from_numpy(inputs), "sample_id": torch.arange(2)})
