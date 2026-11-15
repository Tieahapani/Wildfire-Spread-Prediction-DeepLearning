import pickle

import numpy as np
import pytest
import torch

from wfspread.data.channels import to_inputs
from wfspread.data.cleaning import clean_features
from wfspread.models.unet import UNet
from wfspread.serve.predict import FirePredictor, load_weights


class _NotAWeight:
    pass


def _small_unet(seed=0):
    torch.manual_seed(seed)
    return UNet(enc_channels=[8, 16], bottleneck_ch=16).eval()


def _predictor(raw_features, statistics_for):
    return FirePredictor(_small_unet(), statistics_for(raw_features(4, seed=1)))


def test_weights_load_from_a_lightning_checkpoint_or_a_state_dict(tmp_path):
    trained = _small_unet(seed=0)
    lightning_style = {"model." + key: value for key, value in trained.state_dict().items()}
    torch.save({"epoch": 3, "state_dict": lightning_style}, tmp_path / "best.ckpt")
    torch.save(trained.state_dict(), tmp_path / "weights.pt")
    inputs = torch.randn(1, 13, 16, 16)

    for name in ("best.ckpt", "weights.pt"):
        model = load_weights(_small_unet(seed=1), tmp_path / name)
        assert not model.training
        torch.testing.assert_close(model(inputs), trained(inputs))


def test_checkpoints_that_would_run_code_are_refused(tmp_path):
    torch.save({"state_dict": _small_unet().state_dict(), "x": _NotAWeight()}, tmp_path / "bad.pt")

    with pytest.raises(pickle.UnpicklingError):
        load_weights(_small_unet(), tmp_path / "bad.pt")


def test_mismatched_architecture_is_an_error(tmp_path):
    torch.save(UNet(enc_channels=[8, 16, 32]).state_dict(), tmp_path / "other.pt")

    with pytest.raises(RuntimeError):
        load_weights(_small_unet(), tmp_path / "other.pt")


def test_predictions_are_probabilities_per_pixel(raw_features, statistics_for):
    predictor = _predictor(raw_features, statistics_for)
    raw = raw_features(3, seed=2)
    for name in ("tmmn", "tmmx", "vs", "sph", "erc", "th"):
        raw[name][0, 0, 0] = 0.0

    probs = predictor.predict(raw)

    assert probs.shape == (3, 16, 16)
    assert probs.dtype == np.float32
    assert np.all((probs >= 0) & (probs <= 1))


def test_prediction_matches_cleaning_then_the_model(raw_features, statistics_for):
    predictor = _predictor(raw_features, statistics_for)
    raw = raw_features(2, seed=2)

    expected = torch.sigmoid(
        predictor.model(torch.from_numpy(to_inputs(clean_features(raw, predictor.statistics))))
    )

    np.testing.assert_allclose(predictor.predict(raw), expected.detach().numpy(), atol=1e-6)


def test_single_tile_and_no_day_after_mask_needed(raw_features, statistics_for):
    predictor = _predictor(raw_features, statistics_for)
    tile = {name: values[0] for name, values in raw_features(1, seed=2).items()}
    del tile["FireMask"]

    assert predictor.predict(tile).shape == (1, 16, 16)


def test_missing_feature_is_an_error(raw_features, statistics_for):
    predictor = _predictor(raw_features, statistics_for)
    raw = raw_features(1, seed=2)
    del raw["vs"]

    with pytest.raises(ValueError, match="vs"):
        predictor.predict(raw)
