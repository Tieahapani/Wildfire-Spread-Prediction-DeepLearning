import pytest
import torch

from wfspread.constants import CH, NDWS_CHANNELS
from wfspread.models.unet import UNet
from wfspread.models.wrappers import ChannelSubset


def test_dropped_channel_has_no_effect_on_the_output():
    torch.manual_seed(0)
    model = ChannelSubset(UNet(in_ch=12, enc_channels=[8, 16], bottleneck_ch=16), ["prev_fire"])
    model.eval()
    inputs = torch.randn(2, len(NDWS_CHANNELS), 16, 16)
    changed = inputs.clone()
    changed[:, CH["prev_fire"]] = 5.0

    torch.testing.assert_close(model(inputs), model(changed))


def test_weights_are_stored_under_the_wrapped_model_only():
    model = ChannelSubset(UNet(in_ch=12, enc_channels=[8, 16], bottleneck_ch=16), ["prev_fire"])

    keys = list(model.state_dict())

    assert all(key.startswith("model.") for key in keys)
    assert "keep" not in keys


def test_unknown_channel_is_rejected():
    with pytest.raises(ValueError, match="unknown channels"):
        ChannelSubset(UNet(), ["fuel_moisture"])
