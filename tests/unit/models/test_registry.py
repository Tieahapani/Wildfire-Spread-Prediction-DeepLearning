from pathlib import Path

import pytest
import torch
import yaml

from wfspread.models.registry import MODELS, build_model, load_model
from wfspread.models.unet import UNet

MODEL_CONFIGS = sorted((Path(__file__).resolve().parents[3] / "configs" / "model").glob("*.yaml"))


def test_build_model_passes_arguments():
    model = build_model("unet", in_ch=12)
    assert isinstance(model, UNet)
    assert model(torch.randn(1, 12, 16, 16)).shape == (1, 16, 16)


def test_unknown_model_name_is_rejected():
    with pytest.raises(ValueError, match="unknown model 'cnn'"):
        build_model("cnn")


def test_unknown_argument_is_rejected():
    with pytest.raises(ValueError, match="does not accept"):
        build_model("unet", depth=4)


def test_config_needs_a_name(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("args: {in_ch: 13}\n")
    with pytest.raises(ValueError, match="needs `name`"):
        load_model(path)


@pytest.mark.parametrize("path", MODEL_CONFIGS, ids=lambda p: p.stem)
def test_every_model_config_builds_and_returns_logits(path):
    model = load_model(path)
    assert model(torch.randn(2, 13, 64, 64)).shape == (2, 64, 64)


def test_every_registered_model_has_a_config():
    names = {yaml.safe_load(p.read_text())["name"] for p in MODEL_CONFIGS}
    assert names == set(MODELS)
