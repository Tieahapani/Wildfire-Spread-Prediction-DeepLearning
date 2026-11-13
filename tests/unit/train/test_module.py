from dataclasses import replace

import pytest
import torch
from torch import nn

from wfspread.train.config import TrainConfig
from wfspread.train.losses import masked_focal
from wfspread.train.module import FireModule, warmup_cosine


class _TinyNet(nn.Module):
    def __init__(self, channels: int = 13):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 1, 1)
        )

    def forward(self, inputs):
        return self.net(inputs).squeeze(1)


def test_warmup_rises_linearly():
    assert warmup_cosine(0, 4, 100, 0.0) == pytest.approx(0.25)
    assert warmup_cosine(3, 4, 100, 0.0) == pytest.approx(1.0)


def test_cosine_decays_to_min_ratio():
    assert warmup_cosine(4, 4, 100, 0.01) == pytest.approx(1.0)
    assert warmup_cosine(52, 4, 100, 0.01) == pytest.approx(0.505)
    assert warmup_cosine(100, 4, 100, 0.01) == pytest.approx(0.01)


def test_forward_returns_float32_tile_logits():
    module = FireModule(_TinyNet(), TrainConfig())
    logits = module(torch.randn(2, 13, 8, 8))
    assert logits.shape == (2, 8, 8)
    assert logits.dtype == torch.float32


def test_forward_rejects_a_channel_dimension_in_the_output():
    model = nn.Conv2d(13, 1, 1)
    with pytest.raises(ValueError, match="expected"):
        FireModule(model, TrainConfig())(torch.randn(2, 13, 8, 8))


def test_config_from_yaml_keeps_defaults(tmp_path):
    path = tmp_path / "train.yaml"
    path.write_text("lr: 0.001\nseed: 7\n")
    config = TrainConfig.from_yaml(path)
    assert (config.lr, config.seed, config.patience) == (0.001, 7, 15)


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "train.yaml"
    path.write_text("learning_rate: 0.001\n")
    with pytest.raises(ValueError, match="learning_rate"):
        TrainConfig.from_yaml(path)


def test_repo_default_config_loads():
    from wfspread.train.run import REPO_ROOT

    assert TrainConfig.from_yaml(REPO_ROOT / "configs/train/default.yaml") == TrainConfig()


def test_config_base_file_supplies_the_other_settings(tmp_path):
    (tmp_path / "base.yaml").write_text("lr: 0.001\npos_weight: 2.0\n")
    path = tmp_path / "arm.yaml"
    path.write_text("base: base.yaml\npos_weight: 3.0\n")
    config = TrainConfig.from_yaml(path)
    assert (config.lr, config.pos_weight) == (0.001, 3.0)


def test_config_base_cannot_include_itself(tmp_path):
    path = tmp_path / "loop.yaml"
    path.write_text("base: loop.yaml\n")
    with pytest.raises(ValueError, match="includes itself"):
        TrainConfig.from_yaml(path)


LOSS_ARMS = {
    "loss-bce": ({"bce": 1.0}, 1.0),
    "loss-wbce": ({"bce": 1.0}, 3.0),
    "loss-focal": ({"focal": 1.0}, 1.0),
    "loss-combo": ({"bce": 0.4, "dice": 0.3, "focal": 0.3}, 3.0),
}


@pytest.mark.parametrize("arm", LOSS_ARMS)
def test_loss_comparison_arms_differ_only_in_the_loss(arm):
    from wfspread.train.run import REPO_ROOT

    config = TrainConfig.from_yaml(REPO_ROOT / "configs" / "experiment" / f"{arm}.yaml")
    terms, pos_weight = LOSS_ARMS[arm]
    assert (config.loss_terms, config.pos_weight) == (terms, pos_weight)
    loss_settings = {"loss_terms": terms, "pos_weight": pos_weight}
    assert config == replace(TrainConfig(), **loss_settings)


def test_module_trains_with_the_configured_loss():
    torch.manual_seed(0)
    module = FireModule(_TinyNet(), TrainConfig(loss_terms={"focal": 1.0}))
    inputs = torch.randn(2, 13, 8, 8)
    next_fire = torch.randint(0, 2, (2, 8, 8), dtype=torch.int8)
    with torch.no_grad():
        loss = module.loss_fn(module(inputs), next_fire)
        expected = masked_focal(module(inputs), next_fire)
    assert loss.item() == pytest.approx(expected.item())


def test_module_rejects_an_unknown_loss():
    with pytest.raises(ValueError, match="unknown loss"):
        FireModule(_TinyNet(), TrainConfig(loss_terms={"hinge": 1.0}))
