import pytest
import torch

from wfspread.eval.final import (
    PERSISTENCE,
    ModelSpec,
    best_checkpoint,
    build_scored_model,
    collect,
    read_manifest,
    seed_mean_pr_auc,
    validation_thresholds,
)
from wfspread.eval.metrics import pr_auc
from wfspread.models.unet import UNet
from wfspread.models.wrappers import ChannelSubset


def _tiles(n=4, size=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 2, (n, size, size), generator=generator, dtype=torch.int8)
    prev = torch.randint(0, 2, (n, size, size), generator=generator, dtype=torch.int8)
    return labels, prev


def test_persistence_predicts_todays_fire():
    labels, prev = _tiles()
    prev[0, 0, 0] = -1
    dataset = [
        {"inputs": torch.zeros(13, 8, 8), "prev_fire": p, "next_fire": y, "sample_id": i}
        for i, (y, p) in enumerate(zip(labels, prev, strict=True))
    ]

    probs, next_fire, _ = collect(None, dataset, torch.device("cpu"))

    torch.testing.assert_close(probs, (prev == 1).float())
    assert torch.equal(next_fire, labels)
    assert probs[0, 0, 0] == 0.0


def test_seed_mean_pr_auc_averages_runs():
    labels, prev = _tiles()
    generator = torch.Generator().manual_seed(1)
    runs = torch.rand(4, 2, 8, 8, generator=generator)

    expected = (pr_auc(runs[:, 0], labels) + pr_auc(runs[:, 1], labels)) / 2

    assert seed_mean_pr_auc(runs, labels, prev) == pytest.approx(expected)


def test_each_run_gets_its_own_threshold():
    labels, _ = _tiles()
    perfect = labels.float()
    shifted = labels.float() * 0.5 + 0.2

    thresholds = validation_thresholds(torch.stack([perfect, shifted], dim=1), labels)

    assert thresholds == pytest.approx([1.0, 0.7])


def test_ablation_checkpoint_loads_into_the_wrapped_model(tmp_path):
    torch.manual_seed(0)
    trained = ChannelSubset(UNet(in_ch=12, enc_channels=[8, 16], bottleneck_ch=16), ["prev_fire"])
    state = {"model." + key: value for key, value in trained.state_dict().items()}
    torch.save({"state_dict": state}, tmp_path / "best-003.ckpt")
    config = tmp_path / "unet.yaml"
    config.write_text("name: unet\nargs: {in_ch: 13, enc_channels: [8, 16], bottleneck_ch: 16}\n")
    spec = ModelSpec("ablation", model_config=config, drop_channels=("prev_fire",))

    model = build_scored_model(spec, tmp_path / "best-003.ckpt")

    inputs = torch.randn(1, 13, 8, 8)
    torch.testing.assert_close(model(inputs), trained.eval()(inputs))


def test_best_checkpoint_needs_exactly_one(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    with pytest.raises(ValueError, match="found 0"):
        best_checkpoint(tmp_path)


def test_manifest_resolves_paths_and_checks_comparisons(tmp_path):
    (tmp_path / "final.yaml").write_text(
        "models:\n"
        "  persistence: {kind: persistence}\n"
        "  unet: {model: unet.yaml, runs: [a/s0, a/s1]}\n"
        "comparisons:\n  - [unet, persistence]\n"
    )

    specs, comparisons = read_manifest(tmp_path / "final.yaml", tmp_path / "runs")

    assert specs[0].kind == PERSISTENCE
    assert specs[1].model_config == (tmp_path / "unet.yaml").resolve()
    assert specs[1].runs == (tmp_path / "runs/a/s0", tmp_path / "runs/a/s1")
    assert comparisons == [("unet", "persistence")]

    (tmp_path / "bad.yaml").write_text("models:\n  unet: {}\ncomparisons:\n  - [unet, mlp]\n")
    with pytest.raises(ValueError, match="unlisted"):
        read_manifest(tmp_path / "bad.yaml", tmp_path)
