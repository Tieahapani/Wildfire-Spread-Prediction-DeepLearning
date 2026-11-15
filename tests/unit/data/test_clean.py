import importlib.util
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]

_spec = importlib.util.spec_from_file_location("clean", REPO_ROOT / "scripts/data/clean.py")
clean = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(clean)


def test_statistics_skip_wind_direction_and_record_the_log_transform(raw_features):
    features = raw_features(2, seed=0)
    thresholds = {
        name: {"lower": float(np.min(a)), "upper": float(np.max(a))} for name, a in features.items()
    }

    statistics = clean.calculate_preprocessing_statistics(features, thresholds)

    assert "th" not in statistics
    assert statistics["pr"]["transform"] == "log1p"
    assert statistics["vs"]["transform"] == "none"


def test_eval_split_uses_training_statistics_and_keeps_every_tile(
    tmp_path, raw_features, statistics_for
):
    train_stats = statistics_for(raw_features(2, seed=1, size=64))
    val = raw_features(3, seed=2, size=64)
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
