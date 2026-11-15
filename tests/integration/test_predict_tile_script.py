import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from wfspread.models.unet import UNet

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_script_draws_a_map_for_each_tile(tmp_path, raw_features, statistics_for):
    raw = raw_features(3, seed=0, size=64)
    ndws = tmp_path / "data" / "processed" / "ndws"
    ndws.mkdir(parents=True)
    with h5py.File(ndws / "val.h5", "w") as f:
        group = f.create_group("features")
        for name, values in raw.items():
            group.create_dataset(name, data=values)
        f.create_dataset("tile_ids", data=np.array([b"f:0", b"f:1", b"f:2"]))
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps(statistics_for(raw_features(4, seed=1))))
    args = {"enc_channels": [8, 16], "bottleneck_ch": 16}
    model_config = tmp_path / "model.yaml"
    model_config.write_text(yaml.safe_dump({"name": "unet", "args": args}))
    torch.save(UNet(**args).state_dict(), tmp_path / "weights.pt")
    out_dir = tmp_path / "maps"

    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "serve" / "predict_tile.py"),
        "--checkpoint",
        str(tmp_path / "weights.pt"),
        "--model",
        str(model_config),
        "--stats",
        str(stats),
        "--index",
        "0",
        "2",
        "--out-dir",
        str(out_dir),
        "--device",
        "cpu",
    ]
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), "WF_DATA": str(tmp_path / "data")}
    printed = subprocess.run(command, capture_output=True, text=True, check=True, env=env).stdout

    assert "val-00002: f:2" in printed
    for stem in ("val-00000", "val-00002"):
        assert (out_dir / f"{stem}.png").stat().st_size > 0
        probs = np.load(out_dir / f"{stem}.npy")
        assert probs.shape == (64, 64)
        assert np.all((probs >= 0) & (probs <= 1))
