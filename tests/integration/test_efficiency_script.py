import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_script_measures_the_unet_and_saves_json(tmp_path):
    out = tmp_path / "efficiency.json"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval" / "efficiency.py"),
        "--model",
        str(REPO_ROOT / "configs" / "model" / "unet.yaml"),
        "--device",
        "cpu",
        "--batch-size",
        "1",
        "--out",
        str(out),
    ]
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    printed = subprocess.run(command, capture_output=True, text=True, check=True, env=env).stdout

    assert "| unet | 1,929,889 |" in printed
    unet = json.loads(out.read_text())["models"]["unet"]
    assert unet["parameters"] == 1_929_889
    assert unet["flops_per_tile"] > 1e9
    assert unet["inference"][0]["batch_size"] == 1
