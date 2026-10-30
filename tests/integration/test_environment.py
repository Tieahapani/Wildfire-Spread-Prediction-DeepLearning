"""Environment smoke tests: run with `bin/wf pytest -q`."""
import struct
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
NDWS = ROOT / "data" / "raw" / "ndws"
GALILEO = ROOT / "data" / "models" / "galileo"


def test_container_torch_pinned():
    # The NGC build must never be replaced by a PyPI wheel.
    assert "nv26.9" in torch.__version__ or "nv26.09" in torch.__version__


def test_gpu_available():
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_capability(0) == (12, 1)


def _count_records(path: Path) -> int:
    n = 0
    with open(path, "rb") as fh:
        while header := fh.read(12):
            (length,) = struct.unpack("<Q", header[:8])
            fh.seek(length + 4, 1)
            n += 1
    return n


@pytest.mark.parametrize("split,expected", [("train", 14979), ("eval", 1877), ("test", 1689)])
def test_ndws_counts(split, expected):
    files = sorted(NDWS.glob(f"next_day_wildfire_spread_{split}_*.tfrecord"))
    assert files, f"missing NDWS {split} files"
    assert sum(_count_records(f) for f in files) == expected


@pytest.mark.parametrize("size", ["nano", "tiny", "base"])
def test_galileo_weights_present(size):
    assert (GALILEO / size / "encoder.pt").stat().st_size > 0
    assert (GALILEO / size / "config.json").exists()
