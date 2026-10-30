"""Environment checks: container build, GPU, and integrity of the downloaded data."""

import struct
from pathlib import Path

import pytest
import torch

NDWS_SPLIT_SIZES = {
    "train": 14979,
    "eval": 1877,
    "test": 1689,
}  # Huot et al. 2022; NDWS "eval" = our val


def test_container_torch_is_pinned():
    # A PyPI wheel replacing the NGC build would lose the GB10-tuned kernels.
    assert "nv26.9" in torch.__version__ or "nv26.09" in torch.__version__


@pytest.mark.gpu
def test_gpu_is_gb10():
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_capability(0) == (12, 1)


def _count_tfrecords(path: Path) -> int:
    # TFRecord framing: u64 length, u32 length CRC, payload, u32 payload CRC.
    n = 0
    with open(path, "rb") as fh:
        while header := fh.read(12):
            (length,) = struct.unpack("<Q", header[:8])
            fh.seek(length + 4, 1)
            n += 1
    return n


@pytest.mark.data
@pytest.mark.parametrize(("split", "expected"), NDWS_SPLIT_SIZES.items())
def test_ndws_split_sizes(data_root, split, expected):
    files = sorted(
        (data_root / "raw" / "ndws").glob(f"next_day_wildfire_spread_{split}_*.tfrecord")
    )
    assert files, f"missing NDWS {split} files"
    assert sum(_count_tfrecords(f) for f in files) == expected


@pytest.mark.data
@pytest.mark.parametrize("size", ["nano", "tiny", "base"])
def test_galileo_weights_present(data_root, size):
    model_dir = data_root / "models" / "galileo" / size
    assert (model_dir / "encoder.pt").stat().st_size > 0
    assert (model_dir / "config.json").is_file()
