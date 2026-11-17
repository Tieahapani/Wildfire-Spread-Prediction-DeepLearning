"""Tests for the NDWS data loader."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from wfspread.data.batch import check_batch
from wfspread.data.loader import NDWSDataset, create_dataloader


@pytest.fixture
def tiny_data(tmp_path: Path) -> Path:
    """Write a small cleaned HDF5 that the loader can read."""
    n, h, w = 8, 64, 64
    rng = np.random.RandomState(42)

    for split in ("train", "val"):
        path = tmp_path / f"{split}.h5"
        with h5py.File(path, "w") as f:
            f.attrs["num_records"] = n
            grp = f.create_group("features")
            for name in (
                "NDVI",
                "elevation",
                "erc",
                "pdsi",
                "population",
                "pr",
                "sph",
                "tmmn",
                "tmmx",
                "vs",
            ):
                grp.create_dataset(name, data=rng.randn(n, h, w).astype(np.float32))
            grp.create_dataset("th", data=rng.uniform(0, 360, (n, h, w)).astype(np.float32))
            fire = rng.choice([-1, 0, 1], size=(n, h, w), p=[0.02, 0.90, 0.08])
            grp.create_dataset("FireMask", data=fire.astype(np.float32))
            grp.create_dataset("PrevFireMask", data=fire.astype(np.float32))
            f.create_dataset("record_indices", data=np.arange(100, 100 + n, dtype=np.int64))

    return tmp_path


class TestNDWSDataset:
    """The dataset returns valid batches."""

    def test_len(self, tiny_data: Path):
        ds = NDWSDataset(tiny_data, split="train", augment=True)
        assert len(ds) == 8

    def test_getitem_batch_format(self, tiny_data: Path):
        ds = NDWSDataset(tiny_data, split="val", augment=False)
        sample = ds[0]
        assert "next_fire" in sample
        assert "prev_fire" in sample
        assert "inputs" in sample
        assert "sample_id" in sample
        assert sample["inputs"].dtype == torch.float32
        assert sample["inputs"].shape == (13, 64, 64)
        assert sample["next_fire"].dtype == torch.int8
        assert sample["prev_fire"].dtype == torch.int8

    def test_sample_id_uses_record_index(self, tiny_data: Path):
        ds = NDWSDataset(tiny_data, split="val", augment=False)
        sample = ds[0]
        assert sample["sample_id"].item() == 100

    def test_augmented_sample_valid(self, tiny_data: Path):
        ds = NDWSDataset(tiny_data, split="train", augment=True)
        sample = ds[0]
        assert sample["inputs"].shape == (13, 64, 64)
        assert torch.isfinite(sample["inputs"]).all()

    def test_crop(self, tiny_data: Path):
        ds = NDWSDataset(tiny_data, split="train", augment=True, crop_size=32)
        sample = ds[0]
        assert sample["inputs"].shape == (13, 32, 32)
        assert sample["next_fire"].shape == (32, 32)


class TestCreateDataloader:
    """The factory returns working loaders."""

    def test_train_loader(self, tiny_data: Path):
        loader = create_dataloader(tiny_data, split="train", batch_size=4, num_workers=0)
        batch = next(iter(loader))
        check_batch(batch)
        assert batch["inputs"].shape[0] == 4

    def test_val_loader_no_drop(self, tiny_data: Path):
        loader = create_dataloader(tiny_data, split="val", batch_size=4, num_workers=0)
        total = sum(b["inputs"].shape[0] for b in loader)
        assert total == 8

    def test_mask_values(self, tiny_data: Path):
        loader = create_dataloader(tiny_data, split="val", batch_size=8, num_workers=0)
        batch = next(iter(loader))
        for key in ("prev_fire", "next_fire"):
            vals = set(batch[key].unique().tolist())
            assert vals <= {-1, 0, 1}
