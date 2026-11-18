"""PyTorch dataset and DataLoader for cleaned NDWS wildfire tiles.

Reads cleaned HDF5 files from ``data/processed/ndws-v1/`` and returns batches
in the format defined in ``docs/batch-format.md``.
"""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from wfspread.constants import FIRE, IGNORE_INDEX, NO_FIRE
from wfspread.data.batch import Batch
from wfspread.data.channels import to_inputs
from wfspread.data.transforms import transform_tile


class NDWSDataset(Dataset):
    """One cleaned NDWS split, with optional augmentation.

    Args:
        data_dir: Directory containing cleaned HDF5 files.
        split: One of ``"train"``, ``"val"`` or ``"test"``.
        augment: Apply random flips and 90° rotations (training only).
        crop_size: Random square crop side; ``None`` keeps the full 64×64 tile.
        preload: Read the entire split into memory at init. Uses ~3 GB for
            training but removes the HDF5 read bottleneck, making epochs much
            faster.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        augment: bool = False,
        crop_size: int | None = None,
        preload: bool = False,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split: {split}")
        if augment and split != "train":
            raise ValueError("Augmentation is only allowed for training")
        if crop_size is not None and not 1 <= crop_size <= 64:
            raise ValueError("crop_size must be between 1 and 64")

        self.path = Path(data_dir) / f"{split}.h5"
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        self.split = split
        self.augment = augment
        self.crop_size = crop_size
        self._file: h5py.File | None = None
        self._pid: int | None = None
        self._cache: dict[str, np.ndarray] | None = None
        self._sample_ids: np.ndarray | None = None

        with h5py.File(self.path, "r") as f:
            self._len = int(f.attrs["num_records"])
            if not {"FireMask", "PrevFireMask"} <= set(f["features"]):
                raise ValueError("HDF5 file is missing required fire masks")
            if preload:
                self._cache = {name: f["features"][name][:] for name in f["features"]}
                self._sample_ids = f["record_indices"][:]

    def __len__(self) -> int:
        return self._len

    def _get_file(self) -> h5py.File:
        """Open an HDF5 handle per worker process."""
        pid = os.getpid()
        if self._file is None or self._pid != pid:
            self._file = h5py.File(self.path, "r")
            self._pid = pid
        return self._file

    def __getitem__(self, index: int) -> Batch:
        if index < 0 or index >= self._len:
            raise IndexError(index)

        if self._cache is not None:
            features = {name: self._cache[name][index] for name in self._cache}
            sample_id = int(self._sample_ids[index])
        else:
            f = self._get_file()
            features = {name: f["features"][name][index] for name in f["features"]}
            sample_id = int(f["record_indices"][index])

        if self.augment:
            features = transform_tile(
                features,
                horizontal_flip=bool(torch.randint(2, ()).item()),
                vertical_flip=bool(torch.randint(2, ()).item()),
                rotations_ccw=int(torch.randint(4, ()).item()),
            )

        if self.crop_size is not None:
            h, w = features["FireMask"].shape
            size = self.crop_size
            top = int(torch.randint(h - size + 1, ()).item())
            left = int(torch.randint(w - size + 1, ()).item())
            features = {
                name: arr[top : top + size, left : left + size] for name, arr in features.items()
            }

        prev_fire = _to_mask(features["PrevFireMask"])
        next_fire = _to_mask(features["FireMask"])
        inputs = to_inputs(features)

        return Batch(
            inputs=torch.from_numpy(np.ascontiguousarray(inputs, dtype=np.float32)),
            sample_id=torch.tensor(sample_id, dtype=torch.int64),
            prev_fire=torch.from_numpy(np.ascontiguousarray(prev_fire)),
            next_fire=torch.from_numpy(np.ascontiguousarray(next_fire)),
        )


def _to_mask(raw: np.ndarray) -> np.ndarray:
    """Convert the raw float32 fire mask to int8 {-1, 0, 1}."""
    mask = np.full(raw.shape, NO_FIRE, dtype=np.int8)
    mask[raw == FIRE] = FIRE
    mask[raw == IGNORE_INDEX] = IGNORE_INDEX
    return mask


def create_dataloader(
    data_dir: str | Path,
    split: str = "train",
    batch_size: int = 64,
    num_workers: int = 4,
    crop_size: int | None = None,
    seed: int = 0,
    preload: bool = False,
) -> DataLoader:
    """Create a reproducible NDWS DataLoader.

    Args:
        data_dir: Path to ``ndws-v1/`` with the cleaned HDF5 files.
        split: ``"train"``, ``"val"`` or ``"test"``.
        batch_size: Tiles per batch.
        num_workers: Parallel data loading workers.
        crop_size: Random square crop side for training; ``None`` uses full tiles.
        seed: Seed for shuffling and augmentation.
        preload: Load the entire split into memory for faster epochs.

    Returns:
        A DataLoader yielding ``Batch`` dicts.
    """
    training = split == "train"
    dataset = NDWSDataset(
        data_dir=data_dir,
        split=split,
        augment=training,
        crop_size=crop_size,
        preload=preload,
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        drop_last=training,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        generator=generator,
    )
