"""PyTorch dataset and DataLoader for cleaned NDWS wildfire tiles."""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import torch
from scripts.data.transforms import transform_tile
from torch.utils.data import DataLoader, Dataset

from wfspread.data.channels import to_inputs


class NDWSDataset(Dataset):
    """Load cleaned NDWS tiles and convert them to model-ready samples."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        augment: bool = False,
        crop_size: int | None = None,
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

        self._file = None
        self._pid = None

        with h5py.File(self.path, "r") as f:
            self.length = int(f.attrs["num_records"])

            if not {"FireMask", "PrevFireMask"} <= set(f["features"]):
                raise ValueError("HDF5 file is missing required fire masks")

    def __len__(self):
        return self.length

    def _get_file(self):
        """Open an HDF5 handle separately in each worker process."""
        pid = os.getpid()

        if self._file is None or self._pid != pid:
            self._file = h5py.File(self.path, "r")
            self._pid = pid

        return self._file

    def __getitem__(self, index):
        if index < 0 or index >= self.length:
            raise IndexError(index)

        f = self._get_file()

        features = {name: f["features"][name][index] for name in f["features"]}

        # Apply identical spatial transformations to every feature.
        if self.augment:
            features = transform_tile(
                features,
                horizontal_flip=bool(torch.randint(2, ()).item()),
                vertical_flip=bool(torch.randint(2, ()).item()),
                rotations_ccw=int(torch.randint(4, ()).item()),
            )

        # Optional shared random crop.
        if self.crop_size is not None:
            height, width = features["FireMask"].shape
            size = self.crop_size

            top = int(torch.randint(height - size + 1, ()).item())
            left = int(torch.randint(width - size + 1, ()).item())

            features = {
                name: array[top : top + size, left : left + size]
                for name, array in features.items()
            }

        # Preserve raw masks for loss and metrics.
        prev_fire = np.asarray(features["PrevFireMask"], dtype=np.int8)
        target = np.asarray(features["FireMask"], dtype=np.int8)

        # Convert environmental features to model channels.
        inputs = to_inputs(features)
        inputs = np.ascontiguousarray(inputs, dtype=np.float32)

        return {
            "inputs": torch.from_numpy(inputs),
            "prev_fire": torch.from_numpy(np.ascontiguousarray(prev_fire)),
            "target": torch.from_numpy(np.ascontiguousarray(target)),
            # Temporary split-local position; see sample_id note below.
            "sample_id": torch.tensor(index, dtype=torch.int64),
        }


def create_dataloader(
    data_dir: str | Path,
    split: str = "train",
    batch_size: int = 64,
    num_workers: int = 4,
    crop_size: int | None = None,
    seed: int = 42,
) -> DataLoader:
    """Create a reproducible NDWS DataLoader."""
    training = split == "train"

    dataset = NDWSDataset(
        data_dir=data_dir,
        split=split,
        augment=training,
        crop_size=crop_size,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

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
