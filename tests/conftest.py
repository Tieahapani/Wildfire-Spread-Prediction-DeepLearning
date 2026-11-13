"""Shared pytest fixtures: dataset locations and fake tiles in the batch format."""

import os
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch import Tensor

from wfspread.constants import CH, IGNORE_INDEX

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def data_root() -> Path:
    # bin/wf points WF_DATA at the shared copy, so personal clones see the same data.
    return Path(os.environ.get("WF_DATA", REPO_ROOT / "data"))


def _fake_tiles(n_tiles: int, seed: int, size: int = 16) -> dict[str, Tensor]:
    g = torch.Generator().manual_seed(seed)
    inputs = torch.randn(n_tiles, 13, size, size, generator=g)
    prev = torch.zeros(n_tiles, size, size, dtype=torch.int8)
    for i in range(n_tiles):
        r, c = torch.randint(2, size - 6, (2,), generator=g).tolist()
        prev[i, r : r + 3, c : c + 3] = 1
    next_fire = prev.clone()
    next_fire[:, :, 1:] |= prev[:, :, :-1]
    next_fire[:, 0, :] = IGNORE_INDEX
    inputs[:, CH["prev_fire"]] = prev.float()
    return {
        "inputs": inputs,
        "prev_fire": prev,
        "next_fire": next_fire,
        "sample_id": torch.arange(n_tiles, dtype=torch.int64),
    }


@pytest.fixture
def fake_tiles() -> Callable[..., dict[str, Tensor]]:
    """Builds fake tiles where tomorrow's fire is today's fire grown one pixel right.

    Call it as `fake_tiles(n_tiles, seed, size=16)`. The top row of `next_fire` is
    unknown (-1), so mask handling is exercised too.
    """
    return _fake_tiles
