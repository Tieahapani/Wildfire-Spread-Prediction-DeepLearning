"""Shared pytest fixtures: dataset and model locations from the bin/wf environment."""

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def data_root() -> Path:
    # bin/wf points WF_DATA at the shared copy, so personal clones see the same data.
    return Path(os.environ.get("WF_DATA", REPO_ROOT / "data"))
