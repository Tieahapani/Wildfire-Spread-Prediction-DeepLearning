"""Check exact duplicate NDWS tiles across train and validation."""

import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

EXPECTED_COUNTS = {
    "train": 14_979,
    "val": 1_877,
    "test": 1_689,
}

FEATURE_NAMES = (
    "FireMask",
    "NDVI",
    "PrevFireMask",
    "elevation",
    "erc",
    "pdsi",
    "population",
    "pr",
    "sph",
    "th",
    "tmmn",
    "tmmx",
    "vs",
)

TILE_SHAPE = (64, 64)


def hash_tile(h5_file: h5py.File, index: int) -> str:
    """Calculate a SHA-256 hash of all features in one tile."""
    digest = hashlib.sha256()

    for name in FEATURE_NAMES:
        values = np.asarray(
            h5_file["features"][name][index],
            dtype=np.float32,
        )

        if values.shape != TILE_SHAPE:
            raise ValueError(f"{name}, tile {index}: unexpected shape {values.shape}")

        values = np.ascontiguousarray(values, dtype="<f4")
        digest.update(values.tobytes())

    return digest.hexdigest()


def inspect_split(path: Path, split: str) -> dict:
    """Read one tile at a time and calculate its hash."""
    hashes = {}
    duplicate_count = 0

    with h5py.File(path, "r") as h5_file:
        actual_count = int(h5_file.attrs["num_records"])

        if actual_count != EXPECTED_COUNTS[split]:
            raise ValueError(
                f"{split}: expected {EXPECTED_COUNTS[split]} records, found {actual_count}"
            )

        if set(h5_file["features"].keys()) != set(FEATURE_NAMES):
            raise ValueError(f"{split}: unexpected feature names")

        for index in range(actual_count):
            tile_hash = hash_tile(h5_file, index)

            if tile_hash in hashes:
                duplicate_count += 1
            else:
                hashes[tile_hash] = index

    return {
        "count": actual_count,
        "hashes": hashes,
        "within_split_duplicates": duplicate_count,
    }


def main() -> None:
    """Compare train and validation hashes; check test count only."""
    data_root = Path(os.environ["WF_DATA"])
    data_dir = data_root / "processed" / "ndws"

    train = inspect_split(data_dir / "train.h5", "train")
    val = inspect_split(data_dir / "val.h5", "val")

    train_hashes = set(train["hashes"])
    val_hashes = set(val["hashes"])

    shared_hashes = train_hashes & val_hashes

    test_path = data_dir / "test.h5"

    with h5py.File(test_path, "r") as test_file:
        test_count = int(test_file.attrs["num_records"])

    if test_count != EXPECTED_COUNTS["test"]:
        raise ValueError(f"test: expected {EXPECTED_COUNTS['test']} records, found {test_count}")

    report = {
        "dataset": "NDWS",
        "method": "SHA-256 over all 13 float32 feature arrays",
        "train": {
            "records": train["count"],
            "within_split_duplicates": train["within_split_duplicates"],
        },
        "validation": {
            "records": val["count"],
            "within_split_duplicates": val["within_split_duplicates"],
        },
        "train_validation": {
            "shared_unique_hashes": len(shared_hashes),
            "overlap_verified": len(shared_hashes) == 0,
        },
        "test": {
            "records": test_count,
            "count_verified": True,
            "hash_check": "not performed",
        },
    }

    report_path = data_dir / "overlap_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print("\nNDWS HASH OVERLAP REPORT")
    print("=" * 35)
    print(f"Training records: {train['count']:,}")
    print(f"Validation records: {val['count']:,}")
    print(f"Test records: {test_count:,}")
    print()
    print(
        "Training duplicates:",
        train["within_split_duplicates"],
    )
    print(
        "Validation duplicates:",
        val["within_split_duplicates"],
    )
    print("Train/validation shared hashes:", len(shared_hashes))
    print("Test hash check: skipped (counts only)")
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
