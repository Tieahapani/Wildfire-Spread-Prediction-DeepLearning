"""Convert NDWS TFRecords into incrementally written HDF5 files."""

import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
from tfrecord.reader import tfrecord_loader

EXPECTED_COUNTS = {
    "train": 14_979,
    "val": 1_877,
    "test": 1_689,
}

SPLIT_PATTERNS = {
    "train": "*_train_*.tfrecord",
    "val": "*_eval_*.tfrecord",
    "test": "*_test_*.tfrecord",
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


def get_split_files(record_dir: Path, split: str) -> list[Path]:
    """Find all TFRecord files belonging to a split."""
    files = sorted(record_dir.glob(SPLIT_PATTERNS[split]))

    if not files:
        raise FileNotFoundError(f"No TFRecord files found for {split} in {record_dir}")

    return files


def validate_record(record: dict) -> None:
    """Validate the features and shapes of an NDWS record."""
    if set(record) != set(FEATURE_NAMES):
        missing = set(FEATURE_NAMES) - set(record)
        unexpected = set(record) - set(FEATURE_NAMES)
        raise ValueError(f"Feature mismatch. Missing: {missing}; unexpected: {unexpected}")

    for name in FEATURE_NAMES:
        values = np.asarray(record[name])

        if values.shape != (4096,):
            raise ValueError(f"{name} has unexpected shape {values.shape}")

        if not np.issubdtype(values.dtype, np.floating):
            raise ValueError(f"{name} has unexpected dtype {values.dtype}")


def convert_split(record_dir: Path, output_dir: Path, split: str) -> int:
    """Convert one dataset split, processing one record at a time."""
    files = get_split_files(record_dir, split)
    output_path = output_dir / f"{split}.h5"

    if output_path.exists():
        raise FileExistsError(
            f"{output_path} already exists. Refusing to overwrite processed data."
        )

    temporary_path = None
    total_records = 0

    try:
        with tempfile.NamedTemporaryFile(
            dir=output_dir,
            suffix=".h5",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        with h5py.File(temporary_path, "w") as h5_file:
            features_group = h5_file.create_group("features")

            for name in FEATURE_NAMES:
                features_group.create_dataset(
                    name,
                    shape=(0, *TILE_SHAPE),
                    maxshape=(None, *TILE_SHAPE),
                    chunks=(1, *TILE_SHAPE),
                    dtype=np.float32,
                    compression="lzf",
                )

            string_dtype = h5py.string_dtype(encoding="utf-8")

            tile_ids = h5_file.create_dataset(
                "tile_ids",
                shape=(0,),
                maxshape=(None,),
                chunks=(256,),
                dtype=string_dtype,
            )

            source_files = h5_file.create_dataset(
                "source_files",
                shape=(0,),
                maxshape=(None,),
                chunks=(256,),
                dtype=string_dtype,
            )

            record_indices = h5_file.create_dataset(
                "record_indices",
                shape=(0,),
                maxshape=(None,),
                chunks=(256,),
                dtype=np.int64,
            )

            for source_file in files:
                print(f"Processing {source_file.name}...")

                for record_index, record in enumerate(tfrecord_loader(str(source_file), None)):
                    validate_record(record)

                    current_index = total_records
                    new_size = current_index + 1

                    for name in FEATURE_NAMES:
                        dataset = features_group[name]
                        dataset.resize(new_size, axis=0)
                        dataset[current_index] = np.asarray(
                            record[name],
                            dtype=np.float32,
                        ).reshape(TILE_SHAPE)

                    tile_id = f"{source_file.name}:{record_index}"

                    tile_ids.resize(new_size, axis=0)
                    tile_ids[current_index] = tile_id

                    source_files.resize(new_size, axis=0)
                    source_files[current_index] = source_file.name

                    record_indices.resize(new_size, axis=0)
                    record_indices[current_index] = record_index

                    total_records = new_size

            if total_records != EXPECTED_COUNTS[split]:
                raise ValueError(
                    f"{split}: expected {EXPECTED_COUNTS[split]} records, found {total_records}"
                )

            h5_file.attrs["split"] = split
            h5_file.attrs["num_records"] = total_records
            h5_file.attrs["tile_shape"] = TILE_SHAPE
            h5_file.attrs["feature_names"] = np.asarray(
                FEATURE_NAMES,
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            h5_file.attrs["tile_id_format"] = "source_filename:record_index"

        temporary_path.replace(output_path)

    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    print(f"Finished {split}: {total_records:,} records")
    print(f"Output: {output_path}")

    return total_records


def main() -> None:
    """Convert all NDWS splits and write dataset metadata."""
    data_root = Path(os.environ["WF_DATA"])
    record_dir = data_root / "raw" / "ndws"
    output_dir = data_root / "processed" / "ndws"

    output_dir.mkdir(parents=True, exist_ok=True)

    targets = [output_dir / f"{split}.h5" for split in EXPECTED_COUNTS]
    targets.append(output_dir / "metadata.json")

    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError(
            f"Existing outputs found: {existing}. "
            "Move or remove them before starting a new conversion."
        )

    metadata = {
        "dataset": "NDWS",
        "tile_shape": TILE_SHAPE,
        "features": list(FEATURE_NAMES),
        "tile_id_format": "source_filename:record_index",
        "splits": {},
    }

    for split, expected_count in EXPECTED_COUNTS.items():
        actual_count = convert_split(record_dir, output_dir, split)

        metadata["splits"][split] = {
            "expected_records": expected_count,
            "actual_records": actual_count,
            "verified": actual_count == expected_count,
        }

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print("\nAll splits converted successfully.")
    print(f"Total records: {sum(EXPECTED_COUNTS.values()):,}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
