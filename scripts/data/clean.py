"""Clean NDWS data and calculate training statistics.

Responsibilities:
- Remove duplicate training tiles.
- Remove entirely unknown training tiles.
- Apply feature-specific missing-value rules.
- Apply log1p to skewed features.
- Calculate training clipping thresholds.
- Generate histograms.
- Save cleaned data and training statistics.
- Process the validation split with the training statistics.

Wind direction stays in degrees with NaN where missing: the loader rotates it
with the image and then converts it to sin and cos.

The original HDF5 files must remain unchanged.

Example:
    bin/wf python scripts/data/clean.py --splits train val
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

EMPTY_TILE_INDICES = [
    60,
    689,
    753,
    2039,
    2675,
    3734,
    3870,
    3888,
    5146,
    5690,
    6917,
    6941,
    6963,
    7542,
    7591,
    8155,
    8649,
    9206,
    9225,
    9895,
    10503,
    11056,
    11077,
    11375,
    11400,
    11702,
    11973,
    12269,
    12550,
    13070,
    13081,
    13815,
    14038,
    14765,
    14775,
]


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

EXPECTED_COUNTS = {
    "train": 14979,
    "val": 1877,
    "test": 1689,
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

LOWER_PERCENTILE = 0.1
UPPER_PERCENTILE = 99.9

ENVIRONMENTAL_FEATURES = tuple(
    name for name in FEATURE_NAMES if name not in ("FireMask", "PrevFireMask")
)

# Heavily right-skewed, so log1p before clipping and standardizing.
LOG_FEATURES = ("pr", "population")

# Circular: 359 and 1 degrees are neighbours, so no clipping or z-score.
CIRCULAR_FEATURES = ("th",)

# Small negative precipitation is resampling noise and becomes 0. Below this
# (about 0.002% of training pixels) it is treated as missing.
PR_NEGATIVE_LIMIT_MM = -1.0

STATISTICS_FILE = "training_statistics.json"


# --------------------------------------------------
# 2. Paths
# --------------------------------------------------


def get_paths(output_dir=None):
    """Set input and output directories."""
    data_root = Path(os.environ["WF_DATA"])

    input_dir = data_root / "processed" / "ndws"
    if output_dir is None:
        output_dir = data_root / "processed" / "ndws-v1"

    output_dir.mkdir(parents=True, exist_ok=True)

    return input_dir, output_dir


# --------------------------------------------------
# 3. Load data
# --------------------------------------------------


def load_split(path):
    """Read features and original tile identifiers."""
    with h5py.File(path, "r") as f:
        print("Available keys:", list(f.keys()))

        num_tiles = int(f.attrs["num_records"])

        features = {}

        for name in FEATURE_NAMES:
            array = f["features"][name][:]

            # Accept either flattened or spatial format
            if array.shape not in [(num_tiles, 4096), (num_tiles, 64, 64)]:
                raise ValueError(f"{name}: unexpected shape {array.shape}")

            features[name] = array

        tile_ids = f["tile_ids"][:]
        record_indices = f["record_indices"][:]

        assert len(tile_ids) == num_tiles
        assert len(record_indices) == num_tiles

        print(f"Loaded {num_tiles} tiles")

        for name, array in features.items():
            print(f"{name}: {array.shape}")

    return features, tile_ids, record_indices


# --------------------------------------------------
# 4. Identify duplicates
# --------------------------------------------------


def hash_tile(features, index):
    """Calculate SHA-256 across all 13 features."""
    digest = hashlib.sha256()

    for name in FEATURE_NAMES:
        values = np.asarray(
            features[name][index],
            dtype=np.float32,
        )

        values = np.ascontiguousarray(values, dtype="<f4")

        digest.update(values.tobytes())

    return digest.hexdigest()


def find_duplicates(features):
    """Identify duplicate training records."""
    hashes = {}

    keep_indices = []
    duplicate_indices = []

    num_tiles = len(features["FireMask"])

    for index in range(num_tiles):
        tile_hash = hash_tile(features, index)

        if tile_hash in hashes:
            duplicate_indices.append(index)

        else:
            hashes[tile_hash] = index
            keep_indices.append(index)

    print(f"Training tiles: {num_tiles}")
    print(f"Duplicate tiles: {len(duplicate_indices)}")

    return (
        np.array(keep_indices),
        np.array(duplicate_indices),
    )


# --------------------------------------------------
# 5. Identify empty tiles
# --------------------------------------------------


def find_empty_tiles(features):
    """Find tiles whose entire FireMask is unknown."""
    fire_mask = features["FireMask"]

    empty_mask = np.all(fire_mask == -1, axis=tuple(range(1, fire_mask.ndim)))

    empty_indices = np.where(empty_mask)[0]

    print(f"Empty tiles: {len(empty_indices)}")

    return empty_mask, empty_indices


def remove_bad_tiles(features, tile_ids, record_indices):
    """Remove duplicate and entirely unknown training tiles."""
    _, duplicate_indices = find_duplicates(features)

    _, empty_indices = find_empty_tiles(features)

    # Combine indices so overlapping removals are counted once
    remove_indices = np.union1d(duplicate_indices, empty_indices)

    # Get indices of tiles to retain
    keep_indices = np.setdiff1d(np.arange(len(features["FireMask"])), remove_indices)

    # Filter all features consistently
    cleaned_features = {}

    for name in FEATURE_NAMES:
        cleaned_features[name] = features[name][keep_indices]

    # Keep metadata aligned with the features
    cleaned_tile_ids = tile_ids[keep_indices]
    cleaned_record_indices = record_indices[keep_indices]

    print("\nCLEANING SUMMARY")
    print("----------------")
    print(f"Original: {len(tile_ids)}")
    print(f"Duplicates: {len(duplicate_indices)}")
    print(f"Empty: {len(empty_indices)}")
    print(f"Removed: {len(remove_indices)}")
    print(f"Remaining: {len(keep_indices)}")

    return (
        cleaned_features,
        cleaned_tile_ids,
        cleaned_record_indices,
        duplicate_indices,
        empty_indices,
    )


def verify_cleaning(duplicate_indices, empty_indices, keep_indices, num_original):
    """Verify expected duplicate and empty tile counts."""
    print("\nCLEANING VERIFICATION")
    print("---------------------")

    print("Expected duplicates: 424")
    print(f"Found duplicates: {len(duplicate_indices)}")

    print("Expected empty tiles: 35")
    print(f"Found empty tiles: {len(empty_indices)}")

    print(f"Original tiles: {num_original}")
    print(f"Remaining tiles: {len(keep_indices)}")

    # Verify against known expected counts
    assert len(duplicate_indices) == 424
    assert len(empty_indices) == 35

    # Verify all removal indices are unique
    removal_indices = np.union1d(duplicate_indices, empty_indices)

    assert len(keep_indices) + len(removal_indices) == num_original

    print("Cleaning verification passed!")


# --------------------------------------------------
# 6. Missing-value rules
# --------------------------------------------------


def apply_missing_rules(features):
    """Apply feature-specific missing-value rules.

    Environmental missing values become NaN.
    Fire masks retain -1 for unknown labels.
    """
    cleaned = {name: array.astype(np.float32, copy=True) for name, array in features.items()}

    # Convert non-finite environmental values to NaN
    environmental_features = [
        name for name in FEATURE_NAMES if name not in ("FireMask", "PrevFireMask")
    ]

    for name in environmental_features:
        cleaned[name][~np.isfinite(cleaned[name])] = np.nan

    # Detect shared zero-filled weather pixels
    shared_zero_mask = (
        (cleaned["tmmn"] == 0)
        & (cleaned["tmmx"] == 0)
        & (cleaned["vs"] == 0)
        & (cleaned["sph"] == 0)
        & (cleaned["erc"] == 0)
        & (cleaned["th"] == 0)
    )

    weather_features = ["tmmn", "tmmx", "vs", "sph", "erc", "th"]

    for name in weather_features:
        cleaned[name][shared_zero_mask] = np.nan

    print(f"Shared missing pixels: {shared_zero_mask.sum():,}")

    # Feature-specific validity rules
    cleaned["vs"][cleaned["vs"] < 0] = np.nan

    cleaned["sph"][cleaned["sph"] < 0] = np.nan

    cleaned["tmmn"][cleaned["tmmn"] <= 200] = np.nan

    cleaned["tmmx"][cleaned["tmmx"] <= 200] = np.nan

    cleaned["th"][(cleaned["th"] < 0) | (cleaned["th"] > 360)] = np.nan

    cleaned["erc"][cleaned["erc"] < 0] = np.nan

    pr = cleaned["pr"]
    pr[pr < PR_NEGATIVE_LIMIT_MM] = np.nan
    pr[pr < 0] = 0.0

    return cleaned


def apply_log_transforms(features):
    """Apply log1p in place to the skewed features; NaN stays NaN."""
    for name in LOG_FEATURES:
        features[name] = np.log1p(features[name])

    return features


def report_missing_values(features):
    """Report NaN counts after applying missing-value rules."""
    print("\nMISSING VALUE REPORT")
    print("=" * 45)

    for name, array in features.items():
        nan_count = np.isnan(array).sum()
        total = array.size
        percentage = (nan_count / total) * 100

        print(f"{name:12} NaNs: {nan_count:10,} ({percentage:.4f}%)")


def inspect_minus_one_values(features):
    """Count exact -1 values in every feature."""
    print("\nEXACT -1 VALUE REPORT")
    print("=" * 45)

    for name, array in features.items():
        count = np.sum(array == -1)
        percentage = (count / array.size) * 100

        print(f"{name:12} -1 values: {count:12,} ({percentage:.4f}%)")


# --------------------------------------------------
# 7. Generate histograms
# --------------------------------------------------


def plot_histograms(features, output_dir):
    """Generate histograms and inspect candidate clipping thresholds."""
    # Fire masks are excluded because they are categorical
    environmental_features = [
        name for name in FEATURE_NAMES if name not in ("FireMask", "PrevFireMask")
    ]

    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    axes = axes.flatten()

    candidate_thresholds = {}

    for i, name in enumerate(environmental_features):
        values = np.asarray(features[name]).flatten()

        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError(f"No valid values for {name}")

        lower = np.nanpercentile(values, LOWER_PERCENTILE)

        upper = np.nanpercentile(values, UPPER_PERCENTILE)
        candidate_thresholds[name] = {
            "lower": float(lower),
            "upper": float(upper),
        }

        # Plot distribution within percentile range
        axes[i].hist(
            values,
            bins=100,
            range=(lower, upper),
            color="steelblue",
            edgecolor="black",
            alpha=0.8,
        )

        axes[i].set_title(name)
        axes[i].set_xlabel("Feature value")
        axes[i].set_ylabel("Pixel count")

        axes[i].axvline(
            lower,
            color="red",
            linestyle="--",
            label="0.1th percentile",
        )

        axes[i].axvline(
            upper,
            color="orange",
            linestyle="--",
            label="99.9th percentile",
        )

        axes[i].legend(fontsize=7)

        print(f"{name}: lower={lower:.6f}, upper={upper:.6f}")

    # Hide unused subplot
    for i in range(len(environmental_features), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        "NDWS Environmental Feature Distributions",
        fontsize=16,
    )

    fig.tight_layout()

    # Save histogram image
    histogram_path = output_dir / "histograms.png"

    fig.savefig(histogram_path, dpi=150)
    plt.close(fig)

    # Save candidate thresholds
    threshold_path = output_dir / "candidate_thresholds.json"

    threshold_path.write_text(json.dumps(candidate_thresholds, indent=4))

    print(f"\nHistograms saved to: {histogram_path}")
    print(f"Candidate thresholds saved to: {threshold_path}")

    return candidate_thresholds


def calculate_preprocessing_statistics(
    features,
    thresholds,
):
    """Calculate clipping, median and normalization statistics.

    Uses training data only.
    """
    statistics = {}

    for name in ENVIRONMENTAL_FEATURES:
        if name in CIRCULAR_FEATURES:
            continue

        values = features[name].astype(np.float64)

        valid = np.isfinite(values)

        if not np.any(valid):
            raise ValueError(f"No valid training values for {name}")

        observed = values[valid]

        lower = thresholds[name]["lower"]
        upper = thresholds[name]["upper"]

        # Clip observed training values
        clipped = np.clip(observed, lower, upper)

        # Calculate statistics on observed,
        # clipped training values
        median = np.median(clipped)
        mean = np.mean(clipped)
        std = np.std(clipped)

        if std == 0 or not np.isfinite(std):
            raise ValueError(f"Invalid standard deviation for {name}")

        statistics[name] = {
            "transform": "log1p" if name in LOG_FEATURES else "none",
            "lower": float(lower),
            "upper": float(upper),
            "median": float(median),
            "mean": float(mean),
            "std": float(std),
            "valid_count": len(clipped),
        }

    return statistics


def apply_preprocessing(features, statistics):
    """Apply clipping, median imputation and normalization."""
    processed = {name: array.astype(np.float32, copy=True) for name, array in features.items()}

    for name, stats in statistics.items():
        values = processed[name]
        print(f"\nProcessing: {name}")
        print(f"Median: {stats['median']}")
        print(f"Mean: {stats['mean']}")
        print(f"Std: {stats['std']}")
        print(f"NaNs before: {np.isnan(values).sum()}")
        print(f"Infs before: {np.isinf(values).sum()}")

        lower = stats["lower"]
        upper = stats["upper"]
        median = stats["median"]

        # 1. Clip valid observations
        valid = np.isfinite(values)

        values[valid] = np.clip(values[valid], lower, upper)

        # 2. Impute missing values
        values[~valid] = median

        # Debug missing-value imputation
        print(f"{name}: NaNs remaining after imputation: {np.isnan(values).sum()}")

        # 3. Normalize
        processed[name] = (values - stats["mean"]) / stats["std"]
        print(f"NaNs after normalization: {np.isnan(processed[name]).sum()}")

        print(f"Infs after normalization: {np.isinf(processed[name]).sum()}")

    return processed


# --------------------------
# Saving the Data
# --------------------------


def save_cleaned_data(features, tile_ids, record_indices, output_path, split):
    """Save processed features and metadata to an HDF5 file."""
    with h5py.File(output_path, "w") as f:
        feature_group = f.create_group("features")

        for name, array in features.items():
            feature_group.create_dataset(name, data=array, compression="gzip")

        f.create_dataset("tile_ids", data=tile_ids)
        f.create_dataset("record_indices", data=record_indices)

        f.attrs["num_records"] = len(tile_ids)
        f.attrs["tile_shape"] = TILE_SHAPE
        f.attrs["split"] = split
        f.attrs["feature_names"] = list(features.keys())


def check_processed(processed_features):
    """Check that only wind direction may still be missing, and print z-score moments."""
    print("\nNORMALIZATION CHECK")
    print("-------------------")

    for name in ENVIRONMENTAL_FEATURES:
        values = processed_features[name]

        if name in CIRCULAR_FEATURES:
            finite = values[np.isfinite(values)]
            assert finite.min() >= 0, f"{name} below 0 degrees"
            assert finite.max() <= 360, f"{name} above 360 degrees"
            print(f"{name}: degrees, missing={np.isnan(values).sum():,}")
            continue

        assert np.all(np.isfinite(values)), f"Non-finite values remain in {name}"
        print(f"{name}: mean={values.mean():.4f}, std={values.std():.4f}")


# --------------------------
# ------     MAIN     ------
# --------------------------


def clean_train(input_dir, output_dir):
    """Clean the training split, then save it and the statistics it defines."""
    features, tile_ids, record_indices = load_split(input_dir / "train.h5")

    # Inspect original missing-value encodings
    inspect_minus_one_values(features)

    (
        cleaned_features,
        cleaned_tile_ids,
        cleaned_record_indices,
        duplicate_indices,
        empty_indices,
    ) = remove_bad_tiles(
        features,
        tile_ids,
        record_indices,
    )

    keep_indices = np.setdiff1d(
        np.arange(len(features["FireMask"])), np.union1d(duplicate_indices, empty_indices)
    )

    verify_cleaning(duplicate_indices, empty_indices, keep_indices, len(features["FireMask"]))
    del features

    cleaned_features = apply_missing_rules(cleaned_features)
    report_missing_values(cleaned_features)
    cleaned_features = apply_log_transforms(cleaned_features)

    thresholds = plot_histograms(cleaned_features, output_dir)
    statistics = calculate_preprocessing_statistics(cleaned_features, thresholds)

    processed_features = apply_preprocessing(cleaned_features, statistics)
    check_processed(processed_features)

    stats_path = output_dir / STATISTICS_FILE
    stats_path.write_text(json.dumps(statistics, indent=4))

    save_cleaned_data(
        processed_features,
        cleaned_tile_ids,
        cleaned_record_indices,
        output_dir / "train.h5",
        "train",
    )
    print(f"\nStatistics: {stats_path}")

    return statistics


def clean_eval_split(input_dir, output_dir, split, statistics):
    """Clean a validation or test split with the training statistics.

    No tiles are removed, so every tile is scored and the original order is kept.
    """
    features, tile_ids, record_indices = load_split(input_dir / f"{split}.h5")

    cleaned_features = apply_missing_rules(features)
    report_missing_values(cleaned_features)
    cleaned_features = apply_log_transforms(cleaned_features)

    processed_features = apply_preprocessing(cleaned_features, statistics)
    check_processed(processed_features)

    save_cleaned_data(
        processed_features, tile_ids, record_indices, output_dir / f"{split}.h5", split
    )


def main():
    """Clean the requested splits; statistics always come from the training split."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=["train", "val"],
        help="test only for the final evaluation",
    )
    parser.add_argument("--out-dir", type=Path, help="default: $WF_DATA/processed/ndws-v1")
    args = parser.parse_args()

    input_dir, output_dir = get_paths(args.out_dir)

    if "train" in args.splits:
        statistics = clean_train(input_dir, output_dir)
    else:
        statistics = json.loads((output_dir / STATISTICS_FILE).read_text())

    for split in ("val", "test"):
        if split in args.splits:
            clean_eval_split(input_dir, output_dir, split, statistics)

    print("\nCLEANING COMPLETE")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
