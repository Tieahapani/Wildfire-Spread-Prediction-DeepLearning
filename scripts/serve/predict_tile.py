"""Draw fire probability maps for NDWS tiles with a trained checkpoint.

For each tile it saves the probabilities as `.npy` and a figure with today's fire,
the predicted probability for tomorrow, and tomorrow's actual fire.

Example:
    bin/wf python scripts/serve/predict_tile.py \
        --checkpoint "$WF_RUNS/<run>/checkpoints/last.ckpt" \
        --model configs/model/unet.yaml --split val --index 0 7 42 --out-dir maps
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

from wfspread.constants import FIRE, IGNORE_INDEX
from wfspread.serve.predict import INPUT_FEATURES, FirePredictor


def read_raw_tile(path: Path, index: int) -> tuple[dict[str, np.ndarray], str]:
    """Raw features of one tile from a converted split file, and its tile ID."""
    with h5py.File(path, "r") as f:
        names = (*INPUT_FEATURES, "FireMask")
        raw = {name: f["features"][name][index].reshape(64, 64) for name in names}
        tile_id = f["tile_ids"][index]
    return raw, tile_id.decode() if isinstance(tile_id, bytes) else str(tile_id)


def draw(raw: dict[str, np.ndarray], probs: np.ndarray, title: str, path: Path) -> None:
    """Save today's fire, the predicted probability and tomorrow's fire side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    masks = {"Fire today": raw["PrevFireMask"], "Fire tomorrow (actual)": raw["FireMask"]}
    cmap = ListedColormap(["#bdbdbd", "#ffffff", "#d7301f"])
    for ax, (name, mask) in zip(axes[[0, 2]], masks.items(), strict=True):
        ax.imshow(mask, cmap=cmap, vmin=IGNORE_INDEX, vmax=FIRE, interpolation="nearest")
        ax.set_title(f"{name}\n(grey = unknown)")
    shown = axes[1].imshow(probs, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title("Predicted fire probability, tomorrow")
    fig.colorbar(shown, ax=axes[1], fraction=0.046)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Predict and draw each requested tile."""
    data = Path(os.environ.get("WF_DATA", "data")) / "processed"
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="model config of the run")
    parser.add_argument("--stats", type=Path, default=data / "ndws-v1" / "training_statistics.json")
    # The test split is left out until the one-time final evaluation.
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--index", type=int, nargs="+", required=True, help="tile positions")
    parser.add_argument("--out-dir", type=Path, default=Path("maps"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    predictor = FirePredictor.from_files(args.checkpoint, args.model, args.stats, args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for index in args.index:
        raw, tile_id = read_raw_tile(data / "ndws" / f"{args.split}.h5", index)
        start_s = time.perf_counter()
        probs = predictor.predict(raw)[0]
        elapsed_ms = (time.perf_counter() - start_s) * 1e3

        stem = f"{args.split}-{index:05d}"
        np.save(args.out_dir / f"{stem}.npy", probs)
        draw(raw, probs, f"{args.split} tile {index} ({tile_id})", args.out_dir / f"{stem}.png")
        print(
            f"{stem}: {tile_id}, max probability {probs.max():.3f}, "
            f"{elapsed_ms:.1f} ms including cleaning -> {args.out_dir / stem}.png"
        )


if __name__ == "__main__":
    main()
