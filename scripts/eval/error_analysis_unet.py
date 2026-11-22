"""Error analysis for the final U-Net on the validation split.

Breaks down performance by:
  - Fire size (small / medium / large tiles)
  - Growth vs shrink (fire expands vs contracts next day)
  - New-fire pixels (pixels not burning today that burn tomorrow)
  - Fire edges (pixels adjacent to today's fire boundary)
  - Unknown pixels (fraction of unknowns in the tile)
  - Predicted vs actual fire area

Loads the best checkpoint from the final 3-seed runs (seed 0 by default)
and produces a Markdown report.

Usage:
    bin/wf python scripts/eval/error_analysis_unet.py
    bin/wf python scripts/eval/error_analysis_unet.py --seed 1
    bin/wf python scripts/eval/error_analysis_unet.py --nofire  # analyze no-fire-mask model
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as f_nn
from torch import Tensor
from torch.utils.data import DataLoader

from wfspread.constants import FIRE, IGNORE_INDEX
from wfspread.data.loader import NDWSDataset
from wfspread.eval.metrics import (
    best_f1_threshold,
    pr_auc,
    spread_mask,
    threshold_scores,
    valid_mask,
)
from wfspread.models.unet import UNet
from wfspread.train.config import TrainConfig
from wfspread.train.module import FireModule

RUNS_DIR = Path("/srv/sfsu_srv/wf-spread/runs/final-unet")
DATA_DIR = Path("/srv/sfsu_srv/wf-spread/data/processed/ndws-v1")

BEST_ENC = [48, 96, 192, 384]
BEST_BOTTLENECK = 128
BEST_DROPOUT = 0.4


def load_model(run_dir: Path, nofire: bool = False) -> FireModule:
    """Load the best checkpoint from a run directory."""
    ckpt = sorted(run_dir.glob("checkpoints/best-*.ckpt"))[-1]
    in_ch = 12 if nofire else 13
    model = UNet(
        in_ch=in_ch, enc_channels=BEST_ENC, bottleneck_ch=BEST_BOTTLENECK, dropout=BEST_DROPOUT
    )
    if nofire:

        class DropPrevFire(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.model = m

            def forward(self, x):
                return self.model(x[:, :-1])

        model = DropPrevFire(model)
    config = TrainConfig(seed=0, lr=5.17e-4, loss_terms={"bce": 1.0}, pos_weight=3.0)
    module = FireModule.load_from_checkpoint(ckpt, model=model, config=config, map_location="cpu")
    module.eval()
    return module


@torch.no_grad()
def collect_predictions(
    module: FireModule, loader: DataLoader, device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Run inference and return (probs, next_fire, prev_fire, sample_ids) on CPU."""
    module = module.to(device)
    all_probs, all_next, all_prev, all_ids = [], [], [], []
    for batch in loader:
        logits = module(batch["inputs"].to(device))
        all_probs.append(torch.sigmoid(logits).cpu())
        all_next.append(batch["next_fire"])
        all_prev.append(batch["prev_fire"])
        all_ids.append(batch["sample_id"])
    return torch.cat(all_probs), torch.cat(all_next), torch.cat(all_prev), torch.cat(all_ids)


def tile_fire_pixels(mask: Tensor) -> Tensor:
    """Count fire pixels per tile. mask shape (B, H, W)."""
    return (mask == FIRE).view(mask.shape[0], -1).sum(dim=1)


def tile_valid_pixels(mask: Tensor) -> Tensor:
    """Count valid (non-unknown) pixels per tile."""
    return (mask != IGNORE_INDEX).view(mask.shape[0], -1).sum(dim=1)


def tile_unknown_frac(mask: Tensor) -> Tensor:
    """Fraction of unknown pixels per tile."""
    total = mask.shape[1] * mask.shape[2]
    unknown = (mask == IGNORE_INDEX).view(mask.shape[0], -1).sum(dim=1).float()
    return unknown / total


def fire_edge_mask(prev_fire: Tensor) -> Tensor:
    """Pixels adjacent to (but not inside) today's fire. Shape (B, H, W)."""
    fire = (prev_fire == FIRE).float().unsqueeze(1)  # (B, 1, H, W)
    kernel = torch.ones(1, 1, 3, 3, device=fire.device)
    dilated = f_nn.conv2d(fire, kernel, padding=1).squeeze(1) > 0  # (B, H, W)
    return dilated & (prev_fire != FIRE)


def analyze_by_group(
    probs: Tensor,
    next_fire: Tensor,
    prev_fire: Tensor,
    group_indices: dict[str, Tensor],
    threshold: float,
) -> list[dict]:
    """Compute metrics for each named group of tile indices."""
    rows = []
    for name, idx in group_indices.items():
        if len(idx) == 0:
            rows.append(
                {
                    "group": name,
                    "n_tiles": 0,
                    "pr_auc": "-",
                    "f1": "-",
                    "precision": "-",
                    "recall": "-",
                }
            )
            continue
        p, nf, _pf = probs[idx], next_fire[idx], prev_fire[idx]
        vm = valid_mask(nf)
        has_fire = (nf[vm] == FIRE).any()
        if not has_fire:
            rows.append(
                {
                    "group": name,
                    "n_tiles": len(idx),
                    "pr_auc": "no fire",
                    "f1": "-",
                    "precision": "-",
                    "recall": "-",
                }
            )
            continue
        auc = pr_auc(p, nf)
        scores = threshold_scores(p, nf, threshold)
        rows.append(
            {
                "group": name,
                "n_tiles": len(idx),
                "pr_auc": f"{auc:.4f}",
                "f1": f"{scores.f1:.4f}",
                "precision": f"{scores.precision:.4f}",
                "recall": f"{scores.recall:.4f}",
            }
        )
    return rows


def format_table(rows: list[dict], title: str) -> str:
    """Format rows as a Markdown table."""
    if not rows:
        return f"### {title}\n\nNo data.\n"
    keys = list(rows[0].keys())
    lines = [f"### {title}\n"]
    lines.append("| " + " | ".join(keys) + " |")
    lines.append("|" + "---|" * len(keys))
    for row in rows:
        lines.append("| " + " | ".join(str(row[k]) for k in keys) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Run error analysis and write a Markdown report."""
    parser = argparse.ArgumentParser(description="U-Net error analysis on validation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nofire", action="store_true", help="Analyze the no-fire-mask model")
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    arm = "final-unet-nofire" if args.nofire else "final-unet"
    arm_label = "U-Net (no fire mask)" if args.nofire else "U-Net"

    # Find the run directory
    import csv

    with open(RUNS_DIR / "runs.csv") as f:
        for row in csv.DictReader(f):
            if row["arm"] == arm and int(row["seed"]) == args.seed:
                run_dir = RUNS_DIR / row["run_dir"]
                break
        else:
            raise ValueError(f"No run found for arm={arm}, seed={args.seed}")

    print(f"Loading {arm_label} (seed {args.seed}) from {run_dir.name}")
    module = load_model(run_dir, nofire=args.nofire)

    print("Loading validation data (preload)...")
    val_ds = NDWSDataset(args.data, split="val", augment=False, preload=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    print("Running inference...")
    device = torch.device(args.device)
    probs, next_fire, prev_fire, _sample_ids = collect_predictions(module, val_loader, device)
    n_tiles = probs.shape[0]
    print(f"  {n_tiles} tiles, {probs.shape[1]}x{probs.shape[2]} pixels each")

    # Find threshold on val
    threshold = best_f1_threshold(probs, next_fire)
    print(f"  Best F1 threshold: {threshold:.4f}")

    # Overall metrics
    overall_auc = pr_auc(probs, next_fire)
    overall_scores = threshold_scores(probs, next_fire, threshold)
    spread_auc = pr_auc(probs, next_fire, spread_mask(next_fire, prev_fire))
    spread_scores = threshold_scores(probs, next_fire, threshold, spread_mask(next_fire, prev_fire))

    sections = []
    sections.append(f"# Error Analysis: {arm_label} (seed {args.seed})\n")
    sections.append(
        f"**Overall:** PR-AUC {overall_auc:.4f}, F1 {overall_scores.f1:.4f}, "
        f"Precision {overall_scores.precision:.4f}, Recall {overall_scores.recall:.4f}\n"
    )
    sections.append(
        f"**Spread region only:** PR-AUC {spread_auc:.4f}, F1 {spread_scores.f1:.4f}, "
        f"Precision {spread_scores.precision:.4f}, Recall {spread_scores.recall:.4f}\n"
    )
    sections.append(f"**Threshold:** {threshold:.4f}\n")

    # --- 1. Fire size breakdown ---
    today_fire = tile_fire_pixels(prev_fire)
    tomorrow_fire = tile_fire_pixels(next_fire)
    # Terciles based on tomorrow's fire size
    no_fire_idx = torch.where(tomorrow_fire == 0)[0]
    has_fire = torch.where(tomorrow_fire > 0)[0]
    if len(has_fire) > 0:
        fire_counts = tomorrow_fire[has_fire]
        t33 = torch.quantile(fire_counts.float(), 0.33).item()
        t66 = torch.quantile(fire_counts.float(), 0.66).item()
        small = has_fire[fire_counts <= t33]
        medium = has_fire[(fire_counts > t33) & (fire_counts <= t66)]
        large = has_fire[fire_counts > t66]
        groups = {
            f"no fire (n={len(no_fire_idx)})": no_fire_idx,
            f"small (≤{int(t33)} px)": small,
            f"medium ({int(t33) + 1}–{int(t66)} px)": medium,
            f"large (>{int(t66)} px)": large,
        }
    else:
        groups = {"no fire": no_fire_idx}
    sections.append(
        format_table(
            analyze_by_group(probs, next_fire, prev_fire, groups, threshold),
            "By fire size (tomorrow's fire pixels)",
        )
    )

    # --- 2. Growth vs shrink ---
    diff = tomorrow_fire.float() - today_fire.float()
    has_fire_today = today_fire > 0
    growing = torch.where(has_fire_today & (diff > 0))[0]
    stable = torch.where(has_fire_today & (diff == 0))[0]
    shrinking = torch.where(has_fire_today & (diff < 0))[0]
    new_ignition = torch.where((~has_fire_today) & (tomorrow_fire > 0))[0]
    no_fire_both = torch.where((~has_fire_today) & (tomorrow_fire == 0))[0]
    groups = {
        "growing": growing,
        "stable": stable,
        "shrinking": shrinking,
        "new ignition": new_ignition,
        "no fire both days": no_fire_both,
    }
    sections.append(
        format_table(
            analyze_by_group(probs, next_fire, prev_fire, groups, threshold),
            "By fire dynamics (growth vs shrink)",
        )
    )

    # --- 3. New-fire (spread region) metrics by tile ---
    sm = spread_mask(next_fire, prev_fire)
    new_fire_per_tile = ((next_fire == FIRE) & sm).view(n_tiles, -1).sum(dim=1)
    has_new_fire = torch.where(new_fire_per_tile > 0)[0]
    no_new_fire = torch.where(new_fire_per_tile == 0)[0]
    if len(has_new_fire) > 0:
        nf_counts = new_fire_per_tile[has_new_fire]
        med = torch.median(nf_counts).item()
        few_new = has_new_fire[nf_counts <= med]
        many_new = has_new_fire[nf_counts > med]
        groups = {
            "no new fire": no_new_fire,
            f"few new fire (≤{int(med)} px)": few_new,
            f"many new fire (>{int(med)} px)": many_new,
        }
    else:
        groups = {"no new fire": no_new_fire}
    sections.append(
        format_table(
            analyze_by_group(probs, next_fire, prev_fire, groups, threshold),
            "By new-fire pixels (spread region)",
        )
    )

    # --- 4. Unknown pixel fraction ---
    unk_frac = tile_unknown_frac(next_fire)
    low_unk = torch.where(unk_frac < 0.01)[0]
    mid_unk = torch.where((unk_frac >= 0.01) & (unk_frac < 0.10))[0]
    high_unk = torch.where(unk_frac >= 0.10)[0]
    groups = {
        "< 1% unknown": low_unk,
        "1–10% unknown": mid_unk,
        "≥ 10% unknown": high_unk,
    }
    sections.append(
        format_table(
            analyze_by_group(probs, next_fire, prev_fire, groups, threshold),
            "By unknown pixel fraction",
        )
    )

    # --- 5. Fire edge analysis ---
    edge = fire_edge_mask(prev_fire)
    edge_fire = ((next_fire == FIRE) & edge).view(n_tiles, -1).sum(dim=1)
    edge_total = edge.view(n_tiles, -1).sum(dim=1)
    has_edge = torch.where(edge_total > 0)[0]
    if len(has_edge) > 0:
        edge_fire_rate = edge_fire[has_edge].float() / edge_total[has_edge].float()
        med_rate = torch.median(edge_fire_rate).item()
        low_edge = has_edge[edge_fire_rate <= med_rate]
        high_edge = has_edge[edge_fire_rate > med_rate]
        groups = {
            "no fire edge": torch.where(edge_total == 0)[0],
            f"low edge fire (≤{med_rate:.1%})": low_edge,
            f"high edge fire (>{med_rate:.1%})": high_edge,
        }
    else:
        groups = {"no fire edges": torch.arange(n_tiles)}
    sections.append(
        format_table(
            analyze_by_group(probs, next_fire, prev_fire, groups, threshold),
            "By fire edge activity",
        )
    )

    # --- 6. Predicted vs actual fire area ---
    predicted_fire = (probs >= threshold).view(n_tiles, -1).sum(dim=1).float()
    actual_fire = tomorrow_fire.float()
    has_any = torch.where((predicted_fire > 0) | (actual_fire > 0))[0]
    if len(has_any) > 0:
        ratio = predicted_fire[has_any] / actual_fire[has_any].clamp(min=1)
        under = has_any[ratio < 0.5]
        about_right = has_any[(ratio >= 0.5) & (ratio <= 2.0)]
        over = has_any[ratio > 2.0]
        no_pred_no_actual = torch.where((predicted_fire == 0) & (actual_fire == 0))[0]
        groups = {
            "underpredicts (< 0.5×)": under,
            "about right (0.5–2×)": about_right,
            "overpredicts (> 2×)": over,
            "no fire predicted or actual": no_pred_no_actual,
        }
    else:
        groups = {"no fire": torch.arange(n_tiles)}
    sections.append(
        format_table(
            analyze_by_group(probs, next_fire, prev_fire, groups, threshold),
            "Predicted vs actual fire area",
        )
    )

    # --- 7. Summary statistics ---
    sections.append("### Summary statistics\n")
    sections.append(f"- Total tiles: {n_tiles}")
    sections.append(f"- Tiles with fire today: {int((today_fire > 0).sum())}")
    sections.append(f"- Tiles with fire tomorrow: {int((tomorrow_fire > 0).sum())}")
    sections.append(f"- Tiles with new fire (spread): {int((new_fire_per_tile > 0).sum())}")
    sections.append(f"- Mean fire pixels today: {today_fire.float().mean():.1f}")
    sections.append(f"- Mean fire pixels tomorrow: {tomorrow_fire.float().mean():.1f}")
    sections.append(f"- Mean unknown fraction: {unk_frac.mean():.3f}")
    sections.append("")

    # Write report
    report = "\n".join(sections)
    out_path = RUNS_DIR / f"error_analysis_{arm}_s{args.seed}.md"
    out_path.write_text(report)
    print(f"\nReport saved to {out_path}")
    print("\n" + report)


if __name__ == "__main__":
    main()
