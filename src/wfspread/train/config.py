"""Training settings, loaded from a YAML file so every run comes from a config."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TrainConfig:
    """Optimizer, schedule, stopping and precision settings for one run.

    Attributes:
        seed: Seed for torch, numpy, Python and data loader workers.
        max_epochs: Upper limit on epochs; early stopping usually ends sooner.
        lr: Peak AdamW learning rate.
        weight_decay: AdamW weight decay.
        warmup_epochs: Epochs of linear warmup before cosine decay.
        min_lr: Learning rate at the end of the cosine decay.
        grad_clip: Gradient norm limit.
        loss_terms: Weight per loss term, from "bce", "focal" and "dice"; the loss is
            their weighted sum.
        pos_weight: Weight on fire pixels in the BCE term.
        focal_gamma: Focusing strength of the focal term.
        focal_alpha: Weight on fire pixels in the focal term.
        patience: Validation epochs without a better PR-AUC before stopping.
        precision: Lightning precision setting, for example "32-true" or "bf16-mixed".
        check_batches: Validate every batch against the batch format; slower, for debugging.
        diagnostics_every_n_steps: Log learning rate, gradient norm and per-layer weight
            change every this many optimizer steps; 0 turns diagnostics off.
    """

    seed: int = 0
    max_epochs: int = 100
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 2
    min_lr: float = 1e-6
    grad_clip: float = 1.0
    loss_terms: dict[str, float] = field(default_factory=lambda: {"bce": 1.0})
    pos_weight: float = 1.0
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    patience: int = 15
    precision: str = "32-true"
    check_batches: bool = False
    diagnostics_every_n_steps: int = 50

    @classmethod
    def from_yaml(cls, path: Path) -> TrainConfig:
        """Load settings from a YAML file; missing keys keep their defaults.

        Raises:
            ValueError: If the file has a key this class does not define.
        """
        values = yaml.safe_load(Path(path).read_text()) or {}
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown training settings in {path}: {sorted(unknown)}")
        return cls(**values)
