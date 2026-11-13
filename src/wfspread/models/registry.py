"""Build models by name, so a run chooses its model from a config file."""

from __future__ import annotations

import inspect
from pathlib import Path

import yaml
from torch import nn

from wfspread.models.unet import UNet

MODELS: dict[str, type[nn.Module]] = {
    "unet": UNet,
}


def build_model(name: str, **args: object) -> nn.Module:
    """Create a registered model.

    Args:
        name: Key in `MODELS`, for example "unet".
        **args: Constructor arguments for that model.

    Returns:
        A new model mapping inputs (B, C, H, W) to fire logits (B, H, W).

    Raises:
        ValueError: If the name is not registered or an argument is not accepted.
    """
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; registered: {sorted(MODELS)}")
    model_class = MODELS[name]
    accepted = set(inspect.signature(model_class).parameters)
    unknown = set(args) - accepted
    if unknown:
        raise ValueError(f"{name} does not accept {sorted(unknown)}; accepted: {sorted(accepted)}")
    return model_class(**args)


def load_model(path: Path) -> nn.Module:
    """Create the model described by a YAML file with keys `name` and optional `args`.

    Raises:
        ValueError: If the file has other keys, or `build_model` rejects it.
    """
    values = yaml.safe_load(Path(path).read_text()) or {}
    extra = set(values) - {"name", "args"}
    if extra or "name" not in values:
        raise ValueError(f"{path} needs `name` and optional `args`; found {sorted(values)}")
    return build_model(values["name"], **(values.get("args") or {}))
