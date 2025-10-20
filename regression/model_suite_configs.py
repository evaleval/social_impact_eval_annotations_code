"""Configuration presets for the Eval Cards regression models."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from dataclasses import dataclass, field


CONFIG_PATH_ENV_VAR = "EVAL_CARDS_CONFIG_JSON"


@dataclass(frozen=True)
class ModelConfig:
    """Describe how to build and sample one of the regression models."""

    name: str
    description: str
    build_kwargs: Dict[str, Any] = field(default_factory=dict)
    sample_kwargs: Dict[str, Any] = field(default_factory=dict)


DEFAULT_CONFIG_FILE = Path(__file__).with_name("regression_model_presets.json")


def _coerce_model_config(entry: Dict[str, Any]) -> ModelConfig:
    """Convert a mapping into :class:`ModelConfig` with sensible defaults."""

    try:
        name = entry["name"]
        description = entry.get("description", "")
    except KeyError as exc:  # noqa: B904 - surface missing keys cleanly
        raise ValueError("Model definitions must include at least 'name'.") from exc

    build_kwargs = dict(entry.get("build_kwargs", {}))
    sample_kwargs = dict(entry.get("sample_kwargs", {}))
    return ModelConfig(name=name, description=description, build_kwargs=build_kwargs, sample_kwargs=sample_kwargs)


def _load_json_config(path: str | os.PathLike[str]) -> Tuple[List[ModelConfig], Dict[str, Any]]:
    """Load model configs and defaults from ``path``.

    The file can either be a plain list of model dictionaries or a mapping with
    ``models`` (list) and optional ``defaults`` (mapping) entries.
    """

    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))

    if isinstance(payload, list):
        models_source: Iterable[Dict[str, Any]] = payload
        defaults: Dict[str, Any] = {}
    elif isinstance(payload, dict):
        models_source = payload.get("models", [])
        defaults = dict(payload.get("defaults", {}))
    else:
        raise ValueError("Configuration JSON must be a list or a mapping containing 'models'.")

    models: List[ModelConfig] = []
    for entry in models_source:
        if not isinstance(entry, dict):
            raise ValueError("Each model configuration must be a JSON object.")
        models.append(_coerce_model_config(entry))

    return models, defaults


def _load_embedded_defaults() -> Tuple[List[ModelConfig], Dict[str, Any]]:
    """Load the shipped regression presets from :mod:`regression_model_presets.json`."""

    configs, defaults = _load_json_config(DEFAULT_CONFIG_FILE)
    if not configs:
        raise RuntimeError("Embedded regression_model_presets.json must include model definitions.")
    return configs, defaults


def _resolve_configs_from_source() -> Tuple[List[ModelConfig], Dict[str, Any]]:
    """Return model configs and optional defaults from environment overrides."""

    base_configs, base_defaults = _load_embedded_defaults()

    config_path = os.getenv(CONFIG_PATH_ENV_VAR, "").strip()
    if not config_path:
        return base_configs, base_defaults

    configs, overrides = _load_json_config(config_path)
    if not configs:
        configs = base_configs

    merged_defaults = dict(base_defaults)
    merged_defaults.update(overrides)
    return configs, merged_defaults


REGRESSION_MODEL_CONFIGS, CONFIG_FILE_DEFAULTS = _resolve_configs_from_source()

MODEL_CONFIG_LOOKUP = {cfg.name: cfg for cfg in REGRESSION_MODEL_CONFIGS}

__all__ = [
    "CONFIG_FILE_DEFAULTS",
    "ModelConfig",
    "MODEL_CONFIG_LOOKUP",
    "REGRESSION_MODEL_CONFIGS",
]
