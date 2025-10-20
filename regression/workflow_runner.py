"""Utilities to run the Eval Cards regression models."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import arviz as az
import pymc as pm

from analyse_models import build_analysis_model
from data_processing import load_annotations, prepare_data
from model_suite_configs import CONFIG_FILE_DEFAULTS, MODEL_CONFIG_LOOKUP, ModelConfig


def _resolve_environment_defaults() -> Dict[str, Any]:
    """Collect default runtime options from environment variables."""

    defaults: Dict[str, Any] = dict(CONFIG_FILE_DEFAULTS)

    env_overrides = {
        "data_dir": os.getenv("EVAL_CARDS_DATA_DIR"),
        "output_root": os.getenv("EVAL_CARDS_OUTPUT_DIR"),
        "backend": os.getenv("EVAL_CARDS_BACKEND"),
        "cores": os.getenv("EVAL_CARDS_CORES"),
    }

    for key, value in env_overrides.items():
        if value is not None:
            defaults[key] = value

    data_dir = defaults.get("data_dir", "data")
    output_root = Path(defaults.get("output_root", "airflow_outputs"))
    backend = str(defaults.get("backend", "numpyro")).lower()
    
    # Add error handling for cores environment variable casting
    cores_value = defaults.get("cores", 8)
    try:
        cores = int(cores_value)
        if cores <= 0:
            raise ValueError(f"cores must be positive, got {cores}")
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Invalid value for cores: {cores_value!r}. Must be a positive integer."
        ) from e

    return {
        "data_dir": data_dir,
        "output_root": output_root,
        "backend": backend,
        "cores": cores,
    }


def _load_config(model_name: str) -> ModelConfig:
    try:
        return MODEL_CONFIG_LOOKUP[model_name]
    except KeyError as exc:  # noqa: B904 - surface invalid names directly
        available = ", ".join(sorted(MODEL_CONFIG_LOOKUP))
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}") from exc


def run_model(
    model_name: str,
    *,
    data_dir: Optional[str] = None,
    output_root: Optional[str | Path] = None,
    backend: Optional[str] = None,
    cores: Optional[int] = None,
    sample_overrides: Optional[Dict[str, Any]] = None,
) -> str:
    """Execute one of the regression models and persist basic artefacts.

    Parameters
    ----------
    model_name:
        Identifier from :mod:`model_suite_configs`.
    data_dir:
        Directory containing ``MAIN_ANNOTATIONS_MERGED.tsv``. Defaults to
        ``EVAL_CARDS_DATA_DIR`` or the ``defaults.data_dir`` entry inside
        ``EVAL_CARDS_CONFIG_JSON`` (falling back to ``data``).
    output_root:
        Directory where per-model outputs are written. Defaults to the
        ``EVAL_CARDS_OUTPUT_DIR`` environment variable, then
        ``defaults.output_root`` from ``EVAL_CARDS_CONFIG_JSON``, or
        ``airflow_outputs``.
    backend:
        PyMC sampler backend (``pymc``, ``nutpie``, or ``numpyro``). Defaults to the
        ``EVAL_CARDS_BACKEND`` environment variable, then
        ``defaults.backend`` from ``EVAL_CARDS_CONFIG_JSON`` (falling back to
        ``numpyro``).
    cores:
        Number of worker cores used for sampling. Defaults to
        ``EVAL_CARDS_CORES`` or the ``defaults.cores`` entry in the JSON config, or
        ``8`` when neither override is set.
    sample_overrides:
        Optional dictionary merged into the default sampling configuration.

    Returns
    -------
    str
        Path to the directory containing outputs for ``model_name``.
    """

    defaults = _resolve_environment_defaults()
    config = _load_config(model_name)

    resolved_backend = (backend or defaults["backend"]).lower()
    resolved_data_dir = data_dir or defaults["data_dir"]
    resolved_output_root = Path(output_root or defaults["output_root"])
    resolved_cores = int(cores or defaults["cores"])

    sample_kwargs: Dict[str, Any] = dict(config.sample_kwargs)
    if sample_overrides:
        sample_kwargs.update(sample_overrides)

    target_accept = float(sample_kwargs.pop("target_accept", 0.99))
    max_treedepth = int(sample_kwargs.pop("max_treedepth", 12))
    sample_kwargs.setdefault("draws", 3000)
    sample_kwargs.setdefault("tune", 3000)
    sample_kwargs.setdefault("chains", 4)
    sample_kwargs.setdefault("random_seed", 2025)
    sample_kwargs["chains"] = int(sample_kwargs["chains"])

    df = load_annotations(resolved_data_dir)
    prep = prepare_data(df)

    build_output = build_analysis_model(prep, **config.build_kwargs)

    if isinstance(build_output, tuple):
        model, _ = build_output
    else:
        model = build_output

    chain_method = "parallel" if resolved_cores and resolved_cores > 1 else "sequential"

    with model:
        if resolved_backend == "numpyro":
            if not hasattr(pm, "sampling_jax"):
                raise RuntimeError(
                    "The numpyro backend requires installing PyMC with JAX support."
                )
            import numpyro

            device_count = max(resolved_cores, sample_kwargs["chains"])
            numpyro.set_host_device_count(int(device_count))
            idata = pm.sampling_jax.sample_numpyro_nuts(
                draws=int(sample_kwargs["draws"]),
                tune=int(sample_kwargs["tune"]),
                chains=sample_kwargs["chains"],
                target_accept=target_accept,
                random_seed=int(sample_kwargs["random_seed"]),
                nuts_kwargs={"max_tree_depth": max_treedepth},
                chain_method=chain_method,
            )
        else:
            pm_kwargs = dict(sample_kwargs)
            pm_kwargs.update(
                dict(
                    target_accept=target_accept,
                    cores=resolved_cores,
                )
            )
            mp_ctx = "spawn" if resolved_backend == "nutpie" else None
            if resolved_backend == "pymc":
                step = pm.NUTS(target_accept=target_accept, max_treedepth=max_treedepth)
                idata = pm.sample(step=step, mp_ctx=mp_ctx, **pm_kwargs)
            elif resolved_backend == "nutpie":
                # nutpie may or may not support max_treedepth; if not, omit it
                idata = pm.sample(nuts_sampler=resolved_backend, mp_ctx=mp_ctx, **pm_kwargs)
            else:
                # fallback for other backends
                idata = pm.sample(nuts_sampler=resolved_backend, mp_ctx=mp_ctx, **pm_kwargs)

        idata.extend(pm.sample_posterior_predictive(idata, var_names=["y"]))
        idata.extend(pm.compute_log_likelihood(idata))

    model_dir = resolved_output_root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    az.to_netcdf(idata, model_dir / "posterior.nc")

    summary = az.summary(
        idata,
        var_names=None,
        kind="all",
        stat_focus="median",
        hdi_prob=0.95,
        round_to=4,
        skipna=True,
    ).reset_index(names=["parameter"])
    summary.to_csv(model_dir / "summary.csv", index=False)

    record = {
        "model": model_name,
        "description": config.description,
        "backend": resolved_backend,
        "build_kwargs": config.build_kwargs,
        "sample_kwargs": dict(sample_kwargs, target_accept=target_accept, max_treedepth=max_treedepth),
    }
    with open(model_dir / "config.json", "w", encoding="utf-8") as fp:
        json.dump(record, fp, indent=2, sort_keys=True)

    return str(model_dir)


__all__ = ["run_model"]
