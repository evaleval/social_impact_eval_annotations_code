"""Airflow DAG orchestrating the Eval Cards regression model suite."""

from __future__ import annotations

import os
from datetime import datetime

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator

from model_suite_configs import CONFIG_FILE_DEFAULTS, REGRESSION_MODEL_CONFIGS
from workflow_runner import run_model


DEFAULT_ARGS = {
    "owner": "eval_cards",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
}


_CONFIG_CHAIN_POOL = str(CONFIG_FILE_DEFAULTS.get("chain_pool", "pymc_chains"))
CHAIN_POOL_NAME = os.getenv("EVAL_CARDS_CHAIN_POOL", _CONFIG_CHAIN_POOL).strip()


def _task_kwargs(model_name: str, *, chains: int) -> dict:
    """Construct ``PythonOperator`` kwargs for ``model_name``."""

    kwargs = {
        "task_id": f"run_{model_name}",
        "python_callable": run_model,
        "op_kwargs": {"model_name": model_name},
    }

    if CHAIN_POOL_NAME:
        kwargs["pool"] = CHAIN_POOL_NAME
        kwargs["pool_slots"] = chains

    return kwargs


dag = DAG(
    dag_id="eval_cards_regression_models",
    description="Run the Eval Cards regression models (1–6) with PyMC.",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["eval-cards", "pymc", "regression"],
)

dag.doc_md = """
## Eval Cards regression models

This DAG orchestrates the six PyMC regression models used for the Eval Cards
analysis suite, including Model 6 which swaps the spline calendar effect for a
Gaussian process. Tasks reuse the helper utilities in :mod:`workflow_runner`
and respect the ``EVAL_CARDS_*`` environment variables for runtime configuration:

- ``EVAL_CARDS_DATA_DIR``: directory containing ``MAIN_ANNOTATIONS_MERGED.tsv`` (default: ``data``)
- ``EVAL_CARDS_OUTPUT_DIR``: output parent directory for per-model artefacts (default: ``airflow_outputs``)
- ``EVAL_CARDS_BACKEND``: PyMC sampler backend (default: ``numpyro``)
- ``EVAL_CARDS_CORES``: number of worker cores (default: ``8``)
- ``EVAL_CARDS_CHAIN_POOL``: optional pool coordinating parallel runs based on chain counts (default: ``pymc_chains``)
- ``EVAL_CARDS_CONFIG_JSON``: optional JSON file declaring runtime defaults and model overrides
"""

completion = EmptyOperator(task_id="models_complete", dag=dag)

for cfg in REGRESSION_MODEL_CONFIGS:
    chains = int(cfg.sample_kwargs.get("chains", 1))
    task = PythonOperator(dag=dag, **_task_kwargs(cfg.name, chains=chains))
    task >> completion

__all__ = ["dag"]
