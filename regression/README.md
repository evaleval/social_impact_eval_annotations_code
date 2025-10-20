# eval_cards_regression

Hierarchical ordinal regression pipelines for analysing the "Who Measures What Matters?"
annotation corpus. The project extracts design matrices from the merged TSV data,
constructs a multi-level ordinal regression in PyMC, and reproduces the publication
figures and tables without relying on the original exploratory workflow.

## Environment setup

### Poetry workflow
1. Install [Poetry](https://python-poetry.org/) (version 2.0 or newer is required).
2. From the repository root, install the Python dependencies:
   ```bash
   poetry install
   ```
   The default dependency set now bundles Apache Airflow (for the regression-suite
   DAG), so ensure you are using Python 3.12 when creating your environment.
3. Activate the environment for commands:
   ```bash
   poetry run python analyse_models.py --help
   ```

### Docker workflow
Build and run the container for an isolated environment:
```bash
docker build -t eval-cards-regression .
docker run --rm -v $(pwd):/app eval-cards-regression \
  poetry run python analyse_models.py --data-dir data --output-dir outputs
```
The container image installs Poetry 2, resolves all dependencies, and exposes the
command-line interface as the default entry point.

## Running the analysis

The CLI expects the merged annotation TSV (`MAIN_ANNOTATIONS_MERGED.tsv`) inside the
`--data-dir` directory. A typical invocation is:
```bash
poetry run python analyse_models.py \
  --data-dir data \
  --output-dir outputs \
  --backend numpyro \
  --year-mode gp \
  --chains 8 --cores 8 \
  --draws 3000 --tune 3000
```

Key options:
- `--backend`: choose among `numpyro` (default, requires JAX support), `nutpie`, or
  `pymc`.
- `--adapt-delta` and `--max-treedepth`: fine-tune the NUTS sampler across backends.
- `--year-mode`: pick `spline`, `gp`, or `linear` treatments of calendar effects.
- `--year-std`, `--year-u`, `--year-idx`: override the year covariates if custom
  preprocessing is required.
- Hierarchical prior controls such as `--lkj-eta`, `--sd-reg-int`, and
  `--sd-year` mirror the published defaults but are fully configurable from the CLI.

All figures are written to `<output-dir>/fig` and tabular summaries to
`<output-dir>/table`. When sampling with the NumPyro backend the script automatically
matches the host device count to the requested parallelism so that multi-chain runs
fully utilise available CPU cores.

## Airflow orchestration

The repository also ships with an Apache Airflow DAG that runs the six regression
models (including the Gaussian-process specification for Model 6).

- DAG module: `dags/regression_models_dag.py`
- Helper for standalone execution: `workflow_runner.run_model`

Each task honours the following environment variables, which makes it easy to
configure deployments without editing the DAG:

- `EVAL_CARDS_DATA_DIR`: location of `MAIN_ANNOTATIONS_MERGED.tsv` (defaults to
  `data`)
- `EVAL_CARDS_OUTPUT_DIR`: parent folder where per-model artefacts (posterior
  samples, summaries, and config JSON) are written (defaults to `airflow_outputs`)
- `EVAL_CARDS_BACKEND`: PyMC sampler backend (`nutpie`, `pymc`, or `numpyro`; defaults
  to `numpyro`)
- `EVAL_CARDS_CORES`: number of worker cores made available to PyMC (defaults to `8`)
- `EVAL_CARDS_CHAIN_POOL`: optional Airflow pool name used to gate parallel model
  runs based on their chain counts (defaults to `pymc_chains`)
- `EVAL_CARDS_CONFIG_JSON`: optional path to a JSON file that declares runtime
  defaults and overrides the shipped model configurations. The repository ships
  with `regression_model_presets.json`, which contains the canonical
  definitions for all six regression models and their default runtime settings.

Create an Airflow pool whose slot count matches the number of sampling cores you plan
to make available (for example, 16 slots for a 16-core worker) and set
`EVAL_CARDS_CHAIN_POOL` to that pool name. Each task will claim a number of slots equal
to its configured chain count, which lets independent models execute in parallel
without oversubscribing the sampler backend. If the variable is unset or an empty
string, the DAG falls back to Airflow's default pool.

When `EVAL_CARDS_CONFIG_JSON` is provided, its `defaults` object can define any of the
variables above (for example `{"data_dir": "s3://bucket", "cores": 16}`). The same
file may also express the model suite directly in JSON. Copy
`regression_model_presets.json` as a starting point or author a minimal template
like:

```json
{
  "defaults": {
    "data_dir": "data",
    "output_root": "airflow_outputs",
    "backend": "numpyro",
    "cores": 8,
    "chain_pool": "pymc_chains"
  },
  "models": [
    {
      "name": "model_gp_year",
      "description": "Gaussian-process year effect with shared hyperparameters.",
      "build_kwargs": {
        "year_mode": "gp",
        "sd_year": 0.25,
        "gp_jitter": 1e-6,
        "gp_shared_hypers": true
      },
      "sample_kwargs": {
        "draws": 3000,
        "tune": 3000,
        "chains": 8,
        "target_accept": 0.99,
        "max_treedepth": 12
      }
    }
  ]
}
```

Any keys you omit inside `build_kwargs` or `sample_kwargs` default to empty
objects, so only the parameters that diverge from the baked-in definitions are
required.

Add additional model objects to the `models` array to cover the rest of the
regression suite. If you only need to tweak the runtime defaults, omit the
`models` array entirely and the built-in presets remain in effect.

Load the DAG into an Airflow deployment, set the environment variables if the
defaults are unsuitable, and trigger the `eval_cards_regression_models` DAG. Downstream
analyses can rely on consistent artefact locations under the configured output
directory, irrespective of whether the tasks run sequentially or in parallel.

## Outputs

The workflow saves forest plots, posterior summaries, year-smooth visualisations
for pooled and per-outcome effects (supporting both spline and Gaussian-process
representations), and variance-partitioning tables/plots. Pooled year-smooth
plots now spell out the pooling mode in their titles (for example, `pooled (mean)`
or `pooled (weighted)`, with `pooled (custom)` used for user-supplied weights).
Use the generated CSV and PDF files for downstream reporting or manuscript
integration. Set `--save-idata` if you wish to persist the full `InferenceData`
object for further exploration in ArviZ.
