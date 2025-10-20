"""Command-line entry point for the hierarchical regression analysis."""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

logger = logging.getLogger(__name__)

from data_processing import load_annotations, prepare_data
from model_builder import build_model
from diagnostics import generate_diagnostic_plots, generate_pair_plots
from plotting import (
    compute_icc_vpc,
    plot_beta_forest,
    plot_icc_components,
    save_year_conditional_smooth_by_outcome,
    save_year_conditional_smooth_pooled,
    summarize_beta,
    write_beta_latex_table,
)


def _parse_sequence_arg(value: str, dtype: type) -> np.ndarray:
    """Parse a CLI argument representing a numeric sequence."""

    try:
        if os.path.exists(value):
            loaded = np.load(value, allow_pickle=False)
            if isinstance(loaded, np.lib.npyio.NpzFile):
                first_key = next(iter(loaded.files))
                data = np.asarray(loaded[first_key], dtype=dtype)
            else:
                data = np.asarray(loaded, dtype=dtype)
        else:
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                data = np.fromstring(value, sep=",", dtype=dtype)
            else:
                if isinstance(parsed, (list, tuple)):
                    data = np.asarray(parsed, dtype=dtype)
                else:
                    data = np.fromstring(value, sep=",", dtype=dtype)
    except Exception as exc:  # noqa: BLE001 - surface parsing issues directly
        raise argparse.ArgumentTypeError(f"Unable to parse sequence argument: {exc}") from exc

    if data.size == 0:
        raise argparse.ArgumentTypeError("Parsed sequence argument is empty.")

    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hierarchical ordinal regression analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default="data", help="Directory with TSV annotations.")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Output directory for figures, tables, and results.",
    )
    parser.add_argument("--draws", type=int, default=3000, help="MCMC draws per chain.")
    parser.add_argument("--tune", type=int, default=3000, help="Tuning steps per chain.")
    parser.add_argument("--chains", type=int, default=8, help="Number of MCMC chains.")
    parser.add_argument("--cores", type=int, default=8, help="CPU cores for sampling.")
    parser.add_argument(
        "--backend",
        choices=("numpyro", "nutpie", "pymc"),
        default="numpyro",
        help="PyMC sampling backend.",
    )
    parser.add_argument("--seed", type=int, default=2025, help="Random seed.")
    parser.add_argument(
        "--adapt-delta",
        type=float,
        default=0.99,
        help="Target acceptance rate for NUTS.",
    )
    parser.add_argument(
        "--max-treedepth",
        type=int,
        default=12,
        help="Maximum tree depth for NUTS.",
    )
    parser.add_argument(
        "--font-family",
        default="Times New Roman",
        help="Matplotlib font family.",
    )
    parser.add_argument(
        "--save-idata",
        default=None,
        help="Path to save InferenceData as NetCDF.",
    )
    for level in ("region", "country", "provider", "firstparty"):
        parser.add_argument(
            f"--intercept-corr-{level}",
            action=argparse.BooleanOptionalAction,
            default=False,
            help=f"Enable LKJ prior for {level} intercepts.",
        )
        parser.add_argument(
            f"--slopes-{level}",
            action=argparse.BooleanOptionalAction,
            default=False,
            help=f"Include varying slopes for {level}.",
        )
        parser.add_argument(
            f"--slope-corr-{level}",
            action=argparse.BooleanOptionalAction,
            default=False,
            help=f"Enable correlation for {level} slopes.",
        )
    parser.add_argument("--lkj-eta", type=float, default=6.0, help="LKJ prior shape.")
    parser.add_argument("--sd-reg-int", type=float, default=0.4, help="Region intercept prior scale.")
    parser.add_argument("--sd-cty-dev", type=float, default=0.4, help="Country deviation prior scale.")
    parser.add_argument("--sd-prov-dev", type=float, default=0.4, help="Provider deviation prior scale.")
    parser.add_argument("--sd-fp-int", type=float, default=0.4, help="First-party intercept prior scale.")
    parser.add_argument("--sd-sigma", type=float, default=1.0, help="Observation noise prior scale.")
    parser.add_argument("--cutpoint-prior", default="dirichlet", help="Cutpoint prior family.")
    parser.add_argument(
        "--dirichlet-center",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Center Dirichlet cutpoint prior.",
    )
    parser.add_argument(
        "--separate-slope-sd",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use separate slope standard deviations.",
    )
    parser.add_argument(
        "--pop-slope-corr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Model population-level slope correlations.",
    )
    parser.add_argument("--year-mode", default="spline", help="Year effect type.")
    parser.add_argument("--sd-year", type=float, default=0.25, help="Year effect prior scale.")
    parser.add_argument("--spline-df", type=int, default=12, help="Spline basis degrees of freedom.")
    parser.add_argument("--spline-degree", type=int, default=5, help="Spline polynomial degree.")
    parser.add_argument(
        "--return-components",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Return auxiliary model components.",
    )
    parser.add_argument(
        "--year-std",
        type=str,
        default=None,
        help="Override for standardized year covariate (JSON, CSV, or .npy/.npz file).",
    )
    parser.add_argument(
        "--year-u",
        type=str,
        default=None,
        help="Override for unique standardized years (JSON, CSV, or .npy/.npz file).",
    )
    parser.add_argument(
        "--year-idx",
        type=str,
        default=None,
        help="Override for year indices (JSON, CSV, or .npy/.npz file).",
    )
    return parser.parse_args()


def build_analysis_model(prep: Dict[str, Any], **config: Any):
    """Instantiate the configurable hierarchical ordinal regression model."""

    intercept_corr = config.get(
        "intercept_corr",
        {"region": False, "country": False, "provider": False, "firstparty": False},
    )
    slopes = config.get(
        "slopes", {"region": False, "country": False, "provider": False, "firstparty": False}
    )
    slope_corr = config.get(
        "slope_corr",
        {"region": False, "country": False, "provider": False, "firstparty": False},
    )

    return_components = config.get("return_components", True)

    model = build_model(
        coords=prep["coords"],
        out_idx=prep["out_idx"],
        reg_idx=prep["reg_idx"],
        cty_idx=prep["cty_idx"],
        prov_idx=prep["prov_idx"],
        fp_idx=prep["fp_idx"],
        W=prep["W"],
        y_obs=prep["y_obs"],
        K=prep["K"],
        link="logit",
        firstparty_role="predictor",
        C2R=prep["C2R"],
        P2C=prep["P2C"],
        reg_country_counts=prep["reg_country_counts"],
        cty_prov_counts=prep["cty_prov_counts"],
        intercept_corr=intercept_corr,
        slopes=slopes,
        slope_corr=slope_corr,
        lkj_eta=config.get("lkj_eta", 6.0),
        sd_reg_int=config.get("sd_reg_int", 0.4),
        sd_cty_dev=config.get("sd_cty_dev", 0.4),
        sd_prov_dev=config.get("sd_prov_dev", 0.4),
        sd_fp_int=config.get("sd_fp_int", 0.4),
        sd_sigma=config.get("sd_sigma", 1.0),
        cutpoint_prior=config.get("cutpoint_prior", "dirichlet"),
        dirichlet_center=config.get("dirichlet_center", True),
        separate_slope_sd=config.get("separate_slope_sd", False),
        pop_slope_corr=config.get("pop_slope_corr", True),
        year_std=config.get("year_std", prep["year_std"]),
        year_u=config.get("year_u", prep["year_u"]),
        year_idx=config.get("year_idx", prep["year_idx"]),
        year_mode=config.get("year_mode", "spline"),
        sd_year=config.get("sd_year", 0.25),
        spline_df=config.get("spline_df", 12),
        spline_degree=config.get("spline_degree", 5),
        return_components=return_components,
    )

    if return_components:
        pymc_model = model.get("model")
        if pymc_model is None:
            raise KeyError("Model construction did not provide a 'model' entry.")
        components = {key: value for key, value in model.items() if key != "model"}
        return pymc_model, components

    return model


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    plt.rcParams["font.family"] = args.font_family

    df = load_annotations(args.data_dir)
    prep = prepare_data(df)

    intercept_corr = {
        level: getattr(args, f"intercept_corr_{level}")
        for level in ("region", "country", "provider", "firstparty")
    }
    slopes = {level: getattr(args, f"slopes_{level}") for level in intercept_corr}
    slope_corr = {level: getattr(args, f"slope_corr_{level}") for level in intercept_corr}

    config: Dict[str, Any] = dict(
        intercept_corr=intercept_corr,
        slopes=slopes,
        slope_corr=slope_corr,
        lkj_eta=args.lkj_eta,
        sd_reg_int=args.sd_reg_int,
        sd_cty_dev=args.sd_cty_dev,
        sd_prov_dev=args.sd_prov_dev,
        sd_fp_int=args.sd_fp_int,
        sd_sigma=args.sd_sigma,
        cutpoint_prior=args.cutpoint_prior,
        dirichlet_center=args.dirichlet_center,
        separate_slope_sd=args.separate_slope_sd,
        pop_slope_corr=args.pop_slope_corr,
        year_mode=args.year_mode,
        sd_year=args.sd_year,
        spline_df=args.spline_df,
        spline_degree=args.spline_degree,
        return_components=args.return_components,
    )

    if args.year_std is not None:
        config["year_std"] = _parse_sequence_arg(args.year_std, float)
    if args.year_u is not None:
        config["year_u"] = _parse_sequence_arg(args.year_u, float)
    if args.year_idx is not None:
        config["year_idx"] = _parse_sequence_arg(args.year_idx, int).astype("int64")

    model, _ = build_analysis_model(prep, **config)
    backend = args.backend.lower()
    if backend == "numpyro" and not hasattr(pm, "sampling_jax"):
        raise RuntimeError(
            "The numpyro backend requires installing PyMC with JAX support. "
            "Install via 'pip install pymc[jax]' or add the 'jax' extra when using Poetry."
        )
    chain_method = "parallel" if args.cores and args.cores > 1 else "sequential"
    sample_kwargs = dict(
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        random_seed=args.seed,
        target_accept=args.adapt_delta,
    )

    with model:
        if backend == "numpyro":
            import numpyro

            device_count = args.cores or args.chains or 1
            numpyro.set_host_device_count(int(device_count))

            idata = pm.sampling_jax.sample_numpyro_nuts(
                **sample_kwargs,
                nuts_kwargs={"max_tree_depth": args.max_treedepth},
                chain_method=chain_method,
            )
        else:
            pm_kwargs = dict(sample_kwargs)
            pm_kwargs.update(dict(cores=args.cores, max_treedepth=args.max_treedepth))
            mp_ctx = "spawn" if backend == "nutpie" else None
            idata = pm.sample(nuts_sampler=backend, mp_ctx=mp_ctx, **pm_kwargs)
        idata.extend(pm.sample_posterior_predictive(idata, var_names=["y"]))
        idata.extend(pm.compute_log_likelihood(idata))

    output_dir = args.output_dir
    fig_dir = os.path.join(output_dir, "fig")
    table_dir = os.path.join(output_dir, "table")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    if args.save_idata:
        az.to_netcdf(idata, args.save_idata)

    summary = az.summary(
        idata,
        var_names=None,
        kind="all",
        stat_focus="median",
        hdi_prob=0.95,
        round_to=4,
        skipna=True,
    ).reset_index(names=["parameter"])
    summary.to_csv(os.path.join(table_dir, "model_summary.csv"), index=False)

    summary_beta, sig_mask = summarize_beta(idata)
    summary_beta.to_csv(os.path.join(table_dir, "beta_summary.csv"), index=False)
    write_beta_latex_table(summary_beta, sig_mask, os.path.join(table_dir, "beta.tex"))

    plot_beta_forest(idata, os.path.join(fig_dir, "forest_slopes.pdf"))

    save_year_conditional_smooth_pooled(
        idata,
        prep["out_idx"],
        prep["year_mean"],
        prep["year_sd"],
        pool="weighted",
        out_path=os.path.join(fig_dir, "year_pooled_weighted.pdf"),
        degree=5,
    )
    save_year_conditional_smooth_pooled(
        idata,
        prep["out_idx"],
        prep["year_mean"],
        prep["year_sd"],
        pool="mean",
        out_path=os.path.join(fig_dir, "year_pooled_mean.pdf"),
        degree=5,
    )
    save_year_conditional_smooth_by_outcome(
        idata,
        prep["out_idx"],
        prep["year_mean"],
        prep["year_sd"],
        out_dir=fig_dir,
        degree=5,
    )

    icc_df, icc_overall = compute_icc_vpc(
        idata,
        out_idx=prep["out_idx"],
        reg_idx=prep["reg_idx"],
        cty_idx=prep["cty_idx"],
        prov_idx=prep["prov_idx"],
        fp_idx=prep["fp_idx"],
        link="logit",
        include_year=True,
        include_firstparty_group=("firstparty_int" in idata.posterior.data_vars),
    )
    icc_df.to_csv(os.path.join(table_dir, "icc_components.csv"), index=False)
    icc_overall.to_frame(name="ICC_mean").to_csv(
        os.path.join(table_dir, "icc_overall.csv"), index_label="component"
    )
    plot_icc_components(icc_df, os.path.join(fig_dir, "variance_components.pdf"))

    diag_dir = os.path.join(fig_dir, "diagnostics")
    pair_dir = os.path.join(diag_dir, "pairs")
    generate_diagnostic_plots(idata, diag_dir)
    generate_pair_plots(idata, pair_dir)

    logger.info("Analysis complete. Outputs saved to:")
    logger.info("  Figures: %s", fig_dir)
    logger.info("  Tables: %s", table_dir)
    if args.save_idata:
        logger.info("  InferenceData: %s", args.save_idata)


if __name__ == "__main__":
    main()
