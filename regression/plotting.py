"""Plotting and post-processing helpers for the regression workflow."""


from __future__ import annotations

import os
import re
from typing import Iterable, Optional, Sequence, Tuple

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.ticker import MultipleLocator
from patsy import build_design_matrices, dmatrix


def _stack_samples(da: xr.DataArray) -> xr.DataArray:
    """Stack chain and draw dimensions."""
    return da.stack(s=("chain", "draw")).transpose(
        "s", *[d for d in da.dims if d not in ("chain", "draw")]
    )


def _sqexp_kernel(xa: np.ndarray, xb: np.ndarray, ell: float, sigma: float) -> np.ndarray:
    """Squared exponential covariance evaluated on two column vectors."""
    ell = float(max(ell, 1e-8))
    sigma = float(max(sigma, 1e-8))
    diff = xa - xb.T
    return (sigma**2) * np.exp(-0.5 * (diff**2) / (ell**2))


def _gp_conditional_draws(
    f_year: xr.DataArray,
    ell_year: xr.DataArray,
    sigma_year: xr.DataArray,
    x_obs: np.ndarray,
    x_new: np.ndarray,
    *,
    jitter: float = 1e-6,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample GP draws at new points conditioned on latent function draws."""

    if rng is None:
        rng = np.random.default_rng(0)

    x_obs = np.asarray(x_obs, dtype=float).reshape(-1, 1)
    x_new = np.asarray(x_new, dtype=float).reshape(-1, 1)
    if x_new.size == 0:
        J = f_year.sizes.get("outcome", 1)
        S = f_year.sizes.get("chain", 1) * f_year.sizes.get("draw", 1)
        return np.empty((0, J, S))

    F_sd = _stack_samples(f_year)
    f_arr = np.asarray(F_sd).transpose(1, 2, 0)  # (year_unique, outcome, sample)
    Y, J, S = f_arr.shape

    ell_stack = ell_year.stack(sample=("chain", "draw"))
    sig_stack = sigma_year.stack(sample=("chain", "draw"))

    if "outcome" in ell_stack.dims:
        ell_vals = np.asarray(ell_stack.transpose("outcome", "sample"))
    else:
        ell_vals = np.asarray(ell_stack)[None, :]
    if "outcome" in sig_stack.dims:
        sig_vals = np.asarray(sig_stack.transpose("outcome", "sample"))
    else:
        sig_vals = np.asarray(sig_stack)[None, :]

    if ell_vals.shape[0] == 1 and J > 1:
        ell_vals = np.broadcast_to(ell_vals, (J, ell_vals.shape[1]))
    if sig_vals.shape[0] == 1 and J > 1:
        sig_vals = np.broadcast_to(sig_vals, (J, sig_vals.shape[1]))

    eye_obs = np.eye(Y)
    eye_new = np.eye(x_new.shape[0])

    preds = np.empty((x_new.shape[0], J, S), dtype=float)
    for s in range(S):
        for j in range(J):
            ell = ell_vals[j, s]
            sig = sig_vals[j, s]
            if not np.isfinite(ell) or not np.isfinite(sig):
                preds[:, j, s] = np.nan
                continue
            K_xx = _sqexp_kernel(x_obs, x_obs, ell, sig) + jitter * eye_obs
            try:
                L = np.linalg.cholesky(K_xx)
            except np.linalg.LinAlgError:
                L = np.linalg.cholesky(K_xx + 10 * jitter * eye_obs)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, f_arr[:, j, s]))
            K_xnew_x = _sqexp_kernel(x_new, x_obs, ell, sig)
            mean = K_xnew_x @ alpha
            K_xnew_xnew = _sqexp_kernel(x_new, x_new, ell, sig) + jitter * eye_new
            v = np.linalg.solve(L, K_xnew_x.T)
            cov = K_xnew_xnew - v.T @ v
            cov = (cov + cov.T) / 2
            try:
                chol = np.linalg.cholesky(cov)
                sample = mean + chol @ rng.standard_normal(mean.shape[0])
            except np.linalg.LinAlgError:
                sample = mean
            preds[:, j, s] = sample
    return preds


def summarize_beta(idata: az.InferenceData) -> Tuple[pd.DataFrame, pd.Series]:
    """Return the beta summary dataframe and a mask for coefficients excluding zero."""
    summary = (
        az.summary(
            idata,
            var_names=r"^beta$",
            filter_vars="regex",
            kind="all",
            stat_focus="median",
            hdi_prob=0.95,
            round_to=4,
            skipna=True,
        )
        .reset_index(names=["parameter"])
        .assign(
            parameter=lambda d: d["parameter"]
            .astype(str)
            .str.replace(r"^beta\[(.*)\]$", r"\1", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.replace(r"\bopen_c\b", "open", regex=True)
        )
        .rename(columns={"parameter": "parameter, outcome index"})
    )

    hdi_cols = [
        c
        for c in summary.columns
        if isinstance(c, str)
        and (c.lower().startswith("eti_") or c.lower().startswith("hdi_"))
    ]
    if len(hdi_cols) < 2:
        hdi_cols = [
            c
            for c in summary.columns
            if isinstance(c, str) and ("eti" in c.lower() or "hdi" in c.lower())
        ]

    if hdi_cols:
        low = summary[hdi_cols].min(axis=1)
        high = summary[hdi_cols].max(axis=1)
        mask = low.notna() & high.notna() & ((low > 0) | (high < 0))
    else:
        mask = pd.Series(False, index=summary.index)

    return summary, mask


def escape_latex(value: str) -> str:
    """Escape LaTeX control characters except for existing commands."""
    if not isinstance(value, str) or value.startswith("\\"):
        return value
    trans = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(trans.get(ch, ch) for ch in value)


def write_beta_latex_table(
    summary: pd.DataFrame,
    significant_mask: pd.Series,
    out_path: str,
) -> str:
    """Format the beta summary into a LaTeX table and save it."""
    df_out = summary.copy()

    def _recode_idx(val):
        if not isinstance(val, str):
            return val
        m = re.match(r"^(.*?,\s*)(\d+)(\s*)$", val)
        if not m:
            return val
        prefix, num, suffix = m.groups()
        try:
            n = int(num)
        except ValueError:
            return val
        return f"{prefix}{n + 1}{suffix}" if 0 <= n <= 6 else val

    col_param = "parameter, outcome index"
    if col_param in df_out.columns:
        df_out[col_param] = df_out[col_param].map(_recode_idx)

    num_cols = df_out.select_dtypes(include=[np.number]).columns.tolist()
    df_out[num_cols] = df_out[num_cols].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")

    if significant_mask.any():
        df_out.loc[significant_mask, num_cols] = df_out.loc[
            significant_mask, num_cols
        ].map(lambda s: f"\\textbf{{{s}}}" if isinstance(s, str) and s else s)

    text_cols = [c for c in df_out.columns if c not in num_cols]
    for col in text_cols:
        df_out[col] = df_out[col].map(escape_latex)

    df_out.columns = [escape_latex(str(c)) for c in df_out.columns]

    latex_table = df_out.to_latex(index=False, escape=False)
    wrapped = "\\resizebox{\\textwidth}{!}{%\n" + latex_table + "\n}"

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(wrapped)
    return out_path

def plot_beta_forest(idata: az.InferenceData, out_path: str) -> str:
    """Save a customized forest plot highlighting slope coefficients.
    
    Returns the output path where the figure was saved.
    """
    fig = None
    try:
        ax = az.plot_forest(
            idata,
            kind="ridgeplot",
            var_names=[r"^beta(?!.*_chol)", r"^beta_pop(?!.*_chol)"],
            filter_vars="regex",
            combined=True,
            hdi_prob=0.95,
            ridgeplot_quantiles=[0.5],
            quartiles=False,
            ridgeplot_alpha=0.75,
            ridgeplot_overlap=1,
            ridgeplot_truncate=True,
            colors=["#92D4F3"],
            figsize=(8, 16),
        )
        ax0 = ax[0] if isinstance(ax, (list, np.ndarray)) else ax
        ax0.axvline(0, color="red", alpha=0.75, linestyle="-.", linewidth=1.25)

        ax0.xaxis.set_major_locator(MultipleLocator(1.0))
        ax0.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax0.margins(x=0, y=0)
        yt = np.asarray(ax0.get_yticks())
        if yt.size >= 2:
            step = float(np.median(np.diff(yt)))
        else:
            step = 1.0
        if yt.size > 0:
            ax0.set_ylim(yt[0] - 0.5 * step, yt[-1] + 2 * step)
        if hasattr(ax0, "spines") and "top" in ax0.spines:
            ax0.spines["top"].set_visible(False)
        try:
            ax0.figure.canvas.draw()
        except Exception:
            pass

        orig_labels = [t.get_text() for t in ax0.get_yticklabels()]

        def _get_group(lbl):
            if not isinstance(lbl, str):
                return None
            m = re.match(r"^(beta(?:_pop)?)\[(.*)\]$", lbl)
            if m:
                return m.group(1)
            if lbl.startswith("beta_pop"):
                return "beta_pop"
            if lbl.startswith("beta"):
                return "beta"
            return None

        def _clean_label(lbl):
            if not isinstance(lbl, str):
                return lbl
            s = lbl
            m = re.match(r"^(beta(?:_pop)?)\[(.*)\]$", s)
            if m:
                s = m.group(2)
            elif s.startswith("beta_pop"):
                s = re.sub(r"^beta_pop\s*", "", s)
            elif s.startswith("beta"):
                s = re.sub(r"^beta\s*", "", s)
            s = s.replace("[", "").replace("]", "")
            s = re.sub(r"\b([0-6])\b", lambda m: str(int(m.group(1)) + 1), s)
            s = re.sub(r"\s*,\s*", ", ", s).strip()
            s = s.replace("open_c", "open")
            return s

        groups = [_get_group(s) for s in orig_labels]
        clean = [_clean_label(s) for s in orig_labels]
        if any(isinstance(s, str) and s for s in clean):
            ax0.set_yticklabels(clean)
        yticks = ax0.get_yticks()

        def _add_horizontal_group_label(group_name, label_text, x_pos=-0.14, fontsize=18):
            idxs = [i for i, g in enumerate(groups) if g == group_name]
            if not idxs:
                return
            pos = [yticks[i] + 0.25 for i in idxs if i < len(yticks)]
            if not pos:
                return
            y_mid = 0.5 * (min(pos) + max(pos))
            ax0.text(
                x_pos,
                y_mid,
                label_text,
                transform=ax0.get_yaxis_transform(),
                rotation=0,
                va="center",
                ha="right",
                fontsize=fontsize,
                fontweight="bold",
                color="black",
                alpha=0.95,
                clip_on=False,
                zorder=10,
            )

        _add_horizontal_group_label("beta", "individual outcomes")
        _add_horizontal_group_label("beta_pop", "weighted average")

        fig = ax0.figure
        plt.tight_layout()
        fig.subplots_adjust(left=0.28)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=600, bbox_inches="tight")
        return out_path
    finally:
        if fig is not None:
            plt.close(fig)

def plot_year_conditional_smooth(
    idata: az.InferenceData,
    out_idx: Sequence[int],
    year_mean: float,
    year_sd: float,
    *,
    outcome: Optional[int] = None,
    pool: Optional[Iterable[float] | str] = None,
    hdi_probs: Sequence[float] = (0.95, 0.50),
    center: bool = True,
    degree: int = 3,
    grid: str | int | Sequence[float] = "unique",
    ax: Optional[plt.Axes] = None,
    title_prefix: str = "Conditional smooth for year",
) -> Tuple[plt.Axes, pd.DataFrame]:
    """Plot the conditional year smooth for spline or GP specifications."""
    year_std_obs = np.asarray(idata.constant_data["year_std"]).reshape(-1)
    year_u = np.asarray(idata.constant_data["year_u"]).reshape(-1)
    year_idx = np.asarray(idata.constant_data["year_idx"]).astype(int)
    years_obs = year_u[year_idx]

    def _get_outcome_coord(varname):
        if varname in idata.posterior and "outcome" in idata.posterior[varname].coords:
            return np.asarray(idata.posterior[varname].coords["outcome"].values)
        return None

    oc = _get_outcome_coord("weights_year")
    J = int(idata.posterior.sizes["outcome"])

    w = None
    out_label = outcome
    pool_mode: Optional[str] = None
    if pool is None:
        if outcome is None:
            counts = np.bincount(out_idx, minlength=J)
            j_star = int(np.argmax(counts))
            out_label = oc[j_star] if oc is not None else j_star
        else:
            if (oc is not None) and (outcome in list(oc)):
                # Add validation for np.where result before indexing
                matches = np.where(oc == outcome)[0]
                if len(matches) == 0:
                    raise ValueError(f"Outcome {outcome!r} not found in outcome coordinates")
                j_star = int(matches[0])
                out_label = outcome
            else:
                j_star = int(outcome)
                out_label = outcome
    else:
        if isinstance(pool, str):
            pool_mode = pool
            if pool == "mean":
                w = np.ones(J) / J
            elif pool == "weighted":
                counts = np.bincount(out_idx, minlength=J).astype(float)
                w = counts / counts.sum() if counts.sum() > 0 else np.ones(J) / J
            else:
                raise ValueError("pool must be 'mean', 'weighted', or an array-like of weights")
        else:
            w = np.asarray(pool, dtype=float)
            if w.shape != (J,):
                raise ValueError(f"custom weights must have length {J}, got {w.shape[0]}")
            s = w.sum()
            if s <= 0:
                raise ValueError("custom weights must have a positive sum")
            w = w / s
            pool_mode = "custom"
        if pool_mode is None:
            pool_mode = "custom"
        out_label = f"pooled ({pool_mode})"

    if grid == "unique":
        years_grid = np.unique(years_obs)
    elif isinstance(grid, int):
        years_grid = np.linspace(years_obs.min(), years_obs.max(), grid).astype(float)
    else:
        years_grid = np.array(list(grid), dtype=float)

    df_map = (
        pd.DataFrame({"year": years_obs, "z": year_std_obs}).groupby("year")["z"].mean()
    )
    if np.all(np.isin(years_grid, df_map.index.values)):
        z_grid = df_map.loc[years_grid].values
    else:
        coeffs = np.polyfit(df_map.index.values.astype(float), df_map.values, 1)
        z_grid = np.polyval(coeffs, years_grid)
    x_plot = z_grid * year_sd + year_mean

    has_spline = "weights_year" in idata.posterior
    has_gp = "f_year" in idata.posterior
    if not (has_spline or has_gp):
        raise KeyError(
            "Neither 'weights_year' (spline) nor 'f_year' (GP) found in idata.posterior."
        )

    if has_spline:
        n_basis = int(idata.posterior.sizes["year_basis"])
        formula = (
            f"bs(year_std, df={n_basis} - 1, degree={degree}, include_intercept=False)"
        )
        B_train = dmatrix(formula, data={"year_std": year_std_obs}, return_type="dataframe")
        info = B_train.design_info
        Xg = np.asarray(build_design_matrices([info], {"year_std": z_grid})[0])
        W_year = idata.posterior["weights_year"]
        W_sd = W_year.stack(sample=("chain", "draw")).transpose(
            "year_basis", "outcome", "sample"
        )
        if w is None:
            Wj = W_sd.isel(outcome=j_star)
            f = Xg @ np.asarray(Wj)
            method = "spline"
        else:
            w_da = xr.DataArray(w, dims=["outcome"])
            Wpool = (W_sd * w_da).sum("outcome")
            f = Xg @ np.asarray(Wpool)
            method = "spline (pooled)"
    else:
        method = "GP"
        f_year = idata.posterior["f_year"]
        ell_year = idata.posterior.get("ell_year")
        sigma_year = idata.posterior.get("sigma_year")
        if ell_year is None or sigma_year is None:
            raise KeyError("GP mode requires 'ell_year' and 'sigma_year' in the posterior")

        pos = pd.Index(np.round(year_u, 10)).get_indexer(np.round(years_grid, 10))
        if np.all(pos >= 0):
            F_sd = f_year.stack(sample=("chain", "draw")).transpose(
                "year_unique", "outcome", "sample"
            )
            if w is None:
                Fj = F_sd.isel(outcome=j_star, year_unique=xr.DataArray(pos, dims="g"))
                f = np.asarray(Fj)
            else:
                w_da = xr.DataArray(w, dims=["outcome"])
                Fpool = (
                    (F_sd * w_da)
                    .sum("outcome")
                    .isel(year_unique=xr.DataArray(pos, dims="g"))
                )
                f = np.asarray(Fpool)
                method = "GP (pooled)"
        else:
            preds = _gp_conditional_draws(
                f_year,
                ell_year,
                sigma_year,
                year_u,
                years_grid,
                jitter=1e-6,
            )
            if w is None:
                f = preds[:, j_star, :]
            else:
                f = np.tensordot(preds, w, axes=([1], [0]))
                method = "GP (pooled)"

    if center:
        f = f - f.mean(axis=0, keepdims=True)

    probs = sorted(hdi_probs, reverse=True)
    qs = [(1 - p) / 2 * 100 for p in probs], [(1 + p) / 2 * 100 for p in probs]
    f_med = np.median(f, axis=1)
    bands = [np.percentile(f, [lo, hi], axis=1) for lo, hi in zip(*qs)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    for (lo, hi), p in zip(bands, probs):
        ax.fill_between(
            x_plot,
            lo,
            hi,
            alpha=0.12 if p >= 0.90 else 0.20,
            label=f"{int(p*100)}% HDI",
        )
    ax.plot(x_plot, f_med, lw=2.5, label="Year smooth (link scale)")
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("Year")
    ax.set_ylabel(
        "Partial effect (link scale, centered)" if center else "Partial effect (link scale)"
    )
    if pool is None and isinstance(out_label, (int, np.integer)):
        label_text = f"outcome {int(out_label) + 1}"
    else:
        label_text = out_label
    ax.set_title(f"{title_prefix} — {method} — {label_text}")
    ax.legend(frameon=False)

    summary = pd.DataFrame({"year": years_grid, "median": f_med})
    for b, p in zip(bands, probs):
        summary[f"lo{int(p*100)}"] = b[0]
        summary[f"hi{int(p*100)}"] = b[1]
    return ax, summary

def _get_figure_from_obj(obj) -> plt.Figure:
    """Return a matplotlib figure from various ArviZ plotting outputs."""
    if isinstance(obj, np.ndarray):
        try:
            obj = obj.ravel()[0]
        except Exception:
            pass
    for attr in ("figure", "fig"):
        fig = getattr(obj, attr, None)
        if fig is not None:
            return fig
    getf = getattr(obj, "get_figure", None)
    if callable(getf):
        try:
            return getf()
        except Exception:
            pass
    if hasattr(obj, "savefig"):
        return obj
    return plt.gcf()


def save_year_conditional_smooth_pooled(
    idata: az.InferenceData,
    out_idx: Sequence[int],
    year_mean: float,
    year_sd: float,
    *,
    pool: str,
    out_path: str,
    hdi_probs: Sequence[float] = (0.95, 0.50),
    degree: int = 3,
) -> str:
    """Save a pooled year conditional smooth plot.
    
    Returns the output path where the figure was saved.
    """
    fig = None
    try:
        ax, _ = plot_year_conditional_smooth(
            idata,
            out_idx,
            year_mean,
            year_sd,
            pool=pool,
            hdi_probs=hdi_probs,
            degree=degree,
        )
        fig = _get_figure_from_obj(ax)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.0)
        return out_path
    finally:
        if fig is not None:
            plt.close(fig)


def save_year_conditional_smooth_by_outcome(
    idata: az.InferenceData,
    out_idx: Sequence[int],
    year_mean: float,
    year_sd: float,
    *,
    out_dir: str,
    hdi_probs: Sequence[float] = (0.95, 0.50),
    degree: int = 3,
) -> Sequence[str]:
    """Save year conditional smooth plots for each outcome.
    
    Returns a list of output paths where figures were saved.
    """
    paths = []
    J = int(idata.posterior.sizes["outcome"])
    os.makedirs(out_dir, exist_ok=True)
    for outcome in range(J):
        fig = None
        try:
            ax, _ = plot_year_conditional_smooth(
                idata,
                out_idx,
                year_mean,
                year_sd,
                outcome=outcome,
                hdi_probs=hdi_probs,
                degree=degree,
            )
            fig = _get_figure_from_obj(ax)
            out_path = os.path.join(out_dir, f"year_smooth_outcome_{outcome}.pdf")
            fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.0)
            paths.append(out_path)
        finally:
            if fig is not None:
                plt.close(fig)
    return paths


def _gather_component(posterior, var_name, idx_vec, level_dim, out_idx):
    da = _stack_samples(posterior[var_name]).transpose("s", level_dim, "outcome")
    arr = da.values
    S = arr.shape[0]
    idx = np.asarray(idx_vec, dtype=int)
    out = np.asarray(out_idx, dtype=int)
    return arr[np.arange(S)[:, None], idx[None, :], out[None, :]]


def compute_icc_vpc(
    idata: az.InferenceData,
    *,
    out_idx: Sequence[int],
    reg_idx: Sequence[int],
    cty_idx: Sequence[int],
    prov_idx: Sequence[int],
    fp_idx: Optional[Sequence[int]] = None,
    link: str = "logit",
    include_year: bool = True,
    include_firstparty_group: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Compute ICC/VPC style variance contributions for the regression model."""
    posterior = idata.posterior
    J = posterior.sizes["outcome"]
    out = np.asarray(out_idx, dtype=int)

    comps = {}
    comps["region"] = _gather_component(posterior, "region_int", reg_idx, "region", out_idx)
    comps["country"] = _gather_component(posterior, "country_dev", cty_idx, "country", out_idx)
    comps["provider"] = _gather_component(posterior, "provider_dev", prov_idx, "provider", out_idx)

    if include_firstparty_group and fp_idx is not None and "firstparty_int" in posterior:
        comps["firstparty(group)"] = _gather_component(
            posterior, "firstparty_int", fp_idx, "firstparty", out_idx
        )

    if include_year and "year_effect" in posterior:
        comps["year"] = (
            _stack_samples(posterior["year_effect"]).transpose("s", "obs").values
        )

    resid_var = (np.pi**2) / 3.0 if link == "logit" else 1.0

    rows = []
    for j in range(J):
        mask = out == j
        if mask.sum() == 0:
            continue
        comp_names = list(comps.keys())
        row_sums_all = []
        var_sum_all = []
        for s in range(next(iter(comps.values())).shape[0]):
            C = np.vstack([comps[name][s, mask] for name in comp_names])
            Sigma = np.cov(C, bias=False)
            tot = Sigma.sum()
            var_sum_all.append(tot)
            row_sums_all.append(Sigma.sum(axis=1))
        row_sums_all = np.asarray(row_sums_all)
        var_sum_all = np.asarray(var_sum_all)
        denom = var_sum_all + resid_var
        icc_total = var_sum_all / denom
        icc_comp = row_sums_all / denom[:, None]

        for g_idx, g in enumerate(comp_names):
            vals = icc_comp[:, g_idx]
            rows.append(
                {
                    "outcome": j,
                    "component": g,
                    "ICC_mean": float(vals.mean()),
                    "ICC_05": float(np.quantile(vals, 0.05)),
                    "ICC_95": float(np.quantile(vals, 0.95)),
                }
            )
        rows.append(
            {
                "outcome": j,
                "component": "total_random",
                "ICC_mean": float(icc_total.mean()),
                "ICC_05": float(np.quantile(icc_total, 0.05)),
                "ICC_95": float(np.quantile(icc_total, 0.95)),
            }
        )

    df = pd.DataFrame(rows).sort_values(["outcome", "component"]).reset_index(drop=True)
    counts = np.bincount(out, minlength=J).astype(float)
    w = counts / counts.sum()
    pivot = df.pivot_table(
        index="component", columns="outcome", values="ICC_mean", aggfunc="mean"
    ).fillna(0.0)
    overall_weighted = (pivot @ w).sort_values(ascending=False)
    return df, overall_weighted


def plot_icc_components(icc_df: pd.DataFrame, out_path: str) -> str:
    """Create the stacked ICC component bar plot.
    
    Returns the output path where the figure was saved. If no ICC components 
    are available for plotting, creates a placeholder figure with a message.
    """
    df_comp = icc_df[icc_df["component"] != "total_random"].copy()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig = None
    try:
        if df_comp.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.axis("off")
            ax.set_title("No ICC components available for plotting", fontsize=16)
            fig.savefig(out_path, dpi=600, bbox_inches="tight")
            return out_path

        comp_order = (
            df_comp.groupby("component")["ICC_mean"].mean().sort_values(ascending=False).index
        )
        pv = (
            df_comp.pivot_table(index="outcome", columns="component", values="ICC_mean", aggfunc="mean")
            .reindex(columns=comp_order)
            .sort_index()
            .fillna(0.0)
        )
        row_sums = pv.sum(axis=1).replace(0, np.nan)
        pv_norm = pv.div(row_sums, axis=0).fillna(0.0)

        cmap = plt.cm.get_cmap("tab20", pv_norm.shape[1])
        colors = [cmap(i) for i in range(pv_norm.shape[1])]

        fig, ax = plt.subplots(figsize=(12, 7))
        bar_width = 0.55
        pv_norm.plot(kind="bar", stacked=True, ax=ax, color=colors, width=bar_width)
        ax.margins(x=0.18)
        ax.set_xlabel("Outcome", fontsize=16)
        ax.set_ylabel("Proportion (normalized to 1)", fontsize=16)
        ax.set_title("Variance decomposition by component (normalized per outcome)", fontsize=20)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(
            title="Component",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=False,
            prop={"size": 14},
            title_fontsize=14,
        )
        ax.tick_params(axis="both", labelsize=14)
        if hasattr(ax, "spines"):
            if "top" in ax.spines:
                ax.spines["top"].set_visible(False)
            if "right" in ax.spines:
                ax.spines["right"].set_visible(False)

        def _recode_tick(val):
            try:
                n = int(str(val).strip())
                if 0 <= n <= 6:
                    return str(n + 1)
            except Exception:
                pass
            return str(val)

        xticks = ax.get_xticks()
        ax.set_xticklabels([_recode_tick(v) for v in pv_norm.index], rotation=0, fontsize=14)

        inside_thresh = 0.06
        side_dx = 0.22
        for i, outcome in enumerate(pv_norm.index):
            cum = 0.0
            for j, comp in enumerate(pv_norm.columns):
                v = float(pv_norm.iloc[i, j])
                if v <= 0:
                    continue
                y_center = cum + v / 2.0
                x_center = xticks[i]
                if v >= inside_thresh:
                    ax.text(
                        x_center,
                        y_center,
                        f"{v:.3f}",
                        ha="center",
                        va="center",
                        fontsize=12,
                        color="black",
                        zorder=5,
                    )
                else:
                    x_text = x_center + (bar_width / 2.0) + side_dx
                    ax.annotate(
                        f"{v:.3f}",
                        xy=(x_center + bar_width / 2.0, y_center),
                        xytext=(x_text, y_center),
                        textcoords="data",
                        ha="left",
                        va="center",
                        fontsize=12,
                        color="black",
                        arrowprops=dict(arrowstyle="-", color="0.3", lw=1.0),
                        zorder=6,
                        clip_on=False,
                    )
                cum += v

        fig.tight_layout()
        fig.savefig(out_path, dpi=600, bbox_inches="tight")
        return out_path
    finally:
        if fig is not None:
            plt.close(fig)
