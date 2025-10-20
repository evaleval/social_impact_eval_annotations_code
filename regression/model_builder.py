"""Model construction utilities for the hierarchical regression.

Example
-------
The unified predictor matrix only includes the first-party indicator when
explicitly requested. For example, disabling the predictor leaves the
original design matrix unchanged::

    >>> import numpy as np
    >>> coords = {
    ...     "obs": np.arange(3),
    ...     "outcome": ["turnout"],
    ...     "region": ["r0"],
    ...     "country": ["c0"],
    ...     "provider": ["p0"],
    ...     "firstparty": ["f0"],
    ...     "pred": ["intercept"],
    ... }
    >>> args = dict(
    ...     coords=coords,
    ...     out_idx=np.zeros(3, dtype=int),
    ...     reg_idx=np.zeros(3, dtype=int),
    ...     cty_idx=np.zeros(3, dtype=int),
    ...     prov_idx=np.zeros(3, dtype=int),
    ...     fp_idx=np.zeros(3, dtype=int),
    ...     W=np.ones((3, 1)),
    ...     y_obs=np.zeros(3, dtype=int),
    ...     K=3,
    ...     include_firstparty=False,
    ...     C2R=np.ones((1, 1)),
    ...     P2C=np.ones((1, 1)),
    ...     reg_country_counts=np.ones(1),
    ...     cty_prov_counts=np.ones(1),
    ... )
    >>> model = build_model(**args)
    >>> "firstparty" in model.coords["pred_all"]
    False
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from patsy import dmatrix


def build_model(
    *,
    coords,
    out_idx,
    reg_idx,
    cty_idx,
    prov_idx,
    fp_idx,
    W,
    y_obs,
    K,
    # Unified optional predictors
    include_firstparty: bool = True,
    firstparty_role: str = "predictor",  # predictor | group
    shared_slopes: bool = False,
    link: str = "logit",
    year_std=None,
    year_u=None,
    year_idx=None,
    year_mode: str | None = "linear",  # None|none|linear|spline|gp
    pop_slope_corr: bool = False,
    intercept_corr: dict | None = None,
    slopes: dict | None = None,
    slope_corr: dict | None = None,
    lkj_eta: float = 3.0,
    sd_reg_int: float = 0.5,
    sd_cty_dev: float = 0.2,
    sd_prov_dev: float = 0.2,
    sd_fp_int: float = 0.5,
    sd_sigma: float = 0.2,
    sd_slope: float = 0.25,
    sd_year: float = 0.3,
    separate_slope_sd: bool = False,
    cutpoint_prior: str = "dirichlet",
    dirichlet_center: bool = True,
    # Year spline / gp
    spline_df: int = 8,
    spline_degree: int = 3,
    gp_jitter: float = 1e-6,
    gp_shared_hypers: bool = True,
    # Hierarchy structure matrices
    C2R=None,
    P2C=None,
    reg_country_counts=None,
    cty_prov_counts=None,
    # Data mutability
    mutable: tuple = ("W_total", "y_obs", "year_std", "year_u", "year_idx", "out_idx"),
    return_components: bool = False,
):
    if slopes is None:
        slopes = {
            "region": True,
            "country": False,
            "provider": False,
            "firstparty": False,
        }
    if intercept_corr is None:
        intercept_corr = {
            "region": True,
            "country": True,
            "provider": True,
            "firstparty": True,
        }
    if slope_corr is None:
        slope_corr = {
            "region": False,
            "country": False,
            "provider": False,
            "firstparty": False,
        }
    if link not in {"logit", "probit"}:
        raise ValueError("link must be 'logit' or 'probit'")
    if firstparty_role not in {"predictor", "group"}:
        raise ValueError("firstparty_role must be 'predictor' or 'group'")
    if cutpoint_prior not in {"dirichlet", "normal"}:
        raise ValueError("cutpoint_prior must be 'dirichlet' or 'normal'")
    if include_firstparty and firstparty_role == "group":
        raise ValueError("include_firstparty=True but role='group' (would duplicate).")

    with pm.Model(coords=coords) as m:
        J = len(m.coords["outcome"])
        N = W.shape[0]

        # Helper for mutable wrapping
        def maybe_data(name, value):
            if name in mutable and value is not None:
                return pm.Data(name, value)
            return value

        # ----- Build unified predictor matrix -----
        pred_names = list(m.coords["pred"])
        W_parts = [W]

        if include_firstparty and firstparty_role == "predictor":
            fp_centered = (fp_idx - np.mean(fp_idx)).reshape(N, 1)
            W_parts.append(fp_centered)
            pred_names.append("firstparty")

        W_total = np.concatenate(W_parts, axis=1)
        if "pred_all" not in m.coords:
            m.add_coord("pred_all", pred_names)
        # print(pred_names)
        W_total_tt = maybe_data("W_total", W_total)

        # Wrap other observed indices
        y_tt = maybe_data("y_obs", y_obs)
        out_tt = maybe_data("out_idx", out_idx)
        reg_tt = maybe_data("reg_idx", reg_idx)
        cty_tt = maybe_data("cty_idx", cty_idx)
        prov_tt = maybe_data("prov_idx", prov_idx)
        fp_tt = maybe_data("fp_idx", fp_idx)
        year_std_tt = maybe_data("year_std", year_std)
        year_u_tt = maybe_data("year_u", year_u)
        year_idx_tt = maybe_data("year_idx", year_idx)

        if year_u is not None and "year_unique" not in m.coords:
            m.add_coord("year_unique", np.asarray(year_u, dtype=float))

        # Structural matrices
        C2R_tt = pt.as_tensor_variable(C2R) if C2R is not None else None
        P2C_tt = pt.as_tensor_variable(P2C) if P2C is not None else None
        reg_cnts_tt = (
            pt.as_tensor_variable(reg_country_counts).reshape((-1, 1))
            if reg_country_counts is not None
            else None
        )
        cty_cnts_tt = (
            pt.as_tensor_variable(cty_prov_counts).reshape((-1, 1))
            if cty_prov_counts is not None
            else None
        )

        # ----- Intercepts -----
        def make_intercepts(label, sd_base, correlated):
            if not correlated:
                sd = pm.HalfNormal(f"{label}_sd", sd_base, dims=("outcome",))
                z = pm.Normal(f"z_{label}_int", 0.0, 1.0, dims=(label, "outcome"))
                return pm.Deterministic(f"{label}_int", z * sd, dims=(label, "outcome"))
            sd_dist = pm.HalfNormal.dist(sd_base, shape=J)
            L, _, _ = pm.LKJCholeskyCov(
                f"{label}_chol", n=J, eta=lkj_eta, sd_dist=sd_dist, compute_corr=True
            )
            z = pm.Normal(f"z_{label}_int", 0.0, 1.0, dims=(label, "outcome"))
            return pm.Deterministic(f"{label}_int", z @ L.T, dims=(label, "outcome"))

        B_reg_int = make_intercepts(
            "region", sd_reg_int, intercept_corr.get("region", True)
        )
        B_cty_raw = make_intercepts(
            "country", sd_cty_dev, intercept_corr.get("country", True)
        )
        cty_means_by_reg = (C2R_tt.T @ B_cty_raw) / reg_cnts_tt
        B_cty_dev = pm.Deterministic(
            "country_dev",
            B_cty_raw - (C2R_tt @ cty_means_by_reg),
            dims=("country", "outcome"),
        )

        B_prov_raw = make_intercepts(
            "provider", sd_prov_dev, intercept_corr.get("provider", True)
        )
        prov_means_by_cty = (P2C_tt.T @ B_prov_raw) / cty_cnts_tt
        B_prov_dev = pm.Deterministic(
            "provider_dev",
            B_prov_raw - (P2C_tt @ prov_means_by_cty),
            dims=("provider", "outcome"),
        )

        if firstparty_role == "group":
            B_fp_int = make_intercepts(
                "firstparty", sd_fp_int, intercept_corr.get("firstparty", True)
            )
        else:
            B_fp_int = None

        # ----- Base linear predictor (intercepts) -----
        eta = (
            B_reg_int[reg_tt, out_tt]
            + B_cty_dev[cty_tt, out_tt]
            + B_prov_dev[prov_tt, out_tt]
        )
        if firstparty_role == "group":
            eta += B_fp_int[fp_tt, out_tt]

        # ----- Population slopes (shared vs outcome-specific) -----
        P_total = len(pred_names)
        if shared_slopes:
            sd_beta = pm.HalfNormal("sd_beta", sd_sigma)
            beta_raw = pm.Normal("beta_raw", 0.0, 1.0, dims=("pred_all",))
            beta_pop = pm.Deterministic(
                "beta_pop", beta_raw * sd_beta, dims=("pred_all",)
            )
            beta = pm.Deterministic(
                "beta",
                pt.repeat(beta_pop[:, None], J, axis=1),
                dims=("pred_all", "outcome"),
            )
        else:
            if pop_slope_corr:
                sd_dist = pm.HalfNormal.dist(sd_sigma, shape=J)
                L_beta, _, _ = pm.LKJCholeskyCov(
                    "beta_chol", n=J, eta=lkj_eta, sd_dist=sd_dist, compute_corr=True
                )
                z_beta = pm.Normal("z_beta", 0.0, 1.0, dims=("pred_all", "outcome"))
                beta = pm.Deterministic(
                    "beta", z_beta @ L_beta.T, dims=("pred_all", "outcome")
                )
            else:
                sd_out = pm.HalfNormal("sd_beta_out", sd_sigma, dims=("outcome",))
                z_beta = pm.Normal("z_beta", 0.0, 1.0, dims=("pred_all", "outcome"))
                beta = pm.Deterministic(
                    "beta", z_beta * sd_out, dims=("pred_all", "outcome")
                )
            # Weighted average for 1-D summary
            outcome_counts = np.bincount(out_idx, minlength=J).astype("float64")
            weights = outcome_counts / outcome_counts.sum()
            beta_pop = pm.Deterministic(
                "beta_pop", (beta * weights).sum(axis=1), dims=("pred_all",)
            )

        # ----- Group-level slopes -----
        def make_slopes(label):
            if not slopes.get(label, False):
                return None
            if shared_slopes:
                # group-level deviations common across outcomes
                if slope_corr.get(label, False):
                    # correlation across predictors not modeled here (keep simple)
                    sd = pm.HalfNormal(f"{label}_slope_sd", sd_slope)
                    z = pm.Normal(
                        f"z_{label}_slope", 0.0, 1.0, dims=(label, "pred_all")
                    )
                    return pm.Deterministic(
                        f"{label}_slope", z * sd, dims=(label, "pred_all")
                    )
                else:
                    z = pm.Normal(
                        f"{label}_slope", 0.0, sd_slope, dims=(label, "pred_all")
                    )
                    return z
            # outcome-specific slopes
            if slope_corr.get(label, False):
                sd_dist = pm.HalfNormal.dist(sd_slope, shape=J)
                L, _, _ = pm.LKJCholeskyCov(
                    f"{label}_slope_chol",
                    n=J,
                    eta=lkj_eta,
                    sd_dist=sd_dist,
                    compute_corr=True,
                )
                z = pm.Normal(
                    f"z_{label}_slope", 0.0, 1.0, dims=(label, "pred_all", "outcome")
                )
                if separate_slope_sd:
                    slope_sd = pm.HalfNormal(
                        f"{label}_slope_sd_raw", sd_slope, dims=("pred_all", "outcome")
                    )
                    z = z * slope_sd
                return pm.Deterministic(
                    f"{label}_slope",
                    pt.matmul(z, L.T),
                    dims=(label, "pred_all", "outcome"),
                )
            else:
                s = pm.HalfNormal(
                    f"{label}_slope_sd", sd_slope, dims=("pred_all", "outcome")
                )
                z = pm.Normal(
                    f"z_{label}_slope", 0.0, 1.0, dims=(label, "pred_all", "outcome")
                )
                return pm.Deterministic(
                    f"{label}_slope", z * s, dims=(label, "pred_all", "outcome")
                )

        S_reg = make_slopes("region")
        S_cty = make_slopes("country")
        S_pro = make_slopes("provider")
        S_fp = make_slopes("firstparty") if firstparty_role == "group" else None

        # ----- Fixed + random slope contributions -----
        if shared_slopes:
            fe = (W_total_tt * beta_pop[None, :]).sum(axis=1)
            if S_reg is not None:
                fe += (W_total_tt * S_reg[reg_tt, :]).sum(axis=1)
            if S_cty is not None:
                fe += (W_total_tt * S_cty[cty_tt, :]).sum(axis=1)
            if S_pro is not None:
                fe += (W_total_tt * S_pro[prov_tt, :]).sum(axis=1)
            if S_fp is not None:
                fe += (W_total_tt * S_fp[fp_tt, :]).sum(axis=1)
        else:
            fe_mat = W_total_tt @ beta  # (N,outcome)
            fe = fe_mat[pt.arange(N), out_tt]
            if S_reg is not None:
                fe += pt.sum(W_total_tt * S_reg[reg_tt, :, out_tt], axis=1)
            if S_cty is not None:
                fe += pt.sum(W_total_tt * S_cty[cty_tt, :, out_tt], axis=1)
            if S_pro is not None:
                fe += pt.sum(W_total_tt * S_pro[prov_tt, :, out_tt], axis=1)
            if S_fp is not None:
                fe += pt.sum(W_total_tt * S_fp[fp_tt, :, out_tt], axis=1)
        eta += fe

        # ----- Year effect (if modeled separately) -----
        if year_mode in {None, "none"} or year_std_tt is None:
            year_contrib = pt.zeros((N,))
            year_components = {}
        else:
            if year_mode == "linear":
                beta_year = pm.Normal("beta_year", 0.0, sd_year, dims=("outcome",))
                year_mat = pt.outer(year_std_tt, beta_year)
                year_contrib = year_mat[pt.arange(N), out_tt]
                year_components = {"beta_year": beta_year}
            elif year_mode == "spline":
                spline_formula = (
                    "bs(year_std, "
                    f"df={spline_df}, "
                    f"degree={spline_degree}, "
                    "include_intercept=False)"
                )
                B_np = dmatrix(spline_formula, data={"year_std": year_std})
                if "year_basis" not in m.coords:
                    m.add_coord("year_basis", np.arange(B_np.shape[1]))
                B_basis = pm.Data("B_year_basis", B_np)
                weights_year = pm.Normal(
                    "weights_year", 0.0, sd_year, dims=("year_basis", "outcome")
                )
                year_mat = B_basis @ weights_year
                year_contrib = year_mat[pt.arange(N), out_tt]
                year_components = {"weights_year": weights_year}
            elif year_mode == "gp":
                if year_u_tt is None or year_idx_tt is None:
                    raise ValueError("year_u and year_idx required for GP mode.")
                yu = year_u_tt[:, None]
                if gp_shared_hypers:
                    ell_year = pm.HalfNormal("ell_year", 1.0)
                    sigma_year = pm.HalfNormal("sigma_year", sd_year)
                    cov = sigma_year**2 * pm.gp.cov.ExpQuad(1, ell_year)
                    if gp_jitter:
                        cov = cov + pm.gp.cov.WhiteNoise(gp_jitter)
                    gp = pm.gp.Latent(cov_func=cov)
                    f_cols = [gp.prior(f"f_year_{j}", X=yu) for j in range(J)]
                    hyper_dict = {"ell_year": ell_year, "sigma_year": sigma_year}
                else:
                    ell_year = pm.HalfNormal("ell_year", 1.0, dims=("outcome",))
                    sigma_year = pm.HalfNormal("sigma_year", sd_year, dims=("outcome",))
                    f_cols = []
                    for j in range(J):
                        cov_j = (sigma_year[j] ** 2) * pm.gp.cov.ExpQuad(1, ell_year[j])
                        if gp_jitter:
                            cov_j = cov_j + pm.gp.cov.WhiteNoise(gp_jitter)
                        gp_j = pm.gp.Latent(cov_func=cov_j)
                        f_cols.append(gp_j.prior(f"f_year_{j}", X=yu))
                    hyper_dict = {"ell_year": ell_year, "sigma_year": sigma_year}
                f_year = pt.stack(f_cols, axis=1)
                f_year = pm.Deterministic(
                    "f_year", f_year, dims=("year_unique", "outcome")
                )
                year_mat = f_year[year_idx_tt, :]
                year_contrib = year_mat[pt.arange(N), out_tt]
                year_components = {**hyper_dict, "f_year": f_year}
            else:
                raise ValueError("Unhandled year_mode")
        eta = eta + year_contrib
        pm.Deterministic("year_effect", year_contrib, dims=("obs",))
        pm.Deterministic("eta_obs", eta, dims=("obs",))

        # ----- Cutpoints -----
        if cutpoint_prior == "dirichlet":
            cut_spacings = pm.Dirichlet(
                "cut_spacings", a=pt.ones((J, K - 1)), dims=("outcome", "threshold")
            )
            raw_cp = pt.cumsum(cut_spacings, axis=-1) * float(K)
            if dirichlet_center:
                cp_loc = pm.Normal("cp_loc", 0.0, 5.0, dims=("outcome",))
                centered = raw_cp - pt.mean(raw_cp, axis=-1, keepdims=True)
                cutpoints = pm.Deterministic(
                    "cutpoints",
                    centered + cp_loc[..., None],
                    dims=("outcome", "threshold"),
                )
            else:
                cutpoints = pm.Deterministic(
                    "cutpoints", raw_cp, dims=("outcome", "threshold")
                )
        else:
            raw_cuts = pm.Normal("raw_cuts", 0.0, 1.0, dims=("outcome", "threshold"))
            cutpoints = pm.Deterministic(
                "cutpoints", pt.sort(raw_cuts, axis=-1), dims=("outcome", "threshold")
            )

        c_obs = cutpoints[out_tt]

        # ----- Likelihood -----
        if link == "logit":
            pm.OrderedLogistic(
                "y", eta=eta, cutpoints=c_obs, observed=y_tt, dims=("obs",)
            )
        else:
            pm.OrderedProbit(
                "y", eta=eta, cutpoints=c_obs, observed=y_tt, dims=("obs",)
            )

        if return_components:
            comp = {
                "model": m,
                "beta": beta,
                "beta_pop": beta_pop,
                "B_reg_int": B_reg_int,
                "B_cty_dev": B_cty_dev,
                "B_prov_dev": B_prov_dev,
                "B_fp_int": B_fp_int,
                "S_reg": S_reg,
                "S_cty": S_cty,
                "S_pro": S_pro,
                "S_fp": S_fp,
                "eta": eta,
                "cutpoints": cutpoints,
                "year_effect": year_contrib,
                "shared_slopes": shared_slopes,
            }
            comp.update(year_components)
            return comp
    return m
