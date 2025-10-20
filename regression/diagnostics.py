"""Posterior diagnostic plotting utilities for the regression workflow."""

from __future__ import annotations

import os
import re
from typing import Dict, Iterable, List, Sequence, Tuple

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

_PLOT_RC = {"plot.max_subplots": None}


def _get_figure(obj) -> plt.Figure:
    """Return the matplotlib figure associated with an ArviZ plot result."""

    if isinstance(obj, np.ndarray):
        for item in obj.ravel():
            if hasattr(item, "figure") and item.figure is not None:
                return item.figure
    if hasattr(obj, "figure") and obj.figure is not None:
        return obj.figure
    if hasattr(obj, "fig") and obj.fig is not None:
        return obj.fig
    raise ValueError("Unable to infer matplotlib Figure from object.")


def _format_coord_value(value: object) -> str:
    """Convert coordinate values to a compact string representation."""

    if isinstance(value, (np.generic, np.bool_)):
        value = value.item()
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8")
        except Exception:  # noqa: BLE001
            return repr(value)
    return str(value)


def _slugify(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "group"


def _triangular_indices(indices: Iterable[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
    filtered: List[Tuple[int, ...]] = []
    for idx in indices:
        if len(idx) >= 2 and idx[1] > idx[0]:
            continue
        filtered.append(idx)
    return filtered


def _flatten_variable(
    da: xr.DataArray,
    name: str,
    *,
    max_columns: int = 8,
    triangular: bool = False,
) -> List[Tuple[str, np.ndarray]]:
    """Return labelled scalar slices for the provided posterior variable."""

    dims = [d for d in da.dims if d not in ("chain", "draw")]
    chains = int(da.sizes.get("chain", 1))
    draws = int(da.sizes.get("draw", 1))
    ordered = da.transpose("chain", "draw", *dims, missing_dims="ignore")
    arr = np.asarray(ordered)
    arr = arr.reshape(chains, draws, -1)

    coords: List[np.ndarray] = []
    for dim in dims:
        if dim in ordered.coords:
            coords.append(np.asarray(ordered.coords[dim].values))
        else:
            coords.append(np.arange(int(ordered.sizes[dim])))

    shape = [len(c) for c in coords]
    if shape:
        iterator = list(np.ndindex(*shape))
    else:
        iterator = [()]

    if triangular and len(shape) >= 2:
        iterator = _triangular_indices(iterator)

    columns: List[Tuple[str, np.ndarray]] = []
    limit = min(max_columns, len(iterator))
    for idx in iterator[:limit]:
        parts = []
        for dim, dim_idx, coord in zip(dims, idx, coords):
            parts.append(f"{dim}={_format_coord_value(coord[dim_idx])}")
        label = name if not parts else f"{name}[{', '.join(parts)}]"
        if shape:
            flat_index = np.ravel_multi_index(idx, tuple(shape))
        else:
            flat_index = 0
        flat = arr[..., flat_index].copy()
        columns.append((label, flat))
    return columns


def _get_dataset_value(dataset: xr.Dataset | None, name: str):
    if dataset is None or name not in dataset:
        return None
    return dataset[name]


def _subset_idata_by_obs(idata: az.InferenceData, obs_idx: np.ndarray) -> az.InferenceData | None:
    """Return an InferenceData containing only the selected observations."""

    if obs_idx.size == 0:
        return None
    pp_y = _get_dataset_value(getattr(idata, "posterior_predictive", None), "y")
    obs_y = None
    observed = getattr(idata, "observed_data", None)
    if observed is not None:
        for key in ("y", "y_obs"):
            if key in observed:
                obs_y = observed[key]
                break
    if pp_y is None or obs_y is None:
        return None
    try:
        pp_sel = pp_y.isel(obs=obs_idx)
        obs_sel = obs_y.isel(obs=obs_idx)
    except Exception:  # noqa: BLE001
        return None
    return az.InferenceData(
        posterior_predictive=xr.Dataset({"y": pp_sel}),
        observed_data=xr.Dataset({"y": obs_sel}),
    )


def _iter_outcome_subsets(idata: az.InferenceData):
    const_data = getattr(idata, "constant_data", None)
    out_idx_da = _get_dataset_value(const_data, "out_idx")
    if out_idx_da is None:
        return
    out_idx = np.asarray(out_idx_da).astype(int)
    unique_outcomes = np.unique(out_idx)
    outcome_coords = None
    posterior = getattr(idata, "posterior", None)
    if posterior is not None and "outcome" in posterior.coords:
        try:
            outcome_coords = np.asarray(posterior.coords["outcome"].values)
        except Exception:  # noqa: BLE001
            outcome_coords = None
    for outcome_id in unique_outcomes:
        mask = np.nonzero(out_idx == int(outcome_id))[0]
        if mask.size == 0:
            continue
        if outcome_coords is not None and 0 <= int(outcome_id) < len(outcome_coords):
            label = _format_coord_value(outcome_coords[int(outcome_id)])
        else:
            label = str(int(outcome_id))
        subset = _subset_idata_by_obs(idata, mask)
        if subset is None:
            continue
        yield outcome_id, label, subset


def _classify_group(var_name: str) -> Tuple[str, bool, int]:
    """Return the group label, triangular flag, and column limit for a variable."""

    if var_name.endswith("_chol"):
        base = var_name[:-5].replace("_", " ").strip()
        group = f"{base.title()} cholesky"
        return group, True, 15
    if var_name == "beta":
        return "Population slopes", False, 12
    if var_name == "beta_pop":
        return "Pooled slopes", False, 8
    if var_name in {"ell_year", "sigma_year"}:
        return "Year GP hyperparameters", False, 4
    if var_name.endswith("_sd"):
        base = var_name[:-3].replace("_", " ").strip()
        return f"{base.title()} scales", False, 8
    if var_name.endswith("_rho") or var_name.endswith("_corr"):
        base = var_name.replace("_", " ").strip()
        return base.title(), False, 8
    return "", False, 0


def generate_pair_plots(
    idata: az.InferenceData,
    out_dir: str,
    *,
    max_groups: int | None = None,
) -> Dict[str, str]:
    """Generate grouped pair plots for posterior parameters.
    
    Returns a dictionary mapping group names to output file paths. Skipped 
    groups are not included in the returned dictionary.
    """

    os.makedirs(out_dir, exist_ok=True)
    posterior = idata.posterior
    diverging = None
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        diverging = np.asarray(idata.sample_stats["diverging"])

    grouped: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    for name, da in posterior.data_vars.items():
        group, triangular, limit = _classify_group(name)
        if not group or limit <= 0:
            continue
        columns = _flatten_variable(da, name, max_columns=limit, triangular=triangular)
        if len(columns) < 2:
            continue
        grouped.setdefault(group, []).extend(columns)

    saved: Dict[str, str] = {}
    for idx, (group, columns) in enumerate(sorted(grouped.items(), key=lambda kv: kv[0])):
        if max_groups is not None and idx >= max_groups:
            break
        if len(columns) < 2:
            continue
        var_dict = {}
        for label, values in columns:
            var_dict[label] = values
        # limit per group to manageable size
        max_cols = min(len(var_dict), 12)
        selected_items = list(var_dict.items())[:max_cols]
        posterior_dict = {label: values for label, values in selected_items}
        sample_stats = {"diverging": diverging} if diverging is not None else None
        group_idata = az.from_dict(posterior=posterior_dict, sample_stats=sample_stats)
        n_vars = len(posterior_dict)
        size = max(6.0, 2.5 * n_vars)
        figsize = (size, size)
        fig = None
        try:
            with az.rc_context(rc=_PLOT_RC):
                axes = az.plot_pair(
                    group_idata,
                    var_names=list(posterior_dict.keys()),
                    divergences=diverging is not None,
                    marginals=True,
                    kind="kde",
                    textsize=10,
                    figsize=figsize,
                )
            fig = _get_figure(axes)
            plt.tight_layout()
            path = os.path.join(out_dir, f"pair_{_slugify(group)}.pdf")
            fig.savefig(path, dpi=400, bbox_inches="tight")
            saved[group] = path
        finally:
            if fig is not None:
                plt.close(fig)
    return saved


def _select_small_vars(posterior: xr.Dataset, max_elements: int = 12, max_vars: int = 25) -> List[str]:
    names: List[str] = []
    for name, da in posterior.data_vars.items():
        dims = [d for d in da.dims if d not in ("chain", "draw")]
        if not dims:
            names.append(name)
            continue
        size = int(np.prod([int(da.sizes[d]) for d in dims]))
        if size <= max_elements:
            names.append(name)
    return names[:max_vars]


def generate_diagnostic_plots(
    idata: az.InferenceData,
    out_dir: str,
    *,
    var_names: Sequence[str] | None = None,
) -> Dict[str, str]:
    """Generate posterior diagnostic figures and save them to *out_dir*.
    
    Returns a dictionary mapping diagnostic names to output file paths. Plots 
    that fail to generate or cannot be saved are skipped and not included in 
    the returned dictionary.
    """

    os.makedirs(out_dir, exist_ok=True)
    posterior = idata.posterior
    if var_names is None:
        var_names = _select_small_vars(posterior)
    else:
        var_names = list(var_names)

    plot_specs = []
    if var_names:
        plot_specs.extend(
            [
                ("trace", lambda: az.plot_trace(idata, var_names=var_names, compact=True)),
                ("rank", lambda: az.plot_rank(idata, var_names=var_names)),
                ("autocorr", lambda: az.plot_autocorr(idata, var_names=var_names)),
                ("ess", lambda: az.plot_ess(idata, var_names=var_names)),
                ("mcse", lambda: az.plot_mcse(idata, var_names=var_names)),
                ("rhat", lambda: az.plot_rhat(idata, var_names=var_names)),
                ("density", lambda: az.plot_density(idata, var_names=var_names, shade=0.1)),
            ]
        )
    plot_specs.append(("energy", lambda: az.plot_energy(idata)))

    # Posterior predictive checks
    pp_data = getattr(idata, "posterior_predictive", None)
    if pp_data is not None and len(pp_data.data_vars):
        plot_specs.append(
            (
                "ppc_aggregate",
                lambda: az.plot_ppc(
                    idata,
                    data_pairs={"y": "y"},
                    var_names=["y"],
                    kind="hdi",
                    num_pp_samples=100,
                ),
            )
        )
        plot_specs.append(
            (
                "ppc_pit_aggregate",
                lambda: az.plot_ppc(
                    idata,
                    data_pairs={"y": "y"},
                    var_names=["y"],
                    kind="pit",
                    num_pp_samples=100,
                ),
            )
        )
        for _, label, subset in _iter_outcome_subsets(idata):
            name = f"ppc_outcome_{label}"
            pit_name = f"ppc_pit_outcome_{label}"

            def _plot_subset(kind: str, sub=subset):
                return az.plot_ppc(
                    sub,
                    data_pairs={"y": "y"},
                    var_names=["y"],
                    kind=kind,
                    num_pp_samples=100,
                )

            plot_specs.append((name, lambda _sub=subset: _plot_subset("hdi", _sub)))
            plot_specs.append((pit_name, lambda _sub=subset: _plot_subset("pit", _sub)))
    # LOO / k-hat diagnostics
    khat_result = None
    try:
        khat_result = az.loo(idata, pointwise=True)
    except Exception:  # noqa: BLE001
        khat_result = None
    if khat_result is not None:
        plot_specs.append(("khat", lambda: az.plot_khat(khat_result)))

    saved: Dict[str, str] = {}
    for name, plot_func in plot_specs:
        fig = None
        try:
            try:
                with az.rc_context(rc=_PLOT_RC):
                    axes = plot_func()
            except Exception:  # noqa: BLE001
                continue
            
            try:
                fig = _get_figure(axes)
            except ValueError:
                if isinstance(axes, plt.Figure):
                    fig = axes
                else:
                    # Unable to extract figure - skip this plot
                    continue
            
            try:
                plt.tight_layout()
                path = os.path.join(out_dir, f"diagnostic_{_slugify(name)}.pdf")
                fig.savefig(path, dpi=400, bbox_inches="tight")
                saved[name] = path
            except Exception:  # noqa: BLE001
                # If saving fails, skip this plot but continue with others
                continue
        finally:
            if fig is not None:
                plt.close(fig)
    return saved
