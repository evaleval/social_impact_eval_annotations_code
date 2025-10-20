"""Data loading and preprocessing utilities for the regression analysis."""

from __future__ import annotations

from os import path as osp
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def load_annotations(data_dir: str) -> pd.DataFrame:
    """Load the merged annotation table used by the regression models."""
    path = osp.join(data_dir, "MAIN_ANNOTATIONS_MERGED.tsv")
    return pd.read_csv(path, sep="\t")


def prepare_data(df: pd.DataFrame) -> Dict[str, object]:
    """Prepare design matrices, indices, and metadata from the raw annotations."""
    df = df.copy()

    for col in ["openness", "sector", "region", "country", "provider"]:
        df[col] = df[col].astype(str).str.strip()
    df["openness"] = df["openness"].str.lower()

    score_levels = np.sort(df["score"].dropna().unique())
    score_map = {v: i for i, v in enumerate(score_levels)}
    df["score0"] = df["score"].map(score_map).astype("int64")
    K = len(score_levels)

    category_levels = np.sort(df["category"].dropna().unique())
    category_map = {v: i for i, v in enumerate(category_levels)}
    df["outcome"] = df["category"].map(category_map).astype("int64")
    J = len(category_levels)

    df["fp_id"] = df["is_first_party"].astype(bool).astype("int64")

    def index_col(col: str) -> Tuple[pd.Series, pd.Index]:
        levels = pd.Index(np.sort(df[col].unique()), name=col)
        mapping = {v: i for i, v in enumerate(levels)}
        return df[col].map(mapping).astype("int64"), levels

    df["region_id"], region_levels = index_col("region")
    df["country_id"], country_levels = index_col("country")
    df["provider_id"], provider_levels = index_col("provider")

    df["open_bin"] = (df["openness"] == "open").astype("int64")
    open_mean = df["open_bin"].mean()
    df["open_c"] = df["open_bin"] - open_mean

    sector_ref = df["sector"].value_counts().idxmax()
    sector_levels = [s for s in sorted(df["sector"].unique()) if s != sector_ref]
    sector_dum = pd.get_dummies(df["sector"])[sector_levels].astype("int64")

    W = np.column_stack([df["open_c"].to_numpy(), sector_dum.to_numpy()]).astype("float32")
    pred_names = ["open_c"] + sector_levels

    y_obs = df["score0"].to_numpy().astype("int64")
    out_idx = df["outcome"].to_numpy().astype("int64")
    reg_idx = df["region_id"].to_numpy().astype("int64")
    cty_idx = df["country_id"].to_numpy().astype("int64")
    prov_idx = df["provider_id"].to_numpy().astype("int64")
    fp_idx = df["fp_id"].to_numpy().astype("int64")

    year_raw = df["year"].to_numpy()
    year_mean = float(year_raw.mean())
    year_sd = float(year_raw.std() if year_raw.std() > 0 else 1.0)
    year_std = (year_raw - year_mean) / year_sd
    year_u, year_idx = np.unique(year_std, return_inverse=True)
    Y = int(year_u.size)

    R = len(region_levels)
    C = len(country_levels)
    G = len(provider_levels)
    F = len(df["fp_id"].unique())

    C2R = np.zeros((C, R), dtype="float32")
    tmp = df[["country_id", "region_id"]].drop_duplicates()
    C2R[tmp["country_id"].to_numpy(), tmp["region_id"].to_numpy()] = 1.0

    P2C = np.zeros((G, C), dtype="float32")
    tmp = df[["provider_id", "country_id"]].drop_duplicates()
    P2C[tmp["provider_id"].to_numpy(), tmp["country_id"].to_numpy()] = 1.0

    reg_country_counts = C2R.sum(axis=0)
    cty_prov_counts = P2C.sum(axis=0)

    coords = dict(
        obs=np.arange(len(df)),
        outcome=np.arange(J),
        region=np.arange(R),
        country=np.arange(C),
        provider=np.arange(G),
        firstparty=np.arange(F),
        pred=np.array(pred_names, dtype=object),
        threshold=np.arange(K - 1),
        year=np.arange(Y),
    )

    meta = dict(
        score_map=score_map,
        category_map=category_map,
        sector_reference=sector_ref,
        open_mean=open_mean,
        region_levels=region_levels.tolist(),
        country_levels=country_levels.tolist(),
        provider_levels=provider_levels.tolist(),
    )

    return dict(
        df=df,
        W=W,
        y_obs=y_obs,
        out_idx=out_idx,
        reg_idx=reg_idx,
        cty_idx=cty_idx,
        prov_idx=prov_idx,
        fp_idx=fp_idx,
        K=K,
        J=J,
        C2R=C2R,
        P2C=P2C,
        reg_country_counts=reg_country_counts,
        cty_prov_counts=cty_prov_counts,
        year_std=year_std,
        year_u=year_u,
        year_idx=year_idx,
        Y=Y,
        year_mean=year_mean,
        year_sd=year_sd,
        coords=coords,
        meta=meta,
        pred_names=pred_names,
    )
