#!/usr/bin/env python3
"""
export_guardrailed_residuals.py

Generate a derived data product of guardrailed (post-admissible-correction)
supernova residuals for public release.

Output columns:
  - SN identifier (if available)
  - z, RA, DEC
  - Delta_clean        : baseline + SALT2-leakage removed residual
  - Delta_hat_corr     : admissible correction operator prediction
  - Delta_post         : Delta_clean - Delta_hat_corr

This product is intended for conditional tests only.
It is NOT claimed to represent pure physical residuals.

Run:
  python3 scripts/export_guardrailed_residuals.py \
    --sn data/pantheonplus/distance_moduli/Pantheon_SH0ES.dat \
    --out data/derived/pantheon_guardrailed_residuals.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------
# Utilities (identical semantics to paper / Test 8)
# ------------------------------------------------------------

def smooth_baseline(z: np.ndarray, mu: np.ndarray, span: float = 0.035, grid_points: int = 320) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    mu = np.asarray(mu, dtype=float)

    if z.size == 0:
        return mu.copy()

    idx = np.argsort(z)
    zs = z[idx]
    mus = mu[idx]

    if np.unique(zs).size < 2:
        return np.full_like(mu, float(np.mean(mus)))

    grid_points = min(grid_points, zs.size)
    z_grid = np.linspace(float(zs[0]), float(zs[-1]), grid_points)

    bw = max(span, 1e-4)
    delta = (z_grid[:, None] - zs[None, :]) / bw
    w = np.exp(-0.5 * delta * delta)
    ws = np.sum(w, axis=1)
    ws = np.where(ws == 0.0, 1.0, ws)

    mu_grid = (w @ mus) / ws
    return np.interp(z, z_grid, mu_grid)


def remove_salt2_leakage(resid: np.ndarray, x1: np.ndarray, c: np.ndarray) -> np.ndarray:
    resid = np.asarray(resid, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    c = np.asarray(c, dtype=float)

    X = np.column_stack([x1, c])
    m = np.isfinite(resid) & np.isfinite(x1) & np.isfinite(c)
    if np.sum(m) < 3:
        return resid.copy()

    beta, *_ = np.linalg.lstsq(X[m], resid[m], rcond=None)
    out = resid.copy()
    out[m] = resid[m] - (X[m] @ beta)
    return out


def build_design_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Admissible feature class (same as paper/Test 8):
      - x1, c
      - bias fields (if present)
      - error proxies (if present)
      - sky footprint basis
      - survey fixed effects (drop one)
    """
    cols: List[str] = []
    blocks: List[np.ndarray] = []

    for c in ["x1", "c"]:
        if c in df.columns:
            blocks.append(df[c].to_numpy(dtype=float)[:, None])
            cols.append(c)

    for c in ["biasCor_m_b", "biasCor_m_b_COVSCALE", "biasCor_m_b_COVADD"]:
        if c in df.columns:
            blocks.append(df[c].to_numpy(dtype=float)[:, None])
            cols.append(c)

    for c in ["m_b_corr_err_DIAG", "m_b_corr_err_VPEC", "zHDERR", "zCMBERR"]:
        if c in df.columns:
            blocks.append(df[c].to_numpy(dtype=float)[:, None])
            cols.append(c)

    if "RA" in df.columns and "DEC" in df.columns:
        ra = np.deg2rad(df["RA"].to_numpy(dtype=float))
        dec = np.deg2rad(df["DEC"].to_numpy(dtype=float))
        blocks += [
            np.sin(ra)[:, None],
            np.cos(ra)[:, None],
            np.sin(dec)[:, None],
            np.cos(dec)[:, None],
        ]
        cols += ["sinRA", "cosRA", "sinDEC", "cosDEC"]

    if "IDSURVEY" in df.columns:
        s = df["IDSURVEY"].to_numpy(dtype=int)
        uniq = np.unique(s)
        for u in uniq[1:]:
            blocks.append((s == u).astype(float)[:, None])
            cols.append(f"FE_IDSURVEY_{u}")

    if not blocks:
        X = np.ones((len(df), 1))
        return X, ["intercept_only"]

    X = np.hstack(blocks)
    X = np.hstack([np.ones((len(df), 1)), X])
    return X, ["intercept"] + cols


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sn", required=True, help="Pantheon_SH0ES.dat path")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--span", type=float, default=0.035)
    args = ap.parse_args()

    sn_path = Path(args.sn).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(sn_path, sep=r"\s+", comment="#")

    required = ["MU_SH0ES", "zCMB", "x1", "c", "RA", "DEC", "IDSURVEY"]
    mask = np.ones(len(df), dtype=bool)
    for c in required:
        mask &= df[c].notna().to_numpy() if c == "IDSURVEY" else np.isfinite(df[c].to_numpy(dtype=float))
    df = df.loc[mask].copy()

    mu = df["MU_SH0ES"].to_numpy(float)
    z = df["zCMB"].to_numpy(float)
    x1 = df["x1"].to_numpy(float)
    c = df["c"].to_numpy(float)

    # Δ_clean
    mu_hat = smooth_baseline(z, mu, span=args.span)
    resid = mu - mu_hat
    delta_clean = remove_salt2_leakage(resid, x1, c)

    # admissible correction
    X, names = build_design_matrix(df)
    m = np.all(np.isfinite(X), axis=1) & np.isfinite(delta_clean)
    beta, *_ = np.linalg.lstsq(X[m], delta_clean[m], rcond=None)

    delta_hat = np.full_like(delta_clean, np.nan)
    delta_hat[m] = X[m] @ beta

    delta_post = delta_clean - delta_hat

    out = pd.DataFrame({
        "z": z,
        "RA": df["RA"].to_numpy(float),
        "DEC": df["DEC"].to_numpy(float),
        "Delta_clean": delta_clean,
        "Delta_hat_corr": delta_hat,
        "Delta_post": delta_post,
    })

    if "CID" in df.columns:
        out.insert(0, "CID", df["CID"].astype(str).to_numpy())

    out.to_csv(out_path, index=False)

    print("============================================================")
    print("Guardrailed residual export complete")
    print(f"n = {len(out)}")
    print(f"features = {len(names)}")
    print(f"wrote: {out_path}")
    print("============================================================")


if __name__ == "__main__":
    main()