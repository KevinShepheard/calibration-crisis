#!/usr/bin/env python3
"""
Test 6 — Residual structure and population covariate association.

Computes an ordering-dependent structure metric on cleaned residuals
and evaluates rank correlations with SALT2 population parameters.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parent  # repository root when script sits at top level
SN_DATA = PROJECT_ROOT / "data/pantheonplus/distance_moduli/Pantheon_SH0ES.dat"


def smooth_baseline(z: np.ndarray, mu: np.ndarray, span: float = 0.035) -> np.ndarray:
    z = z.astype(float)
    mu = mu.astype(float)
    idx = np.argsort(z)
    z_s = z[idx]
    mu_s = mu[idx]

    if np.unique(z_s).size < 2:
        return np.full_like(mu, float(mu.mean()))

    z_grid = np.linspace(z_s[0], z_s[-1], min(320, z_s.size))
    bw = max(span, 1e-4)

    delta = (z_grid[:, None] - z_s[None, :]) / bw
    w = np.exp(-0.5 * delta**2)
    wsum = np.sum(w, axis=1)
    wsum[wsum == 0] = 1.0
    mu_grid = (w @ mu_s) / wsum
    return np.interp(z, z_grid, mu_grid)


def remove_salt2_leakage(resid: np.ndarray, x1: np.ndarray, c: np.ndarray) -> np.ndarray:
    X = np.column_stack([x1, c])
    if X.shape[0] < 2:
        return resid.copy()
    beta, *_ = np.linalg.lstsq(X, resid, rcond=None)
    return resid - X @ beta


def structure_metric(y: np.ndarray, z: np.ndarray, window: int = 41) -> float:
    idx = np.argsort(z)
    y_s = y[idx]

    w = max(9, window | 1)
    half = w // 2

    med = np.empty_like(y_s)
    for i in range(y_s.size):
        lo = max(0, i - half)
        hi = min(y_s.size, i + half + 1)
        med[i] = float(np.median(y_s[lo:hi]))

    r = y_s - med
    mad = float(np.median(np.abs(r - np.median(r))))
    if mad <= 0:
        mad = float(np.std(r) + 1e-9)

    return float(np.mean(np.abs(r)) / mad)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--span", type=float, default=0.035)
    ap.add_argument("--window", type=int, default=41)
    args = ap.parse_args()

    df = pd.read_csv(SN_DATA, sep=r"\s+", comment="#")

    mask = (
        np.isfinite(df["MU_SH0ES"])
        & np.isfinite(df["zCMB"])
        & np.isfinite(df["x1"])
        & np.isfinite(df["c"])
        & df["IDSURVEY"].notna()
    )
    df = df.loc[mask].copy()

    mu = df["MU_SH0ES"].to_numpy(float)
    z = df["zCMB"].to_numpy(float)
    x1 = df["x1"].to_numpy(float)
    c = df["c"].to_numpy(float)

    mu_hat = smooth_baseline(z, mu, span=args.span)
    resid = mu - mu_hat
    resid_clean = remove_salt2_leakage(resid, x1, c)

    base_struct = structure_metric(resid_clean, z, window=args.window)

    r_x1, p_x1 = spearmanr(resid_clean, x1)
    r_c, p_c = spearmanr(resid_clean, c)

    print("TEST 6 — STRUCTURE METRIC AND POPULATION ASSOCIATIONS")
    print(f"  n = {resid_clean.size}")
    print(f"  structure_metric = {base_struct:.6f}")
    print(f"  resid vs x1: rho={r_x1:.6f}, p={p_x1:.3e}")
    print(f"  resid vs c : rho={r_c:.6f}, p={p_c:.3e}")


if __name__ == "__main__":
    main()
