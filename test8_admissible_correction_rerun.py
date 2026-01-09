#!/usr/bin/env python3
"""
Test 8 — Admissible correction construction and re-evaluation.

Builds a linear correction model from survey, population, and footprint
features and re-evaluates structure and dipole diagnostics on corrected
or censored residuals.

Modes:
  --mode correction : linear prediction of residuals from admissible features
  --mode censor     : restrict analysis to strata with sufficient sample support
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent  # repository root when script sits at top level
SN_DATA = PROJECT_ROOT / "data/pantheonplus/distance_moduli/Pantheon_SH0ES.dat"


def _as_float(df: pd.DataFrame, col: str) -> np.ndarray:
    return np.asarray(df[col].to_numpy(), dtype=float)


def smooth_baseline(z: np.ndarray, mu: np.ndarray, span: float = 0.035, grid_points: int = 320) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    mu = np.asarray(mu, dtype=float)
    if z.size == 0:
        return mu.copy()
    idx = np.argsort(z)
    z_sorted = z[idx]
    mu_sorted = mu[idx]
    if np.unique(z_sorted).size < 2:
        return np.full_like(mu, float(np.mean(mu_sorted)))
    grid_points = int(min(grid_points, z_sorted.size))
    z_grid = np.linspace(float(z_sorted[0]), float(z_sorted[-1]), grid_points)
    bw = float(max(span, 1e-4))
    delta = (z_grid[:, None] - z_sorted[None, :]) / bw
    w = np.exp(-0.5 * delta * delta)
    wsum = np.sum(w, axis=1)
    wsum = np.where(wsum == 0.0, 1.0, wsum)
    mu_grid = (w @ mu_sorted) / wsum
    return np.interp(z, z_grid, mu_grid)


def remove_salt2_leakage(resid: np.ndarray, x1: np.ndarray, c: np.ndarray) -> np.ndarray:
    resid = np.asarray(resid, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    c = np.asarray(c, dtype=float)
    X = np.column_stack([x1, c])
    if X.shape[0] < 2:
        return resid.copy()
    beta, *_ = np.linalg.lstsq(X, resid, rcond=None)
    return resid - (X @ beta)


def unit_sky_vectors(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    return np.column_stack([np.cos(ra) * np.cos(dec), np.sin(ra) * np.cos(dec), np.sin(dec)])


def vector_to_ra_dec(vec: np.ndarray) -> Tuple[float, float]:
    v = np.asarray(vec, dtype=float)
    amp = float(np.linalg.norm(v))
    if amp <= 0:
        return float("nan"), float("nan")
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    ra = float(np.rad2deg(np.arctan2(y, x)) % 360.0)
    dec = float(np.rad2deg(np.arcsin(np.clip(z / amp, -1.0, 1.0))))
    return ra, dec


def fit_dipole(x3: np.ndarray, resid: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    x3 = np.asarray(x3, dtype=float)
    y = np.asarray(resid, dtype=float)
    vec, *_ = np.linalg.lstsq(x3, y, rcond=None)
    vec = np.asarray(vec, dtype=float)
    amp = float(np.linalg.norm(vec))
    ra, dec = vector_to_ra_dec(vec)
    return vec, amp, ra, dec


def structure_metric(y: np.ndarray, z: np.ndarray, window: int = 41) -> float:
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    n = y.size
    if n < max(25, window):
        return float("nan")
    idx = np.argsort(z)
    y_s = y[idx]
    w = int(max(9, window | 1))
    half = w // 2
    med = np.empty_like(y_s)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        med[i] = float(np.median(y_s[lo:hi]))
    r = y_s - med
    mad = float(np.median(np.abs(r - np.median(r))))
    mad = mad if mad > 0 else float(np.std(r) + 1e-9)
    return float(np.mean(np.abs(r)) / mad)


def zbin_index(z: np.ndarray, width: float) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    w = float(max(width, 1e-6))
    return np.floor(z / w).astype(int)


def build_design_matrix(df: pd.DataFrame, use_survey_fe: bool = True) -> Tuple[np.ndarray, List[str]]:
    cols: List[str] = []
    blocks: List[np.ndarray] = []

    # baseline continuous features
    for c in ["x1", "c"]:
        if c in df.columns:
            v = np.asarray(df[c].to_numpy(), dtype=float)
            blocks.append(v[:, None])
            cols.append(c)

    # bias fields
    for c in ["biasCor_m_b", "biasCor_m_b_COVSCALE", "biasCor_m_b_COVADD"]:
        if c in df.columns:
            v = np.asarray(df[c].to_numpy(), dtype=float)
            blocks.append(v[:, None])
            cols.append(c)

    # error proxies
    for c in ["m_b_corr_err_DIAG", "m_b_corr_err_VPEC", "zHDERR", "zCMBERR"]:
        if c in df.columns:
            v = np.asarray(df[c].to_numpy(), dtype=float)
            blocks.append(v[:, None])
            cols.append(c)

    # footprint basis
    if "RA" in df.columns and "DEC" in df.columns:
        ra = np.deg2rad(np.asarray(df["RA"].to_numpy(), dtype=float))
        dec = np.deg2rad(np.asarray(df["DEC"].to_numpy(), dtype=float))
        blocks.append(np.sin(ra)[:, None]); cols.append("sinRA")
        blocks.append(np.cos(ra)[:, None]); cols.append("cosRA")
        blocks.append(np.sin(dec)[:, None]); cols.append("sinDEC")
        blocks.append(np.cos(dec)[:, None]); cols.append("cosDEC")

    # survey fixed effects
    if use_survey_fe and "IDSURVEY" in df.columns:
        s = np.asarray(df["IDSURVEY"].to_numpy(), dtype=int)
        uniq = np.unique(s)
        # drop one category to avoid collinearity with intercept
        if uniq.size > 1:
            base = int(uniq[0])
            for u in uniq[1:]:
                colname = f"FE_IDSURVEY_{int(u)}"
                blocks.append((s == int(u)).astype(float)[:, None])
                cols.append(colname)

    if not blocks:
        X = np.ones((len(df), 1), dtype=float)
        return X, ["intercept_only"]

    X = np.hstack(blocks)
    # prepend intercept
    X = np.hstack([np.ones((len(df), 1), dtype=float), X])
    return X, ["intercept"] + cols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--span", type=float, default=0.035)
    ap.add_argument("--window", type=int, default=41)
    ap.add_argument("--mode", choices=["correction", "censor"], default="correction")
    ap.add_argument("--zbin", type=float, default=0.02)
    ap.add_argument("--min-cell", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(SN_DATA, sep=r"\s+", comment="#")

    required = ["MU_SH0ES", "zCMB", "x1", "c", "RA", "DEC", "IDSURVEY"]
    mask = np.ones(len(df), dtype=bool)
    for col in required:
        if col == "IDSURVEY":
            mask &= df[col].notna().to_numpy()
        else:
            mask &= np.isfinite(df[col].to_numpy(dtype=float))

    df = df.loc[mask].copy()
    if len(df) == 0:
        raise SystemExit("No valid rows after filtering required columns.")

    mu = _as_float(df, "MU_SH0ES")
    z = _as_float(df, "zCMB")
    x1 = _as_float(df, "x1")
    c = _as_float(df, "c")
    ra = _as_float(df, "RA")
    dec = _as_float(df, "DEC")
    survey = np.asarray(df["IDSURVEY"].to_numpy(), dtype=int)

    mu_hat = smooth_baseline(z, mu, span=float(args.span))
    resid = mu - mu_hat
    resid_clean = remove_salt2_leakage(resid, x1, c)

    x3 = unit_sky_vectors(ra, dec)
    base_vec, base_amp, base_ra, base_dec = fit_dipole(x3, resid_clean)
    base_struct = structure_metric(resid_clean, z, window=int(args.window))

    rng = np.random.default_rng(int(args.seed))

    if args.mode == "correction":
        X, names = build_design_matrix(df, use_survey_fe=True)
        # fit linear model: resid_clean ~ X
        beta, *_ = np.linalg.lstsq(X, resid_clean, rcond=None)
        pred = X @ beta
        corrected = resid_clean - pred

        vec, amp, ra_d, dec_d = fit_dipole(x3, corrected)
        s = structure_metric(corrected, z, window=int(args.window))

        print("TEST 8 — ADMISSIBLE CORRECTION AND RE-EVALUATION")
        print(f"  n={len(corrected)}  features={len(names)}")
        print(f"  baseline: struct={base_struct:.6f}  dipole_amp={base_amp:.6f}  RA={base_ra:.2f}  Dec={base_dec:.2f}")
        print(f"  corrected: struct={s:.6f}  dipole_amp={amp:.6f}  RA={ra_d:.2f}  Dec={dec_d:.2f}")

    else:
        # censoring: keep only selection-stable cells by (survey,zbin,x1bin,cbin) with min count
        zb = zbin_index(z, float(args.zbin))
        x1b = zbin_index(x1, 0.5)
        cb = zbin_index(c, 0.05)

        key = np.column_stack([survey, zb, x1b, cb])
        keys, counts = np.unique(key, axis=0, return_counts=True)
        keep_cells = {tuple(k): int(cn) for k, cn in zip(keys, counts) if int(cn) >= int(args.min_cell)}

        keep = np.array([tuple(k) in keep_cells for k in key], dtype=bool)
        corrected = resid_clean[keep]
        z_k = z[keep]
        x3_k = x3[keep]

        vec, amp, ra_d, dec_d = fit_dipole(x3_k, corrected)
        s = structure_metric(corrected, z_k, window=int(min(args.window, max(25, len(corrected) // 10) | 1)))

        print("TEST 8 — CENSORING MODEL + RERUN GUARDRAILS")
        print(f"  keep={int(np.sum(keep))}/{len(keep)}  min_cell={int(args.min_cell)}  zbin={float(args.zbin)}")
        print(f"  baseline: struct={base_struct:.6f}  dipole_amp={base_amp:.6f}  RA={base_ra:.2f}  Dec={base_dec:.2f}")
        print(f"  censored: struct={s:.6f}  dipole_amp={amp:.6f}  RA={ra_d:.2f}  Dec={dec_d:.2f}")


if __name__ == "__main__":
    main()
