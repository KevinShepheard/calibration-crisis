#!/usr/bin/env python3
"""
guardrail_operator_viz.py

Operator-level visualization for admissible correction structure in
Type Ia supernova residuals.

This script generates the single figure used in the accompanying paper.
It is designed to make the geometry of the inference explicit rather than
to optimize aesthetic presentation.

The visualization has two components:

1) Operator phase portrait
   A scatter of the learned admissible correction (Δ̂_corr) against the
   cleaned residuals (Δ_clean). Alignment along a tilted one-dimensional
   manifold indicates that the dominant residual variance lies within the
   admissible correction subspace, rather than in an orthogonal physical
   component.

2) Structure-collapse curve
   The ordering-dependent structure metric S(·) evaluated on
   Δ_clean − λ Δ̂_corr for λ ∈ [0, 1]. Rapid collapse at small λ indicates
   that the observed redshift-ordered structure is largely attributable
   to admissible (selection-calibration) components.

Key design choices:

- The baseline μ(z) is removed using a non-parametric Gaussian kernel
  smoother. No cosmological model is assumed.
- SALT2 population leakage (x1, c) is projected out before visualization.
- The admissible correction is learned from survey, footprint, bias,
  and error-proxy features only.
- The structure metric is order-sensitive by construction and probes
  coherence along redshift, not variance reduction.

This script is deterministic, self-contained, and produces the figure
byte-for-byte as used in the paper.

Run:
  python scripts/guardrail_operator_viz.py \
    --sn data/pantheonplus/distance_moduli/Pantheon_SH0ES.dat \
    --out out/guardrail_operator_viz.png

Dependencies:
  numpy, pandas, matplotlib
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# -----------------------------
# Utilities
# -----------------------------

def _as_float(df: pd.DataFrame, col: str) -> np.ndarray:
    return np.asarray(df[col].to_numpy(), dtype=float)


def smooth_baseline(z: np.ndarray, mu: np.ndarray, span: float = 0.035, grid_points: int = 320) -> np.ndarray:
    """
    Gaussian-kernel smoothing on a z-grid, then interpolate back to each z.
    Pure baseline; not cosmology.
    """
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
    """
    Linear projection resid ~ [x1, c]. Remove fitted component.
    """
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


def structure_metric(y: np.ndarray, z: np.ndarray, window: int = 41) -> float:
    """
    Your Test8-style ordering-dependent metric:
      sort by z, subtract running-median baseline, normalize by MAD, then mean |residual|.
    """
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    m = np.isfinite(y) & np.isfinite(z)
    y = y[m]
    z = z[m]

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


def build_design_matrix(df: pd.DataFrame, use_survey_fe: bool = True) -> Tuple[np.ndarray, List[str]]:
    """
    Same admissible feature class as your Test 8 script:
      - x1, c
      - bias fields if present
      - error proxies if present
      - footprint basis: sin/cos RA/DEC
      - survey fixed effects (drop one)
    """
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
        if uniq.size > 1:
            base = int(uniq[0])
            for u in uniq[1:]:
                blocks.append((s == int(u)).astype(float)[:, None])
                cols.append(f"FE_IDSURVEY_{int(u)}")

    # if nothing, intercept-only
    if not blocks:
        X = np.ones((len(df), 1), dtype=float)
        return X, ["intercept_only"]

    X = np.hstack(blocks)
    X = np.hstack([np.ones((len(df), 1), dtype=float), X])
    return X, ["intercept"] + cols


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sn", type=str, required=True, help="Path to Pantheon_SH0ES.dat (abs or relative to PROJECT_ROOT)")
    ap.add_argument("--out", type=str, default="out/guardrail_operator_viz.png", help="Output PNG path")
    ap.add_argument("--span", type=float, default=0.035, help="baseline smoothing span")
    ap.add_argument("--window", type=int, default=41, help="structure metric window")
    ap.add_argument("--alpha", type=float, default=0.35, help="scatter alpha")
    ap.add_argument("--seed", type=int, default=0, help="(reserved) deterministic hooks")
    args = ap.parse_args()

    sn_path = (PROJECT_ROOT / args.sn).resolve() if not Path(args.sn).is_absolute() else Path(args.sn).resolve()
    out_path = (PROJECT_ROOT / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(sn_path, sep=r"\s+", comment="#")

    # minimal required columns for this visualization
    required = ["MU_SH0ES", "zCMB", "x1", "c", "RA", "DEC", "IDSURVEY"]
    mask = np.ones(len(df), dtype=bool)
    for col in required:
        if col == "IDSURVEY":
            mask &= df[col].notna().to_numpy()
        else:
            mask &= np.isfinite(df[col].to_numpy(dtype=float))
    df = df.loc[mask].copy()

    if len(df) < 50:
        raise SystemExit(f"Too few valid rows after filtering: n={len(df)}")

    mu = _as_float(df, "MU_SH0ES")
    z = _as_float(df, "zCMB")
    x1 = _as_float(df, "x1")
    c = _as_float(df, "c")

    # Δ_clean
    mu_hat = smooth_baseline(z, mu, span=float(args.span))
    resid = mu - mu_hat
    delta_clean = remove_salt2_leakage(resid, x1, c)

    # admissible correction: Δ_hat_corr = X beta (fit to Δ_clean)
    X, names = build_design_matrix(df, use_survey_fe=True)
    # robustly fit on finite rows
    m = np.all(np.isfinite(X), axis=1) & np.isfinite(delta_clean)
    beta, *_ = np.linalg.lstsq(X[m], delta_clean[m], rcond=None)
    delta_hat_corr = np.full_like(delta_clean, np.nan, dtype=float)
    delta_hat_corr[m] = (X[m] @ beta)

    # post-correction
    delta_post = delta_clean - delta_hat_corr

    # structure collapse curve S(Δ_clean - λ Δ_hat_corr)
    lambdas = np.linspace(0.0, 1.0, 101)
    S = np.array([structure_metric(delta_clean - lam * delta_hat_corr, z, window=int(args.window)) for lam in lambdas])

    # reference structure values
    S0 = structure_metric(delta_clean, z, window=int(args.window))
    S1 = structure_metric(delta_post, z, window=int(args.window))

    # -----------------------------
    # Plot: 2-panel figure
    # -----------------------------
    plt.figure(figsize=(8.4, 7.2))

    # Panel A: phase portrait y vs yhat
    ax1 = plt.subplot(2, 1, 1)
    mm = np.isfinite(delta_clean) & np.isfinite(delta_hat_corr)
    ax1.scatter(delta_hat_corr[mm], delta_clean[mm], s=12, alpha=float(args.alpha))
    # identity line y=x
    lo = np.nanpercentile(np.hstack([delta_hat_corr[mm], delta_clean[mm]]), 1.0)
    hi = np.nanpercentile(np.hstack([delta_hat_corr[mm], delta_clean[mm]]), 99.0)
    ax1.plot([lo, hi], [lo, hi], linewidth=1)
    ax1.axhline(0.0, linewidth=1)
    ax1.axvline(0.0, linewidth=1)
    ax1.set_title("Operator phase portrait: admissible prediction vs cleaned residual")
    ax1.set_xlabel(r"$\widehat{\Delta}_{\rm corr}$  (admissible correction)  [mag]")
    ax1.set_ylabel(r"$\Delta_{\rm clean}$  (baseline + SALT2 leakage removed)  [mag]")

    # Panel B: structure collapse curve
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(lambdas, S, linewidth=2)
    ax2.axvline(0.0, linewidth=1)
    ax2.axvline(1.0, linewidth=1)
    ax2.axhline(S0, linewidth=1)
    ax2.axhline(S1, linewidth=1)
    ax2.set_title("Structure-collapse under admissible correction")
    ax2.set_xlabel(r"blend $\lambda$ in  $\Delta_{\rm clean} - \lambda\,\widehat{\Delta}_{\rm corr}$")
    ax2.set_ylabel(r"ordering-dependent structure metric  $S(\cdot)$")
    ax2.text(
        0.98, 0.02,
        f"S(λ=0) = {S0:.3f}\n"
        f"S(λ=1) = {S1:.3f}\n"
        f"features = {len(names)}",
        transform=ax2.transAxes,
        va="bottom",
        ha="right",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="none",
            alpha=0.85
        )
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()

    print("============================================================")
    print("guardrail_operator_viz.py COMPLETE")
    print(f"n={len(df)}  features={len(names)}")
    print(f"wrote: {out_path}")
    print("============================================================")


if __name__ == "__main__":
    main()