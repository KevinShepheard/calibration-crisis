#!/usr/bin/env python3
"""
Test 1 — Baseline residual structure and covariate association.

Computes an ordering-dependent structure statistic on redshift-ordered
supernova residuals and evaluates rank correlations with selected
population covariates.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import entropy, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent  # repository root when script sits at top level
DATA = PROJECT_ROOT / "data/pantheonplus/distance_moduli/Pantheon_SH0ES.dat"


def load():
    return pd.read_csv(DATA, sep=r"\s+", comment="#")


def structure_stat(residuals, bins=20):
    hist, _ = np.histogram(residuals, bins=bins, density=True)
    hist = hist[hist > 0]
    return entropy(hist)


def main():
    df = load()
    df = df[np.isfinite(df["MU_SH0ES"]) & np.isfinite(df["zHD"])]

    # Explicit NumPy casting to ensure consistent downstream numerical behavior
    mu = np.asarray(df["MU_SH0ES"], dtype=float)
    z = np.asarray(df["zHD"], dtype=float)

    resid = mu - np.mean(mu)

    order = np.argsort(z)
    resid_ord = resid[order]

    print("STRUCTURE STATISTICS")
    print(f"  baseline_structure: {structure_stat(resid_ord):.6f}")

    # Survey-conditional structure estimates
    survey_stats = []
    for _, g in df.groupby("IDSURVEY"):
        if len(g) < 30:
            continue
        mu_g = np.asarray(g["MU_SH0ES"], dtype=float)
        r_g = mu_g - np.mean(mu_g)
        survey_stats.append(structure_stat(np.sort(r_g)))

    if survey_stats:
        print(f"  cond_IDSURVEY_mean: {np.mean(survey_stats):.6f}")

    # Rank correlations with population covariates
    print("\nCORRELATIONS")
    for key in ("x1", "c", "HOST_LOGMASS"):
        col = np.asarray(df[key], dtype=float)
        mask = np.isfinite(col)

        if mask.sum() < 30:
            continue

        rho, p = spearmanr(col[mask], resid[mask])
        print(f"  {key}: rho={rho:.4f}, p={p:.3e}")


if __name__ == "__main__":
    main()
