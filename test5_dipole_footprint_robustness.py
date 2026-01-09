#!/usr/bin/env python3
"""
Test 5 — Dipole robustness under survey and footprint stratification.

Evaluates residual dipole estimates under:
  - Per-survey fits
  - Survey jackknife removal
  - Matched hemisphere resampling stratified by survey and redshift
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent  # repository root when script sits at top level
SN_DATA = PROJECT_ROOT / "data/pantheonplus/distance_moduli/Pantheon_SH0ES.dat"


@dataclass(frozen=True)
class DipoleFit:
    vec: np.ndarray  # shape (3,)
    amp: float
    ra_deg: float
    dec_deg: float


def _as_float_array(s: pd.Series) -> np.ndarray:
    return np.asarray(s.to_numpy(), dtype=float)


def load_sn_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", comment="#")
    return df


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
    bandwidth = float(max(span, 1e-4))

    delta = (z_grid[:, None] - z_sorted[None, :]) / bandwidth
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
        raise ValueError("zero dipole vector")
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    ra = float(np.rad2deg(np.arctan2(y, x)) % 360.0)
    dec = float(np.rad2deg(np.arcsin(np.clip(z / amp, -1.0, 1.0))))
    return ra, dec


def fit_dipole(x3: np.ndarray, resid: np.ndarray, weights: np.ndarray | None = None) -> DipoleFit:
    x3 = np.asarray(x3, dtype=float)
    y = np.asarray(resid, dtype=float)

    if weights is not None:
        w = np.asarray(weights, dtype=float)
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        sw = np.sqrt(w)
        A = x3 * sw[:, None]
        b = y * sw
    else:
        A = x3
        b = y

    vec, *_ = np.linalg.lstsq(A, b, rcond=None)
    vec = np.asarray(vec, dtype=float)
    amp = float(np.linalg.norm(vec))
    ra, dec = vector_to_ra_dec(vec) if amp > 0 else (float("nan"), float("nan"))
    return DipoleFit(vec=vec, amp=amp, ra_deg=ra, dec_deg=dec)


def isotropic_null_pvalue(x3: np.ndarray, resid: np.ndarray, nperm: int, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    x3 = np.asarray(x3, dtype=float)
    y = np.asarray(resid, dtype=float)

    base = fit_dipole(x3, y).amp
    null_amps: List[float] = []
    for _ in range(int(nperm)):
        yp = rng.permutation(y)
        vecp, *_ = np.linalg.lstsq(x3, yp, rcond=None)
        null_amps.append(float(np.linalg.norm(vecp)))
    null = np.asarray(null_amps, dtype=float)
    return float(np.mean(null >= base))


def zbin_index(z: np.ndarray, width: float) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    w = float(max(width, 1e-6))
    return np.floor(z / w).astype(int)


def matched_hemisphere_sample(
    rng: np.random.Generator,
    side_a: np.ndarray,
    side_b: np.ndarray,
    survey: np.ndarray,
    zbin: np.ndarray,
) -> np.ndarray:
    """
    Return an index mask selecting a matched subsample from both sides
    with equal counts per (survey,zbin) stratum.

    Output mask selects the union of matched samples from A and B.
    """
    side_a = np.asarray(side_a, dtype=bool)
    side_b = np.asarray(side_b, dtype=bool)
    survey = np.asarray(survey, dtype=int)
    zbin = np.asarray(zbin, dtype=int)

    strata_keys = np.unique(np.column_stack([survey, zbin]), axis=0)
    keep = np.zeros_like(side_a, dtype=bool)

    for s, zb in strata_keys:
        in_stratum = (survey == s) & (zbin == zb)
        idx_a = np.flatnonzero(in_stratum & side_a)
        idx_b = np.flatnonzero(in_stratum & side_b)
        if idx_a.size == 0 or idx_b.size == 0:
            continue
        k = int(min(idx_a.size, idx_b.size))
        take_a = rng.choice(idx_a, size=k, replace=False)
        take_b = rng.choice(idx_b, size=k, replace=False)
        keep[take_a] = True
        keep[take_b] = True

    return keep


def meta_combine_directions(unit_dirs: np.ndarray, weights: np.ndarray) -> DipoleFit:
    unit_dirs = np.asarray(unit_dirs, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    if unit_dirs.size == 0 or float(np.sum(weights)) <= 0:
        return DipoleFit(vec=np.zeros(3, dtype=float), amp=0.0, ra_deg=float("nan"), dec_deg=float("nan"))

    v = (unit_dirs * weights[:, None]).sum(axis=0)
    amp = float(np.linalg.norm(v))
    if amp <= 0:
        return DipoleFit(vec=v, amp=0.0, ra_deg=float("nan"), dec_deg=float("nan"))
    ra, dec = vector_to_ra_dec(v)
    return DipoleFit(vec=v, amp=amp, ra_deg=ra, dec_deg=dec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--span", type=float, default=0.035)
    ap.add_argument("--zbin", type=float, default=0.02, help="z-bin width used for matched hemispheres")
    ap.add_argument("--nperm", type=int, default=2000, help="permutation count for isotropic null")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-survey-n", type=int, default=50)
    args = ap.parse_args()

    df = load_sn_table(SN_DATA)
    required = ["MU_SH0ES", "RA", "DEC", "zCMB", "x1", "c", "IDSURVEY"]
    mask = np.ones(len(df), dtype=bool)
    for col in required:
        mask &= np.isfinite(df[col].to_numpy(dtype=float)) if col != "IDSURVEY" else df[col].notna().to_numpy()

    df = df.loc[mask].copy()
    if len(df) == 0:
        raise SystemExit("No valid rows after filtering required columns.")

    mu = _as_float_array(df["MU_SH0ES"])
    z = _as_float_array(df["zCMB"])
    ra = _as_float_array(df["RA"])
    dec = _as_float_array(df["DEC"])
    x1 = _as_float_array(df["x1"])
    c = _as_float_array(df["c"])
    survey = np.asarray(df["IDSURVEY"].to_numpy(), dtype=int)

    mu_hat = smooth_baseline(z, mu, span=float(args.span))
    resid = mu - mu_hat
    resid_clean = remove_salt2_leakage(resid, x1, c)

    x3 = unit_sky_vectors(ra, dec)
    base = fit_dipole(x3, resid_clean)
    p_iso = isotropic_null_pvalue(x3, resid_clean, nperm=int(args.nperm), seed=int(args.seed))

    print("TEST 5 — DIPLOE ADVERSARIAL FOOTPRINT SUITE")
    print("  Residual = MU_SH0ES - smooth(mu(z)); SALT2 x1/c leakage removed")
    print(f"  n={len(resid_clean)}  base_amp={base.amp:.6f}  base_RA={base.ra_deg:.2f}  base_Dec={base.dec_deg:.2f}  p_iso={p_iso:.6g}")

    # 5.1 per-survey dipoles + meta-combine
    print("  5.1 Per-survey dipoles (n >= min-survey-n) and meta-combine:")
    names, counts = np.unique(survey, return_counts=True)
    order = np.argsort(-counts)

    per_dirs: List[np.ndarray] = []
    per_w: List[float] = []
    per_rows: List[Tuple[int, int, float, float, float]] = []

    for i in order:
        s = int(names[i])
        n = int(counts[i])
        if n < int(args.min_survey_n):
            continue
        m = (survey == s)
        fit = fit_dipole(x3[m], resid_clean[m])
        if fit.amp > 0:
            u = fit.vec / float(fit.amp)
            per_dirs.append(u)
            per_w.append(float(np.sqrt(n)))
        per_rows.append((s, n, fit.amp, fit.ra_deg, fit.dec_deg))

    for s, n, amp, ra_s, dec_s in per_rows[:12]:
        print(f"    IDSURVEY {s:>4d}: n={n:<4d} amp={amp:>7.3f}  RA={ra_s:>6.1f}  Dec={dec_s:>6.1f}")

    meta = meta_combine_directions(np.vstack(per_dirs) if per_dirs else np.empty((0, 3)), np.asarray(per_w, dtype=float))
    print(f"    META (sqrt(n) weights): amp={meta.amp:.6f}  RA={meta.ra_deg:.2f}  Dec={meta.dec_deg:.2f}")

    # 5.2 jackknife by survey
    print("  5.2 Jackknife-by-survey (drop one IDSURVEY at a time):")
    jack_angles: List[float] = []
    base_u = base.vec / base.amp if base.amp > 0 else np.zeros(3)

    for i in order[: min(len(order), 20)]:
        s = int(names[i])
        m = survey != s
        if int(np.sum(m)) < 50:
            continue
        fit = fit_dipole(x3[m], resid_clean[m])
        if fit.amp <= 0 or base.amp <= 0:
            continue
        u = fit.vec / fit.amp
        cosang = float(np.clip(np.dot(base_u, u), -1.0, 1.0))
        ang = float(np.rad2deg(np.arccos(cosang)))
        jack_angles.append(ang)
        print(f"    drop IDSURVEY {s:>4d}: n={int(np.sum(m)):<4d} amp={fit.amp:>7.3f} RA={fit.ra_deg:>6.1f} Dec={fit.dec_deg:>6.1f}  Δθ={ang:>6.1f}°")

    if jack_angles:
        a = np.asarray(jack_angles, dtype=float)
        print(f"    jackknife Δθ stats: median={np.median(a):.2f}°  90%={np.percentile(a, 90):.2f}°  max={np.max(a):.2f}°")
    else:
        print("    <no jackknife results produced>")

    # 5.3 matched hemisphere resampling across candidate axes
    print("  5.3 Matched-hemisphere resampling (match IDSURVEY × z-bin):")
    rng = np.random.default_rng(int(args.seed))
    zb = zbin_index(z, float(args.zbin))

    # Candidate axes: fitted dipole direction and random comparison directions
    axes: List[np.ndarray] = []
    if base.amp > 0:
        axes.append(base.vec / base.amp)
    for _ in range(24):
        v = rng.normal(size=3)
        nv = float(np.linalg.norm(v))
        axes.append(v / nv if nv > 0 else np.array([1.0, 0.0, 0.0]))

    amps: List[float] = []
    for j, axis in enumerate(axes):
        dot = x3 @ axis
        side_a = dot >= 0
        side_b = ~side_a
        keep = matched_hemisphere_sample(rng, side_a, side_b, survey, zb)
        if int(np.sum(keep)) < 100:
            continue
        fit = fit_dipole(x3[keep], resid_clean[keep])
        amps.append(fit.amp)
        if j == 0:
            print(f"    axis=base: matched_n={int(np.sum(keep))} amp={fit.amp:.6f} RA={fit.ra_deg:.2f} Dec={fit.dec_deg:.2f}")
    if amps:
        aa = np.asarray(amps, dtype=float)
        print(f"    matched-axes amp stats (incl base): median={np.median(aa):.4f}  90%={np.percentile(aa, 90):.4f}  max={np.max(aa):.4f}")
    else:
        print("    <no matched hemisphere fits produced>")


if __name__ == "__main__":
    main()
