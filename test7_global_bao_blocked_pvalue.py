#!/usr/bin/env python3
"""
Test 7 — Globalized blocked-permutation p-value for BAO parameter search.

Computes a global significance level for the maximum rank correlation
obtained over a set of BAO-consistent (w0, wa) shape projections,
accounting for the search over parameter draws.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# Paths (ONE parent only)

PROJECT_ROOT = Path(__file__).resolve().parent  # repository root when script sits at top level

SN_DATA = PROJECT_ROOT / "data/pantheonplus/distance_moduli/Pantheon_SH0ES.dat"
CHAIN_DIR = PROJECT_ROOT / "data/bao-all"

# Core numerics
def _rank_average(a: np.ndarray) -> np.ndarray:
    """
    Average-rank for ties using pandas (robust + simple).
    Returns ranks in [1..n] as float.
    """
    s = pd.Series(a)
    return s.rank(method="average").to_numpy(dtype=float)


def spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    """
    Spearman correlation via rank transform + Pearson correlation.
    Returns 0.0 if degenerate.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 3:
        return 0.0

    ra = _rank_average(a[m])
    rb = _rank_average(b[m])

    ra = ra - float(np.mean(ra))
    rb = rb - float(np.mean(rb))

    denom = float(np.real(np.sqrt(np.sum(ra * ra) * np.sum(rb * rb))))
    if denom <= 0.0:
        return 0.0

    return float(np.real(np.sum(ra * rb) / denom))


def smooth_baseline(z: np.ndarray, mu: np.ndarray, span: float = 0.035, grid_points: int = 320) -> np.ndarray:
    """
    Gaussian-kernel smooth baseline mu(z). Interpolates back onto original z.
    """
    z = np.asarray(z, dtype=float)
    mu = np.asarray(mu, dtype=float)
    if z.size == 0:
        return mu.copy()

    order = np.argsort(z)
    z_s = z[order]
    mu_s = mu[order]

    if np.unique(z_s).size < 2:
        return np.full_like(mu, float(np.mean(mu)))

    grid_points = int(min(grid_points, z_s.size))
    z_grid = np.linspace(float(z_s[0]), float(z_s[-1]), grid_points)
    bw = max(float(span), 1e-6)

    delta = (z_grid[:, None] - z_s[None, :]) / bw
    w = np.exp(-0.5 * delta * delta)
    ws = np.sum(w, axis=1)
    ws = np.where(ws == 0.0, 1.0, ws)
    mu_grid = (w @ mu_s) / ws

    return np.interp(z, z_grid, mu_grid)


def remove_salt2_leakage(resid: np.ndarray, x1: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Project out linear dependence on (x1, c).
    """
    resid = np.asarray(resid, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    c = np.asarray(c, dtype=float)

    X = np.column_stack([x1, c])
    beta, *_ = np.linalg.lstsq(X, resid, rcond=None)
    return resid - (X @ beta)


def mu_shape_proxy(z: np.ndarray, w0: float, wa: float, omegam: float) -> np.ndarray:
    """
    Pure shape proxy (not a physical mu_theory). We only need a smooth 1D family over z.
    """
    z = np.asarray(z, dtype=float)
    return (w0 * z) + (wa * z / (1.0 + z)) + (omegam * z * z)


def blocked_permutation(
    resid: np.ndarray,
    idsurvey: np.ndarray,
    zbin: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Shuffle residuals within each (IDSURVEY × z-bin) block.
    Pylance-safe, NumPy-safe (no object-dtype unique).
    """
    resid_perm = resid.copy()

    surveys = np.unique(idsurvey)
    zbins = np.unique(zbin)

    for sid in surveys:
        for zb in zbins:
            m = (idsurvey == sid) & (zbin == zb)
            n = int(np.sum(m))
            if n > 1:
                resid_perm[m] = rng.permutation(resid_perm[m])

    return resid_perm


def as_float(x: object) -> float:
    """Coerce pandas / numpy scalars to a real Python float (Pylance-safe)."""
    v = np.asarray(x)
    if v.size != 1:
        raise ValueError("Expected scalar")
    if np.iscomplexobj(v):
        v = np.real(v)
    return float(v.item())

# IO
def load_sn() -> pd.DataFrame:
    df = pd.read_csv(SN_DATA, sep=r"\s+", comment="#")
    # conservative mask
    m = (
        np.isfinite(df["MU_SH0ES"])
        & np.isfinite(df["zHD"])
        & np.isfinite(df["x1"])
        & np.isfinite(df["c"])
        & (df["IS_CALIBRATOR"] == 0)
        & (df["USED_IN_SH0ES_HF"] == 0)
    )
    return df.loc[m].reset_index(drop=True)


def load_chain_stack(max_draws: int = 800) -> pd.DataFrame:
    files = sorted(CHAIN_DIR.glob("chain.*.txt"))
    if not files:
        raise FileNotFoundError(f"No chain.*.txt found in {CHAIN_DIR}")

    cols = [
        "weight", "minuslogpost", "w", "wa", "hrdrag", "omm", "As",
        "omch2", "omegam", "omegamh2", "omegal", "zrei", "YHe", "Y_p",
        "DHBBN", "A", "clamp", "age", "rdrag", "zdrag", "H0rdrag",
    ]

    frames: list[pd.DataFrame] = []
    for f in files:
        arr = np.loadtxt(f)
        if arr.ndim == 1:
            arr = arr[None, :]
        use = min(arr.shape[1], len(cols))
        df = pd.DataFrame(arr[:, :use], columns=cols[:use])
        frames.append(df)

    chain = pd.concat(frames, ignore_index=True)
    n = int(min(max_draws, len(chain)))
    chain = chain.sample(n=n, random_state=0).reset_index(drop=True)
    return chain


# Main
def main() -> None:
    rng = np.random.default_rng(0)

    sn = load_sn()
    z = sn["zHD"].to_numpy(dtype=float)
    mu = sn["MU_SH0ES"].to_numpy(dtype=float)
    x1 = sn["x1"].to_numpy(dtype=float)
    c = sn["c"].to_numpy(dtype=float)
    idsurvey = sn["IDSURVEY"].astype(str).to_numpy()

    mu_hat = smooth_baseline(z, mu)
    resid = mu - mu_hat
    resid_clean = remove_salt2_leakage(resid, x1, c)

    # z-bins for blocked shuffles
    q = np.quantile(z, [0.25, 0.50, 0.75])
    zbin = np.digitize(z, q)  # 0..3

    chain = load_chain_stack(max_draws=800)

    # Observed statistic: maximum absolute rank correlation over draws
    rho_abs_obs_list: list[float] = []
    for i in range(len(chain)):
        w0 = as_float(chain.loc[i, "w"])
        wa = as_float(chain.loc[i, "wa"])
        om = as_float(chain.loc[i, "omegam"])

        shape = mu_shape_proxy(z, w0=w0, wa=wa, omegam=om)
        r = spearman_rho(resid_clean, shape)
        rho_abs_obs_list.append(abs(r))

    rho_abs_obs = np.asarray(rho_abs_obs_list, dtype=float)
    max_rho_obs = float(np.real(np.max(rho_abs_obs)))

    # Blocked-permutation null distribution of the maximum statistic
    nperm = 1000
    max_rho_null = np.empty(nperm, dtype=float)

    for p in range(nperm):
        resid_perm = blocked_permutation(resid_clean, idsurvey, zbin, rng)

        best = 0.0
        for i in range(len(chain)):
            w0 = as_float(chain.loc[i, "w"])
            wa = as_float(chain.loc[i, "wa"])
            om = as_float(chain.loc[i, "omegam"])

            shape = mu_shape_proxy(z, w0=w0, wa=wa, omegam=om)
            r = spearman_rho(resid_perm, shape)
            ar = abs(r)
            if ar > best:
                best = ar

        max_rho_null[p] = best

    p_global = float(np.mean(max_rho_null >= max_rho_obs))

    print("TEST 7 — GLOBALIZED BLOCKED P-VALUE (w0–wa search)")
    print(f"  Chain dir: {CHAIN_DIR}")
    print(f"  SN used: n={len(sn)}")
    print(f"  Draws evaluated: n={len(chain)}")
    print(f"  Observed max |rho|: {max_rho_obs:.6f}")
    print(f"  Global blocked p-value: {p_global:.6g}")
    print(f"  Null mean(max|rho|): {float(np.real(np.mean(max_rho_null))):.6f}")
    print(f"  Null std (max|rho|): {float(np.real(np.std(max_rho_null))):.6f}")


if __name__ == "__main__":
    main()
