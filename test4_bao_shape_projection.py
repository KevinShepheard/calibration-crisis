#!/usr/bin/env python3
"""
Test 4 — BAO-only (w0, wa) projection onto cleaned SN residuals.

Evaluates rank correlation between residual structure and BAO-consistent
distance-modulus shape variations without using an H0 anchor.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent  # repository root when script sits at top level

SN_DATA = PROJECT_ROOT / "data/pantheonplus/distance_moduli/Pantheon_SH0ES.dat"
CHAIN_DIR = PROJECT_ROOT / "data/bao-all"

_C_LIGHT_KMS = 299792.458


# =========================
# Pylance-safe statistics
# =========================
def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
	x = np.asarray(x, dtype=float)
	y = np.asarray(y, dtype=float)

	rx = x.argsort().argsort().astype(float)
	ry = y.argsort().argsort().astype(float)

	rx -= rx.mean()
	ry -= ry.mean()

	den = np.sqrt(np.sum(rx**2) * np.sum(ry**2))
	if den == 0:
		return float("nan")
	return float(np.sum(rx * ry) / den)

# SN preprocessing
def smooth_baseline(z: np.ndarray, y: np.ndarray, span: float = 0.035, grid_points: int = 320) -> np.ndarray:
	z = np.asarray(z, dtype=float)
	y = np.asarray(y, dtype=float)

	if z.size == 0:
		return y.copy()

	idx = np.argsort(z)
	zs = z[idx]
	ys = y[idx]

	if np.unique(zs).size < 2:
		return np.full_like(y, float(np.mean(y)))

	grid_points = int(min(grid_points, zs.size))
	z_grid = np.linspace(float(zs[0]), float(zs[-1]), grid_points)

	bw = max(float(span), 1e-4)
	delta = (z_grid[:, None] - zs[None, :]) / bw
	w = np.exp(-0.5 * delta**2)
	ws = np.sum(w, axis=1)
	ws = np.where(ws == 0.0, 1.0, ws)

	y_grid = (w @ ys) / ws
	return np.interp(z, z_grid, y_grid)


def remove_salt2_leakage(resid: np.ndarray, x1: np.ndarray, c: np.ndarray) -> np.ndarray:
	resid = np.asarray(resid, dtype=float)
	x1 = np.asarray(x1, dtype=float)
	c = np.asarray(c, dtype=float)

	X = np.column_stack([np.ones_like(x1), x1, c])
	coef, *_ = np.linalg.lstsq(X, resid, rcond=None)
	return resid - X @ coef


def load_sn_clean() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	df = pd.read_csv(SN_DATA, sep=r"\s+", comment="#")

	mask = (
		np.isfinite(df["MU_SH0ES"])
		& np.isfinite(df["zHD"])
		& np.isfinite(df["x1"])
		& np.isfinite(df["c"])
		& np.isfinite(df["IDSURVEY"])
		& (df["IS_CALIBRATOR"] == 0)
	)

	if "USED_IN_SH0ES_HF" in df.columns:
		mask &= np.isfinite(df["USED_IN_SH0ES_HF"])
		mask &= (df["USED_IN_SH0ES_HF"] == 0)

	df = df[mask].copy()
	if len(df) < 200:
		raise SystemExit(f"Too few SN after masking: n={len(df)}")

	mu = df["MU_SH0ES"].to_numpy(dtype=float)
	z = df["zHD"].to_numpy(dtype=float)
	x1 = df["x1"].to_numpy(dtype=float)
	c = df["c"].to_numpy(dtype=float)
	survey = df["IDSURVEY"].astype(int).to_numpy()

	mu_hat = smooth_baseline(z, mu)
	resid = mu - mu_hat
	resid_clean = remove_salt2_leakage(resid, x1, c)

	return z, resid_clean, survey


# CPL distance–redshift parameterization
def Ez_cpl(z: np.ndarray, omegam: float, w0: float, wa: float) -> np.ndarray:
	z = np.asarray(z, dtype=float)
	ode = 1.0 - float(omegam)
	a = 1.0 / (1.0 + z)
	fde = (a ** (-3.0 * (1.0 + float(w0) + float(wa)))) * np.exp(3.0 * float(wa) * (a - 1.0))
	return np.sqrt(float(omegam) * (1.0 + z) ** 3 + ode * fde)


def mu_theory(z: np.ndarray, omegam: float, w0: float, wa: float, h: float) -> np.ndarray:
	z = np.asarray(z, dtype=float)
	zmax = float(np.max(z))
	if zmax <= 0:
		return np.full_like(z, np.nan, dtype=float)

	zz = np.linspace(0.0, zmax, 4096)
	Ez = Ez_cpl(zz, omegam=float(omegam), w0=float(w0), wa=float(wa))
	invE = 1.0 / Ez

	dz = float(zz[1] - zz[0])
	chi = np.cumsum((invE[:-1] + invE[1:]) * dz * 0.5)
	chi = np.concatenate([[0.0], chi])

	H0 = 100.0 * float(h)
	dc = (_C_LIGHT_KMS / H0) * np.interp(z, zz, chi)
	dl = (1.0 + z) * dc
	return 5.0 * np.log10(dl) + 25.0


# Blocked permutation null construction
def blocked_perm_pvalue(
	x: np.ndarray,
	y: np.ndarray,
	survey: np.ndarray,
	z: np.ndarray,
	nperm: int = 3000,
	z_bins: int = 10,
	seed: int = 0,
) -> float:
	rng = np.random.default_rng(seed)

	x = np.asarray(x, dtype=float)
	y = np.asarray(y, dtype=float)
	z = np.asarray(z, dtype=float)
	survey = np.asarray(survey)

	rho_obs = spearman_rho(x, y)
	if not np.isfinite(rho_obs):
		return float("nan")

	edges = np.quantile(z, np.linspace(0.0, 1.0, z_bins + 1))
	edges[0] = -np.inf
	edges[-1] = np.inf
	zbin = np.digitize(z, edges[1:-1], right=False)

	extreme = 0
	for _ in range(nperm):
		y_perm = y.copy()
		for s in np.unique(survey):
			ms = (survey == s)
			if not np.any(ms):
				continue
			for b in range(z_bins):
				idx = np.where(ms & (zbin == b))[0]
				if idx.size > 1:
					y_perm[idx] = y_perm[rng.permutation(idx)]

		rho_p = spearman_rho(x, y_perm)
		if np.isfinite(rho_p) and abs(rho_p) >= abs(rho_obs):
			extreme += 1

	return float((extreme + 1) / (nperm + 1))

# Chain loader (explicit cols)
CHAIN_COLS = [
	"weight", "minuslogpost", "w", "wa", "hrdrag", "omm", "As", "omch2",
	"omegam", "omegamh2", "omegal", "zrei", "YHe", "Y_p", "DHBBN", "A",
	"clamp", "age", "rdrag", "zdrag", "H0rdrag",
	"chi2__BAO", "minuslogprior", "minuslogprior__0", "chi2",
	"chi2__desi_y1_cosmo_bindings.cobaya_likelihoods.desi_bao_all",
]


def load_chain_stack() -> np.ndarray:
	paths = sorted(glob.glob(str(CHAIN_DIR / "chain.*.txt")))
	if not paths:
		raise FileNotFoundError(f"No chain.*.txt found in {CHAIN_DIR}")
	mats = []
	for p in paths:
		m = np.loadtxt(p)
		if m.ndim == 1:
			m = m.reshape(1, -1)
		mats.append(np.asarray(m, dtype=float))
	return np.vstack(mats)

# Main
def main() -> None:
	z, resid_clean, survey = load_sn_clean()
	chain = load_chain_stack()

	if chain.shape[1] != len(CHAIN_COLS):
		raise SystemExit(f"Chain column mismatch: got {chain.shape[1]}, expected {len(CHAIN_COLS)}")

	col = {name: i for i, name in enumerate(CHAIN_COLS)}
	w0 = chain[:, col["w"]]
	wa = chain[:, col["wa"]]
	omegam = chain[:, col["omegam"]]
	omegamh2 = chain[:, col["omegamh2"]]

	h = np.sqrt(np.where(omegam > 0, omegamh2 / omegam, np.nan))

	valid = np.isfinite(w0) & np.isfinite(wa) & np.isfinite(omegam) & np.isfinite(h)
	valid &= (omegam > 0.05) & (omegam < 0.6) & (h > 0.3) & (h < 1.2)

	idx = np.where(valid)[0]
	if idx.size < 500:
		raise SystemExit(f"Too few valid draws after filtering: n={idx.size}")

	rng = np.random.default_rng(0)
	draws = rng.choice(idx, size=min(800, idx.size), replace=False)
	draws = draws.astype(int)  # ensure Python-int indexing behavior downstream

	# Reference draw: median parameters
	w0_ref = float(np.median(w0[draws]))
	wa_ref = float(np.median(wa[draws]))
	om_ref = float(np.median(omegam[draws]))
	h_ref = float(np.median(h[draws]))

	mu_ref = mu_theory(z, omegam=om_ref, w0=w0_ref, wa=wa_ref, h=h_ref)

	rhos = np.empty(draws.size, dtype=float)
	best_rho = 0.0
	best_i: int = int(draws[0])

	for k, i in enumerate(draws):
		ii = int(i)  # enforce scalar index
		mui = mu_theory(
			z,
			omegam=float(omegam[ii]),
			w0=float(w0[ii]),
			wa=float(wa[ii]),
			h=float(h[ii]),
		)
		dmu = mui - mu_ref
		rho = spearman_rho(resid_clean, dmu)
		rhos[k] = rho
		if np.isfinite(rho) and abs(rho) > abs(best_rho):
			best_rho = float(rho)
			best_i = ii

	# Blocked null on best draw
	mu_best = mu_theory(
		z,
		omegam=float(omegam[best_i]),
		w0=float(w0[best_i]),
		wa=float(wa[best_i]),
		h=float(h[best_i]),
	)
	p_block = blocked_perm_pvalue(
		x=resid_clean,
		y=mu_best - mu_ref,
		survey=survey,
		z=z,
		nperm=3000,
		z_bins=10,
		seed=0,
	)

	print("TEST 4 — BAO-ONLY w0–wa PROJECTION")
	print("  Residual = MU_SH0ES - smooth(mu(z)); SALT2 x1/c leakage removed; calibrators and SH0ES HF removed")
	print(f"  SN used: n={z.size}")
	print(f"  Chain dir: {CHAIN_DIR}")
	print(f"  Valid draws: n={idx.size}  evaluated: n={draws.size}")
	print("  Coupling distribution over draws: rho(resid_clean, Δmu_draw(z) - Δmu_ref(z))")
	print(f"    mean rho: {float(np.nanmean(rhos)):.6f}")
	print(f"    std  rho: {float(np.nanstd(rhos)):.6f}")
	print(f"    max |rho|: {float(np.nanmax(np.abs(rhos))):.6f}")
	print("  Most explanatory draw (by |rho|):")
	print(f"    idx={best_i}  rho={best_rho:.6f}")
	print(f"    omegam={float(omegam[best_i]):.4f}  h={float(h[best_i]):.4f}  w0={float(w0[best_i]):.4f}  wa={float(wa[best_i]):.4f}")
	print("  Blocked-permutation null (IDSURVEY × z-bin shuffle) on best draw:")
	print(f"    p_block={p_block:.6f}  (nperm=3000)")


if __name__ == "__main__":
	main()
