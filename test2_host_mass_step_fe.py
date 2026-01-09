#!/usr/bin/env python3
"""
Test 2 — Host-mass step under survey-fixed-effects and blocked nulls.

Evaluates a binary host-mass indicator after removal of redshift baseline
and SALT2 (x1, c) leakage, including survey fixed effects and
blocked-permutation significance.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import math

PROJECT_ROOT = Path(__file__).resolve().parent  # repository root when script sits at top level
DATA = PROJECT_ROOT / "data/pantheonplus/distance_moduli/Pantheon_SH0ES.dat"


def smooth_baseline(z, y, span=0.035, grid_points=320):
	if len(z) == 0:
		return np.asarray(y)

	idx = np.argsort(z)
	z_sorted = z[idx]
	y_sorted = y[idx]
	if len(np.unique(z_sorted)) < 2:
		return np.full_like(y, np.mean(y))

	grid_points = min(grid_points, len(z_sorted))
	z_grid = np.linspace(z_sorted[0], z_sorted[-1], grid_points)
	bandwidth = max(span, 1e-4)

	delta = (z_grid[:, None] - z_sorted[None, :]) / bandwidth
	weights = np.exp(-0.5 * delta**2)
	ws = weights.sum(axis=1)
	ws = np.where(ws == 0, 1.0, ws)
	y_grid = (weights @ y_sorted) / ws
	return np.interp(z, z_grid, y_grid)


def remove_salt2_leakage(resid, x1, c):
	# include intercept; avoids forcing leakage fit through origin
	X = np.column_stack([np.ones_like(x1), x1, c])
	coef, *_ = np.linalg.lstsq(X, resid, rcond=None)
	return resid - X @ coef


def fit_step_with_survey_fe(r, step, survey_codes, weights=None):
	"""
	Linear model with survey fixed effects and a binary step term.

	Returns the step coefficient, its standard error, t-statistic,
	and a normal-approximation p-value.
	"""
	n = r.size
	if n < 10:
		return np.nan, np.nan, np.nan, np.nan

	# One-hot encoding of survey labels with reference category dropped
	surv = np.asarray(survey_codes)
	uniq = np.unique(surv)
	if uniq.size < 2:
		# fallback: just intercept + step
		X = np.column_stack([np.ones(n), step])
	else:
		ref = uniq[0]
		dummies = np.column_stack([(surv == u).astype(float) for u in uniq[1:]])
		X = np.column_stack([np.ones(n), dummies, step])

	if weights is None:
		Wsqrt = None
		Xw = X
		yw = r
	else:
		w = np.asarray(weights, dtype=float)
		w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
		Wsqrt = np.sqrt(w)
		Xw = X * Wsqrt[:, None]
		yw = r * Wsqrt

	beta_hat, *_ = np.linalg.lstsq(Xw, yw, rcond=None)

	# residual variance
	yhat = Xw @ beta_hat
	res = yw - yhat
	# dof = n - rank
	rank = np.linalg.matrix_rank(Xw)
	dof = max(1, n - rank)
	s2 = float((res @ res) / dof)

	# covariance of beta_hat = s2 * (X'X)^{-1}
	XtX = Xw.T @ Xw
	try:
		cov = s2 * np.linalg.inv(XtX)
	except np.linalg.LinAlgError:
		return np.nan, np.nan, np.nan, np.nan

	# beta is last coefficient
	beta = float(beta_hat[-1])
	se = float(np.sqrt(cov[-1, -1]))
	tstat = beta / se if se > 0 else np.nan

	# Two-sided p-value using normal approximation
	if np.isfinite(tstat):
		# two-sided using math.erf to avoid np.math attribute warning
		p = float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(tstat) / math.sqrt(2.0)))))
	else:
		p = np.nan

	return beta, se, tstat, p


def blocked_permutation_pvalue(r, step, survey, z, weights, nperm=2000, z_bins=10, seed=0):
	"""
	Blocked permutation test for the step coefficient.

	Step labels are shuffled within (survey, z-bin) strata to preserve
	survey composition and redshift distribution.
	"""
	rng = np.random.default_rng(seed)

	# define z bins by quantiles globally (robust, simple)
	qs = np.linspace(0, 1, z_bins + 1)
	edges = np.quantile(z, qs)
	# protect against duplicates
	edges[0] = -np.inf
	edges[-1] = np.inf
	zbin = np.digitize(z, edges[1:-1], right=False)

	# observed
	beta_obs, *_ = fit_step_with_survey_fe(r, step, survey, weights=weights)
	if not np.isfinite(beta_obs):
		return np.nan, beta_obs

	extreme = 0
	for _ in range(nperm):
		step_perm = step.copy()
		# shuffle within blocks
		for s in np.unique(survey):
			ms = (survey == s)
			if not np.any(ms):
				continue
			for b in range(z_bins):
				mb = ms & (zbin == b)
				idx = np.where(mb)[0]
				if idx.size < 2:
					continue
				step_perm[idx] = step_perm[rng.permutation(idx)]
		beta_p, *_ = fit_step_with_survey_fe(r, step_perm, survey, weights=weights)
		if np.isfinite(beta_p) and abs(beta_p) >= abs(beta_obs):
			extreme += 1

	# +1 smoothing
	p = (extreme + 1) / (nperm + 1)
	return float(p), float(beta_obs)


def main():
	df = pd.read_csv(DATA, sep=r"\s+", comment="#")

	need = ["m_b_corr", "zCMB", "x1", "c", "HOST_LOGMASS", "IDSURVEY"]
	mask = np.ones(len(df), dtype=bool)
	for col in need:
		mask &= np.isfinite(df[col])
	df = df[mask].copy()
	if len(df) == 0:
		raise SystemExit("No valid data for Test 2 tightening")

	mb = np.asarray(df["m_b_corr"], dtype=float)
	z = np.asarray(df["zCMB"], dtype=float)
	x1 = np.asarray(df["x1"], dtype=float)
	c = np.asarray(df["c"], dtype=float)
	mass = np.asarray(df["HOST_LOGMASS"], dtype=float)
	survey = df["IDSURVEY"].astype(int).to_numpy()

	# optional weights (if present)
	if "m_b_corr_err_DIAG" in df.columns:
		mb_err = np.asarray(df["m_b_corr_err_DIAG"], dtype=float)
		w = np.where(np.isfinite(mb_err) & (mb_err > 0), 1.0 / (mb_err**2), 0.0)
	else:
		w = None

	# residual object consistent with Test 3 logic
	mb_hat = smooth_baseline(z, mb)
	r0 = mb - mb_hat
	r = remove_salt2_leakage(r0, x1, c)

	# step indicator
	step = (mass >= 10.0).astype(float)

	# fit step with survey fixed effects
	beta, se, tstat, p = fit_step_with_survey_fe(r, step, survey, weights=w)

	# scatter difference after removing survey FE + step
	# build fitted values using the same solver path: re-fit and compute model residuals
	# (reuse function by reconstructing design inside it, but simplest: just compute groupwise residuals of r)
	high = r[step == 1.0]
	low = r[step == 0.0]
	scatter_diff = float(np.var(high, ddof=1) - np.var(low, ddof=1))

	# blocked permutation p-value
	p_block, beta_obs = blocked_permutation_pvalue(r, step, survey, z, w, nperm=2000, z_bins=10, seed=0)

	print("HOST MASS STEP — TIGHTENED (z-baseline removed, x1/c leakage removed, survey FE)")
	print(f"  Δ_M (beta): {beta:.5f} mag")
	print(f"  SE(beta): {se:.5f}")
	print(f"  t: {tstat:.3f}  p(N(0,1) approx): {p:.3e}")
	print(f"  Blocked-permutation p-value: {p_block:.4f}")
	print(f"  Scatter difference (high − low) on cleaned residuals: {scatter_diff:.5f}")


if __name__ == "__main__":
	main()
