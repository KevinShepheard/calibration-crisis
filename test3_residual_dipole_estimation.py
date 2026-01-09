#!/usr/bin/env python3
"""
Test 3 — Residual dipole estimation and directional stability checks.

Fits a dipole to cleaned supernova residuals and evaluates its amplitude,
direction, and robustness under isotropic nulls, subsamples, and
survey-dependent partitions.
"""
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent  # repository root when script sits at top level
DATA = PROJECT_ROOT / "data/pantheonplus/distance_moduli/Pantheon_SH0ES.dat"


KNOWN_BULK_FLOW_DIRECTIONS = [
	{
		"label": "Planck CMB kinematic dipole",
		"ra": 167.94,
		"dec": -6.93,
	},
	{
		"label": "Watkins et al. (2009) bulk flow",
		"ra": 168.0,
		"dec": -38.0,
	},
	{
		"label": "Carrick et al. (2015) 2M++ prediction",
		"ra": 162.0,
		"dec": -7.0,
	},
]


SURVEY_FOOTPRINT_TEMPLATES = [
	{
		"label": "DES Year-5 wide field (approx)",
		"ra_bounds": (60.0, 100.0),
		"dec_bounds": (-62.0, -40.0),
		"note": "southern, concentrated on RA 4h-6h",
	},
	{
		"label": "SDSS main survey (approx)",
		"ra_bounds": (122.0, 250.0),
		"dec_bounds": (-10.0, 62.0),
		"note": "northern cap plus equatorial strip",
	},
	{
		"label": "2MASS Redshift Survey (approx)",
		"ra_bounds": (0.0, 360.0),
		"dec_bounds": (-30.0, 90.0),
		"note": "all-sky with |b| > 5 deg",
	},
]


def vector_to_ra_dec(vec):
	norm = np.linalg.norm(vec)
	if norm == 0:
		raise ValueError("Dipole vector has zero magnitude")

	x, y, z = vec
	ra = np.rad2deg(np.arctan2(y, x)) % 360
	dec = np.rad2deg(np.arcsin(np.clip(z / norm, -1.0, 1.0)))
	return ra, dec


def minimal_angular_difference(angle_a, angle_b):
	diff = (angle_a - angle_b + 180) % 360 - 180
	return abs(diff)


def angular_separation(ra_a, dec_a, ra_b, dec_b):
	ra_a_rad = np.deg2rad(ra_a)
	dec_a_rad = np.deg2rad(dec_a)
	ra_b_rad = np.deg2rad(ra_b)
	dec_b_rad = np.deg2rad(dec_b)

	sin_a, cos_a = np.sin(dec_a_rad), np.cos(dec_a_rad)
	sin_b, cos_b = np.sin(dec_b_rad), np.cos(dec_b_rad)
	cos_delta_ra = np.cos(ra_a_rad - ra_b_rad)

	cos_theta = sin_a * sin_b + cos_a * cos_b * cos_delta_ra
	cos_theta = np.clip(cos_theta, -1.0, 1.0)
	return np.rad2deg(np.arccos(cos_theta))


def ra_in_range(ra, ra_min, ra_max):
	ra = ra % 360
	ra_min = ra_min % 360
	ra_max = ra_max % 360
	if ra_min <= ra_max:
		return ra_min <= ra <= ra_max
	return ra >= ra_min or ra <= ra_max


def distance_to_ra_range(ra, ra_min, ra_max):
	if ra_in_range(ra, ra_min, ra_max):
		return 0.0
	return min(
		minimal_angular_difference(ra, ra_min),
		minimal_angular_difference(ra, ra_max),
	)


def distance_to_dec_range(dec, dec_min, dec_max):
	if dec_min <= dec <= dec_max:
		return 0.0
	return min(abs(dec - dec_min), abs(dec - dec_max))


def smooth_baseline(z, mu, span=0.035, grid_points=320):
	if len(z) == 0:
		return np.asarray(mu)

	idx = np.argsort(z)
	z_sorted = z[idx]
	mu_sorted = mu[idx]
	if len(np.unique(z_sorted)) < 2:
		return np.full_like(mu, np.mean(mu))

	grid_points = min(grid_points, len(z_sorted))
	z_grid = np.linspace(z_sorted[0], z_sorted[-1], grid_points)
	bandwidth = max(span, 1e-4)
	delta = (z_grid[:, None] - z_sorted[None, :]) / bandwidth
	weights = np.exp(-0.5 * delta**2)
	weights_sum = weights.sum(axis=1)
	weights_sum = np.where(weights_sum == 0, 1.0, weights_sum)
	mu_grid = (weights @ mu_sorted) / weights_sum
	return np.interp(z, z_grid, mu_grid)


def remove_salt2_leakage(resid, x1, c):
	design = np.column_stack([x1, c])
	if design.shape[0] < 2:
		return resid

	coeffs, *_ = np.linalg.lstsq(design, resid, rcond=None)
	return resid - design @ coeffs


def fit_dipole(x, resid, weights=None):
	if weights is not None:
		weights = np.asarray(weights, dtype=float)
		sqrt_w = np.sqrt(weights)
		projected_x = x * sqrt_w[:, None]
		projected_resid = resid * sqrt_w
	else:
		projected_x = x
		projected_resid = resid

	dipole, *_ = np.linalg.lstsq(projected_x, projected_resid, rcond=None)
	amp = np.linalg.norm(dipole)
	return dipole, amp


def load():
	return pd.read_csv(DATA, sep=r"\s+", comment="#")


def report_case(label, x, resid, mask, weights=None):
	count = mask.sum()
	if count == 0:
		print(f"    {label}: skipped (no objects)")
		return

	dipole, amp = fit_dipole(x[mask], resid[mask], None if weights is None else weights[mask])
	ra, dec = vector_to_ra_dec(dipole)
	print(
		f"    {label}: n={count}, amp={amp:.3f}, RA={ra:.1f} deg, Dec={dec:.1f} deg"
	)


def main():
	df = load()
	required = ["MU_SH0ES", "RA", "DEC", "zCMB", "x1", "c"]
	mask = np.ones(len(df), dtype=bool)
	for col in required:
		mask &= np.isfinite(df[col])

	df = df[mask].copy()
	if len(df) == 0:
		raise SystemExit("No valid data to run the anisotropy test")

	mu = np.asarray(df["MU_SH0ES"], dtype=float)
	ra = np.asarray(df["RA"], dtype=float)
	dec = np.asarray(df["DEC"], dtype=float)
	z = np.asarray(df["zCMB"], dtype=float)
	x1 = np.asarray(df["x1"], dtype=float)
	c = np.asarray(df["c"], dtype=float)
	survey = df["IDSURVEY"].fillna("missing").astype(str).to_numpy()
	vpec = np.asarray(df["VPEC"], dtype=float)
	vpec_err = np.asarray(df["VPECERR"], dtype=float)

	mu_hat = smooth_baseline(z, mu)
	residual = mu - mu_hat
	residual_clean = remove_salt2_leakage(residual, x1, c)

	ra_rad = np.deg2rad(ra)
	dec_rad = np.deg2rad(dec)
	x = np.column_stack([
		np.cos(ra_rad) * np.cos(dec_rad),
		np.sin(ra_rad) * np.cos(dec_rad),
		np.sin(dec_rad),
	])

	base_dipole, base_amp = fit_dipole(x, residual_clean)
	direction_ra, direction_dec = vector_to_ra_dec(base_dipole)

	rng = np.random.default_rng(0)
	null_amps = []
	for _ in range(2000):
		permuted = rng.permutation(residual_clean)
		d_null, *_ = np.linalg.lstsq(x, permuted, rcond=None)
		null_amps.append(np.linalg.norm(d_null))

	null_amps = np.asarray(null_amps)
	pval = np.mean(null_amps >= base_amp)

	print("TEST 3 — RESIDUAL DIPOLE ESTIMATION")
	print("  Residual = MU_SH0ES - smooth(mu(z)); SALT2 x1/c leakage removed before dipole fit")
	print(f"  Dipole vector: {base_dipole}")
	print(f"  Dipole amplitude: {base_amp:.6f}")
	print(f"  Isotropic null p-value: {pval:.4f}")
	print(f"  Dipole direction: RA={direction_ra:.2f} deg, Dec={direction_dec:.2f} deg")

	unit_direction = base_dipole / base_amp
	dot_products = np.clip(x @ unit_direction, -1.0, 1.0)
	angular_separations = np.rad2deg(np.arccos(dot_products))
	angular_stats = np.percentile(angular_separations, [50, 90])

	print("  Sample coverage vs. dipole direction:")
	print(
		f"    nearest data point {angular_separations.min():.2f} deg, "
		f"median separation {angular_stats[0]:.2f} deg, "
		f"90th percentile {angular_stats[1]:.2f} deg"
	)

	print("  Comparison to reference dipole directions:")
	for entry in KNOWN_BULK_FLOW_DIRECTIONS:
		sep = angular_separation(
			direction_ra,
			direction_dec,
			entry["ra"],
			entry["dec"],
		)
		print(
			f"    {entry['label']}: RA={entry['ra']:.2f} deg, Dec={entry['dec']:.2f} deg, "
			f"separation {sep:.1f} deg"
		)

	ra_bounds = (np.min(ra), np.max(ra))
	dec_bounds = (np.min(dec), np.max(dec))
	footprints = [
		{
			"label": "Pantheon+ SH0ES sample",
			"ra_bounds": ra_bounds,
			"dec_bounds": dec_bounds,
			"note": "data-derived",
		},
		*SURVEY_FOOTPRINT_TEMPLATES,
	]

	print("  Survey footprint comparison:")
	for fp in footprints:
		ra_offset = distance_to_ra_range(
			direction_ra, fp["ra_bounds"][0], fp["ra_bounds"][1]
		)
		dec_offset = distance_to_dec_range(
			direction_dec, fp["dec_bounds"][0], fp["dec_bounds"][1]
		)
		ra_inside = ra_offset == 0.0
		dec_inside = dec_offset == 0.0
		note = fp.get("note")
		note_text = f" ({note})" if note else ""
		print(
			f"    {fp['label']}{note_text}: RA range {fp['ra_bounds'][0]:.1f}-{fp['ra_bounds'][1]:.1f} deg, "
			f"Dec range {fp['dec_bounds'][0]:.1f}-{fp['dec_bounds'][1]:.1f} deg, "
			f"RA inside? {ra_inside}, Dec inside? {dec_inside}, "
			f"RA offset {ra_offset:.1f} deg, Dec offset {dec_offset:.1f} deg"
		)

	print("  Subsample and weighting sensitivity checks:")
	full_mask = np.ones(len(residual_clean), dtype=bool)
	report_case("z < 0.05", x, residual_clean, z < 0.05)
	report_case("z < 0.1", x, residual_clean, z < 0.1)

	vpec_err_weights = np.zeros_like(vpec_err)
	valid_err = np.isfinite(vpec_err) & (vpec_err > 0)
	if np.any(valid_err):
		clipped_err = np.clip(vpec_err[valid_err], 1e-3, 1e-1)
		vpec_err_weights[valid_err] = 1.0 / clipped_err**2
		vpec_err_weights = np.clip(vpec_err_weights, 0, 1e6)

	vpec_magnitude_weights = np.zeros_like(vpec)
	valid_vpec = np.isfinite(vpec)
	if np.any(valid_vpec):
		clipped_vpec = np.clip(np.abs(vpec[valid_vpec]), 1.0, 5e3)
		vpec_magnitude_weights[valid_vpec] = clipped_vpec

	report_case("velocity-error weights (1/VPECERR^2)", x, residual_clean, full_mask, vpec_err_weights)
	report_case("velocity-magnitude weights (|VPEC|)", x, residual_clean, full_mask, vpec_magnitude_weights)

	print("    Per-survey dipoles (IDSURVEY subset):")
	survey_names, counts = np.unique(survey, return_counts=True)
	order = np.argsort(-counts)
	printed = 0
	max_per_survey = 12
	for idx in order:
		if printed >= max_per_survey:
			break
		name = survey_names[idx]
		count = counts[idx]
		if count < 25:
			continue
		report_case(f"IDSURVEY {name}", x, residual_clean, survey == name)
		printed += 1
	if printed == 0:
		print("    <no surveys with 25+ objects>")


if __name__ == "__main__":
	main()
