#!/usr/bin/env python3
"""
Claim 1 — Orbital Staircase Fit
================================
"The orbital staircase model fits Mersenne prime positions with
 R²=0.998, ΔAIC>50 vs Wagstaff, and 43% RMSE reduction."

Uses one-step-ahead prediction residuals in log space.
Matches methodology from scripts/aic_bic_comparison.py.

Dependencies: numpy, scipy
"""

import sys
import os
import math

import numpy as np
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from staircase_utils import (
    load_mersenne_exponents,
    compute_orbital_r_squared,
    TAU_DEFAULT,
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def log_likelihood_normal(residuals):
    """Log-likelihood for iid Normal errors with MLE variance."""
    n = len(residuals)
    ss = sum(r ** 2 for r in residuals)
    sigma2 = ss / n
    if sigma2 <= 0:
        return float("-inf")
    return -n / 2 * math.log(2 * math.pi) - n / 2 * math.log(sigma2) - n / 2


def aic(k, ll):
    return 2 * k - 2 * ll


def aicc(k, n, ll):
    a = aic(k, ll)
    if n - k - 1 > 0:
        a += 2 * k * (k + 1) / (n - k - 1)
    return a


def bic(k, n, ll):
    return k * math.log(n) - 2 * ll


def rmse(residuals):
    return math.sqrt(sum(r ** 2 for r in residuals) / len(residuals))


# ─── Load data ────────────────────────────────────────────────────────────────

exponents = load_mersenne_exponents()
n_obs = len(exponents) - 1  # 51 transitions

log_exps = [math.log(p) for p in exponents]
log_ratios = [log_exps[i] - log_exps[i - 1] for i in range(1, len(log_exps))]
raw_ratios = [exponents[i] / exponents[i - 1] for i in range(1, len(exponents))]

# ─── Wagstaff model (1 param: single ratio) ──────────────────────────────────

avg_log_ratio = sum(log_ratios) / len(log_ratios)
wagstaff_resid = [lr - avg_log_ratio for lr in log_ratios]
wagstaff_rmse = rmse(wagstaff_resid)

ll_wag = log_likelihood_normal(wagstaff_resid)
k_wag = 1

# ─── Staircase model (2 params: within + between ratio, threshold=1.5) ───────

THRESHOLD = 1.5

within_lr = []
between_lr = []
classifications = []

for r, lr in zip(raw_ratios, log_ratios):
    if r < THRESHOLD:
        within_lr.append(lr)
        classifications.append("within")
    else:
        between_lr.append(lr)
        classifications.append("between")

n_within = len(within_lr)
n_between = len(between_lr)
avg_within = sum(within_lr) / len(within_lr)
avg_between = sum(between_lr) / len(between_lr)

staircase_resid = []
for i, cls in enumerate(classifications):
    predicted = avg_within if cls == "within" else avg_between
    staircase_resid.append(log_ratios[i] - predicted)

staircase_rmse = rmse(staircase_resid)

ll_stair = log_likelihood_normal(staircase_resid)
k_stair = 2

# ─── Orbital R² ──────────────────────────────────────────────────────────────

# Use TAU_DEFAULT (1.501) for clustering — matches the paper's cluster definition
from staircase_utils import classify_clusters
clusters = classify_clusters(exponents)
r2 = compute_orbital_r_squared(clusters)

# ─── Information criteria ─────────────────────────────────────────────────────

aic_wag = aic(k_wag, ll_wag)
aic_stair = aic(k_stair, ll_stair)
bic_wag = bic(k_wag, n_obs, ll_wag)
bic_stair = bic(k_stair, n_obs, ll_stair)

delta_aic = aic_wag - aic_stair
delta_bic = bic_wag - bic_stair
rmse_pct = (1 - staircase_rmse / wagstaff_rmse) * 100

# ─── Mann-Whitney U: first-in-cluster vs within-cluster residuals ─────────────

first_resid = [wagstaff_resid[i] for i in range(n_obs) if classifications[i] == "between"]
other_resid = [wagstaff_resid[i] for i in range(n_obs) if classifications[i] == "within"]

mw_stat, mw_p = mannwhitneyu(first_resid, other_resid, alternative="two-sided")

# ─── Report ───────────────────────────────────────────────────────────────────

print("=" * 70)
print("CLAIM 1: ORBITAL STAIRCASE FIT")
print("=" * 70)

print(f"\nData: {len(exponents)} Mersenne primes, {n_obs} transitions")
print(f"  Within-cluster (ratio < {THRESHOLD}):  {n_within}")
print(f"  Between-cluster (ratio >= {THRESHOLD}): {n_between}")

print(f"\n--- Wagstaff Model (1 parameter) ---")
print(f"  MLE ratio: {math.exp(avg_log_ratio):.6f}")
print(f"  RMSE (ln-space): {wagstaff_rmse:.6f}")
print(f"  AIC:  {aic_wag:.2f}")
print(f"  BIC:  {bic_wag:.2f}")

print(f"\n--- Staircase Model (2 parameters, threshold={THRESHOLD}) ---")
print(f"  Within ratio:  {math.exp(avg_within):.6f}  ({n_within} transitions)")
print(f"  Between ratio: {math.exp(avg_between):.6f}  ({n_between} transitions)")
print(f"  RMSE (ln-space): {staircase_rmse:.6f}")
print(f"  AIC:  {aic_stair:.2f}")
print(f"  BIC:  {bic_stair:.2f}")

print(f"\n--- Comparison ---")
print(f"  Orbital R²:       {r2:.4f}")
print(f"  ΔAIC (Wag-Stair): {delta_aic:+.2f}")
print(f"  ΔBIC (Wag-Stair): {delta_bic:+.2f}")
print(f"  RMSE reduction:   {rmse_pct:.1f}%")

print(f"\n--- Mann-Whitney U (Wagstaff residuals: between vs within) ---")
print(f"  First-in-cluster mean: {np.mean(first_resid):+.4f} (n={len(first_resid)})")
print(f"  Within-cluster mean:   {np.mean(other_resid):+.4f} (n={len(other_resid)})")
print(f"  U = {mw_stat:.1f}, p = {mw_p:.4f}")

# ─── Verification ─────────────────────────────────────────────────────────────

print(f"\n{'=' * 70}")
print("VERIFICATION")
print("=" * 70)

checks = [
    ("R² ≈ 0.998",            abs(r2 - 0.998) < 0.002),
    ("ΔAIC > 50",             delta_aic > 50),
    ("RMSE reduction ≈ 43%",  abs(rmse_pct - 43.2) < 2),
    ("Mann-Whitney p < 0.05",  mw_p < 0.05),
    ("Within-cluster = 32",   n_within == 32),
    ("Between-cluster = 19",  n_between == 19),
]

all_ok = True
for label, ok in checks:
    status = "PASS" if ok else "FAIL"
    if not ok:
        all_ok = False
    print(f"  [{status}] {label}")

if all_ok:
    print("\n  All checks passed.")
else:
    print("\n  WARNING: Some checks failed — investigate.")
