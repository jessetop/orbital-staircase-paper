#!/usr/bin/env python3
"""
Threshold Independence Test — Does R² alone select tau = 3/2?

The cold referee critique: tau = 3/2 was chosen post-hoc to minimize |unity - 1|.

This script tests whether INDEPENDENT criteria — adjusted R², delta-AIC,
MDL, and leave-one-out cross-validation — also select tau ~ 3/2, without
any reference to the unity equation.

If multiple criteria converge on the same threshold, the post-hoc critique
is destroyed: you can't cherry-pick a value that multiple unrelated metrics
independently prefer.

Key design choice: We require >= 10 clusters for any fit metric, because
a 3-point line always has near-perfect R². The staircase model claims
EXPONENTIAL structure across MANY clusters — we enforce that.

Three independent selection criteria (none use unity):
  1. Adjusted R² of log-linear staircase fit (penalizes overfitting)
  2. MDL (Minimum Description Length) — information-theoretic model selection
  3. Leave-one-out cross-validation error on staircase predictions

Plus two comparison metrics:
  4. Delta-AIC (staircase vs Wagstaff one-step-ahead)
  5. |unity - 1| (for reference only, NOT used in threshold selection)
"""

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from staircase_utils import (
    load_mersenne_exponents,
    classify_clusters,
    get_singletons,
    compute_orbital_base,
    compute_orbital_r_squared,
    compute_escape_velocity,
    consecutive_pairing,
    unity_equation,
    EGAMMA_OVER_LN2,
)

MIN_CLUSTERS = 10  # Minimum clusters for a meaningful staircase fit


# ─── Staircase fit helpers ────────────────────────────────────────────────────

def staircase_fit(clusters):
    """
    Fit log(center_n) = intercept + slope * n via OLS.
    Returns (slope, intercept, residuals, R², adj_R², log_centers, predicted).
    """
    if len(clusters) < 3:
        return None
    centers = [math.exp(sum(math.log(x) for x in c) / len(c)) for c in clusters]
    log_centers = np.array([math.log(c) for c in centers])
    n = len(log_centers)
    x = np.arange(n)
    x_mean = x.mean()
    y_mean = log_centers.mean()
    ss_xx = np.sum((x - x_mean) ** 2)
    ss_yy = np.sum((log_centers - y_mean) ** 2)
    if ss_xx == 0 or ss_yy == 0:
        return None
    slope = np.sum((x - x_mean) * (log_centers - y_mean)) / ss_xx
    intercept = y_mean - slope * x_mean
    predicted = intercept + slope * x
    residuals = log_centers - predicted
    ss_res = np.sum(residuals ** 2)
    r_squared = 1.0 - ss_res / ss_yy
    # Adjusted R²: penalizes having fewer data points
    if n <= 2:
        adj_r_squared = r_squared
    else:
        adj_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) / (n - 2)
    return slope, intercept, residuals, r_squared, adj_r_squared, log_centers, predicted


def staircase_log_likelihood(clusters):
    """
    Log-likelihood of the two-rate staircase model.

    Within-cluster steps: log(p_{i+1}/p_i) ~ N(mu_w, sigma_w²)
    Between-cluster steps: log(first_next / last_prev) ~ N(mu_b, sigma_b²)

    Returns (log_likelihood, n_params, n_observations).
    n_params counts: mu_w, sigma_w, mu_b, sigma_b + 1 for the threshold itself.
    """
    within_steps = []
    between_steps = []

    for c in clusters:
        for j in range(len(c) - 1):
            within_steps.append(math.log(c[j + 1] / c[j]))

    for i in range(1, len(clusters)):
        between_steps.append(math.log(clusters[i][0] / clusters[i - 1][-1]))

    total_ll = 0.0
    n_params = 1  # threshold itself is a parameter
    n_obs = len(within_steps) + len(between_steps)

    if len(within_steps) >= 2:
        ws = np.array(within_steps)
        mu_w = ws.mean()
        sigma2_w = np.var(ws, ddof=1)
        if sigma2_w <= 1e-20:
            sigma2_w = 1e-10
        total_ll += -0.5 * len(ws) * math.log(2 * math.pi * sigma2_w) \
                    - 0.5 * np.sum((ws - mu_w) ** 2) / sigma2_w
        n_params += 2
    elif len(within_steps) == 1:
        n_params += 1

    if len(between_steps) >= 2:
        bs = np.array(between_steps)
        mu_b = bs.mean()
        sigma2_b = np.var(bs, ddof=1)
        if sigma2_b <= 1e-20:
            sigma2_b = 1e-10
        total_ll += -0.5 * len(bs) * math.log(2 * math.pi * sigma2_b) \
                    - 0.5 * np.sum((bs - mu_b) ** 2) / sigma2_b
        n_params += 2
    elif len(between_steps) == 1:
        n_params += 1

    return total_ll, n_params, n_obs


def wagstaff_log_likelihood(exponents):
    """
    Log-likelihood under Wagstaff's one-rate model.

    All steps log(p_{i+1}/p_i) ~ N(mu, sigma²) — single geometric spacing.
    Returns (log_likelihood, n_params).
    """
    if len(exponents) < 3:
        return None, None
    log_steps = np.array([math.log(exponents[i + 1] / exponents[i])
                          for i in range(len(exponents) - 1)])
    n = len(log_steps)
    mu = log_steps.mean()
    sigma2 = np.var(log_steps, ddof=1)
    if sigma2 <= 1e-20:
        sigma2 = 1e-10
    ll = -0.5 * n * math.log(2 * math.pi * sigma2) - 0.5 * np.sum((log_steps - mu) ** 2) / sigma2
    return ll, 2  # 2 params: mu, sigma


def compute_delta_aic(exponents, clusters):
    """
    AIC comparison: staircase vs Wagstaff single-rate.
    Positive delta-AIC = staircase is better.
    """
    ll_w, k_w = wagstaff_log_likelihood(exponents)
    if ll_w is None:
        return None
    ll_s, k_s, _ = staircase_log_likelihood(clusters)
    aic_w = 2 * k_w - 2 * ll_w
    aic_s = 2 * k_s - 2 * ll_s
    return aic_w - aic_s  # positive = staircase wins


def compute_mdl(clusters):
    """
    Minimum Description Length: MDL = -LL + (k/2) * log(n).
    Lower = better (more efficient coding of the data).
    """
    ll, k, n = staircase_log_likelihood(clusters)
    if n <= 0:
        return None
    return -ll + (k / 2.0) * math.log(n)


def compute_loocv_error(clusters):
    """
    Leave-one-out cross-validation on the staircase log-linear fit.

    Uses the analytic PRESS formula for OLS:
      PRESS_i = residual_i / (1 - h_ii)
    where h_ii is the leverage of point i.

    Returns mean squared prediction error.
    """
    if len(clusters) < 4:
        return None
    centers = [math.exp(sum(math.log(x) for x in c) / len(c)) for c in clusters]
    log_centers = np.array([math.log(c) for c in centers])
    n = len(log_centers)
    x = np.arange(n)
    x_mean = x.mean()
    ss_xx = np.sum((x - x_mean) ** 2)
    if ss_xx == 0:
        return None

    y_mean = log_centers.mean()
    slope = np.sum((x - x_mean) * (log_centers - y_mean)) / ss_xx
    intercept = y_mean - slope * x_mean
    predicted = intercept + slope * x
    residuals = log_centers - predicted

    # Hat matrix diagonal for simple OLS
    h = 1.0 / n + (x - x_mean) ** 2 / ss_xx
    press_residuals = residuals / (1.0 - h)
    return float(np.mean(press_residuals ** 2))


def compute_unity_at_tau(exponents, tau):
    """Return |unity - 1| at this threshold, or None if insufficient data."""
    clusters = classify_clusters(exponents, tau=tau)
    if len(clusters) < 5:
        return None
    singletons = get_singletons(clusters)
    if len(singletons) < 4:
        return None
    orbital = compute_orbital_base(clusters)
    escape = compute_escape_velocity(clusters)
    if orbital is None or escape is None or orbital <= 0 or escape <= 0:
        return None
    result = consecutive_pairing(singletons)
    if result is None:
        return None
    _, _, mean_decay = result
    u = unity_equation(mean_decay, escape, orbital)
    return abs(u - 1.0)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 84)
    print("THRESHOLD INDEPENDENCE TEST")
    print("Does staircase R-squared alone select tau = 3/2?")
    print("=" * 84)

    exponents = load_mersenne_exponents()
    print(f"\nLoaded {len(exponents)} Mersenne prime exponents.")
    print(f"Minimum cluster count for fit metrics: {MIN_CLUSTERS}")

    # ── Compute all consecutive ratios as natural threshold candidates ────
    ratios = []
    for i in range(len(exponents) - 1):
        ratios.append(exponents[i + 1] / exponents[i])

    unique_ratios = sorted(set(ratios))
    print(f"Unique consecutive ratios: {len(unique_ratios)}")

    # Generate test thresholds at every structural transition
    EPS = 1e-6
    test_taus = set()
    for r in unique_ratios:
        if 1.01 < r < 10.0:
            test_taus.add(r + EPS)
            test_taus.add(r - EPS)
    test_taus = sorted(test_taus)
    print(f"Test thresholds: {len(test_taus)}")

    # ══════════════════════════════════════════════════════════════════════
    # PART 1: Full threshold sweep with >= MIN_CLUSTERS constraint
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 84}")
    print(f"PART 1: R-squared vs threshold sweep (>= {MIN_CLUSTERS} clusters required)")
    print(f"{'=' * 84}\n")

    all_results = []

    for tau in test_taus:
        clusters = classify_clusters(exponents, tau=tau)
        n_clusters = len(clusters)
        n_singletons = len(get_singletons(clusters))

        fit = staircase_fit(clusters)
        r_squared = fit[3] if fit else None
        adj_r_squared = fit[4] if fit else None

        delta_aic = compute_delta_aic(exponents, clusters)
        mdl = compute_mdl(clusters)
        cv_error = compute_loocv_error(clusters)
        unity_dist = compute_unity_at_tau(exponents, tau)

        all_results.append({
            "tau": tau,
            "n_clusters": n_clusters,
            "n_singletons": n_singletons,
            "r_squared": r_squared,
            "adj_r_squared": adj_r_squared,
            "delta_aic": delta_aic,
            "mdl": mdl,
            "cv_error": cv_error,
            "unity_dist": unity_dist,
        })

    # Apply minimum cluster constraint
    valid = [r for r in all_results
             if r["adj_r_squared"] is not None and r["n_clusters"] >= MIN_CLUSTERS]
    print(f"Valid results (adj-R² computable, >= {MIN_CLUSTERS} clusters): {len(valid)}")

    # Print table
    header = (f"{'tau':>10s}  {'clust':>5s}  {'sing':>5s}  "
              f"{'adj-R2':>8s}  {'R2':>8s}  {'dAIC':>8s}  {'MDL':>9s}  "
              f"{'CV-err':>10s}  {'|u-1|':>8s}")
    print(f"\n{header}")
    print("-" * len(header))

    for r in valid:
        u_str = f"{r['unity_dist']:8.4f}" if r['unity_dist'] is not None else "     N/A"
        mdl_str = f"{r['mdl']:9.2f}" if r['mdl'] is not None else "      N/A"
        daic_str = f"{r['delta_aic']:8.2f}" if r['delta_aic'] is not None else "     N/A"
        cv_str = f"{r['cv_error']:10.6f}" if r['cv_error'] is not None else "       N/A"
        print(f"{r['tau']:10.6f}  {r['n_clusters']:5d}  {r['n_singletons']:5d}  "
              f"{r['adj_r_squared']:8.6f}  {r['r_squared']:8.6f}  {daic_str}  {mdl_str}  "
              f"{cv_str}  {u_str}")

    # ══════════════════════════════════════════════════════════════════════
    # PART 2: Optimal threshold by each criterion
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 84}")
    print(f"PART 2: Optimal threshold for each criterion (>= {MIN_CLUSTERS} clusters)")
    print(f"{'=' * 84}\n")

    criteria = {}

    # Best adjusted R² (maximize)
    best_ar2 = max(valid, key=lambda r: r["adj_r_squared"])
    criteria["Adjusted R-squared"] = best_ar2["tau"]
    print(f"  MAX adj-R²:    tau = {best_ar2['tau']:.6f}  "
          f"(adj-R² = {best_ar2['adj_r_squared']:.6f}, "
          f"R² = {best_ar2['r_squared']:.6f}, "
          f"{best_ar2['n_clusters']} clusters)")

    # Best delta-AIC (maximize)
    aic_valid = [r for r in valid if r["delta_aic"] is not None]
    if aic_valid:
        best_aic = max(aic_valid, key=lambda r: r["delta_aic"])
        criteria["Delta-AIC"] = best_aic["tau"]
        print(f"  MAX delta-AIC: tau = {best_aic['tau']:.6f}  "
              f"(dAIC = {best_aic['delta_aic']:.2f}, "
              f"{best_aic['n_clusters']} clusters)")

    # Best MDL (minimize)
    mdl_valid = [r for r in valid if r["mdl"] is not None]
    if mdl_valid:
        best_mdl = min(mdl_valid, key=lambda r: r["mdl"])
        criteria["MDL"] = best_mdl["tau"]
        print(f"  MIN MDL:       tau = {best_mdl['tau']:.6f}  "
              f"(MDL = {best_mdl['mdl']:.2f}, "
              f"{best_mdl['n_clusters']} clusters)")

    # Best CV error (minimize)
    cv_valid = [r for r in valid if r["cv_error"] is not None]
    if cv_valid:
        best_cv = min(cv_valid, key=lambda r: r["cv_error"])
        criteria["CV-error"] = best_cv["tau"]
        print(f"  MIN CV-error:  tau = {best_cv['tau']:.6f}  "
              f"(CV = {best_cv['cv_error']:.6f}, "
              f"{best_cv['n_clusters']} clusters)")

    # Unity reference
    unity_valid = [r for r in valid if r["unity_dist"] is not None]
    if unity_valid:
        best_unity = min(unity_valid, key=lambda r: r["unity_dist"])
        criteria["|unity-1| (reference)"] = best_unity["tau"]
        print(f"  MIN |unity-1|: tau = {best_unity['tau']:.6f}  "
              f"(|u-1| = {best_unity['unity_dist']:.6f})  [reference only]")

    # ══════════════════════════════════════════════════════════════════════
    # PART 3: Top-5 by each criterion
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 84}")
    print("PART 3: Top-5 thresholds by each independent criterion")
    print(f"{'=' * 84}")

    print(f"\n  Top-5 by Adjusted R-squared (higher = better):")
    for i, r in enumerate(sorted(valid, key=lambda r: -r["adj_r_squared"])[:5]):
        u_str = f"|u-1|={r['unity_dist']:.4f}" if r['unity_dist'] is not None else "|u-1|=N/A"
        print(f"    #{i+1}: tau={r['tau']:.6f}  adj-R²={r['adj_r_squared']:.6f}  "
              f"clusters={r['n_clusters']}  {u_str}")

    if aic_valid:
        print(f"\n  Top-5 by delta-AIC (higher = staircase wins more):")
        for i, r in enumerate(sorted(aic_valid, key=lambda r: -r["delta_aic"])[:5]):
            u_str = f"|u-1|={r['unity_dist']:.4f}" if r['unity_dist'] is not None else "|u-1|=N/A"
            print(f"    #{i+1}: tau={r['tau']:.6f}  dAIC={r['delta_aic']:.2f}  "
                  f"clusters={r['n_clusters']}  {u_str}")

    if mdl_valid:
        print(f"\n  Top-5 by MDL (lower = better):")
        for i, r in enumerate(sorted(mdl_valid, key=lambda r: r["mdl"])[:5]):
            u_str = f"|u-1|={r['unity_dist']:.4f}" if r['unity_dist'] is not None else "|u-1|=N/A"
            print(f"    #{i+1}: tau={r['tau']:.6f}  MDL={r['mdl']:.2f}  "
                  f"clusters={r['n_clusters']}  {u_str}")

    if cv_valid:
        print(f"\n  Top-5 by CV-error (lower = better prediction):")
        for i, r in enumerate(sorted(cv_valid, key=lambda r: r["cv_error"])[:5]):
            u_str = f"|u-1|={r['unity_dist']:.4f}" if r['unity_dist'] is not None else "|u-1|=N/A"
            print(f"    #{i+1}: tau={r['tau']:.6f}  CV={r['cv_error']:.6f}  "
                  f"clusters={r['n_clusters']}  {u_str}")

    # ══════════════════════════════════════════════════════════════════════
    # PART 4: MDL threshold selection (dedicated analysis)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 84}")
    print("PART 4: MDL threshold selection (information-theoretic)")
    print(f"{'=' * 84}\n")

    if mdl_valid:
        print("  MDL = -log_likelihood + (k/2) * log(n)")
        print(f"  where k includes threshold as a parameter")
        print(f"  Optimal: tau = {best_mdl['tau']:.6f} with MDL = {best_mdl['mdl']:.2f}")
        print(f"  This corresponds to {best_mdl['n_clusters']} clusters")

        # Show MDL at tau ~ 1.5 for comparison
        near_15 = [r for r in mdl_valid if abs(r["tau"] - 1.5) < 0.1]
        if near_15:
            best_near = min(near_15, key=lambda r: r["mdl"])
            print(f"\n  MDL near tau = 1.5: tau = {best_near['tau']:.6f}, "
                  f"MDL = {best_near['mdl']:.2f}, "
                  f"{best_near['n_clusters']} clusters")
            print(f"  MDL penalty for 1.5 vs optimal: "
                  f"{best_near['mdl'] - best_mdl['mdl']:.2f}")

    # ══════════════════════════════════════════════════════════════════════
    # PART 5: Cross-validation (dedicated analysis)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 84}")
    print("PART 5: Leave-one-out cross-validation")
    print(f"{'=' * 84}\n")

    if cv_valid:
        print(f"  LOOCV minimizes prediction error on held-out cluster centers.")
        print(f"  Optimal: tau = {best_cv['tau']:.6f} with CV = {best_cv['cv_error']:.6f}")
        print(f"  This corresponds to {best_cv['n_clusters']} clusters")

        near_15 = [r for r in cv_valid if abs(r["tau"] - 1.5) < 0.1]
        if near_15:
            best_near = min(near_15, key=lambda r: r["cv_error"])
            print(f"\n  CV near tau = 1.5: tau = {best_near['tau']:.6f}, "
                  f"CV = {best_near['cv_error']:.6f}, "
                  f"{best_near['n_clusters']} clusters")
            # Rank of 1.5 among all thresholds
            sorted_cv = sorted(cv_valid, key=lambda r: r["cv_error"])
            rank = next(i for i, r in enumerate(sorted_cv)
                        if abs(r["tau"] - best_near["tau"]) < 1e-8) + 1
            print(f"  Rank of tau ~ 1.5 among {len(cv_valid)} thresholds: #{rank}")

    # ══════════════════════════════════════════════════════════════════════
    # PART 6: Convergence analysis
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 84}")
    print("PART 6: Convergence analysis — do independent criteria agree?")
    print(f"{'=' * 84}\n")

    target = 1.5
    # Use two tolerance levels
    for tol_label, tolerance in [("strict (within 0.10)", 0.10),
                                  ("loose (within 0.20)", 0.20)]:
        n_agree = 0
        n_independent = 0
        print(f"  Tolerance: {tol_label}")
        for name, tau_opt in criteria.items():
            is_ref = "reference" in name
            near = abs(tau_opt - target) < tolerance
            delta = tau_opt - target
            status = "YES" if near else f"NO  (delta = {delta:+.3f})"
            marker = " [reference only]" if is_ref else ""
            print(f"    {name:>25s}: tau* = {tau_opt:.6f}  near 3/2? {status}{marker}")
            if not is_ref:
                n_independent += 1
                if near:
                    n_agree += 1

        print(f"  => {n_agree} / {n_independent} independent criteria agree\n")

    # ══════════════════════════════════════════════════════════════════════
    # PART 7: Rank analysis — where does tau ~ 1.5 place in each ranking?
    # ══════════════════════════════════════════════════════════════════════
    print(f"{'=' * 84}")
    print("PART 7: Rank of tau ~ 1.5 under each criterion")
    print(f"{'=' * 84}\n")

    # Find the result closest to tau = 1.5
    r_at_15 = min(valid, key=lambda r: abs(r["tau"] - 1.5))
    print(f"  Closest threshold to 1.5: tau = {r_at_15['tau']:.6f}")
    print(f"  Clusters: {r_at_15['n_clusters']}, Singletons: {r_at_15['n_singletons']}")
    print()

    # Rank by each criterion
    by_ar2 = sorted(valid, key=lambda r: -r["adj_r_squared"])
    rank_ar2 = next(i for i, r in enumerate(by_ar2)
                    if abs(r["tau"] - r_at_15["tau"]) < 1e-8) + 1
    print(f"  Adjusted R²:   rank {rank_ar2:>3d} / {len(valid):>3d}  "
          f"(value = {r_at_15['adj_r_squared']:.6f}, "
          f"best = {by_ar2[0]['adj_r_squared']:.6f})")

    if aic_valid:
        by_aic = sorted(aic_valid, key=lambda r: -r["delta_aic"])
        rank_aic = next(i for i, r in enumerate(by_aic)
                        if abs(r["tau"] - r_at_15["tau"]) < 1e-8) + 1
        print(f"  Delta-AIC:     rank {rank_aic:>3d} / {len(aic_valid):>3d}  "
              f"(value = {r_at_15['delta_aic']:.2f}, "
              f"best = {by_aic[0]['delta_aic']:.2f})")

    if mdl_valid:
        by_mdl = sorted(mdl_valid, key=lambda r: r["mdl"])
        rank_mdl = next(i for i, r in enumerate(by_mdl)
                        if abs(r["tau"] - r_at_15["tau"]) < 1e-8) + 1
        print(f"  MDL:           rank {rank_mdl:>3d} / {len(mdl_valid):>3d}  "
              f"(value = {r_at_15['mdl']:.2f}, "
              f"best = {by_mdl[0]['mdl']:.2f})")

    if cv_valid:
        by_cv = sorted(cv_valid, key=lambda r: r["cv_error"])
        rank_cv = next(i for i, r in enumerate(by_cv)
                       if abs(r["tau"] - r_at_15["tau"]) < 1e-8) + 1
        print(f"  CV-error:      rank {rank_cv:>3d} / {len(cv_valid):>3d}  "
              f"(value = {r_at_15['cv_error']:.6f}, "
              f"best = {by_cv[0]['cv_error']:.6f})")

    if unity_valid:
        by_unity = sorted(unity_valid, key=lambda r: r["unity_dist"])
        r_at_15_u = min(unity_valid, key=lambda r: abs(r["tau"] - 1.5))
        rank_u = next(i for i, r in enumerate(by_unity)
                      if abs(r["tau"] - r_at_15_u["tau"]) < 1e-8) + 1
        print(f"  |unity-1|:     rank {rank_u:>3d} / {len(unity_valid):>3d}  "
              f"(value = {r_at_15_u['unity_dist']:.6f}, "
              f"best = {by_unity[0]['unity_dist']:.6f})  [reference]")

    # Percentile analysis
    print(f"\n  Percentile of tau ~ 1.5 (lower = better for that criterion):")
    print(f"    Adjusted R²: top {100 * rank_ar2 / len(valid):.1f}%")
    if aic_valid:
        print(f"    Delta-AIC:   top {100 * rank_aic / len(aic_valid):.1f}%")
    if mdl_valid:
        print(f"    MDL:         top {100 * rank_mdl / len(mdl_valid):.1f}%")
    if cv_valid:
        print(f"    CV-error:    top {100 * rank_cv / len(cv_valid):.1f}%")

    # ══════════════════════════════════════════════════════════════════════
    # HEADLINE
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 84}")
    print("HEADLINE RESULT")
    print(f"{'=' * 84}\n")

    # Determine the story from the data
    all_ranks = [rank_ar2]
    rank_labels = ["adj-R²"]
    n_total = len(valid)
    if aic_valid:
        all_ranks.append(rank_aic)
        rank_labels.append("dAIC")
    if mdl_valid:
        all_ranks.append(rank_mdl)
        rank_labels.append("MDL")
    if cv_valid:
        all_ranks.append(rank_cv)
        rank_labels.append("CV")

    mean_percentile = np.mean([100 * r / n_total for r in all_ranks])
    top_quartile = sum(1 for r in all_ranks if r <= n_total * 0.25)

    print(f"  tau ~ 1.5 ranks in top {mean_percentile:.0f}% on average across "
          f"{len(all_ranks)} criteria.")
    print(f"  In top quartile for {top_quartile}/{len(all_ranks)} criteria: "
          f"{', '.join(l for l, r in zip(rank_labels, all_ranks) if r <= n_total * 0.25)}")

    # Check which criteria have tau ~ 1.5 as the global optimum
    optima_near_15 = sum(1 for name, tau in criteria.items()
                         if "reference" not in name and abs(tau - 1.5) < 0.15)
    print(f"\n  Criteria with global optimum near 1.5 (within 0.15): "
          f"{optima_near_15}/{len([c for c in criteria if 'reference' not in c])}")

    if optima_near_15 >= 2:
        print("\n  CONCLUSION: Multiple independent criteria select tau near 3/2.")
        print("  The post-hoc critique is refuted.")
    elif top_quartile >= 2:
        print(f"\n  CONCLUSION: tau = 3/2 is competitive (top quartile in "
              f"{top_quartile} criteria)")
        print("  but is NOT the unique global optimum of all metrics.")
        print("  The independence argument partially supports tau = 3/2.")
    else:
        print(f"\n  CONCLUSION: The independence argument for tau = 3/2 is WEAK.")
        print("  Other criteria prefer different thresholds.")
        print("  This is an honest result — report it transparently.")

    # ══════════════════════════════════════════════════════════════════════
    # Save JSON results
    # ══════════════════════════════════════════════════════════════════════
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "threshold_independence_results.json")

    json_results = {
        "n_exponents": len(exponents),
        "min_clusters": MIN_CLUSTERS,
        "n_test_thresholds": len(test_taus),
        "n_valid": len(valid),
        "optimal_thresholds": criteria,
        "tau_15_ranks": {
            "adj_r_squared": {"rank": rank_ar2, "total": len(valid),
                              "value": r_at_15["adj_r_squared"]},
        },
        "mean_percentile": round(mean_percentile, 1),
        "top_quartile_count": top_quartile,
        "sweep_data": [
            {k: v for k, v in r.items()}
            for r in valid
        ],
    }
    if aic_valid:
        json_results["tau_15_ranks"]["delta_aic"] = {
            "rank": rank_aic, "total": len(aic_valid),
            "value": r_at_15["delta_aic"]}
    if mdl_valid:
        json_results["tau_15_ranks"]["mdl"] = {
            "rank": rank_mdl, "total": len(mdl_valid),
            "value": r_at_15["mdl"]}
    if cv_valid:
        json_results["tau_15_ranks"]["cv_error"] = {
            "rank": rank_cv, "total": len(cv_valid),
            "value": r_at_15["cv_error"]}

    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
