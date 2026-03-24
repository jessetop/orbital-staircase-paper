#!/usr/bin/env python3
"""
Global Pipeline Monte Carlo Test
=================================
Addresses the "garden of forking paths" critique by testing the JOINT
probability that a random Wagstaff-Poisson sequence satisfies ALL four
empirical claims simultaneously, even when the threshold tau is chosen
post-hoc to minimize |unity - 1|.

For each synthetic sequence:
  1. Sweep ALL natural discontinuity points (unique consecutive ratios)
     and pick the threshold that minimizes |unity - 1|.
  2. At that best threshold, compute four metrics:
     a. Staircase R^2  (exponential fit of cluster geometric centers)
     b. DAIC vs single-rate Wagstaff (one-step-ahead residuals)
     c. |unity - 1|    (decay^2 * escape * orbital)
     d. |Spearman r|   of log(jump) vs cluster index (scale independence)
  3. Check whether the synthetic sequence JOINTLY beats the real data on
     ALL four metrics.

The fraction that beat on all four = global p-value.

Usage:
    python global_pipeline_mc.py [--nsims N] [--seed S] [--workers W]
"""

import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.stats import spearmanr

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
    generate_wagstaff_sequence,
    TAU_DEFAULT,
)

# ── Real-data thresholds (from individual claim scripts) ─────────────────────

REAL_R2 = 0.9975      # computed: 0.997564
REAL_DAIC = 55.7      # computed: 56.46
REAL_UNITY_DIST = 0.045  # computed: 0.04443
REAL_SPEARMAN_ABS = 0.008  # computed: 0.00722


# ── Helpers ──────────────────────────────────────────────────────────────────

def log_likelihood_normal(residuals):
    """Log-likelihood for iid Normal errors with MLE variance."""
    n = len(residuals)
    ss = sum(r * r for r in residuals)
    sigma2 = ss / n
    if sigma2 <= 0:
        return float("-inf")
    return -n / 2 * math.log(2 * math.pi) - n / 2 * math.log(sigma2) - n / 2


def aic(k, ll):
    return 2 * k - 2 * ll


def compute_daic(exponents, tau):
    """
    Compute DAIC = AIC_wagstaff - AIC_staircase using one-step-ahead
    residuals in log space, classifying each transition as within or
    between by comparing the raw ratio against tau.

    Returns DAIC or None if classification is degenerate.
    """
    if len(exponents) < 6:
        return None

    log_exps = [math.log(p) for p in exponents]
    log_ratios = [log_exps[i] - log_exps[i - 1] for i in range(1, len(log_exps))]
    raw_ratios = [exponents[i] / exponents[i - 1] for i in range(1, len(exponents))]

    # Wagstaff: single mean log-ratio
    avg_lr = sum(log_ratios) / len(log_ratios)
    wag_resid = [lr - avg_lr for lr in log_ratios]
    ll_wag = log_likelihood_normal(wag_resid)

    # Staircase: separate within / between means
    within_lr = []
    between_lr = []
    classifications = []
    for r, lr in zip(raw_ratios, log_ratios):
        if r < tau:
            within_lr.append(lr)
            classifications.append("w")
        else:
            between_lr.append(lr)
            classifications.append("b")

    if len(within_lr) < 1 or len(between_lr) < 1:
        return None

    avg_w = sum(within_lr) / len(within_lr)
    avg_b = sum(between_lr) / len(between_lr)

    stair_resid = []
    for i, cls in enumerate(classifications):
        predicted = avg_w if cls == "w" else avg_b
        stair_resid.append(log_ratios[i] - predicted)

    ll_stair = log_likelihood_normal(stair_resid)

    return aic(1, ll_wag) - aic(2, ll_stair)


def compute_spearman_escape(clusters):
    """
    |Spearman r| of log(jump_ratio) vs cluster index.
    Returns |rho| or None.
    """
    if len(clusters) < 4:
        return None
    jumps = []
    for i in range(1, len(clusters)):
        ratio = clusters[i][0] / clusters[i - 1][-1]
        if ratio <= 0:
            return None
        jumps.append(math.log(ratio))
    indices = list(range(len(jumps)))
    if len(jumps) < 3:
        return None
    rho, _ = spearmanr(indices, jumps)
    if np.isnan(rho):
        return None
    return abs(rho)


def evaluate_at_threshold(exponents, tau):
    """
    Evaluate all four metrics for a given sequence at a given threshold.
    Returns dict of metrics or None if any metric is uncomputable.
    """
    clusters = classify_clusters(exponents, tau=tau)

    if len(clusters) < 3:
        return None

    # R^2
    r2 = compute_orbital_r_squared(clusters)
    if r2 is None:
        return None

    # DAIC
    daic = compute_daic(exponents, tau)
    if daic is None:
        return None

    # Unity
    singletons = get_singletons(clusters)
    orbital = compute_orbital_base(clusters)
    escape = compute_escape_velocity(clusters)

    if orbital is None or escape is None or orbital <= 0 or escape <= 0:
        return None

    unity_dist = None
    if len(singletons) >= 4:
        result = consecutive_pairing(singletons)
        if result is not None:
            _, _, mean_decay = result
            u = unity_equation(mean_decay, escape, orbital)
            if 0 < u < 100:
                unity_dist = abs(u - 1.0)

    if unity_dist is None:
        return None

    # Spearman escape
    spearman_abs = compute_spearman_escape(clusters)
    if spearman_abs is None:
        return None

    return {
        "r2": r2,
        "daic": daic,
        "unity_dist": unity_dist,
        "spearman_abs": spearman_abs,
        "tau": tau,
        "n_clusters": len(clusters),
        "n_singletons": len(singletons),
    }


def find_best_threshold(exponents):
    """
    Sweep all natural discontinuity points (unique consecutive ratios)
    and return the metrics at the threshold minimizing |unity - 1|.

    This mirrors the post-hoc selection the referee criticized:
    the analyst could pick the best-looking threshold.
    """
    if len(exponents) < 6:
        return None

    # Compute all consecutive ratios as candidate thresholds
    ratios = sorted(set(
        exponents[i] / exponents[i - 1]
        for i in range(1, len(exponents))
    ))

    # Test thresholds placed between each pair of consecutive ratios
    # Also include values just above each unique ratio
    candidate_taus = []
    for r in ratios:
        candidate_taus.append(r + 1e-9)  # just above each ratio

    # Also add midpoints between consecutive ratios for finer sweep
    for i in range(len(ratios) - 1):
        candidate_taus.append((ratios[i] + ratios[i + 1]) / 2)

    # Filter to reasonable range (must have both within and between)
    candidate_taus = [t for t in candidate_taus if 1.01 < t < max(ratios)]

    if not candidate_taus:
        return None

    best = None
    best_unity_dist = float("inf")

    for tau in candidate_taus:
        result = evaluate_at_threshold(exponents, tau)
        if result is None:
            continue
        if result["unity_dist"] < best_unity_dist:
            best_unity_dist = result["unity_dist"]
            best = result

    return best


def simulate_one(args):
    """
    Simulate a single synthetic sequence and evaluate it.
    Returns a dict of metrics or None.
    """
    seed_offset, n_target, p_max = args
    rng = np.random.default_rng(seed_offset)
    exponents = generate_wagstaff_sequence(rng, n_target=n_target, p_max=p_max)

    if len(exponents) < 6:
        return None

    return find_best_threshold(exponents)


def compute_real_metrics():
    """Compute the four metrics for the real Mersenne prime sequence."""
    exponents = load_mersenne_exponents()
    clusters = classify_clusters(exponents, tau=TAU_DEFAULT)
    singletons = get_singletons(clusters)

    r2 = compute_orbital_r_squared(clusters)
    daic = compute_daic(exponents, TAU_DEFAULT)
    spearman_abs = compute_spearman_escape(clusters)

    orbital = compute_orbital_base(clusters)
    escape = compute_escape_velocity(clusters)
    _, _, mean_decay = consecutive_pairing(singletons)
    u = unity_equation(mean_decay, escape, orbital)
    unity_dist = abs(u - 1.0)

    return {
        "r2": r2,
        "daic": daic,
        "unity_dist": unity_dist,
        "spearman_abs": spearman_abs,
        "n_clusters": len(clusters),
        "n_singletons": len(singletons),
        "tau": TAU_DEFAULT,
        "orbital": orbital,
        "escape": escape,
        "mean_decay": mean_decay,
        "unity": u,
    }


def main():
    parser = argparse.ArgumentParser(description="Global pipeline Monte Carlo test")
    parser.add_argument("--nsims", type=int, default=1000,
                        help="Number of simulations (default: 1000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (default: 42)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: cpu_count)")
    args = parser.parse_args()

    n_sims = args.nsims
    base_seed = args.seed
    n_workers = args.workers or max(1, cpu_count() - 1)

    print("=" * 72)
    print("GLOBAL PIPELINE MONTE CARLO TEST")
    print("Garden-of-forking-paths defense: joint probability of all 4 claims")
    print("=" * 72)

    # ── Step 1: Real data metrics ────────────────────────────────────────────
    real = compute_real_metrics()

    print(f"\n--- Real Mersenne Prime Sequence (52 exponents) ---")
    print(f"  Threshold tau:    {real['tau']}")
    print(f"  Clusters:         {real['n_clusters']}")
    print(f"  Singletons:       {real['n_singletons']}")
    print(f"  Staircase R^2:    {real['r2']:.6f}")
    print(f"  DAIC:             {real['daic']:.2f}")
    print(f"  |unity - 1|:      {real['unity_dist']:.4f}")
    print(f"  |Spearman rho|:   {real['spearman_abs']:.4f}")

    print(f"\n--- Thresholds for synthetic sequences to beat ---")
    print(f"  R^2          >= {REAL_R2}")
    print(f"  DAIC         >= {REAL_DAIC}")
    print(f"  |unity - 1|  <= {REAL_UNITY_DIST}")
    print(f"  |Spearman r| <= {REAL_SPEARMAN_ABS}")

    # ── Step 2: Run Monte Carlo ──────────────────────────────────────────────
    print(f"\n--- Running {n_sims:,} simulations (seed={base_seed}, workers={n_workers}) ---")

    task_args = [
        (base_seed + i, 52, 200_000_000)
        for i in range(n_sims)
    ]

    t0 = time.time()

    with Pool(n_workers) as pool:
        results = pool.map(simulate_one, task_args, chunksize=max(1, n_sims // (n_workers * 4)))

    elapsed = time.time() - t0

    # ── Step 3: Analyze results ──────────────────────────────────────────────
    valid_results = [r for r in results if r is not None]
    n_valid = len(valid_results)
    n_skipped = n_sims - n_valid

    r2_vals = np.array([r["r2"] for r in valid_results])
    daic_vals = np.array([r["daic"] for r in valid_results])
    unity_vals = np.array([r["unity_dist"] for r in valid_results])
    spearman_vals = np.array([r["spearman_abs"] for r in valid_results])
    n_singletons_vals = np.array([r["n_singletons"] for r in valid_results])

    # Individual metric pass rates
    beat_r2 = r2_vals >= REAL_R2
    beat_daic = daic_vals >= REAL_DAIC
    beat_unity = unity_vals <= REAL_UNITY_DIST
    beat_spearman = spearman_vals <= REAL_SPEARMAN_ABS

    # Joint pass
    beat_all = beat_r2 & beat_daic & beat_unity & beat_spearman

    # Fractions
    frac_r2 = float(beat_r2.sum()) / n_valid if n_valid > 0 else 0
    frac_daic = float(beat_daic.sum()) / n_valid if n_valid > 0 else 0
    frac_unity = float(beat_unity.sum()) / n_valid if n_valid > 0 else 0
    frac_spearman = float(beat_spearman.sum()) / n_valid if n_valid > 0 else 0
    frac_all = float(beat_all.sum()) / n_valid if n_valid > 0 else 0

    # How many have >= 4 singletons
    has_4_singletons = int((n_singletons_vals >= 4).sum())

    # ── Step 4: Print results ────────────────────────────────────────────────
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Valid simulations: {n_valid:,} / {n_sims:,} ({n_skipped:,} skipped)")
    print(f"Sequences with >= 4 singletons: {has_4_singletons:,} / {n_valid:,}")

    print(f"\n{'=' * 72}")
    print("INDIVIDUAL METRIC PASS RATES (post-hoc best threshold)")
    print(f"{'=' * 72}")
    print(f"  {'Metric':<20s}  {'Threshold':>12s}  {'Count':>8s}  {'Fraction':>10s}")
    print(f"  {'-'*20}  {'-'*12}  {'-'*8}  {'-'*10}")
    print(f"  {'R^2':<20s}  {'>= ' + str(REAL_R2):>12s}  {int(beat_r2.sum()):>8,}  {frac_r2:>10.6f}")
    print(f"  {'DAIC':<20s}  {'>= ' + str(REAL_DAIC):>12s}  {int(beat_daic.sum()):>8,}  {frac_daic:>10.6f}")
    print(f"  {'|unity-1|':<20s}  {'<= ' + str(REAL_UNITY_DIST):>12s}  {int(beat_unity.sum()):>8,}  {frac_unity:>10.6f}")
    print(f"  {'|Spearman r|':<20s}  {'<= ' + str(REAL_SPEARMAN_ABS):>12s}  {int(beat_spearman.sum()):>8,}  {frac_spearman:>10.6f}")

    print(f"\n{'=' * 72}")
    print("GLOBAL p-VALUE (joint probability of beating ALL 4 metrics)")
    print(f"{'=' * 72}")
    print(f"  Sequences beating all 4: {int(beat_all.sum()):,} / {n_valid:,}")
    print(f"  GLOBAL p-value:          {frac_all:.6f}")

    if frac_all == 0:
        print(f"  Upper bound (1/N):       {1.0/n_valid:.6f}")
        print(f"\n  No synthetic sequence jointly matched all 4 metrics.")
        print(f"  p < {1.0/n_valid:.4f} (conservative upper bound)")

    # ── Distribution summaries ───────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("DISTRIBUTION OF METRICS ACROSS SYNTHETIC SEQUENCES")
    print(f"{'=' * 72}")

    def print_dist(name, vals, real_val, better_dir):
        p5, p50, p95 = np.percentile(vals, [5, 50, 95])
        print(f"\n  {name}:")
        print(f"    Mean:   {vals.mean():.6f}")
        print(f"    Median: {p50:.6f}")
        print(f"    5th %%:  {p5:.6f}")
        print(f"    95th %%: {p95:.6f}")
        print(f"    Real:   {real_val:.6f}  ({better_dir})")

    print_dist("Staircase R^2", r2_vals, real["r2"], "higher is better")
    print_dist("DAIC", daic_vals, real["daic"], "higher is better")
    print_dist("|unity - 1|", unity_vals, real["unity_dist"], "lower is better")
    print_dist("|Spearman r|", spearman_vals, real["spearman_abs"], "lower is better")

    # ── Pairwise joint rates ─────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("PAIRWISE JOINT PASS RATES")
    print(f"{'=' * 72}")
    labels = ["R^2", "DAIC", "|unity-1|", "|Spearman|"]
    masks = [beat_r2, beat_daic, beat_unity, beat_spearman]
    for i in range(4):
        for j in range(i + 1, 4):
            joint = (masks[i] & masks[j]).sum()
            frac = joint / n_valid if n_valid > 0 else 0
            print(f"  {labels[i]:>12s} & {labels[j]:<12s}: {int(joint):>6,} / {n_valid:,} = {frac:.6f}")

    # Triple joint
    for i in range(4):
        triple = np.ones(n_valid, dtype=bool)
        skip_label = labels[i]
        included = []
        for j in range(4):
            if j != i:
                triple &= masks[j]
                included.append(labels[j])
        frac = triple.sum() / n_valid if n_valid > 0 else 0
        print(f"  {' & '.join(included)}: {int(triple.sum()):>6,} / {n_valid:,} = {frac:.6f}")

    # ── Save to JSON ─────────────────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "global_mc_results.json")

    output = {
        "config": {
            "n_sims": n_sims,
            "base_seed": base_seed,
            "n_workers": n_workers,
            "n_valid": n_valid,
            "n_skipped": n_skipped,
            "elapsed_seconds": round(elapsed, 1),
        },
        "real_data": {
            "r2": real["r2"],
            "daic": real["daic"],
            "unity_dist": real["unity_dist"],
            "spearman_abs": real["spearman_abs"],
            "tau": real["tau"],
            "n_clusters": real["n_clusters"],
            "n_singletons": real["n_singletons"],
        },
        "thresholds": {
            "r2_ge": REAL_R2,
            "daic_ge": REAL_DAIC,
            "unity_dist_le": REAL_UNITY_DIST,
            "spearman_abs_le": REAL_SPEARMAN_ABS,
        },
        "results": {
            "global_p_value": frac_all,
            "global_count": int(beat_all.sum()),
            "individual_p_values": {
                "r2": frac_r2,
                "daic": frac_daic,
                "unity_dist": frac_unity,
                "spearman_abs": frac_spearman,
            },
            "individual_counts": {
                "r2": int(beat_r2.sum()),
                "daic": int(beat_daic.sum()),
                "unity_dist": int(beat_unity.sum()),
                "spearman_abs": int(beat_spearman.sum()),
            },
            "sequences_with_4plus_singletons": has_4_singletons,
        },
        "distributions": {
            "r2": {
                "mean": float(r2_vals.mean()),
                "median": float(np.median(r2_vals)),
                "p5": float(np.percentile(r2_vals, 5)),
                "p95": float(np.percentile(r2_vals, 95)),
            },
            "daic": {
                "mean": float(daic_vals.mean()),
                "median": float(np.median(daic_vals)),
                "p5": float(np.percentile(daic_vals, 5)),
                "p95": float(np.percentile(daic_vals, 95)),
            },
            "unity_dist": {
                "mean": float(unity_vals.mean()),
                "median": float(np.median(unity_vals)),
                "p5": float(np.percentile(unity_vals, 5)),
                "p95": float(np.percentile(unity_vals, 95)),
            },
            "spearman_abs": {
                "mean": float(spearman_vals.mean()),
                "median": float(np.median(spearman_vals)),
                "p5": float(np.percentile(spearman_vals, 5)),
                "p95": float(np.percentile(spearman_vals, 95)),
            },
        },
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {out_path}")

    # ── Final verdict ────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("VERDICT")
    print(f"{'=' * 72}")
    if frac_all == 0:
        print(f"  In {n_valid:,} synthetic Wagstaff-Poisson sequences (with post-hoc")
        print(f"  threshold optimization), ZERO jointly matched or exceeded the real")
        print(f"  Mersenne prime sequence on all 4 metrics.")
        print(f"")
        print(f"  Global p < {1.0/n_valid:.4f}")
        print(f"")
        print(f"  The observed regularities are NOT an artifact of multiple testing")
        print(f"  or threshold selection. Even granting the analyst full freedom to")
        print(f"  choose tau post-hoc, the joint structure is significant.")
    else:
        print(f"  Global p-value = {frac_all:.6f}")
        print(f"  ({int(beat_all.sum())} of {n_valid:,} synthetic sequences jointly beat all 4 metrics)")

    print()


if __name__ == "__main__":
    main()
