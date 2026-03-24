#!/usr/bin/env python3
"""
Claim 4 — Unity Equation Reproducibility Script
================================================

Claim: "Consecutive pairing of 7 singletons produces unity = 0.956,
        |unity - 1| = 0.044, significant at p < 0.01."

The unity equation:  decay^2 * escape * orbital = 1

Where:
  - decay   = mean ratio of consecutive log10-gap pairs among singletons
  - escape  = geometric mean of inter-cluster jump ratios
  - orbital = exponential growth rate of cluster geometric centers

This script:
  1. Computes the observed unity value from the 52 known Mersenne primes.
  2. Runs 500K Monte Carlo simulations (5 seeds x 100K) using the
     Wagstaff/Poisson null model to estimate p-value with variance.
  3. Reports p-values at multiple thresholds for transparency.

Monte Carlo p-values depend on random seed and simulation count. We report
results across 5 independent seeds (500K total simulations) to demonstrate
stability. Individual 10K-simulation runs may vary by +/-50% around the mean.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from staircase_utils import (
    load_mersenne_exponents, classify_clusters, get_singletons,
    compute_orbital_base, compute_escape_velocity,
    consecutive_pairing, unity_equation,
    run_unity_mc, TAU_DEFAULT
)

import numpy as np


# ─── Configuration ────────────────────────────────────────────────────────────

SEEDS = [42, 137, 256, 512, 1024]
N_SIMS_PER_SEED = 100_000
P_THRESHOLDS = [0.01, 0.02, 0.044, 0.05, 0.10]


def main():
    print("=" * 65)
    print("Claim 4: Unity Equation  (decay^2 * escape * orbital = 1)")
    print("=" * 65)

    # ─── Step 1: Observed values ──────────────────────────────────────────────

    exponents = load_mersenne_exponents()
    clusters = classify_clusters(exponents, tau=TAU_DEFAULT)
    singletons = get_singletons(clusters)
    multi = [c for c in clusters if len(c) > 1]

    orbital = compute_orbital_base(clusters)
    escape = compute_escape_velocity(clusters)
    pairing = consecutive_pairing(singletons)

    if pairing is None:
        print("ERROR: consecutive_pairing returned None")
        sys.exit(1)

    gaps, decay_rates, mean_decay = pairing
    observed_unity = unity_equation(mean_decay, escape, orbital)
    observed_dist = abs(observed_unity - 1.0)

    print(f"\nData: {len(exponents)} known Mersenne prime exponents")
    print(f"Clusters: {len(clusters)} total, {len(multi)} multi, "
          f"{len(singletons)} singletons")
    print(f"Tau (threshold): {TAU_DEFAULT}")

    print(f"\nSingletons: {singletons}")
    print(f"Gaps (log10 pairs):  {[round(g, 3) for g in gaps]}")
    print(f"Decay rates:         {[round(d, 3) for d in decay_rates]}")
    print(f"Mean decay:          {mean_decay:.4f}")
    print(f"Escape velocity:     {escape:.4f}")
    print(f"Orbital base:        {orbital:.4f}")

    print(f"\nUnity = {mean_decay:.4f}^2 * {escape:.4f} * {orbital:.4f} "
          f"= {observed_unity:.4f}")
    print(f"|unity - 1| = {observed_dist:.4f}")

    # ─── Verify expected numbers ──────────────────────────────────────────────

    checks_passed = True

    if len(clusters) != 19:
        print(f"\nWARNING: Expected 19 clusters, got {len(clusters)}")
        checks_passed = False

    if len(singletons) != 7:
        print(f"\nWARNING: Expected 7 singletons, got {len(singletons)}")
        checks_passed = False

    expected_gaps = [1.616, 0.686, 0.286]
    for i, (got, want) in enumerate(zip(gaps, expected_gaps)):
        if abs(got - want) > 0.01:
            print(f"\nWARNING: Gap {i} = {got:.3f}, expected ~{want}")
            checks_passed = False

    expected_decays = [0.425, 0.417]
    for i, (got, want) in enumerate(zip(decay_rates, expected_decays)):
        if abs(got - want) > 0.01:
            print(f"\nWARNING: Decay rate {i} = {got:.3f}, expected ~{want}")
            checks_passed = False

    if abs(mean_decay - 0.4207) > 0.01:
        print(f"\nWARNING: Mean decay = {mean_decay:.4f}, expected ~0.4207")
        checks_passed = False

    if abs(observed_unity - 0.956) > 0.01:
        print(f"\nWARNING: Unity = {observed_unity:.4f}, expected ~0.956")
        checks_passed = False

    if abs(observed_dist - 0.044) > 0.01:
        print(f"\nWARNING: |unity-1| = {observed_dist:.4f}, expected ~0.044")
        checks_passed = False

    if checks_passed:
        print("\nAll observed values match expected numbers.")

    # ─── Step 2: Multi-seed Monte Carlo ───────────────────────────────────────

    print(f"\n{'=' * 65}")
    print(f"Monte Carlo: {len(SEEDS)} seeds x {N_SIMS_PER_SEED:,} sims "
          f"= {len(SEEDS) * N_SIMS_PER_SEED:,} total")
    print(f"Null model: Wagstaff/Poisson (uniform in log-log space)")
    print(f"{'=' * 65}")

    seed_results = []
    all_dists = []

    for seed in SEEDS:
        t0 = time.time()
        dists = run_unity_mc(N_SIMS_PER_SEED, seed, tau=TAU_DEFAULT)
        elapsed = time.time() - t0

        n_valid = len(dists)
        p_value = float(np.sum(dists <= observed_dist) / n_valid) if n_valid > 0 else float("nan")

        seed_results.append({
            "seed": seed,
            "n_valid": n_valid,
            "p_value": p_value,
            "elapsed": elapsed,
        })
        all_dists.append(dists)

        print(f"  Seed {seed:>5d}: {n_valid:>7,} valid, "
              f"p = {p_value:.4f}  ({elapsed:.1f}s)")

    # ─── Variance table ──────────────────────────────────────────────────────

    p_values = np.array([r["p_value"] for r in seed_results])
    p_mean = float(p_values.mean())
    p_std = float(p_values.std())

    print(f"\n{'Seed':>6s}    {'N_valid':>9s}    {'p(\u22640.044)':>10s}")
    print("\u2500" * 38)
    for r in seed_results:
        print(f"{r['seed']:>6d}    {r['n_valid']:>9,}    {r['p_value']:>10.4f}")
    print("\u2500" * 38)
    print(f"{'Mean':>6s}    {'':>9s}    {p_mean:.4f} \u00b1 {p_std:.4f}")

    # ─── Step 3: P-values at various thresholds ──────────────────────────────

    combined = np.concatenate(all_dists)
    n_combined = len(combined)

    print(f"\nCombined: {n_combined:,} valid simulations")
    print(f"\n{'Threshold':>10s}    {'p-value':>10s}    {'Count':>8s}")
    print("\u2500" * 38)
    for thresh in P_THRESHOLDS:
        count = int(np.sum(combined <= thresh))
        p = count / n_combined if n_combined > 0 else float("nan")
        marker = " <-- observed" if abs(thresh - observed_dist) < 0.001 else ""
        print(f"{thresh:>10.3f}    {p:>10.4f}    {count:>8,}{marker}")

    # ─── Summary ──────────────────────────────────────────────────────────────

    print(f"\n{'=' * 65}")
    print("SUMMARY")
    print(f"{'=' * 65}")
    print(f"Observed unity:      {observed_unity:.4f}")
    print(f"|unity - 1|:         {observed_dist:.4f}")
    print(f"p-value (combined):  {float(np.sum(combined <= observed_dist) / n_combined):.4f}")
    print(f"p-value (mean+/-sd): {p_mean:.4f} +/- {p_std:.4f}")

    if p_mean < 0.01:
        print(f"\nClaim SUPPORTED: p < 0.01 (mean across {len(SEEDS)} seeds)")
    elif p_mean < 0.05:
        print(f"\nClaim PARTIALLY SUPPORTED: p < 0.05 but not < 0.01")
    else:
        print(f"\nClaim NOT SUPPORTED at p < 0.05")

    print(f"""
NOTE: Monte Carlo p-values depend on random seed and simulation count. We report
results across {len(SEEDS)} independent seeds ({len(SEEDS) * N_SIMS_PER_SEED:,} total simulations) to demonstrate
stability. Individual 10K-simulation runs may vary by +/-50% around the mean.""")


if __name__ == "__main__":
    main()
