#!/usr/bin/env python3
"""
Claim 2: Escape velocity shows no detectable scale dependence.

The escape velocity (geometric mean of inter-cluster jump ratios) is
approximately constant across the full exponent range, with no trend
as cluster index increases.

Expected results:
  - Pearson r = -0.007, p = 0.977 (no correlation with cluster index)
  - Escape velocity ≈ 2.015
"""

import sys
import os
import math

import numpy as np
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from staircase_utils import load_mersenne_exponents, classify_clusters, compute_escape_velocity

def main():
    exponents = load_mersenne_exponents()
    clusters = classify_clusters(exponents)

    print("=" * 70)
    print("CLAIM 2: ESCAPE VELOCITY SCALE INDEPENDENCE")
    print("=" * 70)

    # Compute individual jump ratios
    jumps = []
    for i in range(1, len(clusters)):
        ratio = clusters[i][0] / clusters[i - 1][-1]
        jumps.append({
            "from_cluster": i - 1,
            "to_cluster": i,
            "from_exp": clusters[i - 1][-1],
            "to_exp": clusters[i][0],
            "ratio": ratio,
            "log_ratio": math.log(ratio),
        })

    # Overall escape velocity
    escape = compute_escape_velocity(clusters)
    print(f"\nOverall escape velocity: {escape:.4f}")
    print(f"Number of inter-cluster jumps: {len(jumps)}")

    # Test for trend: correlate jump ratio with cluster index
    indices = np.array([j["from_cluster"] for j in jumps])
    log_ratios = np.array([j["log_ratio"] for j in jumps])

    r, p = sp_stats.pearsonr(indices, log_ratios)
    print(f"\nScale dependence test (Pearson correlation):")
    print(f"  r = {r:.3f}")
    print(f"  p = {p:.3f}")

    if p > 0.05:
        print(f"  Result: NO significant trend (p > 0.05)")
    else:
        print(f"  Result: Significant trend detected (p < 0.05)")

    # Also test with Spearman (nonparametric)
    rho, p_spearman = sp_stats.spearmanr(indices, log_ratios)
    print(f"\nSpearman rank correlation:")
    print(f"  rho = {rho:.3f}")
    print(f"  p = {p_spearman:.3f}")

    # Show individual jumps
    print(f"\n{'Jump':>4}  {'From':>12}  {'To':>12}  {'Ratio':>8}  {'ln(ratio)':>10}")
    print("-" * 55)
    for j in jumps:
        print(f"  {j['from_cluster']:>2}→{j['to_cluster']:<2}"
              f"  {j['from_exp']:>12,}  {j['to_exp']:>12,}"
              f"  {j['ratio']:>8.3f}  {j['log_ratio']:>10.4f}")

    # Split into halves and compare
    mid = len(jumps) // 2
    early_log = [j["log_ratio"] for j in jumps[:mid]]
    late_log = [j["log_ratio"] for j in jumps[mid:]]

    early_esc = math.exp(np.mean(early_log))
    late_esc = math.exp(np.mean(late_log))

    print(f"\nEarly clusters (0-{mid-1}): escape = {early_esc:.4f}")
    print(f"Late clusters ({mid}-{len(jumps)-1}):  escape = {late_esc:.4f}")
    print(f"Difference: {abs(early_esc - late_esc):.4f} ({abs(early_esc - late_esc) / escape * 100:.1f}%)")

    # Verification
    print(f"\n{'=' * 70}")
    print("VERIFICATION")
    print("=" * 70)
    checks = [
        ("Escape velocity", escape, 2.015, 0.01),
        ("Pearson |r| < 0.1", abs(r), 0.0, 0.1),
        ("Pearson p > 0.05", p, 1.0, 0.95),  # just check p > 0.05
    ]
    all_ok = True
    for name, got, expected, tol in checks:
        ok = abs(got - expected) < tol
        status = "OK" if ok else "MISMATCH"
        print(f"  {name}: got {got:.4f}, expected ~{expected}, [{status}]")
        if not ok:
            all_ok = False

    if all_ok:
        print("\n  All checks passed.")
    else:
        print("\n  WARNING: Some checks failed — investigate.")


if __name__ == "__main__":
    main()
