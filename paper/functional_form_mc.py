#!/usr/bin/env python3
"""
Functional-form look-elsewhere Monte Carlo test.

Addresses the referee's central critique: the unity equation d^2*v*b ~ 1 was
chosen post-hoc from a large space of possible monomial forms d^a * v^b * b^c.
How special is it really?

Approach:
1. For the REAL data: enumerate all d^a * v^b * b^c with integer exponents
   a,b,c in [-3,3] (excluding 0,0,0). Record how many land within 5% of a
   "nice number" (integers 1-20, unit fractions 1/2 through 1/10).

2. For 1,000 synthetic Wagstaff sequences: find the best threshold, compute
   d, v, b, search ALL monomial forms, record the minimum distance to any
   nice number.

3. Report: what fraction of synthetic sequences produce a monomial product
   as close to a nice number as the real data's best (d^2*v*b = 0.956,
   distance 0.044 from 1)?
"""

import json
import math
import os
import sys
import time
from itertools import product as iterproduct

import numpy as np

# Add parent to path for staircase_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from staircase_utils import (
    classify_clusters,
    compute_escape_velocity,
    compute_orbital_base,
    consecutive_pairing,
    generate_wagstaff_sequence,
    get_singletons,
    load_mersenne_exponents,
)

# ─── Nice numbers ─────────────────────────────────────────────────────────────

NICE_NUMBERS = list(range(1, 21)) + [1/k for k in range(2, 11)]
# integers 1-20 plus unit fractions 1/2, 1/3, ..., 1/10


def min_relative_distance_to_nice(value):
    """Minimum relative distance |value - nice| / nice across all nice numbers."""
    if value <= 0 or not math.isfinite(value):
        return float('inf')
    return min(abs(value - n) / n for n in NICE_NUMBERS)


# ─── Monomial enumeration ─────────────────────────────────────────────────────

def get_exponent_triples(lo=-3, hi=3):
    """All (a,b,c) in [lo,hi]^3 excluding (0,0,0)."""
    triples = []
    for a, b, c in iterproduct(range(lo, hi + 1), repeat=3):
        if a == 0 and b == 0 and c == 0:
            continue
        triples.append((a, b, c))
    return triples


def compute_monomial(d, v, b, a, eb, ec):
    """Compute d^a * v^eb * b^ec."""
    try:
        val = (d ** a) * (v ** eb) * (b ** ec)
        if not math.isfinite(val) or val <= 0:
            return None
        return val
    except (OverflowError, ZeroDivisionError, ValueError):
        return None


# ─── Real data analysis ───────────────────────────────────────────────────────

def analyze_real_data(exponents):
    """Analyze all monomial forms for the real Mersenne prime data."""
    clusters = classify_clusters(exponents, tau=1.501)
    b = compute_orbital_base(clusters)
    v = compute_escape_velocity(clusters)
    singletons = get_singletons(clusters)
    result = consecutive_pairing(singletons)
    if result is None:
        raise ValueError("Cannot compute decay for real data")
    _, _, d = result

    triples = get_exponent_triples()
    results = []
    best_dist = float('inf')
    best_triple = None
    best_value = None
    best_target = None

    for (ea, eb, ec) in triples:
        val = compute_monomial(d, v, b, ea, eb, ec)
        if val is None:
            continue
        dist = min_relative_distance_to_nice(val)
        # Find which nice number is closest
        closest_nice = min(NICE_NUMBERS, key=lambda n: abs(val - n) / n)
        rel_dist = abs(val - closest_nice) / closest_nice

        results.append({
            "exponents": [ea, eb, ec],
            "value": val,
            "closest_nice": closest_nice,
            "relative_distance": rel_dist,
        })

        if dist < best_dist:
            best_dist = dist
            best_triple = (ea, eb, ec)
            best_value = val
            best_target = closest_nice

    # Sort by distance
    results.sort(key=lambda r: r["relative_distance"])

    # Count within 5%
    within_5pct = sum(1 for r in results if r["relative_distance"] < 0.05)

    # The specific d^2*v*b value
    unity_val = d**2 * v * b
    unity_dist = min_relative_distance_to_nice(unity_val)

    return {
        "d": d, "v": v, "b": b,
        "unity_value": unity_val,
        "unity_relative_distance": unity_dist,
        "total_monomials": len(results),
        "within_5pct": within_5pct,
        "within_5pct_fraction": within_5pct / len(results) if results else 0,
        "best_monomial": {
            "exponents": list(best_triple) if best_triple else None,
            "value": best_value,
            "closest_nice": best_target,
            "relative_distance": best_dist,
        },
        "top_10": [
            {
                "exponents": r["exponents"],
                "value": round(r["value"], 6),
                "closest_nice": r["closest_nice"],
                "relative_distance": round(r["relative_distance"], 6),
                "label": f"d^{r['exponents'][0]} * v^{r['exponents'][1]} * b^{r['exponents'][2]}"
            }
            for r in results[:10]
        ],
        # Where does d^2*v*b rank?
        "unity_rank": next(
            (i + 1 for i, r in enumerate(results)
             if r["exponents"] == [2, 1, 1]),
            None
        ),
    }


# ─── Synthetic sequence analysis ─────────────────────────────────────────────

def find_best_threshold(exponents):
    """Find the threshold that minimizes |unity - 1| across natural breaks."""
    if len(exponents) < 6:
        return None, None, None, None, None

    # Get all unique consecutive ratios as candidate thresholds
    ratios = [exponents[i] / exponents[i - 1] for i in range(1, len(exponents))]
    unique_ratios = sorted(set(ratios))

    # Test thresholds just above each ratio
    best_tau = None
    best_unity_dist = float('inf')
    best_d = best_v = best_b = None

    for r in unique_ratios:
        tau = r + 0.001
        if tau < 1.05 or tau > 5.0:
            continue

        cls = classify_clusters(exponents, tau=tau)
        if len(cls) < 5:
            continue

        b = compute_orbital_base(cls)
        v = compute_escape_velocity(cls)
        if b is None or v is None or b <= 0 or v <= 0:
            continue

        singles = get_singletons(cls)
        if len(singles) < 4:
            continue

        result = consecutive_pairing(singles)
        if result is None:
            continue

        _, _, d = result
        if d <= 0:
            continue

        unity = d**2 * v * b
        dist = abs(unity - 1.0)
        if dist < best_unity_dist:
            best_unity_dist = dist
            best_tau = tau
            best_d, best_v, best_b = d, v, b

    return best_tau, best_d, best_v, best_b, best_unity_dist


def analyze_synthetic(exponents):
    """
    For a synthetic sequence: find best threshold, then search ALL monomial
    forms for minimum distance to any nice number.
    """
    tau, d, v, b = find_best_threshold(exponents)[:4]
    if d is None or v is None or b is None:
        return None
    if d <= 0 or v <= 0 or b <= 0:
        return None

    triples = get_exponent_triples()
    best_dist = float('inf')

    for (ea, eb, ec) in triples:
        val = compute_monomial(d, v, b, ea, eb, ec)
        if val is None:
            continue
        dist = min_relative_distance_to_nice(val)
        if dist < best_dist:
            best_dist = dist

    return best_dist


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("FUNCTIONAL-FORM LOOK-ELSEWHERE MONTE CARLO TEST")
    print("=" * 70)

    exponents = load_mersenne_exponents()

    # ── Part 1: Real data analysis ──
    print("\n--- Part 1: Real Data Analysis ---")
    real = analyze_real_data(exponents)
    print(f"  d = {real['d']:.4f}, v = {real['v']:.4f}, b = {real['b']:.4f}")
    print(f"  Unity product d^2*v*b = {real['unity_value']:.4f}")
    print(f"  Unity relative distance to 1 = {real['unity_relative_distance']:.4f}")
    print(f"  Total monomials evaluated: {real['total_monomials']}")
    print(f"  Within 5% of a nice number: {real['within_5pct']} ({real['within_5pct_fraction']:.1%})")
    print(f"  d^2*v*b rank among all monomials: #{real['unity_rank']}")
    print(f"\n  Best monomial: d^{real['best_monomial']['exponents'][0]} * "
          f"v^{real['best_monomial']['exponents'][1]} * "
          f"b^{real['best_monomial']['exponents'][2]} = "
          f"{real['best_monomial']['value']:.6f} "
          f"(target {real['best_monomial']['closest_nice']}, "
          f"dist {real['best_monomial']['relative_distance']:.4%})")

    print("\n  Top 10 closest monomials:")
    for i, entry in enumerate(real['top_10'], 1):
        print(f"    {i:2d}. {entry['label']:25s} = {entry['value']:12.6f}  "
              f"-> {entry['closest_nice']:6} (dist {entry['relative_distance']:.4%})")

    # The real data's best distance across ALL monomials
    real_best_dist = real['best_monomial']['relative_distance']
    # The specific d^2*v*b distance
    real_unity_dist = real['unity_relative_distance']

    # ── Part 2: Synthetic MC ──
    N_SIMS = 1000
    print(f"\n--- Part 2: Synthetic Monte Carlo (N={N_SIMS}) ---")
    rng = np.random.default_rng(42)

    synthetic_best_dists = []
    n_valid = 0
    n_beat_unity = 0  # synthetic best dist <= real unity dist (0.044)
    n_beat_best = 0   # synthetic best dist <= real best dist (across ALL monomials)

    t0 = time.time()
    for i in range(N_SIMS):
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (N_SIMS - i - 1) / rate
            print(f"  Sim {i+1}/{N_SIMS} ({elapsed:.1f}s elapsed, ~{remaining:.0f}s remaining)")

        exps = generate_wagstaff_sequence(rng, n_target=52, p_max=200_000_000)
        dist = analyze_synthetic(exps)
        if dist is None:
            continue

        n_valid += 1
        synthetic_best_dists.append(dist)

        if dist <= real_unity_dist:
            n_beat_unity += 1
        if dist <= real_best_dist:
            n_beat_best += 1

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s")
    print(f"  Valid simulations: {n_valid}/{N_SIMS}")

    synthetic_best_dists = np.array(synthetic_best_dists)

    # ── Results ──
    print("\n--- Results ---")
    p_unity = n_beat_unity / n_valid if n_valid > 0 else None
    p_best = n_beat_best / n_valid if n_valid > 0 else None

    print(f"  Real data d^2*v*b distance to 1: {real_unity_dist:.4f}")
    print(f"  Real data best monomial distance: {real_best_dist:.6f}")
    print(f"  Synthetic sequences with ANY monomial as close as d^2*v*b is to 1: "
          f"{n_beat_unity}/{n_valid} (p = {p_unity:.4f})" if p_unity is not None else "  N/A")
    print(f"  Synthetic sequences with ANY monomial as close as real best: "
          f"{n_beat_best}/{n_valid} (p = {p_best:.4f})" if p_best is not None else "  N/A")

    if len(synthetic_best_dists) > 0:
        print(f"\n  Synthetic best-distance distribution:")
        print(f"    Mean:   {synthetic_best_dists.mean():.4f}")
        print(f"    Median: {np.median(synthetic_best_dists):.4f}")
        print(f"    5th pct:  {np.percentile(synthetic_best_dists, 5):.4f}")
        print(f"    95th pct: {np.percentile(synthetic_best_dists, 95):.4f}")

    # ── Save results ──
    output = {
        "description": (
            "Functional-form look-elsewhere test. For each synthetic Wagstaff sequence, "
            "the best threshold is found, d/v/b computed, and ALL monomial forms "
            "d^a * v^b * b^c (a,b,c in [-3,3], excl. 0,0,0) are searched for the "
            "minimum relative distance to any 'nice number' (integers 1-20, unit "
            "fractions 1/2 through 1/10)."
        ),
        "real_data": {
            "d": round(real['d'], 6),
            "v": round(real['v'], 6),
            "b": round(real['b'], 6),
            "unity_product": round(real['unity_value'], 6),
            "unity_relative_distance": round(real_unity_dist, 6),
            "total_monomials": real['total_monomials'],
            "within_5pct_of_nice": real['within_5pct'],
            "within_5pct_fraction": round(real['within_5pct_fraction'], 4),
            "best_monomial": {
                "exponents": real['best_monomial']['exponents'],
                "value": round(real['best_monomial']['value'], 6),
                "closest_nice": real['best_monomial']['closest_nice'],
                "relative_distance": round(real_best_dist, 6),
            },
            "unity_rank_among_all_monomials": real['unity_rank'],
            "top_10": real['top_10'],
        },
        "monte_carlo": {
            "n_sims": N_SIMS,
            "n_valid": n_valid,
            "seed": 42,
            "runtime_seconds": round(elapsed, 1),
            "n_beat_unity_distance": n_beat_unity,
            "p_value_vs_unity": round(p_unity, 6) if p_unity is not None else None,
            "n_beat_best_distance": n_beat_best,
            "p_value_vs_best": round(p_best, 6) if p_best is not None else None,
            "synthetic_best_distance_stats": {
                "mean": round(float(synthetic_best_dists.mean()), 6) if len(synthetic_best_dists) > 0 else None,
                "median": round(float(np.median(synthetic_best_dists)), 6) if len(synthetic_best_dists) > 0 else None,
                "p5": round(float(np.percentile(synthetic_best_dists, 5)), 6) if len(synthetic_best_dists) > 0 else None,
                "p95": round(float(np.percentile(synthetic_best_dists, 95)), 6) if len(synthetic_best_dists) > 0 else None,
            },
        },
        "interpretation": (
            "p_value_vs_unity: fraction of synthetic sequences where the best monomial "
            "(across all 342 forms) lands as close to a nice number as the real data's "
            "d^2*v*b = 0.956 lands to 1 (distance 4.4%). This is the key metric -- it "
            "tells us whether the unity equation is special even after accounting for "
            "the freedom to choose among hundreds of functional forms."
        ),
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "functional_form_results.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    return output


if __name__ == "__main__":
    main()
