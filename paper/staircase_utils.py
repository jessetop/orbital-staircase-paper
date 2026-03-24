"""
Shared utilities for reproducing Orbital Staircase paper claims.

All paper scripts import from this module to ensure consistent
cluster classification, constant computation, and Monte Carlo methods.

Threshold convention: tau = 1.501 (just above 3/2). This handles the
exact ratio M(2)/M(3) = 5/3 ≈ 1.667 correctly, but more importantly
avoids a boundary artifact where M(2)->M(3) has ratio exactly 1.500
which would be missed by strict > 1.5.
"""

import json
import math
import os

import numpy as np

# ─── Constants ────────────────────────────────────────────────────────────────

TAU_DEFAULT = 1.501  # cluster threshold, just above 3/2
EGAMMA_OVER_LN2 = math.exp(0.5772156649) / math.log(2)  # Wagstaff constant ~2.569

# ─── Data loading ─────────────────────────────────────────────────────────────

def _data_path(filename):
    """Resolve path relative to repo root."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "data", filename)


def load_mersenne_exponents():
    """Load the 52 known Mersenne prime exponents from data/mersenne_exponents.json."""
    with open(_data_path("mersenne_exponents.json")) as f:
        return json.load(f)


# ─── Clustering ───────────────────────────────────────────────────────────────

def classify_clusters(exponents, tau=TAU_DEFAULT):
    """
    Group exponents into clusters by consecutive ratio threshold.

    Two consecutive exponents p_i, p_{i+1} are in the same cluster if
    p_{i+1} / p_i <= tau.  Otherwise p_{i+1} starts a new cluster.

    Returns list of clusters, each a list of exponents.
    """
    if len(exponents) < 2:
        return [list(exponents)] if exponents else []

    clusters = [[exponents[0]]]
    for i in range(1, len(exponents)):
        if exponents[i] / exponents[i - 1] > tau:
            clusters.append([exponents[i]])
        else:
            clusters[-1].append(exponents[i])
    return clusters


def get_singletons(clusters):
    """Return exponents of clusters with exactly one member."""
    return [c[0] for c in clusters if len(c) == 1]


def get_multi_clusters(clusters):
    """Return clusters with more than one member."""
    return [c for c in clusters if len(c) > 1]


# ─── Staircase constants ─────────────────────────────────────────────────────

def compute_orbital_base(clusters):
    """
    Orbital base: exponential growth rate of cluster geometric centers.

    Fits log(center_n) = intercept + n * log(base) via OLS.
    Returns base = exp(slope).
    """
    if len(clusters) < 3:
        return None
    centers = [math.exp(sum(math.log(x) for x in c) / len(c)) for c in clusters]
    log_centers = np.array([math.log(c) for c in centers])
    x = np.arange(len(log_centers))
    x_mean = x.mean()
    y_mean = log_centers.mean()
    ss_xx = np.sum((x - x_mean) ** 2)
    if ss_xx == 0:
        return None
    slope = np.sum((x - x_mean) * (log_centers - y_mean)) / ss_xx
    return math.exp(slope)


def compute_orbital_r_squared(clusters):
    """R-squared of the log-linear fit of cluster centers."""
    if len(clusters) < 3:
        return None
    centers = [math.exp(sum(math.log(x) for x in c) / len(c)) for c in clusters]
    log_centers = np.array([math.log(c) for c in centers])
    x = np.arange(len(log_centers))
    x_mean = x.mean()
    y_mean = log_centers.mean()
    ss_xx = np.sum((x - x_mean) ** 2)
    ss_yy = np.sum((log_centers - y_mean) ** 2)
    if ss_xx == 0 or ss_yy == 0:
        return None
    ss_xy = np.sum((x - x_mean) * (log_centers - y_mean))
    return (ss_xy ** 2) / (ss_xx * ss_yy)


def compute_escape_velocity(clusters):
    """
    Escape velocity: geometric mean of inter-cluster jump ratios.

    Jump ratio = first exponent of cluster_{i+1} / last exponent of cluster_i.
    """
    if len(clusters) < 2:
        return None
    jumps = [clusters[i][0] / clusters[i - 1][-1] for i in range(1, len(clusters))]
    if not jumps:
        return None
    return math.exp(sum(math.log(j) for j in jumps) / len(jumps))


def compute_intercept(clusters):
    """Intercept C in center(n) = C * base^n."""
    if len(clusters) < 3:
        return None
    centers = [math.exp(sum(math.log(x) for x in c) / len(c)) for c in clusters]
    log_centers = np.array([math.log(c) for c in centers])
    x = np.arange(len(log_centers))
    x_mean = x.mean()
    y_mean = log_centers.mean()
    ss_xx = np.sum((x - x_mean) ** 2)
    if ss_xx == 0:
        return None
    slope = np.sum((x - x_mean) * (log_centers - y_mean)) / ss_xx
    intercept = y_mean - slope * x_mean
    return math.exp(intercept)


# ─── Singleton pairing ───────────────────────────────────────────────────────

def consecutive_pairing(singletons):
    """
    Pair consecutive singletons in log10 space (zero degrees of freedom).

    Pairs: (0,1), (2,3), (4,5), ... Last singleton unpaired if count is odd.

    Returns (gaps, decay_rates, mean_decay) or None if insufficient data.
    """
    log10_s = [math.log10(s) for s in singletons]

    gaps = []
    for i in range(0, len(log10_s) - 1, 2):
        gap = log10_s[i + 1] - log10_s[i]
        if gap <= 0:
            return None
        gaps.append(gap)

    if len(gaps) < 2:
        return None

    decay_rates = [gaps[i] / gaps[i - 1] for i in range(1, len(gaps))]
    if any(d <= 0 for d in decay_rates):
        return None

    mean_decay = sum(decay_rates) / len(decay_rates)
    return gaps, decay_rates, mean_decay


def unity_equation(decay, escape, orbital):
    """Compute unity = decay^2 * escape * orbital."""
    return decay ** 2 * escape * orbital


# ─── Monte Carlo ──────────────────────────────────────────────────────────────

def generate_wagstaff_sequence(rng, n_target=52, p_min=2, p_max=200_000_000):
    """
    Generate synthetic Mersenne prime exponent sequence using Wagstaff/Poisson model.

    Exponents are modeled as a Poisson process uniform in log-log space.

    Args:
        rng: numpy.random.Generator instance (for reproducibility)
        n_target: expected number of primes (Poisson mean)
        p_min: minimum exponent
        p_max: maximum exponent
    """
    ll_min = math.log(math.log(max(p_min, 2.1)))
    ll_max = math.log(math.log(p_max))
    n_actual = rng.poisson(n_target)
    if n_actual < 6:
        n_actual = 6
    ll_samples = rng.uniform(ll_min, ll_max, n_actual)
    ll_samples.sort()
    exponents = sorted(set(int(math.exp(math.exp(ll))) for ll in ll_samples))
    return exponents


def run_unity_mc(n_sims, seed, tau=TAU_DEFAULT, n_target=52, p_max=200_000_000):
    """
    Monte Carlo: generate random Wagstaff sequences, apply consecutive pairing,
    compute |unity - 1| for each.

    Returns array of |unity - 1| values for valid simulations.
    """
    rng = np.random.default_rng(seed)
    unity_dists = []

    for _ in range(n_sims):
        exps = generate_wagstaff_sequence(rng, n_target=n_target, p_max=p_max)
        cls = classify_clusters(exps, tau=tau)

        if len(cls) < 5:
            continue

        singles = get_singletons(cls)
        if len(singles) < 4:
            continue

        orb = compute_orbital_base(cls)
        esc = compute_escape_velocity(cls)
        if orb is None or esc is None or orb <= 0 or esc <= 0:
            continue

        result = consecutive_pairing(singles)
        if result is None:
            continue

        _, _, mean_decay = result
        u = unity_equation(mean_decay, esc, orb)
        if 0 < u < 100:
            unity_dists.append(abs(u - 1.0))

    return np.array(unity_dists)


def multi_seed_mc(n_sims_per_seed, seeds, tau=TAU_DEFAULT):
    """
    Run Monte Carlo across multiple seeds and report variance.

    Returns dict with per-seed results and aggregate statistics.
    """
    results = {}
    all_dists = []

    for seed in seeds:
        dists = run_unity_mc(n_sims_per_seed, seed, tau=tau)
        all_dists.append(dists)
        results[seed] = {
            "n_valid": len(dists),
            "mean_unity_dist": float(dists.mean()) if len(dists) > 0 else None,
            "median_unity_dist": float(np.median(dists)) if len(dists) > 0 else None,
        }

    combined = np.concatenate(all_dists)
    results["combined"] = {
        "n_valid": len(combined),
        "total_sims": n_sims_per_seed * len(seeds),
    }

    return results, combined
