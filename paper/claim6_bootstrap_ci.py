"""
Claim 6 — Bootstrap confidence intervals for all staircase constants.

Resamples the 19 observed clusters WITH REPLACEMENT (10K iterations),
recomputes orbital base, escape velocity, decay rate, unity product,
and orbital R² each time.  Reports 95% percentile CIs, correlation
matrix, propagated uncertainty on unity, Wagstaff overlap check, and
M53 prediction interval.

Requires: numpy, scipy
"""

import sys
import os
import math

import numpy as np
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from staircase_utils import (
    load_mersenne_exponents, classify_clusters, get_singletons,
    compute_orbital_base, compute_escape_velocity, consecutive_pairing,
    unity_equation, TAU_DEFAULT
)

# ─── Configuration ────────────────────────────────────────────────────────────

N_BOOT = 10_000
SEED = 42
WAGSTAFF_CONSTANT = math.exp(0.5772156649) / math.log(2)  # ~2.569
CI_LO, CI_HI = 2.5, 97.5  # percentile bounds for 95% CI

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _compute_orbital_r_squared(clusters):
    """R² of log-linear fit of cluster geometric centers."""
    if len(clusters) < 3:
        return np.nan
    centers = [math.exp(sum(math.log(x) for x in c) / len(c)) for c in clusters]
    log_centers = np.array([math.log(c) for c in centers])
    x = np.arange(len(log_centers))
    x_mean = x.mean()
    y_mean = log_centers.mean()
    ss_xx = np.sum((x - x_mean) ** 2)
    ss_yy = np.sum((log_centers - y_mean) ** 2)
    if ss_xx == 0 or ss_yy == 0:
        return np.nan
    ss_xy = np.sum((x - x_mean) * (log_centers - y_mean))
    return float((ss_xy ** 2) / (ss_xx * ss_yy))


def _compute_intercept(clusters):
    """Intercept C in center(n) = C * base^n."""
    if len(clusters) < 3:
        return np.nan
    centers = [math.exp(sum(math.log(x) for x in c) / len(c)) for c in clusters]
    log_centers = np.array([math.log(c) for c in centers])
    x = np.arange(len(log_centers))
    x_mean = x.mean()
    y_mean = log_centers.mean()
    ss_xx = np.sum((x - x_mean) ** 2)
    if ss_xx == 0:
        return np.nan
    slope = np.sum((x - x_mean) * (log_centers - y_mean)) / ss_xx
    intercept = y_mean - slope * x_mean
    return math.exp(intercept)


def _predict_m53(clusters):
    """Predict next cluster center from orbital fit, return exponent."""
    base = compute_orbital_base(clusters)
    C = _compute_intercept(clusters)
    if base is None or C is None:
        return np.nan
    n_next = len(clusters)  # 0-indexed, so len = next index
    return C * base ** n_next


def _constants_from_clusters(clusters):
    """Compute all staircase constants from a list of clusters.

    Returns (orbital_base, escape_vel, decay_rate, unity, r_squared)
    or a tuple of NaNs if computation fails.
    """
    nan5 = (np.nan,) * 5

    orb = compute_orbital_base(clusters)
    if orb is None or orb <= 0:
        return nan5

    esc = compute_escape_velocity(clusters)
    if esc is None or esc <= 0:
        return nan5

    singletons = get_singletons(clusters)
    result = consecutive_pairing(singletons)
    if result is None:
        return nan5
    _, _, decay = result

    unity = unity_equation(decay, esc, orb)
    r2 = _compute_orbital_r_squared(clusters)

    return orb, esc, decay, unity, r2


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(SEED)

    # 1. Load data, classify clusters
    exponents = load_mersenne_exponents()
    clusters = classify_clusters(exponents, tau=TAU_DEFAULT)
    n_clusters = len(clusters)

    print(f"Loaded {len(exponents)} Mersenne prime exponents")
    print(f"Classified into {n_clusters} clusters (tau={TAU_DEFAULT})")
    print()

    # 2. Observed constants
    obs_orb, obs_esc, obs_decay, obs_unity, obs_r2 = _constants_from_clusters(clusters)
    obs_m53 = _predict_m53(clusters)

    print("=" * 64)
    print("OBSERVED CONSTANTS")
    print("=" * 64)
    print(f"  Orbital base     : {obs_orb:.3f}")
    print(f"  Escape velocity  : {obs_esc:.3f}")
    print(f"  Decay rate       : {obs_decay:.3f}")
    print(f"  Unity product    : {obs_unity:.3f}")
    print(f"  Orbital R²       : {obs_r2:.6f}")
    print(f"  M53 prediction   : {obs_m53:,.0f}")
    print()

    # 3. Bootstrap: resample clusters WITH REPLACEMENT
    boot_orb = np.full(N_BOOT, np.nan)
    boot_esc = np.full(N_BOOT, np.nan)
    boot_decay = np.full(N_BOOT, np.nan)
    boot_unity = np.full(N_BOOT, np.nan)
    boot_r2 = np.full(N_BOOT, np.nan)
    boot_m53 = np.full(N_BOOT, np.nan)

    cluster_arr = np.array(clusters, dtype=object)
    n_valid = 0

    for i in range(N_BOOT):
        idx = rng.integers(0, n_clusters, size=n_clusters)
        resampled = [list(cluster_arr[j]) for j in idx]
        # Sort clusters by first element to maintain ordering
        resampled.sort(key=lambda c: c[0])

        orb, esc, decay, unity, r2 = _constants_from_clusters(resampled)
        boot_orb[i] = orb
        boot_esc[i] = esc
        boot_decay[i] = decay
        boot_unity[i] = unity
        boot_r2[i] = r2
        boot_m53[i] = _predict_m53(resampled)

        if not np.isnan(orb):
            n_valid += 1

    valid_pct = 100 * n_valid / N_BOOT

    # 4. Report 95% CIs (percentile method)
    def ci(arr):
        valid = arr[~np.isnan(arr)]
        if len(valid) < 100:
            return (np.nan, np.nan)
        return (float(np.percentile(valid, CI_LO)),
                float(np.percentile(valid, CI_HI)))

    ci_orb = ci(boot_orb)
    ci_esc = ci(boot_esc)
    ci_decay = ci(boot_decay)
    ci_unity = ci(boot_unity)
    ci_r2 = ci(boot_r2)
    ci_m53 = ci(boot_m53)

    print("=" * 64)
    print(f"BOOTSTRAP 95% CIs  ({N_BOOT:,} iterations, {n_valid:,} valid = {valid_pct:.1f}%)")
    print("=" * 64)
    fmt = "  {:<18s}: {:.3f}   CI [{:.3f}, {:.3f}]"
    print(fmt.format("Orbital base", obs_orb, *ci_orb))
    print(fmt.format("Escape velocity", obs_esc, *ci_esc))
    print(fmt.format("Decay rate", obs_decay, *ci_decay))
    print(fmt.format("Unity product", obs_unity, *ci_unity))
    print(f"  {'Orbital R²':<18s}: {obs_r2:.6f}   CI [{ci_r2[0]:.6f}, {ci_r2[1]:.6f}]")
    print(f"  {'M53 prediction':<18s}: {obs_m53:>12,.0f}   CI [{ci_m53[0]:>12,.0f}, {ci_m53[1]:>12,.0f}]")
    print()

    # 5. Correlation matrix between orbital / escape / decay
    mask = ~(np.isnan(boot_orb) | np.isnan(boot_esc) | np.isnan(boot_decay))
    v_orb = boot_orb[mask]
    v_esc = boot_esc[mask]
    v_decay = boot_decay[mask]

    corr_matrix = np.corrcoef(np.vstack([v_orb, v_esc, v_decay]))

    print("=" * 64)
    print("CORRELATION MATRIX  (orbital, escape, decay)")
    print("=" * 64)
    labels = ["Orbital", "Escape ", "Decay  "]
    print(f"  {'':>10s}  {'Orbital':>8s}  {'Escape':>8s}  {'Decay':>8s}")
    for i, lab in enumerate(labels):
        row = "  ".join(f"{corr_matrix[i, j]:8.4f}" for j in range(3))
        print(f"  {lab:>10s}  {row}")
    print()

    # 6. Propagated uncertainty on unity (first-order error propagation)
    #    unity = decay^2 * escape * orbital
    #    d(unity)/d(decay)   = 2*decay * escape * orbital
    #    d(unity)/d(escape)  = decay^2 * orbital
    #    d(unity)/d(orbital) = decay^2 * escape
    sigma_d = float(np.std(v_decay, ddof=1))
    sigma_e = float(np.std(v_esc, ddof=1))
    sigma_o = float(np.std(v_orb, ddof=1))

    du_dd = 2 * obs_decay * obs_esc * obs_orb
    du_de = obs_decay ** 2 * obs_orb
    du_do = obs_decay ** 2 * obs_esc

    # Include covariance terms for full propagation
    cov_matrix = np.cov(np.vstack([v_decay, v_esc, v_orb]))
    grad = np.array([du_dd, du_de, du_do])
    sigma_unity_sq = grad @ cov_matrix @ grad
    sigma_unity = math.sqrt(max(sigma_unity_sq, 0))

    print("=" * 64)
    print("PROPAGATED UNCERTAINTY ON UNITY (first-order)")
    print("=" * 64)
    print(f"  sigma(decay)     : {sigma_d:.4f}")
    print(f"  sigma(escape)    : {sigma_e:.4f}")
    print(f"  sigma(orbital)   : {sigma_o:.4f}")
    print(f"  sigma(unity)     : {sigma_unity:.4f}")
    print(f"  Unity observed   : {obs_unity:.4f} +/- {sigma_unity:.4f}")
    print(f"  Unity 1.0 within : {abs(1.0 - obs_unity) / sigma_unity:.2f} sigma")
    print()

    # 7. Wagstaff constant overlap check
    print("=" * 64)
    print("WAGSTAFF CONSTANT CHECK")
    print("=" * 64)
    print(f"  Wagstaff constant (e^gamma / ln2) : {WAGSTAFF_CONSTANT:.3f}")
    print(f"  Orbital base observed              : {obs_orb:.3f}")
    print(f"  Orbital base 95% CI                : [{ci_orb[0]:.3f}, {ci_orb[1]:.3f}]")
    inside = ci_orb[0] <= WAGSTAFF_CONSTANT <= ci_orb[1]
    print(f"  Wagstaff inside orbital CI?        : {'YES' if inside else 'NO'}")
    if inside:
        # Where does Wagstaff sit in the bootstrap distribution?
        pct = float(np.mean(v_orb <= WAGSTAFF_CONSTANT) * 100)
        print(f"  Wagstaff percentile in bootstrap   : {pct:.1f}%")
    print()

    # 8. M53 prediction interval
    valid_m53 = boot_m53[~np.isnan(boot_m53)]
    print("=" * 64)
    print("M53 PREDICTION INTERVAL")
    print("=" * 64)
    print(f"  Point estimate   : {obs_m53:>14,.0f}")
    print(f"  Bootstrap median : {np.median(valid_m53):>14,.0f}")
    print(f"  95% PI           : [{np.percentile(valid_m53, CI_LO):>14,.0f}, "
          f"{np.percentile(valid_m53, CI_HI):>14,.0f}]")
    print(f"  90% PI           : [{np.percentile(valid_m53, 5):>14,.0f}, "
          f"{np.percentile(valid_m53, 95):>14,.0f}]")
    print(f"  Valid samples    : {len(valid_m53):,} / {N_BOOT:,}")
    print()

    print("Done.")


if __name__ == "__main__":
    main()
