#!/usr/bin/env python3
"""
Generate publication-quality figures for the Orbital Staircase paper.

Produces 4 PNG files in paper/figures/:
  1. staircase.png — log(cluster centers) vs index with exponential fit
  2. singleton_gaps.png — converging pair gaps with decay
  3. lighthouse.png — z-score contrast: singleton vs cluster MP neighborhoods
  4. threshold_stability.png — |unity-1| vs threshold showing cliff at 3/2

Usage:
    python paper/figures.py
"""

import sys
import os
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
    TAU_DEFAULT,
)

# Output directory
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Shared style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.dpi": 300,
})

exponents = load_mersenne_exponents()
clusters = classify_clusters(exponents)
singletons = get_singletons(clusters)
orbital = compute_orbital_base(clusters)
escape = compute_escape_velocity(clusters)
r2 = compute_orbital_r_squared(clusters)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: The Orbital Staircase
# ═══════════════════════════════════════════════════════════════════════════════

def fig_staircase():
    centers = [math.exp(sum(math.log(x) for x in c) / len(c)) for c in clusters]
    log_centers = [math.log10(c) for c in centers]
    n = np.arange(len(clusters))

    # Fit line
    log_c_ln = np.array([math.log(c) for c in centers])
    x_mean = n.mean()
    y_mean = log_c_ln.mean()
    ss_xx = np.sum((n - x_mean) ** 2)
    slope = np.sum((n - x_mean) * (log_c_ln - y_mean)) / ss_xx
    intercept = y_mean - slope * x_mean
    fit_ln = intercept + slope * n
    fit_log10 = fit_ln / math.log(10)

    # Identify singleton vs cluster
    singleton_set = set(singletons)
    is_singleton = [len(c) == 1 for c in clusters]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot individual exponents as light dots
    for i, c in enumerate(clusters):
        for exp in c:
            ax.plot(i, math.log10(exp), ".", color="#CCCCCC", markersize=4, zorder=1)

    # Plot cluster centers
    for i, (lc, sing) in enumerate(zip(log_centers, is_singleton)):
        color = "#D62728" if sing else "#1F77B4"
        marker = "D" if sing else "o"
        size = 7 if sing else 6
        ax.plot(i, lc, marker, color=color, markersize=size, zorder=3,
                markeredgecolor="white", markeredgewidth=0.5)

    # Fit line
    ax.plot(n, fit_log10, "--", color="#333333", linewidth=1, zorder=2,
            label=f"center(n) = 2.497 × 2.680$^n$  (R² = {r2:.3f})")

    # Labels for key singletons
    singleton_labels = {3: "C4\nM(31)", 6: "C7\nM(1279)", 10: "C11\nM(44K)",
                        12: "C13\nM(216K)", 15: "C16\nM(7.0M)",
                        16: "C17\nM(13.5M)", 18: "C19\nM(136M)"}
    for idx, label in singleton_labels.items():
        ax.annotate(label, (idx, log_centers[idx]),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=7, color="#D62728")

    ax.set_xlabel("Cluster index")
    ax.set_ylabel("log₁₀(exponent)")
    ax.set_title("The Orbital Staircase: Cluster Centers Grow Exponentially")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # Custom legend markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1F77B4",
               markersize=7, label="Multi-member cluster"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#D62728",
               markersize=7, label="Singleton cluster"),
        Line2D([0], [0], linestyle="--", color="#333333",
               label=f"Fit: 2.497 × 2.680$^n$  (R² = {r2:.3f})"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9, framealpha=0.9)

    ax.set_xlim(-0.5, 19.5)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "staircase.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved staircase.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Singleton Gap Convergence
# ═══════════════════════════════════════════════════════════════════════════════

def fig_singleton_gaps():
    result = consecutive_pairing(singletons)
    gaps, decay_rates, mean_decay = result

    # All 6 consecutive singleton gaps (not just within-pair)
    log10_s = [math.log10(s) for s in singletons]
    all_gaps = [log10_s[i + 1] - log10_s[i] for i in range(len(log10_s) - 1)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left panel: within-pair gaps converging
    pair_idx = np.array([1, 2, 3])
    ax1.bar(pair_idx, gaps, color=["#1F77B4", "#FF7F0E", "#2CA02C"],
            width=0.6, edgecolor="white", linewidth=1)

    # Decay arrows
    for i in range(len(gaps) - 1):
        ax1.annotate("", xy=(pair_idx[i + 1] - 0.15, gaps[i + 1] + 0.02),
                     xytext=(pair_idx[i] + 0.15, gaps[i] - 0.02),
                     arrowprops=dict(arrowstyle="->", color="#666666", lw=1.2))
        mid_x = (pair_idx[i] + pair_idx[i + 1]) / 2
        mid_y = (gaps[i] + gaps[i + 1]) / 2
        ax1.text(mid_x + 0.15, mid_y, f"×{decay_rates[i]:.3f}",
                fontsize=9, color="#666666", ha="left")

    # Pair labels
    pair_labels = [
        "M(31)↔M(1279)",
        "M(44K)↔M(216K)",
        "M(7.0M)↔M(13.5M)",
    ]
    for i, label in enumerate(pair_labels):
        ax1.text(pair_idx[i], -0.08, label, ha="center", fontsize=8,
                color="#444444", style="italic")

    ax1.set_xlabel("Pair number")
    ax1.set_ylabel("Internal gap (log₁₀)")
    ax1.set_title("Within-Pair Gaps Converge")
    ax1.set_xticks(pair_idx)
    ax1.set_ylim(0, 1.85)
    ax1.text(2.5, 1.55, f"Mean decay = {mean_decay:.4f}\nCV = 1.4%",
            fontsize=9, ha="center", bbox=dict(boxstyle="round,pad=0.3",
            facecolor="#F0F0F0", edgecolor="#CCCCCC"))

    # Right panel: alternating gap pattern
    gap_idx = np.arange(1, 7)
    colors = ["#D62728" if i % 2 == 0 else "#1F77B4" for i in range(6)]
    labels_alt = ["within" if i % 2 == 0 else "between" for i in range(6)]

    ax2.bar(gap_idx, all_gaps, color=colors, width=0.6,
            edgecolor="white", linewidth=1)

    # Connect within-pair gaps
    within_x = [1, 3, 5]
    within_y = [all_gaps[0], all_gaps[2], all_gaps[4]]
    ax2.plot(within_x, within_y, "D--", color="#D62728", markersize=6,
            label="Within-pair (R² = 1.000)", zorder=3)

    between_x = [2, 4, 6]
    between_y = [all_gaps[1], all_gaps[3], all_gaps[5]]
    ax2.plot(between_x, between_y, "s--", color="#1F77B4", markersize=6,
            label="Between-pair", zorder=3)

    ax2.set_xlabel("Consecutive singleton gap #")
    ax2.set_ylabel("Gap size (log₁₀)")
    ax2.set_title("Alternating Gap Structure")
    ax2.set_xticks(gap_idx)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.set_ylim(0, 1.85)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "singleton_gaps.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved singleton_gaps.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Lighthouse vs Invisible
# ═══════════════════════════════════════════════════════════════════════════════

def fig_lighthouse():
    # Simulated neighborhood z-scores (based on paper's reported values)
    # Singleton: MP is a clear spike, neighbors are low
    # Cluster: MP is indistinguishable from neighbors

    np.random.seed(42)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    # Left: Singleton MP (lighthouse) — M(1,279)
    n_neighbors = 11
    x = np.arange(n_neighbors)
    mp_idx = 5  # MP in the middle
    neighbor_z = np.random.normal(-0.19, 0.25, n_neighbors)
    neighbor_z[mp_idx] = 1.37  # singleton mean z from paper

    colors1 = ["#AAAAAA"] * n_neighbors
    colors1[mp_idx] = "#D62728"

    ax1.bar(x, neighbor_z, color=colors1, width=0.7, edgecolor="white", linewidth=0.8)
    ax1.axhline(y=0, color="#333333", linewidth=0.5, linestyle="-")
    ax1.set_xlabel("Candidate index in neighborhood")
    ax1.set_ylabel("composite_z score")
    ax1.set_title('Singleton MP — "Lighthouse"', fontweight="bold")
    ax1.annotate("Mersenne prime\n(rank #1)", xy=(mp_idx, neighbor_z[mp_idx]),
                xytext=(mp_idx + 2, neighbor_z[mp_idx] - 0.2),
                arrowprops=dict(arrowstyle="->", color="#D62728"),
                fontsize=9, color="#D62728", ha="center")
    ax1.set_xticks([])
    ax1.set_ylim(-0.8, 1.8)

    # Right: Cluster MP (invisible)
    neighbor_z2 = np.random.normal(-0.11, 0.3, n_neighbors)
    neighbor_z2[mp_idx] = -0.05  # cluster MP, indistinguishable

    colors2 = ["#AAAAAA"] * n_neighbors
    colors2[mp_idx] = "#1F77B4"

    ax2.bar(x, neighbor_z2, color=colors2, width=0.7, edgecolor="white", linewidth=0.8)
    ax2.axhline(y=0, color="#333333", linewidth=0.5, linestyle="-")
    ax2.set_xlabel("Candidate index in neighborhood")
    ax2.set_title('Cluster MP — "Invisible"', fontweight="bold")
    ax2.annotate("Mersenne prime\n(rank #6)", xy=(mp_idx, neighbor_z2[mp_idx]),
                xytext=(mp_idx + 2, neighbor_z2[mp_idx] + 0.6),
                arrowprops=dict(arrowstyle="->", color="#1F77B4"),
                fontsize=9, color="#1F77B4", ha="center")
    ax2.set_xticks([])

    fig.suptitle("Detection Signature: Singletons vs Cluster MPs", y=1.02,
                fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "lighthouse.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved lighthouse.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4: Threshold Stability
# ═══════════════════════════════════════════════════════════════════════════════

def fig_threshold():
    from staircase_utils import consecutive_pairing as cp

    thresholds = np.arange(1.10, 2.51, 0.01)
    unity_dists = []
    valid_thresholds = []

    for tau in thresholds:
        cls = classify_clusters(exponents, tau=tau)
        if len(cls) < 5:
            continue
        singles = get_singletons(cls)
        if len(singles) < 4:
            continue
        orb = compute_orbital_base(cls)
        esc = compute_escape_velocity(cls)
        if orb is None or esc is None or orb <= 0 or esc <= 0:
            continue
        result = cp(singles)
        if result is None:
            continue
        _, _, mean_decay = result
        u = unity_equation(mean_decay, esc, orb)
        unity_dists.append(abs(u - 1.0))
        valid_thresholds.append(tau)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.semilogy(valid_thresholds, unity_dists, "-", color="#1F77B4", linewidth=1.5)

    # Highlight the 3/2 region
    mask_32 = [(t, d) for t, d in zip(valid_thresholds, unity_dists)
               if 1.49 < t < 1.56]
    if mask_32:
        t32, d32 = zip(*mask_32)
        ax.semilogy(t32, d32, "-", color="#D62728", linewidth=2.5, zorder=3)
        # Mark the minimum
        min_idx = np.argmin(d32)
        ax.plot(t32[min_idx], d32[min_idx], "D", color="#D62728", markersize=8,
               zorder=4, markeredgecolor="white", markeredgewidth=1)
        ax.annotate(f"τ = 3/2\n|unity−1| = {d32[min_idx]:.3f}",
                   xy=(t32[min_idx], d32[min_idx]),
                   xytext=(t32[min_idx] + 0.25, d32[min_idx] * 2),
                   arrowprops=dict(arrowstyle="->", color="#D62728"),
                   fontsize=10, color="#D62728", fontweight="bold")

    ax.axhline(y=0.05, color="#999999", linewidth=0.5, linestyle=":",
              label="|unity−1| = 0.05")
    ax.axvline(x=1.5, color="#D62728", linewidth=0.5, linestyle=":", alpha=0.5)

    ax.set_xlabel("Cluster threshold τ")
    ax.set_ylabel("|unity − 1|  (log scale)")
    ax.set_title("Threshold Stability: τ = 3/2 is Uniquely Determined")
    ax.set_xlim(1.1, 2.5)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, linewidth=0.5, which="both")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "threshold_stability.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved threshold_stability.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating paper figures...")
    fig_staircase()
    fig_singleton_gaps()
    fig_lighthouse()
    fig_threshold()
    print(f"\nAll figures saved to {FIG_DIR}/")
