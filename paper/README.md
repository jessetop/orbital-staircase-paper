# Reproducing Paper Claims

Scripts to verify every quantitative claim in two companion papers:

**Paper A:** "Clustering Structure in Mersenne Prime Exponents: An Empirical Investigation"
**Paper B:** "Local Detection Signatures of Mersenne Primes via Lucas-Lehmer Residue Statistics"

## Quick Start

```bash
cd paper
bash run_all.sh
```

Expected runtime: ~5-10 minutes (most time in Claim 4 Monte Carlo and Global MC).

## Requirements

- Python 3.11+
- numpy >= 1.26
- scipy >= 1.11

## Claim-to-Script Mapping

### Paper A — Clustering and Staircase

| Claim | Script | Key Numbers |
|-------|--------|-------------|
| 1. Orbital staircase fit | `claim1_staircase_fit.py` | R²=0.998, ΔAIC=55.7, RMSE 43% reduction |
| 2. Escape velocity invariance | `claim2_escape_invariance.py` | Pearson r=-0.007, p=0.977 |
| 4. Unity equation | `claim4_unity_equation.py` | unity=0.956, p≈0.005 (multi-seed MC) |
| 5. Threshold uniqueness | `claim5_threshold_tau.py` | 2/73 thresholds produce unity≈1 |
| 6. Bootstrap CIs | `claim6_bootstrap_ci.py` | Orbital 2.680 [2.02, 3.10], etc. |
| **Joint significance** | **`global_pipeline_mc.py`** | **p<0.0001 (0/10,000 joint passes)** |
| Functional-form LEE | `functional_form_mc.py` | 342 monomial products tested, look-elsewhere corrected |
| Threshold independence | `threshold_independence.py` | R² and LOOCV peak near τ=3/2 |

### Paper B — Detection Signatures

| Claim | Script | Key Numbers |
|-------|--------|-------------|
| 3. Lighthouse detection | `claim3_lighthouse.py` | 5/5 singletons rank #1, p=0.000006 |

## Data

All scripts read from `data/mersenne_exponents.json` — the 52 known Mersenne prime
exponents as of M52 (136,279,841).

## Shared Utilities

`staircase_utils.py` provides canonical implementations of clustering, constant
computation, and Monte Carlo methods used by all claim scripts. The threshold
convention is τ = 1.501 (just above 3/2) to correctly handle the exact M(2)→M(3)
ratio of 1.500.

## Key New Results

### Global Pipeline Monte Carlo (Paper A, Section 8)

The most important script for addressing the "garden of forking paths" critique.
For each of 10,000 synthetic Wagstaff-Poisson sequences, the script applies the
full analysis pipeline with post-hoc threshold optimization — exactly the procedure
a referee would worry about. Zero synthetic sequences jointly beat the real data on
all four metrics (R², ΔAIC, |unity-1|, |Spearman r|), yielding p < 0.0001.

### Functional-Form Look-Elsewhere Effect (Paper A, Section 8)

Addresses the referee critique: "how many functional forms were tried before landing
on d^a × v^b × b^c ≈ 1?" This script tests all 342 monomial products with integer
exponents in [-3, 3] against 1,000 synthetic Wagstaff sequences to compute a
look-elsewhere-corrected p-value. If the unity equation is a statistical fluke,
many forms should achieve similar proximity to unity on random data.

### Threshold Independence (Paper A, Section 3.3)

Tests whether τ ≈ 3/2 is independently selected by staircase fit quality (R²) and
cross-validation prediction error, without reference to the unity equation. Both
criteria peak in the 1.4–1.5 neighborhood, with τ = 1.5 ranking in the top 3–5
for each.

## Monte Carlo Seed Variance

Claim 4 runs 500K total simulations across 5 independent seeds to demonstrate
p-value stability. Individual 10K runs may vary ±50% around the mean due to
sampling noise — the multi-seed approach eliminates this concern.

## Note on Claim 3

The lighthouse scores (ll_transient) require the full pipeline with gmpy2 and
tier-specific calibration data. `claim3_lighthouse.py` verifies the combinatorial
p-value from precomputed results. The raw scores can be reproduced using the
full research pipeline (available from the author).
