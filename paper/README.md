# Reproducing Paper Claims

Scripts to verify every quantitative claim in "The Orbital Staircase: A Multi-Scale
Clustering Framework for Mersenne Prime Exponents."

## Quick Start

```bash
cd paper
bash run_all.sh
```

Expected runtime: ~3-5 minutes (most time in Claim 4 Monte Carlo).

## Requirements

- Python 3.11+
- numpy >= 1.26
- scipy >= 1.11

## Claim-to-Script Mapping

| Claim | Script | Key Numbers |
|-------|--------|-------------|
| 1. Orbital staircase fit | `claim1_staircase_fit.py` | R²=0.998, ΔAIC=55.7, RMSE 43% reduction |
| 2. Escape velocity invariance | `claim2_escape_invariance.py` | Pearson r=-0.007, p=0.977 |
| 3. Lighthouse detection | `claim3_lighthouse.py` | 5/5 singletons rank #1, p=0.000006 |
| 4. Unity equation | `claim4_unity_equation.py` | unity=0.956, p≈0.005 (multi-seed MC) |
| 5. Threshold uniqueness | `claim5_threshold_tau.py` | 2/73 thresholds produce unity≈1 |
| 6. Bootstrap CIs | `claim6_bootstrap_ci.py` | Orbital 2.680 [2.02, 3.10], etc. |

## Data

All scripts read from `data/mersenne_exponents.json` — the 52 known Mersenne prime
exponents as of M52 (136,279,841).

## Shared Utilities

`staircase_utils.py` provides canonical implementations of clustering, constant
computation, and Monte Carlo methods used by all claim scripts. The threshold
convention is τ = 1.501 (just above 3/2) to correctly handle the exact M(2)→M(3)
ratio of 1.500.

## Monte Carlo Seed Variance

Claim 4 runs 500K total simulations across 5 independent seeds to demonstrate
p-value stability. Individual 10K runs may vary ±50% around the mean due to
sampling noise — the multi-seed approach eliminates this concern.

## Note on Claim 3

The lighthouse scores (ll_transient) require the full pipeline with gmpy2 and
tier-specific calibration data. `claim3_lighthouse.py` verifies the combinatorial
p-value from precomputed results. The raw scores can be reproduced using the
pipeline scripts in `scripts/`.
