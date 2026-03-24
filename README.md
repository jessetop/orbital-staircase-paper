# Hierarchical Structure in the Mersenne Prime Distribution

Reproducibility scripts for two companion papers:

**Paper A:** "Clustering Structure in Mersenne Prime Exponents: An Empirical Investigation" (Toporowski, 2026)

**Paper B:** "Local Detection Signatures of Mersenne Primes via Lucas-Lehmer Residue Statistics" (Toporowski, 2026)

## Quick Start

```bash
pip install numpy scipy
cd paper
bash run_all.sh
```

Expected runtime: ~5-10 minutes (includes global Monte Carlo).

## Repository Structure

```
paper/                              Reproducibility scripts
  staircase_utils.py                Shared utilities (clustering, constants, MC)

  # Paper A — Clustering and Staircase
  claim1_staircase_fit.py           R²=0.998, ΔAIC=55.7, RMSE 43% reduction
  claim2_escape_invariance.py       Escape velocity scale independence
  claim4_unity_equation.py          Unity equation MC (p<0.01, multi-seed)
  claim5_threshold_tau.py           Threshold τ=3/2 uniqueness (2/73)
  claim6_bootstrap_ci.py            Bootstrap confidence intervals
  global_pipeline_mc.py             Joint significance test (p<0.0001)
  threshold_independence.py         Independent threshold selection (R², LOOCV, MDL)

  # Paper B — Detection Signatures
  claim3_lighthouse.py              Singleton detection (5/5, p=0.000006)

  # Shared
  figures.py                        Publication figures (4 PNGs)
  run_all.sh                        Run all claim scripts
  README.md                         Detailed claim-to-script mapping

data/
  mersenne_exponents.json           The 52 known Mersenne prime exponents
```

## Key Results

| Claim | Script | Result |
|-------|--------|--------|
| Staircase fit | claim1 | R²=0.998, ΔAIC=55.7 vs Wagstaff |
| Escape velocity | claim2 | Scale-invariant (r=-0.007, p=0.977) |
| Lighthouse detection | claim3 | 5/5 singletons rank #1 (p=0.000006) |
| Unity equation | claim4 | p~0.005 (multi-seed MC) |
| Threshold uniqueness | claim5 | Only τ≈3/2 works (next best 64x worse) |
| Bootstrap CIs | claim6 | Orbital [2.023, 3.096], Escape [1.561, 2.288] |
| **Joint significance** | **global_pipeline_mc** | **p < 0.0001 (0/10,000 joint passes)** |
| Threshold independence | threshold_independence | R² and LOOCV peak near τ=3/2 |

## Requirements

- Python 3.11+
- numpy >= 1.26
- scipy >= 1.11

## Papers

Available on Zenodo: [DOI pending]

## License

MIT
