# Hierarchical Structure in the Mersenne Prime Distribution

Reproducibility scripts for "Hierarchical Structure in the Mersenne Prime Distribution: A Multi-Scale Analysis" (Toporowski, 2026).

## Quick Start

```bash
pip install numpy scipy
cd paper
bash run_all.sh
```

Expected runtime: ~3-5 minutes.

## Repository Structure

```
paper/                          Reproducibility scripts
  staircase_utils.py            Shared utilities (clustering, constants, MC)
  claim1_staircase_fit.py       R²=0.998, ΔAIC=55.7, RMSE 43% reduction
  claim2_escape_invariance.py   Escape velocity scale independence
  claim3_lighthouse.py          Singleton detection (5/5, p=0.000006)
  claim4_unity_equation.py      Unity equation MC (p<0.01, multi-seed)
  claim5_threshold_tau.py       Threshold τ=3/2 uniqueness (2/73)
  claim6_bootstrap_ci.py        Bootstrap confidence intervals
  run_all.sh                    Run all claims
  README.md                     Detailed claim-to-script mapping
data/
  mersenne_exponents.json       The 52 known Mersenne prime exponents
```

## Requirements

- Python 3.11+
- numpy >= 1.26
- scipy >= 1.11

## Paper

The paper is available on Zenodo: [DOI pending]

## License

MIT
