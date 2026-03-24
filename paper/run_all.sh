#!/bin/bash
# Run all paper claim verification scripts.
# Usage: cd paper && bash run_all.sh
#
# Expected runtime: ~5-10 minutes (dominated by claim4 Monte Carlo and global MC)
# Requirements: Python 3.11+, numpy, scipy

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

PYTHON="${PYTHON:-python3}"
PASS=0
FAIL=0

run_claim() {
    local name="$1"
    local script="$2"
    echo ""
    echo "================================================================"
    echo "  RUNNING: $name"
    echo "================================================================"
    if $PYTHON "$script"; then
        echo ""
        echo "  ✓ $name completed successfully"
        PASS=$((PASS + 1))
    else
        echo ""
        echo "  ✗ $name FAILED (exit code $?)"
        FAIL=$((FAIL + 1))
    fi
}

echo "================================================================"
echo "  ORBITAL STAIRCASE — PAPER CLAIM VERIFICATION"
echo "  Running all reproducibility scripts"
echo "================================================================"

run_claim "Claim 1: Staircase Fit (AIC/BIC, RMSE, R²)" "paper/claim1_staircase_fit.py"
run_claim "Claim 2: Escape Velocity Scale Independence" "paper/claim2_escape_invariance.py"
run_claim "Claim 3: Lighthouse Detection" "paper/claim3_lighthouse.py"
run_claim "Claim 4: Unity Equation (Monte Carlo)" "paper/claim4_unity_equation.py"
run_claim "Claim 5: Threshold τ = 3/2 Uniqueness" "paper/claim5_threshold_tau.py"
run_claim "Claim 6: Bootstrap Confidence Intervals" "paper/claim6_bootstrap_ci.py"
run_claim "Global Pipeline MC (Joint Significance)" "paper/global_pipeline_mc.py"
run_claim "Threshold Independence (R², LOOCV, MDL)" "paper/threshold_independence.py"

echo ""
echo "================================================================"
echo "  SUMMARY: $PASS passed, $FAIL failed"
echo "================================================================"

if [ $FAIL -gt 0 ]; then
    exit 1
fi
