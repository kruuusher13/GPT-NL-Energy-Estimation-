#!/usr/bin/env bash
# v2 orchestration entry point.
#
# Two modes:
#   bash run_all.sh submit    # 00 + 01: generate configs, fire jobs
#   bash run_all.sh harvest   # 03 + 04 + 05 + 06: harvest, fit, validate, loops-check
#
# Live monitoring is interactive and stays separate (02_monitor.sh).
#
# Typical 12-hour flow:
#   bash run_all.sh submit             # H0 — fire jobs, walk away
#   (queue time, jobs run, ...)
#   bash run_all.sh harvest            # H4+ — once jobs land
#   bash 02_monitor.sh                 # optional: live EKF updates as jobs finish

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

case "${1:-}" in
  submit)
    bash 00_gen_configs.sh
    bash 01_submit_all.sh
    ;;
  harvest)
    bash 03_harvest.sh
    python3 04_fit_ridge.py
    bash 05_validate.sh
    bash 06_loops_integrate.sh
    python3 07_plot.py
    echo
    echo "v2 sweep complete. Artifacts:"
    echo "  measurements/measurements.csv"
    echo "  measurements/ols_fits.json"
    echo "  measurements/loops_validation.csv"
    echo "  figures/fig_*.png"
    ;;
  *)
    echo "Usage: bash run_all.sh {submit|harvest}"
    exit 1
    ;;
esac
