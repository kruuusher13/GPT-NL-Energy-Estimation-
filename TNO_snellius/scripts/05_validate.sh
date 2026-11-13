#!/usr/bin/env bash
# Validation: CASE-EKF vs raw-EAR vs constant-power baseline, on held-out replicate.
#
# This is the bit that makes the "novel calibrated CI" claim falsifiable.
# Prints a MAPE + CI-coverage table. If CASE-EKF doesn't beat raw-EAR in
# extrapolation MAPE, that's a real finding too — say so in the paper.
#
# Also cross-checks the roofline regime against EAR's own phase classification
# (CPU_BOUND / MEM_BOUND / MIX_COMP), if EAR populated those columns.
#
# Usage:
#   source env.sh
#   bash 05_validate.sh

set -euo pipefail
source "$(dirname "$0")/env.sh"

[ -f "$MEAS_DIR/measurements.csv" ] || { echo "No measurements yet."; exit 1; }
[ -f "$MEAS_DIR/ols_fits.json" ]    || { echo "Run 04_fit_ridge.py first."; exit 1; }

python3 - "$MEAS_DIR/measurements.csv" "$MEAS_DIR/ols_fits.json" <<'PY'
import csv, json, math, sys
from collections import defaultdict
import numpy as np

meas_path, fits_path = sys.argv[1:3]

rows = list(csv.DictReader(open(meas_path)))
fits = json.load(open(fits_path))["fits"]
AVG_TOKENS = json.load(open(fits_path))["avg_tokens"]

# Held-out: replicate 3.  Train on replicates 1 & 2.
def split(rows):
    train = [r for r in rows if r["replicate"] != "3"]
    test  = [r for r in rows if r["replicate"] == "3"]
    return train, test

def phi(stage, n, t, features):
    v = []
    for f in features:
        if f == "n":      v.append(n)
        elif f == "n_t":  v.append(n * t)
        elif f == "n_logn": v.append(n * math.log(max(n, 1)))
    v.append(1.0)
    return np.array(v)

def case_ekf_predict(stage, n, t):
    f = fits[stage]
    return float(np.array(f["theta"]) @ phi(stage, n, t, f["features"]))

def constant_power_predict(stage, n, t, train):
    """Baseline: mean DC power across training × predicted-time-from-n."""
    pwrs = [float(r["dc_node_power_w"]) for r in train if r["stage"] == stage]
    times = [(float(r["size"]), float(r["time_sec"]))
             for r in train if r["stage"] == stage]
    if not pwrs or not times: return None
    mean_p = sum(pwrs) / len(pwrs)
    # Time-vs-n linear fit (1-feature)
    ns = np.array([x[0] for x in times]); ts = np.array([x[1] for x in times])
    A = np.vstack([ns, np.ones_like(ns)]).T
    a, b = np.linalg.lstsq(A, ts, rcond=None)[0]
    return mean_p * (a * n + b)

def linear_n_predict(stage, n, t, train):
    """Baseline: ignore roofline. Always use phi=n."""
    sub = [r for r in train if r["stage"] == stage]
    if len(sub) < 2: return None
    ns = np.array([float(r["size"]) for r in sub])
    Es = np.array([float(r["dc_node_power_w"]) * float(r["time_sec"]) for r in sub])
    A = np.vstack([ns, np.ones_like(ns)]).T
    a, b = np.linalg.lstsq(A, Es, rcond=None)[0]
    return float(a * n + b)

def mape(y_true, y_pred):
    e = [abs((p - t) / t) for t, p in zip(y_true, y_pred) if t > 0 and p is not None]
    return 100 * sum(e) / len(e) if e else float("nan")

# ── Run comparison per stage ───────────────────────────────────────────────
by_stage = defaultdict(list)
for r in rows:
    by_stage[r["stage"]].append(r)

print(f"{'stage':<25}  {'n_test':>6}  {'CASE-EKF':>10}  {'raw-EAR':>10}  {'const-P':>10}  {'linear-n':>10}")
print("-" * 80)

overall = {"case": [], "raw": [], "const": [], "lin": []}
for stage, stage_rows in by_stage.items():
    if stage not in fits: continue
    train, test = split(stage_rows)
    if not test: continue

    y_true = [float(r["dc_node_power_w"]) * float(r["time_sec"]) for r in test]
    ns     = [float(r["size"]) for r in test]

    y_case = [case_ekf_predict(stage, n, AVG_TOKENS) for n in ns]
    y_const = [constant_power_predict(stage, n, AVG_TOKENS, train) for n in ns]
    y_lin  = [linear_n_predict(stage, n, AVG_TOKENS, train) for n in ns]
    # Raw-EAR baseline = mean training energy (the dumbest possible predictor)
    train_E = [float(r["dc_node_power_w"]) * float(r["time_sec"])
               for r in train if r["stage"] == stage]
    y_raw  = [sum(train_E) / len(train_E) if train_E else None] * len(y_true)

    m_case = mape(y_true, y_case)
    m_raw  = mape(y_true, y_raw)
    m_cst  = mape(y_true, y_const)
    m_lin  = mape(y_true, y_lin)

    print(f"{stage:<25}  {len(y_true):>6}  {m_case:>9.1f}%  {m_raw:>9.1f}%  {m_cst:>9.1f}%  {m_lin:>9.1f}%")

    overall["case"].extend(zip(y_true, y_case))
    overall["raw"].extend(zip(y_true, y_raw))
    overall["const"].extend(zip(y_true, y_const))
    overall["lin"].extend(zip(y_true, y_lin))

print("-" * 80)
for label, key in [("CASE-EKF", "case"), ("raw-EAR", "raw"),
                   ("const-P", "const"), ("linear-n", "lin")]:
    ts = [t for t, _ in overall[key]]; ps = [p for _, p in overall[key]]
    print(f"  overall MAPE  {label:<10} = {mape(ts, ps):.1f}%")

# ── Roofline vs EAR phase tag cross-check ─────────────────────────────────
# EAR's app CSV may include CPU_BOUND / MEM_BOUND columns (depends on EAR
# version). If we got CPI in measurements.csv we can do a coarse own-check:
#   high CPI + low CPU_GFLOPS → memory-bound
#   low CPI + high CPU_GFLOPS → compute-bound
print()
print("EAR-derived bound check (CPI > 1 → memory-bound, CPU_GFLOPS > 50 → compute-bound):")
for stage, srows in by_stage.items():
    cpis = [float(r["cpi"]) for r in srows if r.get("cpi") not in (None, "", "0")]
    gflops = [float(r["cpu_gflops"]) for r in srows if r.get("cpu_gflops")]
    if not cpis or not gflops: continue
    mc = sum(cpis)/len(cpis); mg = sum(gflops)/len(gflops)
    bound = "MEM" if mc > 1 else ("CPU" if mg > 50 else "MIX")
    print(f"  {stage:<25}  CPI={mc:.2f}  CPU_GFLOPS={mg:.2f}  → {bound}-bound")
PY
