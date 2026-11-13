#!/usr/bin/env python3
"""
Per-stage ridge OLS with leave-one-replicate-out CV.

This is the only Python you'll touch in v2. Everything else is bash.
Math:
  Closed-form ridge:     theta = (XtX + lambda*I)^-1 Xt y
  CV objective:          minimize sum of squared held-out residuals
  Prediction interval:   t_{0.975, dof} * sigma * sqrt(1 + x* (XtX+lI)^-1 x*)

Input:  measurements/measurements.csv  (from 03_harvest.sh)
Output: measurements/ols_fits.json     (used by 02_monitor.sh and 05_validate.sh)

Why ridge: with 12 candidate features and ~12 rows per stage, plain OLS overfits.
Ridge with CV-tuned lambda is the smallest possible "real" upgrade over plain lstsq.
"""

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WS_V2 = HERE.parent
MEAS = WS_V2 / "measurements" / "measurements.csv"
OUT = WS_V2 / "measurements" / "ols_fits.json"

# ── Per-stage feature spec ───────────────────────────────────────────────────
# Pre-execution features: known BEFORE the run from corpus metadata only.
# These are the ones that go into Mode A prediction.
#
# 'n', 'n_t', 'n_logn' are derived from size (and a fixed avg-tokens assumption).
PRE_FEATURES = {
    "data_splitting":       ["n"],
    "string_normalization": ["n", "n_t"],
    "heuristic_filtering":  ["n", "n_t"],
    "deduplication":        ["n", "n_logn"],
}

# Average tokens/doc (placeholder — replace with empirical mean once you parse
# the parquet files; for now this matches v1's AVG_TOKENS = 583).
AVG_TOKENS = 583.0


def build_feature_matrix(rows, features):
    """Returns (X with intercept appended, y) where y is energy_j."""
    X, y = [], []
    for r in rows:
        n = float(r["size"])
        t = AVG_TOKENS
        feat = []
        for f in features:
            if f == "n":      feat.append(n)
            elif f == "n_t":  feat.append(n * t)
            elif f == "n_logn": feat.append(n * math.log(max(n, 1)))
            else: raise ValueError(f"unknown feature {f}")
        feat.append(1.0)  # intercept
        X.append(feat)
        # Energy = DC_NODE_POWER_W * TIME_SEC  (canonical EAR-derived energy)
        E = float(r["dc_node_power_w"]) * float(r["time_sec"])
        y.append(E)
    return np.array(X), np.array(y)


def ridge_fit(X, y, lam):
    """Closed-form ridge regression."""
    XtX = X.T @ X
    p = X.shape[1]
    # Don't regularize the intercept (last column)
    reg = lam * np.eye(p)
    reg[-1, -1] = 0.0
    theta = np.linalg.solve(XtX + reg, X.T @ y)
    return theta


def loo_replicate_cv(rows, features, lambdas):
    """
    Leave-one-replicate-out CV: each fold holds out all measurements with a
    given replicate index. Returns lambda* that minimizes mean held-out MSE.
    """
    reps = sorted({r["replicate"] for r in rows})
    if len(reps) < 2:
        return 0.0  # not enough replicates to do CV — fall back to plain OLS

    best_lam, best_mse = None, float("inf")
    for lam in lambdas:
        sq_err = []
        for held_out in reps:
            train = [r for r in rows if r["replicate"] != held_out]
            test  = [r for r in rows if r["replicate"] == held_out]
            if not train or not test:
                continue
            X_tr, y_tr = build_feature_matrix(train, features)
            X_te, y_te = build_feature_matrix(test,  features)
            theta = ridge_fit(X_tr, y_tr, lam)
            pred  = X_te @ theta
            sq_err.extend((pred - y_te) ** 2)
        mse = float(np.mean(sq_err)) if sq_err else float("inf")
        if mse < best_mse:
            best_mse, best_lam = mse, lam
    return best_lam


def fit_stage(stage, rows, features):
    X, y = build_feature_matrix(rows, features)
    n_obs, p = X.shape
    if n_obs < p + 1:
        print(f"[{stage}] only {n_obs} rows for {p} params — skipping")
        return None

    lambdas = np.logspace(-3, 4, 30)
    lam = loo_replicate_cv(rows, features, lambdas)

    theta = ridge_fit(X, y, lam)
    resid = y - X @ theta
    sigma2 = float(resid @ resid) / max(n_obs - p, 1)

    XtX = X.T @ X
    reg = lam * np.eye(p); reg[-1, -1] = 0.0
    XtX_inv = np.linalg.inv(XtX + reg)

    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")

    # Mean survival rate from observed n_in / n_out (where present)
    surv = []
    for r in rows:
        try:
            n_in = float(r["n_in"]); n_out = float(r["n_out"])
            if n_in > 0:
                surv.append(n_out / n_in)
        except (ValueError, TypeError):
            pass

    return {
        "stage": stage,
        "features": features,
        "n_obs": n_obs,
        "lambda": float(lam),
        "theta": theta.tolist(),
        "sigma2": sigma2,
        "XtX_inv": XtX_inv.tolist(),
        "r2": r2,
        "empirical_survival_rate_mean": (sum(surv) / len(surv)) if surv else None,
        "empirical_survival_rate_n": len(surv),
    }


def main():
    if not MEAS.exists():
        sys.exit(f"No measurements.csv at {MEAS} — run 03_harvest.sh first")

    rows_by_stage = defaultdict(list)
    with open(MEAS) as f:
        for r in csv.DictReader(f):
            rows_by_stage[r["stage"]].append(r)

    fits = {}
    for stage in PRE_FEATURES:
        rows = rows_by_stage.get(stage, [])
        if not rows:
            print(f"[{stage}] no rows — skipping")
            continue
        fit = fit_stage(stage, rows, PRE_FEATURES[stage])
        if fit is None:
            continue
        fits[stage] = fit
        a = fit["theta"][0]; b = fit["theta"][-1]
        print(f"[{stage}] n_obs={fit['n_obs']}  λ={fit['lambda']:.3g}  "
              f"R²={fit['r2']:.3f}  E ≈ {a:.4g}·{fit['features'][0]} + ... + {b:.3g}  "
              f"survival={fit['empirical_survival_rate_mean']}")

    with open(OUT, "w") as f:
        json.dump({"avg_tokens": AVG_TOKENS, "fits": fits}, f, indent=2)
    print(f"[fit] Wrote {OUT}")


if __name__ == "__main__":
    main()
