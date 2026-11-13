import csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_ekf import PipelineEKF

csv_path, fits_path, state_path, stage = sys.argv[1:5]
# eacct -c is HEADERLESS: parse by position. The real measurement row has
# STEP_ID (col2) == "0"; the pseudo-step "4294967291" carries only zeros.
# col26 = TIME_SEC, col32 = DC_NODE_POWER_W  (1-based).
E_obs = 0.0
with open(csv_path) as f:
    for parts in csv.reader(f, delimiter=';'):
        if len(parts) < 33:
            continue
        if parts[1].strip() in ("", "4294967291"):
            continue
        try:
            t = float(parts[25]); p = float(parts[31])
        except ValueError:
            continue
        if p > 0 and t > 0 and p * t > E_obs:
            E_obs = p * t
if E_obs <= 0:
    sys.exit(0)

state = json.load(open(state_path))
ekf = PipelineEKF(fits_path, n0=state["x"][0], t0=state["x"][1])
import numpy as np
ekf.P = np.array(state["P"]); ekf.history = state["history"]
ekf.update(stage, E_obs)

state["x"] = ekf.x.tolist()
state["P"] = ekf.P.tolist()
state["history"] = ekf.history
json.dump(state, open(state_path, "w"), indent=2)

STAGES = ["data_splitting", "string_normalization", "heuristic_filtering", "deduplication"]
# Only predict stages that actually have a fit in ols_fits.json — deduplication
# has no v2 fit yet, and predicting it would KeyError in lib_ekf._E_hat.
remaining = [s for s in STAGES[STAGES.index(stage)+1:] if s in ekf.fits] \
            if stage in STAGES else []
if remaining:
    rows = ekf.predict_all(remaining)
    total_E = sum(r["E_hat"] for r in rows)
    total_std = (sum(r["std"]**2 for r in rows))**0.5
    print(f"  remaining: {total_E/1e3:.1f} kJ ± {2*total_std/1e3:.1f} kJ  ({len(remaining)} stages)")
