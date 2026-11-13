"""
PipelineEKF — minimal 2-state EKF for energy estimation.

State:  x = [n_current, t_bar]
Modes:
  predict_all(stages)    Mode A — pre-execution, no EAR readings
  update(stage, E_obs)   Mode B — absorb a real EAR energy reading
  skip_update(stage)     Mode C — graceful degradation, inflate process noise

This is the same math as v1's notebook, lifted into a library so monitor and
notebook share one source of truth. Loaded by 02_monitor.sh via `python3 -c`.
"""

import json
import math
from pathlib import Path

import numpy as np

# ── Two-sided 95% Student-t quantile, scipy-free ─────────────────────────────────
# The monitor runs under system python3, which has numpy but not scipy. The only
# thing we needed scipy for was stats.t.ppf(0.975, df). Table for df 1..30 (exact
# to 3 dp); Cornish-Fisher expansion for df>30 (accurate to ~1e-3 there).
_T_TABLE_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def t_ppf_975(df):
    """Inverse CDF of Student-t at p=0.975 (i.e. the 95% two-sided multiplier)."""
    df = int(round(df))
    if df < 1:
        df = 1
    if df in _T_TABLE_975:
        return _T_TABLE_975[df]
    z = 1.959963985
    g1 = (z ** 3 + z) / 4.0
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96.0
    g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384.0
    return z + g1 / df + g2 / df ** 2 + g3 / df ** 3


class PipelineEKF:
    def __init__(self, fits_path, n0, t0, survival_default=None,
                 sigma_n_frac=0.08, sigma_t_frac=0.05):
        with open(fits_path) as f:
            blob = json.load(f)
        self.fits = blob["fits"]
        self.avg_tokens = blob["avg_tokens"]

        self.x = np.array([float(n0), float(t0)])
        self.P = np.diag([(sigma_n_frac * n0) ** 2, (sigma_t_frac * t0) ** 2])
        self.history = []
        self.survival_default = survival_default or {}

    # ── feature builder ─────────────────────────────────────────────────────
    def _phi(self, stage, n, t):
        features = self.fits[stage]["features"]
        vec = []
        for f in features:
            if f == "n":      vec.append(n)
            elif f == "n_t":  vec.append(n * t)
            elif f == "n_logn": vec.append(n * math.log(max(n, 1)))
            else: raise ValueError(f)
        vec.append(1.0)  # intercept
        return np.array(vec)

    def _E_hat(self, stage, n, t):
        f = self.fits[stage]
        phi = self._phi(stage, n, t)
        theta = np.array(f["theta"])
        E = float(theta @ phi)
        XtX_inv = np.array(f["XtX_inv"])
        var = float(f["sigma2"] * (1 + phi @ XtX_inv @ phi))
        return E, var

    def _H(self, stage, n, t):
        """Jacobian dE/dx (only first feature drives derivatives)."""
        f = self.fits[stage]
        feature = f["features"][0]
        a = f["theta"][0]
        if feature == "n":      dn, dt = a, 0.0
        elif feature == "n_t":  dn, dt = a * t, a * n
        elif feature == "n_logn": dn, dt = a * (math.log(max(n, 1)) + 1), 0.0
        else: raise ValueError(feature)
        return np.array([dn, dt])

    def _survival(self, stage):
        f = self.fits[stage]
        emp = f.get("empirical_survival_rate_mean")
        if emp is not None and emp > 0:
            return emp
        return self.survival_default.get(stage, 1.0)

    def _propagate(self, stage, extra_Q=None):
        s = self._survival(stage)
        F = np.array([[s, 0.0], [0.0, 1.0]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T
        if extra_Q is not None:
            self.P = self.P + extra_Q
        self.x[0] = max(self.x[0], 1.0)

    # ── public API ──────────────────────────────────────────────────────────
    def predict_stage(self, stage):
        n, t = self.x
        E, var_ols = self._E_hat(stage, n, t)
        H = self._H(stage, n, t)
        var_state = float(H @ self.P @ H)
        return E, math.sqrt(var_ols + var_state)

    def update(self, stage, E_observed):
        n, t = self.x
        E, var_ols = self._E_hat(stage, n, t)
        H = self._H(stage, n, t)
        S = float(H @ self.P @ H) + var_ols
        K = (self.P @ H) / S
        innov = E_observed - E
        self.x = self.x + K * innov
        self.P = (np.eye(2) - np.outer(K, H)) @ self.P
        self.history.append({"stage": stage, "E_obs": E_observed,
                             "E_pred": E, "innov": innov, "mode": "B"})
        self._propagate(stage)

    def skip_update(self, stage):
        n, t = self.x
        _, var_ols = self._E_hat(stage, n, t)
        H = self._H(stage, n, t)
        Q = np.outer(H, H) * var_ols
        self.history.append({"stage": stage, "mode": "C"})
        self._propagate(stage, extra_Q=Q)

    def predict_all(self, stages):
        x_sim = self.x.copy(); P_sim = self.P.copy()
        rows = []
        for stage in stages:
            n, t = x_sim
            E, var_ols = self._E_hat(stage, n, t)
            H = self._H(stage, n, t)
            var_state = float(H @ P_sim @ H)
            std = math.sqrt(var_ols + var_state)
            dof = self.fits[stage]["n_obs"] - len(self.fits[stage]["features"]) - 1
            tc = t_ppf_975(max(dof, 1))
            rows.append({
                "stage": stage, "n_in": int(n), "E_hat": E, "std": std,
                "ci_low": E - tc * std, "ci_high": E + tc * std,
            })
            s = self._survival(stage)
            F = np.array([[s, 0.0], [0.0, 1.0]])
            x_sim = F @ x_sim
            P_sim = F @ P_sim @ F.T
        return rows
