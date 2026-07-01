"""
Extended Kalman Filter (EKF) for live pipeline energy estimation.

The EKF starts with a model prediction (wide confidence band) and sequentially
recalibrates as each parallel task reports its measured energy. An innovation
gate rejects contaminated shared-node readings.

State: per-stage energy total.
Prior: model prediction (physics or ML).
Update: each task reading → recalibrate stage → update total estimate + CI.

This is the core of the live monitor (energy_demo.py).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from gptnl_energy.data import (
    PIPELINE_STAGES,
    corpus_chars,
    CALIBRATION_CORPUS,
    CALIBRATION_CHARS,
    load_fits,
)


class LiveEstimator:
    """Online Kalman estimator for pipeline total energy.

    State = per-stage energy. Prior = model prediction.
    Each task reading recalibrates the stage online (blend prior ↔ data-extrapolation),
    with an innovation gate that rejects contaminated readings.

    Usage:
        est = LiveEstimator(fits, n=400000, corpus="american_stories",
                           tasks_per_stage={"heuristic_filtering": 16})
        for measurement in stream:
            ok, ratio = est.update(stage="heuristic_filtering", energy_j=42000)
            total, ci = est.estimate()
            print(f"{total/1000:.1f} kJ ± {ci/1000:.1f}")
    """

    def __init__(
        self,
        fits: Dict,
        n: int,
        corpus: str,
        stages: Optional[List[str]] = None,
        tasks_per_stage: Optional[Dict[str, int]] = None,
        prior_rel: float = 0.45,
        k0: float = 2.0,
        gate_ratio: float = 2.5,
    ):
        self.k0 = k0
        self.gate_ratio = gate_ratio

        self.stages = stages or [s for s in PIPELINE_STAGES if s in fits]
        # Model prediction per stage (with cross-corpus transfer)
        self.pred: Dict[str, float] = {}
        chars = corpus_chars(corpus)
        for s in self.stages:
            f = fits.get(s, {"c1": 0.0, "c0": 0.0})
            c1 = f["c1"]
            if corpus != CALIBRATION_CORPUS:
                c1 = c1 / CALIBRATION_CHARS * chars
            self.pred[s] = max(f["c0"] + c1 * n, 1.0)

        self.M = {}
        for s in self.stages:
            self.M[s] = max(tasks_per_stage.get(s, 1) if tasks_per_stage else 1, 1)

        # Initial variance per stage (prior uncertainty)
        self.P0: Dict[str, float] = {}
        for s in self.stages:
            self.P0[s] = (prior_rel * self.pred[s]) ** 2

        self.seen: Dict[str, List[float]] = {s: [] for s in self.stages}
        self.rejected: int = 0
        self._total_observed: float = 0.0

    def update(self, stage: str, energy_j: float) -> Tuple[bool, float]:
        """Feed one task reading into the estimator.

        Returns (accepted: bool, ratio_to_stage_budget: float).
        Gate fires when a single reading exceeds the whole stage's prediction
        by gate_ratio × — robust to heterogeneous sub-jobs (e.g., dedup's
        dominant signature pass).
        """
        sp = self.pred.get(stage, 1.0)
        ratio = energy_j / max(sp, 1e-9)

        if ratio > self.gate_ratio:
            # Contaminated spike → substitute with model prediction per-task share
            self.seen[stage].append(sp / self.M.get(stage, 1))
            self.rejected += 1
            return False, ratio

        self.seen[stage].append(energy_j)
        self._total_observed += energy_j
        return True, ratio

    def estimate(self) -> Tuple[float, float]:
        """Compute current total energy estimate + 95% CI half-width (Joules).

        For unstarted stages: pure prior.
        For completed stages: sum of measured values.
        For partially-observed stages: blend prior with data-extrapolation.
        """
        total = 0.0
        total_var = 0.0

        for s in self.stages:
            k = len(self.seen[s])
            M = self.M[s]
            prior = self.pred[s]

            if k == 0:
                # Not started → pure prior
                est = prior
                var = self.P0[s]
            elif k >= M:
                # Stage complete → known (only measurement noise)
                est = sum(self.seen[s])
                var = (0.03 * est) ** 2
            else:
                # Partial → blend prior ↔ data
                measured = sum(self.seen[s])
                extrap = measured * M / k  # data-only stage total
                w = k / (k + self.k0)       # trust data more as k grows
                est = w * extrap + (1 - w) * prior

                if k > 1:
                    spread = np.var(self.seen[s])
                else:
                    spread = (0.30 * np.mean(self.seen[s])) ** 2
                var = (1 - w) ** 2 * self.P0[s] + spread * max(M - k, 0)

            total += est
            total_var += var

        ci = 1.96 * math.sqrt(max(total_var, 0))
        return total, ci

    @property
    def total_predicted(self) -> float:
        """Pre-run model prediction (sum of per-stage priors)."""
        return sum(self.pred.values())

    @property
    def total_observed(self) -> float:
        """Sum of all accepted measurements so far."""
        return self._total_observed

    @property
    def acceptance_rate(self) -> float:
        """Fraction of readings accepted by the innovation gate."""
        total_readings = sum(len(v) for v in self.seen.values())
        if total_readings == 0:
            return 1.0
        return 1.0 - self.rejected / total_readings


def make_estimator_from_fits(
    fits_path: str,
    n: int,
    corpus: str,
    stages: Optional[List[str]] = None,
    tasks_per_stage: Optional[Dict[str, int]] = None,
    **kwargs,
) -> LiveEstimator:
    """Convenience: load fits JSON → LiveEstimator."""
    fits = load_fits(fits_path)
    return LiveEstimator(fits, n, corpus, stages, tasks_per_stage, **kwargs)


def make_estimator_from_model(
    model,
    n: int,
    corpus: str,
    stages: Optional[List[str]] = None,
    tasks_per_stage: Optional[Dict[str, int]] = None,
    **kwargs,
) -> LiveEstimator:
    """Convenience: use a trained model's predictions as the EKF prior."""
    preds = model.predict(n, corpus)
    stages = stages or [s for s in PIPELINE_STAGES if s in preds]

    # Convert model predictions to OLS-fit format expected by LiveEstimator
    fits = {}
    for s in stages:
        if s in preds:
            fits[s] = {"c1": 0.0, "c0": preds[s], "sigma": preds[s] * 0.3}

    return LiveEstimator(fits, n, corpus, stages, tasks_per_stage, **kwargs)
