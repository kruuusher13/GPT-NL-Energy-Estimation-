"""
Swappable model registry for energy prediction.

Models implement a common interface and are registered by name.
Users call `get_model("ols")` or `get_model("gbm")` — the rest is plug-and-play.

Architecture:
  - BaseEnergyModel: fit/predict interface
  - PerStageModel: predicts each pipeline stage independently (OLS, Ridge)
  - JointModel: predicts all stages from features in one model (GBM, MLP, FTT)
  - TotalModel: predicts whole-run total energy (Kalman-FTT)
"""

from __future__ import annotations

import abc
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from gptnl_energy.data import PIPELINE_STAGES, load_fits, corpus_chars, CALIBRATION_CORPUS, CALIBRATION_CHARS

logger = logging.getLogger(__name__)

# ── Registry ──────────────────────────────────────────────────────────────────
_registry: Dict[str, type] = {}


def register_model(name: str):
    """Decorator to register a model class."""
    def decorator(cls):
        _registry[name] = cls
        return cls
    return decorator


def get_model(name: str, **kwargs) -> "BaseEnergyModel":
    """Factory: get a model by name. Raises KeyError if not registered."""
    if name not in _registry:
        raise KeyError(f"Unknown model '{name}'. Available: {list(_registry)}")
    return _registry[name](**kwargs)


def available_models() -> List[str]:
    """List all registered model names."""
    return sorted(_registry.keys())


# ── Base class ────────────────────────────────────────────────────────────────
class BaseEnergyModel(abc.ABC):
    """Every energy model must implement fit() and predict()."""

    def __init__(self, **kwargs):
        self._metadata: Dict[str, Any] = {}

    @abc.abstractmethod
    def fit(self, df: pd.DataFrame, target: str = "energy_j"):
        """Train the model on measurement data."""
        ...

    @abc.abstractmethod
    def predict(self, n: int, corpus: str, n_tasks: int = 1,
                stages: Optional[List[str]] = None) -> Dict[str, float]:
        """Predict energy (Joules) for each stage and total."""
        ...

    def predict_kwh(self, n: int, corpus: str, n_tasks: int = 1,
                    pue: float = 1.20) -> float:
        """Predict total energy in kWh (with PUE overhead)."""
        result = self.predict(n, corpus, n_tasks)
        return result["total_j"] / 3.6e6 * pue

    def save(self, path: str):
        """Persist model to disk."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({"state": self._get_state(), "metadata": self._metadata}, path)

    @classmethod
    def load(cls, path: str) -> "BaseEnergyModel":
        """Load model from disk."""
        data = joblib.load(path)
        model = cls()
        model._set_state(data["state"])
        model._metadata = data.get("metadata", {})
        return model

    def _get_state(self) -> Dict:
        """Override in subclasses to serialize model-specific state."""
        return {}

    def _set_state(self, state: Dict):
        """Override in subclasses to restore model-specific state."""
        pass


# ── Per-stage physics model (OLS) ─────────────────────────────────────────────
@register_model("ols")
@register_model("linear")
class PerStageOLSModel(BaseEnergyModel):
    """Calibrated per-stage linear model: E_stage = c0 + c1 * n.

    Loads coefficients from a JSON fits file. For unseen corpora, transfers
    via the per-character constant (energy/char is ~stable across corpora).
    """

    def __init__(self, fits_path: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self._fits_path = fits_path
        self._fits: Dict = {}

    def fit(self, df: pd.DataFrame, target: str = "energy_j"):
        """Fit per-stage OLS from data. Uses grouped (size-averaged) regression."""
        stages = [s for s in PIPELINE_STAGES if s in df.stage.unique()]
        fits = {}
        for stage in stages:
            sub = df[df.stage == stage]
            grp = sub.groupby("n")[target].mean().reset_index().sort_values("n")
            if len(grp) < 2:
                continue
            A = np.vstack([grp.n.values, np.ones(len(grp))]).T
            (c1, c0), residuals, _, _ = np.linalg.lstsq(A, grp[target].values, rcond=None)
            sigma = float(np.std(residuals)) if len(residuals) > 0 else 0.0
            fits[stage] = {"c1": float(c1), "c0": float(c0), "sigma": sigma}
        self._fits = fits
        self._metadata["num_stages"] = len(fits)
        return self

    def load_fits(self, fits_path: str):
        """Load pre-computed OLS coefficients from JSON."""
        self._fits = load_fits(fits_path)
        self._fits_path = fits_path

    def predict(self, n: int, corpus: str, n_tasks: int = 1,
                stages: Optional[List[str]] = None) -> Dict[str, float]:
        if stages is None:
            stages = [s for s in PIPELINE_STAGES if s in self._fits]

        chars = corpus_chars(corpus)
        per_stage = {}
        total = 0.0
        total_var = 0.0

        for stage in stages:
            fit = self._fits.get(stage, {"c1": 0.0, "c0": 0.0, "sigma": 0.0})
            c1, c0 = fit["c1"], fit["c0"]

            # Cross-corpus transfer: scale c1 by chars/doc ratio
            if corpus != CALIBRATION_CORPUS:
                c1 = c1 / CALIBRATION_CHARS * chars

            energy = c0 + c1 * n
            per_stage[stage] = max(energy, 0.0)
            total += energy
            total_var += fit.get("sigma", 0.0) ** 2

        import math
        per_stage["total_j"] = total
        per_stage["ci_95_j"] = 1.96 * math.sqrt(max(total_var, 0))
        return per_stage

    def _get_state(self) -> Dict:
        return {"fits": self._fits, "fits_path": self._fits_path}

    def _set_state(self, state: Dict):
        self._fits = state["fits"]
        self._fits_path = state.get("fits_path")


# ── Sklearn-based joint model ─────────────────────────────────────────────────
class _SklearnJointModel(BaseEnergyModel):
    """Base for models that predict all stages from a feature matrix.

    Subclasses set `_model_name` and override `_build_model()`.
    """

    _model_name: str = "sklearn"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model: Any = None
        self._features: List[str] = []
        self._log_scale: bool = True

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build the standard feature matrix."""
        X = pd.DataFrame()
        X["ln"] = np.log10(df.n.clip(lower=1))
        X["lchars"] = np.log10(df.chars.clip(lower=1))
        X["lnt"] = np.log2(df.ntasks.clip(lower=1))
        # Stage as integer index
        stage_map = {s: i for i, s in enumerate(PIPELINE_STAGES)}
        X["sidx"] = df.stage.map(stage_map).fillna(0).astype(int)
        self._features = list(X.columns)
        return X

    def fit(self, df: pd.DataFrame, target: str = "energy_j"):
        X = self._build_features(df)
        y = df[target].values
        self._model = self._build_model()
        if self._log_scale:
            y = np.log10(y.clip(min=1e-6))
        self._model.fit(X.values, y)
        self._metadata["features"] = self._features
        return self

    def predict(self, n: int, corpus: str, n_tasks: int = 1,
                stages: Optional[List[str]] = None) -> Dict[str, float]:
        if stages is None:
            stages = [s for s in PIPELINE_STAGES if s in self._fits] if hasattr(self, '_fits') else PIPELINE_STAGES

        chars = corpus_chars(corpus)
        rows = []
        stage_map = {s: i for i, s in enumerate(PIPELINE_STAGES)}
        for stage in stages:
            rows.append({
                "ln": np.log10(max(n, 1)),
                "lchars": np.log10(max(chars, 1)),
                "lnt": np.log2(max(n_tasks, 1)),
                "sidx": stage_map.get(stage, 0),
            })
        X = pd.DataFrame(rows)[self._features]
        preds = self._model.predict(X.values)
        if self._log_scale:
            preds = 10 ** preds

        result = {stage: float(preds[i]) for i, stage in enumerate(stages)}
        result["total_j"] = sum(result.values())
        return result

    @abc.abstractmethod
    def _build_model(self):
        ...

    def _get_state(self) -> Dict:
        return {"model": self._model, "features": self._features, "log_scale": self._log_scale}

    def _set_state(self, state: Dict):
        self._model = state["model"]
        self._features = state["features"]
        self._log_scale = state.get("log_scale", True)


@register_model("ridge")
class RidgeModel(_SklearnJointModel):
    _model_name = "ridge"

    def _build_model(self):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


@register_model("gbm")
class GBMModel(_SklearnJointModel):
    _model_name = "gbm"

    def _build_model(self):
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            max_iter=300, max_depth=3, learning_rate=0.06,
            min_samples_leaf=3, random_state=0,
        )


@register_model("mlp")
class MLPModel(_SklearnJointModel):
    _model_name = "mlp"

    def _build_model(self):
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 32), max_iter=2000,
                alpha=1e-3, random_state=0,
            ),
        )
