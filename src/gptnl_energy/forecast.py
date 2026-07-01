"""
Pre-run energy forecast — the one-shot prediction tool.

Usage:
    from gptnl_energy.forecast import Forecast

    f = Forecast.from_fits("paper/data/ols_fits_with_dedup.json")
    result = f.predict(n=400000, corpus="american_stories")
    print(f"{result.total_kwh:.2f} kWh  —  €{result.eur_cost:.2f}")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from gptnl_energy.data import (
    PIPELINE_STAGES,
    CORPUS_CHARS,
    CALIBRATION_CORPUS,
    CALIBRATION_CHARS,
    DEDUP_OOM_N,
    DEFAULT_PUE,
    DEFAULT_EUR_PER_KWH,
    DEFAULT_KG_CO2_PER_KWH,
    load_fits,
    corpus_chars,
)
from gptnl_energy.models import BaseEnergyModel


@dataclass
class StageForecast:
    """Per-stage energy forecast."""
    stage: str
    energy_j: float
    band_j: float  # 95% CI half-width
    share_pct: float


@dataclass
class ForecastResult:
    """Complete pipeline energy forecast."""
    n: int
    corpus: str
    n_tasks: int
    hardware: str
    stages: List[StageForecast] = field(default_factory=list)
    total_j: float = 0.0
    total_band_j: float = 0.0

    @property
    def total_kj(self) -> float:
        return self.total_j / 1000

    @property
    def total_mj(self) -> float:
        return self.total_j / 1e6

    @property
    def total_kwh(self, pue: float = DEFAULT_PUE) -> float:
        return self.total_j / 3.6e6 * pue

    @property
    def eur_cost(self, eur_per_kwh: float = DEFAULT_EUR_PER_KWH,
                 pue: float = DEFAULT_PUE) -> float:
        return self.total_kwh * eur_per_kwh

    @property
    def kg_co2(self, kg_per_kwh: float = DEFAULT_KG_CO2_PER_KWH,
               pue: float = DEFAULT_PUE) -> float:
        return self.total_kwh * kg_per_kwh

    @property
    def dedup_oom_warning(self) -> bool:
        return self.n >= DEDUP_OOM_N and any(
            s.stage == "deduplication" for s in self.stages
        )

    def format_terminal(self, pue: float = DEFAULT_PUE,
                        eur_per_kwh: float = DEFAULT_EUR_PER_KWH,
                        kg_per_kwh: float = DEFAULT_KG_CO2_PER_KWH) -> str:
        """Render as a terminal-friendly table."""
        chars = corpus_chars(self.corpus)
        lines = [
            f"",
            f"  Energy forecast — {self.n:,} docs | corpus={self.corpus} "
            f"(~{chars:,} chars/doc) | n_tasks={self.n_tasks} | {self.hardware}",
            f"  {'=' * 66}",
            f"  {'stage':<26}{'energy':>12}{'± band':>12}   share",
            f"  {'-' * 66}",
        ]
        for s in self.stages:
            lines.append(
                f"  {s.stage:<26}{s.energy_j/1000:>9.1f} kJ"
                f"{s.band_j/1000:>9.1f} kJ{s.share_pct:>7.0f}%"
            )
        lines.extend([
            f"  {'-' * 66}",
            f"  {'TOTAL PIPELINE':<26}{self.total_j/1e6:>9.2f} MJ"
            f"{self.total_band_j/1e6:>9.2f} MJ",
            f"  {'=' * 66}",
            f"  =  {self.total_kwh:8.2f} kWh   (PUE {pue})",
            f"  =  EUR {self.eur_cost:8.2f}   (@ EUR {eur_per_kwh}/kWh)",
            f"  =  {self.kg_co2:8.2f} kg CO2  (@ {kg_per_kwh} kg/kWh)",
        ])
        if self.dedup_oom_warning:
            lines.extend([
                f"",
                f"  ⚠  WARNING: deduplication OOMs at ~{DEDUP_OOM_N:,} docs "
                f"(signature pass).",
                f"           This run will FAIL at dedup — shard it or raise memory.",
            ])
        lines.append("")
        return "\n".join(lines)


class Forecast:
    """Pre-run energy forecaster.

    Can use either:
      - Physics model (OLS fits from JSON): Forecast.from_fits(path)
      - Any trained model: Forecast.from_model(model)
    """

    def __init__(self, fits: Optional[Dict] = None, model: Optional[BaseEnergyModel] = None):
        self._fits = fits
        self._model = model

    @classmethod
    def from_fits(cls, fits_path: str) -> "Forecast":
        """Create forecaster from OLS coefficient fits JSON."""
        return cls(fits=load_fits(fits_path))

    @classmethod
    def from_model(cls, model: BaseEnergyModel) -> "Forecast":
        """Create forecaster from a trained model."""
        return cls(model=model)

    def predict(
        self,
        n: int,
        corpus: str = CALIBRATION_CORPUS,
        n_tasks: int = 1,
        hardware: str = "genoa",
        stages: Optional[List[str]] = None,
    ) -> ForecastResult:
        """Predict total pipeline energy for a given configuration."""
        chars = corpus_chars(corpus)

        if self._model is not None:
            preds = self._model.predict(n, corpus, n_tasks, stages)
            total_j = preds.get("total_j", sum(
                v for k, v in preds.items() if k not in ("total_j", "ci_95_j")
            ))
            band_j = preds.get("ci_95_j", total_j * 0.3)

            # Build per-stage breakdown
            stage_preds = []
            for s in stages or PIPELINE_STAGES:
                ej = preds.get(s, 0.0)
                stage_preds.append(StageForecast(
                    stage=s, energy_j=ej, band_j=band_j / len(stages or PIPELINE_STAGES),
                    share_pct=ej / total_j * 100 if total_j > 0 else 0,
                ))
        else:
            # Physics model
            fits = self._fits or {}
            stages_list = stages or [s for s in PIPELINE_STAGES if s in fits]
            stage_preds = []
            total_j = 0.0
            total_var = 0.0

            for s in stages_list:
                fit = fits.get(s, {"c1": 0.0, "c0": 0.0, "sigma": 0.0})
                c1, c0 = fit["c1"], fit["c0"]

                # Cross-corpus transfer: per-character constant
                if corpus != CALIBRATION_CORPUS:
                    c1 = c1 / CALIBRATION_CHARS * chars

                ej = c0 + c1 * n
                band = 1.96 * fit.get("sigma", 0.0)
                total_j += ej
                total_var += fit.get("sigma", 0.0) ** 2
                stage_preds.append(StageForecast(
                    stage=s, energy_j=ej, band_j=band, share_pct=0.0,
                ))

            band_j = 1.96 * math.sqrt(max(total_var, 0))

            # Recompute shares
            for sp in stage_preds:
                sp.share_pct = sp.energy_j / total_j * 100 if total_j > 0 else 0

        return ForecastResult(
            n=n, corpus=corpus, n_tasks=n_tasks, hardware=hardware,
            stages=stage_preds, total_j=total_j, total_band_j=band_j,
        )
