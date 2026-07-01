"""
FT-Transformer neural models for energy prediction (PyTorch).

These are OPTIONAL — import only if torch is available. The model registry
auto-detects availability and skips registration if torch is missing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.info("torch not available — FT-Transformer models disabled")


if HAS_TORCH:
    from gptnl_energy.models import BaseEnergyModel, register_model, PIPELINE_STAGES
    from gptnl_energy.data import corpus_chars

    NUM_FEATURES = ["ln", "lchars", "lnt"]


    class FeatureTokenizerTransformer(nn.Module):
        """FT-Transformer: each numeric feature → token, + stage embedding + CLS head.

        Architecture mirrors the original train_and_save.py implementation.
        """

        def __init__(self, n_num: int, n_stage: int, d: int = 32,
                     heads: int = 4, layers: int = 2):
            super().__init__()
            self.num_tokens = nn.ModuleList([nn.Linear(1, d) for _ in range(n_num)])
            self.stage_emb = nn.Embedding(n_stage, d)
            self.cls_token = nn.Parameter(torch.randn(1, 1, d))
            encoder_layer = nn.TransformerEncoderLayer(
                d, heads, d * 2, dropout=0.1, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, layers)
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
            self._cfg = dict(n_num=n_num, n_stage=n_stage, d=d, heads=heads, layers=layers)

        def forward(self, xnum: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
            tokens = [self.num_tokens[i](xnum[:, i:i + 1]) for i in range(xnum.shape[1])]
            tokens.append(self.stage_emb(xs))
            t = torch.stack(tokens, dim=1)
            b = t.shape[0]
            t = torch.cat([self.cls_token.expand(b, -1, -1), t], dim=1)
            return self.head(self.transformer(t)[:, 0]).squeeze(-1)

    @register_model("ftt")
    @register_model("ft-transformer")
    class FTTransformerModel(BaseEnergyModel):
        """FT-Transformer: per-stage prediction from numeric features + stage embedding."""

        def __init__(self, d: int = 32, heads: int = 4, layers: int = 2,
                     epochs: int = 400, lr: float = 0.01, **kwargs):
            super().__init__(**kwargs)
            self._d = d
            self._heads = heads
            self._layers = layers
            self._epochs = epochs
            self._lr = lr
            self._model: Optional[FeatureTokenizerTransformer] = None
            self._mu: Optional[torch.Tensor] = None
            self._sd: Optional[torch.Tensor] = None
            self._stage_map: Dict[str, int] = {}
            self._log_scale = True

        def fit(self, df: pd.DataFrame, target: str = "energy_j"):
            stages_in_data = [s for s in PIPELINE_STAGES if s in df.stage.unique()]
            self._stage_map = {s: i for i, s in enumerate(stages_in_data)}

            Xn = torch.tensor(
                df[NUM_FEATURES].values, dtype=torch.float32
            )
            Xs = torch.tensor(
                df.stage.map(self._stage_map).fillna(0).values
            )
            y = torch.tensor(
                np.log10(df[target].clip(lower=1e-6).values), dtype=torch.float32
            )

            self._mu = Xn.mean(0)
            self._sd = Xn.std(0) + 1e-6
            Xn = (Xn - self._mu) / self._sd

            self._model = FeatureTokenizerTransformer(
                len(NUM_FEATURES), len(stages_in_data),
                d=self._d, heads=self._heads, layers=self._layers,
            )

            opt = torch.optim.Adam(
                self._model.parameters(), self._lr, weight_decay=1e-4
            )
            loss_fn = nn.SmoothL1Loss()

            for _ in range(self._epochs):
                opt.zero_grad()
                loss = loss_fn(self._model(Xn, Xs), y)
                loss.backward()
                opt.step()

            self._model.eval()
            self._metadata["epochs"] = self._epochs
            self._metadata["stages"] = stages_in_data
            return self

        def predict(self, n: int, corpus: str, n_tasks: int = 1,
                    stages: Optional[List[str]] = None) -> Dict[str, float]:
            if self._model is None:
                raise RuntimeError("Model not trained. Call fit() or load() first.")

            if stages is None:
                stages = self._metadata.get("stages", PIPELINE_STAGES)

            chars = corpus_chars(corpus)
            rows = []
            for stage in stages:
                rows.append([
                    np.log10(max(n, 1)),
                    np.log10(max(chars, 1)),
                    np.log2(max(n_tasks, 1)),
                ])

            Xn = torch.tensor(rows, dtype=torch.float32)
            Xn = (Xn - self._mu) / self._sd
            Xs = torch.tensor([self._stage_map.get(s, 0) for s in stages])

            with torch.no_grad():
                preds_log = self._model(Xn, Xs).numpy()

            preds = 10 ** preds_log
            result = {stage: float(preds[i]) for i, stage in enumerate(stages)}
            result["total_j"] = sum(result.values())
            return result

        def _get_state(self) -> Dict:
            return {
                "model_state": self._model.state_dict() if self._model else None,
                "cfg": self._model._cfg if self._model else None,
                "mu": self._mu,
                "sd": self._sd,
                "stage_map": self._stage_map,
                "log_scale": self._log_scale,
            }

        def _set_state(self, state: Dict):
            cfg = state["cfg"]
            self._model = FeatureTokenizerTransformer(
                cfg["n_num"], cfg["n_stage"],
                d=cfg["d"], heads=cfg["heads"], layers=cfg["layers"],
            )
            self._model.load_state_dict(state["model_state"])
            self._model.eval()
            self._mu = state["mu"]
            self._sd = state["sd"]
            self._stage_map = state.get("stage_map", {})
            self._log_scale = state.get("log_scale", True)


    @register_model("kalman-ftt")
    class KalmanFTTModel(BaseEnergyModel):
        """FT-Transformer trained on Kalman filter trajectories for cold whole-run prediction.

        Predicts TOTAL energy from (n, chars) for an unseen corpus — no per-stage breakdown.
        """

        def __init__(self, d: int = 24, heads: int = 4, layers: int = 2,
                     epochs: int = 400, lr: float = 0.008, **kwargs):
            super().__init__(**kwargs)
            self._d = d
            self._heads = heads
            self._layers = layers
            self._epochs = epochs
            self._lr = lr
            self._model: Optional[KalmanFTTModel._TabFTT] = None
            self._mu: Optional[torch.Tensor] = None
            self._sd: Optional[torch.Tensor] = None
            self._rates: Dict[str, float] = {}
            self._stage_order: List[str] = []

        class _TabFTT(nn.Module):
            """Plain tabular FT-Transformer (no stage embedding — whole-run prediction)."""
            def __init__(self, n_num: int, d: int = 24, heads: int = 4, layers: int = 2):
                super().__init__()
                self.tok = nn.ModuleList([nn.Linear(1, d) for _ in range(n_num)])
                self.cls = nn.Parameter(torch.randn(1, 1, d))
                enc = nn.TransformerEncoderLayer(d, heads, d * 2, 0.1, batch_first=True)
                self.tr = nn.TransformerEncoder(enc, layers)
                self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
                self.cfg = dict(n_num=n_num, d=d, heads=heads, layers=layers)

            def forward(self, x):
                toks = [self.tok[i](x[:, i:i + 1]) for i in range(x.shape[1])]
                t = torch.stack(toks, 1)
                b = t.shape[0]
                t = torch.cat([self.cls.expand(b, -1, -1), t], 1)
                return self.head(self.tr(t)[:, 0]).squeeze(-1)

        def fit(self, df: pd.DataFrame, target: str = "energy_j"):
            CPU = ["data_splitting", "string_normalization", "heuristic_filtering", "deduplication"]
            d1 = df[(df.ntasks == 1) & df.stage.isin(CPU)].copy()

            # Per-stage energy per character rates (physics prior)
            rates = {}
            for s in CPU:
                sub = d1[d1.stage == s]
                if not sub.empty:
                    rates[s] = float(np.median(
                        sub[target] / (sub.chars * sub.n)
                    ))
            self._rates = rates
            self._stage_order = CPU

            # Build Kalman trajectory rows
            X, Y = [], []
            for slug, g in d1.groupby("slug"):
                sv = g.groupby("stage")[target].mean()
                if not all(s in sv.index for s in CPU):
                    continue
                vals = np.array([sv[s] for s in CPU])
                chars = g.chars.iloc[0]
                n = g.n.iloc[0]
                cum = np.concatenate([[0.0], np.cumsum(vals)])

                for j in range(len(CPU) + 1):
                    observed = cum[j]
                    remaining_prior = sum(
                        rates[s] * chars * n for s in CPU[j:]
                    )
                    kalman_partial = observed + remaining_prior
                    X.append([
                        np.log10(n),
                        np.log10(chars),
                        j / len(CPU),
                        np.log10(kalman_partial + 1),
                        np.log10(observed + 1),
                    ])
                    Y.append(np.log10(vals.sum()))

            X = np.array(X, np.float32)
            Y = np.array(Y, np.float32)

            self._mu = torch.tensor(X.mean(0))
            self._sd = torch.tensor(X.std(0) + 1e-6)

            xt = (torch.tensor(X) - self._mu) / self._sd
            yt = torch.tensor(Y)

            self._model = self._TabFTT(X.shape[1], self._d, self._heads, self._layers)
            opt = torch.optim.Adam(
                self._model.parameters(), self._lr, weight_decay=2e-3
            )
            loss_fn = nn.SmoothL1Loss()

            for _ in range(self._epochs):
                opt.zero_grad()
                loss = loss_fn(self._model(xt), yt)
                loss.backward()
                opt.step()

            self._model.eval()
            self._metadata["rates"] = self._rates
            return self

        def predict(self, n: int, corpus: str, n_tasks: int = 1,
                    stages: Optional[List[str]] = None) -> Dict[str, float]:
            if self._model is None:
                raise RuntimeError("Model not trained.")

            chars = corpus_chars(corpus)

            # Cold prediction: nothing observed yet (j=0)
            prior_total = sum(
                self._rates.get(s, 0) * chars * n
                for s in self._stage_order
            )
            x = np.array([[
                np.log10(n),
                np.log10(chars),
                0.0,
                np.log10(prior_total + 1),
                0.0,
            ]], np.float32)

            xt = (torch.tensor(x) - self._mu) / self._sd

            with torch.no_grad():
                pred_log = self._model(xt).item()

            total_j = float(10 ** pred_log)
            return {"total_j": total_j, "prior_estimate_j": prior_total}

        def _get_state(self) -> Dict:
            return {
                "model_state": self._model.state_dict() if self._model else None,
                "cfg": self._model.cfg if self._model else None,
                "mu": self._mu,
                "sd": self._sd,
                "rates": self._rates,
                "stage_order": self._stage_order,
            }

        def _set_state(self, state: Dict):
            cfg = state["cfg"]
            self._model = self._TabFTT(
                cfg["n_num"], d=cfg["d"], heads=cfg["heads"], layers=cfg["layers"]
            )
            self._model.load_state_dict(state["model_state"])
            self._model.eval()
            self._mu = state["mu"]
            self._sd = state["sd"]
            self._rates = state.get("rates", {})
            self._stage_order = state.get("stage_order", [])
