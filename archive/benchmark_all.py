#!/usr/bin/env python3
"""One honest leaderboard: every model evaluated under IDENTICAL held-out splits.
Models: Linear(OLS), Ridge, GBM (HistGradientBoosting), MLP, FT-Transformer.
Axes:   size-extrapolation (leave-one-DOC-COUNT-out) and corpus-transfer (leave-one-CORPUS-out).
Targets: wall-time and energy. Metric: MRE %.  -> benchmark_all_results.json + a printed table."""
import sys, re, json
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut
np.random.seed(0); torch.manual_seed(0)

CSV   = sys.argv[1] if len(sys.argv) > 1 else "paper/data/measurements_generalized.csv"
CHARS = {"american_news": 1798, "github_code": 3151, "german": 48881, "euro_parliament": 181075}
STAGES = ["data_splitting", "string_normalization", "heuristic_filtering",
          "deduplication", "toxic_language_detection"]
def corp(s):
    s = str(s).lower()
    if s.startswith(("ghd", "ghub")) or "github" in s: return "github_code"
    if s.startswith(("gerd", "gerg", "ger")):          return "german"
    if s.startswith(("epd", "epg", "eu")) or "parl" in s: return "euro_parliament"
    return "american_news"
def nt(s):
    m = re.search(r"_n(\d+)_", str(s)); return int(m.group(1)) if m else 1

df = pd.read_csv(CSV)
df["corpus"] = df.slug.map(corp); df["ntasks"] = df.slug.map(nt)
df["n"] = pd.to_numeric(df["size"], errors="coerce")
df["energy_j"] = df.dc_node_power_w.fillna(0) * df.time_sec
df = df[df.stage.isin(STAGES) & (df.time_sec > 0) & (df.energy_j > 0) & df.n.notna()].copy()
df["chars"] = df.corpus.map(CHARS).fillna(2000)
df["sidx"]  = df.stage.map({s: i for i, s in enumerate(STAGES)})
df["ln"] = np.log10(df.n); df["lchars"] = np.log10(df.chars); df["lnt"] = np.log2(df.ntasks.clip(lower=1))
NUM = ["ln", "lchars", "lnt"]; FEAT = NUM + ["sidx"]

class FTT(nn.Module):
    def __init__(self, n_num, n_stage, d=32, heads=4, layers=2):
        super().__init__()
        self.num = nn.ModuleList([nn.Linear(1, d) for _ in range(n_num)])
        self.stage = nn.Embedding(n_stage, d); self.cls = nn.Parameter(torch.randn(1, 1, d))
        enc = nn.TransformerEncoderLayer(d, heads, d * 2, 0.1, batch_first=True)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
    def forward(self, xnum, xs):
        toks = [self.num[i](xnum[:, i:i + 1]) for i in range(xnum.shape[1])]
        toks.append(self.stage(xs)); t = torch.stack(toks, 1); b = t.shape[0]
        t = torch.cat([self.cls.expand(b, -1, -1), t], 1)
        return self.head(self.tr(t)[:, 0]).squeeze(-1)

def fit_ftt(tr, te, target, seeds=(0,), epochs=160):
    Xn = torch.tensor(tr[NUM].values, dtype=torch.float32); Xs = torch.tensor(tr.sidx.values)
    y = torch.tensor(np.log10(tr[target].values), dtype=torch.float32)
    mu, sd = Xn.mean(0), Xn.std(0) + 1e-6; Xn = (Xn - mu) / sd
    Xnt = (torch.tensor(te[NUM].values, dtype=torch.float32) - mu) / sd; Xst = torch.tensor(te.sidx.values)
    preds = []
    for s in seeds:
        torch.manual_seed(s)
        m = FTT(len(NUM), len(STAGES)); opt = torch.optim.Adam(m.parameters(), 1e-2, weight_decay=1e-4)
        lf = nn.SmoothL1Loss()
        for _ in range(epochs):
            opt.zero_grad(); lf(m(Xn, Xs), y).backward(); opt.step()
        m.eval()
        with torch.no_grad(): preds.append(m(Xnt, Xst).numpy())
    return 10 ** np.mean(preds, axis=0)

def sk(model):
    return make_pipeline(StandardScaler(), model)

MODELS = {
    "Linear (OLS)":   lambda: sk(LinearRegression()),
    "Ridge":          lambda: sk(Ridge(alpha=1.0)),
    "GBM":            lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=3,
                                  learning_rate=0.06, min_samples_leaf=3),
    "MLP":            lambda: sk(MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000,
                                  alpha=1e-3, random_state=0)),
}
def mre(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float); k = a > 0
    return float(np.mean(np.abs(p[k] - a[k]) / a[k]) * 100)

AXES = {"size-extrapolation": "n", "corpus-transfer": "corpus"}
results = {}
for target, tname in [("time_sec", "time"), ("energy_j", "energy")]:
    results[tname] = {}
    for axis, gcol in AXES.items():
        groups = df[gcol].astype(str).values; logo = LeaveOneGroupOut()
        preds = {name: ([], []) for name in list(MODELS) + ["FT-Transformer"]}
        for tri, tei in logo.split(df, df[target], groups):
            tr, te = df.iloc[tri], df.iloc[tei]
            for name, make in MODELS.items():
                m = make(); m.fit(tr[FEAT], np.log10(tr[target]))
                p = 10 ** m.predict(te[FEAT])
                preds[name][0].extend(te[target].values); preds[name][1].extend(p)
            pf = fit_ftt(tr, te, target)
            preds["FT-Transformer"][0].extend(te[target].values); preds["FT-Transformer"][1].extend(pf)
        results[tname][axis] = {name: mre(a, p) for name, (a, p) in preds.items()}

with open("benchmark_all_results.json", "w") as f:
    json.dump(results, f, indent=2)

cols = [(t, a) for t in ["time", "energy"] for a in AXES]
names = list(MODELS) + ["FT-Transformer"]
print("\nLEADERBOARD — MRE %, lower is better (identical leave-one-group-out splits)\n")
hdr = f"{'model':<16}" + "".join(f"{t[:4]+'/'+a.split('-')[0]:>18}" for t, a in cols)
print(hdr); print("-" * len(hdr))
for name in names:
    row = f"{name:<16}" + "".join(f"{results[t][a][name]:>17.1f}%" for t, a in cols)
    print(row)
print("\nwrote benchmark_all_results.json")
