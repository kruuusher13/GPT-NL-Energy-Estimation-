#!/usr/bin/env python3
"""Train EVERY model on the full data and SAVE the artifacts to models/.
Produces real, reloadable trained networks (not just eval runs):
  - per-stage energy/time predictors: Linear, Ridge, GBM, MLP, FT-Transformer
  - the Kalman-trained FT-Transformer (cold whole-run total)
Writes models/*.pt (torch), models/*.joblib (sklearn), a manifest (models/MODELS.md +
models_manifest.json) with each model's held-out accuracy pulled from the benchmark JSONs.

Classes are importable without side effects (training runs only on `python train_and_save.py`)."""
import os, re, json, time
import numpy as np, pandas as pd, torch, torch.nn as nn, joblib
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

CSV   = "paper/data/measurements_generalized.csv"
CHARS = {"american_news": 1798, "github_code": 3151, "german": 48881, "euro_parliament": 181075}
STAGES = ["data_splitting", "string_normalization", "heuristic_filtering",
          "deduplication", "toxic_language_detection"]
CPU = ["data_splitting", "string_normalization", "heuristic_filtering", "deduplication"]
NUM = ["ln", "lchars", "lnt"]; FEAT = NUM + ["sidx"]

def corp(s):
    s = str(s).lower()
    if s.startswith(("ghd", "ghub")) or "github" in s: return "github_code"
    if s.startswith(("gerd", "gerg", "ger")):          return "german"
    if s.startswith(("epd", "epg", "eu")) or "parl" in s: return "european_parliament"
    return "american_news"
def nt(s):
    m = re.search(r"_n(\d+)_", str(s)); return int(m.group(1)) if m else 1
def n_params(m): return sum(p.numel() for p in m.parameters())

class FTT(nn.Module):
    """FT-Transformer: each numeric feature -> token, + stage embedding + CLS -> energy/time."""
    def __init__(self, n_num, n_stage, d=32, heads=4, layers=2):
        super().__init__()
        self.num = nn.ModuleList([nn.Linear(1, d) for _ in range(n_num)])
        self.stage = nn.Embedding(n_stage, d); self.cls = nn.Parameter(torch.randn(1, 1, d))
        enc = nn.TransformerEncoderLayer(d, heads, d * 2, 0.1, batch_first=True)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self.cfg = dict(n_num=n_num, n_stage=n_stage, d=d, heads=heads, layers=layers)
    def forward(self, xnum, xs):
        toks = [self.num[i](xnum[:, i:i + 1]) for i in range(xnum.shape[1])]
        toks.append(self.stage(xs)); t = torch.stack(toks, 1); b = t.shape[0]
        t = torch.cat([self.cls.expand(b, -1, -1), t], 1)
        return self.head(self.tr(t)[:, 0]).squeeze(-1)

class FTTtab(nn.Module):
    """FT-Transformer over plain numeric features (Kalman-trained whole-run total)."""
    def __init__(self, n_num, d=24, heads=4, layers=2):
        super().__init__()
        self.tok = nn.ModuleList([nn.Linear(1, d) for _ in range(n_num)])
        self.cls = nn.Parameter(torch.randn(1, 1, d))
        enc = nn.TransformerEncoderLayer(d, heads, d * 2, 0.1, batch_first=True)
        self.tr = nn.TransformerEncoder(enc, layers); self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self.cfg = dict(n_num=n_num, d=d, heads=heads, layers=layers)
    def forward(self, x):
        toks = [self.tok[i](x[:, i:i + 1]) for i in range(x.shape[1])]
        t = torch.stack(toks, 1); b = t.shape[0]; t = torch.cat([self.cls.expand(b, -1, -1), t], 1)
        return self.head(self.tr(t)[:, 0]).squeeze(-1)

def load_df():
    df = pd.read_csv(CSV)
    df["corpus"] = df.slug.map(corp); df["ntasks"] = df.slug.map(nt)
    df["n"] = pd.to_numeric(df["size"], errors="coerce")
    df["energy_j"] = df.dc_node_power_w.fillna(0) * df.time_sec
    df = df[df.stage.isin(STAGES) & (df.time_sec > 0) & (df.energy_j > 0) & df.n.notna()].copy()
    df["chars"] = df.corpus.map(CHARS).fillna(2000)
    df["sidx"]  = df.stage.map({s: i for i, s in enumerate(STAGES)})
    df["ln"] = np.log10(df.n); df["lchars"] = np.log10(df.chars); df["lnt"] = np.log2(df.ntasks.clip(lower=1))
    return df

def main():
    torch.manual_seed(0); np.random.seed(0)
    os.makedirs("models", exist_ok=True)
    df = load_df()
    bench = json.load(open("benchmark_all_results.json")) if os.path.exists("benchmark_all_results.json") else {}
    kal   = json.load(open("kalman_transformer_results.json")) if os.path.exists("kalman_transformer_results.json") else {}
    def bm(tname, model):
        try:    return {a: round(bench[tname][a][model], 1) for a in bench[tname]}
        except Exception: return None

    manifest = []
    def record(name, fname, kind, target, params, acc):
        manifest.append(dict(model=name, file=fname, type=kind, target=target, params=params, accuracy=acc))

    SK = {
        "linear": (lambda: make_pipeline(StandardScaler(), LinearRegression()), "Linear (OLS)"),
        "ridge":  (lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)), "Ridge"),
        "gbm":    (lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.06, min_samples_leaf=3), "GBM"),
        "mlp":    (lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, alpha=1e-3, random_state=0)), "MLP"),
    }
    print("Training + saving per-stage models ...")
    for target, tname in [("time_sec", "time"), ("energy_j", "energy")]:
        y = np.log10(df[target].values)
        for key, (make, bench_name) in SK.items():
            m = make().fit(df[FEAT].values, y)
            f = f"models/{key}_{tname}.joblib"; joblib.dump({"model": m, "feat": FEAT, "log10": True}, f)
            record(key.upper(), f, "sklearn", tname, "-", bm(tname, bench_name)); print(f"  saved {f}")
        Xn = torch.tensor(df[NUM].values, dtype=torch.float32); Xs = torch.tensor(df.sidx.values)
        mu, sd = Xn.mean(0), Xn.std(0) + 1e-6; Xnn = (Xn - mu) / sd; yt = torch.tensor(y, dtype=torch.float32)
        m = FTT(len(NUM), len(STAGES)); opt = torch.optim.Adam(m.parameters(), 1e-2, weight_decay=1e-4); lf = nn.SmoothL1Loss()
        for _ in range(400):
            opt.zero_grad(); lf(m(Xnn, Xs), yt).backward(); opt.step()
        f = f"models/ftt_{tname}.pt"
        torch.save({"state_dict": m.state_dict(), "cfg": m.cfg, "mu": mu, "sd": sd,
                    "num_feats": NUM, "stages": STAGES, "chars": CHARS, "log10": True}, f)
        record("FT-Transformer", f, "torch", tname, n_params(m), bm(tname, "FT-Transformer"))
        print(f"  saved {f}  ({n_params(m):,} params)")

    print("Training + saving the Kalman-trained transformer ...")
    d1 = df[(df.ntasks == 1) & df.stage.isin(CPU)].copy()
    rates = {s: float(np.median(d1[d1.stage == s].energy_j / (d1[d1.stage == s].chars * d1[d1.stage == s].n))) for s in CPU}
    X, Y = [], []
    for slug, g in d1.groupby("slug"):
        sv = g.groupby("stage").energy_j.mean()
        if not all(s in sv.index for s in CPU): continue
        vals = np.array([sv[s] for s in CPU]); chars = g.chars.iloc[0]; n = g.n.iloc[0]; cum = np.concatenate([[0], np.cumsum(vals)])
        for j in range(len(CPU) + 1):
            kp = cum[j] + sum(rates[s] * chars * n for s in CPU[j:])
            X.append([np.log10(n), np.log10(chars), j / len(CPU), np.log10(kp + 1), np.log10(cum[j] + 1)]); Y.append(np.log10(vals.sum()))
    X, Y = np.array(X, np.float32), np.array(Y, np.float32); mu2, sd2 = X.mean(0), X.std(0) + 1e-6
    mk = FTTtab(X.shape[1]); opt = torch.optim.Adam(mk.parameters(), 8e-3, weight_decay=2e-3); lf = nn.SmoothL1Loss()
    xt = torch.tensor((X - mu2) / sd2); yt = torch.tensor(Y)
    for _ in range(400):
        opt.zero_grad(); lf(mk(xt), yt).backward(); opt.step()
    f = "models/kalman_transformer_energy.pt"
    torch.save({"state_dict": mk.state_dict(), "cfg": mk.cfg, "mu": mu2, "sd": sd2, "rates": rates,
                "feats": ["log_n", "log_chars", "frac_observed", "log_kalman_partial", "log_observed"],
                "stages": CPU, "chars": CHARS, "log10": True}, f)
    km = kal.get("energy", {})
    acc = {"cold_baseline%": round(km["baseline"], 1), "cold_kalman%": round(km["kalman"], 1)} if km else None
    record("Kalman-trained FT-Transformer", f, "torch", "energy(total,cold)", n_params(mk), acc)
    print(f"  saved {f}  ({n_params(mk):,} params)")

    json.dump(manifest, open("models_manifest.json", "w"), indent=2)
    lines = ["# Trained models (saved artifacts)", "",
             f"_Generated {time.strftime('%Y-%m-%d %H:%M')} by `train_and_save.py` from `{CSV}`._", "",
             "| model | file | type | target | params | held-out accuracy (MRE %) |",
             "|---|---|---|---|---|---|"]
    for m in manifest:
        a = "-" if not m["accuracy"] else ", ".join(f"{k}={v}" for k, v in m["accuracy"].items())
        lines.append(f"| {m['model']} | `{m['file']}` | {m['type']} | {m['target']} | {m['params']} | {a} |")
    open("models/MODELS.md", "w").write("\n".join(lines) + "\n")
    print("\n" + "=" * 70)
    print(f"SAVED {len(manifest)} trained models to ./models/  (see models/MODELS.md)")

if __name__ == "__main__":
    main()
