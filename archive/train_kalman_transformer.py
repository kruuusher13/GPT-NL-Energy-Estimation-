#!/usr/bin/env python3
"""Yesterday's idea, measured honestly: STRENGTHEN COLD (cross-corpus) PREDICTION by
training the tabular transformer on the KALMAN FILTER's streaming trajectory data.

Why this could work
-------------------
The plain transformer overfits cross-corpus because we have only 4 corpora -> a handful
of (corpus -> total) rows. But the Kalman filter that drives the live monitor produces a
whole TRAJECTORY per run: after each stage reports, its posterior mean of the run's TOTAL
is a fresh, valid (context -> total) training example. Replaying the filter over every run
turns each run into (#stages+1) supervised rows. Those extra "warm" states anchor the
feature->total mapping, regularising the transformer so its COLD (no-data) prediction for an
unseen corpus improves.

We compare, leave-one-CORPUS-out, both evaluated at the COLD state (nothing observed yet):
  BASELINE   : transformer trained only on the cold row of each training run.
  KALMAN-AUG : transformer trained on the full Kalman trajectory (cold + warm states).

No leakage: the prior used to build trajectories is fit on TRAINING corpora only; the test
corpus contributes only its (chars, n) at cold inference. Honest MRE either way.
"""
import sys, re, json
import numpy as np, pandas as pd, torch, torch.nn as nn
torch.manual_seed(0); np.random.seed(0)

CSV   = sys.argv[1] if len(sys.argv) > 1 else "paper/data/measurements_generalized.csv"
CHARS = {"american_news": 1798, "github_code": 3151, "german": 48881, "euro_parliament": 181075}
STAGES = ["data_splitting", "string_normalization", "heuristic_filtering", "deduplication"]

def corpus_of(s):
    s = str(s).lower()
    if s.startswith(("ghd", "ghub")) or "github" in s: return "github_code"
    if s.startswith(("gerd", "gerg", "ger")):          return "german"
    if s.startswith(("epd", "epg", "eu")) or "parl" in s: return "euro_parliament"
    return "american_news"
def ntasks_of(s):
    m = re.search(r"_n(\d+)_", str(s)); return int(m.group(1)) if m else 1

df = pd.read_csv(CSV)
df["corpus"] = df.slug.map(corpus_of); df["ntasks"] = df.slug.map(ntasks_of)
df["n"] = pd.to_numeric(df["size"], errors="coerce")

def build_runs(target):
    """One row per (slug) run that has all 4 CPU stages: ordered per-stage values + meta."""
    d = df[(df.ntasks == 1) & df.stage.isin(STAGES) & (df.time_sec > 0) & df.n.notna()].copy()
    d["val"] = d.time_sec if target == "time" else d.dc_node_power_w.fillna(0) * d.time_sec
    d = d[d.val > 0]
    runs = []
    for slug, g in d.groupby("slug"):
        sv = g.groupby("stage").val.mean()
        if not all(s in sv.index for s in STAGES): continue
        vals = np.array([sv[s] for s in STAGES])
        runs.append(dict(corpus=corpus_of(slug), n=float(g.n.iloc[0]),
                         chars=CHARS.get(corpus_of(slug), 2000), vals=vals, total=vals.sum()))
    return runs

def stage_rates(runs):
    """Per-stage energy/time PER CHARACTER, median over the given runs (the physics prior)."""
    acc = {s: [] for s in STAGES}
    for r in runs:
        for i, s in enumerate(STAGES):
            acc[s].append(r["vals"][i] / (r["chars"] * r["n"]))
    return {s: float(np.median(v)) for s, v in acc.items()}

def prior_total(rates, chars, n):
    return sum(rates[s] * chars * n for s in STAGES)

def make_rows(runs, rates, augment):
    """Each run -> Kalman trajectory rows. Cold row j=0 always; warm rows j=1..4 if augment.
    Features: [log10 n, log10 chars, frac_observed, log10 kalman_partial, log10(observed+1)]."""
    X, Y = [], []
    for r in runs:
        S = len(STAGES); cum = np.concatenate([[0.0], np.cumsum(r["vals"])])
        js = range(S + 1) if augment else [0]
        for j in js:
            observed = cum[j]
            remaining_prior = sum(rates[s] * r["chars"] * r["n"] for s in STAGES[j:])
            kalman_partial = observed + remaining_prior          # Kalman posterior mean of TOTAL
            X.append([np.log10(r["n"]), np.log10(r["chars"]), j / S,
                      np.log10(kalman_partial + 1), np.log10(observed + 1)])
            Y.append(np.log10(r["total"]))
    return np.array(X, np.float32), np.array(Y, np.float32)

class FTT(nn.Module):
    """FT-Transformer over the numeric features (each feature -> token, + CLS)."""
    def __init__(self, n_num, d=24, heads=4, layers=2):
        super().__init__()
        self.tok = nn.ModuleList([nn.Linear(1, d) for _ in range(n_num)])
        self.cls = nn.Parameter(torch.randn(1, 1, d))
        enc = nn.TransformerEncoderLayer(d, heads, d * 2, 0.1, batch_first=True)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
    def forward(self, x):
        toks = [self.tok[i](x[:, i:i + 1]) for i in range(x.shape[1])]
        t = torch.stack(toks, 1); b = t.shape[0]
        t = torch.cat([self.cls.expand(b, -1, -1), t], 1)
        return self.head(self.tr(t)[:, 0]).squeeze(-1)

def train_eval(Xtr, Ytr, Xte, epochs=400, seeds=(0, 1, 2, 3, 4)):
    """Ensemble over a few seeds (same treatment for both methods) to denoise the estimate."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    xt = torch.tensor((Xtr - mu) / sd); yt = torch.tensor(Ytr); xe = torch.tensor((Xte - mu) / sd)
    preds = []
    for s in seeds:
        torch.manual_seed(s)
        m = FTT(Xtr.shape[1]); opt = torch.optim.Adam(m.parameters(), 8e-3, weight_decay=2e-3)
        lossf = nn.SmoothL1Loss()
        for _ in range(epochs):
            opt.zero_grad(); loss = lossf(m(xt), yt); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            preds.append(m(xe).numpy())
    return 10 ** np.mean(preds, axis=0)

def mre(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float); k = a > 0
    return float(np.mean(np.abs(p[k] - a[k]) / a[k]) * 100)

results = {}
for target in ["time", "energy"]:
    runs = build_runs(target)
    corpora = sorted({r["corpus"] for r in runs})
    base_p, base_a, kal_p, kal_a, per = [], [], [], [], {}
    for held in corpora:                                   # leave-one-CORPUS-out (cold transfer)
        tr = [r for r in runs if r["corpus"] != held]
        te = [r for r in runs if r["corpus"] == held]
        if not te or not tr: continue
        rates = stage_rates(tr)                            # prior from TRAIN corpora only
        # COLD test rows (nothing observed) for the held-out corpus
        Xte = np.array([[np.log10(r["n"]), np.log10(r["chars"]), 0.0,
                         np.log10(prior_total(rates, r["chars"], r["n"]) + 1), 0.0]
                        for r in te], np.float32)
        yte = [r["total"] for r in te]
        Xb, Yb = make_rows(tr, rates, augment=False)       # baseline: cold rows only
        Xk, Yk = make_rows(tr, rates, augment=True)        # kalman-augmented: full trajectories
        pb = train_eval(Xb, Yb, Xte); pk = train_eval(Xk, Yk, Xte)
        base_p += list(pb); base_a += yte; kal_p += list(pk); kal_a += yte
        per[held] = dict(baseline=mre(yte, pb), kalman=mre(yte, pk),
                         rows_baseline=len(Yb), rows_kalman=len(Yk))
    results[target] = dict(baseline=mre(base_a, base_p), kalman=mre(kal_a, kal_p), per_corpus=per)

with open("kalman_transformer_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("=" * 64)
print("COLD cross-corpus prediction (leave-one-corpus-out), MRE %")
print("=" * 64)
for target in ["time", "energy"]:
    r = results[target]
    tag = "WALL-TIME" if target == "time" else "ENERGY"
    drop = (r["baseline"] - r["kalman"]) / r["baseline"] * 100
    print(f"\n{tag}:  plain transformer {r['baseline']:6.1f}%   ->   "
          f"Kalman-trained {r['kalman']:6.1f}%   ({drop:+.0f}% change)")
    for c, v in r["per_corpus"].items():
        print(f"    {c:<18} {v['baseline']:7.1f}% -> {v['kalman']:7.1f}% "
              f"(rows {v['rows_baseline']}->{v['rows_kalman']})")
print("\nwrote kalman_transformer_results.json")
