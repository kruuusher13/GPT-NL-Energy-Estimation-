#!/usr/bin/env python3
"""
Generalized energy model — the ML ladder Julio asked for.

OLS (per-stage physics baseline)  ->  GBM (gradient-boosted trees)  ->  MLP (neural).
The point is NOT to beat OLS on one stage's curve (OLS is already R^2~0.999 there).
It is to predict energy ACROSS stage / size / n_tasks / hardware / corpus from one
model — i.e. generalize to configs we never calibrated, removing calibration runs.

Honest evaluation: grouped cross-validation.
  - LeaveOneSizeOut   : train on some doc-counts, predict an unseen doc-count (extrapolation)
  - LeaveOneCorpusOut : train on some corpora, predict an unseen corpus  (transfer)
We report held-out MRE per model. If OLS/GBM already wins, we SAY so — that is a result.

Usage:
  python3 ml_energy_model.py                      # uses paper/data/*.csv (tonight)
  python3 ml_energy_model.py FRESH_HARVEST.csv    # point at the morning harvest
"""
import sys, re, glob, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut

DATA = "/Users/hornet/Projects/TNO/paper/data"
DEFAULT_CSVS = [f"{DATA}/measurements.csv", f"{DATA}/measurements_shared_multicorpus.csv"]

STAGES = ["data_splitting", "string_normalization", "heuristic_filtering",
          "deduplication", "toxic_language_detection"]

# Measured average characters/document per corpus (probe_corpora.py, 2026-06-14).
# This is the transfer bridge: energy-per-CHARACTER is ~constant across corpora, so
# a model that sees `chars` can predict a corpus it never calibrated. Spans 100x.
CORPUS_CHARS = {
    "american_stories":    1798,
    "cc_github_opencode":  3151,
    "cc_german_pd":       48881,
    "european_parliament":181075,
}

# slug-prefix -> canonical corpus name (covers all tag generations: size*, amst*,
# amg*, euparl*, epg*, ghub*, gerg*).
def parse_corpus(slug: str) -> str:
    s = slug.lower()
    if s.startswith(("ghd", "ghub", "gh_")) or "github" in s:
        return "cc_github_opencode"
    if s.startswith(("gerd", "gerg", "ger", "german")) or "german" in s:
        return "cc_german_pd"
    if s.startswith(("epd", "epg", "eu")) or "euparl" in s or "parliament" in s:
        return "european_parliament"
    return "american_stories"   # amd/amg/amst/size*

def parse_ntasks(slug: str) -> int:
    m = re.search(r"_n(\d+)_", slug)
    return int(m.group(1)) if m else 1

# ── load + build the unified feature table ──────────────────────────────────────
def load(csvs):
    frames = []
    for c in csvs:
        for f in glob.glob(c):
            try:
                frames.append(pd.read_csv(f))
            except Exception as e:
                print(f"  [warn] skip {f}: {e}")
    if not frames:
        sys.exit("No measurement CSVs found.")
    df = pd.concat(frames, ignore_index=True)

    # energy: CPU stages = DC node power x time; GPU (toxic) = + GPU0 power x time
    for col in ["dc_node_power_w", "gpu0_power_w", "time_sec", "size"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df["gpu0_power_w"] = df["gpu0_power_w"].fillna(0.0)
    df["energy_j"] = (df["dc_node_power_w"].fillna(0.0) + df["gpu0_power_w"]) * df["time_sec"]

    df = df[df["stage"].isin(STAGES)].copy()
    df["corpus"] = df["slug"].map(parse_corpus)
    df["n_tasks"] = df["slug"].map(parse_ntasks)
    df["n"] = df["size"].astype(float)
    df["avg_chars"] = df["corpus"].map(CORPUS_CHARS).fillna(2000.0)
    df["chars"] = df["n"] * df["avg_chars"]   # total characters processed

    # one physical reading per (slug, stage): sum energy over array tasks / sub-jobs,
    # wall-time = max (array tasks run concurrently)
    g = (df.groupby(["slug", "stage", "corpus", "n_tasks", "n", "avg_chars", "chars",
                     "partition"], dropna=False)
           .agg(energy_j=("energy_j", "sum"), time_sec=("time_sec", "max"))
           .reset_index())
    g = g[(g["energy_j"] > 0) & np.isfinite(g["energy_j"])]
    return g

def mre(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    m = y_true > 0
    return float(np.mean(np.abs(y_pred[m] - y_true[m]) / y_true[m]) * 100)

# ── models ──────────────────────────────────────────────────────────────────────
def feat_matrix(df):
    """Features for the ML models: stage (one-hot), log n, n_tasks, hardware, corpus."""
    # NOTE: deliberately NOT one-hot the corpus — we want the model to generalize to
    # an UNSEEN corpus via chars/doc, not memorize a corpus dummy. Stage + hardware
    # are one-hot; size/parallelism/doc-size are continuous.
    X = pd.get_dummies(df[["stage", "partition"]].astype(str), dummy_na=False)
    X["log_n"] = np.log10(df["n"].clip(lower=1))
    X["log_chars"] = np.log10(df["chars"].clip(lower=1))      # the transfer bridge
    X["log_avg_chars"] = np.log10(df["avg_chars"].clip(lower=1))
    X["log_ntasks"] = np.log2(df["n_tasks"].clip(lower=1))
    return X.astype(float)

def ols_perstage_predict(train, test, target="energy_j"):
    """OLS baseline: per-stage y = c0 + c1 n, fit on train, predict test."""
    pred = np.full(len(test), np.nan)
    for st in train["stage"].unique():
        tr = train[train["stage"] == st]
        if len(tr) < 2:
            c1, c0 = 0.0, tr[target].mean() if len(tr) else 0.0
        else:
            A = np.vstack([tr["n"].values, np.ones(len(tr))]).T
            (c1, c0), *_ = np.linalg.lstsq(A, tr[target].values, rcond=None)
        mask = (test["stage"] == st).values
        pred[mask] = c1 * test["n"].values[mask] + c0
    return pred

def run_cv(df, group_col, label, target="energy_j"):
    logo = LeaveOneGroupOut()
    groups = df[group_col].astype(str).values
    if len(np.unique(groups)) < 2:
        print(f"  [{label}] only one {group_col} group present — skipping (need fresh data)")
        return
    rows = []
    Xall = feat_matrix(df); yall = df[target].values
    Xall = Xall[yall > 0]; groups = groups[yall > 0]; dfv = df[yall > 0]; yall = yall[yall > 0]
    for name, model in [
        ("GBM", HistGradientBoostingRegressor(max_iter=300, max_depth=3,
                                              learning_rate=0.06, min_samples_leaf=3)),
        ("MLP", make_pipeline(StandardScaler(),
                              MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-2,
                                           max_iter=4000, random_state=0))),
    ]:
        yt, yp = [], []
        for tr, te in logo.split(Xall, yall, groups):
            model.fit(Xall.iloc[tr], np.log10(yall[tr]))
            yp.extend(10 ** model.predict(Xall.iloc[te])); yt.extend(yall[te])
        rows.append((name, mre(yt, yp)))
    # OLS per-stage baseline under same splits (filtered frame + same target)
    yt, yp = [], []
    for tr, te in logo.split(Xall, yall, groups):
        p = ols_perstage_predict(dfv.iloc[tr], dfv.iloc[te], target)
        ok = np.isfinite(p)
        yp.extend(p[ok]); yt.extend(yall[te][ok])
    rows.insert(0, ("OLS (per-stage physics)", mre(yt, yp)))

    print(f"\n  Held-out MRE  —  {label}  (leave-one-{group_col}-out)")
    print("  " + "-" * 46)
    for name, v in rows:
        print(f"    {name:<28} {v:7.1f}%")
    best = min(rows, key=lambda r: r[1])
    print(f"  -> best: {best[0]} ({best[1]:.1f}%)")

def main():
    csvs = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_CSVS
    df = load(csvs)
    print(f"Loaded {len(df)} stage-readings | stages={sorted(df.stage.unique())}")
    print(f"  corpora={sorted(df.corpus.unique())}  n_tasks={sorted(df.n_tasks.unique())}"
          f"  sizes={sorted(df.n.unique())}")
    for target, tname in [("time_sec", "WALL-TIME (clean, co-tenancy-invariant)"),
                          ("energy_j", "ENERGY (raw node power — contaminated at n_tasks>1)")]:
        print(f"\n################ TARGET: {tname} ################")
        run_cv(df, "n", "extrapolate to an unseen DOC-COUNT", target)
        run_cv(df, "corpus", "transfer to an unseen CORPUS", target)
    print("\nNote: wall-time is the honest target (energy = P_eff x time). Raw-power energy "
          "is inflated at n_tasks>1 (harvest sums whole-node power across array tasks). "
          "Model train-cost is seconds on CPU; a GPU transformer would cost more energy "
          "than the calibration it replaces — keep OLS as physics, ML as the transfer layer.")

if __name__ == "__main__":
    main()
