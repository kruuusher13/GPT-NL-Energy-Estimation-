#!/usr/bin/env python3
"""
Train g — the coefficient predictor (level 2 of the smart model).

g: per-stage context features -> the stage's energy/time coefficient c1.
Features are DERIVED FROM THE ENERGY DATA ITSELF (no corpus probe needed):
  chars_per_doc  ~ io_mbs * time / n_docs      (bytes read per doc, from data_splitting)
  survival       = n_out / n_in                (quality-filter keep rate = content signal)
  cpi, io_mbs, cpu_gflops, cpu_util            (per-stage compute character)

We fit each (stage, corpus) law to get the coefficient label, attach features, then
do LEAVE-ONE-CORPUS-OUT: train g on the other corpora, predict the held-out corpus's
coefficients, apply the physics, and score the held-out WHOLE-run prediction.
This is the only test that proves cold cross-corpus prediction.

Usage: python3 train_g.py [measurements.csv]
"""
import sys, re
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

CSV = sys.argv[1] if len(sys.argv)>1 else "paper/data/measurements_generalized.csv"
STAGES = ["data_splitting","string_normalization","heuristic_filtering","deduplication"]
KNOWN_CHARS = {"american_stories":1798,"cc_github_opencode":3151,
               "cc_german_pd":48881,"european_parliament":181075}

TAG2CORP = {"amd":"american_stories","amg":"american_stories","amst":"american_stories",
            "ghd":"cc_github_opencode","ghub":"cc_github_opencode",
            "gerd":"cc_german_pd","gerg":"cc_german_pd",
            "epd":"european_parliament","epg":"european_parliament","eu":"european_parliament",
            "aud":"auditdienstrijk","bel":"belgian_journal","eng":"cc_english_pd",
            "oal":"cc_openalex","dan":"dansknaw","dpc":"dpc","kbb":"kb_pd_books",
            "nat":"naturalis","rec":"rechtspraak","twk":"tweedekamer",
            "utr":"utrechtsarchief","woo":"woogle"}

def corp(s):
    s=s.lower(); tag=s.split("_")[0]
    if tag in TAG2CORP: return TAG2CORP[tag]
    if s.startswith(("amst","amg","amd","size","dedupval","demolive")): return "american_stories"
    if s.startswith(("euparl","epd","epg","eu")): return "european_parliament"
    if s.startswith(("ghd","ghub","gh")): return "cc_github_opencode"
    if s.startswith(("gerd","gerg","ger")): return "cc_german_pd"
    if s.startswith("mix"): return "MIX"          # mixed corpus — excluded from g table
    return tag
def nt(s):
    m=re.search(r"_n(\d+)_",s); return int(m.group(1)) if m else 1

def load():
    df=pd.read_csv(CSV)
    df["corpus"]=df.slug.map(corp); df["ntasks"]=df.slug.map(nt)
    df["n"]=pd.to_numeric(df["size"],errors="coerce")
    for c in ["time_sec","dc_node_power_w","io_mbs","cpi","cpu_gflops","cpu_util","n_in","n_out"]:
        df[c]=pd.to_numeric(df.get(c),errors="coerce")
    df["energy_j"]=df.dc_node_power_w.fillna(0)*df.time_sec
    return df[(df.ntasks==1)&(df.time_sec>0)&df.n.notna()&(df.energy_j>0)]

def chars_per_doc(df, c):
    if c in KNOWN_CHARS: return KNOWN_CHARS[c]
    sub=df[(df.corpus==c)&(df.stage=="data_splitting")]          # bytes read / docs
    if sub.empty: return np.nan
    bpd=(sub.io_mbs*sub.time_sec*1e6/sub.n).replace([np.inf,-np.inf],np.nan).dropna()
    return float(bpd.median()) if len(bpd) else np.nan

def fit_c1(sub, target):
    g=sub.groupby("n")[target].mean().reset_index().sort_values("n")
    if len(g)<2: return None
    A=np.vstack([g.n.values,np.ones(len(g))]).T
    (c1,c0),*_=np.linalg.lstsq(A,g[target].values,rcond=None)
    return c1,c0

def build_table(df, target):
    """one row per (stage, corpus): coefficient label + derived features."""
    rows=[]
    for c in df.corpus.unique():
        if c=="MIX": continue                     # mixed corpus is not a clean training row
        ch=chars_per_doc(df,c)
        if not np.isfinite(ch) or ch<=0: continue
        for st in STAGES:
            sub=df[(df.corpus==c)&(df.stage==st)]
            f=fit_c1(sub,target)
            if not f: continue
            c1,c0=f
            surv=(sub.n_out/sub.n_in).replace([np.inf,-np.inf],np.nan).mean()
            rows.append(dict(stage=st,corpus=c,c1=c1,c0=c0,chars=ch,
                             cpi=sub.cpi.mean(),io=sub.io_mbs.mean(),
                             gflops=sub.cpu_gflops.mean(),util=sub.cpu_util.mean(),
                             surv=surv if np.isfinite(surv) else 0.5,
                             N=sub.n.max()))
    return pd.DataFrame(rows)

FEATS=["lchars","lcpi","lio","lgflops","util","surv"]
def featmat(t):
    X=pd.DataFrame()
    X["lchars"]=np.log10(t.chars.clip(lower=1))
    X["lcpi"]=np.log10(t.cpi.clip(lower=1e-3))
    X["lio"]=np.log10(t.io.clip(lower=1e-3))
    X["lgflops"]=np.log10(t.gflops.clip(lower=1e-6))
    X["util"]=t.util.fillna(50)/100.0
    X["surv"]=t.surv.fillna(0.5)
    return X.fillna(0.0)

def mre(a,p): return abs(p-a)/a*100 if a else np.nan

def main():
    df=load()
    print(f"Loaded {len(df)} ntasks=1 readings | corpora={sorted(df.corpus.unique())} "
          f"({df.corpus.nunique()} total)\n")
    for target,unit in [("time_sec","s"),("energy_j","J")]:
        t=build_table(df,target)
        ncorp=t.corpus.nunique()
        print(f"################ g target={target}  ({len(t)} stage-coeffs, {ncorp} corpora) ################")
        if ncorp<3:
            print("  need >=3 corpora to train g — waiting for broad-corpus data.\n"); continue
        # per-corpus held out; per-stage g predicts c1 from features
        model = (HistGradientBoostingRegressor(max_iter=200,max_depth=2,learning_rate=0.08,
                                               min_samples_leaf=2) if ncorp>=8 else Ridge(alpha=1.0))
        baseline_err=[]; g_err=[]
        for held in t.corpus.unique():
            predN_g=trueN=predN_base=0.0; ok=True
            for st in STAGES:
                tr=t[(t.stage==st)&(t.corpus!=held)]; te=t[(t.stage==st)&(t.corpus==held)]
                if te.empty or len(tr)<2: ok=False; break
                # baseline: mean per-char k from other corpora
                kbase=(tr.c1/tr.chars).mean(); c1_base=kbase*te.chars.values[0]
                # g: learn c1 (log) from features
                Xtr=featmat(tr); Xte=featmat(te)
                m=model
                try:
                    m.fit(Xtr, np.log10(np.abs(tr.c1)+1e-12))
                    c1_g=10**m.predict(Xte)[0]
                except Exception:
                    c1_g=c1_base
                N=te.N.values[0]; c0=tr.c0.mean()
                predN_g+=c1_g*N+c0; predN_base+=c1_base*N+c0; trueN+=te.c1.values[0]*N+te.c0.values[0]
            if ok and trueN>0:
                g_err.append(mre(trueN,predN_g)); baseline_err.append(mre(trueN,predN_base))
        if g_err:
            print(f"  LEAVE-ONE-CORPUS-OUT whole-run MRE:")
            print(f"    per-char baseline : median {np.median(baseline_err):5.1f}%   mean {np.mean(baseline_err):5.1f}%")
            print(f"    g (learned)       : median {np.median(g_err):5.1f}%   mean {np.mean(g_err):5.1f}%   "
                  f"[{'g WINS' if np.median(g_err)<np.median(baseline_err) else 'baseline wins'}]")
        print()

if __name__=="__main__":
    main()
