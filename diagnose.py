#!/usr/bin/env python3
"""Diagnose WHERE the prediction error comes from, so we fix the right thing.
Tests the REAL use case (calibrate on small sizes -> predict the full run of the
SAME corpus = interpolation), per stage, per corpus, for time AND energy, at
ntasks=1 (the clean regime). Also checks the per-character transfer law."""
import sys, re
import numpy as np, pandas as pd

CSV = sys.argv[1] if len(sys.argv) > 1 else "paper/data/measurements_generalized.csv"
CORPUS_CHARS = {"american_stories":1798,"cc_github_opencode":3151,
                "cc_german_pd":48881,"european_parliament":181075}

def corp(s):
    s=s.lower()
    if s.startswith(("ghd","ghub")) or "github" in s: return "cc_github_opencode"
    if s.startswith(("gerd","ger")) or "german" in s: return "cc_german_pd"
    if s.startswith(("epd","epg","eu")) or "parl" in s: return "european_parliament"
    if s.startswith("mix"): return "MIXED"
    return "american_stories"   # amd/amg/amst/size*
def nt(s):
    m=re.search(r"_n(\d+)_",s); return int(m.group(1)) if m else 1

df = pd.read_csv(CSV)
df["corpus"]=df.slug.map(corp); df["ntasks"]=df.slug.map(nt)
df["n"]=pd.to_numeric(df["size"],errors="coerce")
df["energy_j"]=df.dc_node_power_w.fillna(0)*df.time_sec
df = df[(df.energy_j>0)&(df.time_sec>0)&df.n.notna()]

def mre(a,p):
    a,p=np.asarray(a,float),np.asarray(p,float); m=a>0
    return np.mean(np.abs(p[m]-a[m])/a[m])*100 if m.any() else np.nan

def calib_predict(sub, target):
    """fit y=c0+c1 n on all-but-largest size, predict largest. returns (mre, n_pred)."""
    g=sub.groupby("n")[target].mean().reset_index().sort_values("n")
    if len(g)<3: return None,0
    tr,te=g.iloc[:-1],g.iloc[-1:]
    A=np.vstack([tr.n,np.ones(len(tr))]).T
    (c1,c0),*_=np.linalg.lstsq(A,tr[target].values,rcond=None)
    pred=c1*te.n.values+c0
    return mre(te[target].values,pred), int(te.n.values[0])

print(f"Loaded {len(df)} readings | ntasks={sorted(df.ntasks.unique())}\n")

for target in ["time_sec","energy_j"]:
    print(f"################ CALIBRATE->PREDICT (same corpus, ntasks=1) — target={target} ################")
    d1=df[df.ntasks==1]
    hdr=f"  {'stage':<24}{'corpus':<20}{'held-out n':>11}{'MRE':>9}"
    print(hdr); print("  "+"-"*64)
    errs=[]
    for st in ["data_splitting","string_normalization","heuristic_filtering","deduplication","toxic_language_detection"]:
        for c in ["american_stories","cc_github_opencode","cc_german_pd","european_parliament"]:
            sub=d1[(d1.stage==st)&(d1.corpus==c)]
            m,npred=calib_predict(sub,target)
            if m is not None:
                print(f"  {st:<24}{c:<20}{npred:>11,}{m:>8.1f}%"); errs.append(m)
    if errs: print(f"  -> median MRE: {np.median(errs):.1f}%   mean: {np.mean(errs):.1f}%   (n={len(errs)} cells)")
    print()

# ── per-character transfer law: is energy/char and time/char stable across corpora? ──
print("################ PER-CHARACTER LAW (ntasks=1, largest size per corpus) ################")
d1=df[df.ntasks==1]
for st in ["string_normalization","heuristic_filtering"]:
    print(f"\n  {st}:")
    print(f"    {'corpus':<20}{'n':>9}{'J/char':>12}{'ms/char':>12}")
    for c in ["american_stories","cc_github_opencode","cc_german_pd","european_parliament"]:
        sub=d1[(d1.stage==st)&(d1.corpus==c)]
        if sub.empty: continue
        big=sub[sub.n==sub.n.max()]
        chars=big.n.mean()*CORPUS_CHARS.get(c,2000)
        jpc=big.energy_j.mean()/chars; mspc=big.time_sec.mean()/chars*1000
        print(f"    {c:<20}{int(big.n.mean()):>9,}{jpc:>12.5f}{mspc:>12.5f}")
