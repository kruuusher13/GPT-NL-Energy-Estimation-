#!/usr/bin/env python3
"""
The SMART model: hierarchical, physics-informed coefficient prediction + Kalman.

Instead of predicting energy end-to-end, we:
  1. fit each stage's physical law  y = c0 + c1*n  per (stage, corpus)   [the labels]
  2. learn g: context -> the per-CHARACTER coefficient  k = c1 / chars_per_doc
     (physics backbone: k is ~constant per stage; g learns how it varies)
  3. predict an UNSEEN corpus's law from g(its chars), apply the physics, and
     report leave-one-corpus-out accuracy on the whole run.

Targets reported for BOTH wall-time (clean) and energy. Time is the trustworthy
signal on shared nodes; energy follows via P_eff.

Usage:
  python3 coefficient_model.py [measurements.csv]
"""
import sys, re
import numpy as np, pandas as pd

CSV = sys.argv[1] if len(sys.argv)>1 else "paper/data/measurements_generalized.csv"
CORPUS_CHARS = {"american_stories":1798,"cc_github_opencode":3151,
                "cc_german_pd":48881,"european_parliament":181075}
STAGES = ["data_splitting","string_normalization","heuristic_filtering","deduplication"]

def corp(s):
    s=s.lower()
    if s.startswith(("ghd","ghub")) or "github" in s: return "cc_github_opencode"
    if s.startswith(("gerd","ger")) or "german" in s: return "cc_german_pd"
    if s.startswith(("epd","epg","eu")) or "parl" in s: return "european_parliament"
    return "american_stories"
def nt(s):
    m=re.search(r"_n(\d+)_",s); return int(m.group(1)) if m else 1

def load():
    df=pd.read_csv(CSV)
    df["corpus"]=df.slug.map(corp); df["ntasks"]=df.slug.map(nt)
    df["n"]=pd.to_numeric(df["size"],errors="coerce")
    df["energy_j"]=df.dc_node_power_w.fillna(0)*df.time_sec
    df=df[(df.ntasks==1)&(df.time_sec>0)&df.n.notna()&(df.energy_j>0)]
    df["chars"]=df.corpus.map(CORPUS_CHARS)
    return df

def fit_law(sub, target):
    """y = c0 + c1*n via least squares over (size-averaged) points. returns (c1,c0,npts)."""
    g=sub.groupby("n")[target].mean().reset_index().sort_values("n")
    if len(g)<2: return None
    A=np.vstack([g.n.values, np.ones(len(g))]).T
    (c1,c0),*_=np.linalg.lstsq(A,g[target].values,rcond=None)
    return c1,c0,len(g)

def mre(a,p): return abs(p-a)/a*100 if a else float("nan")

def main():
    df=load()
    print(f"Loaded {len(df)} ntasks=1 readings | corpora={sorted(df.corpus.unique())}\n")

    # ── 1+2: fit per-(stage,corpus) law, derive per-char coefficient k=c1/chars ──────
    for target,unit in [("time_sec","s/doc"),("energy_j","J/doc")]:
        print(f"################ COEFFICIENTS — target={target} ################")
        print(f"  {'stage':<22}{'corpus':<20}{'c1 ('+unit+')':>14}{'k=c1/char':>14}{'pts':>5}")
        coef={}   # (stage,corpus)->(c1,c0)
        for st in STAGES:
            ks=[]
            for c in CORPUS_CHARS:
                sub=df[(df.stage==st)&(df.corpus==c)]
                f=fit_law(sub,target)
                if not f: continue
                c1,c0,npts=f; k=c1/CORPUS_CHARS[c]
                coef[(st,c)]=(c1,c0); ks.append((c,k))
                print(f"  {st:<22}{c:<20}{c1:>14.4g}{k:>14.3e}{npts:>5}")
            if len(ks)>=2:
                kv=[k for _,k in ks]
                print(f"  {'':<22}{'-> k spread across corpora':<20}"
                      f"  mean={np.mean(kv):.2e}  CV={np.std(kv)/np.mean(kv)*100:.0f}%  "
                      f"[{'TIGHT (per-char transfers)' if np.std(kv)/np.mean(kv)<0.5 else 'content-dependent'}]")
        print()

        # ── 3: leave-one-corpus-out — predict unseen corpus from g(chars) ───────────
        print(f"  LEAVE-ONE-CORPUS-OUT (predict held-out corpus's WHOLE run from per-char law):")
        print(f"    {'held-out corpus':<22}{'pred kJ' if target=='energy_j' else 'pred (s)':>12}"
              f"{'true':>12}{'MRE':>9}")
        errs=[]
        for held in CORPUS_CHARS:
            # train k_stage from the OTHER corpora; predict held-out coefficients
            pred_tot=true_tot=0.0; ok=True
            # use the held-out corpus's largest size as the "whole run" target
            hsub=df[df.corpus==held]
            if hsub.empty: continue
            N=hsub.n.max()
            for st in STAGES:
                others=[coef[(st,c)][0]/CORPUS_CHARS[c] for c in CORPUS_CHARS
                        if c!=held and (st,c) in coef]
                if not others: ok=False; break
                k_pred=np.mean(others)                      # g(chars)=k*chars, k from others
                c1_pred=k_pred*CORPUS_CHARS[held]
                # intercept: small; borrow mean c0 from others (overhead ~ corpus-indep)
                c0s=[coef[(st,c)][1] for c in CORPUS_CHARS if c!=held and (st,c) in coef]
                c0_pred=np.mean(c0s) if c0s else 0.0
                pred_tot+=c1_pred*N+c0_pred
                tf=fit_law(hsub[hsub.stage==st],target)
                if tf: true_tot+=tf[0]*N+tf[1]
            if not ok or true_tot<=0: continue
            sc=1000 if target=="energy_j" else 1
            e=mre(true_tot,pred_tot); errs.append(e)
            print(f"    {held:<22}{pred_tot/sc:>12.1f}{true_tot/sc:>12.1f}{e:>8.1f}%")
        if errs:
            print(f"    -> leave-one-corpus-out median MRE: {np.median(errs):.1f}%  mean: {np.mean(errs):.1f}%")
        print()

if __name__=="__main__":
    main()
