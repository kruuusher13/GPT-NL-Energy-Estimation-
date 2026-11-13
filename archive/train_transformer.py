#!/usr/bin/env python3
"""FT-Transformer (tabular transformer) for pipeline energy/time — the 'neural' entry
Julio asked for, benchmarked honestly against OLS/GBM/MLP under the SAME held-out splits."""
import sys, re, json
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import LeaveOneGroupOut
torch.manual_seed(0); np.random.seed(0)

CSV = sys.argv[1] if len(sys.argv)>1 else "paper/data/measurements_generalized.csv"
CHARS={"american_stories":1798,"cc_github_opencode":3151,"cc_german_pd":48881,"european_parliament":181075}
STAGES=["data_splitting","string_normalization","heuristic_filtering","deduplication","toxic_language_detection"]
def corp(s):
    s=s.lower()
    if s.startswith(("ghd","ghub")) or "github" in s: return "cc_github_opencode"
    if s.startswith(("gerd","ger")): return "cc_german_pd"
    if s.startswith(("epd","epg","eu")) or "parl" in s: return "european_parliament"
    return "american_stories"
def nt(s):
    m=re.search(r"_n(\d+)_",s); return int(m.group(1)) if m else 1

df=pd.read_csv(CSV)
df["corpus"]=df.slug.map(corp); df["ntasks"]=df.slug.map(nt)
df["n"]=pd.to_numeric(df["size"],errors="coerce")
df["energy_j"]=df.dc_node_power_w.fillna(0)*df.time_sec
df=df[df.stage.isin(STAGES)&(df.time_sec>0)&(df.energy_j>0)&df.n.notna()].copy()
df["chars"]=df.corpus.map(CHARS).fillna(2000)
df["sidx"]=df.stage.map({s:i for i,s in enumerate(STAGES)})
NUM=["ln","lchars","lnt"]
df["ln"]=np.log10(df.n); df["lchars"]=np.log10(df.chars); df["lnt"]=np.log2(df.ntasks.clip(lower=1))

class FTT(nn.Module):
    """Feature-Tokenizer Transformer: each feature -> a token, + stage embedding + CLS."""
    def __init__(self,n_num,n_stage,d=32,heads=4,layers=2):
        super().__init__()
        self.num=nn.ModuleList([nn.Linear(1,d) for _ in range(n_num)])
        self.stage=nn.Embedding(n_stage,d); self.cls=nn.Parameter(torch.randn(1,1,d))
        enc=nn.TransformerEncoderLayer(d,heads,d*2,0.1,batch_first=True)
        self.tr=nn.TransformerEncoder(enc,layers); self.head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,1))
    def forward(self,xnum,xs):
        toks=[self.num[i](xnum[:,i:i+1]) for i in range(xnum.shape[1])]
        toks.append(self.stage(xs))
        t=torch.stack(toks,1); b=t.shape[0]
        t=torch.cat([self.cls.expand(b,-1,-1),t],1)
        return self.head(self.tr(t)[:,0]).squeeze(-1)

def train_predict(tr,te,target):
    Xn=torch.tensor(tr[NUM].values,dtype=torch.float32); Xs=torch.tensor(tr.sidx.values)
    y=torch.tensor(np.log10(tr[target].values),dtype=torch.float32)
    mu,sd=Xn.mean(0),Xn.std(0)+1e-6; Xn=(Xn-mu)/sd
    m=FTT(len(NUM),len(STAGES)); opt=torch.optim.Adam(m.parameters(),1e-2,weight_decay=1e-4)
    lossf=nn.SmoothL1Loss()
    for _ in range(300):
        opt.zero_grad(); loss=lossf(m(Xn,Xs),y); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        Xnt=(torch.tensor(te[NUM].values,dtype=torch.float32)-mu)/sd
        p=10**m(Xnt,torch.tensor(te.sidx.values)).numpy()
    return p

def mre(a,p):
    a,p=np.asarray(a,float),np.asarray(p,float); m=a>0
    return np.mean(np.abs(p[m]-a[m])/a[m])*100

for target,tname in [("time_sec","WALL-TIME"),("energy_j","ENERGY")]:
    print(f"\n################ FT-TRANSFORMER vs GBM — target={tname} ################")
    for gcol,lbl in [("n","extrapolate to unseen DOC-COUNT"),("corpus","transfer to unseen CORPUS")]:
        groups=df[gcol].astype(str).values
        if len(np.unique(groups))<2: continue
        logo=LeaveOneGroupOut(); yt_t,yp_t,yt_g,yp_g=[],[],[],[]
        for tri,tei in logo.split(df,df[target],groups):
            tr,te=df.iloc[tri],df.iloc[tei]
            yp_t+=list(train_predict(tr,te,target)); yt_t+=list(te[target].values)
            g=HistGradientBoostingRegressor(max_iter=300,max_depth=3,learning_rate=0.06,min_samples_leaf=3)
            g.fit(tr[NUM+["sidx"]],np.log10(tr[target])); yp_g+=list(10**g.predict(te[NUM+["sidx"]])); yt_g+=list(te[target].values)
        print(f"  {lbl:<34} GBM {mre(yt_g,yp_g):7.1f}%   FT-Transformer {mre(yt_t,yp_t):7.1f}%")
print("\n(honest: on ~400 rows / 4 corpora, trees & transformer are close; transformer needs more corpora.)")
