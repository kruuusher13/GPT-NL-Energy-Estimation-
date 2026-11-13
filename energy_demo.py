#!/usr/bin/env python3
"""
GPT-NL pipeline energy — LIVE estimation demo (per-task EKF).

A terminal dashboard that shows the pipeline's total-energy estimate converging in
real time as each parallel TASK finishes: the point estimate, the shrinking 95%
confidence band, the streaming task readings, and the innovation gate rejecting
contaminated shared-node spikes. The estimate starts as a pure model PREDICTION
(wide band) and is recalibrated online toward the measured truth as readings arrive.

This is the EKF / "calibrate as it runs" story made visible — now PER TASK, not per stage.

  python3 energy_demo.py                      # auto-pick a run, live TUI
  python3 energy_demo.py --run amg_n16_size400000_rep1 --speed 0.12
  python3 energy_demo.py --test               # no UI, verify the estimator math
  python3 energy_demo.py --gate-demo          # inject a contaminated reading to show the gate

Data: paper/data/measurements_raw_generalized.csv (one row per parallel task).
"""
import argparse, json, math, time, random, sys
import pandas as pd, numpy as np

RAW = "/Users/hornet/Projects/TNO/paper/data/measurements_raw_generalized.csv"
FITS = "/Users/hornet/Projects/TNO/paper/data/ols_fits_with_dedup.json"
CORPUS_CHARS = {"american_stories":1798,"cc_github_opencode":3151,
                "cc_german_pd":48881,"european_parliament":181075}
CAL_CHARS = CORPUS_CHARS["american_stories"]
STAGE_ORDER = ["data_splitting","string_normalization","heuristic_filtering",
               "toxic_language_detection","deduplication"]
TOXIC_FIT = {"c1":0.135,"c0":6843.0}

def corp(s):
    s=s.lower()
    if s.startswith(("ghd","ghub")) or "github" in s: return "cc_github_opencode"
    if s.startswith(("gerd","ger")) or "german" in s: return "cc_german_pd"
    if s.startswith(("epd","epg","eu")) or "parl" in s: return "european_parliament"
    return "american_stories"

def load_fits():
    d=json.load(open(FITS)); fits={}
    for st,f in d.get("fits",d).items():
        th=f.get("theta") or [f.get("marginal_theta",[0])[0],f.get("intercept_j",0)]
        fits[st]={"c1":float(th[0]),"c0":float(th[1])}
    fits.setdefault("toxic_language_detection",TOXIC_FIT)
    return fits

def model_pred_j(fits, stage, n, corpus):
    """per-stage energy prediction (J), transferred across corpora via chars/doc."""
    f=fits.get(stage,{"c1":0.0,"c0":0.0}); c1=f["c1"]
    if corpus!="american_stories" and corpus in CORPUS_CHARS:
        c1=c1/CAL_CHARS*CORPUS_CHARS[corpus]
    return max(f["c0"]+c1*n, 1.0)

# ── per-task sequential estimator (the EKF, online-calibrated) ───────────────────
class LiveEstimator:
    """State = per-stage energy. Prior = model. Each task reading recalibrates the
    stage online (blend prior <-> data-extrapolation), with an innovation gate that
    rejects contaminated readings. Total = sum of stages; CI from summed variance."""
    def __init__(self, fits, n, corpus, stages, tasks_per_stage, prior_rel=0.45, k0=2.0, gate_ratio=2.5):
        self.k0=k0; self.gate_ratio=gate_ratio
        self.stages=stages
        self.pred={s:model_pred_j(fits,s,n,corpus) for s in stages}
        self.M={s:max(tasks_per_stage.get(s,1),1) for s in stages}
        self.P0={s:(prior_rel*self.pred[s])**2 for s in stages}
        self.seen={s:[] for s in stages}        # accepted per-task energies
        self.rejected=0
    def update(self, stage, energy_j):
        """returns (accepted: bool, ratio-to-stage-budget). Gate fires only when a single
        reading exceeds the WHOLE stage's predicted energy by gate_ratio x (= contamination);
        robust to heterogeneous sub-jobs like dedup's dominant signature pass."""
        sp=self.pred[stage]
        ratio=energy_j/max(sp,1e-9)
        if ratio>self.gate_ratio:                       # contaminated spike -> use prediction
            self.seen[stage].append(sp/self.M[stage]); self.rejected+=1
            return False, ratio
        self.seen[stage].append(energy_j)
        return True, ratio
    def estimate(self):
        tot=0.0; var=0.0
        for s in self.stages:
            k=len(self.seen[s]); M=self.M[s]; prior=self.pred[s]
            if k==0:                                       # not started -> pure prior
                est=prior; v=self.P0[s]
            elif k>=M:                                     # stage COMPLETE -> known
                est=sum(self.seen[s]); v=(0.03*est)**2     # only measurement noise
            else:                                          # partial -> blend prior <-> data
                measured=sum(self.seen[s])
                extrap=measured*M/k                        # data-only stage total
                w=k/(k+self.k0)                            # trust data more as k grows
                est=w*extrap+(1-w)*prior
                spread=np.var(self.seen[s]) if k>1 else (0.30*np.mean(self.seen[s]))**2
                v=(1-w)**2*self.P0[s]+spread*max(M-k,0)    # prior-residual + sampling unc
            tot+=est; var+=v
        return tot, 1.96*math.sqrt(max(var,0))

def build_stream(run, gate_demo=False):
    df=pd.read_csv(RAW)
    df=df[df.slug==run].copy()
    if df.empty: sys.exit(f"no rows for run '{run}'")
    df["energy_j"]=df.dc_node_power_w.fillna(0)*df.time_sec
    df=df[df.energy_j>0]
    df["ord"]=df.stage.map(lambda s: STAGE_ORDER.index(s) if s in STAGE_ORDER else 9)
    df=df.sort_values(["ord","job_id","time_sec"]).reset_index(drop=True)
    clean_truth=df.energy_j.sum()                 # truth BEFORE any contamination
    if gate_demo and len(df)>3:                   # inject contamination on a MAJOR stage reading
        i=int(df["energy_j"].values.argmax())     # the biggest (clearly a main stage)
        df.iloc[i, df.columns.get_loc("energy_j")] *= 6.0
    df.attrs["clean_truth"]=clean_truth
    return df

def make_estimator(df, run):
    n=int(df["size"].iloc[0]); c=corp(run)
    stages=[s for s in STAGE_ORDER if s in set(df.stage)]
    tps=df.groupby("stage").size().to_dict()
    return LiveEstimator(load_fits(), n, c, stages, tps), n, c, stages

# ── rendering ────────────────────────────────────────────────────────────────────
SHORT={"data_splitting":"split","string_normalization":"normalize",
       "heuristic_filtering":"heuristic","toxic_language_detection":"toxic",
       "deduplication":"dedup"}

def render(run, n, corpus, est, truth, hist, last_rows, prior0, width=140, times=None):
    import plotext as plt
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.layout import Layout
    from rich.console import Group

    side_w=46
    graph_w=max(54, width-side_w-8)
    # x-axis = elapsed wall-clock TIME if we have it, else reading index
    if times and len(times)==len(hist):
        xs=[t/60.0 for t in times]; xlab="elapsed time (min)"      # predicted energy vs TIME
    else:
        xs=list(range(1,len(hist)+1)); xlab="stages completed"
    e=[h[0]/1000 for h in hist]; up=[(h[0]+h[1])/1000 for h in hist]; lo=[(h[0]-h[1])/1000 for h in hist]

    # ── GRAPH 1: estimate (+CI) over time, vs pre-run prediction and truth ──────────
    plt.clf(); plt.theme("pro"); plt.plotsize(graph_w,14)
    plt.plot(xs, [prior0/1000]*len(xs), color="white")              # pre-run prediction (flat)
    if truth is not None:
        plt.plot(xs, [truth/1000]*len(xs), color="green+")          # measured truth (flat)
    plt.plot(xs, up, color="magenta"); plt.plot(xs, lo, color="magenta")  # CI band
    plt.plot(xs, e, color="cyan+", marker="braille")                # estimate (bright)
    plt.xlabel(xlab); plt.ylabel("kJ")
    graph1=Text.from_ansi(plt.build())
    key1=Text.from_markup("  [cyan]━ estimate[/]  [magenta]━ 95% CI[/]  [white]━ pre-run prediction[/]  "
                          +("[green]━ measured truth[/]" if truth is not None else "[dim](truth at completion)[/]"))

    # ── GRAPH 2: REAL energy readings from the kernel (EAR), measured kJ per stage ───
    order=[s for s in STAGE_ORDER if s in est.seen and est.seen[s]]
    labels=[SHORT.get(s,s[:6]) for s in order]
    vals=[sum(est.seen[s])/1000 for s in order]
    plt.clf(); plt.theme("pro"); plt.plotsize(graph_w,9)
    if vals:
        plt.bar(labels, vals, color="orange", width=0.4)
    plt.ylabel("kJ"); plt.title("")
    graph2=Text.from_ansi(plt.build())

    cur,ci=hist[-1]; ciw=ci/cur*100 if cur else 0
    s=Table(show_header=False,box=None,pad_edge=False)
    s.add_column(style="bold cyan",no_wrap=True); s.add_column(no_wrap=True)
    s.add_row("run", f"[yellow]{run}[/]")
    s.add_row("docs", f"{n:,}  [dim]{corpus}[/]")
    s.add_row("predicted", f"{prior0/1000:7.1f} kJ  [dim](pre-run)[/]")
    s.add_row("estimate", f"[bold cyan]{cur/1000:7.1f} kJ[/] ± {ci/1000:.0f}")
    if truth is not None:
        err=abs(cur-truth)/truth*100 if truth else 0
        s.add_row("truth", f"[bold green]{truth/1000:7.1f} kJ[/]")
        s.add_row("error", f"[{'bold green' if err<10 else 'yellow'}]{err:.1f}%[/]")
    else:
        meas=sum(sum(v) for v in est.seen.values())
        s.add_row("status", "[bold green]● LIVE[/]")
        s.add_row("measured", f"[green]{meas/1000:7.1f} kJ[/] [dim]so far[/]")
    s.add_row("CI width", f"{ciw:.0f}%")
    s.add_row("readings", f"{len(hist)}   [dim]gated[/] [red]{est.rejected}[/]")

    t=Table(show_header=True,header_style="bold",box=None,expand=True,pad_edge=False)
    for col,j in (("#","right"),("stage","left"),("kJ","right"),("gate","center"),
                  ("est","right"),("±","right")):
        t.add_column(col,justify=j,no_wrap=True)
    for r in last_rows[-10:]:
        g="[green]✓[/]" if r["ok"] else "[red]✗REJ[/]"
        t.add_row(str(r["i"]), SHORT.get(r["stage"],r["stage"][:9]),
                  f"{r['e']/1000:.1f}", g, f"{r['est']/1000:.0f}", f"{r['ci']/1000:.0f}")

    lay=Layout()
    lay.split_column(Layout(name="head",size=3), Layout(name="body"))
    lay["head"].update(Panel(Text(" GPT-NL PIPELINE — LIVE ENERGY ESTIMATION  (per-task EKF: predict → calibrate as it runs)",
                                  style="bold white on blue")))
    lay["body"].split_row(Layout(name="g"), Layout(name="side",size=side_w))
    lay["g"].split_column(Layout(name="g1"), Layout(name="g2",size=11))
    lay["g"]["g1"].update(Panel(Group(graph1,key1),
                          title="predicted energy vs time  (estimate → truth, CI shrinking)",border_style="cyan"))
    lay["g"]["g2"].update(Panel(graph2,title="real energy readings from EAR (measured kJ per stage)",
                          border_style="orange3"))
    lay["side"].split_column(Panel(s,title="state",border_style="yellow"),
                             Panel(t,title="streaming readings",border_style="white"))
    return lay

def run_test(df, run):
    est,n,c,stages=make_estimator(df,run)
    truth=df.attrs.get("clean_truth", df.energy_j.sum())
    for _,r in df.iterrows(): est.update(r.stage, r.energy_j)
    cur,ci=est.estimate()
    print(f"run={run} n={n:,} corpus={c}")
    print(f"  pre-run prediction : {sum(est.pred.values())/1000:8.1f} kJ")
    print(f"  final estimate     : {cur/1000:8.1f} kJ  ± {ci/1000:.1f}")
    print(f"  measured truth     : {truth/1000:8.1f} kJ")
    print(f"  final error        : {abs(cur-truth)/truth*100:.1f}%   gate rejects={est.rejected}")

# ── LIVE mode: drive the EKF off ACTUAL running SLURM jobs on Snellius ────────────
SSH_HOST="rmalik@snellius.surf.nl"
RUNS_REMOTE="/projects/0/prjs0986/energy_studies/test_romir/runs"

def fetch_live(run):
    """SSH to Snellius, return [(stage, csv_id, energy_j)] for every completed EAR CSV
    of this run so far (dedup contributes several sub-job CSVs)."""
    import subprocess
    cmd=(f'for f in $(find "{RUNS_REMOTE}/{run}" -path "*emission/ear_db_*_apps.csv" 2>/dev/null); do '
         'stage=$(echo "$f" | sed -E "s#.*/stage[0-9]+_([^/]+)/.*#\\1#"); b=$(basename "$f"); '
         'e=$(awk -F";" \'NR==1{for(i=1;i<=NF;i++)c[$i]=i;next} $1=="JOBID"{next} '
         '{t=$(c["TIME_SEC"]);p=$(c["DC_NODE_POWER_W"]); if(t>0&&p>0)s+=p*t} END{printf "%.0f", s+0}\' "$f"); '
         'echo "$stage|$b|$e"; done')
    try:
        out=subprocess.run(["ssh","-o","BatchMode=yes","-o","ConnectTimeout=20",SSH_HOST,cmd],
                           capture_output=True,text=True,timeout=40).stdout
    except Exception:
        return []
    res=[]
    for ln in out.splitlines():
        p=ln.split("|")
        if len(p)==3 and p[2].strip():
            try: res.append((p[0], p[1], float(p[2])))
            except ValueError: pass
    return res

def run_live_ssh(run, poll=8.0, timeout=2400):
    from rich.live import Live
    from rich.console import Console
    W=Console().width
    n=int(__import__("re").search(r"size(\d+)",run).group(1)); c=corp(run)
    stages=["data_splitting","string_normalization","heuristic_filtering","deduplication"]
    est=LiveEstimator(load_fits(), n, c, stages, {s:1 for s in stages})
    prior0=sum(est.pred.values()); hist=[]; rows=[]; times=[]; fed=set(); idle=0; t0=time.time()
    with Live(refresh_per_second=8, screen=True) as live:
        while time.time()-t0 < timeout:
            new=0
            for (stage,cid,e) in fetch_live(run):
                if (stage,cid) in fed or stage not in est.pred: continue
                ok,ratio=est.update(stage,e); fed.add((stage,cid))
                est.M[stage]=len(est.seen[stage])           # treat seen-so-far as complete
                cur,ci=est.estimate(); hist.append((cur,ci)); times.append(time.time()-t0)
                rows.append({"i":len(fed),"stage":stage,"e":e,"ok":ok,"est":cur,"ci":ci}); new+=1
            if not hist:                                     # nothing finished yet: show pure prior
                hist.append((prior0,1.96*math.sqrt(sum(est.P0.values())))); times.append(time.time()-t0)
            live.update(render(run,n,c,est,None,hist,rows,prior0,W,times))
            seen_stages={r["stage"] for r in rows}
            idle = idle+1 if new==0 else 0
            if seen_stages>=set(stages) and idle>=3: break   # all 4 stages in + settled
            time.sleep(poll)
        # hold the final frame until the user quits
        try:
            while True: time.sleep(0.5)
        except KeyboardInterrupt:
            pass
    cur,ci=hist[-1]
    print(f"\nLIVE done: {run}  final estimate {cur/1000:.1f} kJ ± {ci/1000:.1f}  "
          f"({len(fed)} readings, {est.rejected} gated)")

def run_live(df, run, speed):
    from rich.live import Live
    from rich.console import Console
    W=Console().width
    est,n,c,stages=make_estimator(df,run)
    truth=df.attrs.get("clean_truth", df.energy_j.sum()); prior0=sum(est.pred.values())
    hist=[]; rows=[]; times=[]; tcum=0.0
    with Live(refresh_per_second=12, screen=True) as live:
        # start frame: pure pre-run prediction with its wide band (so you SEE it converge)
        hist.append((prior0,1.96*math.sqrt(sum(est.P0.values())))); times.append(0.0)
        live.update(render(run,n,c,est,truth,hist,rows,prior0,W,times)); time.sleep(1.2)
        for i,(_,r) in enumerate(df.iterrows(),1):
            ok,z=est.update(r.stage, r.energy_j)
            cur,ci=est.estimate(); hist.append((cur,ci))
            tcum+=float(r.time_sec); times.append(tcum)             # cumulative pipeline time
            rows.append({"i":i,"stage":r.stage,"e":r.energy_j,"ok":ok,"est":cur,"ci":ci})
            live.update(render(run,n,c,est,truth,hist,rows,prior0,W,times))
            time.sleep(speed)
        # HOLD the final frame until the user quits (Ctrl-C), so it never just vanishes
        try:
            while True: time.sleep(0.5)
        except KeyboardInterrupt:
            pass

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run",default=None)
    ap.add_argument("--speed",type=float,default=0.7)
    ap.add_argument("--test",action="store_true")
    ap.add_argument("--gate-demo",action="store_true")
    ap.add_argument("--live",action="store_true",help="drive off ACTUAL running Snellius jobs")
    ap.add_argument("--poll",type=float,default=8.0)
    a=ap.parse_args()
    if a.live:
        if not a.run: sys.exit("--live needs --run SLUG (a currently-running run)")
        run_live_ssh(a.run, a.poll); return
    if a.run is None:                            # default to the smooth 16-reading am-stories run
        df=pd.read_csv(RAW); df=df[df.dc_node_power_w.fillna(0)>0]
        if (df.slug=="amg_n1_size400000_rep1").any():
            a.run="amg_n1_size400000_rep1"
        else:
            n1=df[df.slug.str.contains("_n1_") | df.slug.str.match(r"size\d")]
            pick=n1 if not n1.empty else df
            a.run=pick.groupby("slug").size().sort_values().index[-1]
    df=build_stream(a.run, a.gate_demo)
    if a.test: run_test(df,a.run)
    else: run_live(df,a.run,a.speed)

if __name__=="__main__":
    main()
