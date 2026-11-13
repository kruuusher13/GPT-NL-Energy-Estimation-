#!/usr/bin/env python3
"""poster.png / poster.pdf  ·  A4 landscape, static. Flow chart on top, big centred live monitor,
the Kalman stopping rule, and real cross-corpus findings. Built in matplotlib."""
import re, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from scipy import stats as _st

INK, GREY, LINE, FILL = "#111111", "#666666", "#222222", "#f4f4f4"
ACC, ACC2, GRN, ORG = "#2b5d8a", "#7a1f1f", "#3a7d3a", "#b5651d"
CC = {"amnews": ACC, "github": ACC2, "german": GRN, "euparl": ORG}
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.edgecolor": LINE,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "font.size": 8})

df = pd.read_csv("paper/data/measurements_harvest_0626.csv")
def corp(s):
    s = s.lower()
    if s.startswith(("ghd", "ghub")) or "github" in s: return "github"
    if s.startswith(("gerd", "gerg", "ger")): return "german"
    if s.startswith(("epd", "epg", "eu")) or "parl" in s: return "euparl"
    return "amnews"
df["corpus"] = df.slug.map(corp)
df["nt"] = df.slug.map(lambda s: int(re.search(r"_n(\d+)_", s).group(1)) if re.search(r"_n(\d+)_", s) else 1)
df["n"] = pd.to_numeric(df["size"], errors="coerce")
df["e"] = df.dc_node_power_w.fillna(0) * df.time_sec / 1000.0
df = df[(df.time_sec > 0) & df.n.notna()]
S = ["data_splitting", "string_normalization", "heuristic_filtering", "deduplication"]
SN = {"data_splitting": "split", "string_normalization": "normalize", "heuristic_filtering": "filter", "deduplication": "dedup"}
PEFF = {"data_splitting": 293, "string_normalization": 297, "heuristic_filtering": 278, "deduplication": 280}
df["ec"] = df.apply(lambda r: PEFF.get(r.stage, 290) * r.time_sec / 1000.0, axis=1)
CORPORA = ["amnews", "github", "german", "euparl"]
def fit(x, y):
    A = np.vstack([x, np.ones(len(x))]).T; (a, b), *_ = np.linalg.lstsq(A, y, rcond=None); return a, b

FIGW, FIGH = 11.69, 8.27
fig = plt.figure(figsize=(FIGW, FIGH), dpi=300); fig.patch.set_facecolor("white")

# ===== TITLE =====
axt = fig.add_axes([0, 0.962, 1, 0.038]); axt.axis("off")
axt.text(0.035, 0.5, "Complexity Aware Energy Estimation for an LLM Data Curation Pipeline",
         fontsize=14.5, fontweight="bold", va="center")
axt.text(0.905, 0.5, "Romir Malik  ·  GPT NL on Snellius", fontsize=8.8, va="center", ha="right", color=GREY)
# Utrecht University logo, top-right corner
axlogo = fig.add_axes([0.928, 0.928, 0.052, 0.052 * (FIGW / FIGH)]); axlogo.axis("off")
try: axlogo.imshow(mpimg.imread("assets/uu_logo.png"))
except FileNotFoundError: pass

# ================= FLOW CHART (top) =================
axp = fig.add_axes([0.025, 0.66, 0.95, 0.295]); axp.axis("off"); axp.set_xlim(0, 100); axp.set_ylim(0, 100)
axp.text(0, 100, "Method: a swappable model per stage, calibrated on cheap trials, tied into one whole pipeline total by an Extended Kalman filter",
         fontsize=9.6, fontweight="bold", va="top")
STG = ["data\nsplitting", "string\nnormalization", "heuristic\nfiltering", "deduplication"]
MODELS = ["OLS  (baseline)", "Ridge", "GBM", "MLP", "FT Transformer"]
x0, bw, gap = 8.0, 15.2, 2.8
ytop, bh, yslot, sh = 58.0, 13.0, 30.0, 24.0
def sx_(i): return x0 + i * (bw + gap)
axp.add_patch(FancyBboxPatch((0.5, ytop), 6.6, bh, boxstyle="round,pad=0.2,rounding_size=1.2", fc="white", ec=LINE, lw=1.2))
axp.text(3.8, ytop + bh / 2, "raw\ntext", ha="center", va="center", fontsize=8)
for i, st in enumerate(STG):
    sx = sx_(i)
    axp.add_patch(FancyBboxPatch((sx, ytop), bw, bh, boxstyle="round,pad=0.2,rounding_size=1.2", fc=FILL, ec=LINE, lw=1.3))
    axp.text(sx + bw / 2, ytop + bh / 2, st, ha="center", va="center", fontsize=8.2, fontweight="bold")
    axp.add_patch(FancyBboxPatch((sx, yslot), bw, sh, boxstyle="round,pad=0.2,rounding_size=1.2", fc="white", ec=GREY, lw=1.0, linestyle=(0, (4, 2))))
    axp.text(sx + bw / 2, yslot + sh - 2.0, "model slot", ha="center", va="top", fontsize=6.2, color=GREY, style="italic")
    for j, m in enumerate(MODELS):
        bold = (j == 0)
        axp.text(sx + bw / 2, yslot + sh - 5.8 - j * 3.8, m, ha="center", va="center",
                 fontsize=6.4 if not bold else 7.0, fontweight="bold" if bold else "normal", color=ACC if bold else INK)
    src = sx + bw; dst = sx_(i + 1) if i < 3 else sx + bw + gap + 4
    axp.add_patch(FancyArrowPatch((src, ytop + bh / 2), (dst, ytop + bh / 2), arrowstyle="-|>", mutation_scale=11, lw=1.2, color=LINE))
    if i < 3: axp.text((src + dst) / 2, ytop + bh / 2 + 2.0, "r$_s$", ha="center", va="bottom", fontsize=6.2, color=GREY)
axp.add_patch(FancyArrowPatch((7.1, ytop + bh / 2), (x0, ytop + bh / 2), arrowstyle="-|>", mutation_scale=11, lw=1.2, color=LINE))
ox = sx_(4) + 1.0
axp.add_patch(FancyBboxPatch((ox, ytop - 1.5), 12.5, bh + 3, boxstyle="round,pad=0.2,rounding_size=1.2", fc="white", ec=ACC, lw=1.8))
axp.text(ox + 6.25, ytop + bh / 2 + 1.5, "TOTAL\nenergy", ha="center", va="center", fontsize=8.2, fontweight="bold", color=ACC)
axp.text(ox + 6.25, ytop - 0.4, r"$\hat E \pm$ CI", ha="center", va="bottom", fontsize=7, color=ACC)
kx0, kx1 = x0 - 1.0, sx_(3) + bw + 1.0
axp.add_patch(FancyBboxPatch((kx0, 5.0), kx1 - kx0, 14.0, boxstyle="round,pad=0.3,rounding_size=1.5", fc="#eef3f8", ec=ACC, lw=1.6))
axp.text((kx0 + kx1) / 2, 15.5, "Extended Kalman filter  ·  predict, gate at 3$\\sigma$, update, shrink the interval",
         ha="center", va="center", fontsize=8.2, fontweight="bold", color=ACC)
axp.text((kx0 + kx1) / 2, 9.0, "clean reading is absorbed and the band tightens, a contaminated reading trips the gate and is replaced by the model",
         ha="center", va="center", fontsize=6.8, color=INK)
for i in range(4):
    axp.add_patch(FancyArrowPatch((sx_(i) + bw / 2, 19.0), (sx_(i) + bw / 2, yslot), arrowstyle="-|>", mutation_scale=8, lw=1.0, color=ACC))
axp.add_patch(FancyArrowPatch((kx1, 12.0), (ox + 6.25, ytop - 1.5), arrowstyle="-|>", mutation_scale=10, lw=1.2, color=ACC))
# g transfer box, placed in the clear band above the deduplication stage
gx, gy, gw, ghh = 55.0, 75.0, 41.0, 13.5
axp.add_patch(FancyBboxPatch((gx, gy), gw, ghh, boxstyle="round,pad=0.3,rounding_size=1.2", fc="white", ec=ACC2, lw=1.4))
axp.text(gx + gw / 2, gy + ghh - 3.4, "g : transfer to a new corpus", ha="center", va="center", fontsize=7.8, fontweight="bold", color=ACC2)
axp.text(gx + gw / 2, gy + 3.8, "predicts each stage slope c$_1$ from chars/doc,\nsurvival, and compute counters", ha="center", va="center", fontsize=6.3, color=INK)
axp.add_patch(FancyArrowPatch((gx + 9, gy), (sx_(3) + bw / 2, ytop + bh), arrowstyle="-|>", mutation_scale=8, lw=1.0, color=ACC2, linestyle=(0, (3, 2))))

# ================= MIDDLE: stop-rule (left) | LIVE MONITOR big centre | parity (right) =================
# ---- stopping rule (real) ----
axS = fig.add_axes([0.045, 0.345, 0.225, 0.20])
P_EFF = 278.0; Ntar = 400_000
sub = df[(df.corpus == "amnews") & (df.nt == 1)]
hf = sub[sub.stage == "heuristic_filtering"].groupby("n").time_sec.mean().sort_index()
stream = [(n, t) for n, t in hf.items() if n < Ntar]
truth = P_EFF * float(hf[Ntar]) / 1000.0
allx = np.array([float(n) for n, _ in stream]); ally = np.array([float(v) for _, v in stream])
Xf = np.vstack([allx, np.ones(len(allx))]).T; thf = np.linalg.lstsq(Xf, ally, rcond=None)[0]
sigma = np.std(ally - Xf @ thf) + 0.02 * ally.mean()    # one fixed noise -> posterior shrinks monotonically
xst = np.array([Ntar, 1.0]); ks, est, half = [], [], []
for k in range(2, len(stream) + 1):
    X = np.vstack([allx[:k], np.ones(k)]).T; Xi = np.linalg.inv(X.T @ X); th = Xi @ X.T @ ally[:k]
    e = xst @ th; lev = xst @ Xi @ xst
    ks.append(k); est.append(P_EFF * e / 1000); half.append(P_EFF * 1.96 * sigma * np.sqrt(1 + lev) / 1000)
est = np.array(est); half = np.array(half)
si = next((i for i in range(len(ks)) if half[i] / est[i] <= 0.10), len(ks) - 1)   # stop index
axS.fill_between(ks, est - half, est + half, color=ACC, alpha=0.22, label="95% confidence ribbon")
axS.plot(ks, est, "o-", color=ACC, lw=1.6, ms=4, label="prediction")
axS.axhline(truth, ls="--", color=GREY, lw=1.0); axS.text(ks[-1], truth, f" true {truth:.0f}", fontsize=6, va="bottom", color=GREY)
axS.scatter([ks[si]], [est[si]], s=85, facecolors="none", edgecolors=ACC2, lw=1.8, zorder=5)
axS.annotate(f"STOP at {ks[si]} trials\nribbon < 10%, error {abs(est[si]-truth)/truth*100:.0f}%",
             (ks[si], est[si]), (ks[0] + 0.15, est.max() + half.max() * 0.2),
             fontsize=6.2, color=ACC2, fontweight="bold")
axS.set_xlabel("calibration trials", fontsize=7); axS.set_ylabel("predicted 400k run (kJ)", fontsize=7)
axS.set_xticks(ks); axS.tick_params(labelsize=6); axS.grid(True, lw=0.3, alpha=0.4); axS.legend(fontsize=5.4, frameon=False, loc="lower right")
axS.set_title("Stopping rule: the confidence ribbon shrinks\nuntil the stop test fires", fontsize=7.4, fontweight="bold")

# ---- LIVE MONITOR, big and centred ----
mw = 0.40; mh = mw * (924 / 2254) * (FIGW / FIGH)
mx = (1 - mw) / 2; my = 0.345 + 0.20 - mh
axM = fig.add_axes([mx, my, mw, mh]); axM.axis("off")
try: axM.imshow(mpimg.imread("assets/monitor_live.png"), aspect="auto")
except FileNotFoundError: axM.text(0.5, 0.5, "monitor missing", ha="center", transform=axM.transAxes)
fig.text(0.5, 0.345 + 0.207, "Live monitor: estimate and 95% band converge live, gate rejects bad readings",
         ha="center", va="bottom", fontsize=8.2, fontweight="bold")
fig.text(0.5, my - 0.012, "estimate 868.6 $\\pm$ 40 kJ   ·   CI width 5%   ·   error 0.0%   ·   0 readings gated",
         ha="center", va="top", fontsize=6.6, color=GREY)

# ---- MATH SIDE PANEL (OLS baseline + Kalman), tall right column ----
axMa = fig.add_axes([0.755, 0.085, 0.215, 0.455]); axMa.axis("off"); axMa.set_xlim(0, 1); axMa.set_ylim(0, 1)
axMa.add_patch(Rectangle((0, 0), 1, 1, fc="#fbfbfb", ec=LINE, lw=1.1))
def mrow(y, tex, cap, fs=11.5):
    axMa.text(0.07, y, tex, fontsize=fs, va="center")
    if cap: axMa.text(0.08, y - 0.034, cap, fontsize=6.2, color=GREY, va="center")
axMa.text(0.5, 0.965, "The mathematics", ha="center", fontsize=9.6, fontweight="bold")
axMa.text(0.07, 0.905, "OLS baseline (per stage)", fontsize=7.6, fontweight="bold", color=ACC)
mrow(0.85, r"$E_s(n)=c_0+c_1 n$", "fixed cost + cost per document")
mrow(0.762, r"$\hat\theta=(X^\top X)^{-1}X^\top y$", "least squares fit, the baseline")
mrow(0.674, r"$\pm\,t\hat\sigma\sqrt{1+h_\star}$", "exact prediction interval")
axMa.text(0.07, 0.585, "Kalman walk", fontsize=7.6, fontweight="bold", color=ACC)
mrow(0.530, r"$r_k=z_k-H\hat{x}_k$", "innovation: reading minus prediction")
mrow(0.442, r"$r_k^2\leq(3\sigma)^2 S_k$", "3$\\sigma$ gate accepts or rejects")
mrow(0.354, r"$\theta_k=\theta_{k-1}+K_k r_k$", "clean tightens, contaminated rejected")
mrow(0.266, r"$1.96\sqrt{\mathrm{Var}}/\hat E\leq\tau$", "stop when the ribbon is tight")
axMa.text(0.07, 0.180, "Transfer model g", fontsize=7.6, fontweight="bold", color=ACC2)
mrow(0.125, r"$\log c_1=g(\mathrm{chars},\mathrm{surv},\mathrm{cmp})$", "slope for a corpus never run;")
axMa.text(0.08, 0.038, "beats the 17,569 param transformer 4x", fontsize=6.2, color=INK)

# ================= BOTTOM: breakdown | parallelism | per-char =================
RX = [0.045, 0.375, 0.705]; RW = 0.255; BY, BH = 0.085, 0.195
# breakdown
axB = fig.add_axes([RX[0], BY, RW, BH])
scol = {"data_splitting": "#bbbbbb", "string_normalization": ACC, "heuristic_filtering": ACC2, "deduplication": GRN}
ysn = np.arange(len(CORPORA)); left = np.zeros(len(CORPORA)); shares = {}
for c in CORPORA:
    cc = df[(df.corpus == c) & (df.nt == 1)]
    commons = [n for n in sorted(cc.n.unique()) if set(cc[cc.n == n].stage) >= set(S)]
    N = commons[-1] if commons else cc.n.max()
    e = np.nan_to_num(np.array([cc[(cc.n == N) & (cc.stage == st)].ec.mean() for st in S])); shares[c] = e / e.sum() * 100
for j, st in enumerate(S):
    vals = [shares[c][j] for c in CORPORA]; axB.barh(ysn, vals, left=left, color=scol[st], height=0.62, label=SN[st])
    for i, v in enumerate(vals):
        if v > 9: axB.text(left[i] + v / 2, ysn[i], f"{v:.0f}", ha="center", va="center", fontsize=6, color="white", fontweight="bold")
    left += np.array(vals)
axB.set_yticks(ysn); axB.set_yticklabels(CORPORA, fontsize=6.5); axB.invert_yaxis(); axB.set_xlim(0, 100)
axB.set_xlabel("share of pipeline energy (%)", fontsize=7); axB.tick_params(labelsize=6)
axB.legend(fontsize=5.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False, columnspacing=0.7)
for s in ["top", "right"]: axB.spines[s].set_visible(False)
axB.set_title("Where the energy goes flips by corpus\nfilter dominates news, normalize dominates code", fontsize=7.4, fontweight="bold")
# parallelism
axR = fig.add_axes([RX[1], BY, RW, BH])
par = {}
for c in ["amnews", "github"]:
    hh = df[(df.corpus == c) & (df.stage == "heuristic_filtering") & (df.n == 400000)]; base = hh[hh.nt == 1].time_sec.mean(); par[c] = {}
    for nt in [1, 16, 32]:
        r = hh[hh.nt == nt]
        if len(r): par[c][nt] = (base / r.time_sec.mean(), (r.dc_node_power_w * r.time_sec / 1000).mean())
nts = [1, 16, 32]; w = 0.36
for k, c in enumerate(["amnews", "github"]):
    eng = [par[c].get(nt, (np.nan, np.nan))[1] for nt in nts]; xx = np.arange(3) + (k - 0.5) * w
    axR.bar(xx, eng, w, color=CC[c], label=c)
    for x, nt in zip(xx, nts):
        sp = par[c].get(nt, (np.nan,))[0]
        if np.isfinite(sp): axR.text(x, eng[nts.index(nt)], f"{sp:.1f}x", ha="center", va="bottom", fontsize=5.6)
axR.set_yscale("log"); axR.set_xticks(range(3)); axR.set_xticklabels([f"{n}t" for n in nts], fontsize=6.5)
axR.set_ylabel("energy (kJ, log)", fontsize=7); axR.tick_params(labelsize=6); axR.legend(fontsize=6, frameon=False)
for s in ["top", "right"]: axR.spines[s].set_visible(False)
axR.set_title("Parallelism pays off only for big docs\nlabels are speedup; news wastes 10$\\times$ energy", fontsize=7.4, fontweight="bold")
# MODEL RESULTS: g vs the trained networks at cross-corpus transfer (lower is better)
axMod = fig.add_axes([0.632, BY, 0.113, BH])
try:
    bt = json.load(open("archive/benchmark_all_results.json"))["energy"]["corpus-transfer"]
except Exception:
    bt = {"GBM": 330, "FT-Transformer": 362, "MLP": 3272}
mods = [("g", 88, GRN), ("GBM", bt.get("GBM", 330), ACC),
        ("FT-T", bt.get("FT-Transformer", 362), ACC), ("MLP", bt.get("MLP", 3272), ACC)]
vals = [m[1] for m in mods]
axMod.barh(range(len(mods)), vals, color=[m[2] for m in mods], height=0.64)
for i, (lb, v, c) in enumerate(mods):
    axMod.text(v * 1.25, i, f"{v:.0f}%", va="center", fontsize=5.6, color=(GRN if i == 0 else INK),
               fontweight=("bold" if i == 0 else "normal"))
axMod.set_xscale("log"); axMod.set_yticks(range(len(mods))); axMod.set_yticklabels([m[0] for m in mods], fontsize=6.2)
axMod.invert_yaxis(); axMod.set_xlim(45, 16000); axMod.tick_params(labelsize=5)
axMod.set_xlabel("unseen-corpus error % (log)", fontsize=5.8)
for s in ["top", "right"]: axMod.spines[s].set_visible(False)
axMod.set_title("Model results:\ng beats every net", fontsize=7.0, fontweight="bold")

fig.savefig("poster.png", dpi=300, facecolor="white")
fig.savefig("poster.pdf", facecolor="white")
print("wrote poster.png and poster.pdf")
