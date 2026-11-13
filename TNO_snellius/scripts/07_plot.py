#!/usr/bin/env python3
"""
Produce v2 figures from measurements.csv + ols_fits.json + loops_validation.csv.

No notebook. PNGs land in figures/.

Figures:
  fig_per_stage_fits.png       — scatter + ridge fit + 95% PI per stage
  fig_baselines.png            — MAPE bar chart (CASE-EKF vs raw vs const vs lin)
  fig_loops_vs_apps.png        — apps-level E vs ∫P(t)dt scatter
  fig_phase_check.png          — EAR phase classification cross-check
  fig_ekf_modes.png            — Mode A pre-exec vs Mode B sequential CI narrowing
"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
WS = HERE.parent
MEAS = WS / "measurements" / "measurements.csv"
FITS = WS / "measurements" / "ols_fits.json"
LOOPS = WS / "measurements" / "loops_validation.csv"
FIGS = WS / "figures"
FIGS.mkdir(exist_ok=True)

from lib_ekf import PipelineEKF, t_ppf_975  # noqa: E402

STAGES = ["data_splitting", "string_normalization",
          "heuristic_filtering", "deduplication"]
COLORS = {"data_splitting": "#2196F3", "string_normalization": "#4CAF50",
          "heuristic_filtering": "#FF9800", "deduplication": "#9C27B0"}


def load_measurements():
    rows = list(csv.DictReader(open(MEAS)))
    by_stage = defaultdict(list)
    for r in rows:
        by_stage[r["stage"]].append(r)
    return rows, by_stage


def fig_per_stage_fits():
    rows, by_stage = load_measurements()
    blob = json.load(open(FITS))
    fits, AVG_T = blob["fits"], blob["avg_tokens"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    n_range = np.linspace(500, 200_000, 200)

    for ax, stage in zip(axes.flatten(), STAGES):
        if stage not in fits:
            ax.set_title(f"{stage} — no data"); ax.axis("off"); continue

        f = fits[stage]
        theta = np.array(f["theta"])
        XtX_inv = np.array(f["XtX_inv"])
        col = COLORS[stage]

        # Scatter (measurements)
        ns = [float(r["size"]) for r in by_stage[stage]]
        Es = [float(r["dc_node_power_w"]) * float(r["time_sec"])
              for r in by_stage[stage]]
        ax.scatter(np.array(ns)/1000, np.array(Es)/1000, color=col, alpha=0.7, s=30)

        # Fit + 95% PI
        def phi_vec(n):
            v = []
            for feat in f["features"]:
                if feat == "n":      v.append(n)
                elif feat == "n_t":  v.append(n * AVG_T)
                elif feat == "n_logn": v.append(n * math.log(max(n, 1)))
            v.append(1.0)
            return np.array(v)

        Es_pred = np.array([float(theta @ phi_vec(n)) for n in n_range])
        stds = np.array([math.sqrt(f["sigma2"] * (1 + phi_vec(n) @ XtX_inv @ phi_vec(n)))
                          for n in n_range])
        dof = max(f["n_obs"] - len(f["features"]) - 1, 1)
        tc = t_ppf_975(dof)

        ax.plot(n_range/1000, Es_pred/1000, color=col, lw=2,
                label=f"ridge λ={f['lambda']:.2g}  R²={f['r2']:.2f}")
        ax.fill_between(n_range/1000, (Es_pred - tc*stds)/1000,
                        (Es_pred + tc*stds)/1000, alpha=0.15, color=col)

        ax.set_title(stage.replace('_', ' ').title())
        ax.set_xlabel("Documents (thousands)")
        ax.set_ylabel("Energy (kJ)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    plt.suptitle("Per-stage ridge OLS fits — american-stories, real EAR readings",
                 fontweight="bold")
    plt.tight_layout()
    out = FIGS / "fig_per_stage_fits.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  wrote {out}")


def fig_ekf_modes():
    if not FITS.exists(): return
    blob = json.load(open(FITS))
    AVG_T = blob["avg_tokens"]
    N_TEST = 100_000

    ekf_a = PipelineEKF(FITS, N_TEST, AVG_T)
    rows_a = ekf_a.predict_all(STAGES)
    Es = [r["E_hat"] for r in rows_a]
    stds = [r["std"] for r in rows_a]

    fig, ax = plt.subplots(figsize=(9, 4))
    labels = [s.replace('_', '\n') for s in STAGES]
    x = np.arange(len(STAGES))
    bars = ax.bar(x, np.array(Es)/1e3, yerr=2*np.array(stds)/1e3,
                  color=[COLORS[s] for s in STAGES], alpha=0.85, capsize=5)
    for i, (b, p) in enumerate(zip(bars, Es)):
        pct = 100 * p / sum(Es)
        ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                f" {pct:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Predicted energy (kJ) — 95% CI")
    ax.set_title(f"Mode A pre-execution prediction · {N_TEST:,} docs · 4 stages",
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out = FIGS / "fig_ekf_modes.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  wrote {out}")


def fig_loops_vs_apps():
    if not LOOPS.exists():
        print("  skipping loops figure — no loops_validation.csv yet")
        return
    rows = list(csv.DictReader(open(LOOPS)))
    if not rows: return
    fig, ax = plt.subplots(figsize=(7, 6))
    for stage in STAGES:
        sub = [r for r in rows if r["stage"] == stage]
        if not sub: continue
        xa = [float(r["E_apps_j"])/1000 for r in sub]
        ya = [float(r["E_loops_j"])/1000 for r in sub]
        ax.scatter(xa, ya, color=COLORS[stage], label=stage, s=40, alpha=0.7)
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], "k--", alpha=0.4, label="y = x (perfect)")
    ax.plot([0, lim], [0, lim*1.05], "r:", alpha=0.4, label="±5%")
    ax.plot([0, lim], [0, lim*0.95], "r:", alpha=0.4)
    ax.set_xlabel("Apps-level energy:  DC_NODE_POWER × TIME (kJ)")
    ax.set_ylabel("Loops integration:  ∫P(t)dt (kJ)")
    ax.set_title("EAR aggregate vs. sub-second integration", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_aspect("equal")
    plt.tight_layout()
    out = FIGS / "fig_loops_vs_apps.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  wrote {out}")


def fig_phase_check():
    _, by_stage = load_measurements()
    fig, ax = plt.subplots(figsize=(7, 5))
    for stage, srows in by_stage.items():
        cpis = [float(r["cpi"]) for r in srows
                if r.get("cpi") not in (None, "", "0")]
        gflops = [float(r["cpu_gflops"]) for r in srows if r.get("cpu_gflops")]
        if not cpis or not gflops: continue
        ax.scatter(np.mean(cpis), np.mean(gflops),
                   color=COLORS.get(stage, "k"), s=140, label=stage, alpha=0.85)
        ax.annotate(stage.replace('_', '\n'),
                    (np.mean(cpis), np.mean(gflops)),
                    fontsize=8, xytext=(8, 0), textcoords="offset points")
    ax.axvline(1.0, color="gray", ls="--", alpha=0.5, label="CPI=1 (mem-bound threshold)")
    ax.set_xlabel("Mean CPI (cycles per instruction)")
    ax.set_ylabel("Mean CPU_GFLOPS")
    ax.set_title("EAR-derived bound classification\n(cross-check vs roofline regime)",
                 fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    plt.tight_layout()
    out = FIGS / "fig_phase_check.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  wrote {out}")


def main():
    if not MEAS.exists():
        print(f"No {MEAS} — run 03_harvest.sh first"); return
    print("[plot] generating figures ...")
    fig_per_stage_fits()
    fig_ekf_modes()
    fig_loops_vs_apps()
    fig_phase_check()


if __name__ == "__main__":
    main()
