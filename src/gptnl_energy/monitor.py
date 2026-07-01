"""
Live energy monitor — terminal dashboard with EKF estimation.

Drives the LiveEstimator in real time as tasks complete, showing:
  - Total energy estimate converging toward truth
  - Shrinking 95% confidence band
  - Real-time per-stage energy readings
  - Innovation gate rejections

Two modes:
  - Replay: drive from recorded CSV data (for demos)
  - Live: SSH to Snellius and poll EAR telemetry (for real runs)

Usage:
    from gptnl_energy.monitor import Monitor

    mon = Monitor.from_fits("paper/data/ols_fits_with_dedup.json")
    mon.run_replay("paper/data/measurements_raw_generalized.csv",
                   run="amg_n1_size400000_rep1")
"""

from __future__ import annotations

import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from gptnl_energy.data import (
    PIPELINE_STAGES,
    STAGE_SHORT,
    CALIBRATION_CORPUS,
    load_fits,
    parse_corpus,
)
from gptnl_energy.ekf import LiveEstimator


# ── Snellius SSH config ───────────────────────────────────────────────────────
SSH_HOST = "rmalik@snellius.surf.nl"
RUNS_REMOTE = "/projects/0/prjs0986/energy_studies/test_romir/runs"


def fetch_live_ear(run: str) -> List[Tuple[str, str, float]]:
    """SSH to Snellius, return [(stage, csv_id, energy_j)] for completed EAR CSVs."""
    # Shell script that finds EAR CSV files and sums per-file energy
    script = rf"""
for f in $(find "{RUNS_REMOTE}/{run}" -path "*emission/ear_db_*_apps.csv" 2>/dev/null); do
    stage=$(echo "$f" | sed -E 's#.*/stage[0-9]+_([^/]+)/.*#\1#')
    b=$(basename "$f")
    e=$(awk -F';' 'NR==1{{for(i=1;i<=NF;i++)c[$i]=i;next}}
        $1=="JOBID"{{next}}
        {{t=$(c["TIME_SEC"]); p=$(c["DC_NODE_POWER_W"]); if(t>0&&p>0)s+=p*t}}
        END{{printf "%.0f", s+0}}' "$f")
    echo "$stage|$b|$e"
done
""".strip()
    try:
        out = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", SSH_HOST, script],
            capture_output=True, text=True, timeout=40,
        ).stdout
    except Exception:
        return []

    results = []
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) == 3 and parts[2].strip():
            try:
                results.append((parts[0], parts[1], float(parts[2])))
            except ValueError:
                pass
    return results


class Monitor:
    """Live energy monitor for pipeline runs."""

    def __init__(self, fits: Dict):
        self._fits = fits

    @classmethod
    def from_fits(cls, fits_path: str) -> "Monitor":
        return cls(load_fits(fits_path))

    def run_replay(
        self,
        raw_csv: str,
        run: str,
        speed: float = 0.7,
        gate_demo: bool = False,
    ):
        """Replay a recorded run from the raw measurements CSV.

        Renders a Rich Live dashboard showing the EKF converging.
        """
        from rich.live import Live
        from rich.console import Console

        df = pd.read_csv(raw_csv)
        df = df[df.slug == run].copy()
        if df.empty:
            print(f"No rows for run '{run}'")
            return

        df["energy_j"] = df.dc_node_power_w.fillna(0) * df.time_sec
        df = df[df.energy_j > 0]

        # Order by pipeline stage
        stage_order = {s: i for i, s in enumerate(PIPELINE_STAGES)}
        df["ord"] = df.stage.map(lambda s: stage_order.get(s, 9))
        df = df.sort_values(["ord", "job_id", "time_sec"]).reset_index(drop=True)

        clean_truth = df.energy_j.sum()

        if gate_demo and len(df) > 3:
            i = int(df["energy_j"].values.argmax())
            df.iloc[i, df.columns.get_loc("energy_j")] *= 6.0

        n = int(df["size"].iloc[0])
        corpus = parse_corpus(run)
        stages_in_run = [s for s in PIPELINE_STAGES if s in set(df.stage)]
        tps = df.groupby("stage").size().to_dict()
        est = LiveEstimator(self._fits, n, corpus, stages_in_run, tps)

        prior_total = est.total_predicted
        hist: List[Tuple[float, float]] = []
        rows: List[Dict] = []
        times: List[float] = []
        tcum = 0.0
        console = Console()
        width = console.width

        with Live(refresh_per_second=12, screen=True) as live:
            # Initial frame: pure prior
            hist.append((prior_total, 1.96 * math.sqrt(sum(est.P0.values()))))
            times.append(0.0)
            live.update(
                self._render(width, run, n, corpus, est, clean_truth,
                            hist, rows, prior_total, times)
            )
            time.sleep(1.2)

            for i, (_, row) in enumerate(df.iterrows(), 1):
                ok, _ = est.update(row.stage, row.energy_j)
                cur, ci = est.estimate()
                hist.append((cur, ci))
                tcum += float(row.time_sec)
                times.append(tcum)
                rows.append({
                    "i": i, "stage": row.stage, "e": row.energy_j,
                    "ok": ok, "est": cur, "ci": ci,
                })
                live.update(
                    self._render(width, run, n, corpus, est, clean_truth,
                                hist, rows, prior_total, times)
                )
                time.sleep(speed)

            # Hold final frame
            try:
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass

        cur, ci = hist[-1] if hist else (0, 0)
        err = abs(cur - clean_truth) / clean_truth * 100 if clean_truth else 0
        print(
            f"\nReplay done: {run}  final estimate {cur/1000:.1f} kJ ± {ci/1000:.1f}  "
            f"truth {clean_truth/1000:.1f} kJ  error {err:.1f}%  "
            f"gated {est.rejected}"
        )

    def run_live_ssh(self, run: str, poll: float = 8.0, timeout: float = 2400):
        """Monitor a live Snellius run via SSH polling."""
        from rich.live import Live
        from rich.console import Console

        import re
        m = re.search(r"size(\d+)", run)
        n = int(m.group(1)) if m else 100000
        corpus = parse_corpus(run)
        stages = ["data_splitting", "string_normalization",
                   "heuristic_filtering", "deduplication"]
        est = LiveEstimator(self._fits, n, corpus, stages, {s: 1 for s in stages})
        prior_total = est.total_predicted

        hist: List[Tuple[float, float]] = []
        rows: List[Dict] = []
        times: List[float] = []
        fed: set = set()
        idle = 0
        t0 = time.time()
        console = Console()
        width = console.width

        with Live(refresh_per_second=8, screen=True) as live:
            while time.time() - t0 < timeout:
                new = 0
                for stage, cid, energy in fetch_live_ear(run):
                    if (stage, cid) in fed or stage not in est.pred:
                        continue
                    ok, _ = est.update(stage, energy)
                    fed.add((stage, cid))
                    est.M[stage] = len(est.seen[stage])
                    cur, ci = est.estimate()
                    hist.append((cur, ci))
                    times.append(time.time() - t0)
                    rows.append({
                        "i": len(fed), "stage": stage, "e": energy,
                        "ok": ok, "est": cur, "ci": ci,
                    })
                    new += 1

                if not hist:
                    hist.append((prior_total, 1.96 * math.sqrt(sum(est.P0.values()))))
                    times.append(time.time() - t0)

                live.update(
                    self._render(width, run, n, corpus, est, None,
                                hist, rows, prior_total, times)
                )

                seen_stages = {r["stage"] for r in rows}
                idle = idle + 1 if new == 0 else 0
                if seen_stages >= set(stages) and idle >= 3:
                    break
                time.sleep(poll)

            try:
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass

        cur, ci = hist[-1] if hist else (0, 0)
        print(
            f"\nLIVE done: {run}  final estimate {cur/1000:.1f} kJ ± {ci/1000:.1f}  "
            f"({len(fed)} readings, {est.rejected} gated)"
        )

    def run_test(self, raw_csv: str, run: str):
        """Quick test: run the EKF silently and print final stats."""
        df = pd.read_csv(raw_csv)
        df = df[df.slug == run].copy()
        df["energy_j"] = df.dc_node_power_w.fillna(0) * df.time_sec
        df = df[df.energy_j > 0]

        n = int(df["size"].iloc[0])
        corpus = parse_corpus(run)
        stages_in_run = [s for s in PIPELINE_STAGES if s in set(df.stage)]
        tps = df.groupby("stage").size().to_dict()
        est = LiveEstimator(self._fits, n, corpus, stages_in_run, tps)

        truth = df.energy_j.sum()
        for _, row in df.iterrows():
            est.update(row.stage, row.energy_j)

        cur, ci = est.estimate()
        err = abs(cur - truth) / truth * 100 if truth else 0
        print(f"run={run} n={n:,} corpus={corpus}")
        print(f"  pre-run prediction : {est.total_predicted/1000:8.1f} kJ")
        print(f"  final estimate     : {cur/1000:8.1f} kJ  ± {ci/1000:.1f}")
        print(f"  measured truth     : {truth/1000:8.1f} kJ")
        print(f"  final error        : {err:.1f}%   gate rejects={est.rejected}")

    def _render(self, width, run, n, corpus, est, truth, hist, rows,
                prior_total, times):
        """Build a Rich layout for the live dashboard."""
        import plotext as plt
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
        from rich.layout import Layout
        from rich.console import Group

        side_w = 46
        graph_w = max(54, width - side_w - 8)

        # X-axis: elapsed time or reading index
        if times and len(times) == len(hist):
            xs = [t / 60.0 for t in times]
            xlab = "elapsed time (min)"
        else:
            xs = list(range(1, len(hist) + 1))
            xlab = "stages completed"

        e_vals = [h[0] / 1000 for h in hist]
        up = [(h[0] + h[1]) / 1000 for h in hist]
        lo = [(h[0] - h[1]) / 1000 for h in hist]

        # Graph 1: estimate over time
        plt.clf()
        plt.theme("pro")
        plt.plotsize(graph_w, 14)
        plt.plot(xs, [prior_total / 1000] * len(xs), color="white")
        if truth is not None:
            plt.plot(xs, [truth / 1000] * len(xs), color="green+")
        plt.plot(xs, up, color="magenta")
        plt.plot(xs, lo, color="magenta")
        plt.plot(xs, e_vals, color="cyan+", marker="braille")
        plt.xlabel(xlab)
        plt.ylabel("kJ")
        graph1 = Text.from_ansi(plt.build())

        # Graph 2: per-stage measured energy
        order = [s for s in PIPELINE_STAGES if s in est.seen and est.seen[s]]
        labels = [STAGE_SHORT.get(s, s[:6]) for s in order]
        vals = [sum(est.seen[s]) / 1000 for s in order]
        plt.clf()
        plt.theme("pro")
        plt.plotsize(graph_w, 9)
        if vals:
            plt.bar(labels, vals, color="orange", width=0.4)
        plt.ylabel("kJ")
        graph2 = Text.from_ansi(plt.build())

        # Stats panel
        cur, ci = hist[-1]
        ciw = ci / cur * 100 if cur else 0
        stats = Table(show_header=False, box=None, pad_edge=False)
        stats.add_column(style="bold cyan", no_wrap=True)
        stats.add_column(no_wrap=True)
        stats.add_row("run", f"[yellow]{run}[/]")
        stats.add_row("docs", f"{n:,}  [dim]{corpus}[/]")
        stats.add_row("predicted", f"{prior_total/1000:7.1f} kJ  [dim](pre-run)[/]")
        stats.add_row("estimate", f"[bold cyan]{cur/1000:7.1f} kJ[/] ± {ci/1000:.0f}")
        if truth is not None:
            err = abs(cur - truth) / truth * 100 if truth else 0
            stats.add_row("truth", f"[bold green]{truth/1000:7.1f} kJ[/]")
            stats.add_row("error", f"[{'bold green' if err < 10 else 'yellow'}]{err:.1f}%[/]")
        else:
            meas = est.total_observed
            stats.add_row("status", "[bold green]● LIVE[/]")
            stats.add_row("measured", f"[green]{meas/1000:7.1f} kJ[/] [dim]so far[/]")
        stats.add_row("CI width", f"{ciw:.0f}%")
        stats.add_row("readings", f"{len(hist)}   [dim]gated[/] [red]{est.rejected}[/]")

        # Streaming readings table
        tbl = Table(show_header=True, header_style="bold", box=None,
                     expand=True, pad_edge=False)
        for col, just in (("#", "right"), ("stage", "left"), ("kJ", "right"),
                           ("gate", "center"), ("est", "right"), ("±", "right")):
            tbl.add_column(col, justify=just, no_wrap=True)
        for r in rows[-10:]:
            g = "[green]✓[/]" if r["ok"] else "[red]✗REJ[/]"
            tbl.add_row(
                str(r["i"]),
                STAGE_SHORT.get(r["stage"], r["stage"][:9]),
                f"{r['e']/1000:.1f}",
                g,
                f"{r['est']/1000:.0f}",
                f"{r['ci']/1000:.0f}",
            )

        layout = Layout()
        layout.split_column(Layout(name="head", size=3), Layout(name="body"))
        layout["head"].update(Panel(Text(
            " GPT-NL PIPELINE — LIVE ENERGY ESTIMATION  "
            "(per-task EKF: predict → calibrate as it runs)",
            style="bold white on blue",
        )))
        layout["body"].split_row(Layout(name="g"), Layout(name="side", size=side_w))
        layout["g"].split_column(Layout(name="g1"), Layout(name="g2", size=11))
        layout["g"]["g1"].update(Panel(
            Group(graph1, Text.from_markup(
                "  [cyan]━ estimate[/]  [magenta]━ 95% CI[/]  "
                "[white]━ pre-run prediction[/]  "
                + ("[green]━ measured truth[/]" if truth is not None
                   else "[dim](truth at completion)[/]")
            )),
            title="predicted energy vs time  (estimate → truth, CI shrinking)",
            border_style="cyan",
        ))
        layout["g"]["g2"].update(Panel(
            graph2,
            title="real energy readings from EAR (measured kJ per stage)",
            border_style="orange3",
        ))
        layout["side"].split_column(
            Panel(stats, title="state", border_style="yellow"),
            Panel(tbl, title="streaming readings", border_style="white"),
        )
        return layout
