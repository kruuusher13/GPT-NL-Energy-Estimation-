#!/usr/bin/env bash
# Shared environment for all v2 scripts. Source me, don't run me.
#   source env.sh
#
# Override any of these by exporting before sourcing, e.g.
#   DATA_DIR=/projects/prjs0986/wp12/extracted/american-stories source env.sh

# ── Layout ─────────────────────────────────────────────────────────────────────
export WS_V2="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CONFIGS_DIR="$WS_V2/configs"
export RUNS_DIR="$WS_V2/runs"
export MEAS_DIR="$WS_V2/measurements"
export FIGS_DIR="$WS_V2/figures"

# ── Data ───────────────────────────────────────────────────────────────────────
# Default: ../data/american-stories relative to ws_v2. Override for Snellius:
export DATA_DIR="${DATA_DIR:-/projects/0/prjs0986/energy_studies/datasets/american-stories}"

# ── Sweep grid ─────────────────────────────────────────────────────────────────
# v2 CALIBRATION sweep (22 May): the 1k–100k sweep showed 3/4 stages are
# overhead-bound — jobs finish in seconds, fixed per-job cost (~5–15 s) swamps
# the per-document work, so the marginal-energy slope c1 is unidentifiable.
# This grid is log-spaced up to the full american-stories corpus (~1.4M docs)
# so the work clears the overhead and c1 becomes measurable. The 1k point is
# kept as the fixed-cost (c0) anchor.
export SIZES=(1000 100000 400000 1400000)
export REPLICATES=2
export STAGES=(data_splitting string_normalization heuristic_filtering deduplication)
export PARTITION="${PARTITION:-genoa}"

# ── SLURM defaults ─────────────────────────────────────────────────────────────
# Time per stage. Loose maxima — bumped for the full-corpus calibration sweep.
export TIME_DATA_SPLITTING="01:00:00"
export TIME_STRING_NORM="02:00:00"
export TIME_HEURISTIC="03:00:00"
export TIME_DEDUP="03:00:00"

# Tasks per stage. CALIBRATION runs single-task (=1): the 4-task fan-out gave
# badly load-imbalanced nodes (one node did ~all the work, three idled) plus
# cross-node aggregation noise — and buys nothing for calibration since energy
# is additive. One task = one clean (energy, time) reading per stage.
export NTASKS_DATA_SPLITTING=1
export NTASKS_STRING_NORM=1
export NTASKS_HEURISTIC=1
export NTASKS_DEDUP=1

# Memory per CPU (GB).
export MEM_PER_CPU=1
export MEM_PER_CPU_DEDUP=1

# ── EAR ────────────────────────────────────────────────────────────────────────
export HPC_EAR=true

# ── Pipeline repo ──────────────────────────────────────────────────────────────
# data-curation-pipeline now lives inside ws_v2 (self-contained, no external refs).
# Modules sit at ws_v2/data-curation-modules — pyproject.toml uses ../data-curation-modules
# relative paths so they resolve automatically.
export PIPELINE_REPO="$WS_V2/data-curation-pipeline"

echo "[env] WS_V2=$WS_V2"
echo "[env] DATA_DIR=$DATA_DIR"
echo "[env] PARTITION=$PARTITION"
echo "[env] PIPELINE_REPO=$PIPELINE_REPO"
