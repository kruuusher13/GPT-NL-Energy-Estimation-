#!/usr/bin/env bash
# Submit every generated YAML to the SLURM queue via `poetry run pipeline`.
#
# Records: $MEAS_DIR/submissions.tsv  (slug, submit_time_utc, raw_pipeline_stdout)
# We rely on harvest_all_runs.py (later) to recover job IDs via sacct from the
# submission timestamp + JobName, which is what the v1 flow already does.
#
# Usage:
#   source env.sh
#   bash 01_submit_all.sh             # submit all configs
#   bash 01_submit_all.sh size1000_rep1   # submit one config by slug
#
# Pre-req: you've run 00_gen_configs.sh, and `poetry run pipeline` works in
# $PIPELINE_REPO.

set -euo pipefail
source "$(dirname "$0")/env.sh"

mkdir -p "$MEAS_DIR" "$RUNS_DIR"
LOG="$MEAS_DIR/submissions.tsv"
[ -f "$LOG" ] || echo -e "slug\tsubmit_utc\tconfig_path\treturncode" > "$LOG"

submit_one() {
  local cfg="$1"
  local slug
  slug=$(basename "$cfg" .yaml)
  slug=${slug#run_}

  local utc
  utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)

  echo "[submit] $slug  ($utc)"
  pushd "$PIPELINE_REPO" > /dev/null

  set +e
  poetry run pipeline -s "$cfg" 2>&1 | tee "$MEAS_DIR/submit_${slug}.log"
  local rc=${PIPESTATUS[0]}
  set -e

  popd > /dev/null

  printf "%s\t%s\t%s\t%d\n" "$slug" "$utc" "$cfg" "$rc" >> "$LOG"

  if [ "$rc" -ne 0 ]; then
    echo "  WARN: poetry run pipeline exited $rc — see $MEAS_DIR/submit_${slug}.log"
  fi
}

if [ "$#" -ge 1 ]; then
  cfg="$CONFIGS_DIR/run_${1}.yaml"
  [ -f "$cfg" ] || { echo "No such config: $cfg"; exit 1; }
  submit_one "$cfg"
else
  shopt -s nullglob
  cfgs=("$CONFIGS_DIR"/run_*.yaml)
  [ ${#cfgs[@]} -gt 0 ] || { echo "No configs found in $CONFIGS_DIR — run 00_gen_configs.sh first."; exit 1; }
  echo "[submit] Firing ${#cfgs[@]} runs ..."
  for cfg in "${cfgs[@]}"; do
    submit_one "$cfg"
  done
fi

echo "[submit] Done. Log: $LOG"
echo "[submit] Watch:    watch -n 60 squeue -u \$USER"
echo "[submit] Next:     bash 02_monitor.sh  (live EKF) or wait, then bash 03_harvest.sh"
