#!/usr/bin/env bash
# ISC-EAR technique: sub-second loops data via `eacct -r`, integrate ∫P(t)dt,
# compare to apps-level DC_NODE_POWER_W * TIME_SEC.
#
# If the apps-level aggregate diverges from the integrated loop data by more
# than ~5%, EAR's reported average power was probably wrong (sampling missed a
# spike or trough). That's a measurement-integrity check straight out of the
# ISC-EAR tutorial.
#
# Reads:  measurements.csv (for job IDs)
# Writes: measurements/loops_validation.csv (job_id, stage, E_apps, E_loops, pct_diff)
#
# Usage:
#   source env.sh
#   bash 06_loops_integrate.sh

set -euo pipefail
source "$(dirname "$0")/env.sh"

MEAS="$MEAS_DIR/measurements.csv"
OUT="$MEAS_DIR/loops_validation.csv"
[ -f "$MEAS" ] || { echo "Run 03_harvest.sh first."; exit 1; }

echo "job_id,stage,size,replicate,E_apps_j,E_loops_j,pct_diff" > "$OUT"

# Skip header, iterate rows
tail -n +2 "$MEAS" | while IFS=',' read -r slug size rep stage jobid rest; do
  [ -z "$jobid" ] || [ "$jobid" = "JOBID" ] && continue

  loops_csv=$(mktemp)
  if ! eacct -j "$jobid" -r -c "$loops_csv" > /dev/null 2>&1; then
    rm -f "$loops_csv"; continue
  fi

  # Integrate via trapezoidal rule on (TIME, DC_NODE_POWER_W) loop samples.
  # Loop CSV has many rows per job. Compute apps-level energy from the same row's
  # TIME_SEC * DC_NODE_POWER_W in the apps CSV — but we already have that in
  # measurements.csv, so just reuse.
  E_loops=$(awk -F';' '
    NR == 1 {
      for (i=1;i<=NF;i++) col[$i]=i
      next
    }
    NR > 1 && $col["DC_NODE_POWER_W"] != "" {
      t = $col["TIME_SEC"] + 0
      p = $col["DC_NODE_POWER_W"] + 0
      if (last_t > 0 && t > last_t) {
        # trapezoid: 0.5 * (p_prev + p_cur) * (t_cur - t_prev)
        E += 0.5 * (last_p + p) * (t - last_t)
      }
      last_t = t; last_p = p
    }
    END { print E }
  ' "$loops_csv")

  rm -f "$loops_csv"

  # apps-level energy was harvested earlier — recompute from current row.
  # measurements.csv columns: 1=slug 2=size 3=replicate 4=stage 5=jobid
  #                           6=nodename 7=partition 8=time_sec 9=dc_node_power_w
  E_apps=$(awk -F',' -v j="$jobid" -v s="$stage" '
    $5 == j && $4 == s { print $8 * $9; exit }
  ' "$MEAS")

  if [ -z "$E_apps" ] || [ -z "$E_loops" ] || (( $(echo "$E_apps == 0" | bc -l 2>/dev/null || echo 0) )); then
    continue
  fi

  pct=$(echo "scale=2; ($E_loops - $E_apps) / $E_apps * 100" | bc -l 2>/dev/null || echo "")
  printf "%s,%s,%s,%s,%.1f,%.1f,%s\n" "$jobid" "$stage" "$size" "$rep" "$E_apps" "$E_loops" "$pct" >> "$OUT"
done

echo "[loops] Wrote $OUT"
echo "[loops] Summary (rows where pct_diff > 5%):"
awk -F',' 'NR>1 && ($7+0 > 5 || $7+0 < -5)' "$OUT" || true
