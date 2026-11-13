#!/usr/bin/env bash
# Diagnostic: print the EAR CSV header for every app-level CSV under $RUNS_DIR.
# Use this when 03_harvest.sh aborts because a required column is missing —
# you'll see the actual column names your EAR build is writing and can patch
# the REQUIRED_COLS list in 03_harvest.sh accordingly.
#
# Usage:
#   source env.sh
#   bash 97_inspect_ear.sh
#   bash 97_inspect_ear.sh /path/to/one.csv     # specific file
#   bash 97_inspect_ear.sh --row                # also print the first data row

set -euo pipefail
source "$(dirname "$0")/env.sh"

show_row=false
single_file=""
for arg in "$@"; do
  case "$arg" in
    --row) show_row=true ;;
    *) single_file="$arg" ;;
  esac
done

files=()
if [ -n "$single_file" ]; then
  files=("$single_file")
else
  while IFS= read -r line; do
    [ -n "$line" ] && files+=("$line")
  done < <(find "$RUNS_DIR" -path "*/emission/ear_db_*_apps.csv" 2>/dev/null | sort -u)
  if [ "${#files[@]}" -eq 0 ]; then
    while IFS= read -r line; do
      [ -n "$line" ] && files+=("$line")
    done < <(find "$RUNS_DIR" -path "*/emission/ear_db_*.csv" 2>/dev/null \
             | grep -v '_loops\.csv$' | sort -u)
  fi
fi

if [ "${#files[@]}" -eq 0 ]; then
  echo "No EAR CSVs found under $RUNS_DIR" >&2
  exit 1
fi

echo "── Found ${#files[@]} EAR CSV(s) ──"
echo

for f in "${files[@]}"; do
  echo "File: $f"
  ncols=$(head -n 1 "$f" | awk -F';' '{print NF}')
  nrows=$(( $(wc -l < "$f") - 1 ))
  echo "  columns: $ncols   data rows: $nrows"
  echo "  header (one per line):"
  head -n 1 "$f" | tr ';' '\n' | nl -ba | sed 's/^/    /'
  if $show_row && [ "$nrows" -ge 1 ]; then
    echo "  first data row (one per line, paired with header):"
    paste <(head -n 1 "$f" | tr ';' '\n') <(sed -n '2p' "$f" | tr ';' '\n') | nl -ba | sed 's/^/    /'
  fi
  echo
done
