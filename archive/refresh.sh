#!/usr/bin/env bash
# One command to refresh everything from the latest Snellius runs:
#   harvest -> pull -> retrain coefficient model g -> rebuild the notebook.
# Needs the SSH master session alive (open `ssh snellius.surf.nl` in a terminal first).
set -e
cd "$(dirname "$0")"
HOST=rmalik@snellius.surf.nl
WS=/projects/0/prjs0986/energy_studies/test_romir

echo "[1/4] harvesting on Snellius ..."
ssh -o BatchMode=yes -o ConnectTimeout=30 "$HOST" \
  "cd $WS/scripts && source env.sh >/dev/null 2>&1 && bash 03_harvest.sh 2>&1 | grep -E 'Aggregated|Wrote'"

echo "[2/4] pulling measurements ..."
scp -o BatchMode=yes "$HOST:$WS/measurements/measurements.csv"     paper/data/measurements_generalized.csv
scp -o BatchMode=yes "$HOST:$WS/measurements/measurements_raw.csv" paper/data/measurements_raw_generalized.csv

echo "[3/4] retraining coefficient model g (leave-one-corpus-out) ..."
python3 coefficient_model.py paper/data/measurements_generalized.csv | tail -20
python3 train_g.py          paper/data/measurements_generalized.csv

echo "[4/4] rebuilding the notebook with fresh data ..."
python3 build_notebook.py
python3 -m nbconvert --to notebook --execute --inplace GPTNL_Energy_Explained.ipynb --ExecutePreprocessor.timeout=240
echo "DONE — GPTNL_Energy_Explained.ipynb refreshed."
