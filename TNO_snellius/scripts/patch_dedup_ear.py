#!/usr/bin/env python3
"""
Fix dedup producing no EAR energy CSV.

Root cause: the 4 hand-built dedup sub-executors (signature/buckets/cluster/filter)
in stages/deduplication.py set sbatch_args ear-user-db -> {logs}/emission/ear_db,
but — unlike the regular Stage.run_stage in stage.py (line ~292) — their
env_command never does `mkdir -p {logs}/emission`. EAR then has nowhere to write
its user-db and silently emits nothing. The 3 regular stages have the mkdir and DO
produce CSVs; dedup lacks it and does NOT. This injects the same mkdir into all 4.

Idempotent. Writes a .bak_dedupear backup on first run.
"""
import sys
import pathlib

TARGET = pathlib.Path(
    "data-curation-pipeline/gptnl_data_curation_pipeline/stages/deduplication.py"
)

NEEDLE = (
    '                f"source {venv_path}; export OMP_NUM_THREADS={self.hpc_cpus_per_task}; export PYTHONFAULTHANDLER=1; "\n'
    '                f"{self.env_commands}",'
)
INJECT = (
    '                f"source {venv_path}; export OMP_NUM_THREADS={self.hpc_cpus_per_task}; export PYTHONFAULTHANDLER=1; "\n'
    "                f\"mkdir -p {self.logs_folder / 'emission'}; \"\n"
    '                f"{self.env_commands}",'
)


def main():
    src = TARGET.read_text()
    if INJECT in src and NEEDLE not in src:
        print("[patch-dedup-ear] already patched, nothing to do.")
        return 0
    n = src.count(NEEDLE)
    if n != 4:
        print(f"[patch-dedup-ear] EXPECTED 4 dedup env_command blocks, found {n}. "
              "Aborting without writing so nothing is corrupted.", file=sys.stderr)
        return 2
    bak = TARGET.with_suffix(".py.bak_dedupear")
    if not bak.exists():
        bak.write_text(src)
        print(f"[patch-dedup-ear] backup -> {bak}")
    TARGET.write_text(src.replace(NEEDLE, INJECT))
    print(f"[patch-dedup-ear] injected `mkdir -p .../emission` into {n} dedup "
          "executors. EAR will now have a directory to write its user-db.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
