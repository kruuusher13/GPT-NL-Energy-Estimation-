#!/usr/bin/env python3
"""Patch 02_monitor.sh: eacct -c is headerless, so DictReader misparses it and
the EKF never updates. Replace the name-based parse with positional parsing of
the real EAR measurement row (STEP_ID=0): col26=TIME_SEC, col32=DC_NODE_POWER_W
(1-based) => indices 25 and 31."""
import sys

path = sys.argv[1]
s = open(path).read()

old = '''with open(csv_path) as f:
    row = next(csv.DictReader(f, delimiter=';'), None)
if not row:
    sys.exit(0)

E_obs = float(row.get("DC_NODE_POWER_W", 0) or 0) * float(row.get("TIME_SEC", 0) or 0)
if E_obs <= 0:
    sys.exit(0)'''

new = '''# eacct -c is HEADERLESS: parse by position. The real measurement row has
# STEP_ID (col2) == "0"; the pseudo-step "4294967291" carries only zeros.
# col26 = TIME_SEC, col32 = DC_NODE_POWER_W  (1-based).
E_obs = 0.0
with open(csv_path) as f:
    for parts in csv.reader(f, delimiter=';'):
        if len(parts) < 33:
            continue
        if parts[1].strip() in ("", "4294967291"):
            continue
        try:
            t = float(parts[25]); p = float(parts[31])
        except ValueError:
            continue
        if p > 0 and t > 0 and p * t > E_obs:
            E_obs = p * t
if E_obs <= 0:
    sys.exit(0)'''

n = s.count(old)
if n == 0:
    print("ERROR: target block not found (already patched?)")
    sys.exit(1)
open(path, "w").write(s.replace(old, new))
print(f"patched {n} block(s) in {path}")
