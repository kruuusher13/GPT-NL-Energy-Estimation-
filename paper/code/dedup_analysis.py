#!/usr/bin/env python3
"""Deduplication stage energy: real harvested measurements.

Source: EAR `eacct` database on Snellius, account tnsr72764 (user rmalik), the
clean *exclusive* dedup sweep of 2026-05-22 (DC node power ~277-302 W). The dedup
stage runs as four chained sub-jobs (signature, buckets, cluster, filter); each
sub-job emits its own EAR record. Sizes were enabled by D. Kravchuk's (dkravchuk)
scaled American Stories datasets in the shared project space.

Job->size mapping is by the run dirs' slurm logs under
/projects/0/prjs0986/energy_studies/test_romir/runs/size*_rep*/.../stage4_deduplication.
The 1.4M-document run is OUT_OF_MEMORY in the signature pass (jobs 23035638,
23035656; State=FAILED, exit 0:125) -- a real scaling limit, not a missing record.

We report clean energy E = P_dc * t per sub-job (exclusive => power is clean),
which equals the time-driven estimate E = P_eff * t to within the replicate noise.
"""
import numpy as np

# (size, rep) -> list of (subjob, wall_s, dc_power_W) from the harvested eacct rows
RUNS = {
    (1000, 1):   [("signature", 5.257, 277.7), ("buckets", 13.28, 293.1), ("cluster", 7.75, 321.8), ("filter", 2.66, 363.6)],
    (1000, 2):   [("signature", 21.120, 279.4), ("buckets", 11.32, 275.9), ("cluster", 4.30, 289.4), ("filter", 4.88, 160.7)],
    (100000, 1): [("signature", 91.139, 276.2), ("buckets", 15.67, 263.7), ("cluster", 9.80, 238.9), ("filter", 13.07, 257.8)],
    (100000, 2): [("signature", 91.061, 283.0), ("buckets", 11.47, 310.1), ("cluster", 6.80, 213.4), ("filter", 14.83, 263.2)],
    (400000, 1): [("signature", 369.980, 277.5), ("buckets", 14.89, 258.8), ("cluster", 6.90, 216.1), ("filter", 54.02, 283.8)],
    (400000, 2): [("signature", 354.577, 302.3), ("buckets", 14.89, 258.8), ("cluster", 6.90, 216.1), ("filter", 54.02, 283.8)],  # rep2 sig measured; buc/clu/fil = rep1 (not separately logged)
}
OOM = {1400000: "signature OUT_OF_MEMORY (jobs 23035638, 23035656)"}

def energy_kJ(subjobs):
    return sum(t * p for _, t, p in subjobs) / 1000.0

# per-run dedup totals
per_size = {}
print("== per-run dedup energy (clean exclusive, E = P_dc * t) ==")
for (sz, rep), subs in sorted(RUNS.items()):
    e = energy_kJ(subs)
    sig = next(t for n, t, p in subs if n == "signature")
    sig_share = (next(t * p for n, t, p in subs if n == "signature")) / (e * 1000) * 100
    per_size.setdefault(sz, []).append(e)
    print(f"  {sz:>7} rep{rep}: {e:6.1f} kJ   (signature {sig:6.1f}s, {sig_share:4.1f}% of stage)")

sizes = sorted(per_size)
E = np.array([np.mean(per_size[s]) for s in sizes])
N = np.array(sizes, float)
print("\n== per-size mean dedup energy ==")
for s, e in zip(sizes, E):
    print(f"  {s:>7}: {e:6.1f} kJ")
for s, why in OOM.items():
    print(f"  {s:>7}: FAILED - {why}")

# in-sample linear fit E = c0 + c1*n
A = np.vstack([N, np.ones_like(N)]).T
(c1, c0), *_ = np.linalg.lstsq(A, E * 1000, rcond=None)
pred = A @ np.array([c1, c0])
ss_res = np.sum((E * 1000 - pred) ** 2); ss_tot = np.sum((E * 1000 - np.mean(E * 1000)) ** 2)
r2 = 1 - ss_res / ss_tot
print(f"\n== in-sample fit ==  c0 = {c0:,.0f} J   c1 = {c1:.3f} J/doc   R^2 = {r2:.4f}")

# held-out: calibrate {1k,100k} -> predict 400k
cal = [0, 1]; tst = [2]
Ac = A[cal]; (s1, s0), *_ = np.linalg.lstsq(Ac, (E * 1000)[cal], rcond=None)
ph = Ac @ [s1, s0]
for i in tst:
    p = s1 * N[i] + s0
    mre = abs((E * 1000)[i] - p) / (E * 1000)[i] * 100
    print(f"== held-out ==  calibrate {{1k,100k}} -> predict {int(N[i])}: pred {p/1000:.1f} kJ vs meas {E[i]:.1f} kJ  (MRE {mre:.1f}%)")

# marginal cost between consecutive sizes
print("== marginal dE/dn ==")
for i in range(len(sizes) - 1):
    dm = (E[i + 1] - E[i]) * 1000 / (N[i + 1] - N[i])
    print(f"  {sizes[i]:>7} -> {sizes[i+1]:>7}: {dm:.3f} J/doc")

print(f"\n== clean 100k dedup = {np.mean(per_size[100000]):.1f} kJ "
      f"(vs contaminated single-run 63.5 kJ; co-tenancy over-read ~{63.5/np.mean(per_size[100000]):.1f}x) ==")
