#!/usr/bin/env python3
"""
energy forecast — the "how the team uses what I built" tool.

Before launching a curation run, type the dataset size (and optionally n_tasks /
corpus / hardware) and get the predicted energy of the WHOLE pipeline, with a
confidence band, translated into the numbers a team actually acts on: kWh, euros,
and kg CO2 — plus a per-stage breakdown and a deduplication-OOM warning.

The per-stage law is the calibrated physics  E_stage = c0 + c1 * n  (OLS).
For a corpus we never calibrated, we transfer via the per-CHARACTER constant
(energy/char is ~stable across corpora) so the tool still answers for code, German,
etc. — that is the generalized model in closed form.

Usage:
  python3 forecast.py --n 1400000
  python3 forecast.py --n 400000 --corpus cc_github_opencode --n-tasks 16
  python3 forecast.py --n 100000 --stages data_splitting,string_normalization
"""
import argparse, json, math, sys

FITS = "/Users/hornet/Projects/TNO/paper/data/ols_fits_with_dedup.json"

# measured avg chars/doc per corpus (probe_corpora.py, 2026-06-14) — the transfer key
CORPUS_CHARS = {
    "american_stories":    1798,   # calibration corpus
    "cc_github_opencode":  3151,
    "cc_german_pd":       48881,
    "european_parliament":181075,
}
CAL_CORPUS = "american_stories"
CAL_CHARS = CORPUS_CHARS[CAL_CORPUS]

# pipeline order; toxic_language_detection is filled once its fit is harvested
PIPELINE = ["data_splitting", "string_normalization", "heuristic_filtering",
            "toxic_language_detection", "deduplication"]

DEDUP_OOM_N = 1_400_000      # signature pass OOMs at ~1.4M docs (observed)

# energy -> money/carbon (Netherlands, data-centre; all overridable)
PUE = 1.20                    # facility overhead
EUR_PER_KWH = 0.30
KG_CO2_PER_KWH = 0.30


# toxic-language fit harvested 2026-06-14 from H100 runs (am-stories node energy:
# 100k=20.4kJ, 400k=61.0kJ -> c1=0.135 J/doc, c0=6843 J). Preliminary (2 points).
TOXIC_FIT = {"c1": 0.135, "c0": 6843.0, "sigma": 3000.0}


def load_fits(path):
    d = json.load(open(path))
    fits = {}
    for st, f in d.get("fits", d).items():
        # theta = [c1 (J/doc), c0 (J)]
        th = f.get("theta") or [f.get("marginal_theta", [0])[0], f.get("intercept_j", 0)]
        sig = f.get("sigma2", 0.0)
        fits[st] = {"c1": float(th[0]), "c0": float(th[1]), "sigma": math.sqrt(max(sig, 0))}
    fits.setdefault("toxic_language_detection", TOXIC_FIT)   # H100 stage, harvested
    return fits


def stage_energy_j(fit, n, corpus):
    """Per-stage energy. For non-calibration corpora, transfer via chars/doc:
    c1 is energy/DOC on am-stories -> convert to energy/CHAR -> rescale to this corpus."""
    c1, c0 = fit["c1"], fit["c0"]
    if corpus != CAL_CORPUS and corpus in CORPUS_CHARS:
        c1 = c1 / CAL_CHARS * CORPUS_CHARS[corpus]      # per-char transfer
    return c0 + c1 * n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="number of documents")
    ap.add_argument("--corpus", default="american_stories", choices=list(CORPUS_CHARS))
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--hardware", default="genoa", choices=["genoa", "rome"])
    ap.add_argument("--stages", default=",".join(PIPELINE))
    ap.add_argument("--price", type=float, default=EUR_PER_KWH)
    ap.add_argument("--co2", type=float, default=KG_CO2_PER_KWH)
    ap.add_argument("--fits", default=FITS)
    a = ap.parse_args()

    fits = load_fits(a.fits)
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]

    print(f"\n  Energy forecast — {a.n:,} docs | corpus={a.corpus} "
          f"(~{CORPUS_CHARS[a.corpus]:,} chars/doc) | n_tasks={a.n_tasks} | {a.hardware}")
    print("  " + "=" * 66)
    print(f"  {'stage':<26}{'energy':>12}{'± band':>12}   share")
    print("  " + "-" * 66)

    total_j = total_var = 0.0
    rows = []
    for st in stages:
        if st not in fits:
            print(f"  {st:<26}{'(no fit yet)':>12}")
            continue
        e = stage_energy_j(fits[st], a.n, a.corpus)
        band = 1.96 * fits[st]["sigma"]
        total_j += e; total_var += fits[st]["sigma"] ** 2
        rows.append((st, e, band))
    for st, e, band in rows:
        share = e / total_j * 100 if total_j else 0
        print(f"  {st:<26}{e/1000:>9.1f} kJ{band/1000:>9.1f} kJ{share:>7.0f}%")

    band_j = 1.96 * math.sqrt(total_var)
    # hardware note: coefficients calibrated on genoa; rome differs (per-hardware term)
    hw_note = "" if a.hardware == "genoa" else "  [rome: needs per-hardware factor — calibrate]"
    kwh = total_j / 3.6e6 * PUE
    print("  " + "-" * 66)
    print(f"  {'TOTAL PIPELINE':<26}{total_j/1e6:>9.2f} MJ{band_j/1e6:>9.2f} MJ{hw_note}")
    print("  " + "=" * 66)
    print(f"  =  {kwh:8.2f} kWh   (PUE {PUE})")
    print(f"  =  EUR {kwh*a.price:8.2f}   (@ EUR {a.price}/kWh)")
    print(f"  =  {kwh*a.co2:8.2f} kg CO2  (@ {a.co2} kg/kWh)")

    if "deduplication" in stages and a.n >= DEDUP_OOM_N:
        print(f"\n  WARNING: deduplication OOMs at ~{DEDUP_OOM_N:,} docs (signature pass).")
        print("           This run will FAIL at dedup — shard it or raise memory first.")
    print()


if __name__ == "__main__":
    main()
