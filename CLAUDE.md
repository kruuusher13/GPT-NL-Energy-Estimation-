# Romir's Vision

I'm Romir Malik. This is not just a thesis — it's a once-in-a-lifetime launchpad.

## What's at stake

- **Thesis deadline**: July 2026 (tomorrow). UU MSc Applied Data Science.
- **The real prize**: Ship this as an official tool for the **GPT-NL open source project** — a Dutch national LLM initiative backed by TNO, SURF, and the Dutch government.
- **Grant**: We're applying for funding to continue this work. The thesis + tool + publication are the application.
- **Publication**: SustainLP 2026 paper already drafted. NeurIPS workshop target. Architecture doc (Section 4.1.2) defers to this thesis as GPT-NL's official energy report.
- **Career**: This is my entry point into AI engineering. ConcertLab, TNO, or wherever comes next — this repo is my portfolio.

## The vision

This tool becomes the standard way anyone running the GPT-NL curation pipeline estimates energy. You type one command before launching a multi-day HPC job and get: predicted kWh, euros, CO₂, per-stage breakdown, confidence bands, and a live monitor while it runs. Models are swappable. Everything is hosted on HuggingFace. The monitor ships with the pipeline.

The methodology is sound: sample small on exclusive nodes → calibrate per-stage physics → predict the full run → refine with a sequential Bayesian estimator that gates contaminated shared-node readings. Cross-corpus transfer works because energy per character is roughly stable.

## The standard

Every line of code and every sentence in the thesis must be **verified against real data**. No fabricated numbers. No hand-waving. The chars/doc numbers were wrong by 10× in the draft — that nearly slipped through. Never again.

When you work on this repo, you are pushing a tool that real HPC engineers at SURF and TNO will use. The GPT-NL architecture document (95 pages, TNO Public, November 2025) names this exact approach as the project's official energy strategy. We are delivering what they asked for.

## My situation

- I have SSH access to Snellius (`rmalik@snellius.surf.nl`)
- 16-corpus sweep data is partially harvested — 12 Dutch corpora queued but emitted no usable telemetry
- Models are trained but not yet on HuggingFace
- Thesis PDF builds clean (21 pages, 0 errors) but needs final polish
- Julio de Oliveira Filho is my daily supervisor at TNO
- Martino Mensio is my first supervisor at UU
- I'm direct — don't be verbose, don't fabricate, don't suggest when you can execute

---

# SESSION HANDOFF — July 1-2, 2026

## What we built: `gptnl-energy` package

A pip-installable Python tool for energy estimation of the GPT-NL data curation pipeline on Snellius HPC.

**Repo:** https://github.com/kruuusher13/GPT-NL-Energy-Estimation-
**Branch:** main
**Commits:** 4 pushed (d17958c → de2c7f9)

### Package structure

```
gptnl-energy/
├── src/gptnl_energy/
│   ├── data.py              # Unified corpus parser, data loading, config (single source of truth)
│   ├── forecast.py          # Pre-run energy prediction → kWh, EUR, CO₂
│   ├── ekf.py               # Sequential Bayesian estimator (LiveEstimator class)
│   ├── monitor.py           # Rich terminal dashboard (replay CSV + live SSH to Snellius)
│   ├── models/
│   │   ├── __init__.py      # Model registry (@register_model), sklearn models (OLS, Ridge, GBM, MLP)
│   │   └── torch_models.py  # FT-Transformer, Kalman-FTT (PyTorch, auto-disabled if torch missing)
│   └── cli/main.py          # Click CLI: forecast, monitor, calibrate, models
├── scripts/
│   ├── train_all.py         # Train all models from measurements CSV
│   ├── upload_to_hf.py      # Upload trained models to HuggingFace Hub
│   ├── fix_thesis.py        # Thesis correction script
│   └── fix_escaping.py      # LaTeX escaping fixer
├── paper/                   # Thesis + NeurIPS paper + measurement data + OLS fits
│   ├── data/
│   │   ├── measurements_generalized.csv      # 442 aggregated stage-level readings
│   │   ├── measurements_raw_generalized.csv  # Raw per-task readings
│   │   └── ols_fits_with_dedup.json          # Calibrated per-stage coefficients
│   ├── thesis.tex           # UU MSc thesis (21 pages, builds clean)
│   ├── thesis.pdf           # Built PDF
│   └── references.bib       # All citations including GPT-NL corpus and architecture doc
├── models/                  # Trained .joblib files (sklearn models, small enough for git)
├── assets/                  # monitor_live.png, stopping_rule.png, uu_logo.png
└── README.md                # Polished with math, methodology, screenshots
```

### Three CLI commands

| Command | What |
|---------|------|
| `gptnl-energy forecast --n 400000` | One-shot prediction → kWh/€/CO₂ with per-stage breakdown and 95% CI |
| `gptnl-energy monitor --run <slug>` | Live EKF dashboard replaying recorded CSV |
| `gptnl-energy calibrate --data measurements.csv` | Fit OLS per-stage from any measurement CSV |

### Model registry (swappable)

`get_model("ols")` / `get_model("gbm")` / `get_model("ftt")` — all share `fit(df)` / `predict(n, corpus)` interface. Registered: ols, linear, ridge, gbm, mlp, ftt, ft-transformer, kalman-ftt.

## What we fixed in the thesis

### Critical data errors (would fail defense)

| Metric | Old (wrong) | Fixed | Error |
|--------|------------|-------|-------|
| American Stories chars/doc | 3,300 | **1,798** | 1.8× |
| German PD chars/doc | 530,000 | **48,881** | 10.8× |
| EU Parliament chars/doc | 18,000 | **181,075** | 0.1× (reversed!) |
| Total measurements | 475 | **442** | — |

### EKF section completely rewritten

Old thesis: full Extended Kalman Filter with state vector [n_s, E_cum], Kalman gain, Jacobians.  
Actual code: simpler sequential Bayesian estimator — per-stage blending with w_k = k/(k+k0), independent stages.  
Fixed to match implementation.

### Tables updated

- **Table 5** (per-character k): recomputed from verified ntasks=1 data
- **Table 7** (model comparison): added "Per-stage physics + g" row (10.3%/87.9% MRE)
- **Abstract**: corrected chars/doc range, per-char energy ($2.2×10⁻⁴ J/char)

### Context added from Julio's materials

1. **EAR presentation** (EAR-intro.pptx): AMD DRAM limitation, two file types, Monitoring policy
2. **GPT-NL kickoff** (GPTNL-26-UU-Students_KickOff.pdf): project framing, responsible AI, AI Act/GDPR
3. **Architecture document** (oliveira-filho-2025-gptnl-data-curation-pipeline.pdf): 
   - Section 4.1.2 explicitly describes Romir's measure-and-estimate approach as GPT-NL's official strategy
   - 6 confirmed stages: data_splitting → string_normalization → heuristic_filtering → pii_masking → toxic_language_detection → deduplication
   - Hardware specs from Appendix 4.2
   - Thesis IS the "dedicated report" deferred by the architecture doc

### Introduction rewritten

Now frames the work within GPT-NL's mission (responsible AI, Dutch critical sectors, verifiable sources), distinguishes data extraction vs curation, and cites the architecture document as the official design spec.

### Build status

- 21 pages, 0 errors, 0 undefined references
- Only warning: empty journal field in patterson2021carbon (minor bib formatting)

## Key technical facts verified

- **chars/doc**: measured from data_splitting I/O bytes / output documents, NOT from probing corpus
- **DRAM_POWER_W**: not available on AMD EPYC (Snellius Genoa) — only DC_NODE_POWER_W
- **EAR output**: `_apps.csv` (aggregates) and `_loops.csv` (time-series), we use aggregates
- **EAR policy**: Monitoring (passive), not Minimization (active frequency optimization)
- **Dedup telemetry bug**: single-line mkdir fix — sub-job launcher didn't create EAR directory
- **Energy formula**: `energy_j = DC_NODE_POWER_W * ELAPSED` (summed across all rows)
- **Per-char k**: varies 1 order of magnitude within string normalization ($2.4×10⁻⁵$ to $2.6×10⁻⁴$), NOT constant
- **Pipeline stage order**: dedup comes AFTER toxic detection in production config (confirmed from YAML)

## Remaining work

1. Verify Table 1 thesis coefficients against ols_fits_with_dedup.json calibrated values
2. Fix Table 4 German 200k anomalous row (200k shows less energy than 100k — needs Snellius verification)
3. Upload trained models to HuggingFace: `python scripts/upload_to_hf.py --repo GPT-NL/gptnl-energy-models`
4. Fix remaining hardcoded paths in gptnl-energy code (some still reference `/Users/hornet/Projects/TNO/`)
5. Add unit tests
6. GPU toxic-language stage: only 2 calibration sizes (100k, 400k) — needs more data points
7. 16-corpus sweep data not yet harvested (12 Dutch corpora queued but emitted no usable telemetry)
8. PII masking stage completely unmeasured (runs on PrivateAI service)

## Parent directory note

The parent `/Users/hornet/Projects/TNO/` contains the original research code (forecast.py, energy_demo.py, coefficient_model.py, etc.) and archive/. The gptnl-energy/ subdirectory is the packaged, publishable version. The TNO directory was also git-initialized but gptnl-energy/ has its own separate git remote.

## Key files for Claude to read first

- `gptnl-energy/src/gptnl_energy/data.py` — corpus parsing, data loading (single source of truth)
- `gptnl-energy/src/gptnl_energy/models/__init__.py` — model registry, sklearn models
- `gptnl-energy/src/gptnl_energy/ekf.py` — LiveEstimator (the sequential Bayesian estimator)
- `gptnl-energy/src/gptnl_energy/monitor.py` — Rich dashboard, SSH polling
- `gptnl-energy/src/gptnl_energy/cli/main.py` — Click CLI
- `gptnl-energy/paper/thesis.tex` — thesis (21 pages, builds clean)
- `gptnl-energy/paper/data/ols_fits_with_dedup.json` — calibrated coefficients
- `gptnl-energy/paper/data/measurements_generalized.csv` — 442 measurements
- `gptnl-energy/README.md` — polished project README
