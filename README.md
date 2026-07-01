<div align="center">
<h1>⚡ GPT-NL Energy</h1>
<h3>Sample, Predict, Monitor — energy estimation for the GPT-NL data curation pipeline</h3>

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Models](https://img.shields.io/badge/🤗%20Models-HuggingFace-yellow.svg)](https://huggingface.co/GPT-NL/gptnl-energy-models)

</div>

---

## The Problem

Before committing a multi-day data curation job for an LLM training corpus, the team needs **one number**: how much energy the whole pipeline will consume. The EAR energy meter can't run on production jobs, and shared HPC nodes contaminate readings with co-tenant workloads.

## The Solution

A **sample-then-predict** framework with live monitoring:

1. **Sample**: Run a few small calibration sizes on exclusive nodes (minutes, not hours)
2. **Calibrate**: Fit a two-parameter linear energy model per stage
3. **Predict**: One command → kWh, €, and kg CO₂ for the full production run
4. **Monitor**: Extended Kalman Filter converges the estimate in real time as telemetry arrives

```
┌──────────┐     ┌──────────────┐     ┌──────────────┐
│  Sample  │ ──▶ │  Calibrate   │ ──▶ │   Forecast   │
│  runs    │     │  E=c₀+c₁·n  │     │  kWh/€/CO₂   │
└──────────┘     └──────────────┘     └──────────────┘
                                             │
                                       ┌─────▼──────┐
                                       │   Monitor  │
                                       │  EKF live  │
                                       └────────────┘
```

## Real Numbers

Measured on the **Snellius supercomputer** (AMD Genoa nodes, 5-stage pipeline):

| Corpus | Docs | Chars/doc | Pipeline Energy | Cost (€0.30/kWh) | CO₂ |
|--------|------|-----------|-----------------|--------------------|-----|
| American Stories | 400k | 1,798 | 2.2 MJ (0.73 kWh) | €0.22 | 0.22 kg |
| GitHub Code | 400k | 3,151 | 3.8 MJ (1.27 kWh) | €0.38 | 0.38 kg |
| German PD | 100k | 48,881 | 16.5 MJ (5.50 kWh) | €1.65 | 1.65 kg |
| EU Parliament | 10k | 181,075 | 18.2 MJ (6.07 kWh) | €1.82 | 1.82 kg |

**Key finding**: Energy per *character* is stable at ~1.7×10⁻⁴ J/char across a 100× range in document length — making the predictor portable to unseen corpora.

## Quick Start

```bash
# Install
pip install gptnl-energy

# Download calibrated coefficients (from HuggingFace)
# Or calibrate from your own data:
gptnl-energy calibrate --data measurements.csv --output fits.json

# One-shot prediction
gptnl-energy forecast --n 400000
# →
#   Energy forecast — 400,000 docs | corpus=american_stories (~1,798 chars/doc)
#   ==================================================================
#     data_splitting               17.4 kJ    1.7 kJ      1%
#     string_normalization        175.2 kJ   29.0 kJ      8%
#     heuristic_filtering         624.8 kJ   97.7 kJ     28%
#     toxic_language_detection     60.8 kJ    5.9 kJ      3%
#     deduplication               124.6 kJ   17.7 kJ      6%
#   ------------------------------------------------------------------
#     TOTAL PIPELINE                1.00 MJ  104.6 kJ
#   ==================================================================
#     =    0.33 kWh   (PUE 1.2)
#     =  EUR 0.10     (@ EUR 0.30/kWh)
#     =   0.10 kg CO2  (@ 0.30 kg/kWh)

# Predict for a different corpus
gptnl-energy forecast --n 100000 --corpus cc_german_pd

# Use a different model
gptnl-energy forecast --n 400000 --model gbm

# Live monitoring dashboard (replay)
gptnl-energy monitor --run amg_n1_size400000_rep1
```

## Swappable Models

The tool ships with a model registry — swap models without changing code:

| Model | Type | Best For | Held-out MRE |
|-------|------|----------|--------------|
| `ols` (default) | Per-stage physics | Same-corpus prediction | ~10% |
| `ridge` | Regularized linear | Small data regimes | ~12% |
| `gbm` | Gradient boosting | Cross-corpus transfer | ~30-40% |
| `mlp` | Neural network | Complex interactions | ~50% |
| `ftt` | FT-Transformer | Size extrapolation | ~34% |
| `kalman-ftt` | Kalman-augmented FTT | Cold whole-run prediction | improves baseline |

```python
from gptnl_energy.models import get_model

# Physics model (default, no training needed)
m = get_model("ols")
m.load_fits("paper/data/ols_fits_with_dedup.json")
result = m.predict(n=400000, corpus="american_stories")
print(f"{result['total_j']/1e6:.2f} MJ")

# Swap to GBM
m = get_model("gbm")
m.fit(df)
result = m.predict(n=400000, corpus="cc_github_opencode")
```

## Live EKF Monitor

The Extended Kalman Filter starts with a model prediction (wide confidence band) and converges to the true energy as each parallel task reports its measured energy. An innovation gate automatically rejects contaminated shared-node readings.

```bash
# Replay a recorded run
gptnl-energy monitor --run amg_n16_size400000_rep1 --speed 0.5

# Test the estimator
gptnl-energy monitor --test --run amg_n1_size400000_rep1

# Show innovation gate rejecting contamination
gptnl-energy monitor --gate-demo --run amg_n1_size400000_rep1

# Monitor a LIVE Snellius job (SSH)
gptnl-energy monitor --live --run <current-run-slug>
```

## Methodology

### Three Prediction Layers

**Layer 1 — Physics (OLS)**: `E_stage = c₀ + c₁·n`

Each pipeline stage is modeled as a linear function of document count. Coefficients are calibrated from 2-4 sample sizes on exclusive nodes. R² ≥ 0.997 in-sample.

**Layer 2 — Learned g**: coefficient prediction for unseen corpora

The per-character energy constant `k = c₁ / chars_per_doc` is nearly invariant across corpora. The model `g` predicts `k` from data-derived features (compute intensity, survival rate, I/O rate) — enabling _cold prediction_ for a corpus never measured.

**Layer 3 — Kalman Filter (EKF)**:

Sequential estimator with state = per-stage energy. Starts with the model prior, then blends in live telemetry as readings arrive. Innovation gate rejects contaminated readings (>2.5× stage budget). Converges to ~10% error after all stages complete.

### Innovation Gate Performance

| Scenario | Without Gate | With Gate |
|----------|-------------|-----------|
| Clean run (exclusive node) | 8.3% error | 8.3% error |
| Contaminated (shared node, 2.4× inflation) | +93% error | +8% error |

## Integration with GPT-NL Pipeline

This package is designed to integrate directly with the [GPT-NL data curation pipeline](https://github.com/GPT-NL/data-curation-pipeline):

```bash
# In the pipeline's pyproject.toml:
# gptnl-energy = "^0.1.0"

# Before launching a curation run:
poetry run gptnl-energy forecast --n 1400000 --corpus cc_german_pd

# During the run, monitor energy:
poetry run gptnl-energy monitor --live --run <slug>
```

## Repository Structure

```
gptnl-energy/
├── src/gptnl_energy/
│   ├── __init__.py          # Package entry
│   ├── data.py              # Data loading, corpus parsing, config
│   ├── forecast.py          # Pre-run energy prediction
│   ├── ekf.py               # Extended Kalman Filter estimator
│   ├── monitor.py           # Live terminal dashboard
│   ├── models/
│   │   ├── __init__.py      # Model registry + sklearn models
│   │   └── torch_models.py  # FT-Transformer (PyTorch)
│   └── cli/
│       └── main.py          # Click CLI (forecast, monitor, calibrate)
├── scripts/
│   ├── train_all.py         # Train all models
│   └── upload_to_hf.py      # Upload models to HuggingFace
├── paper/                   # Thesis + data + fitted coefficients
│   ├── data/
│   │   ├── measurements_generalized.csv
│   │   ├── measurements_raw_generalized.csv
│   │   └── ols_fits_with_dedup.json
│   └── thesis.pdf
└── models/                  # Trained model artifacts
```

## Citation

```bibtex
@mastersthesis{malik2026energy,
  title={Complexity-Aware Energy Estimation for an LLM Data Curation Pipeline},
  author={Malik, Romir},
  school={Utrecht University},
  year={2026},
  type={MSc Thesis}
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

Built by [Romir Malik](https://github.com/kruuusher13) · MSc Applied Data Science · Utrecht University · Thesis deadline: July 2026
