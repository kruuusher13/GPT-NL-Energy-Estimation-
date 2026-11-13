# AGENTS.md — GPT-NL Energy Prediction Codebase

This repository contains the code and data for predicting the energy consumption of the GPT-NL LLM data-curation pipeline running on the Snellius supercomputer. The core idea: measure small sample runs, learn how energy scales with document count, then predict full production runs without needing to measure them directly.

## Project Overview

**Problem**: The GPT-NL curation pipeline (splitting → normalization → filtering → deduplication) runs for hours/days on a shared HPC cluster. The energy meter (EAR) can't be left on for production runs, so the actual energy cost is unknown.

**Solution**: Calibrate a linear energy model `E = c0 + c1*n` from small exclusive-node runs, then use an Extended Kalman Filter (EKF) to refine predictions as real readings arrive during execution.

**Key insight**: Energy per *character* is nearly invariant across corpora (~4×10⁻⁴ J/char), making the model portable. Wall-clock time is co-tenancy-invariant (unlike raw power), so we predict time and convert to energy.

## Essential Commands

### Data Pipeline Refresh
```bash
./refresh.sh                    # Full refresh: harvest → pull → retrain → rebuild notebook
```
Requires SSH master session to Snellius (`ssh snellius.surf.nl`). This:
1. Harvests measurements from remote Snellius runs
2. Pulls updated CSV files to `paper/data/`
3. Retrains the coefficient model `g` (leave-one-corpus-out)
4. Rebuilds and executes the explainer notebook

### Model Training & Benchmarking
```bash
python3 train_and_save.py       # Train ALL models (OLS, Ridge, GBM, MLP, FT-Transformer, Kalman-augmented)
python3 benchmark_all.py        # Evaluate all models under identical held-out splits → benchmark_all_results.json
python3 train_kalman_transformer.py  # Train transformer on Kalman trajectory data → kalman_transformer_results.json
python3 coefficient_model.py    # Analyze per-stage coefficients, leave-one-corpus-out evaluation
python3 train_g.py              # Train the learned model g (predicts law coefficients from corpus features)
```

### Live Tools & Forecasting
```bash
python3 forecast.py --n 1400000           # Predict energy before running (kWh / € / CO₂)
python3 energy_demo.py                    # Live EKF dashboard (per-task streaming estimator)
python3 energy_demo.py --gate-demo        # Demonstrate innovation gate rejecting contaminated readings
python3 energy_demo.py --live --run <slug>  # Track real Snellius job
```

### Notebook & Presentation Building
```bash
python3 build_notebook.py                 # Generate GPTNL_Energy_Explained.ipynb
python3 -m nbconvert --to notebook --execute --inplace GPTNL_Energy_Explained.ipynb
python3 build_slides.py                   # Build presentation slides
python3 build_poster.py                   # Build research poster
```

### Paper Compilation (LaTeX)
```bash
cd paper && make                          # Compile main.pdf from main.tex
cd paper && make clean                    # Remove auxiliary files
cd paper && make upgrade                  # Update NeurIPS style files
```

## Directory Structure

```
├── paper/
│   ├── data/                             # Measurement data (CSV)
│   │   ├── measurements_generalized.csv         # Aggregated stage-level measurements
│   │   ├── measurements_raw_generalized.csv     # Raw per-task measurements
│   │   ├── ols_fits_with_dedup.json             # Calibrated OLS coefficients
│   │   └── validation.csv                       # Validation split
│   ├── code/                             # Analysis scripts
│   └── main.tex                          # NeurIPS paper
├── models/                               # Saved trained artifacts
│   ├── MODELS.md                         # Model manifest with accuracy metrics
│   ├── *.joblib                          # Sklearn models (Linear, Ridge, GBM, MLP)
│   └── *.pt                              # PyTorch models (FT-Transformer, Kalman-augmented)
├── *.py                                  # Core scripts (see below)
├── *.sh                                  # Shell scripts for config generation
└── *.ipynb                               # Jupyter notebooks (explainers, slides)
```

## Core Scripts

| Script | Purpose |
|--------|---------|
| `coefficient_model.py` | Fits per-stage physics laws (`y = c0 + c1*n`), evaluates cross-corpus transfer |
| `train_g.py` | Trains learned model `g` that predicts coefficients from corpus features |
| `benchmark_all.py` | Fair comparison: Linear, Ridge, GBM, MLP, FT-Transformer on identical splits |
| `train_kalman_transformer.py` | Augments transformer training with Kalman filter trajectory data |
| `forecast.py` | Pre-run energy forecaster (outputs kWh, €, CO₂) |
| `energy_demo.py` | Live EKF dashboard with innovation gate |
| `train_and_save.py` | Trains and saves all models to `models/` directory |
| `build_notebook.py` | Generates the explainer notebook with real data |

## Data Flow

1. **Harvest**: Remote Snellius runs produce raw EAR telemetry → aggregated into `measurements_generalized.csv`
2. **Calibrate**: Fit OLS per stage on exclusive-node runs → `ols_fits_with_dedup.json`
3. **Train**: Learn model `g` (coefficient predictor) and neural networks → `models/`
4. **Predict**: Use calibrated model + EKF to estimate full-pipeline energy
5. **Validate**: Leave-one-corpus-out and size-extrapolation tests → `benchmark_all_results.json`

## Key Conventions

### Corpus Identification
Corpora are identified by slug patterns in filenames (e.g., `amg_n1_size400000_rep1`):
- `amd/amg/size...` → American Stories news (calibration corpus, ~1798 chars/doc)
- `ghd/ghub...` → GitHub code (~3151 chars/doc)
- `gerd/gerg...` → German text (~48881 chars/doc)
- `epd/epg/eu...` → European Parliament (~181075 chars/doc)

Slug format: `{corpus}_n{tasks}_size{n}_rep{replicate}`

### Stage Order
Pipeline stages execute in this order:
1. `data_splitting` (cheap, I/O-bound)
2. `string_normalization` (scales with characters)
3. `heuristic_filtering` (**most expensive**, dominant stage)
4. `toxic_language_detection` (GPU-accelerated, fast on H100)
5. `deduplication` (super-linear, OOMs at ~1.4M docs)

### Energy Calculation
```python
energy_j = dc_node_power_w * time_sec  # Joules
energy_kJ = energy_j / 1000            # Kilojoules
```

On shared nodes, raw power is contaminated → use wall-time with effective power:
```python
E = P_eff * t  # where P_eff is calibrated once on exclusive runs
```

### Feature Engineering
Models use these features:
- `ln` = log₁₀(n) — number of documents
- `lchars` = log₁₀(chars_per_doc) — average document length
- `lnt` = log₂(ntasks) — parallelism level
- `sidx` = stage index (0–4)

## Important Gotchas

### Shared-Node Contamination
Raw EAR power readings include co-tenant workloads. A controlled experiment showed 2.4× inflation on shared vs. exclusive runs. **Always prefer time-based energy estimates** when working with shared-node data.

### Deduplication OOM
The MinHash signature pass runs out of memory at ~1.4M documents. This is a measured wall, not a bug. The `forecast.py` tool warns about this threshold.

### Missing Telemetry
In the original EAR configuration, deduplication's chained sub-jobs emitted no energy records. A one-line fix recovered them, but historical data may have gaps. Check `ols_fits_with_dedup.json` for availability.

### Cross-Corpus Transfer
Energy per *document* varies wildly between corpora (code vs. parliament transcripts). Energy per *character* is stable (~4×10⁻⁴ J/char). When transferring to unseen corpora, scale coefficients by `chars_per_doc` ratio.

### Model Overfitting
The FT-Transformer wins at size extrapolation (34% MRE vs. 155% for GBM on time prediction) but **overfits on cross-corpus transfer** (308% MRE) because we only have 4 corpora. The 16-corpus run (`gen_allcorpora.sh`) will improve this.

## Testing Approach

There are no formal unit tests. Validation is done through:
1. **Held-out size extrapolation**: Calibrate on small sizes, predict largest held-out size
2. **Leave-one-corpus-out**: Train on N-1 corpora, predict the held-out corpus
3. **Controlled experiments**: Exclusive vs. shared node comparisons
4. **R² checks**: In-sample R² ≥ 0.997 (misleading); held-out R² 0.73–0.96 (honest)

Run benchmarks with:
```bash
python3 benchmark_all.py          # All models, both axes (size + corpus)
python3 train_kalman_transformer.py  # Cold prediction with Kalman augmentation
```

## Dependencies

No `requirements.txt` or `pyproject.toml` exists. Based on imports, you need:
- Python 3.9+
- NumPy, Pandas, Matplotlib
- Scikit-learn (LinearRegression, Ridge, HistGradientBoostingRegressor, MLPRegressor)
- PyTorch (for FT-Transformer models)
- Jupyter/nbformat (for notebook building)
- LaTeX + pdfcrop (for paper compilation)

Install with:
```bash
pip install numpy pandas matplotlib scikit-learn torch jupyter nbformat nbconvert
```

## Architecture Summary

The system has three layers:

1. **Physics layer (fixed)**: Linear energy model `E = c0 + c1*n` per stage. We know the shape; we just fit coefficients from measurements.

2. **Learned model `g`**: Predicts the slope `c1` for an unseen corpus from its features (chars/doc, compute intensity, survival rate after filtering). This is the "AI" part that enables cross-corpus transfer.

3. **Kalman filter (EKF)**: Sequential estimator that blends model predictions with real readings as they arrive. Uses an innovation gate to reject contaminated readings. This is the live tool that converges to accurate estimates during execution.

The honest verdict: **calibrate-and-stream works** (~10% error when calibrated on a sample). **Cold prediction** (unseen corpus, no data) is improving but still frontier work.
