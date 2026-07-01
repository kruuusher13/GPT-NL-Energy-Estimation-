#!/usr/bin/env python3
"""
Train ALL models on measurement data and save artifacts.

One script to replace train_and_save.py, train_transformer.py,
train_kalman_transformer.py, and benchmark_all.py.

Output:
  models/linear_energy.joblib, models/gbm_energy.joblib, ...
  models/ftt_energy.pt, models/kalman_transformer_energy.pt
  paper/data/ols_fits_with_dedup.json (if calibrating)

Usage:
  python scripts/train_all.py --data paper/data/measurements_generalized.csv
  python scripts/train_all.py --data paper/data/measurements_generalized.csv --target time_sec
  python scripts/train_all.py --data paper/data/measurements_generalized.csv --skip-torch
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gptnl_energy.data import load_measurements, PIPELINE_STAGES, CORPUS_CHARS


def fit_ols_per_stage(df, target="energy_j"):
    """Calibrate per-stage linear model and save to JSON."""
    from gptnl_energy.models import PerStageOLSModel

    model = PerStageOLSModel()
    model.fit(df, target)

    fits_path = "paper/data/ols_fits_with_dedup.json"
    os.makedirs(os.path.dirname(fits_path), exist_ok=True)

    # Serialize in the format expected by load_fits()
    fits_data = {"fits": {}}
    for stage, fit in model._fits.items():
        fits_data["fits"][stage] = {
            "stage": stage,
            "features": ["n"],
            "theta": [fit["c1"], fit["c0"]],
            "intercept_j": fit["c0"],
            "marginal_theta": [fit["c1"]],
            "sigma2": fit["sigma"] ** 2,
            "n_obs": len(df[df.stage == stage]["n"].unique()),
        }

    with open(fits_path, "w") as f:
        json.dump(fits_data, f, indent=2)
    print(f"  ✓ Per-stage OLS fits → {fits_path}")
    return model


def train_sklearn(df, target, model_name):
    """Train a sklearn-based model and save."""
    from gptnl_energy.models import get_model

    m = get_model(model_name)
    m.fit(df, target)

    fname = f"models/{model_name}_{'energy' if target == 'energy_j' else 'time'}.joblib"
    os.makedirs("models", exist_ok=True)
    m.save(fname)
    print(f"  ✓ {model_name.upper()} → {fname}")
    return m


def train_ftt(df, target):
    """Train FT-Transformer and save."""
    try:
        from gptnl_energy.models.torch_models import FTTransformerModel
    except ImportError:
        print("  ⚠ FT-Transformer skipped (torch not available)")
        return None

    m = FTTransformerModel(epochs=400)
    m.fit(df, target)

    fname = f"models/ftt_{'energy' if target == 'energy_j' else 'time'}.pt"
    os.makedirs("models", exist_ok=True)
    m.save(fname)
    print(f"  ✓ FT-Transformer → {fname}")
    return m


def train_kalman_ftt(df, target):
    """Train Kalman-augmented FT-Transformer and save."""
    try:
        from gptnl_energy.models.torch_models import KalmanFTTModel
    except ImportError:
        print("  ⚠ Kalman-FTT skipped (torch not available)")
        return None

    m = KalmanFTTModel(epochs=400)
    m.fit(df, target)

    fname = "models/kalman_transformer_energy.pt"
    os.makedirs("models", exist_ok=True)
    m.save(fname)
    print(f"  ✓ Kalman-FTT → {fname}")
    return m


def main():
    ap = argparse.ArgumentParser(description="Train all energy prediction models")
    ap.add_argument("--data", default="paper/data/measurements_generalized.csv",
                    help="Path to measurements CSV")
    ap.add_argument("--target", default="energy_j",
                    choices=["energy_j", "time_sec"])
    ap.add_argument("--skip-torch", action="store_true",
                    help="Skip PyTorch models")
    ap.add_argument("--skip-kalman", action="store_true",
                    help="Skip Kalman FTT (slow)")
    ap.add_argument("--models", default="all",
                    help="Comma-separated: ols,ridge,gbm,mlp,ftt,kalman (or 'all')")
    args = ap.parse_args()

    t0 = time.time()
    print(f"Training models on {args.data} (target={args.target})")
    print("=" * 60)

    df = load_measurements(args.data)
    print(f"Loaded {len(df)} readings | {df.corpus.nunique()} corpora | "
          f"stages={sorted(df.stage.unique())}")

    models_to_train = set(
        args.models.split(",") if args.models != "all"
        else ["ols", "ridge", "gbm", "mlp", "ftt", "kalman"]
    )

    manifest = []

    # 1. OLS (always — it's the physics baseline + produces fits.json)
    if "ols" in models_to_train:
        ols = fit_ols_per_stage(df, args.target)
    else:
        ols = None

    # 2. Ridge
    if "ridge" in models_to_train:
        train_sklearn(df, args.target, "ridge")

    # 3. GBM
    if "gbm" in models_to_train:
        train_sklearn(df, args.target, "gbm")

    # 4. MLP
    if "mlp" in models_to_train:
        train_sklearn(df, args.target, "mlp")

    # 5. FT-Transformer
    if "ftt" in models_to_train and not args.skip_torch:
        train_ftt(df, args.target)

    # 6. Kalman-FTT (energy only)
    if "kalman" in models_to_train and not args.skip_kalman and not args.skip_torch:
        if args.target == "energy_j":
            train_kalman_ftt(df, args.target)
        else:
            print("  ⚠ Kalman-FTT only supports energy_j target, skipping")

    elapsed = time.time() - t0
    print("=" * 60)
    print(f"Training complete in {elapsed:.0f}s. Models in ./models/")
    print("\nNext steps:")
    print("  gptnl-energy forecast --n 400000")
    print("  python scripts/upload_to_hf.py --repo GPT-NL/gptnl-energy-models")


if __name__ == "__main__":
    main()
