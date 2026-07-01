#!/usr/bin/env python3
"""
Upload trained models to HuggingFace Hub.

Usage:
  python scripts/upload_to_hf.py --repo GPT-NL/gptnl-energy-models
  python scripts/upload_to_hf.py --repo your-org/gptnl-energy-models --token hf_xxx
  python scripts/upload_to_hf.py --repo GPT-NL/gptnl-energy-models --private

Requires: pip install huggingface_hub
Set HF_TOKEN env var or pass --token.
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo, upload_folder
    HAS_HF = True
except ImportError:
    HAS_HF = False
    print("huggingface_hub not installed. Run: pip install huggingface_hub")


MODEL_FILES = [
    "linear_energy.joblib",
    "linear_time.joblib",
    "ridge_energy.joblib",
    "ridge_time.joblib",
    "gbm_energy.joblib",
    "gbm_time.joblib",
    "mlp_energy.joblib",
    "mlp_time.joblib",
    "ftt_energy.pt",
    "ftt_time.pt",
    "kalman_transformer_energy.pt",
]

MODEL_CARDS = {
    "linear_energy.joblib": {
        "description": "Per-stage OLS linear model (physics baseline): E = c0 + c1*n per pipeline stage",
        "features": ["n (document count)"],
        "target": "energy_j (Joules)",
        "accuracy": "~10% MRE on held-out size extrapolation",
    },
    "gbm_energy.joblib": {
        "description": "Histogram Gradient Boosting — best ML model for cross-corpus energy transfer",
        "features": ["ln", "lchars", "lnt", "sidx"],
        "target": "energy_j (Joules)",
        "accuracy": "~30-40% MRE on cross-corpus transfer",
    },
    "ftt_energy.pt": {
        "description": "FT-Transformer — neural tabular model with feature tokenization + stage embedding",
        "features": ["ln", "lchars", "lnt", "sidx"],
        "target": "energy_j (Joules)",
        "accuracy": "~34% MRE on size extrapolation (best), ~308% MRE on cross-corpus (overfits with 4 corpora)",
    },
    "kalman_transformer_energy.pt": {
        "description": "FT-Transformer trained on Kalman filter trajectories for cold whole-run prediction",
        "features": ["log_n", "log_chars", "frac_observed", "log_kalman_partial", "log_observed"],
        "target": "total_energy_j (whole pipeline, cold start)",
        "accuracy": "Improves cross-corpus cold prediction vs plain transformer",
    },
}


def main():
    if not HAS_HF:
        sys.exit(1)

    ap = argparse.ArgumentParser(description="Upload trained models to HuggingFace Hub")
    ap.add_argument("--repo", required=True,
                    help="HuggingFace repo ID (e.g., GPT-NL/gptnl-energy-models)")
    ap.add_argument("--token", default=None,
                    help="HF token (or set HF_TOKEN env var)")
    ap.add_argument("--private", action="store_true",
                    help="Create as private repo")
    ap.add_argument("--models-dir", default="models",
                    help="Directory containing trained models")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be uploaded without doing it")
    args = ap.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF token required. Set HF_TOKEN env var or pass --token.")
        sys.exit(1)

    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        print(f"Error: models directory '{models_dir}' not found.")
        print("Run 'python scripts/train_all.py' first.")
        sys.exit(1)

    # Find available model files
    available = [f for f in MODEL_FILES if (models_dir / f).exists()]
    if not available:
        print(f"No trained models found in {models_dir}/")
        print(f"Expected files: {MODEL_FILES}")
        sys.exit(1)

    print(f"Found {len(available)} trained models:")
    for f in available:
        size_mb = (models_dir / f).stat().st_size / 1e6
        card = MODEL_CARDS.get(f, {})
        print(f"  {f:<40} ({size_mb:.1f} MB)  {card.get('description', '')}")

    if args.dry_run:
        print(f"\nWould upload to: {args.repo}")
        return

    # Create or get repo
    api = HfApi(token=token)
    try:
        create_repo(
            args.repo, token=token, private=args.private, exist_ok=True
        )
        print(f"\nRepo: https://huggingface.co/{args.repo}")
    except Exception as e:
        print(f"Error creating repo: {e}")
        sys.exit(1)

    # Upload models
    for f in available:
        local_path = models_dir / f
        print(f"Uploading {f}...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=f,
            repo_id=args.repo,
            token=token,
        )

    # Upload README
    readme = _build_readme(available, args.repo)
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=args.repo,
        token=token,
    )

    print(f"\n✓ Uploaded {len(available)} models to https://huggingface.co/{args.repo}")
    print("\nUsers can now load models with:")
    print(f"  from gptnl_energy.models import get_model")
    print(f"  m = get_model('linear')")
    print(f"  m.load_fits(hf_hub_download('{args.repo}', 'linear_energy.joblib'))")


def _build_readme(files, repo_id):
    lines = [
        "# GPT-NL Energy Prediction Models",
        "",
        "Trained energy prediction models for the GPT-NL data curation pipeline.",
        "These models estimate the energy consumption of each pipeline stage",
        "(data splitting → string normalization → heuristic filtering →",
        "toxic language detection → deduplication) running on the Snellius",
        "supercomputer.",
        "",
        "## Usage",
        "",
        "```bash",
        "pip install gptnl-energy huggingface_hub",
        "",
        "# Download and use a model",
        "from huggingface_hub import hf_hub_download",
        "from gptnl_energy.models import get_model",
        "",
        "model_path = hf_hub_download('GPT-NL/gptnl-energy-models', 'linear_energy.joblib')",
        "model = get_model('linear')",
        "model.load_fits(model_path)",
        "",
        "# Predict energy for 400k documents",
        "result = model.predict(n=400000, corpus='american_stories')",
        "print(f'{result[\"total_j\"]/1e6:.2f} MJ')",
        "```",
        "",
        "## Available Models",
        "",
        "| File | Type | Target | Description |",
        "|---|---|---|---|",
    ]
    for f in files:
        card = MODEL_CARDS.get(f, {})
        lines.append(
            f"| `{f}` | {'PyTorch' if f.endswith('.pt') else 'sklearn'} | "
            f"{card.get('target', '—')} | {card.get('description', '—')} |"
        )

    lines.extend([
        "",
        "## Methodology",
        "",
        "Three prediction layers:",
        "1. **Physics**: Per-stage linear model E = c0 + c1·n (OLS, calibrated from sample runs)",
        "2. **Learned g**: Coefficient predictor for unseen corpora (GBM, MLP, FT-Transformer)",
        "3. **Kalman filter (EKF)**: Online estimator blending model prediction with live telemetry",
        "",
        "See the [GPT-NL Energy repo](https://github.com/GPT-NL/gptnl-energy) for the full pipeline.",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
