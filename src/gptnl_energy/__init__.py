"""
GPT-NL Energy — energy estimation and live monitoring for LLM data curation pipelines.

Prediction layers:
  1. Physics: per-stage linear model E = c0 + c1*n  (calibrated from sample runs)
  2. Learned g: predicts coefficients for unseen corpora from data-derived features
  3. Kalman filter (EKF): online estimator that blends model predictions with live telemetry

Usage:
    gptnl-energy forecast --n 400000
    gptnl-energy monitor --live --run <slug>
"""

from gptnl_energy._version import __version__  # noqa: F401
