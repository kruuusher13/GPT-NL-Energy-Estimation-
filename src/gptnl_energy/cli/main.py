"""
CLI entry point for gptnl-energy.

    gptnl-energy forecast --n 400000
    gptnl-energy forecast --n 100000 --corpus cc_german_pd --model gbm
    gptnl-energy monitor --run amg_n1_size400000_rep1
    gptnl-energy monitor --live --run amg_n16_size400000_rep1
    gptnl-energy calibrate --data measurements.csv --output fits.json
    gptnl-energy models list
"""

import sys
from pathlib import Path

import click

from gptnl_energy import __version__
from gptnl_energy.data import (
    CORPUS_CHARS,
    CALIBRATION_CORPUS,
    PIPELINE_STAGES,
    DEFAULT_PUE,
    DEFAULT_EUR_PER_KWH,
    DEFAULT_KG_CO2_PER_KWH,
)
from gptnl_energy.forecast import Forecast
from gptnl_energy.monitor import Monitor


def _get_default_fits() -> str:
    """Find the default fits JSON."""
    candidates = [
        "paper/data/ols_fits_with_dedup.json",
        "../paper/data/ols_fits_with_dedup.json",
        str(Path(__file__).parent.parent.parent.parent
            / "paper/data/ols_fits_with_dedup.json"),
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return "paper/data/ols_fits_with_dedup.json"


@click.group()
@click.version_option(__version__, prog_name="gptnl-energy")
def main():
    """GPT-NL Energy — predict and monitor energy consumption of the
    data curation pipeline on HPC/SLURM infrastructure."""
    pass


@main.command()
@click.option("--n", type=int, required=True, help="Number of documents to process")
@click.option("--corpus", default=CALIBRATION_CORPUS,
              type=click.Choice(list(CORPUS_CHARS)),
              help="Corpus name (affects per-character energy scaling)")
@click.option("--n-tasks", type=int, default=1, help="Number of parallel SLURM tasks")
@click.option("--hardware", default="genoa",
              type=click.Choice(["genoa", "rome"]),
              help="CPU hardware (affects effective power)")
@click.option("--stages", default=None,
              help="Comma-separated stages (default: all 5)")
@click.option("--model", default="ols",
              type=click.Choice(["ols", "linear", "ridge", "gbm", "mlp", "ftt"]),
              help="Prediction model to use")
@click.option("--model-path", default=None,
              help="Path to pre-trained model file (.joblib or .pt)")
@click.option("--fits", default=None,
              help="Path to OLS coefficient fits JSON")
@click.option("--price", type=float, default=DEFAULT_EUR_PER_KWH,
              help="EUR per kWh")
@click.option("--co2", type=float, default=DEFAULT_KG_CO2_PER_KWH,
              help="kg CO2 per kWh")
@click.option("--pue", type=float, default=DEFAULT_PUE,
              help="Power Usage Effectiveness (datacenter overhead)")
def forecast(n, corpus, n_tasks, hardware, stages, model, model_path,
             fits, price, co2, pue):
    """One-shot energy prediction before launching a run.

    Predicts total pipeline energy in kJ, kWh, EUR, and kg CO2,
    with per-stage breakdown and 95% confidence band.
    """
    stages_list = None
    if stages:
        stages_list = [s.strip() for s in stages.split(",") if s.strip()]

    fits_path = fits or _get_default_fits()

    if model_path:
        import joblib
        data = joblib.load(model_path)
        if isinstance(data, dict) and "state" in data:
            from gptnl_energy.models import get_model
            model_type = data.get("metadata", {}).get("type", "ols")
            m = get_model(model_type)
            m._set_state(data["state"])
            f = Forecast.from_model(m)
        else:
            click.echo("Error: model-path must point to a saved gptnl-energy model",
                       err=True)
            sys.exit(1)
    elif model != "ols":
        try:
            from gptnl_energy.models import get_model
            m = get_model(model)
            m.load_fits(fits_path) if hasattr(m, 'load_fits') else None
            f = Forecast.from_model(m)
        except KeyError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
    else:
        if not Path(fits_path).exists():
            click.echo(
                f"Error: fits file not found at '{fits_path}'. "
                f"Run 'gptnl-energy calibrate' first, or pass --fits.",
                err=True,
            )
            sys.exit(1)
        f = Forecast.from_fits(fits_path)

    result = f.predict(n, corpus, n_tasks, hardware, stages_list)
    click.echo(result.format_terminal(pue, price, co2))


@main.command()
@click.option("--run", default=None,
              help="Run slug (e.g., amg_n1_size400000_rep1)")
@click.option("--csv", default=None,
              help="Path to measurements_raw_generalized.csv")
@click.option("--speed", type=float, default=0.7,
              help="Replay speed (seconds between task updates)")
@click.option("--fits", default=None,
              help="Path to OLS coefficient fits JSON")
@click.option("--test", "test_mode", is_flag=True,
              help="Quick test: run silently and print final stats")
@click.option("--gate-demo", is_flag=True,
              help="Inject a contaminated reading to show the innovation gate")
@click.option("--live", "live_mode", is_flag=True,
              help="Monitor a live Snellius job via SSH")
@click.option("--poll", type=float, default=8.0,
              help="Poll interval for live mode (seconds)")
def monitor(run, csv, speed, fits, test_mode, gate_demo, live_mode, poll):
    """Live energy monitoring with Extended Kalman Filter.

    Replay a recorded run or connect to a live Snellius job.
    Shows the EKF estimate converging in real time as tasks complete.
    """
    fits_path = fits or _get_default_fits()
    if not Path(fits_path).exists():
        click.echo(f"Error: fits file not found at '{fits_path}'", err=True)
        sys.exit(1)

    mon = Monitor.from_fits(fits_path)

    if live_mode:
        if not run:
            click.echo("Error: --live requires --run SLUG", err=True)
            sys.exit(1)
        mon.run_live_ssh(run, poll)
        return

    csv_path = csv or "paper/data/measurements_raw_generalized.csv"
    if not Path(csv_path).exists():
        click.echo(f"Error: raw CSV not found at '{csv_path}'", err=True)
        sys.exit(1)

    # Auto-pick a run if none given
    if run is None:
        import pandas as pd
        df = pd.read_csv(csv_path)
        df = df[df.dc_node_power_w.fillna(0) > 0]
        n1 = df[df.slug.str.contains("_n1_") | df.slug.str.match(r"size\d")]
        pick = n1 if not n1.empty else df
        run = pick.groupby("slug").size().sort_values().index[-1]

    if test_mode:
        mon.run_test(csv_path, run)
    else:
        mon.run_replay(csv_path, run, speed, gate_demo)


@main.command()
@click.option("--data", "data_path", required=True,
              help="Path to measurements CSV")
@click.option("--output", default="fits.json",
              help="Output path for coefficient JSON")
@click.option("--target", default="energy_j",
              type=click.Choice(["energy_j", "time_sec"]),
              help="Target variable to calibrate")
def calibrate(data_path, output, target):
    """Calibrate per-stage energy laws from measurement data.

    Fits E = c0 + c1*n per stage, saves coefficients to JSON.
    This is the first step before forecasting or monitoring.
    """
    import json
    import numpy as np
    import pandas as pd

    from gptnl_energy.data import load_measurements, PIPELINE_STAGES

    df = load_measurements(data_path, ntasks_filter=1)
    stages = [s for s in PIPELINE_STAGES if s in df.stage.unique()]

    fits = {}
    for stage in stages:
        sub = df[df.stage == stage]
        grp = sub.groupby("n")[target].mean().reset_index().sort_values("n")
        if len(grp) < 2:
            click.echo(f"  Skipping {stage}: need ≥2 distinct sizes (got {len(grp)})")
            continue
        A = np.vstack([grp.n.values, np.ones(len(grp))]).T
        (c1, c0), residuals, _, _ = np.linalg.lstsq(
            A, grp[target].values, rcond=None
        )
        sigma2 = float(np.var(residuals)) if len(residuals) > 0 else 0.0
        fits[stage] = {
            "stage": stage,
            "features": ["n"],
            "theta": [float(c1), float(c0)],
            "intercept_j": float(c0),
            "marginal_theta": [float(c1)],
            "sigma2": sigma2,
            "n_obs": len(grp),
        }
        click.echo(f"  {stage:<26} c1={c1:.4g} J/doc  c0={c0:.1f} J  "
                    f"σ={np.sqrt(sigma2):.1f}  (n={len(grp)})")

    with open(output, "w") as f:
        json.dump({"fits": fits}, f, indent=2)

    click.echo(f"\n  Saved {len(fits)} stage fits to {output}")


@main.group()
def models():
    """Manage energy prediction models."""
    pass


@models.command("list")
def models_list():
    """List available models."""
    from gptnl_energy.models import available_models
    click.echo("Available models:")
    for name in available_models():
        click.echo(f"  - {name}")
    click.echo("\nUse --model <name> with the forecast command.")


@models.command("train")
@click.option("--data", "data_path", required=True,
              help="Path to measurements CSV")
@click.option("--model", "model_name", default="ols",
              type=click.Choice(["ols", "ridge", "gbm", "mlp", "ftt"]),
              help="Model type to train")
@click.option("--target", default="energy_j",
              type=click.Choice(["energy_j", "time_sec"]))
@click.option("--output", default=None,
              help="Output path for trained model")
def models_train(data_path, model_name, target, output):
    """Train a model and save it."""
    from gptnl_energy.data import load_measurements
    from gptnl_energy.models import get_model

    df = load_measurements(data_path)
    click.echo(f"Training {model_name} on {len(df)} readings...")

    m = get_model(model_name)
    m.fit(df, target)

    out = output or f"{model_name}_{target}.joblib"
    m.save(out)
    click.echo(f"Saved to {out}")


if __name__ == "__main__":
    main()
