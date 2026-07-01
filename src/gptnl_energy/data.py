"""
Unified data loading, corpus identification, and configuration.

This is the SINGLE source of truth for corpus parsing — every module imports from here.
No more copy-pasted corp() functions with inconsistent naming.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ── Corpus constants ───────────────────────────────────────────────────────────
# Measured average characters per document (probe_corpora.py, 2026-06-14)
CORPUS_CHARS: Dict[str, int] = {
    "american_stories": 1798,
    "cc_github_opencode": 3151,
    "cc_german_pd": 48881,
    "european_parliament": 181075,
}

CALIBRATION_CORPUS = "american_stories"
CALIBRATION_CHARS = CORPUS_CHARS[CALIBRATION_CORPUS]

PIPELINE_STAGES = [
    "data_splitting",
    "string_normalization",
    "heuristic_filtering",
    "toxic_language_detection",
    "deduplication",
]

STAGE_SHORT = {
    "data_splitting": "split",
    "string_normalization": "normalize",
    "heuristic_filtering": "heuristic",
    "toxic_language_detection": "toxic",
    "deduplication": "dedup",
}

# Tag → canonical corpus name mapping (covers all slug generations)
TAG_TO_CORPUS: Dict[str, str] = {
    # American Stories (calibration corpus)
    "amd": "american_stories",
    "amg": "american_stories",
    "amst": "american_stories",
    # GitHub code
    "ghd": "cc_github_opencode",
    "ghub": "cc_github_opencode",
    # German
    "gerd": "cc_german_pd",
    "gerg": "cc_german_pd",
    # European Parliament
    "epd": "european_parliament",
    "epg": "european_parliament",
    "eu": "european_parliament",
    # Additional 16-corpus sweep
    "aud": "auditdienstrijk",
    "bel": "belgian_journal",
    "eng": "cc_english_pd",
    "oal": "cc_openalex",
    "dan": "dansknaw",
    "dpc": "dpc",
    "kbb": "kb_pd_books",
    "nat": "naturalis",
    "rec": "rechtspraak",
    "twk": "tweedekamer",
    "utr": "utrechtsarchief",
    "woo": "woogle",
}

# Energy/carbon constants (Netherlands datacenter, overridable)
DEFAULT_PUE = 1.20
DEFAULT_EUR_PER_KWH = 0.30
DEFAULT_KG_CO2_PER_KWH = 0.30

# Deduplication OOM threshold
DEDUP_OOM_N = 1_400_000


def parse_corpus(slug: str) -> str:
    """Map a run slug to its canonical corpus name. Single source of truth."""
    s = slug.lower()
    tag = s.split("_")[0]
    if tag in TAG_TO_CORPUS:
        return TAG_TO_CORPUS[tag]
    # Heuristic fallbacks
    if s.startswith(("amst", "amg", "amd", "size", "dedupval", "demolive")):
        return "american_stories"
    if s.startswith(("euparl", "epd", "epg", "eu")):
        return "european_parliament"
    if s.startswith(("ghd", "ghub", "gh")) or "github" in s:
        return "cc_github_opencode"
    if s.startswith(("gerd", "gerg", "ger")) or "german" in s:
        return "cc_german_pd"
    if s.startswith("mix"):
        return "MIX"
    return tag  # unknown — pass through


def parse_ntasks(slug: str) -> int:
    """Extract n_tasks from slug (e.g., 'amg_n16_size400000_rep1' → 16)."""
    m = re.search(r"_n(\d+)_", slug)
    return int(m.group(1)) if m else 1


def parse_size(slug: str) -> Optional[int]:
    """Extract document count from slug (e.g., 'amg_n1_size400000_rep1' → 400000)."""
    m = re.search(r"size(\d+)", slug)
    return int(m.group(1)) if m else None


def corpus_chars(corpus: str) -> int:
    """Characters per document for a corpus. Falls back to calibration corpus."""
    return CORPUS_CHARS.get(corpus, CALIBRATION_CHARS)


def load_measurements(
    path: str,
    ntasks_filter: Optional[int] = None,
    min_energy: bool = True,
) -> pd.DataFrame:
    """Load and clean the generalized measurements CSV.

    Returns a DataFrame with added columns: corpus, ntasks, n, energy_j, chars, ln, lchars, lnt.
    """
    df = pd.read_csv(path)
    df["corpus"] = df.slug.map(parse_corpus)
    df["ntasks"] = df.slug.map(parse_ntasks)
    df["n"] = pd.to_numeric(df["size"], errors="coerce")

    for col in ["time_sec", "dc_node_power_w", "io_mbs", "cpi", "cpu_gflops",
                 "cpu_util", "n_in", "n_out"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")

    df["energy_j"] = df.dc_node_power_w.fillna(0) * df.time_sec

    if ntasks_filter is not None:
        df = df[df.ntasks == ntasks_filter]
    if min_energy:
        df = df[(df.energy_j > 0) & (df.time_sec > 0) & df.n.notna()]

    df["chars"] = df.corpus.map(lambda c: corpus_chars(c))
    df["ln"] = np.log10(df.n)
    df["lchars"] = np.log10(df.chars.clip(lower=1))
    df["lnt"] = np.log2(df.ntasks.clip(lower=1))

    return df


def load_fits(path: str) -> Dict:
    """Load per-stage OLS coefficient fits from JSON."""
    with open(path) as f:
        data = json.load(f)

    fits = {}
    import math

    for stage, fit in data.get("fits", data).items():
        theta = fit.get("theta") or [
            fit.get("marginal_theta", [0])[0],
            fit.get("intercept_j", 0),
        ]
        sigma2 = fit.get("sigma2", 0.0)
        fits[stage] = {
            "c1": float(theta[0]),
            "c0": float(theta[1]),
            "sigma": math.sqrt(max(sigma2, 0)),
        }

    # GPU toxic-language stage (H100, preliminary 2-point fit)
    fits.setdefault("toxic_language_detection", {
        "c1": 0.135,
        "c0": 6843.0,
        "sigma": 3000.0,
    })

    return fits


@dataclass
class EnergyConfig:
    """Configuration for energy estimation pipeline."""
    pue: float = DEFAULT_PUE
    eur_per_kwh: float = DEFAULT_EUR_PER_KWH
    kg_co2_per_kwh: float = DEFAULT_KG_CO2_PER_KWH
    hardware: str = "genoa"
    stages: List[str] = field(default_factory=lambda: PIPELINE_STAGES.copy())
    corpus: str = CALIBRATION_CORPUS
    n_tasks: int = 1
    fits_path: Optional[str] = None
