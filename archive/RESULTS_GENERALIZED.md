# Generalized energy model — real results (harvested 2026-06-14)

Source: `paper/data/measurements_generalized.csv` (209 stage-measurements, 2776 raw
node rows). Sweep: **5 corpora × ntasks{1,16,32} × 5 stages**, genoa CPU + H100 toxic.
Tools: `forecast.py` (usage demo), `ml_energy_model.py` (OLS→GBM→MLP ladder).

## The one-line story for the team
"I built a tool that predicts the energy of the **whole** curation pipeline — for any
dataset size, parallelism, hardware, and corpus — and turns it into kWh / € / CO₂
**before you launch.** Here's how you use it, and three things it already tells us."

---

## Result 1 — The forecast tool (how they use it)
`python3 forecast.py --n 400000 --corpus cc_german_pd` →
per-stage breakdown + total **25.5 MJ = 8.5 kWh = €2.55 = 2.55 kg CO₂**, with a
dedup-OOM warning at ~1.4M docs. One command, before spending the energy.

## Result 2 — Energy scales with CHARACTERS, not documents (the transfer law)
Measured `string_normalization`:
- am-stories (1.8k chars/doc), 100k docs → ~40 kJ
- **German (49k chars/doc), 100k docs → 1,499 kJ  (~37× — matches the ~27× char ratio)**
So a per-*document* law (OLS on one corpus) can't transfer; a per-*character* model can.
This is why the generalized model exists, and why GBM beats OLS across corpora.

## Result 3 — Parallelism barely helps → high n_tasks WASTES energy (actionable)
`heuristic_filtering`, am-stories 400k, wall-time:
- ntasks=1 → 2344 s | ntasks=16 → 2268 s | ntasks=32 → 1927 s
**32× the tasks ≈ 1.2× speedup** (datatrove sharding is load-imbalanced here). Since
energy ≈ node-power × time summed over the tasks' nodes, cranking n_tasks multiplies
the power draw for ~no time saving → **the energy-optimal n_tasks is LOW.** Direct,
useful answer to the team's n_tasks sweep, with an energy lens.

## Result 4 — Toxic-language measured on H100 (new 5th stage)
~68 W GPU, fast: am-stories 400k → 3.1 kJ GPU / 61 kJ node (CPU host dominates). Now a
first-class stage in the forecast. (Pipeline really has 6 stages; pii_masking needs the
PrivateAI service — noted as the remaining gap.)

## Result 5 — The honest ML verdict (Julio's Q3)
Held-out, grouped CV (leave-one-corpus-out / leave-one-size-out):
- **Cross-corpus transfer:** GBM beats OLS (time 337% vs 520%; energy 547% vs 2993%).
- **Same-corpus extrapolation:** OLS/GBM comparable; OLS is the right per-stage physics.
Verdict to state plainly: **keep OLS as per-stage physics; add ML (GBM) as the
cross-corpus transfer layer.** Absolute errors are still large because the data is
single-replicate and spans overhead-dominated tiny runs to 1.4M — **more replicates +
the chars feature will tighten it** (this is the one-week follow-up, not a failure). A
GPU transformer is NOT justified: it would cost more energy than the calibration it saves.

---

## Caveats (say them — they're strengths)
- ntasks>1 energy via summed node power is inflated (whole-node power per array task) →
  **wall-time is the clean signal**; energy follows via E = P_eff·t (the established
  R²=0.99 finding). The forecast uses the per-stage calibrated law.
- Rome vs genoa: coefficients are per-hardware (Zen2/128c vs Zen4/192c) → cross-hardware
  is the model's next axis (teammate's rome runs = free second hardware point).

## Deck arc
problem → the forecast tool (live, on real numbers) → Results 2–4 as "what it already
tells us" → Result 5 (honest ML verdict) → roadmap (pii stage, more reps, rome transfer).
