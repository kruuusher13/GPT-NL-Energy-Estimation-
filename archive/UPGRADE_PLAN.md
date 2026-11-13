# Upgrade Plan — answering Julio's three questions

**Context:** Presentation landed well; Julio now understands the calibrate→predict idea. He
raised three things. This plan turns each into a concrete experiment and scopes what is
realistically done **before next week**.

Julio's three points:
1. **What is the criterion for the calibration split?** (5% vs 2% — the team picks these
   ad hoc and he wants a principled rule.)
2. **Is there a major difference between 100k and 400k readings?** 400k is a quarter of the
   dataset — "that is too much data," needs a strategy.
3. **Train a neural net / transformer instead of OLS**, using OLS as the baseline, to get
   better per-step energy equations.

---

## The unifying insight (say this in the meeting)

Q1 and Q2 are **the same question**: *how much data do you need to calibrate, and how do
you know you have enough?* Both are about **sample efficiency**. One experiment answers both.

And there is a trap in Q2 worth stating out loud: if calibrating required 400k (a quarter of
the corpus), the method would be worthless — you'd spend ~25% of the full-run energy just to
predict it. The whole value proposition is that **calibration is a few percent**. So when the
linear stages show "400k ≈ 100k," that is not a problem — **that is the thesis confirming
itself.** The held-out result (calibrate {1k,100k} → predict 400k & 1.4M) is already the
evidence; we just never turned it into a stated *criterion*. The honest exception is
**deduplication**: super-linear, OOMs at 1.4M, so its large end is genuinely informative and
must be sampled differently. Conclusion: **sampling is stage-dependent, not one fixed fraction.**

---

## Workstream A — Principled calibration-split criterion (answers Q1)

**Goal:** replace "5% or 2% because the team does that" with a measured stopping rule.

**Method (calibration-fraction sweep):**
- For each stage, sweep calibration budget f ∈ {0.5%, 1%, 2%, 5%, 10%} of the full corpus.
- For each f: fit E = c0 + c1·n on the points ≤ f·N, predict the full run, record held-out MRE
  and the predicted-energy confidence band.
- **Criterion = the smallest f whose predicted full-run energy lands inside the confidence
  band of the measured full run** (target: predicted within ±X% at chosen confidence).
- Deliverable: one plot, **held-out MRE vs calibration fraction, one curve per stage.** The
  knee of each curve *is* the answer to "why this %."

**Elegant version using what we already built — the EKF stopping rule:**
- The innovation-gated EKF already maintains a running posterior variance on the energy law.
- Natural criterion: **stop calibrating when the posterior predictive interval for full-run
  energy drops below tolerance.** This makes "when do we have enough?" a *measured* event, not
  a guess, and reuses PHASE-EKF instead of bolting on something new.
- This directly upgrades the paper: the EKF goes from "nice online calibration" to "the thing
  that *tells you when to stop measuring*."

**Why it's stage-dependent:** linear stages (splitting, normalization, filtering) saturate
early — a small f is plenty. Dedup has curvature, so its criterion must include at least one
large-n probe. Report a per-stage f, not a global one. This is the honest, defensible answer.

---

## Workstream B — Sample efficiency / 100k-vs-400k (answers Q2)

**Goal:** show whether 400k carries information beyond 100k, per stage, and restructure the
sampling so we stop wasting energy measuring a quarter of the corpus.

**Tests:**
- **Slope-stability test:** is c1 estimated from {1k,100k} statistically different from c1
  including 400k? Use the held-out MRE (already 12.1% for dedup; compute for all stages). If
  the slope is stable, 400k is redundant for that stage → *don't measure it.*
- **Curvature test:** fit linear vs a mild super-linear form (e.g. c0 + c1·n + c2·n·log n) and
  compare. Linear stages: c2 ≈ 0 (400k redundant). Dedup: c2 significant (400k essential, and
  1.4M needs the memory fix before it can be measured at all).

**Restructured sampling strategy (the "strategy" Julio asked for):**
- Replace the single expensive 400k point with a **denser low-end sweep** (more points below
  100k: e.g. 1k, 5k, 20k, 50k, 100k) to pin the slope cheaply.
- Keep large-n probes **only for stages that show curvature** (dedup).
- This is **active sampling**: choose the next n where it most reduces prediction variance —
  which ties directly back to Workstream A's stopping rule.
- Net effect: lower total measurement energy, *and* it strengthens the two honest caveats
  already in the paper (single 400k rep2 with reconstructed sub-jobs; dedup validated 1k–400k
  only).

---

## Workstream C — ML upgrade, done honestly (answers Q3)

**Honest position to hold:** a transformer on 3–6 scalar points per stage is indefensible and
overfits; OLS is the correct estimator for a single-stage scalar law (R²=0.999 — nothing to
improve). Training a big model to predict energy can cost more energy than the method saves.
**So OLS stays as the per-stage physics.** The ML model earns its place as a *different* thing:
a **multi-feature, cross-stage, cross-corpus** energy model whose job is **generalization** —
predicting energy for a config/corpus we never calibrated on (i.e. removing calibration runs).

**Build the feature table first** (prerequisite for any ML): one tidy row per measured run with
features = {stage id, n_docs, n_bytes, concurrency/#tasks, node type, power-cap, corpus stats:
avg doc length, dedup ratio, language mix} and targets = {energy_kJ, wall_time_s}.

**Model ladder (with honest baselines, leave-one-size-out AND leave-one-corpus-out CV):**
1. **OLS / GLS per stage** — baseline (what we have).
2. **Gradient-boosted trees (XGBoost/LightGBM)** — the real workhorse; on tabular data this
   size it will likely beat a transformer. This is the fair "ML upgrade."
3. **Small MLP / FT-Transformer** — only as the "neural" entry Julio asked for, reported
   against OLS *and* GBM. Be ready to report "OLS/GBM already wins" — that is a valid result.

**Physics-informed structure (the honest bridge):** predict power and time separately, enforce
energy = power × time, non-negativity, and monotonicity in n. This injects the structure that
lets a model work with few runs instead of overfitting, and is far more defensible than a
black-box net. This is the version I'd actually put in the paper.

**The compelling pitch for the ML model:** "predict the energy of a *new* stage/corpus/config
without calibrating it" — transfer, not curve-fitting. That is the only framing where a neural
model beats OLS and is worth the page.

---

## One-week scope (realistic before next week)

A transformer trained properly on this little data in a week is **not** realistic. The realistic,
defensible wins this week are A and B (which answer *two* of his three questions) plus an honest
GBM baseline for C. Day-by-day:

- **Day 1–2:** Assemble the unified feature table from all harvested runs (Snellius EAR /
  eacct). Harvest the extra low-end points (5k, 20k, 50k) where cheap — these power both A and B.
- **Day 2–3:** Workstream A — calibration-fraction sweep + MRE-vs-fraction plot; wire the EKF
  posterior-interval stopping rule. Produce the per-stage criterion.
- **Day 3–4:** Workstream B — slope-stability + curvature tests; the restructured sampling
  recommendation. Produce the "400k is redundant for linear stages, essential for dedup" result.
- **Day 4–5:** Workstream C — OLS vs GBM with leave-one-size-out / leave-one-corpus-out CV.
  Report whether ML beats OLS. Sketch (not necessarily train) the physics-informed MLP design.
- **Day 5:** Fold results into one figure + one table + crisp written answers to his 3 questions.

---

## Deliverables for the next meeting

- **Figure:** held-out MRE vs calibration fraction, one curve per stage (the knee = the answer
  to "why this %").
- **Table:** OLS vs GBM vs (MLP) held-out energy error, per stage, with leave-one-out CV.
- **Three one-paragraph answers**, one per Julio question:
  1. Criterion = smallest calibration fraction whose predicted full-run energy enters the
     measured confidence band; operationalized by the EKF stopping rule; reported per stage.
  2. 100k vs 400k: redundant for linear stages (slope stable, that's the thesis working),
     essential for dedup (curvature); → stage-dependent active sampling, denser low end.
  3. OLS stays for per-stage physics; ML added as a cross-stage/cross-corpus transfer model
     with GBM/physics-informed baselines, honestly reporting whether it beats OLS.

---

## Risks & what I'd push back on

- **Don't promise a transformer result next week.** Promise the sample-efficiency criterion
  (answers Q1+Q2 cleanly) + an honest GBM baseline. Over-promising the NN is the main risk.
- **Guard against the energy irony:** quantify the energy *cost* of training the ML model vs
  the calibration energy it saves. If the model costs more than it saves, say so.
- **Few-runs overfitting:** the cross-validation must be leave-one-*group*-out (by size and by
  corpus), or the numbers are meaningless. State this explicitly.
- **Dedup memory fix** is the gating item for ever reaching 1.4M — flag it as a dependency, not
  a this-week deliverable.
