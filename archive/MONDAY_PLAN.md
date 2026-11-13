# Monday plan — generalized energy model, real readings, whole pipeline

**Goal Julio set:** final results + "show the team how they can use what I built."
**Tonight's job:** get *real new readings* (toxic-language stage + ntasks sweep + more
corpora), train the neural net, test the monitor — then build the finished product on top.

---

## The reframe: from per-stage constants to a GENERALIZED model

So far: per-stage law `E = c₀ + c₁·n` at *one* config (ntasks=1, genoa, am-stories).
The generalized solution predicts energy as a function of **all** the knobs the team
actually turns:

    E = f( stage, n, n_tasks, hardware[genoa/rome], corpus[chars/doc, lang], stage-params )

That is exactly the screenshot the team sent (every stage sweeps `n_tasks` + algorithmic
params). **Their sweep grid is my feature list. Their parameters are my training set.**

---

## What the inventory found (Snellius)

- **6 pipeline stages exist**, not 4: data_splitting, string_normalization,
  heuristic_filtering, **pii_masking**, **toxic_language_detection**, deduplication.
  Measured so far: 4. Missing: pii_masking, toxic_language_detection.
- **toxic_language_detection** is runnable: stage file + `local-toxic-language-run.yaml`.
  Defaults `threshold=0.995`, `max_chunk_length=256`. It's a transformer classifier
  (the `chunk_length` knob) → CPU works but is slow; H100 is the fast path.
- **16 corpora** in `datasets/` (american-stories, european_parliament, cc_english_pd,
  cc_german-pd, rechtspraak, tweedekamer, kb-pd-books, …) → cross-corpus fuel.
- Dmytro's `shared/`: am_stories at 10/25/50/100% + eu_parliament (scaled datasets).
- 33 existing runs, **all ntasks=1**. Teammates' folders empty/locked.
- Run harness is a parameterized YAML generator (`00m_gen_multicorpus.sh`) — adding a
  stage or changing ntasks = editing the template. Monitor = `02_monitor.sh`.

## Clean-reading strategy (the ntasks decision)

- Genoa = 192 cores, Rome = 128. Own the whole node (`--exclusive` + high ntasks) →
  `E = DC_NODE_POWER × time` is exact (no co-tenants, no idle-core inflation).
- **Sweep** ntasks ∈ {1, 16, 32, whole-node} so parallelism is a measured feature, not a
  fixed assumption. Whole-node = the clean provisioned-energy anchor (the real bill).

---

## Workstreams (run in parallel; HPC jobs are the long pole → submit first)

**W1 · Submit new runs (long pole — submit tonight, run overnight)**
- Add `toxic_language_detection` (+ optionally `pii_masking`) to the YAML template → the
  WHOLE 6-stage pipeline.
- ntasks sweep {1, 16, 32} + one whole-node anchor.
- ≥2 corpora (am-stories + euro-parliament; add a 3rd if queue is fast).
- EAR on. Tag slugs so they never clobber existing runs.
- Fallback: existing ntasks=1 + dedup + Dmytro data is the *guaranteed* dataset.

**W2 · Test the monitor (`02_monitor.sh`)**
- Run it against a live job from W1; confirm it parses eacct (col26=TIME, col32=DC_POWER)
  and tracks energy live. This is a demo asset ("watch the estimate converge live").

**W3 · Train the neural net (local/fast, while jobs run)**
- Unified feature table: one row per run × stage with the features above.
- Ladder, honest baselines, grouped leave-one-(size/corpus)-out CV:
  OLS → **GBM (likely winner on tabular)** → small MLP/FT-Transformer.
- Pitch: predict energy for an *un-calibrated* config/corpus → removes calibration runs.
- Energy-irony guard: don't burn an H100 to fit ~100 rows; quantify train-cost vs saved.

**W4 · Fold fresh readings → final numbers**
- Morning: harvest W1 → retrain W3 → regenerate figures + the held-out table (now incl.
  toxic, ntasks, multi-corpus).

**W5 · Finished product (the "how they use it" demo)**
- One-command forecast tool: input (stage list, n, n_tasks, hardware, corpus) →
  predicted total energy + CI, translated to **kWh / € / kg CO₂**, per-stage breakdown,
  dedup-OOM flag. A tool the team can picture running before a launch.
- Deck arc: problem → final whole-pipeline results → live forecast demo → decisions it
  unlocks → OLS-vs-NN honest verdict → roadmap (pii_masking, Rome cross-hardware).

---

## Back-pocket answers (bulletproof, for the team Q&A)

- **Calibration split (2% vs 5%)?** Not a fixed %. It's a *stopping rule*: smallest set of
  sizes whose fit predicts the next held-out size within tolerance — operationally, stop
  when the slope stabilizes within its error bar / the EKF posterior interval < tolerance.
  ~1–2% for linear stages; dedup needs a large probe (super-linear). Stage-dependent.
- **100k vs 400k?** Redundant for linear stages (slope stable = thesis confirming itself —
  small calibration suffices); essential for dedup (curvature, OOMs at 1.4M).
- **Rome vs Genoa?** Different CPU generations (Zen 2 / 128c vs Zen 4 / 192c). Same n_tasks
  = different node fraction; different power baselines; different J/doc. Coefficients are
  **per-hardware** → can't pool without a hardware term → that's the cross-hardware model's
  job. Rome (teammate) + Genoa (me) = free cross-hardware dataset.
- **My config?** ntasks=1 clean law → now sweeping ntasks {16,32,whole-node}, genoa, EAR on,
  default stage params, sizes 1k/100k/400k/1.4M, am-stories + euro-parliament + Dmytro scaled.
