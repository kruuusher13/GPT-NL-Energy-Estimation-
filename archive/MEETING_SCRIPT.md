# Meeting script — full-pipeline energy for GPT-NL curation
_Romir Malik · 2026-06-09 · Julio (supervisor) · Martino (EAR/infra) · Dmytro Kravchuk_

A speaking script. It walks the **whole pipeline** — all four stages, including
deduplication — and the energy account we built for it. **Julio's four decisions**
are flagged at the moment each comes up and collected at the end. Tables are plain
text so nothing renders blank. Bold numbers are the ones to say out loud.

**One-sentence version (say this first):** *"We ran the complete four-stage
curation pipeline and built a predictor for its total energy from a few minutes of
cheap calibration — and it stays accurate even when the cluster's telemetry is
contaminated or missing, which for the deduplication stage it was."*

---

## 0. Opening (30 seconds)

> "The deliverable is one number with an honest error bar: before we launch a
> multi-day curation run on a new corpus, what does the *whole* pipeline cost in
> energy? Not three convenient stages — the whole thing, deduplication included,
> because that's the number we actually have to budget. I'll walk the four stages,
> the validation, the two telemetry problems we hit and solved, and then I have four
> decisions for you, Julio, and five EAR questions for Martino."

---

## 1. Why this is hard (1 minute)

We run the GPT-NL curation pipeline on Snellius, instrumented with EAR. The need is
simple: a cheap, pre-run estimate of total energy across all four stages. Two things
get in the way, and we **measured** both — we didn't assume them:

- **Contaminated telemetry.** On a shared node, EAR's DC power is node-level, so it
  includes the neighbours. Same job exclusive vs shared reads **2.4×** the power.
- **Missing telemetry.** Deduplication runs as four sub-jobs that emitted *no* energy
  record at all — a launcher never created the directory EAR writes to.

> "A naive estimator that trusts every reading inherits both: a **+93%** over-estimate
> when it swallows a contaminated reading, and an incomplete total when a stage goes
> dark. The whole point is a full-pipeline number that survives both."

---

## 2. The method spine — calibrate, then predict (1 minute)

The framing you signed off on, Julio. For **each** of the four stages:

1. **Calibrate** a two-parameter model `E = c₀ + c₁·n` on a few *small, cheap,
   exclusive* runs (minutes).
2. **Predict** the energy of the *large* full run we never measured.
3. Pipeline total = sum of all four stage predictions.

---

## 3. The four stages — what each does (1 minute)

| Stage | What it does |
|---|---|
| `data_splitting` | Shards the raw corpus into per-document records |
| `string_normalization` | Per-document text normalisation |
| `heuristic_filtering` | FastText language detection + quality heuristics |
| `deduplication` | MinHash near-dup removal: signature → buckets → cluster → filter |

All four were swept at {1k, 100k, 400k, 1.4M} documents, two replicates each.
(`data_splitting` and `string_normalization` each run as two concurrent tasks — we
sum them.) **Dmytro's scaled American Stories datasets (10/25/50/100%) are what made
the multi-size sweep possible** — worth crediting up front.

---

## 4. Result 1 — in-sample fit is excellent, and that's the trap (2 minutes)

Fit the line on all sizes and every stage looks solved:

| Stage | c₀ (J) | c₁ (J/doc) | in-sample R² |
|---|---|---|---|
| `data_splitting` | 2,308 | 0.038 | 0.9998 |
| `string_normalization` | 14,308 | 0.402 | 0.9977 |
| `heuristic_filtering` | 36,285 | 1.471 | 0.9973 |
| `deduplication` | 7,891 | 0.293 | 0.9988 |

The honest test is **held-out**: calibrate on the cheap {1k, 100k}, predict the
expensive ones.

| Stage | in-sample R² | held-out R² | held-out MRE |
|---|---|---|---|
| `data_splitting` | 0.999 | 0.868 | 14.4% |
| `string_normalization` | 0.997 | 0.730 | 15.5% |
| `heuristic_filtering` | 0.997 | **0.955** | **8.1%** |
| `deduplication` | 0.999 | 0.960 | 12.1% |

> "The picture changes. `heuristic_filtering` predicts the full run well — 0.955, 8%.
> `deduplication` is right behind at 0.96. `string_normalization` drops to 0.73
> because it's genuinely sub-linear; `data_splitting`'s 14% miss is one noisy 1k
> point (cold cache). **The gap between 0.997 and 0.73 is the finding** — held-out
> tells you not just *that* a stage is hard but *which* fix it needs."

> 🟦 **JULIO — DECISION 1.** "Validation regime: calibrate {1k,100k} → predict
> {400k,1.4M}, held-out R²/MRE. Headline that, or switch to leave-one-out? I checked
> leave-largest-out — same ordering, so the story is robust; I just need you to pick
> the one we report."

---

## 5. Result 2 — deduplication: the stage that hid, and the stage that breaks (2 min)

This is the stage that gave us the most trouble and the most signal.

**It hid.** Its four sub-jobs emitted no EAR record — the launcher didn't make the
directory EAR writes to. **One `mkdir` per sub-job fixes it.** Then we swept it on
exclusive nodes (clean power):

| Documents | Dedup energy (kJ) | signature share |
|---|---|---|
| 1k | 9.9 | 35% |
| 100k | **34.8** | 73% |
| 400k | 125.6 | 84% |

> "Two things. The energy concentrates in the `signature` pass — it reads and hashes
> every document — growing to 84% at scale, so a dedup model only has to nail
> signature. And the contamination story repeats: an earlier single dedup run on a
> *shared* node reported **63.5 kJ** at 100k — a **1.8× over-read** of the clean 34.8.
> So dedup energy, like every stage, has to come off clean power, never raw node
> power."

**It breaks.** The 1.4M-document run **runs out of memory** in the signature pass and
is killed after ~17 minutes.

> "This is the payoff of running the *whole* pipeline instead of three convenient
> stages: deduplication is the one stage that doesn't scale to 1.4M — the MinHash
> signature builder holds everything in memory and OOMs. A three-stage study never
> sees it. It's a real engineering wall, and it bounds where our dedup model is valid
> today: 1k–400k, pending a spill-to-disk fix."

> 🟦 **JULIO — DECISION 2.** "Scope: per-stage models summed to a full-pipeline total,
> the EKF for missing/contaminated readings, the time-driven shared-node resolution,
> and the dedup OOM as a named result — is that the SustaiNLP target?"

---

## 6. The estimator — an EKF that survives broken telemetry (2.5 minutes)

Calibrate→predict is the spine; the novelty is the sequential filter wrapped around
it. Three modes, chosen automatically:

- **Mode A — predict.** Pre-run total with a confidence interval, zero telemetry.
- **Mode B — update.** A clean reading arrives → Kalman-correct, tighten downstream.
- **Mode C — reject.** A missing or contaminated reading is caught by a 3σ
  **innovation gate**, rejected, and replaced by the model with a wider band.

Real four-stage run, 100k documents:

| Mode | Result |
|---|---|
| A — pre-run | total **273 kJ**, 95% CI [208, 338], zero telemetry |
| B — online | converges **273 → 253 kJ**; CI half-width tightens 65 → 7 kJ |
| Gate vs contamination | naive swallows the 2.4× reading → **+93%** error; **gated EKF → +8%** |

> "The headline is the last row: **the gate turns a +93% telemetry error into +8%**,
> with no human flagging anything. [Point to figure: left, the band tightening as
> clean stages land; right, the contaminated reading rejected instead of inflating
> the total.] The clean dedup number, by the way, is why the total came down to 273
> from the contaminated 296 we had before."

---

## 7. Why shared-node power is unusable — the controlled experiment (1.5 min)

Same stage, same size, exclusive vs shared:

| Run | DC power | Wall time | CPU util | Energy |
|---|---|---|---|---|
| Exclusive | 275 W | 595 s | 97% | 164 kJ |
| Shared | **669 W** | 595 s | 97% | **398 kJ** |

> "Identical work, identical time, 2.4× the power — that's the neighbours. SLURM's own
> docs say only exclusive allocation gives accurate per-job energy. This is a measured
> fact, and it decides the architecture."

---

## 8. The shared-node fix — drive from time, not power (2 minutes)

We can't use exclusive in production, and calibrating on shared *power* gives **45.8%**
error. Two measured facts fix it:

- **No clean per-job power exists in any regime** — a power model on the clean
  counters hits only **R² = 0.08** (node idle baselines swing 232–640 W and bury the
  job's 50–90 W).
- **Energy tracks wall-time at R² = 0.99**, ~290 W, and wall-time is
  **co-tenancy-invariant** (the 595 s = 595 s above). So `E = P_eff × time`.

| Stage | P_eff (W) | Power-based MRE | Time-based MRE |
|---|---|---|---|
| `data_splitting` | 293 | 54.2% | 12.9% |
| `string_normalization` | 297 | 20.3% | 3.0% |
| `heuristic_filtering` | 278 | 63.1% | 0.2% |
| **Overall** | ~290 | 45.8% | **5.4%** |

> "Driving from time cuts shared-node error **45.8% → 5.4%**, and lets the EKF ingest
> every shared run instead of discarding it. **Exclusive calibration is needed zero
> times going forward** — the contamination finding becomes the reason the right
> observable is time, not power."

---

## 9. Cross-corpus — characters, not documents (1.5 minutes)

A single corpus can't tell you if energy scales with documents or characters (they're
collinear). European Parliament has **256× longer documents** than American Stories:

| Stage | am-stories (J/char) | euparl (J/char) |
|---|---|---|
| `string_normalization` | 4.5×10⁻⁴ | 3.9×10⁻⁴ |

> "Energy per character agrees to **12%** across a 256× document-length gap. So the
> transferable feature is **characters** — calibrate once, read the new corpus's mean
> doc length, predict. That's the result a single-corpus study can't produce."

> 🟦 **JULIO — DECISION 3.** "Endorse *characters* as the headline transferable
> feature? Caveat: euparl is one size, one replicate so far — strong pattern, not yet
> a law."

---

## 10. What's solid, what's next (1 minute)

> "Scope: a SustaiNLP @ EMNLP short paper — a method, an honest held-out evaluation
> across all four stages, a non-obvious finding (characters transfer; dedup OOMs at
> scale), and a reusable systems result (drive shared-node energy from wall-time).
> What's next, in priority order: (1) the dedup memory fix so we can reach 1.4M; (2)
> three replicates per cell; (3) a denser multi-corpus sweep; (4) the GPU stages."

> 🟦 **JULIO — DECISION 4.** "Next campaign: **3 replicates per size** to average
> co-tenancy noise (matches Dmytro's plan) — budget approved?"

---

## Julio — your four decisions, in one place

1. **Validation regime** — calibrate {1k,100k} → predict {400k,1.4M}, held-out R²/MRE.
   Headline this or leave-one-out? *(Recommend held-out — the honest "predict the full
   run from a sample" story.)*
2. **Paper scope/target** — four-stage total + EKF + time-driven resolution + dedup OOM,
   aimed at SustaiNLP @ EMNLP. Confirm?
3. **Characters as the headline feature** — endorse? *(Portable prior; caveat euparl = 1
   size.)*
4. **3 replicates per size** next campaign — budget approved?

## Martino — five EAR questions (end of meeting)

1. Is there a genuinely **per-job / per-cgroup power** field (per-job RAPL via EARL),
   distinct from node `DC_NODE_POWER`? *(Decides whether shared-node per-job energy is
   even possible.)*
2. Can we use **`--exclusive`** on genoa for the calibration campaign as standard
   practice — any charging/policy implication?
3. Can we read a node **idle power** to subtract unused-core draw from light stages?
4. **Dedup fix** — the per-sub-job `mkdir` got EAR records out; is that the sanctioned
   way, or is there a proper EAR hook for chained datatrove sub-steps?
5. **Dedup at 1.4M OOMs** in the signature pass (~500 MB RSS, killed at ~17 min) — is
   there a recommended large-memory or spill configuration for the MinHash stage on
   genoa?

---

## Closing line

> "So: a cheap calibration predicts the full four-stage run; characters make it
> portable; an innovation gate plus a time-driven estimate keep it accurate when the
> telemetry is contaminated or missing — which for dedup it was. Headline numbers:
> +93% error down to +8%, shared-node error 46% down to 5%, and the whole pipeline
> accounted for — including the stage that hid and the stage that breaks. Julio, your
> four calls; Martino, the EAR questions. That unblocks the next campaign."
