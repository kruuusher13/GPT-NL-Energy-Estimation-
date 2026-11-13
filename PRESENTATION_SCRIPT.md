# Presentation script

Five to six minutes, fast. The poster (`poster.pdf`) is the single slide. Read it the way the talk runs: the flow chart on top, the maths panel on the right, then the findings across the middle and bottom. The order below matches the poster exactly.

---

## 1. The flow chart (top) — what the system is (50s)

Point at the top diagram.

"Before a team commits a multi day job to clean a corpus for an LLM, they want one number: how much energy will the whole pipeline burn. The meter on the cluster cannot be trusted on the real runs, so I predict that number from cheap trials.

Read the flow chart left to right. Documents pass through four stages: splitting, normalisation, heuristic filtering, deduplication. Under each stage is a swappable model slot. The baseline in every slot is OLS, a two parameter line. The Extended Kalman filter along the bottom ties the four stages into one total: it predicts each stage, gates the incoming reading at three sigma, updates the total, and shrinks the interval. The red box, g, transfers the model to a corpus I have never run."

## 2. The maths (right panel) — explain it fast (70s)

Point at the maths side panel, top to bottom.

"The maths is three blocks. First, the OLS baseline. Each stage energy is a line, a fixed start up cost c naught plus a cost per document c one. I fit it in closed form, theta hat equals X transpose X inverse X transpose y, and because it is linear I get an exact prediction interval, the plus or minus t sigma root term. That analytic interval is the whole reason I keep the model linear.

Second, the Kalman walk. When a stage reports an energy z, the innovation r is the reading minus the prediction. The gate accepts it only if r squared is within three sigma of the predicted variance S. If it passes, the update nudges the estimate by the gain K times r, so clean readings tighten the total and contaminated ones are rejected and replaced by the model. And the stopping rule: stop sampling when the relative interval, 1.96 root variance over E hat, drops below a tolerance tau.

Third, g. It predicts the log of the slope c one for a new corpus from features, characters per document, survival rate, and compute counters. So you never refit from scratch, you predict the one coefficient that changes."

## 3. Findings — tie the maths to the data (3 min)

Now walk the findings, each one is the maths above proven on real runs across four corpora.

**Stopping rule (left graph), 25s.**
"This is the interval in action. As trials stream in, the 95 percent confidence ribbon shrinks. The stop test fires at three trials, with the prediction already within 7 percent of the true 400k run. You commit the budget without ever running the full size."

**Live monitor (centre), 35s.**
"This is the tool running on a real job. Before the run, a wide forecast band. As each stage reports, the Kalman update folds it in, the band collapses to plus or minus 40 kilojoules, a 5 percent interval, error 0.0 percent, zero readings rejected because they were clean. A contaminated shared node reading would trip the gate instead of poisoning the total. This single panel is predict, gate, update, converge."

**Where energy goes (bottom left), 25s.**
"A finding you only see by running the whole pipeline on several corpora: the dominant stage flips. For news, heuristic filtering is 77 percent of the energy. For code, string normalisation is 66 percent. There is no single stage to optimise, so the estimator has to model all four."

**Parallelism (bottom middle), 25s.**
"This one is operational. Cranking parallelism does not help and wrecks the energy budget. For news at 400k documents, sixteen tasks give no speedup and burn ten times the energy. For code it is different, the documents are large enough that sixteen tasks give a real speedup. So one task is energy optimal for small document corpora."

**Model results (bottom right), 30s.**
"And the question everyone asks: would a neural network do better. I trained the full ladder, gradient boosted trees, an MLP, and a 17,569 parameter transformer. On transfer to a corpus never seen, g, with two parameters per stage, wins at 88 percent error against 330, 362, and over 3000 for the networks. Architecture is not the bottleneck. The number of corpora is."

## 4. Tie it together and close (30s)

"So the whole poster is one loop. The flow chart is the architecture. The maths panel is the engine: an OLS line per stage, a Kalman filter that gates and stops, and g for transfer. The findings are that engine proven on four corpora that differ by 256 times in document length: predict the full run to a median 15.9 percent, stop after a few trials, survive a broken meter, and beat every trained network at transfer.

In one line: model what is cheap to calibrate, walk what you must predict, gate what you cannot trust, and the whole pipeline number stays honest. Thank you."

---

## Fast answers if pushed

- **Transfer errors look large.** Within a corpus, the operational case, the median is 15.9 percent. The large numbers are transfer to a corpus never run, with three corpora to train on. They fall as I add corpora.
- **Why time not power.** Shared node power is contaminated up to 2.4 times. Wall time is co tenancy invariant, so energy equals a calibrated effective power times time.
- **Why keep OLS with a transformer available.** The transformer slots into the same model slot for size extrapolation; the linear model wins at transfer and gives the analytic interval the filter needs.
- **The twelve queued corpora.** They did not emit usable telemetry yet. The model-results panel shows exactly why recovering them matters: corpus count, not model size, is the lever.
