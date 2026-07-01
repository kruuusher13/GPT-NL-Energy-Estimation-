#!/usr/bin/env python3
"""Fix thesis.tex: chars/doc, EKF math, tables, abstract, conclusion."""
import re

with open("paper/thesis.tex", "r") as f:
    content = f.read()

fixes_applied = 0

# Fix 1: Fix any double-escaped LaTeX commands from earlier patch mess
content = content.replace("\\\\textbf{Abstract}", "\\textbf{Abstract}")
content = content.replace("\\\\%", "\\%")
content = content.replace("\\\\times", "\\times")

# Fix 2: Rewrite EKF subsection to match actual implementation
old_ekf = r"""\subsection{The Kalman walk}
\label{sec:ekf}

The pipeline is a chain. Stage $s$ outputs $n_{s+1} = r_s n_s$ documents where $r_s$ is the survival rate. We track a two-dimensional state vector:

\begin{equation}
  \mathbf{x}_s = \begin{bmatrix} n_s \\ E_{\mathrm{cum},s} \end{bmatrix}
  \label{eq:state}
\end{equation}

The predict step propagates the state using the calibrated model. After a stage completes, the EAR reports an energy reading $z$. The innovation is $y = z - H\hat{\mathbf{x}}$ with $H = [0,1]$.

\subsection{The innovation gate}

A reading is accepted only if it is statistically consistent with the model:

\begin{equation}
  y^2 \leq \gamma^2 S, \quad \gamma = 3
  \label{eq:gate}
\end{equation}

If the gate opens, we apply the Kalman update: the estimate shifts toward the reading and the prediction interval tightens. If the gate stays closed, we keep the model prediction and widen the uncertainty. This mechanism makes the estimator robust without human intervention."""

new_ekf = r"""\subsection{The sequential estimator}
\label{sec:ekf}

The pipeline executes each stage as $M_s$ parallel SLURM tasks. After task $k$ of stage $s$ completes, the EAR library records its energy $z_k$. We maintain a per-stage energy estimate $\hat{E}^{(s)}$ that blends the calibrated model prior with the streaming data using a Bayesian update rule.

The estimate after $k$ of $M_s$ tasks of stage $s$ have reported is:

\begin{equation}
  \hat{E}^{(s)}_k = w_k \cdot \left(\hat{E}^{(s)}_0 \cdot \frac{k}{M_s} + z_k \cdot \frac{M_s - k}{M_s}\right) + (1 - w_k) \cdot \hat{E}^{(s)}_0
  \label{eq:blend}
\end{equation}

where $w_k = k / (k + k_0)$ is the blending weight, $M_s$ is the total number of tasks in stage $s$, $\hat{E}^{(s)}_0$ is the model prior for stage $s$, and $k_0 = 2$ controls the speed of trust transition from model to data. When $k = 0$ (no readings), the estimate is the pure model prior. When $k = M_s$ (stage complete), the estimate is the sum of measured energies with only instrument noise.

The variance of the estimate shrinks as more tasks report:

\begin{equation}
  \mathrm{Var}(\hat{E}^{(s)}_k) = (1 - w_k)^2 \cdot P_0^{(s)} + \sigma^2_{\mathrm{sample}} \cdot \max(M_s - k, 0)
  \label{eq:variance}
\end{equation}

with prior variance $P_0^{(s)} = (0.45 \cdot \hat{E}^{(s)}_0)^2$ and sample variance $\sigma^2_{\mathrm{sample}}$ estimated from the variance of the $k$ readings seen so far. The total pipeline variance is the sum of per-stage variances, and the 95\% confidence half-width is $1.96 \sqrt{\sum_s \mathrm{Var}(\hat{E}^{(s)})}$.

\subsection{The innovation gate}

A per-task reading is accepted only if it is physically plausible — a single task's energy cannot exceed the entire stage's predicted budget:

\begin{equation}
  \frac{z_k}{\hat{E}^{(s)}_0} \leq \gamma, \quad \gamma = 2.5
  \label{eq:gate}
\end{equation}

If the gate triggers, the contaminated reading is replaced by the model's per-task share $\hat{E}^{(s)}_0 / M_s$, preventing a single inflated node from distorting the estimate. The gate is designed for shared-node contamination where co-tenant workloads inflate a node's power reading (up to $2.4\times$) without affecting its wall-clock time. On clean exclusive-node runs, the gate is never observed to fire on the dominant stage."""

if old_ekf in content:
    content = content.replace(old_ekf, new_ekf)
    fixes_applied += 1
    print("[OK] EKF section rewritten")

# Fix 3: Per-character k table with verified values
old_ktable = r"""\begin{table}[ht]
\centering
\small
\begin{tabular}{lrrrr}
\toprule
Stage & amnews & github & german & euparl \\
\midrule
String normalization & 1.33e-4 & 3.50e-4 & 4.73e-5 & 4.48e-4 \\
Deduplication & 2.41e-4 & 3.8e-6 & 1.85e-5 & 1.34e-4 \\
\bottomrule
\end{tabular}
\caption{Per-character energy coefficient $k = c_1 / \ell$ (J/char) across four corpora. String normalization shows the strongest invariance.}
\label{tab:chars}
\end{table}

String normalization is the stage that processes text character-by-character. Its per-character coefficient spans 4.73e-5 to 4.48e-4 J/char across a 100x document length range. This is useful as a first-order transfer law."""

new_ktable = r"""\begin{table}[ht]
\centering
\small
\begin{tabular}{lrrrr}
\toprule
Stage & amnews & github & german & euparl \\
\midrule
String normalization & 2.24e-4 & 1.73e-4 & 2.55e-4 & 2.43e-5 \\
Heuristic filtering & 8.18e-4 & 1.71e-4 & 8.79e-4 & 5.16e-5 \\
Deduplication & 1.63e-4 & 1.79e-6 & 1.08e-4 & 5.14e-6 \\
\bottomrule
\end{tabular}
\caption{Per-character energy coefficient $k = c_1 / \ell$ (J/char) across four corpora. Computed from ntasks=1 exclusive-node measurements.}
\label{tab:chars}
\end{table}

The per-character coefficients vary more than a simple constant model would predict. Within string normalization, $k$ spans one order of magnitude ($2.43 \times 10^{-5}$ to $2.55 \times 10^{-4}$ J/char). The per-character transfer law provides a first-order approximation (mean $k$ predicts within 60\% MRE on held-out corpora, Section \ref{tab:g}), but corpus-specific factors---character encoding density, Unicode complexity, and cache behaviour---cause real differences. The learned model g captures these by incorporating compute counters (CPI, GFLOPS, I/O throughput) as additional features."""

if old_ktable in content:
    content = content.replace(old_ktable, new_ktable)
    fixes_applied += 1
    print("[OK] Per-character k table updated")

# Fix 4: Add per-stage physics row to model comparison table
old_models = r"""Linear (OLS) & 278\% & 483\% \\
Ridge & 277\% & 479\% \\
GBM & 153\% & 330\% \\
MLP & 111\% & 3,272\% \\
FT-Transformer & 101\% & 362\% \\"""
new_models = r"""Linear (OLS, no corpus features) & 278\% & 483\% \\
Ridge & 277\% & 479\% \\
GBM & 153\% & 330\% \\
MLP & 111\% & 3,272\% \\
FT-Transformer & 101\% & 362\% \\
\midrule
Per-stage physics + g (this work) & 10.3\% & 87.9\% \\"""
if old_models in content:
    content = content.replace(old_models, new_models)
    fixes_applied += 1
    print("[OK] Model comparison table updated")

# Fix 5: Update the calibration text with verified coefficients
old_calib_text = r"""String normalization on American Stories costs 0.44 J/doc. On German public domain, with 530k characters per document, it costs 25.1 J/doc. This is the per-character invariant at work."""
new_calib_text = r"""String normalization on American Stories costs 0.40 J/doc. On German public domain, with 48,881 characters per document, it costs 12.5 J/doc. The energy ratio (31$\times$) approximately tracks the document-length ratio (27$\times$), consistent with a per-character cost that is roughly stable."""
if old_calib_text in content:
    content = content.replace(old_calib_text, new_calib_text)
    fixes_applied += 1
    print("[OK] Calibration text updated")

# Fix 6: Update conclusion - EKF description and numbers
old_conc_ekf = r"""then walk the live pipeline with an EKF that gates contaminated readings."""
new_conc_ekf = r"""then walk the live pipeline with a sequential Bayesian estimator that gates contaminated readings."""
content = content.replace(old_conc_ekf, new_conc_ekf)

old_char_conc = r"""Energy per character is approximately 1.7e-4 J/char across a 100x document length range."""
new_char_conc = r"""Energy per character varies from $2.4 \times 10^{-5}$ to $8.8 \times 10^{-4}$ J/char across corpora and stages, with string normalization showing the tightest spread (one order of magnitude)."""
content = content.replace(old_char_conc, new_char_conc)

old_ger_conc = r"""2.2 MJ for American Stories at 400k, up to 18 MJ for German public domain at 200k."""
new_ger_conc = r"""2.2 MJ for American Stories at 400k documents, scaling with document count and document length across corpora."""
content = content.replace(old_ger_conc, new_ger_conc)

# Fix 7: Add note about per-stage physics vs naive OLS in model comparison text
old_model_text = r"""Linear models perform well within a corpus but struggle with cross-corpus transfer. GBM is the best learner at both tasks. MLP and FT-Transformer overfit on the limited number of training corpora. The coefficient model g, with 2 parameters per stage, beats every trained network at corpus transfer (87.9\% MRE in Section \ref{tab:g}) because the physics backbone constrains the prediction."""
new_model_text = r"""The naive OLS/GBM/MLP/FT-Transformer models are trained on a flat feature table without the per-stage physics structure. They perform poorly on both tasks because they must learn both the linear energy law and the per-stage identity from data. In contrast, the per-stage physics model (this work) applies the calibrated $E = c_0 + c_1 n$ law per stage with the coefficient model g for cross-corpus transfer, achieving 10.3\% MRE on size extrapolation and 87.9\% MRE on corpus transfer. The key architectural difference is that the physics backbone constrains the prediction to a known functional form, reducing the learning problem to coefficient prediction rather than function discovery."""
if old_model_text in content:
    content = content.replace(old_model_text, new_model_text)
    fixes_applied += 1
    print("[OK] Model comparison text updated")

# Fix 8: Update "five-stage" references throughout
content = content.replace("five-stage GPT-NL curation pipeline", "GPT-NL curation pipeline")
content = content.replace("five-stage GPT-NL data curation pipeline", "GPT-NL data curation pipeline")

# Fix 9: Update the forecast tool output numbers
old_forecast = r"""At 400k documents on American Stories, the tool reports: total pipeline 2.2 MJ (0.73 kWh), energy cost EUR 0.26 at EUR 0.30/kWh with PUE 1.20, carbon 0.26 kg CO2, per-stage breakdown with confidence bands, and a warning if deduplication exceeds memory limit."""
new_forecast = r"""At 400k documents on American Stories, the tool reports: total pipeline 1.0 MJ (0.33 kWh with PUE 1.20), energy cost EUR 0.10 at EUR 0.30/kWh, carbon 0.10 kg CO$_2$, per-stage breakdown with 95\% confidence bands, and a warning if deduplication exceeds its memory limit at 1.4M documents."""
content = content.replace(old_forecast, new_forecast)

# Fix 10: Add GPT-NL corpus citation to the Data section
old_data_start = r"""All measurements were collected on the Snellius genoa partition (AMD EPYC 9654, 192 cores per node). We ran the GPT-NL curation pipeline \cite{gptnl_architecture} implemented with Datatrove \cite{datatrove} across four corpora from the GPT-NL Public Corpus \cite{gptnl_corpus}:"""
# This should already be there from the earlier patch

# Write back
with open("paper/thesis.tex", "w") as f:
    f.write(content)

print(f"\nTotal fixes applied: {fixes_applied}")
print("Thesis written to paper/thesis.tex")
