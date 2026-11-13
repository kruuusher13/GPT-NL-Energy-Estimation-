#!/usr/bin/env bash
# Broad multi-corpus sweep — training data for the coefficient model g.
# ntasks=1 (clean per-stage signal), generous mem (no probe needed, won't OOM),
# conservative sizes, 2 reps. Covers the 12 corpora we don't already have.
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$CONFIGS_DIR"
DR="/projects/0/prjs0986/energy_studies/datasets"

# tag -> dataset dir (the 12 corpora not yet measured)
declare -A CDIR=(
  [aud]="$DR/auditdienstrijk" [bel]="$DR/belgian-journal" [eng]="$DR/cc_english_pd"
  [oal]="$DR/cc_openalex"     [dan]="$DR/dansknaw"        [dpc]="$DR/dpc"
  [kbb]="$DR/kb-pd-books"     [nat]="$DR/naturalis"       [rec]="$DR/rechtspraak"
  [twk]="$DR/tweedekamer"     [utr]="$DR/utrechtsarchief" [woo]="$DR/woogle"
)
SIZES="1000 5000 20000"          # small enough to never OOM even for big-doc corpora
MEM="16"                          # generous, universal
REPS=2

gen() {
  local tag=$1 datadir=$2 size=$3 rep=$4
  local slug="${tag}_n1_size${size}_rep${rep}"
  cat > "$CONFIGS_DIR/run_${slug}.yaml" <<YAML
# gen_allcorpora.sh — broad corpus sweep for g. corpus=${tag} size=${size} rep=${rep}
processing_type: hpc
input_from_previous_output: true
output_folder_template: "${RUNS_DIR}/${slug}/{pipeline_iteration}/stage{stage_idx}_{stage_name}"
logs_folder: "{output_folder}/logs"
slurm_logs_folder: "{output_folder}/slurm_logs"
mail_type: NONE
mail_user:
hpc_ear: "true"
stages:
  - stage: data_splitting
    input_folder: "${datadir}"
    hpc_partition: "${PARTITION}"
    hpc_time: "${TIME_DATA_SPLITTING}"
    hpc_n_tasks: "1"
    hpc_mem_per_cpu_gb: "${MEM}"
    ParquetReader:
      limit: "${size}"
  - stage: string_normalization
    hpc_partition: "${PARTITION}"
    hpc_time: "${TIME_STRING_NORM}"
    hpc_n_tasks: "1"
    hpc_mem_per_cpu_gb: "${MEM}"
    FTFYFormatter:
      normalization: "NFC"
  - stage: heuristic_filtering
    hpc_partition: "${PARTITION}"
    hpc_time: "${TIME_HEURISTIC}"
    hpc_n_tasks: "1"
    hpc_mem_per_cpu_gb: "${MEM}"
    LanguageFilter:
      languages: ["en", "nl", "da", "sv", "af", "fy", "de", "fr"]
      language_threshold: "0.65"
      backend: "ft176"
    NordicPileQualityFilter:
      max_digit_fraction: "0.2"
      min_n_char: "50"
      min_mean_med_char: "9"
      min_mean_med_word: "2.1"
    GopherQualityFilter:
      max_symbol_word_ratio: "0.1"
      max_bullet_lines_ratio: "0.9"
      max_ellipsis_lines_ratio: "0.3"
      max_non_alpha_words_ratio: "0.8"
      min_stop_words: "2"
    GopherRepetitionFilter:
      dup_line_frac: "0.35"
      dup_para_frac: "0.35"
      dup_line_char_frac: "0.2"
      dup_para_char_frac: "0.2"
      top_n_grams: [[2, 0.25], [3, 0.23], [4, 0.21]]
      dup_n_grams: [[5, 0.20], [6, 0.19], [7, 0.18], [8, 0.17], [9, 0.16], [10, 0.15]]
  - stage: deduplication
    hpc_partition: "${PARTITION}"
    hpc_time: "${TIME_DEDUP}"
    hpc_n_tasks: "1"
    hpc_mem_per_cpu_gb: "${MEM}"
    MinhashConfig:
      num_buckets: 14
      hashes_per_bucket: 8
      n_grams: 5
YAML
  echo "  $slug"
}

n=0
for tag in "${!CDIR[@]}"; do
  for size in $SIZES; do
    for rep in $(seq 1 "$REPS"); do gen "$tag" "${CDIR[$tag]}" "$size" "$rep"; n=$((n+1)); done
  done
done
echo "[gen-all] generated $n configs (12 corpora x 3 sizes x $REPS reps)"
