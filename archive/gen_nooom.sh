#!/usr/bin/env bash
# NO-OOM full-reading sweep for the LARGE sizes that previously OOM'd on genoa
# (1 GB/cpu was far too little for the dedup signature pass at >=1.4M docs).
# Runs the WHOLE pipeline on fat_genoa (1.44 TB node, same Zen4 arch as genoa)
# with real memory, ntasks=1 (clean per-stage energy), so we finally get the
# deduplication reading at 1.4M and extra reps at 400k/700k.
# New tag 'amf' -> parsed as american_stories; does not clobber existing runs.
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$CONFIGS_DIR"
DR="/projects/0/prjs0986/energy_studies/datasets"

DATADIR="$DR/american-stories"
SIZES_BIG="400000 700000 1400000"
MEM="${MEM_BIG:-48}"                 # GB/cpu -> ample headroom for dedup at 1.4M
PART="${PARTITION:-fat_genoa}"
REPS="${REPS_BIG:-2}"
# generous walltime: the big stages (heuristic/dedup) at 1.4M are slow
TDS="01:30:00"; TSN="04:00:00"; THF="06:00:00"; TDD="08:00:00"

gen() {
  local size=$1 rep=$2
  local slug="amf_n1_size${size}_rep${rep}"
  cat > "$CONFIGS_DIR/run_${slug}.yaml" <<YAML
# gen_nooom.sh — fat_genoa high-mem full-reading run. size=${size} rep=${rep}
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
    input_folder: "${DATADIR}"
    hpc_partition: "${PART}"
    hpc_time: "${TDS}"
    hpc_n_tasks: "1"
    hpc_mem_per_cpu_gb: "${MEM}"
    ParquetReader:
      limit: "${size}"
  - stage: string_normalization
    hpc_partition: "${PART}"
    hpc_time: "${TSN}"
    hpc_n_tasks: "1"
    hpc_mem_per_cpu_gb: "${MEM}"
    FTFYFormatter:
      normalization: "NFC"
  - stage: heuristic_filtering
    hpc_partition: "${PART}"
    hpc_time: "${THF}"
    hpc_n_tasks: "1"
    hpc_mem_per_cpu_gb: "${MEM}"
    LanguageFilter:
      languages: ["en", "nl", "da", "sv", "af", "fy", "de"]
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
    hpc_partition: "${PART}"
    hpc_time: "${TDD}"
    hpc_n_tasks: "1"
    hpc_mem_per_cpu_gb: "${MEM}"
    MinhashConfig:
      num_buckets: 14
      hashes_per_bucket: 8
      n_grams: 5
YAML
  echo "  $slug  (part=$PART mem=${MEM}G)"
}

n=0
for size in $SIZES_BIG; do
  for rep in $(seq 1 "$REPS"); do gen "$size" "$rep"; n=$((n+1)); done
done
echo "[gen-nooom] generated $n configs (american-stories big sizes, $PART, ${MEM}G/cpu)"
