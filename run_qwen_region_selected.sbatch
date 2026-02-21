#!/bin/bash -l
#SBATCH --job-name=qwen_sel
#SBATCH --partition=h24gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/En_high_performing/logs/qwen_region_selected_%A.out
#SBATCH --error=/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/En_high_performing/logs/qwen_region_selected_%A.err
#SBATCH --account=OD-237777
#SBATCH --gres=gpu:1

set -euo pipefail

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

conda activate deepfake

if [[ -n "${SCRIPT_DIR_OVERRIDE:-}" ]]; then
  SCRIPT_DIR="${SCRIPT_DIR_OVERRIDE}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/run_qwen_region_selected.sbatch" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
elif [[ -n "${SLURM_JOB_SCRIPT:-}" && -f "${SLURM_JOB_SCRIPT}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "$(realpath "${SLURM_JOB_SCRIPT}")")" && pwd)"
elif [[ -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)"
else
  SCRIPT_DIR="$(pwd)"
fi

mkdir -p "${SCRIPT_DIR}/logs"
cd "${SCRIPT_DIR}"

MODEL_ID="${MODEL_ID:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/VLM/Qwen3-VL-8B-Thinking/}"
BACKEND="${BACKEND:-transformers}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-350}"
BATCH_SIZE="${BATCH_SIZE:-1}"

META_CSV="${META_CSV:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/img/region_phone_table_grid.csv}"
SPEC_ROOT="${SPEC_ROOT:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/img/specs/grid/}"
MFA_JSON_ROOT="${MFA_JSON_ROOT:-/scratch3/che489/Ha/interspeech/datasets/vocv4_mfa_aligned/}"
SYSTEM_FILE="${SYSTEM_FILE:-${SCRIPT_DIR}/prompts/region_forensics_system.txt}"
USER_TEMPLATE_FILE="${USER_TEMPLATE_FILE:-${SCRIPT_DIR}/prompts/region_forensics_user.txt}"

OUTPUT_BASE_DIR="${OUTPUT_BASE:-${OUTPUT_BASE_DIR:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/En_high_performing/new}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE_DIR}/selected_samples}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${OUTPUT_DIR}/qwen_region_outputs_selected.jsonl}"

SAMPLE_IDS_PRIMARY="${SAMPLE_IDS_PRIMARY:-hifigan_LA_D_1469404;hn-sinc-nsf-hifi_LA_D_5621016;hn-sinc-nsf_LA_D_3205943;waveglow_LA_D_1469404}"
SAMPLE_IDS_EXTRA="${SAMPLE_IDS_EXTRA:-hifigan_LA_D_6503475;hifigan_LA_D_5495040;hifigan_LA_T_2397067}"
SAMPLE_IDS="${SAMPLE_IDS:-${SAMPLE_IDS_PRIMARY};${SAMPLE_IDS_EXTRA}}"
FILTERED_META_CSV="${FILTERED_META_CSV:-${OUTPUT_DIR}/region_phone_table_selected.csv}"

mkdir -p "${OUTPUT_DIR}" "${OUTPUT_BASE_DIR}/logs"

# Build a filtered metadata CSV that keeps header + rows for selected sample IDs.
awk -F, -v ids="${SAMPLE_IDS}" '
BEGIN {
  n = split(ids, raw, ";");
  for (i = 1; i <= n; i++) {
    gsub(/^ +| +$/, "", raw[i]);
    if (raw[i] != "") keep[raw[i]] = 1;
  }
}
NR == 1 { print; next }
($1 in keep) { print }
' "${META_CSV}" > "${FILTERED_META_CSV}"

echo "[run] MODEL_ID=${MODEL_ID}"
echo "[run] SAMPLE_IDS_PRIMARY=${SAMPLE_IDS_PRIMARY}"
echo "[run] SAMPLE_IDS_EXTRA=${SAMPLE_IDS_EXTRA}"
echo "[run] SAMPLE_IDS(all)=${SAMPLE_IDS}"
echo "[run] FILTERED_META_CSV=${FILTERED_META_CSV}"
echo "[run] OUTPUT_DIR=${OUTPUT_DIR}"

python -u "${SCRIPT_DIR}/qwen_region_artifact_prompt.py" \
  --backend "${BACKEND}" \
  --model-id "${MODEL_ID}" \
  --meta-csv "${FILTERED_META_CSV}" \
  --spec-root "${SPEC_ROOT}" \
  --mfa-json-root "${MFA_JSON_ROOT}" \
  --system-file "${SYSTEM_FILE}" \
  --user-template-file "${USER_TEMPLATE_FILE}" \
  --dtype "${DTYPE}" \
  --attn-implementation "${ATTN_IMPLEMENTATION}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-shards 1 \
  --shard-id 0 \
  --output-dir "${OUTPUT_DIR}" \
  --output-jsonl "${OUTPUT_JSONL}" \
  --overwrite
