#!/bin/bash -l
#SBATCH --job-name=qwen_region_wakefake
#SBATCH --partition=h24gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4
#SBATCH --array=0-7
#SBATCH --output=logs/qwen_region_wakefake_%A_%a.out
#SBATCH --account=OD-237777

set -euo pipefail

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi
conda activate deepfake

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "${SCRIPT_DIR}/logs"
cd "${SCRIPT_DIR}"

SHARD_COUNT="${SHARD_COUNT:-8}"
SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"

MODEL_ID="${MODEL_ID:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/VLM/Qwen3-VL-8B-Thinking/}"
META_CSV="${META_CSV:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/wakefake/region_phone_table_top3.csv}"
SPEC_ROOT="${SPEC_ROOT:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/wakefake/Ms_region_outputs/grid/}"
MFA_JSON_ROOT="${MFA_JSON_ROOT:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/wakefake/mfa_json/}"

SYSTEM_FILE="${SYSTEM_FILE:-${SCRIPT_DIR}/prompts/region_forensics_system.txt}"
USER_TEMPLATE_FILE="${USER_TEMPLATE_FILE:-${SCRIPT_DIR}/prompts/region_forensics_user.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch3/che489/Ha/interspeech/VLM/Qwen3-VL/outputs}"

DTYPE="${DTYPE:-auto}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-600}"
BATCH_SIZE="${BATCH_SIZE:-2}"
DO_SAMPLE="${DO_SAMPLE:-0}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"

EXTRA_ARGS=()
if [[ "${DO_SAMPLE}" == "1" ]]; then
  EXTRA_ARGS+=(--do-sample)
fi

python "${SCRIPT_DIR}/qwen_region_artifact_prompt.py" \
  --model-id "${MODEL_ID}" \
  --meta-csv "${META_CSV}" \
  --spec-root "${SPEC_ROOT}" \
  --mfa-json-root "${MFA_JSON_ROOT}" \
  --system-file "${SYSTEM_FILE}" \
  --user-template-file "${USER_TEMPLATE_FILE}" \
  --dtype "${DTYPE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --batch-size "${BATCH_SIZE}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --num-shards "${SHARD_COUNT}" \
  --shard-id "${SHARD_ID}" \
  --output-dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}" \
  "$@"