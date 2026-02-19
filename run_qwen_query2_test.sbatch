#!/bin/bash -l
#SBATCH --job-name=qwen_q2_test
#SBATCH --partition=h24gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4
#SBATCH --output=logs/qwen_query2_test_%A_%a.out
#SBATCH --account=OD-237777

set -euo pipefail

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi
conda activate deepfake

if [[ -n "${SCRIPT_DIR_OVERRIDE:-}" ]]; then
  SCRIPT_DIR="${SCRIPT_DIR_OVERRIDE}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/run_qwen_query2_test.sbatch" ]]; then
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

MODEL_KEY="${MODEL_KEY:-qwen3_30b_a3b_instruct}"
MODEL_ID="${MODEL_ID:-}"
if [[ -n "${MODEL_TAG:-}" ]]; then
  RESOLVED_MODEL_TAG="${MODEL_TAG}"
elif [[ -n "${MODEL_ID}" ]]; then
  RESOLVED_MODEL_TAG="$(basename "${MODEL_ID%/}")"
else
  RESOLVED_MODEL_TAG="${MODEL_KEY}"
fi

META_CSV="${META_CSV:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/SFT_2turn/stage1_gt_with_transcript.csv}"
IMAGE_FOLDER="${IMAGE_FOLDER:-}"
IMG_STEM_CONTAINS="${IMG_STEM_CONTAINS:-_LA_D_}"
SYSTEM_FILE="${SYSTEM_FILE:-${SCRIPT_DIR}/query2_prompts/query2_system.txt}"
USER_TEMPLATE_FILE="${USER_TEMPLATE_FILE:-${SCRIPT_DIR}/query2_prompts/query2_user.txt}"

OUTPUT_BASE="${OUTPUT_BASE:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/ZS_Q2}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE}/${RESOLVED_MODEL_TAG}_test}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${OUTPUT_DIR}/qwen_query2_outputs.jsonl}"

SHARD_COUNT="${SHARD_COUNT:-1}"
SHARD_ID="${SHARD_ID:-${SLURM_ARRAY_TASK_ID:-0}}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-600}"
BATCH_SIZE="${BATCH_SIZE:-1}"
OVERWRITE="${OVERWRITE:-0}"
DO_SAMPLE="${DO_SAMPLE:-0}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"

mkdir -p "${OUTPUT_DIR}"

echo "[run] SCRIPT_DIR=${SCRIPT_DIR}"
echo "[run] MODEL_KEY=${MODEL_KEY}"
echo "[run] MODEL_ID=${MODEL_ID}"
echo "[run] MODEL_TAG=${RESOLVED_MODEL_TAG}"
echo "[run] META_CSV=${META_CSV}"
echo "[run] IMG_STEM_CONTAINS=${IMG_STEM_CONTAINS}"
echo "[run] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[run] OUTPUT_JSONL=${OUTPUT_JSONL}"
echo "[run] SHARD_ID=${SHARD_ID} SHARD_COUNT=${SHARD_COUNT}"

CMD=(python -u "${SCRIPT_DIR}/qwen_query2_artifact_prompt.py" \
  --model-key "${MODEL_KEY}" \
  --meta-csv "${META_CSV}" \
  --image-folder "${IMAGE_FOLDER}" \
  --img-stem-contains "${IMG_STEM_CONTAINS}" \
  --system-file "${SYSTEM_FILE}" \
  --user-template-file "${USER_TEMPLATE_FILE}" \
  --dtype "${DTYPE}" \
  --attn-implementation "${ATTN_IMPLEMENTATION}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --batch-size "${BATCH_SIZE}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --num-shards "${SHARD_COUNT}" \
  --shard-id "${SHARD_ID}" \
  --output-dir "${OUTPUT_DIR}" \
  --output-jsonl "${OUTPUT_JSONL}")

if [[ -n "${MODEL_ID}" ]]; then
  CMD+=(--model-id "${MODEL_ID}")
fi

if [[ "${DO_SAMPLE}" == "1" ]]; then
  CMD+=(--do-sample)
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  CMD+=(--overwrite)
fi

CMD+=("$@")
echo "[run] ${CMD[*]}"
"${CMD[@]}"
