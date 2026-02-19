#!/bin/bash -l
#SBATCH --job-name=qwen_baseline
#SBATCH --partition=h24gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --output=logs/qwen_baseline_%A_%a.out
#SBATCH --account=OD-237777
#SBATCH --gres=gpu:4

set -euo pipefail

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

conda activate deepfake

if [[ -n "${SCRIPT_DIR_OVERRIDE:-}" ]]; then
  SCRIPT_DIR="${SCRIPT_DIR_OVERRIDE}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/run_qwen_baseline_stage1.sbatch" ]]; then
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

MODEL_ID="${MODEL_ID:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/VLM/Qwen3-VL-235B-A22B-Instruct/}"
META_CSV="${META_CSV:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/stage1_gt.csv}"
SYSTEM_FILE="${SYSTEM_FILE:-${SCRIPT_DIR}/baseline_prompts/baseline_system.txt}"
USER_TEMPLATE_FILE="${USER_TEMPLATE_FILE:-${SCRIPT_DIR}/baseline_prompts/baseline_user.txt}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_strongVLM/}"
SHARD_COUNT="${SHARD_COUNT:-1}"
SHARD_ID="${SHARD_ID:-${SLURM_ARRAY_TASK_ID:-0}}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-200}"
DTYPE="${DTYPE:-auto}"
OVERWRITE="${OVERWRITE:-0}"
SAMPLE_ID_GLOB="${SAMPLE_ID_GLOB:-}"

_model_basename="$(basename "${MODEL_ID%/}")"
MODEL_TAG="${MODEL_TAG:-$(echo "${_model_basename}" | tr -cs 'A-Za-z0-9._-' '_')}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID:-manual}}"
RUN_OUTPUT_ROOT="${OUTPUT_BASE_DIR%/}/${MODEL_TAG}/${RUN_TAG}"

if [[ "${SHARD_COUNT}" -gt 1 ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-${RUN_OUTPUT_ROOT}/shard_${SHARD_ID}_of_${SHARD_COUNT}}"
else
  OUTPUT_DIR="${OUTPUT_DIR:-${RUN_OUTPUT_ROOT}}"
fi
OUTPUT_JSONL="${OUTPUT_JSONL:-${OUTPUT_DIR%/}/qwen_baseline_outputs.jsonl}"

echo "[run] SCRIPT_DIR=${SCRIPT_DIR}"
echo "[run] MODEL_ID=${MODEL_ID}"
echo "[run] MODEL_TAG=${MODEL_TAG}"
echo "[run] RUN_TAG=${RUN_TAG}"
echo "[run] META_CSV=${META_CSV}"
echo "[run] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[run] SHARD_ID=${SHARD_ID} SHARD_COUNT=${SHARD_COUNT}"
echo "[run] SAMPLE_ID_GLOB=${SAMPLE_ID_GLOB}"
echo "[run] OVERWRITE=${OVERWRITE}"

CMD=(python -u "${SCRIPT_DIR}/qwen_baseline_artifact_prompt.py" \
  --model-id "${MODEL_ID}" \
  --meta-csv "${META_CSV}" \
  --image-folder "${IMAGE_FOLDER:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/img/specs/grid}" \
  --system-file "${SYSTEM_FILE}" \
  --user-template-file "${USER_TEMPLATE_FILE}" \
  --dtype "${DTYPE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --num-shards "${SHARD_COUNT}" \
  --shard-id "${SHARD_ID}" \
  --output-dir "${OUTPUT_DIR}" \
  --output-jsonl "${OUTPUT_JSONL}")

if [[ -n "${SAMPLE_ID_GLOB}" ]]; then
  CMD+=(--sample-id-glob "${SAMPLE_ID_GLOB}")
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  CMD+=(--overwrite)
fi

CMD+=("$@")
echo "[run] ${CMD[*]}"
"${CMD[@]}"
