#!/bin/bash -l
#SBATCH --job-name=qwen_En
#SBATCH --partition=h24gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/qwen_region_full_%A_%a.out
#SBATCH --account=OD-237777
#SBATCH --gres=gpu:4

set -euo pipefail

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

conda activate deepfake
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-info}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -n "${SCRIPT_DIR_OVERRIDE:-}" ]]; then
  SCRIPT_DIR="${SCRIPT_DIR_OVERRIDE}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/run_qwen_region_full.sbatch" ]]; then
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

# Defaults mirror qwen_region_artifact_prompt.py; override via env vars if needed.
MODEL_ID="${MODEL_ID:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/VLM/Qwen3-VL-8B-Thinking/}"
BACKEND="${BACKEND:-}"
if [[ -z "${BACKEND}" ]]; then
  if [[ "${MODEL_ID}" == *"-FP8"* ]]; then
    BACKEND="vllm"
  else
    BACKEND="transformers"
  fi
fi
META_CSV="${META_CSV:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/img/region_phone_table_grid.csv}"
SPEC_ROOT="${SPEC_ROOT:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/img/specs/grid/}"
MFA_JSON_ROOT="${MFA_JSON_ROOT:-/scratch3/che489/Ha/interspeech/datasets/vocv4_mfa_aligned/}"
SYSTEM_FILE="${SYSTEM_FILE:-${SCRIPT_DIR}/prompts/region_forensics_system.txt}"
USER_TEMPLATE_FILE="${USER_TEMPLATE_FILE:-${SCRIPT_DIR}/prompts/region_forensics_user.txt}"
OUTPUT_BASE_DIR="${OUTPUT_BASE:-${OUTPUT_BASE_DIR:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/En}}"
CACHE_BASE_DIR="${CACHE_BASE_DIR:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/En_high_performing/cache}"
FLASHINFER_WORKSPACE_DIR="${FLASHINFER_WORKSPACE_DIR:-${CACHE_BASE_DIR}/flashinfer}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_BASE_DIR}}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_BASE_DIR}/triton}"
HF_HOME="${HF_HOME:-${CACHE_BASE_DIR}/huggingface}"
TORCH_HOME="${TORCH_HOME:-${CACHE_BASE_DIR}/torch}"
TMPDIR="${TMPDIR:-${CACHE_BASE_DIR}/tmp}"
SHARD_COUNT="${SHARD_COUNT:-1}"
SHARD_ID="${SHARD_ID:-${SLURM_ARRAY_TASK_ID:-0}}"
if [[ "${SHARD_COUNT}" -gt 1 ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE_DIR}/shard_${SHARD_ID}_of_${SHARD_COUNT}}"
  OUTPUT_JSONL="${OUTPUT_JSONL:-${OUTPUT_BASE_DIR}/qwen_region_outputs_shard_${SHARD_ID}_of_${SHARD_COUNT}.jsonl}"
else
  OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE_DIR}}"
  OUTPUT_JSONL="${OUTPUT_JSONL:-${OUTPUT_DIR}/qwen_region_outputs.jsonl}"
fi
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-600}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
BATCH_SIZE="${BATCH_SIZE:-1}"
OVERWRITE="${OVERWRITE:-0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}"
VLLM_MAX_IMAGES_PER_PROMPT="${VLLM_MAX_IMAGES_PER_PROMPT:-1}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"

mkdir -p "${FLASHINFER_WORKSPACE_DIR}" "${TRITON_CACHE_DIR}" "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}"
export FLASHINFER_WORKSPACE_DIR XDG_CACHE_HOME TRITON_CACHE_DIR HF_HOME TORCH_HOME TMPDIR

echo "[run] SCRIPT_DIR=${SCRIPT_DIR}"
echo "[run] MODEL_ID=${MODEL_ID}"
echo "[run] BACKEND=${BACKEND}"
echo "[run] META_CSV=${META_CSV}"
echo "[run] SPEC_ROOT=${SPEC_ROOT}"
echo "[run] MFA_JSON_ROOT=${MFA_JSON_ROOT}"
echo "[run] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[run] SHARD_ID=${SHARD_ID} SHARD_COUNT=${SHARD_COUNT}"
echo "[run] BATCH_SIZE=${BATCH_SIZE}"
echo "[run] DTYPE=${DTYPE} ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION}"
echo "[run] OVERWRITE=${OVERWRITE}"
echo "[run] CACHE_BASE_DIR=${CACHE_BASE_DIR}"
echo "[run] FLASHINFER_WORKSPACE_DIR=${FLASHINFER_WORKSPACE_DIR}"

CMD=(python -u "${SCRIPT_DIR}/qwen_region_artifact_prompt.py" \
  --backend "${BACKEND}" \
  --model-id "${MODEL_ID}" \
  --meta-csv "${META_CSV}" \
  --spec-root "${SPEC_ROOT}" \
  --mfa-json-root "${MFA_JSON_ROOT}" \
  --system-file "${SYSTEM_FILE}" \
  --user-template-file "${USER_TEMPLATE_FILE}" \
  --dtype "${DTYPE}" \
  --attn-implementation "${ATTN_IMPLEMENTATION}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-shards "${SHARD_COUNT}" \
  --shard-id "${SHARD_ID}" \
  --output-dir "${OUTPUT_DIR}" \
  --output-jsonl "${OUTPUT_JSONL}")

if [[ -n "${TENSOR_PARALLEL_SIZE}" ]]; then
  CMD+=(--tensor-parallel-size "${TENSOR_PARALLEL_SIZE}")
fi

if [[ -n "${VLLM_MAX_MODEL_LEN}" ]]; then
  CMD+=(--vllm-max-model-len "${VLLM_MAX_MODEL_LEN}")
fi

CMD+=(--vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}")
CMD+=(--vllm-max-images-per-prompt "${VLLM_MAX_IMAGES_PER_PROMPT}")

if [[ "${VLLM_ENFORCE_EAGER}" == "1" ]]; then
  CMD+=(--vllm-enforce-eager)
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  CMD+=(--overwrite)
fi

CMD+=("$@")
echo "[run] ${CMD[*]}"
"${CMD[@]}"
