#!/bin/bash -l
#SBATCH --job-name=qwen_region_full
#SBATCH --partition=h24gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/qwen_region_full_%j.out
#SBATCH --account=OD-237777
#SBATCH --gres=gpu:4

set -euo pipefail

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

conda activate deepfake

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
MODEL_ID="${MODEL_ID:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/VLM/Qwen3-VL-30B-A3B-Thinking}"
P1_ROOT="${P1_ROOT:-/scratch3/che489/Ha/interspeech/localization/Ms_region_outputs}"
P2_ROOT="${P2_ROOT:-/scratch3/che489/Ha/interspeech/localization/region_crops_top3}"
P3_ROOT="${P3_ROOT:-/scratch3/che489/Ha/interspeech/localization/region_crops_real}"
META_CSV="${META_CSV:-/scratch3/che489/Ha/interspeech/datasets/region_phone_table_top3_all_with_ptype.csv}"
SYSTEM_FILE="${SYSTEM_FILE:-${SCRIPT_DIR}/prompts/region_forensics_system.txt}"
USER_TEMPLATE_FILE="${USER_TEMPLATE_FILE:-${SCRIPT_DIR}/prompts/region_forensics_user.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch3/che489/Ha/interspeech/localization/qwen3_vlm}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${OUTPUT_DIR}/qwen_region_outputs.jsonl}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-600}"
DTYPE="${DTYPE:-auto}"

echo "[run] SCRIPT_DIR=${SCRIPT_DIR}"
echo "[run] MODEL_ID=${MODEL_ID}"
echo "[run] P2_ROOT=${P2_ROOT}"
echo "[run] OUTPUT_DIR=${OUTPUT_DIR}"

python "${SCRIPT_DIR}/qwen_region_artifact_prompt.py" \
  --model-id "${MODEL_ID}" \
  --p1-root "${P1_ROOT}" \
  --p2-root "${P2_ROOT}" \
  --p3-root "${P3_ROOT}" \
  --meta-csv "${META_CSV}" \
  --system-file "${SYSTEM_FILE}" \
  --user-template-file "${USER_TEMPLATE_FILE}" \
  --dtype "${DTYPE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --output-dir "${OUTPUT_DIR}" \
  --output-jsonl "${OUTPUT_JSONL}" \
  "$@"
