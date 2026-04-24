#!/usr/bin/env bash

###############################################################################
# Lightweight SLURM helper to start a local vLLM server and run `sparc` client.
# This variant uses the grpo2-style env activation (module load miniforge3 +
# configurable CONDA_ENV_PATH) so it works in the same cluster environment as
# run-sparc-grpo2.sbatch.
#
# Usage:
#   sbatch run-sparc2.sbatch [-- <extra sparc args>]
#
# Key env-var overrides at submission time:
#   MODEL             – path or HF model ID to evaluate (default below)
#   CONDA_ENV_PATH    – full path to conda env  (e.g. /path/to/envs/vllm)
#                       if not set, falls back to CONDA_ENV_NAME
#   CONDA_ENV_NAME    – env name used when CONDA_ENV_PATH is not set (default: vllm)
#   LOAD_MINIFORGE    – set to "0" to skip `module load miniforge3` (default: 1)
#   VISION_PLOT_TYPE  – override the inferred plot type (original | path_cell_annotated |
#                       path_cell | cell_annotated | cell)
#   LLM_PORT          – vLLM server port (default: 8000)
#   LLM_GPU_UTIL      – GPU memory utilisation fraction (default: 0.90)
###############################################################################
#SBATCH --job-name=sparc
#SBATCH --partition=scc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:4
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH -c 32
#SBATCH --output=logs/%x-%j.out
#SBATCH -C inet

set -euo pipefail
echo "[$(date +%F\ %T)] Job starting on $HOSTNAME"

# new tmp cache dirs
export TMPDIR=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}

export TRITON_CACHE_DIR=$TMPDIR/triton
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/torchinductor
export TORCH_EXTENSIONS_DIR=$TMPDIR/torch_extensions
export VLLM_CACHE_ROOT=$TMPDIR/vllm
export PIP_CACHE_DIR=$TMPDIR/pip
export XDG_CACHE_HOME=$TMPDIR/xdg

mkdir -p \
  $TRITON_CACHE_DIR \
  $TORCHINDUCTOR_CACHE_DIR \
  $TORCH_EXTENSIONS_DIR \
  $VLLM_CACHE_ROOT \
  $PIP_CACHE_DIR \
  $XDG_CACHE_HOME

###############################################################################
# 1 · User-configurable parameters (override with environment variables)
###############################################################################
MODEL="${MODEL:-lkaesberg/Qwen3-14B-SPaRC-GRPO-8E}"
PORT="${LLM_PORT:-8000}"
TP_SIZE="${SLURM_GPUS_ON_NODE:-4}"   # tensor-parallel size (GPUs on node)
GPU_UTIL="${LLM_GPU_UTIL:-0.90}"
HF_HOME="${HF_HOME:-$SCRATCH/huggingface}"

# Conda / environment activation
# Priority: explicit CONDA_ENV_PATH > build from CONDA_ENV_NAME (default: vllm)
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vllm}"
if [[ -z "${CONDA_ENV_PATH:-}" ]]; then
  CONDA_ENV_PATH="${CONDA_ENVS_PATH:+$CONDA_ENVS_PATH/}${CONDA_ENV_NAME}"
fi
export CONDA_ENV_PATH

# Whether to run `module load miniforge3` before activation (set to "0" to skip)
LOAD_MINIFORGE="${LOAD_MINIFORGE:-1}"

export HF_HOME
mkdir -p "$HF_HOME" logs

###############################################################################
# 2 · Infer visualization plot type from model name suffix
#     Suffixes recognised: _original, _path_cell_annotated, _text
#     VISION_PLOT_TYPE env var overrides inference.
#     This script always runs sparc in visual mode.
###############################################################################
MODEL_BASENAME="$(basename "$MODEL")"

# Infer plot type from suffix when VISION_PLOT_TYPE is not already set
if [[ -z "${VISION_PLOT_TYPE:-}" ]]; then
  case "$MODEL_BASENAME" in
    *_original)           VISION_PLOT_TYPE="original" ;;
    *_path_cell_annotated) VISION_PLOT_TYPE="path_cell_annotated" ;;
    *_text)               VISION_PLOT_TYPE="text" ;;
    *)                    VISION_PLOT_TYPE="" ;;   # unknown suffix fallback to non vision variante
  esac
  PLOT_TYPE_SOURCE="inferred from model name"
else
  PLOT_TYPE_SOURCE="overridden via VISION_PLOT_TYPE"
fi

if [ -n "$VISION_PLOT_TYPE" ]; then
  VISUAL_ARGS=(--visual --plot-type "$VISION_PLOT_TYPE")
else
  VISUAL_ARGS=()
  echo "No board puzzle visualization used (no image)."
fi
VISUAL_MODE="visual (plot-type=${VISION_PLOT_TYPE}, ${PLOT_TYPE_SOURCE})"

echo "Model:        $MODEL"
echo "Port:         $PORT | TP size: $TP_SIZE | HF_HOME: $HF_HOME"
echo "Conda env:    $CONDA_ENV_PATH"
echo "Visual mode:  $VISUAL_MODE"

###############################################################################
# 3 · Activate pre-provisioned environment
###############################################################################
module load gcc/13.2.0
module load cuda/12.6.2

if [[ "${LOAD_MINIFORGE}" != "0" ]]; then
  module load miniforge3
fi

echo "Activating conda env: $CONDA_ENV_PATH"
source activate "$CONDA_ENV_PATH"
echo "CONDA_PREFIX: ${CONDA_PREFIX:-<not set>}"

###############################################################################
# 4 · Launch vLLM (background) and capture logs
###############################################################################
LOG_DIR="$PWD/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/vllm_${SLURM_JOB_ID:-$$}.log"
touch "$LOG_FILE"

echo "Starting vLLM server (log -> $LOG_FILE)"
python -u -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --trust-remote-code \
  >"$LOG_FILE" 2>&1 &
SERVER_PID=$!

# Tail log while server boots so the SLURM job shows progress
tail -n +1 -f "$LOG_FILE" &
TAIL_PID=$!

cleanup() {
  echo "[$(date +%T)] Cleaning up…"
  kill -SIGINT "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  kill "$TAIL_PID"   2>/dev/null || true
}
trap cleanup EXIT INT TERM

###############################################################################
# 5 · Health probe (wait until /v1/models responds)
###############################################################################
echo -n "⌛ Waiting for vLLM to become ready"
for _ in {1..3600}; do
  if curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    echo " … ready."
    break
  fi
  printf "."; sleep 1
done

# Stop tailing logs now (server keeps running in background)
kill "$TAIL_PID" 2>/dev/null || true
echo "[$(date +%T)] Server started (log saved to $LOG_FILE)"

###############################################################################
# 6 · Run sparc client against local server
###############################################################################
export OPENAI_API_BASE="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-LOCALHOST_ONLY}"

echo "Running sparc client (visual mode: $VISUAL_MODE)..."
sparc --api-key "$OPENAI_API_KEY" \
      --base-url "$OPENAI_API_BASE" \
      --model "$MODEL" \
      --results-file ${MODEL##*/} \
      --batch-size 100 \
      "${VISUAL_ARGS[@]}" \
      "$@"

echo "[$(date +%F\ %T)] sparc finished — job complete."
