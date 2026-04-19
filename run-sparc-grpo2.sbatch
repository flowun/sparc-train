#!/usr/bin/env bash
###############################################################################
# SBATCH wrapper for distributed GRPO training.
# Assumes a pre-created "vllm" environment is available on compute nodes
# (activate with: `source activate vllm`).
###############################################################################
#SBATCH --job-name=sparc-grpo
#SBATCH --partition=scc-gpu
#SBATCH --time=48:00:00         # 48:00:00
#SBATCH --mem=256G
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:4       # 4 GPUs per node
#SBATCH -c 32
#SBATCH -C inet&80gb
#SBATCH --output=logs_grpo/%x_%j.log
#SBATCH --error=logs_grpo/%x_%j.log

module load gcc/13.2.0
module load cuda/12.6.2


# Cache problem fix: new cache dir in every job
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


module load miniforge3
CONDA_ENV_NAME=${CONDA_ENV_NAME:-vllm}
CONDA_ENV_PATH="${CONDA_ENVS_PATH:+$CONDA_ENVS_PATH/}${CONDA_ENV_NAME}"
# Export conda path
export CONDA_ENV_PATH

# Activate the vllm environment (cluster-provided)
source activate "$CONDA_ENV_PATH"
echo "Conda env:"
echo "$CONDA_PREFIX"

#
# Dynamic layout from a single TRAIN_NODE_COUNT (or derive from allocation)
# - By default we use 1 node for vLLM and the rest for training
# - You can override with env vars when submitting, e.g.:
#     TRAIN_NODE_COUNT=2 GPUS_PER_NODE=4 sbatch -N 3 run-sparc-grpo-vllm-deep.sbatch
#   Ensure the SBATCH -N equals TRAIN_NODE_COUNT + VLLM_NODE_COUNT

# Central model selection (override by exporting MODEL_NAME before sbatch)
MODEL_NAME=${MODEL_NAME:-lkaesberg/Qwen3-32B-SPaRC-GRPO}
echo "Model: ${MODEL_NAME}"
RUN_NAME_ADDITION=${RUN_NAME_ADDITION:-"8epoch"}
USE_VISION=${USE_VISION:-False}
VISION_PLOT_TYPE=${VISION_PLOT_TYPE:-"original"}  # options: original, path_cell_annotated, text, etc.
CHECKPOINT_STEPS=${CHECKPOINT_STEPS:-}

USE_VISION_NORM=$(echo "${USE_VISION}" | tr '[:upper:]' '[:lower:]')
USE_VISION_ARG=""
if [[ "${USE_VISION_NORM}" == "1" || "${USE_VISION_NORM}" == "true" || "${USE_VISION_NORM}" == "yes" ]]; then
  USE_VISION_ARG="--use_vision_variant"
  RUN_NAME_ADDITION="${RUN_NAME_ADDITION}_${VISION_PLOT_TYPE}"
fi
echo "Use vision variant: ${USE_VISION} (${USE_VISION_ARG:-disabled})"
echo "Vision plot type: ${VISION_PLOT_TYPE}"
CHECKPOINT_ARGS=""
CHECKPOINT_STATUS="disabled"
if [[ -n "${CHECKPOINT_STEPS}" ]]; then
  CHECKPOINT_ARGS="--checkpoint_steps ${CHECKPOINT_STEPS}"
  CHECKPOINT_STATUS="every ${CHECKPOINT_STEPS} steps"
fi
echo "Checkpointing: ${CHECKPOINT_STATUS}"

# Discover allocated hosts
NODELIST=( $(scontrol show hostnames ${SLURM_JOB_NODELIST}) )
TOTAL_NODES=${#NODELIST[@]}

# User-tunable knobs (with safe defaults)
VLLM_NODE_COUNT=${VLLM_NODE_COUNT:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
VLLM_TP_SIZE=${VLLM_TP_SIZE:-${GPUS_PER_NODE}}

# Determine number of training nodes
if [[ -n "${TRAIN_NODE_COUNT}" ]]; then
  TRAIN_NODES_COUNT=${TRAIN_NODE_COUNT}
else
  TRAIN_NODES_COUNT=$(( TOTAL_NODES - VLLM_NODE_COUNT ))
fi

if (( TRAIN_NODES_COUNT < 1 )); then
  echo "Error: TRAIN_NODE_COUNT must be >= 1 (total=${TOTAL_NODES}, vllm_nodes=${VLLM_NODE_COUNT})." >&2
  exit 1
fi

# Split nodes: first TRAIN_NODES_COUNT for training, next 1 for vLLM
TRAIN_NODES=("${NODELIST[@]:0:${TRAIN_NODES_COUNT}}")
VLLM_NODE="${NODELIST[$((TRAIN_NODES_COUNT))]}"

# Derived distributed settings
NUM_MACHINES=${TRAIN_NODES_COUNT}
NUM_PROCESSES=$(( TRAIN_NODES_COUNT * GPUS_PER_NODE ))
MAIN_ADDR="${TRAIN_NODES[0]}"
MAIN_PORT=${MAIN_PORT:-29500}

echo "Train nodes (${NUM_MACHINES}): ${TRAIN_NODES[*]}"
echo "vLLM node:  ${VLLM_NODE}"
echo "GPUs per node: ${GPUS_PER_NODE} | Processes: ${NUM_PROCESSES} | TP size: ${VLLM_TP_SIZE}"

# Launch vLLM server on the dedicated node as a background step
srun --export=ALL --nodes=1 --ntasks=1 --nodelist=${VLLM_NODE} \
  bash -lc "\
    module load gcc/13.2.0; \
    module load cuda/12.6.2; \
    nvidia-smi; \
    module load miniforge3; \
    source activate $CONDA_ENV_PATH; \
    trl vllm-serve \
      --model "${MODEL_NAME}" \
      --host 0.0.0.0 --port 8000 \
      --tensor_parallel_size ${VLLM_TP_SIZE} \
  " &

# Give the server time to start
sleep 30

# Start distributed training across the training nodes
srun --export=ALL --nodes=${NUM_MACHINES} --ntasks=${NUM_MACHINES} --nodelist="${TRAIN_NODES[*]}" \
  bash -lc "\
    module load gcc/13.2.0; \
    module load cuda/12.6.2; \
    nvidia-smi; \
    module load miniforge3; \
    source activate $CONDA_ENV_PATH; \
    accelerate launch \
      --config_file deepspeed_zero3.yaml \
      --num_processes ${NUM_PROCESSES} \
      --num_machines ${NUM_MACHINES} \
      --main_process_ip ${MAIN_ADDR} \
      --main_process_port ${MAIN_PORT} \
      --machine_rank $SLURM_PROCID \
      --rdzv_backend c10d \
      train_grpo.py \
        --model \"${MODEL_NAME}\" \
        --vllm_server_host ${VLLM_NODE} \
        --wandb_project sparc-grpo \
	--wandb_entity "${WANDB_ENTITY}" \
        --run_name_addition ${RUN_NAME_ADDITION} \
        --vision_plot_type ${VISION_PLOT_TYPE} \
        ${USE_VISION_ARG} \
        ${CHECKPOINT_ARGS} \
  "
