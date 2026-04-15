#!/usr/bin/env bash
###############################################################################
# SBATCH wrapper for distributed GRPO training.
# Assumes a pre-created "vllm" environment is available on compute nodes
# (activate with: `source activate vllm`).
###############################################################################
#SBATCH --job-name=sparc-grpo3
#SBATCH --partition=scc-gpu
#SBATCH --time=48:00:00         # 48:00:00
#SBATCH --mem=256G
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:2       # at least 2 GPUs per node (request floor)
#SBATCH -c 32
#SBATCH -C inet&80gb
#SBATCH --output=logs_grpo/%x_%j.log
#SBATCH --error=logs_grpo/%x_%j.log

module load gcc/13.2.0
module load cuda/12.6.2

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

USE_VISION_NORM=$(echo "${USE_VISION}" | tr '[:upper:]' '[:lower:]')
USE_VISION_ARG=""
if [[ "${USE_VISION_NORM}" == "1" || "${USE_VISION_NORM}" == "true" || "${USE_VISION_NORM}" == "yes" ]]; then
  USE_VISION_ARG="--use_vision_variant"
  RUN_NAME_ADDITION="${RUN_NAME_ADDITION}_${VISION_PLOT_TYPE}"
fi
echo "Use vision variant: ${USE_VISION} (${USE_VISION_ARG:-disabled})"
echo "Vision plot type: ${VISION_PLOT_TYPE}"

# Discover allocated hosts
NODELIST=( $(scontrol show hostnames ${SLURM_JOB_NODELIST}) )
TOTAL_NODES=${#NODELIST[@]}

# User-tunable knobs (with safe defaults)
VLLM_NODE_COUNT=${VLLM_NODE_COUNT:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
VLLM_TP_SIZE=${VLLM_TP_SIZE:-}

# Optional GPU type parser hint (defaults to A100 from SBATCH --gres line)
GPU_TYPE=${GPU_TYPE:-A100}

parse_node_state() {
  local node="$1"
  scontrol show node "${node}" | awk '{
    for (i = 1; i <= NF; i++) {
      if ($i ~ /^State=/) {
        split($i, a, "=");
        split(a[2], b, "+");
        print b[1];
        exit;
      }
    }
  }'
}

parse_gpu_count_from_field() {
  local field="$1"
  local gpu_type="$2"
  local count

  count=$(echo "${field}" | sed -nE "s/.*gpu:${gpu_type}:([0-9]+).*/\\1/p")
  if [[ -z "${count}" ]]; then
    count=$(echo "${field}" | sed -nE 's/.*gpu:[^:,]+:([0-9]+).*/\1/p')
  fi
  if [[ -z "${count}" ]]; then
    count=$(echo "${field}" | sed -nE 's/.*gpu:([0-9]+).*/\1/p')
  fi
  if [[ -z "${count}" ]]; then
    count=0
  fi
  echo "${count}"
}

parse_node_gpu_total() {
  local node="$1"
  local field
  field=$(scontrol show node "${node}" | awk '{
    for (i = 1; i <= NF; i++) {
      if ($i ~ /^Gres=/) {
        split($i, a, "=");
        print a[2];
        exit;
      }
    }
  }')
  parse_gpu_count_from_field "${field}" "${GPU_TYPE}"
}

parse_node_gpu_used() {
  local node="$1"
  local field
  field=$(scontrol show node "${node}" | awk '{
    for (i = 1; i <= NF; i++) {
      if ($i ~ /^GresUsed=/) {
        split($i, a, "=");
        print a[2];
        exit;
      }
    }
  }')
  parse_gpu_count_from_field "${field}" "${GPU_TYPE}"
}

rank_state_priority() {
  case "$1" in
    IDLE) echo 0 ;;
    MIXED) echo 1 ;;
    *) echo 2 ;;
  esac
}

declare -A NODE_FREE_GPU
ranked_lines=()
for node in "${NODELIST[@]}"; do
  state=$(parse_node_state "${node}")
  total_gpu=$(parse_node_gpu_total "${node}")
  used_gpu=$(parse_node_gpu_used "${node}")
  free_gpu=$(( total_gpu - used_gpu ))
  if (( free_gpu < 0 )); then
    free_gpu=0
  fi
  priority=$(rank_state_priority "${state}")
  NODE_FREE_GPU["${node}"]=${free_gpu}
  ranked_lines+=( "${priority} ${free_gpu} ${node}" )
done

readarray -t SORTED_NODELIST < <(
  printf '%s\n' "${ranked_lines[@]}" | sort -k1,1n -k2,2nr -k3,3 | awk '{print $3}'
)

if (( ${#SORTED_NODELIST[@]} == TOTAL_NODES )); then
  NODELIST=( "${SORTED_NODELIST[@]}" )
fi

echo "Ranked allocated nodes (idle -> mixed -> others, then free GPUs desc): ${NODELIST[*]}"

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

if (( VLLM_NODE_COUNT != 1 )); then
  echo "Error: run-sparc-grpo3.sbatch currently supports exactly one vLLM node (VLLM_NODE_COUNT=1)." >&2
  exit 1
fi

# Split nodes: first 1 for vLLM, then TRAIN_NODES_COUNT for training
VLLM_NODE="${NODELIST[0]}"
TRAIN_NODES=("${NODELIST[@]:${VLLM_NODE_COUNT}:${TRAIN_NODES_COUNT}}")

if (( ${#TRAIN_NODES[@]} != TRAIN_NODES_COUNT )); then
  echo "Error: expected ${TRAIN_NODES_COUNT} training nodes, got ${#TRAIN_NODES[@]}." >&2
  exit 1
fi

if (( GPUS_PER_NODE < 2 )); then
  GPUS_PER_NODE=2
fi

MIN_ALLOCATED_TRAIN_GPU=999
readarray -t TRAIN_GPU_INFO < <(
  srun --export=ALL --nodes=${TRAIN_NODES_COUNT} --ntasks=${TRAIN_NODES_COUNT} --nodelist="${TRAIN_NODES[*]}" \
    bash -lc 'echo "$(hostname) $(nvidia-smi -L | wc -l)"'
)

if (( ${#TRAIN_GPU_INFO[@]} == 0 )); then
  echo "Error: could not determine allocated GPU counts on training nodes." >&2
  exit 1
fi

for line in "${TRAIN_GPU_INFO[@]}"; do
  node_gpu_count=$(echo "${line}" | awk '{print $2}')
  if [[ -z "${node_gpu_count}" ]]; then
    node_gpu_count=0
  fi
  if (( node_gpu_count < MIN_ALLOCATED_TRAIN_GPU )); then
    MIN_ALLOCATED_TRAIN_GPU=${node_gpu_count}
  fi
done

if (( MIN_ALLOCATED_TRAIN_GPU < 2 )); then
  echo "Error: at least 2 GPUs are required on each training node, found minimum allocated ${MIN_ALLOCATED_TRAIN_GPU}." >&2
  exit 1
fi

GPUS_PER_NODE=${MIN_ALLOCATED_TRAIN_GPU}

VLLM_VISIBLE_GPU=$(srun --export=ALL --nodes=1 --ntasks=1 --nodelist=${VLLM_NODE} bash -lc 'nvidia-smi -L | wc -l')
if [[ -z "${VLLM_VISIBLE_GPU}" ]]; then
  VLLM_VISIBLE_GPU=0
fi

if (( VLLM_VISIBLE_GPU < 1 )); then
  echo "Error: selected vLLM node ${VLLM_NODE} has no allocated/visible GPUs." >&2
  exit 1
fi

if [[ -z "${VLLM_TP_SIZE}" ]]; then
  if (( VLLM_VISIBLE_GPU < GPUS_PER_NODE )); then
    VLLM_TP_SIZE=${VLLM_VISIBLE_GPU}
  else
    VLLM_TP_SIZE=${GPUS_PER_NODE}
  fi
fi

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
  "
