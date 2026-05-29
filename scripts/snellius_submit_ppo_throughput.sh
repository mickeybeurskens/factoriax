#!/bin/bash
# Submit a PPO training-throughput benchmark to Snellius.
#
# Runs ``scripts/ppo_throughput_bench.py`` on one allocated GPU.
# Times one warmup PPO iteration (JIT compile + first launch) and a
# configurable number of steady-state iterations on the easy-rocket
# scenario, then extrapolates to a 100M-step reference budget so the
# paper-side figure can show wall-clock time. Logs every timed
# iteration to Weights & Biases and saves the JSON locally and as a
# wandb artifact.
#
# Workflow once the run finishes:
#   1. ``wandb sync`` confirms the artifact uploaded.
#   2. Download ``ppo_throughput_<label>.json`` from the wandb run page.
#   3. Drop it into ``paper/data/`` as ``ppo_throughput.json`` (or
#      concatenate multiple device files into one ``runs`` array).
#   4. ``uv run python paper/scripts/build_figures.py --only throughput``
#      regenerates the §3.5 figure with the new left-panel data.
#
# Snellius partition reference (from SURF docs):
#   gpu_a100 — 18 cores + 1 A100 + 120 GiB host memory per 1/4 node.
#   gpu_h100 — 16 cores + 1 H100 + 180 GiB host memory per 1/4 node.
#   gpu_mig  — 9 cores + 1 A100 (MIG slice), useful for smoke tests.
#
# ------------------------------------------------------------------
# Override anything on the command line, e.g.:
#   DEVICE_LABEL=h100 PARTITION=gpu_h100 ./scripts/snellius_submit_ppo_throughput.sh
#   NUM_ENVS=1024 MEASURE_ITERS=20 ./scripts/snellius_submit_ppo_throughput.sh
#   REFERENCE_TOTAL_STEPS=1000000000 ./scripts/snellius_submit_ppo_throughput.sh
# ------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROJECT_DIR="${PROJECT_DIR:-${DEFAULT_PROJECT_DIR}}"
PARTITION="${PARTITION:-gpu_a100}"
WALL_TIME="${WALL_TIME:-00:30:00}"
DEVICE_LABEL="${DEVICE_LABEL:-a100}"
NUM_ENVS="${NUM_ENVS:-64}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-128}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-4}"
HIDDEN_DIMS="${HIDDEN_DIMS:-64 64}"
WARMUP_ITERS="${WARMUP_ITERS:-1}"
MEASURE_ITERS="${MEASURE_ITERS:-}"                      # empty -> autoscale from budget
MEASURE_BUDGET_STEPS="${MEASURE_BUDGET_STEPS:-500000000}"
LOG_EVERY_ITER="${LOG_EVERY_ITER:-0}"                   # 0 -> auto
WANDB_PROJECT="${WANDB_PROJECT:-factoriax_ppo_throughput}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-ppo_throughput_${DEVICE_LABEL}}"

if [[ -z "${CPUS_PER_TASK:-}" ]]; then
    case "${PARTITION}" in
        gpu_h100) CPUS_PER_TASK=16 ;;
        gpu_mig)  CPUS_PER_TASK=9 ;;
        *)        CPUS_PER_TASK=18 ;;
    esac
fi

SLURM_OUT_DIR="${SLURM_OUT_DIR:-${HOME}/slurm}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"

mkdir -p "${SLURM_OUT_DIR}"

if [[ ! -f "${PROJECT_DIR}/pyproject.toml" ]]; then
    echo "ERROR: PROJECT_DIR=${PROJECT_DIR} does not contain pyproject.toml." >&2
    exit 1
fi
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "ERROR: no Python at ${VENV_DIR}/bin/python." >&2
    exit 1
fi

echo "Config:"
echo "  PROJECT_DIR          = ${PROJECT_DIR}"
echo "  VENV_DIR             = ${VENV_DIR}"
echo "  PARTITION            = ${PARTITION}"
echo "  WALL_TIME            = ${WALL_TIME}"
echo "  CPUS_PER_TASK        = ${CPUS_PER_TASK}"
echo "  DEVICE_LABEL         = ${DEVICE_LABEL}"
echo "  NUM_ENVS             = ${NUM_ENVS}"
echo "  ROLLOUT_STEPS        = ${ROLLOUT_STEPS}"
echo "  UPDATE_EPOCHS        = ${UPDATE_EPOCHS}"
echo "  NUM_MINIBATCHES      = ${NUM_MINIBATCHES}"
echo "  HIDDEN_DIMS          = ${HIDDEN_DIMS}"
echo "  WARMUP_ITERS         = ${WARMUP_ITERS}"
echo "  MEASURE_ITERS        = ${MEASURE_ITERS:-<autoscale>}"
echo "  MEASURE_BUDGET_STEPS = ${MEASURE_BUDGET_STEPS}"
echo "  LOG_EVERY_ITER       = ${LOG_EVERY_ITER}"
echo "  WANDB_PROJECT        = ${WANDB_PROJECT}"
echo "  WANDB_RUN_NAME       = ${WANDB_RUN_NAME}"

cat <<EOF | sbatch
#!/bin/bash
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --gpus=1
#SBATCH --time=${WALL_TIME}
#SBATCH --job-name=factoriax_ppo_throughput
#SBATCH --output=${SLURM_OUT_DIR}/ppo_throughput_%j.out

set -euo pipefail
export PATH="${VENV_DIR}/bin:\${PATH}"
export VIRTUAL_ENV="${VENV_DIR}"
PY="${VENV_DIR}/bin/python"

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

cd "${PROJECT_DIR}"

echo "=== \$(date) — host \$(hostname) ==="
nvidia-smi --query-gpu=name,memory.free,memory.total,driver_version --format=csv
"\${PY}" --version

MEASURE_ITERS_FLAG=""
if [[ -n "${MEASURE_ITERS}" ]]; then
    MEASURE_ITERS_FLAG="--measure-iters ${MEASURE_ITERS}"
fi

"\${PY}" scripts/ppo_throughput_bench.py \\
    --device-label ${DEVICE_LABEL} \\
    --num-envs ${NUM_ENVS} \\
    --rollout-steps ${ROLLOUT_STEPS} \\
    --update-epochs ${UPDATE_EPOCHS} \\
    --num-minibatches ${NUM_MINIBATCHES} \\
    --hidden-dims ${HIDDEN_DIMS} \\
    --warmup-iters ${WARMUP_ITERS} \\
    \${MEASURE_ITERS_FLAG} \\
    --measure-budget-steps ${MEASURE_BUDGET_STEPS} \\
    --log-every-iter ${LOG_EVERY_ITER} \\
    --wandb-project ${WANDB_PROJECT} \\
    --wandb-run-name ${WANDB_RUN_NAME}
EOF

echo "Submitted."
