#!/bin/bash
# Submit a Factoriax throughput benchmark to Snellius.
#
# Runs ``scripts/throughput_bench.py`` (FactoriaX engine repo) on
# one allocated GPU. Sweeps map_size x batch_size x entity_multiplier
# with 50 trials per config by default, logs every individual trial
# to Weights & Biases plus a per-config summary event that carries
# startup_seconds, median, std, and (optionally) the baseline
# steps-per-sec ceiling. The JSON file is saved both to
# ``./throughput_<label>.json`` inside the job's working tree and to
# the wandb run as an artifact.
#
# Workflow once the run finishes:
#   1. ``wandb sync`` confirms the artifact uploaded.
#   2. Download ``throughput_<label>.json`` from the wandb run page.
#   3. Drop it into ``paper/data/`` in the repo on your laptop.
#   4. Run ``uv run python paper/scripts/experiments/throughput.py`` to
#      merge per-device files into the consolidated ``throughput.json``
#      the figure builder reads.
#
# Snellius partition reference (from SURF docs):
#   gpu_a100 — 18 cores + 1 A100 + 120 GiB host memory per 1/4 node,
#              128 SBU / GPU / hour, max 120h wall time.
#   gpu_h100 — 16 cores + 1 H100 + 180 GiB host memory per 1/4 node,
#              192 SBU / GPU / hour, max 120h wall time.
#   gpu_mig  — 9 cores + 1 A100 (MIG slice), useful for a smoke test.
#
# ------------------------------------------------------------------
# Override anything on the command line, e.g.:
#   DEVICE_LABEL=a100_h100 PARTITION=gpu_h100 ./scripts/snellius_submit_throughput.sh
#   NUM_ENVS_LIST="64 1024 16384" TRIALS=10 ./scripts/snellius_submit_throughput.sh
#   PROJECT_DIR=$HOME/code/factoriax ./scripts/snellius_submit_throughput.sh
# ------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROJECT_DIR="${PROJECT_DIR:-${DEFAULT_PROJECT_DIR}}"
PARTITION="${PARTITION:-gpu_a100}"
WALL_TIME="${WALL_TIME:-02:00:00}"
DEVICE_LABEL="${DEVICE_LABEL:-a100}"
OBS="${OBS:-x_ray_global}"
INNER_STEPS="${INNER_STEPS:-}"                       # empty -> autoscale
TRIALS="${TRIALS:-50}"
TARGET_TRIAL_SECONDS="${TARGET_TRIAL_SECONDS:-1.0}"
MAP_SIZES="${MAP_SIZES:-16 32 128}"
NUM_ENVS_LIST="${NUM_ENVS_LIST:-64 256 1024 4096 16384}"
ENTITY_MULTIPLIERS="${ENTITY_MULTIPLIERS:-0.5 1.0 2.0}"
BASELINE_FLAG="${BASELINE_FLAG:-}"                   # set to --baseline to enable
WANDB_PROJECT="${WANDB_PROJECT:-factoriax_throughput}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-throughput_${DEVICE_LABEL}}"

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
    echo "Set PROJECT_DIR to the factoriax checkout root." >&2
    exit 1
fi
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "ERROR: no Python at ${VENV_DIR}/bin/python." >&2
    echo "Run 'uv sync' on the login node first, or point VENV_DIR at an existing venv." >&2
    exit 1
fi

echo "Config:"
echo "  PROJECT_DIR          = ${PROJECT_DIR}"
echo "  VENV_DIR             = ${VENV_DIR}"
echo "  PARTITION            = ${PARTITION}"
echo "  WALL_TIME            = ${WALL_TIME}"
echo "  CPUS_PER_TASK        = ${CPUS_PER_TASK}"
echo "  DEVICE_LABEL         = ${DEVICE_LABEL}"
echo "  OBS                  = ${OBS}"
echo "  INNER_STEPS          = ${INNER_STEPS:-<autoscale>}"
echo "  TRIALS               = ${TRIALS}"
echo "  TARGET_TRIAL_SECONDS = ${TARGET_TRIAL_SECONDS}"
echo "  MAP_SIZES            = ${MAP_SIZES}"
echo "  NUM_ENVS_LIST        = ${NUM_ENVS_LIST}"
echo "  ENTITY_MULTIPLIERS   = ${ENTITY_MULTIPLIERS}"
echo "  BASELINE_FLAG        = ${BASELINE_FLAG:-<off>}"
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
#SBATCH --job-name=factoriax_throughput
#SBATCH --output=${SLURM_OUT_DIR}/throughput_%j.out

set -euo pipefail
export PATH="${VENV_DIR}/bin:\${PATH}"
export VIRTUAL_ENV="${VENV_DIR}"
PY="${VENV_DIR}/bin/python"

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

cd "${PROJECT_DIR}"

echo "=== \$(date) — host \$(hostname) ==="
nvidia-smi --query-gpu=name,memory.free,memory.total,driver_version --format=csv
"\${PY}" --version

INNER_STEPS_FLAG=""
if [[ -n "${INNER_STEPS}" ]]; then
    INNER_STEPS_FLAG="--inner-steps ${INNER_STEPS}"
fi

"\${PY}" scripts/throughput_bench.py \\
    --device-label ${DEVICE_LABEL} \\
    --obs ${OBS} \\
    --map-sizes ${MAP_SIZES} \\
    --num-envs ${NUM_ENVS_LIST} \\
    --entity-multipliers ${ENTITY_MULTIPLIERS} \\
    --trials ${TRIALS} \\
    --target-trial-seconds ${TARGET_TRIAL_SECONDS} \\
    \${INNER_STEPS_FLAG} \\
    ${BASELINE_FLAG} \\
    --wandb-project ${WANDB_PROJECT} \\
    --wandb-run-name ${WANDB_RUN_NAME}
EOF

echo "Submitted."
