#!/bin/bash
# Submit rocket-benchmark PPO training jobs to Snellius.
#
# Edit the arrays below to run a sweep; each combination submits one
# sbatch job. The default single-element arrays submit one job.
#
# Snellius partition reference (from SURF docs):
#   gpu_a100 — 18 cores + 1 A100 + 120 GiB host memory per 1/4 node,
#              128 SBU / GPU / hour, max 120h wall time.
#   gpu_h100 — 16 cores + 1 H100 + 180 GiB host memory per 1/4 node,
#              192 SBU / GPU / hour, max 120h wall time.
#   gpu_mig  — 9 cores + 1 A100 (MIG slice) + 60 GiB, 64 SBU / hour.
#              Cheap, but SM count is capped — fine for compile/smoke
#              tests, avoid for real training rollouts.
#
# ------------------------------------------------------------------
# One-time setup on the *login node* (only the first time):
#   curl -LsSf https://astral.sh/uv/install.sh | sh
#   source "$HOME/.local/bin/env"
#   cd "<your factoriax checkout>"
#   uv sync
#   uv run wandb login
#
# After that, from inside the factoriax repo on Snellius just run:
#   ./scripts/snellius_submit_rocket.sh
#
# ------------------------------------------------------------------
# Everything user-specific is derived or overridable. Override any
# value on the command line, e.g.:
#   PARTITION=gpu_h100 WALL_TIME=08:00:00 ./scripts/snellius_submit_rocket.sh
#   PROJECT_DIR=$HOME/code/factoriax ./scripts/snellius_submit_rocket.sh
#   UV_CACHE_DIR=/scratch-shared/$USER/uv-cache ./scripts/snellius_submit_rocket.sh
# ------------------------------------------------------------------

set -euo pipefail

# ---- Sweep parameters (edit to add more jobs) --------------------------------

seeds=(0)
total_steps_list=(100000000)         # 100M steps, cheap first test run.
num_envs_list=(2048)
rollout_steps_list=(128)
run_names=("ppo_rocket_100M_a100_test")

# ---- SLURM / environment (override via env vars, defaults below) -------------

# Resolve the repo root from this script's own location so PROJECT_DIR
# works no matter whose home directory the checkout lives in.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROJECT_DIR="${PROJECT_DIR:-${DEFAULT_PROJECT_DIR}}"
PARTITION="${PARTITION:-gpu_a100}"
WALL_TIME="${WALL_TIME:-02:00:00}"
# Snellius allocates 18 cores with 1 A100 and 16 cores with 1 H100; pick
# the safe default per partition unless the caller overrides.
if [[ -z "${CPUS_PER_TASK:-}" ]]; then
    case "${PARTITION}" in
        gpu_h100) CPUS_PER_TASK=16 ;;
        gpu_mig)  CPUS_PER_TASK=9 ;;
        *)        CPUS_PER_TASK=18 ;;
    esac
fi
SLURM_OUT_DIR="${SLURM_OUT_DIR:-${HOME}/slurm}"
WANDB_PROJECT="${WANDB_PROJECT:-factoriax_rocket}"
UV_ENV_FILE="${UV_ENV_FILE:-${HOME}/.local/bin/env}"

mkdir -p "${SLURM_OUT_DIR}"

# Fail fast if the project checkout looks wrong.
if [[ ! -f "${PROJECT_DIR}/pyproject.toml" ]]; then
    echo "ERROR: PROJECT_DIR=${PROJECT_DIR} does not contain pyproject.toml." >&2
    echo "Set PROJECT_DIR to the factoriax checkout root." >&2
    exit 1
fi
if [[ ! -f "${UV_ENV_FILE}" ]]; then
    echo "ERROR: uv env file not found at ${UV_ENV_FILE}." >&2
    echo "Install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

echo "Config:"
echo "  PROJECT_DIR   = ${PROJECT_DIR}"
echo "  PARTITION     = ${PARTITION}"
echo "  WALL_TIME     = ${WALL_TIME}"
echo "  CPUS_PER_TASK = ${CPUS_PER_TASK}"
echo "  SLURM_OUT_DIR = ${SLURM_OUT_DIR}"
echo "  WANDB_PROJECT = ${WANDB_PROJECT}"

# ---- Submit loop -------------------------------------------------------------

for i in "${!seeds[@]}"; do
    seed="${seeds[$i]}"
    total_steps="${total_steps_list[$i]}"
    num_envs="${num_envs_list[$i]}"
    rollout_steps="${rollout_steps_list[$i]}"
    run_name="${run_names[$i]}"

    echo "Submitting: seed=${seed} steps=${total_steps} envs=${num_envs} rollout=${rollout_steps} name=${run_name}"

    cat <<EOF | sbatch
#!/bin/bash
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --gpus=1
#SBATCH --time=${WALL_TIME}
#SBATCH --job-name=rocket_ppo
#SBATCH --output=${SLURM_OUT_DIR}/rocket_%j.out

set -euo pipefail

# --- environment ---
source "${UV_ENV_FILE}"

# JAX tends to grab all VRAM by default; leave some headroom for the
# CUDA context. On A100 80GB this is ~72GB usable.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

cd "${PROJECT_DIR}"

echo "=== \$(date) — host \$(hostname) ==="
nvidia-smi --query-gpu=name,memory.free,memory.total,driver_version --format=csv
uv --version

# uv run uses the already-synced .venv (created via \`uv sync\` on the
# login node). No network or install happens here on the compute node.
uv run python -m baselines.rocket.train_ppo \\
    --num-envs ${num_envs} \\
    --rollout-steps ${rollout_steps} \\
    --total-steps ${total_steps} \\
    --max-timesteps 2000 \\
    --seed ${seed} \\
    --log-interval 32 \\
    --use-wandb \\
    --wandb-project ${WANDB_PROJECT} \\
    --wandb-run-name ${run_name}
EOF

    sleep 0.1
done
