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
#   cd "<your factoriax checkout>"
#   uv sync                 # creates .venv with pinned deps
#   .venv/bin/wandb login   # authenticate wandb
#
# After that, from inside the factoriax repo on Snellius just run:
#   ./scripts/snellius_submit_rocket.sh
#
# Don't have uv? Any of these work on Snellius:
#   pip install --user uv                # simplest
#   curl -LsSf https://astral.sh/uv/install.sh | sh
#   module load 2025 && module load uv   # if available
#
# ------------------------------------------------------------------
# Everything user-specific is derived or overridable. Override any
# value on the command line, e.g.:
#   PARTITION=gpu_h100 WALL_TIME=08:00:00 ./scripts/snellius_submit_rocket.sh
#   PROJECT_DIR=$HOME/code/factoriax ./scripts/snellius_submit_rocket.sh
#   VENV_DIR=$HOME/other_venv ./scripts/snellius_submit_rocket.sh
# ------------------------------------------------------------------

set -euo pipefail

# ---- Sweep parameters (edit to add more jobs) --------------------------------

seeds=(0)
total_steps_list=(5_000_000_000)         # 100M steps, cheap first test run.
num_envs_list=(2048)
rollout_steps_list=(128)
# Tag run names with the benchmark config that defines the task: 8000-tick
# episodes + masked CRAFT_* actions (matches the scripted-agent benchmark).
run_names=("ppo_rocket_craftmask")

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
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"

mkdir -p "${SLURM_OUT_DIR}"

# Fail fast if the project checkout looks wrong.
if [[ ! -f "${PROJECT_DIR}/pyproject.toml" ]]; then
    echo "ERROR: PROJECT_DIR=${PROJECT_DIR} does not contain pyproject.toml." >&2
    echo "Set PROJECT_DIR to the factoriax checkout root." >&2
    exit 1
fi
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "ERROR: no Python at ${VENV_DIR}/bin/python." >&2
    echo "Create the venv first (on the login node, from the repo):" >&2
    echo "    uv sync" >&2
    echo "Or point VENV_DIR at an existing venv:" >&2
    echo "    VENV_DIR=/path/to/.venv ./scripts/snellius_submit_rocket.sh" >&2
    exit 1
fi

echo "Config:"
echo "  PROJECT_DIR   = ${PROJECT_DIR}"
echo "  VENV_DIR      = ${VENV_DIR}"
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
# Call the venv's python directly. No activation, no uv dependency on
# the compute node — whatever created the venv on the login node (uv,
# pip, etc.) is fine as long as .venv/bin/python exists.
export PATH="${VENV_DIR}/bin:\${PATH}"
export VIRTUAL_ENV="${VENV_DIR}"
PY="${VENV_DIR}/bin/python"

# JAX tends to grab all VRAM by default; leave some headroom for the
# CUDA context. On A100 80GB this is ~72GB usable.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

cd "${PROJECT_DIR}"

echo "=== \$(date) — host \$(hostname) ==="
nvidia-smi --query-gpu=name,memory.free,memory.total,driver_version --format=csv
"\${PY}" --version

"\${PY}" -m baselines.rocket.train_ppo \\
    --num-envs ${num_envs} \\
    --rollout-steps ${rollout_steps} \\
    --total-steps ${total_steps} \\
    --max-timesteps 8000 \\
    --obs-radius 7 \\
    --seed ${seed} \\
    --log-interval 32 \\
    --use-wandb \\
    --wandb-project ${WANDB_PROJECT} \\
    --wandb-run-name ${run_name}
EOF

    sleep 0.1
done
