#!/bin/bash
# Submit easy-rocket PPO training jobs to Snellius.
#
# Loads the scenario via ``factoriax.make("EasyRocket-v1")``: a 16x16 procgen
# map with a 2000-step budget. Each parallel env draws its OWN keyed terrain
# from the scenario's reset_fn, so ``num_envs`` also controls how many distinct
# layouts the policy trains across (the whole point of easy_rocket).
#
# Edit the arrays below to run a sweep; each combination submits one sbatch job.
# The default single-element arrays submit one job.
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
#   ./scripts/snellius_submit_easy_rocket.sh
#
# Don't have uv? Any of these work on Snellius:
#   pip install --user uv                # simplest
#   curl -LsSf https://astral.sh/uv/install.sh | sh
#   module load 2025 && module load uv   # if available
#
# ------------------------------------------------------------------
# Everything user-specific is derived or overridable. Override any
# value on the command line, e.g.:
#   PARTITION=gpu_h100 WALL_TIME=08:00:00 ./scripts/snellius_submit_easy_rocket.sh
#   NUM_ENVS=4096 TOTAL_STEPS=2_000_000_000 ./scripts/snellius_submit_easy_rocket.sh
#   PROJECT_DIR=$HOME/code/factoriax ./scripts/snellius_submit_easy_rocket.sh
#
# Edit ``seeds`` and ``fixed_env_flag`` at the top of the script for
# seed / layout-fixing changes.
# ------------------------------------------------------------------

set -euo pipefail

# ---- Sweep parameters (edit to add more jobs) --------------------------------
# A single override (NUM_ENVS / TOTAL_STEPS / ...) seeds the one-job default;
# for a real sweep, edit the arrays directly.

seeds=(0)
total_steps_list=("${TOTAL_STEPS:-1_000_000_000}")   # 1B steps — a real "higher count" test.
num_envs_list=("${NUM_ENVS:-2048}")                  # 16x16 env is tiny; A100 fits thousands.
rollout_steps_list=("${ROLLOUT_STEPS:-128}")
run_names=("${RUN_NAME:-ppo_easy_rocket_procgen}")

# Set to "true" to broadcast a single reset key to all parallel envs (every
# worker draws the same procgen layout — layout-invariance ablation).
# "false" keeps the default per-env keyed reset.
fixed_env_seed=false
fixed_env_flag=""
[[ "${fixed_env_seed}" == "true" ]] && fixed_env_flag="--fixed-env-seed"

# ---- SLURM / environment (override via env vars, defaults below) -------------

# Resolve the repo root from this script's own location so PROJECT_DIR
# works no matter whose home directory the checkout lives in.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROJECT_DIR="${PROJECT_DIR:-${DEFAULT_PROJECT_DIR}}"
PARTITION="${PARTITION:-gpu_a100}"
WALL_TIME="${WALL_TIME:-04:00:00}"
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
WANDB_PROJECT="${WANDB_PROJECT:-factoriax_easy_rocket}"
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
    echo "    VENV_DIR=/path/to/.venv ./scripts/snellius_submit_easy_rocket.sh" >&2
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
#SBATCH --job-name=easy_rocket_ppo
#SBATCH --output=${SLURM_OUT_DIR}/easy_rocket_%j.out

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

# easy_rocket is global-obs (no --obs-radius); each env gets its own procgen
# terrain automatically via make("EasyRocket-v1"), unless --fixed-env-seed
# is passed (then all envs share the same layout).
"\${PY}" -m baselines.easy_rocket.train_ppo \\
    --num-envs ${num_envs} \\
    --rollout-steps ${rollout_steps} \\
    --total-steps ${total_steps} \\
    --max-timesteps 2000 \\
    --seed ${seed} \\
    --log-interval 32 \\
    ${fixed_env_flag} \\
    --use-wandb \\
    --wandb-project ${WANDB_PROJECT} \\
    --wandb-run-name ${run_name}
EOF

    sleep 0.1
done
