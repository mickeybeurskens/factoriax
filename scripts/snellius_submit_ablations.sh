#!/bin/bash
# Submit the easy_rocket entropy + learning-rate ablations to Snellius.
#
# Five (lr, entropy) cells, each over five seeds -> 25 A100 jobs:
#
#   entropy ablation  (lr held at 1e-4):  ent in {0.0, 0.01, 0.03}
#   lr ablation       (ent held at 0.01): lr  in {1e-4, 2.5e-4, 5e-4}
#
# The lr1e-4 / ent0.01 cell is shared between the two ablations, so it is
# submitted once (tagged ent_abblation) rather than twice. num_envs and
# rollout_steps are fixed across the sweep; only lr, entropy and seed vary.
#
# The paper-side pull (paper/scripts/experiments/achievement_curves.py)
# selects runs by the "abblation" substring in the name and then splits
# them by the LOGGED config (ppo.learning_rate / ppo.entropy_coef), so the
# shared cell lands in both ablation figures and the exact float spelling
# in the name is irrelevant.
#
# ------------------------------------------------------------------
# One-time setup on the login node (first time only):
#   cd "<your factoriax checkout>"
#   uv sync                 # creates .venv with pinned deps
#   .venv/bin/wandb login   # authenticate wandb
#
# Then, from the repo root on Snellius:
#   ./scripts/snellius_submit_ablations.sh
#
# Override any value on the command line, e.g.:
#   TOTAL_STEPS=1_500_000_000 ./scripts/snellius_submit_ablations.sh
#   SEEDS="0 1 2" WALL_TIME=00:30:00 ./scripts/snellius_submit_ablations.sh
# ------------------------------------------------------------------

set -euo pipefail

# ---- Sweep definition --------------------------------------------------------
# Each cell is "lr entropy tag". The tag only shapes the human-readable run
# name; the pull keys off config, not the tag.
cells=(
  "1e-4   0.0   ent_abblation"
  "1e-4   0.01  ent_abblation"   # shared centre (also serves the lr ablation)
  "1e-4   0.03  ent_abblation"
  "2.5e-4 0.01  lr_abblation"
  "5e-4   0.01  lr_abblation"
)
# Space-separated so SEEDS="0 1 2" overrides cleanly from the command line.
read -r -a seeds <<< "${SEEDS:-0 1 2 3 4}"

num_envs="${NUM_ENVS:-2048}"
rollout_steps="${ROLLOUT_STEPS:-128}"
# Set from the calibration plateau (~1.3x the step where return flattens).
total_steps="${TOTAL_STEPS:-2_000_000_000}"
run_prefix="${RUN_NAME:-ppo_a100}"

# ---- SLURM / environment (override via env vars, defaults below) -------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROJECT_DIR="${PROJECT_DIR:-${DEFAULT_PROJECT_DIR}}"
PARTITION="${PARTITION:-gpu_a100}"
WALL_TIME="${WALL_TIME:-01:00:00}"
if [[ -z "${CPUS_PER_TASK:-}" ]]; then
    case "${PARTITION}" in
        gpu_h100) CPUS_PER_TASK=16 ;;
        gpu_mig)  CPUS_PER_TASK=9 ;;
        *)        CPUS_PER_TASK=18 ;;
    esac
fi
SLURM_OUT_DIR="${SLURM_OUT_DIR:-${HOME}/slurm}"
WANDB_PROJECT="${WANDB_PROJECT:-factoriax_easy_rocket}"
THROUGHPUT_DIR="${THROUGHPUT_DIR:-${PROJECT_DIR}/throughput_runs}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"

mkdir -p "${SLURM_OUT_DIR}" "${THROUGHPUT_DIR}"

if [[ ! -f "${PROJECT_DIR}/pyproject.toml" ]]; then
    echo "ERROR: PROJECT_DIR=${PROJECT_DIR} does not contain pyproject.toml." >&2
    exit 1
fi
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "ERROR: no Python at ${VENV_DIR}/bin/python. Run 'uv sync' first." >&2
    exit 1
fi

echo "Config:"
echo "  PROJECT_DIR    = ${PROJECT_DIR}"
echo "  PARTITION      = ${PARTITION}"
echo "  WALL_TIME      = ${WALL_TIME}"
echo "  num_envs       = ${num_envs}"
echo "  total_steps    = ${total_steps}"
echo "  seeds          = ${seeds[*]}"
echo "  cells          = ${#cells[@]} (x ${#seeds[@]} seeds = $(( ${#cells[@]} * ${#seeds[@]} )) jobs)"
echo "  WANDB_PROJECT  = ${WANDB_PROJECT}"

# ---- Submit loop -------------------------------------------------------------

for cell in "${cells[@]}"; do
  read -r lr ent tag <<< "${cell}"
  for seed in "${seeds[@]}"; do
    run_name="${run_prefix}_${tag}_lr${lr}_ent${ent}_s${seed}"

    echo "Submitting: ${run_name} (lr=${lr} ent=${ent} seed=${seed} steps=${total_steps})"

    cat <<EOF | sbatch
#!/bin/bash
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --gpus=1
#SBATCH --time=${WALL_TIME}
#SBATCH --job-name=easy_rocket_ablation
#SBATCH --output=${SLURM_OUT_DIR}/easy_rocket_%j.out

set -euo pipefail

export PATH="${VENV_DIR}/bin:\${PATH}"
export VIRTUAL_ENV="${VENV_DIR}"
PY="${VENV_DIR}/bin/python"

# Leave headroom for the CUDA context on the 40GB A100.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

cd "${PROJECT_DIR}"

echo "=== \$(date) — host \$(hostname) ==="
nvidia-smi --query-gpu=name,memory.free,memory.total,driver_version --format=csv
"\${PY}" --version

"\${PY}" -m baselines.easy_rocket.ppo.train_ppo \\
    --num-envs ${num_envs} \\
    --rollout-steps ${rollout_steps} \\
    --total-steps ${total_steps} \\
    --lr ${lr} \\
    --entropy-coef ${ent} \\
    --max-timesteps 2000 \\
    --seed ${seed} \\
    --log-interval 32 \\
    --use-wandb \\
    --wandb-project ${WANDB_PROJECT} \\
    --wandb-run-name ${run_name} \\
    --throughput-json ${THROUGHPUT_DIR}/${run_name}.json
EOF
    sleep 0.1
  done
done
