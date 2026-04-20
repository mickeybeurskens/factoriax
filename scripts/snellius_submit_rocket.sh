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
# One-time setup on the *login node*:
#   # Install uv (~10 MB, no module needed).
#   curl -LsSf https://astral.sh/uv/install.sh | sh
#   source $HOME/.local/bin/env
#
#   # Sync the project's dependencies into .venv (CUDA-enabled JAX wheels).
#   cd /home/mbeurskens/projects/factoriax
#   uv sync
#
#   # Authenticate wandb once so ~/.netrc is populated.
#   uv run wandb login
#
# After the above, this submit script uses `uv run` on the compute node,
# which skips re-installation and just activates the existing .venv.
#
# Optional: if you start hitting your home-directory quota, point uv's
# cache at scratch:
#   echo 'export UV_CACHE_DIR=/scratch-shared/$USER/uv-cache' >> ~/.bashrc
# ------------------------------------------------------------------

set -euo pipefail

# ---- Sweep parameters (edit to add more jobs) --------------------------------

seeds=(0)
total_steps_list=(100000000)         # 100M for a cheap first testing run.
num_envs_list=(2048)
rollout_steps_list=(128)
run_names=("ppo_rocket_100M_a100_test")

# ---- SLURM / environment settings --------------------------------------------

PARTITION="gpu_a100"                 # cheap partition for testing
WALL_TIME="02:00:00"                 # 100M steps on an A100 should be well under 1h; 2h is a safety margin.
CPUS_PER_TASK=18                     # Snellius allocates 18 cores with 1 A100.
SLURM_OUT_DIR="/home/mbeurskens/slurm"
PROJECT_DIR="/home/mbeurskens/projects/factoriax"
WANDB_PROJECT="factoriax_rocket"

mkdir -p "${SLURM_OUT_DIR}"

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
# uv installs a shim script that puts its binary on PATH.
source \$HOME/.local/bin/env

# JAX tends to grab all VRAM by default; leave some headroom for the
# CUDA context. On A100 80GB this is ~72GB usable.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

cd ${PROJECT_DIR}

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
