"""CLI argument helpers for the easy rocket PPO baseline.

Provides ``add_ppo_args`` to register the common PPO hyperparameters on
an ``argparse.ArgumentParser``, and ``ppo_config_from_args`` to extract
them into a ``PPOConfig`` instance.
"""

from __future__ import annotations

import argparse

from baselines.easy_rocket.ppo.config import PPOConfig


def add_ppo_args(parser: argparse.ArgumentParser) -> None:
    """Add shared PPO hyperparameter arguments to a parser.

    Domain-specific arguments should be added separately by each baseline.

    Args:
        parser: ArgumentParser to extend.
    """
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--total-steps", type=int, default=10_000_000)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--no-normalize-obs", action="store_true")
    parser.add_argument("--obs-radius", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="factoriax")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument(
        "--throughput-json",
        type=str,
        default="./ppo_throughput.json",
        help=(
            "Write per-iteration timing + GPU info to this path. "
            "The GPU model (queried from nvidia-smi at start of run) "
            "is suffixed onto the basename so the same path is safe "
            "to reuse across devices and the file name matches the "
            "paper-side consolidator's ``ppo_throughput_*.json`` "
            "glob. Pass an empty string to disable."
        ),
    )


def ppo_config_from_args(
    args: argparse.Namespace,
    **overrides: object,
) -> PPOConfig:
    """Build a PPOConfig from parsed CLI arguments.

    Any keyword arguments override the CLI values, letting baselines
    set domain-appropriate defaults (e.g. a different ``wandb_project``).

    Args:
        args: Parsed argument namespace from ``add_ppo_args``.
        **overrides: Field values that take precedence over CLI args.

    Returns:
        Configured PPOConfig.
    """
    values = {
        "num_envs": args.num_envs,
        "rollout_steps": args.rollout_steps,
        "total_steps": args.total_steps,
        "learning_rate": args.lr,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_eps": args.clip_eps,
        "value_coef": args.value_coef,
        "entropy_coef": args.entropy_coef,
        "update_epochs": args.update_epochs,
        "num_minibatches": args.num_minibatches,
        "max_grad_norm": args.max_grad_norm,
        "normalize_obs": not args.no_normalize_obs,
        "obs_radius": args.obs_radius,
        "seed": args.seed,
        "save_path": args.save_path,
        "log_interval": args.log_interval,
        "use_wandb": args.use_wandb,
        "wandb_project": args.wandb_project,
        "wandb_run_name": args.wandb_run_name,
        "throughput_json": args.throughput_json,
    }
    values.update(overrides)
    return PPOConfig(**values)  # type: ignore[arg-type]
