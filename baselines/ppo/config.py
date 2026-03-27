"""Shared PPO hyperparameters and runtime settings.

Each baseline embeds a ``PPOConfig`` instance as a field in its own
domain-specific config dataclass, following composition over inheritance.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class PPOConfig:
    """Core PPO hyperparameters shared across all baselines.

    Attributes:
        hidden_dims: Sizes of shared MLP hidden layers.
        num_envs: Number of parallel training environments.
        rollout_steps: Steps collected per environment per iteration.
        total_steps: Total environment steps to train for.
        learning_rate: Adam learning rate.
        gamma: Discount factor.
        gae_lambda: GAE lambda smoothing parameter.
        clip_eps: PPO surrogate clipping epsilon.
        value_coef: Weight of the value loss term.
        entropy_coef: Weight of the entropy bonus.
        update_epochs: PPO update epochs per data collection batch.
        num_minibatches: Minibatches per epoch.
        max_grad_norm: Global gradient clipping norm.
        normalize_obs: Whether to apply online observation normalization.
        obs_radius: Half-width of the local observation window in tiles.
        seed: Random seed.
        save_path: Directory for checkpoints (None = disabled).
        log_interval: Iterations between console/W&B log lines.
        use_wandb: Whether to log to Weights & Biases.
        wandb_project: W&B project name.
        wandb_run_name: W&B run name (None = auto-generated).
    """

    hidden_dims: tuple[int, ...] = (256, 256)
    num_envs: int = 64
    rollout_steps: int = 128
    total_steps: int = 10_000_000
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    update_epochs: int = 4
    num_minibatches: int = 8
    max_grad_norm: float = 0.5
    normalize_obs: bool = True
    obs_radius: int = 7
    seed: int = 0
    save_path: str | None = None
    log_interval: int = 10
    use_wandb: bool = False
    wandb_project: str = "factoriax"
    wandb_run_name: str | None = None
