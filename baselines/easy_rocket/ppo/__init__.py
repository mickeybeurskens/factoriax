"""Shared PPO training infrastructure for FactoriaX baselines.

Provides the common building blocks used by all baseline training scripts:
network architecture, observation normalization, GAE computation, and the
PPO update function. Each baseline imports these and wires in its own
reward function, level sampling strategy, and domain-specific configuration.
"""

from baselines.ppo.config import PPOConfig
from baselines.ppo.gae import Transition, compute_gae
from baselines.ppo.loss import PPOHyperParams, make_update_fn
from baselines.ppo.network import ActorCritic
from baselines.ppo.normalization import (
    RunningStats,
    init_running_stats,
    normalize_obs,
    update_running_stats,
)

__all__ = [
    "ActorCritic",
    "PPOConfig",
    "PPOHyperParams",
    "RunningStats",
    "Transition",
    "compute_gae",
    "init_running_stats",
    "make_update_fn",
    "normalize_obs",
    "update_running_stats",
]
