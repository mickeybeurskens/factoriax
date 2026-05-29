"""PPO training infrastructure for the easy rocket baseline.

Provides the building blocks used by ``train_ppo``: network architecture,
observation normalization, GAE computation, and the PPO update function.
``train_ppo`` imports these and wires in the scenario's reward function,
level sampling strategy, and domain-specific configuration.
"""

from baselines.easy_rocket.ppo.config import PPOConfig
from baselines.easy_rocket.ppo.gae import Transition, compute_gae
from baselines.easy_rocket.ppo.loss import PPOHyperParams, make_update_fn
from baselines.easy_rocket.ppo.network import ActorCritic
from baselines.easy_rocket.ppo.normalization import (
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
