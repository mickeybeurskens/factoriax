"""FactoriaX environment module.

The core :class:`FactoriaXEnv` is the engine; achievement evaluation,
auto-reset, action masking, science-pack tallies, and local observation
windows compose on top via lightweight wrappers.
"""

from factoriax.envs.action_mask_wrapper import ActionMaskWrapper
from factoriax.envs.auto_reset_wrapper import AutoResetState, AutoResetWrapper
from factoriax.envs.factoriax_env import FactoriaXEnv
from factoriax.envs.local_observation_wrapper import LocalObservationWrapper
from factoriax.envs.science_tally_wrapper import (
    ScienceTallyState,
    ScienceTallyWrapper,
)

__all__ = [
    "ActionMaskWrapper",
    "AutoResetState",
    "AutoResetWrapper",
    "FactoriaXEnv",
    "LocalObservationWrapper",
    "ScienceTallyState",
    "ScienceTallyWrapper",
]
