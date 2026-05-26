"""FactoriaX environment module.

The core :class:`FactoriaXEnv` is the engine; achievement evaluation,
auto-reset, action masking, science-pack tallies, and local observation
windows compose on top via lightweight wrappers.
"""

from factoriax.engine.envs.action_mask_wrapper import ActionMaskWrapper
from factoriax.engine.envs.auto_reset_wrapper import AutoResetState, AutoResetWrapper
from factoriax.engine.envs.factoriax_env import FactoriaXEnv
from factoriax.engine.envs.local_observation_wrapper import LocalObservationWrapper
from factoriax.engine.envs.science_tally_wrapper import (
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
