"""FactoriaX environment module.

The core :class:`FactoriaXEnv` is the engine; achievement evaluation,
auto-reset, action masking, and science-pack tallies compose on top via
lightweight wrappers. The observation variant is set on the env itself
via the ``obs`` / ``obs_radius`` constructor args — no wrapper needed.
"""

from factoriax.engine.envs.action_mask_wrapper import ActionMaskWrapper
from factoriax.engine.envs.auto_reset_wrapper import AutoResetState, AutoResetWrapper
from factoriax.engine.envs.factoriax_env import FactoriaXEnv
from factoriax.engine.envs.science_tally_wrapper import (
    ScienceTallyState,
    ScienceTallyWrapper,
)

__all__ = [
    "ActionMaskWrapper",
    "AutoResetState",
    "AutoResetWrapper",
    "FactoriaXEnv",
    "ScienceTallyState",
    "ScienceTallyWrapper",
]
