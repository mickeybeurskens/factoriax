"""Composable post-step hooks for :class:`FactoriaXEnv`.

A hook is a pure ``(key, state, params) -> state`` transform applied after the
core physics step. Scenarios layer mechanics onto the shared simulation through
hooks — achievement checking, tallies, and similar additive behavior — without
editing the engine. Hooks run in order and each sees the previous one's output.
"""

from __future__ import annotations

from collections.abc import Callable

import jax

from factoriax.engine.state import EnvParams, EnvState

#: Pure function evaluated against the post-step state to produce achievement
#: condition bits, shape ``(MAX_ACHIEVEMENTS,)`` bool.
AchievementFn = Callable[[EnvState], jax.Array]

#: A post-step transform: takes the PRNG key, the post-step state, and params,
#: and returns a (possibly modified) state.
StepHook = Callable[[jax.Array, EnvState, EnvParams], EnvState]


def achievement_hook(condition_fn: AchievementFn) -> StepHook:
    """Return a step hook that OR-folds ``condition_fn`` into the latched mask.
    
    The returned transform evaluates ``condition_fn`` on the post-step state and
    folds the result into ``state.achievements_unlocked`` with ``|``, so unlocked
    bits latch for the rest of the episode.

    Parameters
    ----------
    condition_fn :
        Pure
    condition_fn: AchievementFn :
        

    Returns
    -------
    type
        A :data:`StepHook` that latches the evaluated bits.

    """

    def hook(key: jax.Array, state: EnvState, params: EnvParams) -> EnvState:
        """

        Parameters
        ----------
        key: jax.Array :
            
        state: EnvState :
            
        params: EnvParams :
            

        Returns
        -------

        """
        del key, params
        return state.replace(
            achievements_unlocked=state.achievements_unlocked | condition_fn(state),
        )

    return hook
