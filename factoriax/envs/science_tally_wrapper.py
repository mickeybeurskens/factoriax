"""Gymnax wrapper that tallies science pack consumption across an episode.

The engine's per-step ``EnvState.science_consumed_step`` is a transient
delta — zero most steps, non-zero whenever a :class:`MachineType.SCIENCE_LAB`
consumed packs this tick. This wrapper folds those deltas into a cumulative
per-type total on a composite state so downstream reward functions,
analyses, or achievements can ask "how much basic vs advanced science has
been consumed so far?" without walking the trajectory.

Pass-through for everything else (observations, action space, reset): the
wrapper is additive and safe to stack underneath any existing wrappers.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces  # type: ignore[import-untyped]

from factoriax.constants import NUM_SCIENCE_PACK_TYPES
from factoriax.envs.factoriax_env import FactoriaXEnv
from factoriax.state import EnvParams, EnvState


class ScienceTallyState(struct.PyTreeNode):  # type: ignore[no-untyped-call]
    """Composite state pairing an env state with a cumulative science total.

    Attributes:
        env_state: Inner environment state.
        total_science_consumed: Running total, shape
            ``(NUM_SCIENCE_PACK_TYPES,)``, int32. Index 0 is basic,
            index 1 is advanced.
    """

    env_state: EnvState
    total_science_consumed: jnp.ndarray


class ScienceTallyWrapper(environment.Environment[ScienceTallyState, EnvParams]):  # type: ignore[misc]
    """Accumulate per-step science pack consumption into a running total.

    Reads ``new_env_state.science_consumed_step`` after every step and
    adds it to ``total_science_consumed``. Exposes the total through the
    wrapped state; does not inject it into the observation vector (that
    decision is left to the benchmark that composes this wrapper).

    Args:
        inner: Any gymnax-compatible env whose state type is :class:`EnvState`
            and which exposes ``science_consumed_step`` on every step.
    """

    def __init__(self, inner: FactoriaXEnv) -> None:
        """Initialize the wrapper.

        Args:
            inner: Core environment instance.
        """
        super().__init__()
        self._inner = inner

    @property
    def default_params(self) -> EnvParams:
        """Return default environment parameters."""
        return self._inner.default_params

    def step_env(
        self,
        key: jax.Array,
        state: ScienceTallyState,
        action: int | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, ScienceTallyState, jax.Array, jax.Array, dict[str, Any]]:
        """Forward to inner, then fold the per-step delta into the total."""
        obs, new_env, reward, done, info = self._inner.step_env(
            key,
            state.env_state,
            action,
            params,
        )
        new_total = state.total_science_consumed + new_env.science_consumed_step.astype(
            state.total_science_consumed.dtype
        )
        new_state = state.replace(
            env_state=new_env,
            total_science_consumed=new_total,
        )
        return obs, new_state, reward, done, info

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, ScienceTallyState]:
        """Reset the inner env and zero the tally."""
        obs, env_state = self._inner.reset_env(key, params)
        state = ScienceTallyState(
            env_state=env_state,
            total_science_consumed=jnp.zeros(NUM_SCIENCE_PACK_TYPES, dtype=jnp.int32),
        )
        return obs, state

    def get_obs(self, state: ScienceTallyState, params: EnvParams) -> jax.Array:
        """Pass-through observation from the inner env."""
        return self._inner.get_obs(state.env_state, params)

    def is_terminal(self, state: ScienceTallyState, params: EnvParams) -> jax.Array:
        """Delegate termination to the inner env."""
        return self._inner.is_terminal(state.env_state, params)

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        """Action space is unchanged."""
        return self._inner.action_space(params)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space is unchanged."""
        return self._inner.observation_space(params)
