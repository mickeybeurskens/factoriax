"""Compose the canonical FactoriaX wrapper stack via ``factoriax.make``.

Demonstrates the recommended way to construct an environment with
local observations, action masking, and auto-reset.

Run::

    uv run python examples/wrapped_env.py
"""

from __future__ import annotations

import jax

import factoriax
from factoriax import (
    Action,
    ActionMaskWrapper,
    AutoResetWrapper,
    LocalObservationWrapper,
)


def main(steps: int = 32) -> None:
    """Build a wrapped env and step it.

    Args:
        steps: How many random NOOP steps to take. Kept tiny by default
            so the example finishes in well under a second.
    """
    env, params = factoriax.make(
        obs="local",
        obs_radius=4,
        auto_reset=True,
        blocked_actions=(int(Action.MINE),),
    )

    # Outermost-to-innermost: AutoReset → ActionMask → LocalObservation
    # → FactoriaXEnv. ``factoriax.make`` is the canonical builder.
    assert isinstance(env, AutoResetWrapper)
    assert isinstance(env._inner, ActionMaskWrapper)  # noqa: SLF001
    assert isinstance(env._inner._inner, LocalObservationWrapper)  # noqa: SLF001

    rng = jax.random.PRNGKey(0)
    rng, key_reset = jax.random.split(rng)
    obs, state = env.reset_env(key_reset, params)
    print(f"obs.shape = {obs.shape}  (local window + per-player scalars)")

    for _ in range(steps):
        rng, key_step = jax.random.split(rng)
        # Action 0 = NOOP. The action-mask wrapper would coerce
        # any blocked action to NOOP transparently.
        obs, state, reward, done, info = env.step_env(key_step, state, 0, params)

    # AutoResetWrapper boxes the engine state; the live EnvState is
    # at ``.env_state``. Wrappers above a wrapper bring their own
    # state shape — read the wrapper's docstring before reaching in.
    print(f"after {steps} steps: timestep={int(state.env_state.timestep)}")


if __name__ == "__main__":
    main(steps=200)
