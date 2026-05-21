"""Unit tests for :class:`factoriax.envs.action_mask_wrapper.ActionMaskWrapper`.

The wrapper rewrites blocked actions to ``Action.NOOP`` before calling
``inner.step_env``. These tests exercise the rewriting logic directly
on a stub inner env so the wrapper's masking can be validated without
paying a real env compile. The previous coverage came indirectly via
``tests/scenarios/test_runner.py``'s per-level mask integration tests
(~21s of XLA compile across three tests); the unit tests below run in
milliseconds and exercise the same property.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import pytest

from factoriax.constants import Action
from factoriax.envs.action_mask_wrapper import ActionMaskWrapper


class _StubInner:
    """Minimal stand-in for the inner env.

    ``step_env`` records the action it was called with so tests can
    assert what ``ActionMaskWrapper`` passed through. Returns a dummy
    five-tuple in the gymnax-style shape.
    """

    default_params = None  # ActionMaskWrapper.__init__ doesn't read this.

    def __init__(self) -> None:
        self.last_action: int | None = None

    def step_env(
        self,
        key: jax.Array,
        state: Any,
        action: int | jax.Array,
        params: Any,
    ) -> tuple[jax.Array, Any, jax.Array, jax.Array, dict[str, Any]]:
        self.last_action = int(jnp.asarray(action))
        return (
            jnp.zeros(1),
            state,
            jnp.float32(0.0),
            jnp.bool_(False),
            {},
        )

    def reset_env(self, key: jax.Array, params: Any) -> tuple[jax.Array, Any]:
        return jnp.zeros(1), object()

    def get_obs(self, state: Any, params: Any) -> jax.Array:
        return jnp.zeros(1)

    def is_terminal(self, state: Any, params: Any) -> jax.Array:
        return jnp.bool_(False)

    def action_space(self, params: Any) -> Any:
        return None

    def observation_space(self, params: Any) -> Any:
        return None


@pytest.fixture
def stub_and_wrapper():
    """Return (stub_inner, wrapper) with RIGHT blocked."""
    inner = _StubInner()
    wrapper = ActionMaskWrapper(inner, [int(Action.RIGHT)])
    return inner, wrapper


def test_blocked_action_rewritten_to_noop(stub_and_wrapper) -> None:
    """A blocked action becomes NOOP at the inner env's call site."""
    inner, wrapper = stub_and_wrapper
    wrapper.step_env(jax.random.PRNGKey(0), object(), int(Action.RIGHT), None)
    assert inner.last_action == int(Action.NOOP)


def test_unblocked_action_passes_through(stub_and_wrapper) -> None:
    """An action not in the mask reaches the inner env unchanged."""
    inner, wrapper = stub_and_wrapper
    wrapper.step_env(jax.random.PRNGKey(0), object(), int(Action.UP), None)
    assert inner.last_action == int(Action.UP)


def test_noop_passes_through(stub_and_wrapper) -> None:
    """NOOP is never blocked even if someone tries to mask it."""
    inner, wrapper = stub_and_wrapper
    wrapper.step_env(jax.random.PRNGKey(0), object(), int(Action.NOOP), None)
    assert inner.last_action == int(Action.NOOP)


def test_empty_mask_passes_everything_through() -> None:
    """A wrapper with no blocked actions is the identity at step time."""
    inner = _StubInner()
    wrapper = ActionMaskWrapper(inner, [])
    wrapper.step_env(jax.random.PRNGKey(0), object(), int(Action.MINE), None)
    assert inner.last_action == int(Action.MINE)


def test_multiple_blocked_actions() -> None:
    """All actions in the mask are blocked; others pass through."""
    inner = _StubInner()
    wrapper = ActionMaskWrapper(
        inner, [int(Action.RIGHT), int(Action.LEFT), int(Action.MINE)]
    )
    wrapper.step_env(jax.random.PRNGKey(0), object(), int(Action.MINE), None)
    assert inner.last_action == int(Action.NOOP)
    wrapper.step_env(jax.random.PRNGKey(0), object(), int(Action.LEFT), None)
    assert inner.last_action == int(Action.NOOP)
    wrapper.step_env(jax.random.PRNGKey(0), object(), int(Action.UP), None)
    assert inner.last_action == int(Action.UP)
