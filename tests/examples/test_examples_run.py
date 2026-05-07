"""Smoke tests for the runnable examples under ``examples/``.

Each example exposes a ``main()`` with a step / seed budget so the
test can call it with a tiny budget. They're pinned against API
drift, not numerical accuracy — the contract is "doesn't crash and
returns the documented dict shape."

Marked ``@pytest.mark.slow`` because each test instantiates a
different wrapper stack and pays JAX's JIT compile cost (~5-10s
per distinct graph). The pre-commit hook skips slow tests; CI runs
them.

Adding a new example: drop ``examples/<name>.py`` and add a single
test method here. The example's ``main()`` should accept the
keyword args the test passes and return ``None`` or a dict.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make ``examples/`` importable as a top-level package for the tests.
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from examples import (  # noqa: E402
    batched_evaluation,
    custom_level,
    custom_reward,
    train_random_policy,
    wrapped_env,
)

pytestmark = pytest.mark.slow


def test_wrapped_env_runs() -> None:
    """``wrapped_env`` builds the canonical stack and steps a few times."""
    wrapped_env.main(steps=2)


def test_train_random_policy_returns_stats() -> None:
    """``train_random_policy`` returns a dict with mean / total reward."""
    stats = train_random_policy.main(steps=8)
    assert set(stats) == {"mean_reward", "total_reward"}
    assert isinstance(stats["mean_reward"], float)


def test_custom_level_round_trips() -> None:
    """``custom_level`` builds, saves, loads, and steps without raising."""
    custom_level.main(steps=2)


def test_custom_reward_runs() -> None:
    """``custom_reward`` returns ore-mining stats."""
    stats = custom_reward.main(steps=8)
    assert set(stats) == {"total_reward", "ore_mined"}
    assert stats["total_reward"] >= 0.0
    assert stats["ore_mined"] >= 0.0


def test_batched_evaluation_runs() -> None:
    """``batched_evaluation`` returns mean/std over a tiny vmap batch."""
    stats = batched_evaluation.main(num_seeds=2, steps=4)
    assert set(stats) == {"mean_return", "std_return"}
    assert stats["std_return"] >= 0.0
