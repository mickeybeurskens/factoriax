"""Every declared achievement set obeys the wire format.

The engine ships no achievements. A scenario declares an ordered tuple, and
the bit order of that tuple is a wire format: a recording, a trained policy,
and a plot all read position, not name. This file runs the same invariants
over every set any scenario declares, so a new set cannot break the format
quietly.

It is not in the mirror because it belongs to no one scenario.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from factoriax.engine.achievements import (
    Achievement,
    achievement_fn,
    achievement_weights,
    index_of,
    max_score,
)
from factoriax.engine.constants import MAX_ACHIEVEMENTS
from factoriax.engine.envs.easy_rocket import EASY_ROCKET_ACHIEVEMENTS
from factoriax.engine.envs.miner_curriculum import MINER_BOOTSTRAP_ACHIEVEMENTS
from factoriax.engine.envs.rocket import ROCKET_ACHIEVEMENTS
from factoriax.playground.play.achievements import FREE_PLAY_ACHIEVEMENTS

#: Every achievement set in the tree, by the name it is exported under.
ALL_SETS: dict[str, tuple[Achievement, ...]] = {
    "Rocket-v1": ROCKET_ACHIEVEMENTS,
    "EasyRocket-v1": EASY_ROCKET_ACHIEVEMENTS,
    "MinerBootstrap-v1": MINER_BOOTSTRAP_ACHIEVEMENTS,
    "free-play": FREE_PLAY_ACHIEVEMENTS,
}

_CASES = pytest.mark.parametrize("achievements", ALL_SETS.values(), ids=ALL_SETS.keys())


@_CASES
def test_ids_are_unique(achievements) -> None:
    """Ids identify bits in logs. Duplicates make the lookup ambiguous."""
    ids = [a.id for a in achievements]
    assert len(ids) == len(set(ids))


@_CASES
def test_ids_are_non_empty_and_stripped(achievements) -> None:
    """Ids become metric keys, so stray whitespace is a real hazard."""
    for a in achievements:
        assert a.id and a.id == a.id.strip()


@_CASES
def test_fits_the_slot_budget(achievements) -> None:
    """A set larger than the state vector loses bits in silence."""
    assert 0 < len(achievements) <= MAX_ACHIEVEMENTS


@_CASES
def test_index_of_round_trips(achievements) -> None:
    """Every declared id resolves back to its own position."""
    for i, a in enumerate(achievements):
        assert index_of(achievements, a.id) == i


@_CASES
def test_weights_are_padded_and_non_negative(achievements) -> None:
    """Padding slots must never pay, and no bit can pay a negative amount."""
    w = achievement_weights(achievements)
    assert w.shape == (MAX_ACHIEVEMENTS,)
    assert w.dtype == jnp.float32
    assert float(jnp.min(w)) >= 0.0
    assert float(jnp.sum(w[len(achievements) :])) == 0.0


@_CASES
def test_max_score_matches_the_weight_vector(achievements) -> None:
    """The advertised ceiling must be what the reward can actually pay."""
    w = achievement_weights(achievements)
    assert float(jnp.sum(w)) == pytest.approx(max_score(achievements))


@_CASES
def test_at_least_one_bit_pays(achievements) -> None:
    """An all-zero set makes its scenario unlearnable in silence."""
    assert max_score(achievements) > 0.0


@_CASES
def test_conditions_evaluate_to_a_padded_bool_vector(
    achievements, state_factory
) -> None:
    """Every set plugs into FactoriaxEnv with one uniform output shape."""
    from factoriax.engine.constants import BlockType

    state = state_factory(world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32))
    bits = achievement_fn(achievements)(state)
    assert bits.shape == (MAX_ACHIEVEMENTS,)
    assert bits.dtype == jnp.bool_
    assert not bool(jnp.any(bits[len(achievements) :]))


def test_sets_do_not_share_bit_meanings() -> None:
    """Guards the decision that there is no shared default ladder.

    If two scenarios ever agree bit-for-bit, that is a shared catalogue
    growing back by accident. The analysis code is then tempted to treat
    those bits as comparable across scenarios. It is not.
    """
    seen: dict[tuple[str, ...], str] = {}
    for name, achievements in ALL_SETS.items():
        key = tuple(a.id for a in achievements)
        assert key not in seen, f"{name} duplicates {seen.get(key)}"
        seen[key] = name
