"""Tests for the achievement builders and shared condition helpers.

The engine ships no achievements of its own, so this file covers the
machinery: how a scenario's ordered tuple becomes a padded condition
function and a weight vector, and the helpers those conditions are
built from. Per-ladder behaviour is tested next to whoever declares it.
"""

import jax.numpy as jnp
import pytest

from factoriax.engine.achievements import (
    Achievement,
    achievement_fn,
    achievement_weights,
    any_assembler_has_output,
    any_buffer_nonempty,
    count_machines,
    count_total_items,
    has_machines,
    holds_item,
    index_of,
    max_score,
    mined_at_least,
    total_machines,
    total_ore_mined,
)
from factoriax.engine.constants import (
    MAX_ACHIEVEMENTS,
    NUM_ITEM_TYPES,
    BlockType,
    ItemType,
    Machine,
)

_ALWAYS = Achievement("always", lambda state: jnp.bool_(True))
_NEVER = Achievement("never", lambda state: jnp.bool_(False))


def _dirt(state_factory, **kwargs):
    """One-tile dirt world with the given overrides."""
    return state_factory(
        world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32), **kwargs
    )


def _player_inv(**items: int) -> jnp.ndarray:
    """Build a single-player pouch inventory."""
    inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
    name_to_type = {m.name: int(m) for m in ItemType}
    for name, count in items.items():
        inv = inv.at[0, name_to_type[name]].set(count)
    return inv


class TestAchievement:
    """The per-bit declaration."""

    def test_name_defaults_to_id(self) -> None:
        """Agent-facing bits reuse the id as their metric key."""
        assert _ALWAYS.name == "always"

    def test_explicit_name_wins(self) -> None:
        """A human-facing bit sets its own display name."""
        a = Achievement("first_ore", _ALWAYS.condition, name="First Ore")
        assert a.name == "First Ore"
        assert a.id == "first_ore"

    def test_defaults_are_weight_one_and_no_hint(self) -> None:
        """Unweighted sets pay 1.0 per bit; hints are UI-only."""
        assert _ALWAYS.weight == 1.0
        assert _ALWAYS.hint == ""


class TestAchievementFn:
    """Turning a set into the env's condition function."""

    def test_pads_to_max_achievements(self, state_factory) -> None:
        """Every scenario yields one state shape regardless of bit count."""
        fn = achievement_fn((_ALWAYS, _NEVER))
        bits = fn(_dirt(state_factory))
        assert bits.shape == (MAX_ACHIEVEMENTS,)
        assert bits.dtype == jnp.bool_

    def test_bit_order_follows_declaration_order(self, state_factory) -> None:
        """Index is the wire format — it must track the tuple."""
        bits = achievement_fn((_NEVER, _ALWAYS))(_dirt(state_factory))
        assert not bool(bits[0])
        assert bool(bits[1])

    def test_padding_slots_are_false(self, state_factory) -> None:
        """Unused slots never latch, so they never pay."""
        second = Achievement("also_always", lambda state: jnp.bool_(True))
        bits = achievement_fn((_ALWAYS, second))(_dirt(state_factory))
        assert bool(jnp.all(bits[:2]))
        assert not bool(jnp.any(bits[2:]))

    def test_rejects_duplicate_ids(self) -> None:
        """Two bits sharing an id makes index lookup ambiguous."""
        dupe = Achievement("always", lambda state: jnp.bool_(False))
        with pytest.raises(ValueError, match="duplicate achievement ids"):
            achievement_fn((_ALWAYS, dupe))

    def test_rejects_empty_set(self) -> None:
        """An empty set should be expressed as achievement_fn=None."""
        with pytest.raises(ValueError, match="empty"):
            achievement_fn(())

    def test_rejects_more_than_the_slot_budget(self) -> None:
        """The state vector is fixed width; overflow must not truncate."""
        too_many = tuple(
            Achievement(f"a{i}", lambda state: jnp.bool_(True))
            for i in range(MAX_ACHIEVEMENTS + 1)
        )
        with pytest.raises(ValueError, match="exceeds MAX_ACHIEVEMENTS"):
            achievement_fn(too_many)


class TestAchievementWeights:
    """Turning a set into the reward vector."""

    def test_padded_and_float32(self) -> None:
        """Shape and dtype must match what achievement_reward multiplies."""
        w = achievement_weights((_ALWAYS, _NEVER))
        assert w.shape == (MAX_ACHIEVEMENTS,)
        assert w.dtype == jnp.float32

    def test_weights_align_with_bit_order(self) -> None:
        """A per-bit weight must land on that bit's index."""
        w = achievement_weights(
            (
                Achievement("a", _ALWAYS.condition, weight=3.0),
                Achievement("b", _ALWAYS.condition, weight=8.0),
            )
        )
        assert float(w[0]) == 3.0
        assert float(w[1]) == 8.0

    def test_padding_weights_are_zero(self) -> None:
        """Padding slots must contribute no reward."""
        w = achievement_weights((_ALWAYS,))
        assert float(jnp.sum(w[1:])) == 0.0

    def test_zero_weight_bits_are_free_diagnostics(self) -> None:
        """Bits can latch for analysis while paying nothing."""
        w = achievement_weights(
            (
                Achievement("diag", _ALWAYS.condition, weight=0.0),
                Achievement("goal", _ALWAYS.condition, weight=1.0),
            )
        )
        assert float(w[0]) == 0.0
        assert float(jnp.sum(w)) == 1.0


class TestSetHelpers:
    """max_score and index_of."""

    def test_max_score_sums_weights(self) -> None:
        four = Achievement("b", _ALWAYS.condition, weight=4.0)
        assert max_score((_ALWAYS, four)) == 5.0

    def test_index_of_finds_the_bit(self) -> None:
        assert index_of((_NEVER, _ALWAYS), "always") == 1

    def test_index_of_raises_on_unknown_id(self) -> None:
        """A typo'd id must fail loudly, not silently return a wrong bit."""
        with pytest.raises(KeyError):
            index_of((_ALWAYS,), "nope")


class TestConditionHelpers:
    """The shared predicates scenarios build their bits from."""

    def test_count_total_items_empty(self, state_factory) -> None:
        assert count_total_items(_dirt(state_factory), ItemType.COAL) == 0

    def test_count_total_items_single_player(self, state_factory) -> None:
        state = _dirt(state_factory, player_inventory=_player_inv(COAL=10))
        assert count_total_items(state, ItemType.COAL) == 10

    def test_count_total_items_sums_across_players(self, state_factory) -> None:
        """Inventories are per-player; conditions ask about the team."""
        inv = jnp.zeros((2, NUM_ITEM_TYPES), dtype=jnp.int32)
        inv = inv.at[0, ItemType.IRON_ORE].set(5)
        inv = inv.at[1, ItemType.IRON_ORE].set(7)
        state = _dirt(
            state_factory,
            num_players=2,
            player_positions=jnp.array([[0, 0], [0, 0]], dtype=jnp.int32),
            player_inventory=inv,
        )
        assert count_total_items(state, ItemType.IRON_ORE) == 12

    def test_count_machines_empty_map(self, state_factory) -> None:
        assert count_machines(_dirt(state_factory), Machine.MINER) == 0

    def test_count_machines_single(self, state_factory) -> None:
        state = _dirt(
            state_factory,
            machine_types=jnp.array([[Machine.MINER]], dtype=jnp.int32),
        )
        assert count_machines(state, Machine.MINER) == 1

    def test_holds_item_respects_count(self, state_factory) -> None:
        state = _dirt(state_factory, player_inventory=_player_inv(COAL=2))
        assert bool(holds_item(state, int(ItemType.COAL)))
        assert bool(holds_item(state, int(ItemType.COAL), count=2))
        assert not bool(holds_item(state, int(ItemType.COAL), count=3))

    def test_has_machines_respects_count(self, state_factory) -> None:
        state = _dirt(
            state_factory,
            machine_types=jnp.array(
                [[Machine.MINER, Machine.MINER]], dtype=jnp.int32
            ),
        )
        assert bool(has_machines(state, int(Machine.MINER), count=2))
        assert not bool(has_machines(state, int(Machine.MINER), count=3))

    def test_total_machines_counts_every_type(self, state_factory) -> None:
        state = _dirt(
            state_factory,
            machine_types=jnp.array(
                [[Machine.MINER, Machine.PALLET]], dtype=jnp.int32
            ),
        )
        assert total_machines(state) == 2

    def test_mined_at_least_reads_the_monotone_counter(
        self, state_factory
    ) -> None:
        """Latches on what was ever mined, not what is still held."""
        mined = jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32)
        mined = mined.at[ItemType.IRON_ORE].set(4)
        state = _dirt(state_factory, items_mined=mined)
        assert bool(mined_at_least(state, int(ItemType.IRON_ORE), 4))
        assert not bool(mined_at_least(state, int(ItemType.IRON_ORE), 5))
        assert count_total_items(state, ItemType.IRON_ORE) == 0

    def test_total_ore_mined_sums_the_three_ores(self, state_factory) -> None:
        mined = jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32)
        mined = mined.at[ItemType.COAL].set(1)
        mined = mined.at[ItemType.IRON_ORE].set(2)
        mined = mined.at[ItemType.COPPER_ORE].set(3)
        assert total_ore_mined(_dirt(state_factory, items_mined=mined)) == 6

    def test_any_buffer_nonempty(self, state_factory) -> None:
        state = _dirt(
            state_factory,
            machine_types=jnp.array([[Machine.MINER]], dtype=jnp.int32),
            buffer_type=jnp.array([[ItemType.IRON_ORE]], dtype=jnp.int8),
            buffer_count=jnp.array([[3]], dtype=jnp.int16),
        )
        assert bool(any_buffer_nonempty(state, int(Machine.MINER)))
        assert not bool(any_buffer_nonempty(state, int(Machine.PALLET)))

    def test_any_buffer_nonempty_ignores_empty_buffers(
        self, state_factory
    ) -> None:
        """A placed machine that has produced nothing must not count."""
        state = _dirt(
            state_factory,
            machine_types=jnp.array([[Machine.MINER]], dtype=jnp.int32),
        )
        assert not bool(any_buffer_nonempty(state, int(Machine.MINER)))

    def test_any_assembler_has_output(self, state_factory) -> None:
        state = _dirt(
            state_factory,
            machine_types=jnp.array([[Machine.ASSEMBLER]], dtype=jnp.int32),
            asm_out_type=jnp.array([[ItemType.IRON_PLATE]], dtype=jnp.int8),
            asm_out_count=jnp.array([[1]], dtype=jnp.int16),
        )
        assert bool(any_assembler_has_output(state))

    def test_any_assembler_has_output_empty(self, state_factory) -> None:
        state = _dirt(
            state_factory,
            machine_types=jnp.array([[Machine.ASSEMBLER]], dtype=jnp.int32),
        )
        assert not bool(any_assembler_has_output(state))
