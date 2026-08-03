"""Tests for the playground's free-play achievement ladder."""

import jax.numpy as jnp
import pytest

from factoriax.engine.achievements import index_of
from factoriax.engine.constants import (
    MAX_ACHIEVEMENTS,
    NUM_ITEM_TYPES,
    BlockType,
    ItemType,
    Machine,
)
from factoriax.playground.play.achievements import (
    FREE_PLAY_ACHIEVEMENTS,
    free_play_conditions,
)


def _dirt(state_factory, **kwargs):
    """One-tile dirt world with the given overrides."""
    return state_factory(
        world_map=jnp.array([[BlockType.DIRT]], dtype=jnp.int32), **kwargs
    )


def _player_inv(**items: int) -> jnp.ndarray:
    inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
    name_to_type = {m.name: int(m) for m in ItemType}
    for name, count in items.items():
        inv = inv.at[0, name_to_type[name]].set(count)
    return inv


def _bit(state, achievement_id: str) -> bool:
    i = index_of(FREE_PLAY_ACHIEVEMENTS, achievement_id)
    return bool(free_play_conditions(state)[i])


class TestLadderShape:
    """Structural invariants of the set itself."""

    def test_ids_are_unique(self) -> None:
        ids = [a.id for a in FREE_PLAY_ACHIEVEMENTS]
        assert len(ids) == len(set(ids))

    def test_every_bit_has_a_keyboard_hint(self) -> None:
        """This ladder exists to teach a human. A bit without a hint
        renders an empty footer in the achievement panel."""
        missing = [a.id for a in FREE_PLAY_ACHIEVEMENTS if not a.hint]
        assert missing == []

    def test_every_bit_has_a_display_name_distinct_from_its_id(self) -> None:
        """The panel shows names, so they must be written for reading."""
        unnamed = [a.id for a in FREE_PLAY_ACHIEVEMENTS if a.name == a.id]
        assert unnamed == []

    def test_conditions_pad_to_the_state_width(self, state_factory) -> None:
        bits = free_play_conditions(_dirt(state_factory))
        assert bits.shape == (MAX_ACHIEVEMENTS,)

    def test_no_bit_is_unreachable(self) -> None:
        """Placeholder bits that can never fire were removed. A bit that
        is hardcoded False shows as permanently locked to players."""
        assert not any(
            a.id in {"hull_production", "fuel_production"}
            for a in FREE_PLAY_ACHIEVEMENTS
        )


class TestConditions:
    """What actually unlocks each bit."""

    def test_fresh_state_unlocks_nothing(self, state_factory) -> None:
        assert not bool(jnp.any(free_play_conditions(_dirt(state_factory))))

    def test_first_ore(self, state_factory) -> None:
        mined = jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32)
        mined = mined.at[ItemType.IRON_ORE].set(1)
        assert _bit(_dirt(state_factory, items_mined=mined), "first_ore")

    def test_stockpile_needs_ten(self, state_factory) -> None:
        mined = jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32)
        mined = mined.at[ItemType.IRON_ORE].set(9)
        assert not _bit(_dirt(state_factory, items_mined=mined), "stockpile")
        mined = mined.at[ItemType.IRON_ORE].set(10)
        assert _bit(_dirt(state_factory, items_mined=mined), "stockpile")

    def test_apprentice_engineer_on_any_machine_item(self, state_factory) -> None:
        state = _dirt(state_factory, player_inventory=_player_inv(PALLET=1))
        assert _bit(state, "apprentice_engineer")

    def test_coal_gathered(self, state_factory) -> None:
        state = _dirt(state_factory, player_inventory=_player_inv(COAL=1))
        assert _bit(state, "coal_gathered")

    def test_automated_mining(self, state_factory) -> None:
        state = _dirt(
            state_factory,
            machine_types=jnp.array([[Machine.MINER]], dtype=jnp.int32),
            buffer_type=jnp.array([[ItemType.IRON_ORE]], dtype=jnp.int8),
            buffer_count=jnp.array([[3]], dtype=jnp.int16),
        )
        assert _bit(state, "automated_mining")

    def test_first_pipeline(self, state_factory) -> None:
        state = _dirt(
            state_factory,
            machine_types=jnp.array([[Machine.PALLET]], dtype=jnp.int32),
            buffer_type=jnp.array([[ItemType.IRON_ORE]], dtype=jnp.int8),
            buffer_count=jnp.array([[2]], dtype=jnp.int16),
        )
        assert _bit(state, "first_pipeline")

    def test_assembler_crafted(self, state_factory) -> None:
        state = _dirt(state_factory, player_inventory=_player_inv(ASSEMBLER=1))
        assert _bit(state, "assembler_crafted")

    def test_first_assembly(self, state_factory) -> None:
        state = _dirt(
            state_factory,
            machine_types=jnp.array([[Machine.ASSEMBLER]], dtype=jnp.int32),
            asm_out_type=jnp.array([[ItemType.IRON_PLATE]], dtype=jnp.int8),
            asm_out_count=jnp.array([[1]], dtype=jnp.int16),
        )
        assert _bit(state, "first_assembly")

    def test_rocket_complete(self, state_factory) -> None:
        state = _dirt(
            state_factory,
            machine_types=jnp.array([[Machine.ROCKET]], dtype=jnp.int32),
        )
        assert _bit(state, "rocket_complete")

    def test_first_science_accepts_either_tier(self, state_factory) -> None:
        for item in ("TIER1_SCIENCE_PACK", "TIER2_SCIENCE_PACK"):
            state = _dirt(state_factory, player_inventory=_player_inv(**{item: 1}))
            assert _bit(state, "first_science"), item

    def test_advanced_science_needs_tier_two(self, state_factory) -> None:
        state = _dirt(state_factory, player_inventory=_player_inv(TIER1_SCIENCE_PACK=1))
        assert not _bit(state, "advanced_science")
        state = _dirt(state_factory, player_inventory=_player_inv(TIER2_SCIENCE_PACK=1))
        assert _bit(state, "advanced_science")


class TestMovingParts:
    """Regression: the bit used to ignore the arm its hint demands.

    Before the fix the condition counted pallets only, so placing a lone
    pallet unlocked an achievement whose name and hint both say "an arm
    and a pallet".
    """

    @pytest.mark.parametrize(
        "machines,expected",
        [
            ((Machine.PALLET, Machine.NONE), False),
            ((Machine.ARM, Machine.NONE), False),
            ((Machine.ARM, Machine.PALLET), True),
        ],
        ids=["pallet_only", "arm_only", "both"],
    )
    def test_requires_both(self, state_factory, machines, expected) -> None:
        state = _dirt(
            state_factory,
            machine_types=jnp.array([list(machines)], dtype=jnp.int32),
        )
        assert _bit(state, "moving_parts") is expected

    def test_hint_still_describes_the_condition(self) -> None:
        i = index_of(FREE_PLAY_ACHIEVEMENTS, "moving_parts")
        a = FREE_PLAY_ACHIEVEMENTS[i]
        assert "arm" in a.hint.lower()
        assert "pallet" in a.hint.lower()
