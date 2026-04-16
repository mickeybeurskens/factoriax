"""Tests for science packs, research, and tech gating.

Uses entity-based state where assemblers auto-detect recipes from
inputs in ``ent_asm_in_type`` / ``ent_asm_in_count``.
"""

import jax.numpy as jnp
from jax import random

from factoriax.constants import (
    NUM_ITEM_TYPES,
    RESEARCH_COST,
    Action,
    BlockType,
    ItemType,
)
from factoriax.game_logic import apply_research, factoriax_step
from factoriax.machines import run_assemblers
from factoriax.state import EnvParams


class TestSciencePackRecipes:
    """Assemblers should produce science packs from correct inputs."""

    def test_basic_science_pack_crafts(self, state_factory) -> None:
        """Basic science pack: motor + tin plate -> starts craft."""
        state = state_factory(
            world_map=jnp.full((4, 4), int(BlockType.DIRT), dtype=jnp.int32),
            machine_types=jnp.zeros((4, 4), dtype=jnp.int32).at[0, 0].set(3),
            asm_in_type=jnp.zeros((4, 4, 2), dtype=jnp.int8)
            .at[0, 0, 0]
            .set(int(ItemType.MOTOR))
            .at[0, 0, 1]
            .set(int(ItemType.TIN_PLATE)),
            asm_in_count=jnp.zeros((4, 4, 2), dtype=jnp.int16)
            .at[0, 0, 0]
            .set(5)
            .at[0, 0, 1]
            .set(5),
        )
        new = run_assemblers(state)
        eid = state.tile_entity[0, 0]
        assert int(new.ent_power[eid]) == 8
        assert int(new.ent_asm_in_count[eid, 0]) == 0
        assert int(new.ent_asm_in_count[eid, 1]) == 0

    def test_advanced_science_pack_crafts(self, state_factory) -> None:
        """Advanced science pack: sensor + wafer -> starts craft."""
        state = state_factory(
            world_map=jnp.full((4, 4), int(BlockType.DIRT), dtype=jnp.int32),
            machine_types=jnp.zeros((4, 4), dtype=jnp.int32).at[0, 0].set(3),
            asm_in_type=jnp.zeros((4, 4, 2), dtype=jnp.int8)
            .at[0, 0, 0]
            .set(int(ItemType.SENSOR))
            .at[0, 0, 1]
            .set(int(ItemType.WAFER)),
            asm_in_count=jnp.zeros((4, 4, 2), dtype=jnp.int16)
            .at[0, 0, 0]
            .set(3)
            .at[0, 0, 1]
            .set(3),
        )
        new = run_assemblers(state)
        eid = state.tile_entity[0, 0]
        assert int(new.ent_power[eid]) == 8
        assert int(new.ent_asm_in_count[eid, 0]) == 0
        assert int(new.ent_asm_in_count[eid, 1]) == 0


class TestResearchAction:
    """Research actions should consume science packs and unlock techs."""

    def test_consume_basic_science_pack(self, state_factory) -> None:
        """Consuming a basic science pack increments research progress."""
        p_inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        p_inv = p_inv.at[0, ItemType.BASIC_SCIENCE_PACK].set(5)

        state = state_factory(
            world_map=jnp.full(
                (4, 4),
                int(BlockType.DIRT),
                dtype=jnp.int32,
            ),
            player_inventory=p_inv,
        )
        new = apply_research(state, 0, ItemType.BASIC_SCIENCE_PACK)
        assert int(new.research_progress[0]) == 1
        assert int(new.player_inventory[0, ItemType.BASIC_SCIENCE_PACK]) == 4
        assert not bool(new.research_unlocked[0])

    def test_research_unlocks_at_threshold(self, state_factory) -> None:
        """Tech should unlock when progress reaches RESEARCH_COST."""
        p_inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        p_inv = p_inv.at[0, ItemType.BASIC_SCIENCE_PACK].set(RESEARCH_COST)

        progress = jnp.array(
            [RESEARCH_COST - 1, 0],
            dtype=jnp.int16,
        )

        state = state_factory(
            world_map=jnp.full(
                (4, 4),
                int(BlockType.DIRT),
                dtype=jnp.int32,
            ),
            player_inventory=p_inv,
            research_progress=progress,
        )
        new = apply_research(state, 0, ItemType.BASIC_SCIENCE_PACK)
        assert bool(new.research_unlocked[0])
        assert int(new.research_progress[0]) == RESEARCH_COST

    def test_no_consume_without_science_pack(
        self,
        state_factory,
    ) -> None:
        """Research with zero science packs should be a no-op."""
        p_inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)

        state = state_factory(
            world_map=jnp.full(
                (4, 4),
                int(BlockType.DIRT),
                dtype=jnp.int32,
            ),
            player_inventory=p_inv,
        )
        new = apply_research(state, 0, ItemType.BASIC_SCIENCE_PACK)
        assert int(new.research_progress[0]) == 0

    def test_no_consume_when_already_unlocked(
        self,
        state_factory,
    ) -> None:
        """Research should not consume packs for already-unlocked tech."""
        p_inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        p_inv = p_inv.at[0, ItemType.BASIC_SCIENCE_PACK].set(5)

        state = state_factory(
            world_map=jnp.full(
                (4, 4),
                int(BlockType.DIRT),
                dtype=jnp.int32,
            ),
            player_inventory=p_inv,
            research_unlocked=jnp.array(
                [True, False],
                dtype=jnp.bool_,
            ),
        )
        new = apply_research(state, 0, ItemType.BASIC_SCIENCE_PACK)
        assert int(new.player_inventory[0, ItemType.BASIC_SCIENCE_PACK]) == 5

    def test_advanced_science_unlocks_tech_1(
        self,
        state_factory,
    ) -> None:
        """Advanced science packs should advance tech index 1."""
        p_inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        p_inv = p_inv.at[0, ItemType.ADVANCED_SCIENCE_PACK].set(1)

        state = state_factory(
            world_map=jnp.full(
                (4, 4),
                int(BlockType.DIRT),
                dtype=jnp.int32,
            ),
            player_inventory=p_inv,
        )
        new = apply_research(
            state,
            0,
            ItemType.ADVANCED_SCIENCE_PACK,
        )
        assert int(new.research_progress[1]) == 1
        assert int(new.research_progress[0]) == 0

    def test_research_via_step(self, state_factory) -> None:
        """RESEARCH_BASIC action through factoriax_step should work."""
        p_inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        p_inv = p_inv.at[0, ItemType.BASIC_SCIENCE_PACK].set(3)

        state = state_factory(
            world_map=jnp.full(
                (4, 4),
                int(BlockType.DIRT),
                dtype=jnp.int32,
            ),
            player_inventory=p_inv,
        )
        params = EnvParams(map_width=4, map_height=4, num_players=1)
        rng = random.PRNGKey(0)
        new = factoriax_step(
            rng,
            state,
            int(Action.RESEARCH_BASIC),
            params,
        )
        assert int(new.research_progress[0]) == 1
        assert int(new.player_inventory[0, ItemType.BASIC_SCIENCE_PACK]) == 2
