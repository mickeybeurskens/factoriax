"""Tests for the science lab building.

Every non-trivial science lab invariant is tested here so that:

1. Placing a ``SCIENCE_LAB`` item creates an entity with the right
   machine type.
2. The per-step delta (``EnvState.science_consumed_step``) correctly
   sums the packs in every active lab's input slots and is reset to
   zero on steps where nothing was consumed.
3. Non-science-pack items sitting in a lab slot are left alone (the
   lab is a pack sink, not a generic consumer).

The engine exposes ``science_consumed_step`` as its
input, so these invariants are load-bearing for anything built on top.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from factoriax.engine.constants import (
    BlockType,
    ItemType,
    Machine,
)
from factoriax.engine.step import run_labs

# ---------------------------------------------------------------------------
# Low-level run_labs tests — exercise the reduction directly without an env.
# ---------------------------------------------------------------------------


def _lab_state(
    state_factory,
    lab_slot_types: tuple[tuple[int, int], ...],
    lab_slot_counts: tuple[tuple[int, int], ...],
    lab_positions: tuple[tuple[int, int], ...] = ((3, 3),),
) -> object:
    """Build a state with one or more SCIENCE_LAB entities pre-loaded.

    Args:
        state_factory: pytest fixture.
        lab_slot_types: per-lab ``(slot0_type, slot1_type)``.
        lab_slot_counts: per-lab ``(slot0_count, slot1_count)``.
        lab_positions: per-lab ``(x, y)`` placements.

    Returns:
        ``EnvState`` with labs placed and slots populated.
    """
    h = w = 8
    world_map = jnp.full((h, w), int(BlockType.DIRT), dtype=jnp.int32)
    mt_grid = jnp.full((h, w), int(Machine.NONE), dtype=jnp.int32)
    ait = jnp.zeros((h, w, 2), dtype=jnp.int32)
    aic = jnp.zeros((h, w, 2), dtype=jnp.int32)
    for (x, y), types, counts in zip(
        lab_positions, lab_slot_types, lab_slot_counts, strict=True
    ):
        mt_grid = mt_grid.at[y, x].set(int(Machine.SCIENCE_LAB))
        ait = ait.at[y, x, 0].set(types[0])
        ait = ait.at[y, x, 1].set(types[1])
        aic = aic.at[y, x, 0].set(counts[0])
        aic = aic.at[y, x, 1].set(counts[1])

    return state_factory(
        world_map=world_map,
        machine_types=mt_grid,
        asm_in_type=ait,
        asm_in_count=aic,
    )


class TestRunLabsDelta:
    """``run_labs`` produces the right per-type delta on one or many labs."""

    def test_empty_lab_yields_zero_delta(self, state_factory) -> None:
        state = _lab_state(
            state_factory,
            lab_slot_types=((0, 0),),
            lab_slot_counts=((0, 0),),
        )
        new_state = run_labs(state)
        assert tuple(new_state.science_consumed_step.tolist()) == (0, 0, 0)

    def test_single_basic_pack(self, state_factory) -> None:
        state = _lab_state(
            state_factory,
            lab_slot_types=((int(ItemType.TIER1_SCIENCE_PACK), 0),),
            lab_slot_counts=((5, 0),),
        )
        new_state = run_labs(state)
        assert tuple(new_state.science_consumed_step.tolist()) == (5, 0, 0)

    def test_single_advanced_pack(self, state_factory) -> None:
        state = _lab_state(
            state_factory,
            lab_slot_types=((0, int(ItemType.TIER2_SCIENCE_PACK)),),
            lab_slot_counts=((0, 3),),
        )
        new_state = run_labs(state)
        assert tuple(new_state.science_consumed_step.tolist()) == (0, 3, 0)

    def test_single_ultimate_pack(self, state_factory) -> None:
        state = _lab_state(
            state_factory,
            lab_slot_types=((int(ItemType.TIER3_SCIENCE_PACK), 0),),
            lab_slot_counts=((2, 0),),
        )
        new_state = run_labs(state)
        assert tuple(new_state.science_consumed_step.tolist()) == (0, 0, 2)

    def test_both_pack_types_in_one_lab(self, state_factory) -> None:
        state = _lab_state(
            state_factory,
            lab_slot_types=(
                (
                    int(ItemType.TIER1_SCIENCE_PACK),
                    int(ItemType.TIER2_SCIENCE_PACK),
                ),
            ),
            lab_slot_counts=((2, 4),),
        )
        new_state = run_labs(state)
        assert tuple(new_state.science_consumed_step.tolist()) == (2, 4, 0)

    def test_slots_zeroed_after_consumption(self, state_factory) -> None:
        """Slots that held packs are cleared so double-counting is impossible."""
        state = _lab_state(
            state_factory,
            lab_slot_types=((int(ItemType.TIER1_SCIENCE_PACK), 0),),
            lab_slot_counts=((7, 0),),
        )
        new_state = run_labs(state)
        # Find the active lab entity.
        idx = int(jnp.argmax(new_state.ent_type == int(Machine.SCIENCE_LAB)))
        assert int(new_state.ent_asm_in_type[idx, 0]) == 0
        assert int(new_state.ent_asm_in_count[idx, 0]) == 0

    def test_multi_lab_aggregation(self, state_factory) -> None:
        """Delta sums across every active lab's contributions."""
        state = _lab_state(
            state_factory,
            lab_slot_types=(
                (int(ItemType.TIER1_SCIENCE_PACK), 0),
                (0, int(ItemType.TIER2_SCIENCE_PACK)),
                (
                    int(ItemType.TIER1_SCIENCE_PACK),
                    int(ItemType.TIER2_SCIENCE_PACK),
                ),
            ),
            lab_slot_counts=((4, 0), (0, 1), (2, 2)),
            lab_positions=((1, 1), (3, 3), (5, 5)),
        )
        new_state = run_labs(state)
        # basic: 4 + 0 + 2 = 6, advanced: 0 + 1 + 2 = 3.
        assert tuple(new_state.science_consumed_step.tolist()) == (6, 3, 0)

    def test_non_pack_items_not_consumed(self, state_factory) -> None:
        """A lab slot holding iron plate is left alone, delta is zero."""
        state = _lab_state(
            state_factory,
            lab_slot_types=((int(ItemType.IRON_PLATE), 0),),
            lab_slot_counts=((10, 0),),
        )
        new_state = run_labs(state)
        assert tuple(new_state.science_consumed_step.tolist()) == (0, 0, 0)
        idx = int(jnp.argmax(new_state.ent_type == int(Machine.SCIENCE_LAB)))
        # Non-pack item stays put.
        assert int(new_state.ent_asm_in_type[idx, 0]) == int(ItemType.IRON_PLATE)
        assert int(new_state.ent_asm_in_count[idx, 0]) == 10


# ---------------------------------------------------------------------------
# Full-env integration — ensure factoriax_step wires run_labs in correctly.
# ---------------------------------------------------------------------------


class TestLabInEnvStep:
    """``factoriax_step`` runs the lab and publishes the delta each tick.

    Consumes ``canonical_env_8x8_1p`` (root conftest, session-scoped)
    so the JIT compile of ``env.step_env`` is paid once for the entire
    session instead of twice per test class.
    """

    def test_delta_resets_each_step(self, canonical_env_8x8_1p) -> None:
        """With no labs consuming, delta is zero every step."""
        _, params, jit_step_fn, state = canonical_env_8x8_1p
        _, state, _, _, _ = jit_step_fn(jax.random.PRNGKey(1), state, 0, params)
        assert tuple(state.science_consumed_step.tolist()) == (0, 0, 0)

    def test_step_consumes_and_resets(
        self, canonical_env_8x8_1p, state_factory
    ) -> None:
        """A lab with packs consumes them in one step; next step is zero."""
        _, params, jit_step_fn, _ = canonical_env_8x8_1p
        # Build a state with one loaded lab.
        state = _lab_state(
            state_factory,
            lab_slot_types=(
                (
                    int(ItemType.TIER1_SCIENCE_PACK),
                    int(ItemType.TIER2_SCIENCE_PACK),
                ),
            ),
            lab_slot_counts=((3, 2),),
        )
        _, state_after, _, _, _ = jit_step_fn(jax.random.PRNGKey(1), state, 0, params)
        assert tuple(state_after.science_consumed_step.tolist()) == (3, 2, 0)

        # Second step: slots were zeroed by the first step, so delta is 0.
        _, state_after2, _, _, _ = jit_step_fn(
            jax.random.PRNGKey(2), state_after, 0, params
        )
        assert tuple(state_after2.science_consumed_step.tolist()) == (0, 0, 0)


# ---------------------------------------------------------------------------
# Constants sanity.
# ---------------------------------------------------------------------------


class TestScienceConstants:
    """New constants declared in factoriax.engine.constants cohere."""

    def test_science_pack_index_maps_all_packs(self) -> None:
        from factoriax.engine.tables import SCIENCE_PACK_INDEX

        assert int(SCIENCE_PACK_INDEX[int(ItemType.TIER1_SCIENCE_PACK)]) == 0
        assert int(SCIENCE_PACK_INDEX[int(ItemType.TIER2_SCIENCE_PACK)]) == 1
        assert int(SCIENCE_PACK_INDEX[int(ItemType.TIER3_SCIENCE_PACK)]) == 2
        # Non-pack items map to the sentinel.
        assert int(SCIENCE_PACK_INDEX[int(ItemType.IRON_PLATE)]) == -1
        assert int(SCIENCE_PACK_INDEX[int(ItemType.EMPTY)]) == -1

    def test_science_lab_is_placeable(self) -> None:
        from factoriax.engine.constants import PLACEABLE_ITEM_LIST

        assert int(ItemType.SCIENCE_LAB) in PLACEABLE_ITEM_LIST

    def test_lab_slot_roles(self) -> None:
        """The lab has two INPUT slots in MACHINE_SLOT_ROLES."""
        from factoriax.engine.constants import SlotRole
        from factoriax.playground.editor.slot_display import MACHINE_SLOT_ROLES

        roles = MACHINE_SLOT_ROLES[int(Machine.SCIENCE_LAB)]
        assert roles[0] == int(SlotRole.INPUT)
        assert roles[1] == int(SlotRole.INPUT)
        # Padding slots beyond the lab's two are NONE.
        for i in range(2, len(roles)):
            assert roles[i] == int(SlotRole.NONE)
