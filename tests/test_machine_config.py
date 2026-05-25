"""Tests for the MachineConfig system."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from factoriax.constants import (
    MAX_HEALTH,
    BlockType,
    Direction,
    ItemType,
    MachineType,
)
from factoriax.machine_config import (
    DEFAULT_MACHINE_CONFIG,
    MachineConfig,
    MachineConfigOverride,
)
from factoriax.machine_spec import MACHINE_MAX_STACK
from factoriax.machines import run_conveyor_belts
from factoriax.state import EnvParams, EnvState


class TestMachineConfigDefault:
    def test_default_max_stack_matches_constant(self) -> None:
        """``MachineConfig.default()`` mirrors ``MACHINE_MAX_STACK``."""
        cfg = MachineConfig.default()
        assert cfg.max_stack.shape == MACHINE_MAX_STACK.shape
        assert jnp.all(cfg.max_stack == MACHINE_MAX_STACK)

    def test_default_singleton_value_equality(self) -> None:
        """:data:`DEFAULT_MACHINE_CONFIG` matches :meth:`default`."""
        assert jnp.all(
            DEFAULT_MACHINE_CONFIG.max_stack == MachineConfig.default().max_stack
        )

    def test_envparams_default_matches_module_default(self) -> None:
        """A bare ``EnvParams()`` carries :data:`DEFAULT_MACHINE_CONFIG`."""
        params = EnvParams()
        assert jnp.all(
            params.machine_config.max_stack == DEFAULT_MACHINE_CONFIG.max_stack
        )

    def test_default_max_health_is_max_health_constant(self) -> None:
        """``MachineConfig.default()`` initializes max_health=MAX_HEALTH for
        every machine type."""
        cfg = MachineConfig.default()
        assert cfg.max_health.shape == (len(MachineType),)
        assert cfg.max_health.dtype == jnp.int16
        assert jnp.all(cfg.max_health == MAX_HEALTH)

    def test_envparams_default_carries_max_health(self) -> None:
        """A bare ``EnvParams()`` exposes the default max_health array."""
        params = EnvParams()
        assert jnp.all(params.machine_config.max_health == MAX_HEALTH)


class TestMachineConfigOverrides:
    def test_empty_overrides_returns_self(self) -> None:
        """``with_overrides({})`` is a no-op."""
        base = MachineConfig.default()
        out = base.with_overrides({})
        assert jnp.all(out.max_stack == base.max_stack)

    def test_single_override_changes_only_that_index(self) -> None:
        """Overriding BELT.max_stack=1 leaves other entries untouched."""
        base = MachineConfig.default()
        out = base.with_overrides(
            {int(MachineType.CONVEYOR_BELT): MachineConfigOverride(max_stack=1)}
        )
        belt_idx = int(MachineType.CONVEYOR_BELT)
        assert int(out.max_stack[belt_idx]) == 1
        # Every other index unchanged.
        mask = jnp.arange(out.max_stack.shape[0]) != belt_idx
        assert jnp.all(out.max_stack[mask] == base.max_stack[mask])

    def test_none_field_does_not_override(self) -> None:
        """A :class:`MachineConfigOverride` with all fields ``None``
        leaves the array untouched."""
        base = MachineConfig.default()
        out = base.with_overrides(
            {int(MachineType.PALLET): MachineConfigOverride(max_stack=None)}
        )
        assert jnp.all(out.max_stack == base.max_stack)

    def test_multiple_overrides_all_applied(self) -> None:
        """Overriding two machine types simultaneously updates both."""
        base = MachineConfig.default()
        out = base.with_overrides(
            {
                int(MachineType.PALLET): MachineConfigOverride(max_stack=64),
                int(MachineType.FURNACE): MachineConfigOverride(max_stack=500),
            }
        )
        assert int(out.max_stack[int(MachineType.PALLET)]) == 64
        assert int(out.max_stack[int(MachineType.FURNACE)]) == 500
        # Untouched index unchanged.
        belt_idx = int(MachineType.CONVEYOR_BELT)
        assert int(out.max_stack[belt_idx]) == int(base.max_stack[belt_idx])

    def test_negative_max_stack_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            MachineConfig.default().with_overrides(
                {int(MachineType.PALLET): MachineConfigOverride(max_stack=-1)}
            )

    def test_out_of_range_machine_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            MachineConfig.default().with_overrides(
                {999: MachineConfigOverride(max_stack=1)}
            )

    def test_max_health_override_changes_only_that_index(self) -> None:
        """Overriding FURNACE.max_health=42 leaves other entries untouched."""
        base = MachineConfig.default()
        out = base.with_overrides(
            {int(MachineType.FURNACE): MachineConfigOverride(max_health=42)}
        )
        furnace_idx = int(MachineType.FURNACE)
        assert int(out.max_health[furnace_idx]) == 42
        mask = jnp.arange(out.max_health.shape[0]) != furnace_idx
        assert jnp.all(out.max_health[mask] == base.max_health[mask])
        # Independent: max_stack is untouched by a max_health override.
        assert jnp.all(out.max_stack == base.max_stack)

    def test_max_health_and_max_stack_overrides_compose(self) -> None:
        """Both fields on the same override apply to the same machine type."""
        base = MachineConfig.default()
        out = base.with_overrides(
            {
                int(MachineType.PALLET): MachineConfigOverride(
                    max_stack=64, max_health=200
                )
            }
        )
        idx = int(MachineType.PALLET)
        assert int(out.max_stack[idx]) == 64
        assert int(out.max_health[idx]) == 200

    def test_negative_max_health_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            MachineConfig.default().with_overrides(
                {int(MachineType.PALLET): MachineConfigOverride(max_health=-1)}
            )


class TestMachineConfigEngineWiring:
    """End-to-end: an override actually changes engine behavior."""

    def _make_two_belt_state(self, state_factory, *, src_count: int) -> EnvState:
        """Build a 1x2 grid with two RIGHT-facing belts; src tile holds
        ``src_count`` IRON_PLATE units."""
        shape = (1, 2)
        world = jnp.full(shape, int(BlockType.DIRT), dtype=jnp.int32)
        mt = jnp.full(shape, int(MachineType.CONVEYOR_BELT), dtype=jnp.int32)
        md = jnp.full(shape, int(Direction.RIGHT), dtype=jnp.int8)
        bt = jnp.zeros(shape, dtype=jnp.int8)
        bc = jnp.zeros(shape, dtype=jnp.int16)
        bt = bt.at[0, 0].set(int(ItemType.IRON_PLATE))
        bc = bc.at[0, 0].set(src_count)
        return state_factory(
            world_map=world,
            machine_types=mt,
            machine_direction=md,
            buffer_type=bt,
            buffer_count=bc,
            max_machines=2,
        )

    def test_default_belt_cap_allows_pushing(self, state_factory) -> None:
        """Default belt cap (3) lets the source belt push 1 item east."""
        state = self._make_two_belt_state(state_factory, src_count=1)
        params = EnvParams()
        out = run_conveyor_belts(state, params)
        dst_eid = int(out.tile_entity[0, 1])
        assert int(out.ent_buf_count[dst_eid]) == 1

    def test_override_zero_belt_cap_blocks_push(self, state_factory) -> None:
        """Overriding BELT.max_stack=0 makes the destination belt
        unable to accept anything — the push is rejected and the source
        belt's item stays in place."""
        state = self._make_two_belt_state(state_factory, src_count=1)
        params = EnvParams(
            machine_config=DEFAULT_MACHINE_CONFIG.with_overrides(
                {int(MachineType.CONVEYOR_BELT): MachineConfigOverride(max_stack=0)}
            )
        )
        out = run_conveyor_belts(state, params)
        src_eid = int(out.tile_entity[0, 0])
        dst_eid = int(out.tile_entity[0, 1])
        # Destination didn't accept; source still holds the item.
        assert int(out.ent_buf_count[dst_eid]) == 0
        assert int(out.ent_buf_count[src_eid]) == 1
