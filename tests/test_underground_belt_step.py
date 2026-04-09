"""Test that underground belt tiles don't crash the step function.

Regression: MACHINE_POWER_CONSUMPTION and MACHINE_MINING_RATE had
only 7 elements (MachineTypes 0-6). Underground entries (7) and
exits (8) caused an index-out-of-bounds when the step function
ran refuel_machines or run_miners.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from factoriax.constants import (
    MACHINE_INVENTORY_COUNT_DTYPE,
    NUM_ITEM_TYPES,
    Direction,
    ItemType,
    MachineType,
)
from factoriax.envs.factoriax_env import FactoriaXEnv
from factoriax.machines import update_all_machines
from factoriax.state import EnvParams


class TestUndergroundBeltStepSafety:
    """Underground belt tiles must not crash the machine update loop."""

    def test_update_all_machines_with_underground_tiles(
        self, state_factory,
    ) -> None:
        """update_all_machines must not crash with underground entries/exits."""
        mt = jnp.zeros((6, 10), dtype=jnp.int32)
        mt = mt.at[2, 3].set(MachineType.UNDERGROUND_ENTRY)
        mt = mt.at[2, 6].set(MachineType.UNDERGROUND_EXIT)
        mt = mt.at[2, 2].set(MachineType.CONVEYOR_BELT)
        mt = mt.at[2, 7].set(MachineType.CONVEYOR_BELT)
        md = jnp.zeros((6, 10), dtype=jnp.int32)
        for c in [2, 3, 6, 7]:
            md = md.at[2, c].set(Direction.RIGHT)

        mi = jnp.zeros(
            (6, 10, NUM_ITEM_TYPES), dtype=MACHINE_INVENTORY_COUNT_DTYPE,
        )
        mi = mi.at[2, 2, ItemType.IRON].set(10)

        state = state_factory(
            world_map=jnp.zeros((6, 10), dtype=jnp.int32),
            machine_types=mt,
            machine_direction=md,
            machine_inventory=mi,
        )
        params = EnvParams(map_width=10, map_height=6)

        # This should not raise an IndexError.
        new_state = update_all_machines(state, params)
        assert new_state.machine_inventory.shape == state.machine_inventory.shape

    def test_full_step_with_underground_belt(
        self, state_factory,
    ) -> None:
        """A full env.step_env must not crash with underground tiles."""
        mt = jnp.zeros((6, 10), dtype=jnp.int32)
        mt = mt.at[2, 3].set(MachineType.UNDERGROUND_ENTRY)
        mt = mt.at[2, 6].set(MachineType.UNDERGROUND_EXIT)
        md = jnp.zeros((6, 10), dtype=jnp.int32)
        md = md.at[2, 3].set(Direction.RIGHT)
        md = md.at[2, 6].set(Direction.RIGHT)

        state = state_factory(
            world_map=jnp.zeros((6, 10), dtype=jnp.int32),
            machine_types=mt,
            machine_direction=md,
        )
        params = EnvParams(
            map_width=10, map_height=6, num_players=1,
        )

        env = FactoriaXEnv()
        rng = jax.random.PRNGKey(0)
        _, new_state, _, _, _ = env.step_env(
            rng, state, 0, params,
        )
        assert new_state.machine_types.shape == (6, 10)
