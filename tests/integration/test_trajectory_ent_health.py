"""A machine's health survives the trip through a recorded trajectory.

This file spans two subpackages, so it is not in the mirror.
``engine/placement.py`` writes ``ent_health``, and ``analysis/trajectory.py``
records and replays it. A field the recorder drops is invisible to both sides'
own tests: the engine still writes it, and the recorder still round trips
every field it knows about.
"""

from __future__ import annotations

import jax.numpy as jnp

from factoriax.engine.constants import (
    NUM_ITEM_TYPES,
    BlockType,
    Direction,
    ItemType,
)
from factoriax.engine.placement import place_machine
from factoriax.engine.state import EnvParams


class TestTrajectoryRecordsEntHealth:
    """ent_health survives the state->trajectory->state roundtrip."""

    def test_ent_health_roundtrip_preserves_values(self, state_factory) -> None:
        """A non-trivial ent_health pattern is preserved end-to-end."""
        from factoriax.analysis.trajectory import (
            states_to_trajectory,
            trajectory_to_states,
        )

        inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        inv = inv.at[0, ItemType.MINER].set(1)
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT, BlockType.DIRT], [BlockType.DIRT, BlockType.DIRT]],
                dtype=jnp.int32,
            ),
            player_position=(0, 0),
            player_direction=Direction.RIGHT,
            player_inventory=inv,
        )
        params = EnvParams()
        placed = place_machine(state, params, 0, int(ItemType.MINER))
        eidx = int(placed.tile_entity[0, 1])
        damaged = placed.replace(ent_health=placed.ent_health.at[eidx].set(33))

        traj = states_to_trajectory([damaged])
        # The field is present and shaped (B, T, MAX_M).
        assert traj.ent_health is not None
        assert traj.ent_health.shape == (1, 1, placed.ent_health.shape[0])
        # Roundtrip back into a state list.
        restored = trajectory_to_states(traj, episode=0)
        assert len(restored) == 1
        assert int(restored[0].ent_health[eidx]) == 33
