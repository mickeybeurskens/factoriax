"""Regression tests for renderer inventory and machine activity functions.

These tests guard the pouch inventory model integration in the renderer.
They verify that render_inventory_bar, is_miner_active, and is_arm_active
correctly read from player_inventory and machine_inventory.
"""

import numpy as np

from factoriax.constants import (
    NUM_ITEM_TYPES,
    ItemType,
    MachineType,
)
from factoriax.renderer import (
    INVENTORY_BAR_HEIGHT,
    is_arm_active,
    is_miner_active,
    render_inventory_bar,
)


class TestRenderInventoryBar:
    """Tests for render_inventory_bar reading from player_inventory."""

    def test_shape_and_dtype(self, state_factory) -> None:
        """Output must be RGB uint8 with correct dimensions."""
        import jax.numpy as jnp

        state = state_factory(world_map=jnp.zeros((4, 4), dtype=jnp.int32))
        bar = render_inventory_bar(state, 256)
        assert bar.dtype == np.uint8
        assert bar.shape == (INVENTORY_BAR_HEIGHT, 256, 3)

    def test_populated_inventory_renders(self, state_factory) -> None:
        """Bar renders without crash when player has items."""
        import jax.numpy as jnp

        inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        inv = inv.at[0, int(ItemType.COAL)].set(10)
        inv = inv.at[0, int(ItemType.IRON)].set(5)
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            player_inventory=inv,
        )
        bar = render_inventory_bar(state, 256)
        assert bar.shape == (INVENTORY_BAR_HEIGHT, 256, 3)

    def test_selected_item_produces_highlight(self, state_factory) -> None:
        """Selecting different items produces different pixels."""
        import jax.numpy as jnp

        state = state_factory(world_map=jnp.zeros((4, 4), dtype=jnp.int32))
        bar_a = render_inventory_bar(state, 256, selected_item=1)
        bar_b = render_inventory_bar(state, 256, selected_item=5)
        assert not np.array_equal(bar_a, bar_b)

    def test_all_item_types_selectable(self, state_factory) -> None:
        """Every non-EMPTY item type can be selected without crash."""
        import jax.numpy as jnp

        state = state_factory(world_map=jnp.zeros((4, 4), dtype=jnp.int32))
        for item_type in range(1, NUM_ITEM_TYPES):
            bar = render_inventory_bar(state, 256, selected_item=item_type)
            assert bar.shape == (INVENTORY_BAR_HEIGHT, 256, 3)


class TestIsMinerActive:
    """Tests for is_miner_active reading from machine_inventory."""

    def test_empty_miner_inactive(self, state_factory) -> None:
        """Miner with empty inventory and no power is inactive."""
        import jax.numpy as jnp

        state = state_factory(
            world_map=jnp.full((4, 4), int(ItemType.COAL), dtype=jnp.int32),
            machine_types=jnp.full(
                (4, 4), int(MachineType.MINER), dtype=jnp.int32,
            ),
            block_resources=jnp.full((4, 4), 100, dtype=jnp.int16),
        )
        assert not is_miner_active(state, 0, 0)

    def test_powered_miner_with_resources_active(self, state_factory) -> None:
        """Miner with power and resources below is active."""
        import jax.numpy as jnp

        state = state_factory(
            world_map=jnp.full((4, 4), int(ItemType.COAL), dtype=jnp.int32),
            machine_types=jnp.full(
                (4, 4), int(MachineType.MINER), dtype=jnp.int32,
            ),
            machine_power=jnp.full((4, 4), 5, dtype=jnp.int32),
            block_resources=jnp.full((4, 4), 100, dtype=jnp.int16),
        )
        assert is_miner_active(state, 0, 0)


class TestIsArmActive:
    """Tests for is_arm_active reading from machine_inventory."""

    def test_empty_arm_inactive(self, state_factory) -> None:
        """Arm with empty inventory is inactive."""
        import jax.numpy as jnp


        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=jnp.full(
                (4, 4), int(MachineType.ARM), dtype=jnp.int32,
            ),
        )
        assert not is_arm_active(state, 0, 0)

    def test_arm_with_items_active(self, state_factory) -> None:
        """Arm holding items in its inventory is active."""
        import jax.numpy as jnp

        from factoriax.constants import MACHINE_INVENTORY_COUNT_DTYPE

        machine_inv = jnp.zeros(
            (4, 4, NUM_ITEM_TYPES), dtype=MACHINE_INVENTORY_COUNT_DTYPE,
        )
        machine_inv = machine_inv.at[0, 0, int(ItemType.COAL)].set(1)
        state = state_factory(
            world_map=jnp.zeros((4, 4), dtype=jnp.int32),
            machine_types=jnp.full(
                (4, 4), int(MachineType.ARM), dtype=jnp.int32,
            ),
            machine_inventory=machine_inv,
        )
        assert is_arm_active(state, 0, 0)
