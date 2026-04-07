"""Inventory item-type swap for the interactive player UI.

Pure Python, no pygame dependency, so it can be unit-tested headlessly.
This is a UI-only convenience for rearranging inventory layout and is
intentionally outside the JAX action pipeline. Game-affecting transfers
(deposit, withdraw, rotate) are handled by :mod:`factoriax.game_logic`
so that RL agents and the interactive player share the same mechanics.
"""

from __future__ import annotations

from factoriax.state import EnvState


def swap_inventory_slots(
    state: EnvState,
    player_idx: int,
    type_a: int,
    type_b: int,
) -> EnvState:
    """Swap the counts of two item types in a player's inventory pouch.

    When *type_a* and *type_b* are different, their counts are exchanged.
    Same-type calls are a no-op.

    Args:
        state: Current environment state.
        player_idx: Index of the acting player.
        type_a: First item type index.
        type_b: Second item type index.

    Returns:
        Updated state with counts exchanged between the two types.
    """
    if type_a == type_b:
        return state

    count_a = state.player_inventory[player_idx, type_a]
    count_b = state.player_inventory[player_idx, type_b]
    new_inv = (
        state.player_inventory
        .at[player_idx, type_a].set(count_b)
        .at[player_idx, type_b].set(count_a)
    )
    return state.replace(player_inventory=new_inv)
