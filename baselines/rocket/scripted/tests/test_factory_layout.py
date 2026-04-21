"""Unit tests for the design-time factory layout.

These are pure-Python (no env rollout) so they run in < 1 second.
Any coordinate error in :func:`default_rocket_layout` is caught
here, before the integration tests burn budget.
"""

from __future__ import annotations

import pytest

from baselines.rocket.scripted.factory_layout import (
    CoalSnake,
    FactoryLayout,
    NodeFactory,
    Placement,
    default_rocket_layout,
    validate_layout,
)
from factoriax.benchmarks.rocket import build_rocket_level
from factoriax.constants import Direction, ItemType, MachineType


def _level():
    return build_rocket_level()


def test_default_layout_passes_validation() -> None:
    layout = default_rocket_layout()
    errors = validate_layout(layout, _level())
    assert errors == [], "\n".join(errors)


def test_layout_has_four_node_factories() -> None:
    layout = default_rocket_layout()
    ores = {node.ore for node in layout.nodes}
    assert ores == {
        ItemType.IRON_ORE,
        ItemType.COPPER_ORE,
        ItemType.TIN_ORE,
        ItemType.SILICON,
    }


def test_coal_snake_has_four_miners_and_one_pallet() -> None:
    layout = default_rocket_layout()
    assert len(layout.coal.miners) == 4
    assert layout.coal.pallet.machine == MachineType.PALLET
    for miner in layout.coal.miners:
        assert miner.machine == MachineType.MINER


def test_central_bank_has_two_furnaces_and_two_assemblers() -> None:
    layout = default_rocket_layout()
    machines = [p.machine for p in layout.central_bank]
    assert machines.count(MachineType.FURNACE) == 2
    assert machines.count(MachineType.ASSEMBLER) == 2


def test_every_node_miner_pushes_into_its_pallet() -> None:
    """The whole point of explicit facings — guarded at design time."""
    layout = default_rocket_layout()
    for node in layout.nodes:
        assert node.miner.forward_tile() == node.pallet.tile, (
            f"{node.ore.name} miner @ {node.miner.tile} facing "
            f"{Direction(node.miner.facing).name} forward "
            f"{node.miner.forward_tile()} != pallet {node.pallet.tile}"
        )


def test_coal_snake_chains_correctly() -> None:
    layout = default_rocket_layout()
    for i, miner in enumerate(layout.coal.miners):
        next_tile = (
            layout.coal.miners[i + 1].tile
            if i + 1 < len(layout.coal.miners)
            else layout.coal.pallet.tile
        )
        assert miner.forward_tile() == next_tile


# ---------------------------------------------------------------------------
# Negative tests — validator catches common mistakes
# ---------------------------------------------------------------------------


def test_validator_catches_miner_not_facing_pallet() -> None:
    bad = FactoryLayout(
        nodes=(
            NodeFactory(
                ore=ItemType.IRON_ORE,
                miner=Placement(
                    MachineType.MINER,
                    (8, 9),
                    int(Direction.UP),  # pushes to (8, 8), not pallet
                ),
                pallet=Placement(
                    MachineType.PALLET,
                    (8, 10),
                    int(Direction.DOWN),
                ),
            ),
        ),
        coal=default_rocket_layout().coal,
        central_bank=(),
    )
    errors = validate_layout(bad, _level())
    assert any("pushes to" in e for e in errors), errors


def test_validator_catches_duplicate_tile_claims() -> None:
    good = default_rocket_layout()
    p1 = good.nodes[0].miner
    # Second node's miner reuses the first's tile
    bad_node = NodeFactory(
        ore=ItemType.COPPER_ORE,
        miner=Placement(MachineType.MINER, p1.tile, int(Direction.DOWN)),
        pallet=Placement(
            MachineType.PALLET, (p1.tile[0], p1.tile[1] + 1), int(Direction.DOWN)
        ),
    )
    bad = FactoryLayout(
        nodes=(good.nodes[0], bad_node),
        coal=good.coal,
        central_bank=(),
    )
    errors = validate_layout(bad, _level())
    assert any("already claimed" in e for e in errors), errors


def test_validator_catches_out_of_bounds() -> None:
    bad = FactoryLayout(
        nodes=(
            NodeFactory(
                ore=ItemType.IRON_ORE,
                miner=Placement(MachineType.MINER, (8, 9), int(Direction.DOWN)),
                pallet=Placement(
                    MachineType.PALLET,
                    (8, 10),
                    int(Direction.DOWN),
                ),
            ),
        ),
        coal=CoalSnake(
            miners=(Placement(MachineType.MINER, (-5, 23), int(Direction.DOWN)),),
            pallet=Placement(MachineType.PALLET, (10, 23), int(Direction.LEFT)),
        ),
        central_bank=(),
    )
    errors = validate_layout(bad, _level())
    assert any("out of bounds" in e for e in errors), errors


def test_all_placements_flat_order() -> None:
    layout = default_rocket_layout()
    flat = layout.all_placements()
    # Node pallets come before their miners (so miner's push target
    # already exists).
    for node in layout.nodes:
        pi = flat.index(node.pallet)
        mi = flat.index(node.miner)
        assert pi < mi, f"{node.ore.name}: pallet must place before miner"
    # Coal snake: pallet first, then miners downstream → upstream.
    coal_pallet_i = flat.index(layout.coal.pallet)
    coal_miner_indices = [flat.index(m) for m in layout.coal.miners]
    assert all(coal_pallet_i < i for i in coal_miner_indices)
    # Coal miners themselves in reversed order (M4 before M1).
    assert coal_miner_indices == sorted(coal_miner_indices, reverse=True)


@pytest.mark.parametrize(
    "direction",
    [Direction.LEFT, Direction.RIGHT, Direction.UP, Direction.DOWN],
)
def test_placement_stand_tile_is_one_step_opposite(direction: Direction) -> None:
    p = Placement(MachineType.MINER, (10, 10), int(direction))
    sx, sy = p.stand_tile()
    # Sum of stand + unit(direction) should equal tile.
    from baselines.rocket.scripted.factory_layout import _DIR_DX_DY

    dx, dy = _DIR_DX_DY[int(direction)]
    assert (sx + dx, sy + dy) == p.tile
