"""Design-time factory layout for the rocket benchmark.

Instead of hard-coding tile coordinates inside the goal list, build
a :class:`FactoryLayout` dataclass up-front and run it through
:func:`validate_layout` before any env rollout. Validation catches
the easy mistakes (machine on water, miner doesn't face its pallet,
coal snake broken) so the integration test doesn't have to.

The layout describes:

- **Per-ore-node factories** — a miner + pallet pair for each of
  iron, copper, tin, silicon. Miner is placed on the ore patch with
  a facing chosen so its per-tick push lands in the adjacent pallet.
- **Coal snake** — a chain of miners that funnel into a pallet on
  dirt east of the patch.
- **Central bank** — extra furnaces + assemblers placed near the
  spawn for pipelined bulk production.

All tile coordinates are absolute on the 32×32 rocket level. The
``validate_layout`` function rejects:

- any placement tile in SOLID (water / out-of-bounds)
- any placement tile already occupied by a pre-placed machine
- two placements targeting the same tile
- a miner whose forward tile is not the tile of its paired pallet
- a coal-snake miner whose forward tile isn't the next miner or
  the snake's terminal pallet
- any stand tile (``tile - unit(direction)``) outside the map or on
  a solid block
"""

from __future__ import annotations

from dataclasses import dataclass

from factoriax.constants import Direction, ItemType, MachineType
from factoriax.levels import Level

# Axis deltas per :class:`Direction`. Mirrors
# ``factoriax.constants.DIRECTIONS`` without pulling JAX.
_DIR_DX_DY: dict[int, tuple[int, int]] = {
    int(Direction.LEFT): (-1, 0),
    int(Direction.RIGHT): (1, 0),
    int(Direction.UP): (0, -1),
    int(Direction.DOWN): (0, 1),
}

# Solid block types players / machines can't occupy (mirrors
# ``factoriax.constants.SOLID_BLOCKS`` + OOB marker).
_SOLID_BLOCKS: frozenset[int] = frozenset(
    {int(v) for v in (2,)}  # updated below after import
)


def _solid_set() -> frozenset[int]:
    """Defer the import of BlockType so this module stays light."""
    from factoriax.constants import BlockType

    return frozenset(
        {
            int(BlockType.WATER),
            int(BlockType.OUT_OF_BOUNDS),
            int(BlockType.INVALID),
        }
    )


@dataclass(frozen=True)
class Placement:
    """A single (machine_type, tile, facing) tuple."""

    machine: MachineType
    tile: tuple[int, int]
    facing: int  # Direction

    def stand_tile(self) -> tuple[int, int]:
        """Tile the player must stand on to place this correctly.

        The engine sets ``ent_direction`` to the player's facing at
        placement time, so to end up with ``facing`` we stand at the
        tile one step opposite of that direction.
        """
        dx, dy = _DIR_DX_DY[self.facing]
        return (self.tile[0] - dx, self.tile[1] - dy)

    def forward_tile(self) -> tuple[int, int]:
        """Tile the machine pushes into (for miners) or faces."""
        dx, dy = _DIR_DX_DY[self.facing]
        return (self.tile[0] + dx, self.tile[1] + dy)


@dataclass(frozen=True)
class NodeFactory:
    """Miner + pallet pair for one ore type."""

    ore: ItemType
    miner: Placement
    pallet: Placement


@dataclass(frozen=True)
class CoalSnake:
    """Chain of coal miners feeding a terminal pallet."""

    miners: tuple[Placement, ...]  # ordered upstream → downstream
    pallet: Placement


@dataclass(frozen=True)
class FactoryLayout:
    """The full design-time plan for a rocket-level factory."""

    nodes: tuple[NodeFactory, ...]
    coal: CoalSnake
    central_bank: tuple[Placement, ...]

    def all_placements(self) -> list[Placement]:
        """Flat list in the order they should be executed.

        Order is careful: the pallet in each node factory lands
        first so the miner's later placement finds a valid push
        target; the coal snake is placed terminal-first (pallet →
        M4 → M3 → M2 → M1) so each stand tile stays walkable.
        Central bank last.
        """
        out: list[Placement] = []
        for node in self.nodes:
            out.append(node.pallet)
            out.append(node.miner)
        # Snake: pallet first, then miners downstream → upstream
        # (reversed) so each placement's stand tile is still free.
        out.append(self.coal.pallet)
        out.extend(reversed(self.coal.miners))
        out.extend(self.central_bank)
        return out


# ---------------------------------------------------------------------------
# Default layout for the rocket benchmark level
# ---------------------------------------------------------------------------


def _node_strip(ore: ItemType, patch_x: int, patch_y_bottom: int) -> NodeFactory:
    """Miner at south-edge of patch, pallet one tile further south.

    Miner faces DOWN so it pushes into the pallet. Pallet faces
    DOWN for consistency (facing has no effect for pallets).
    """
    miner_tile = (patch_x, patch_y_bottom)
    pallet_tile = (patch_x, patch_y_bottom + 1)
    return NodeFactory(
        ore=ore,
        miner=Placement(MachineType.MINER, miner_tile, int(Direction.DOWN)),
        pallet=Placement(MachineType.PALLET, pallet_tile, int(Direction.DOWN)),
    )


def default_rocket_layout() -> FactoryLayout:
    """Canonical factory layout for :func:`build_rocket_level`.

    Hard-coded coordinates map to the patch positions in
    :mod:`factoriax.benchmarks.rocket`:

    - iron (7-9, 7-9)
    - copper (22-24, 7-9)
    - tin (22-24, 22-24)
    - silicon (14-16, 3-5)
    - coal (7-9, 22-24)
    """
    nodes = (
        _node_strip(ItemType.IRON_ORE, patch_x=8, patch_y_bottom=9),
        _node_strip(ItemType.COPPER_ORE, patch_x=23, patch_y_bottom=9),
        _node_strip(ItemType.TIN_ORE, patch_x=23, patch_y_bottom=24),
        _node_strip(ItemType.SILICON, patch_x=15, patch_y_bottom=5),
    )

    # Coal snake: M1 (7,22) DOWN → M2 (7,23) RIGHT → M3 (8,23)
    # RIGHT → M4 (9,23) RIGHT → Pallet (10,23).
    snake_miners = (
        Placement(MachineType.MINER, (7, 22), int(Direction.DOWN)),
        Placement(MachineType.MINER, (7, 23), int(Direction.RIGHT)),
        Placement(MachineType.MINER, (8, 23), int(Direction.RIGHT)),
        Placement(MachineType.MINER, (9, 23), int(Direction.RIGHT)),
    )
    coal = CoalSnake(
        miners=snake_miners,
        pallet=Placement(MachineType.PALLET, (10, 23), int(Direction.LEFT)),
    )

    # Central bank: 2 furnaces and 2 assemblers somewhere near spawn.
    # These placements use :func:`free_tile_near_player` at runtime
    # rather than fixed tiles (the spawn area is always free), so
    # the layout just declares machine types — actual tiles chosen
    # by the agent goal.
    central_bank = (
        Placement(MachineType.FURNACE, (-1, -1), int(Direction.DOWN)),
        Placement(MachineType.FURNACE, (-1, -1), int(Direction.DOWN)),
        Placement(MachineType.ASSEMBLER, (-1, -1), int(Direction.DOWN)),
        Placement(MachineType.ASSEMBLER, (-1, -1), int(Direction.DOWN)),
    )

    return FactoryLayout(nodes=nodes, coal=coal, central_bank=central_bank)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_layout(layout: FactoryLayout, level: Level) -> list[str]:
    """Return a list of error strings; empty if the layout is sound.

    Checks are cheap — this is a design-time gate so integration
    tests don't burn 3 minutes to discover an off-by-one coordinate.
    """
    import numpy as np

    errors: list[str] = []
    solid = _solid_set()
    block_map = np.asarray(level.block_map)
    machine_types = np.asarray(level.machine_types)
    h, w = block_map.shape

    claimed: dict[tuple[int, int], MachineType] = {}

    def _check_placement(p: Placement, label: str) -> None:
        x, y = p.tile
        if p.tile == (-1, -1):
            return  # deferred / runtime-resolved; skip.
        if not (0 <= x < w and 0 <= y < h):
            errors.append(f"{label}: tile {p.tile} out of bounds")
            return
        if int(block_map[y, x]) in solid:
            errors.append(f"{label}: tile {p.tile} is on a solid block")
        if int(machine_types[y, x]) != int(MachineType.NONE):
            errors.append(f"{label}: tile {p.tile} already has a pre-placed machine")
        existing = claimed.get(p.tile)
        if existing is not None:
            errors.append(
                f"{label}: tile {p.tile} already claimed by "
                f"{MachineType(existing).name}"
            )
        claimed[p.tile] = p.machine

        stand = p.stand_tile()
        sx, sy = stand
        if not (0 <= sx < w and 0 <= sy < h):
            errors.append(f"{label}: stand tile {stand} out of bounds")
            return
        if int(block_map[sy, sx]) in solid:
            errors.append(f"{label}: stand tile {stand} is solid")

    # Node factories: miner must push into its paired pallet.
    for node in layout.nodes:
        _check_placement(node.pallet, f"{node.ore.name} pallet")
        _check_placement(node.miner, f"{node.ore.name} miner")
        if node.miner.machine != MachineType.MINER:
            errors.append(f"{node.ore.name} node: miner slot isn't a MINER")
        if node.pallet.machine != MachineType.PALLET:
            errors.append(f"{node.ore.name} node: pallet slot isn't a PALLET")
        if node.miner.forward_tile() != node.pallet.tile:
            errors.append(
                f"{node.ore.name} node: miner at {node.miner.tile} facing "
                f"{Direction(node.miner.facing).name} pushes to "
                f"{node.miner.forward_tile()}, not pallet tile {node.pallet.tile}"
            )

    # Coal snake: each miner's forward tile is the next miner or pallet.
    _check_placement(layout.coal.pallet, "coal pallet")
    for i, miner in enumerate(layout.coal.miners):
        _check_placement(miner, f"coal M{i + 1}")
        if miner.machine != MachineType.MINER:
            errors.append(f"coal M{i + 1}: not a MINER")
        expected_next = (
            layout.coal.miners[i + 1].tile
            if i + 1 < len(layout.coal.miners)
            else layout.coal.pallet.tile
        )
        if miner.forward_tile() != expected_next:
            errors.append(
                f"coal M{i + 1}: forward tile {miner.forward_tile()} != "
                f"expected next {expected_next}"
            )

    # Central bank: only type checks for deferred placements.
    for i, p in enumerate(layout.central_bank):
        if p.machine not in (MachineType.FURNACE, MachineType.ASSEMBLER):
            errors.append(f"central bank [{i}]: {p.machine} isn't furnace/assembler")

    return errors
