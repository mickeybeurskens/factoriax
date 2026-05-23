"""Plan the factory layout from the recipe DAG and runtime ore patches.

Two DAG passes over the runtime ``recipe_table`` decide what to build;
runtime reads of ``state.map`` decide where to put it.

Pass 1: walk backwards from the target item, collecting every recipe
the automated chain runs. Each such recipe gets one assembler.

Pass 2: for every ore type the recipes (automated *and* hand-crafted
factory machines) touch, allocate one manual miner plus one factory
miner per automated consumer.

Geometric placement is greedy: miners onto patch tiles, assemblers
onto free tiles near their consuming miners' centroid, arms onto a
free side of each assembler, belts routed by BFS over free tiles.

The output of :func:`plan_factory` is a :class:`FactoryLayout`. Phases
read the layout's per-role lists to know what to craft, where to
walk, and what predicate flips them done.
"""

from __future__ import annotations

import dataclasses
from collections import deque

from baselines.easy_rocket.scripted.state_reader import (
    OrePatch,
    find_patches,
    tile_free,
)
from factoriax.constants import BlockType, Direction, ItemType
from factoriax.recipes import RecipeTable
from factoriax.state import EnvState

# Map ore ItemType -> ore BlockType so we can find the corresponding
# patch on the map.
ORE_ITEM_TO_BLOCK: dict[int, int] = {
    int(ItemType.IRON_ORE): int(BlockType.IRON),
    int(ItemType.COPPER_ORE): int(BlockType.COPPER),
    int(ItemType.TIN_ORE): int(BlockType.TIN),
    int(ItemType.SILICON): int(BlockType.SILICON),
    int(ItemType.COAL): int(BlockType.COAL),
    int(ItemType.LIMESTONE): int(BlockType.LIMESTONE),
}

# Items that are crafted by hand to build the factory itself.
# Their recipes contribute ore demand to the manual miners but never
# become an automated-chain assembler.
HAND_CRAFTED_FACTORY_ITEMS: frozenset[int] = frozenset(
    {
        int(ItemType.MINER),
        int(ItemType.ASSEMBLER),
        int(ItemType.CONVEYOR_BELT),
        int(ItemType.ARM),
    }
)

# 4-direction offsets in (dx, dy).
_DIR_OFFSETS: dict[int, tuple[int, int]] = {
    int(Direction.LEFT): (-1, 0),
    int(Direction.RIGHT): (1, 0),
    int(Direction.UP): (0, -1),
    int(Direction.DOWN): (0, 1),
}


@dataclasses.dataclass(frozen=True)
class MinerPlan:
    """One miner placement in the factory layout."""

    pos: tuple[int, int]
    facing: int
    role: str  # "manual" or "factory"
    source_ore: int  # ItemType of the ore mined
    # ItemType the consumer assembler produces; 0 for manual miners.
    consumer_recipe_output: int


@dataclasses.dataclass(frozen=True)
class AssemblerPlan:
    """One automated assembler placement."""

    pos: tuple[int, int]
    recipe_output: int  # ItemType produced by this assembler


@dataclasses.dataclass(frozen=True)
class ArmPlan:
    """One arm draining an assembler's output to a downstream belt."""

    pos: tuple[int, int]
    facing: int  # direction items flow OUT of the arm
    source_assembler_idx: int  # index into FactoryLayout.assemblers


@dataclasses.dataclass(frozen=True)
class BeltPlan:
    """One conveyor belt tile."""

    pos: tuple[int, int]
    facing: int  # direction items flow on this tile


@dataclasses.dataclass(frozen=True)
class FactoryLayout:
    """Full factory blueprint produced by :func:`plan_factory`."""

    miners: tuple[MinerPlan, ...]
    assemblers: tuple[AssemblerPlan, ...]
    arms: tuple[ArmPlan, ...]
    belts: tuple[BeltPlan, ...]
    rocket_tile: tuple[int, int]
    valid: bool
    error: str | None


# ---------------------------------------------------------------------------
# Pass 1 — recipe DAG analysis
# ---------------------------------------------------------------------------


def _recipe_for_output(table: RecipeTable, item: int) -> int | None:
    """Return the recipe index that produces ``item``, or ``None``."""
    import numpy as np  # noqa: PLC0415

    out_to_recipe = np.asarray(table.output_to_recipe)
    if item < 0 or item >= out_to_recipe.size:
        return None
    idx = int(out_to_recipe[item])
    return idx if idx >= 0 else None


def _recipe_inputs(table: RecipeTable, recipe_idx: int) -> list[int]:
    """Return the non-empty input ItemTypes for a recipe."""
    import numpy as np  # noqa: PLC0415

    items = np.asarray(table.input_items[recipe_idx])
    counts = np.asarray(table.input_counts[recipe_idx])
    return [int(items[i]) for i in range(items.size) if counts[i] > 0]


def _walk_dag(
    table: RecipeTable, target: int, hand_crafted: frozenset[int]
) -> tuple[list[int], set[int]]:
    """Walk the recipe DAG backwards from ``target``.

    Returns ``(automated_recipe_outputs, all_ore_items)``:

    - ``automated_recipe_outputs`` — ItemTypes whose recipe must run
      in an automated assembler. Topologically ordered so producers
      come before consumers. The target is the last entry.
    - ``all_ore_items`` — every raw ore ItemType the recipes
      (automated *and* hand-crafted factory machines) depend on.
    """
    automated_outputs: list[int] = []
    visited_automated: set[int] = set()
    ore_items: set[int] = set()

    def visit(item: int, hand_root: bool) -> None:
        if item in ORE_ITEM_TO_BLOCK:
            ore_items.add(item)
            return
        recipe_idx = _recipe_for_output(table, item)
        if recipe_idx is None:
            return
        for inp in _recipe_inputs(table, recipe_idx):
            visit(inp, hand_root)
        # If this item is hand-crafted to build the factory, do not add
        # an assembler for it; its ore demand was captured above. The
        # target item and every intermediate in the target's chain go
        # in the automated set.
        if hand_root or item in hand_crafted:
            return
        if item not in visited_automated:
            visited_automated.add(item)
            automated_outputs.append(item)

    # Walk the target's chain → automated.
    visit(target, hand_root=False)
    # Walk each hand-crafted factory item's chain → ores only.
    for item in hand_crafted:
        visit(item, hand_root=True)

    return automated_outputs, ore_items


# ---------------------------------------------------------------------------
# Pass 2 — miner allocation
# ---------------------------------------------------------------------------


def _consumers_per_ore(
    table: RecipeTable, automated_outputs: list[int]
) -> dict[int, list[int]]:
    """For each ore type, list the automated assembler outputs that
    consume it.

    The mapping drives the factory-miner count: one factory miner per
    (ore, automated-consumer) pair, per the spec's rule.
    """
    consumers: dict[int, list[int]] = {}
    for output in automated_outputs:
        recipe_idx = _recipe_for_output(table, output)
        if recipe_idx is None:
            continue
        for inp in _recipe_inputs(table, recipe_idx):
            if inp in ORE_ITEM_TO_BLOCK:
                consumers.setdefault(inp, []).append(output)
    return consumers


# ---------------------------------------------------------------------------
# Geometric placement
# ---------------------------------------------------------------------------


def _patch_by_ore(patches: list[OrePatch]) -> dict[int, OrePatch]:
    """Map ore ItemType → patch, via the block-type bridge."""
    block_to_item = {block: item for item, block in ORE_ITEM_TO_BLOCK.items()}
    return {
        block_to_item[p.ore_block]: p for p in patches if p.ore_block in block_to_item
    }


def _free_tile_near(
    state: EnvState,
    cx: int,
    cy: int,
    reserved: set[tuple[int, int]],
) -> tuple[int, int] | None:
    """Return the closest free tile to ``(cx, cy)`` that is not reserved.

    Search radius grows outward; first hit wins.
    """
    h, w = state.map.shape
    for r in range(max(h, w)):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if max(abs(dx), abs(dy)) != r:
                    continue
                x, y = cx + dx, cy + dy
                if (x, y) in reserved:
                    continue
                if tile_free(state, x, y):
                    return (x, y)
    return None


def _free_tiles_near(
    state: EnvState,
    cx: int,
    cy: int,
    reserved: set[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]]:
    """Return up to ``count`` free tiles nearest to ``(cx, cy)``.

    Used by the placement backtracking loop: each automated assembler
    gets a handful of candidate positions ordered by distance to its
    desired centroid; the planner tries them in order until one leads
    to a successful complete layout.
    """
    h, w = state.map.shape
    found: list[tuple[int, int]] = []
    for r in range(max(h, w)):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if max(abs(dx), abs(dy)) != r:
                    continue
                x, y = cx + dx, cy + dy
                if (x, y) in reserved:
                    continue
                if tile_free(state, x, y):
                    found.append((x, y))
                    if len(found) >= count:
                        return found
    return found


def _bfs_belt_route(
    state: EnvState,
    start: tuple[int, int],
    end: tuple[int, int],
    reserved: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    """Find a BFS path of free tiles from ``start`` to ``end``.

    Tiles in ``reserved`` are off-limits except for ``start`` and
    ``end`` themselves (the endpoints are typically machine positions
    occupied by miners / assemblers — the path is the belt tiles
    *between* them). Returns the intermediate tiles only (excludes
    both endpoints). Returns ``None`` if no route exists.
    """
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    frontier: deque[tuple[int, int]] = deque([start])
    found = False
    while frontier:
        cur = frontier.popleft()
        if cur == end:
            found = True
            break
        for dx, dy in _DIR_OFFSETS.values():
            nx, ny = cur[0] + dx, cur[1] + dy
            nxt = (nx, ny)
            if nxt in came_from:
                continue
            if nxt == end:
                came_from[nxt] = cur
                frontier.append(nxt)
                continue
            if nxt in reserved:
                continue
            if not tile_free(state, nx, ny):
                continue
            came_from[nxt] = cur
            frontier.append(nxt)
    if not found:
        return None
    path: list[tuple[int, int]] = []
    cur = end
    while came_from[cur] is not None:
        prev = came_from[cur]
        assert prev is not None
        path.append(cur)
        cur = prev
    path.reverse()
    # Drop the endpoint; intermediates are the belt tiles.
    return path[:-1]


def _describe_blocked_tile(
    state: EnvState,
    pos: tuple[int, int],
    reserved: set[tuple[int, int]],
    reserved_by: dict[tuple[int, int], str],
) -> str:
    """Return a short human-readable reason ``pos`` is unwalkable.

    Used by :func:`_bfs_failure_message` to annotate why a BFS gave up.
    """
    x, y = pos
    h, w = state.map.shape
    if not (0 <= x < w and 0 <= y < h):
        return "out-of-bounds"
    if pos in reserved:
        return f"reserved by {reserved_by.get(pos, 'unknown')}"
    if not tile_free(state, x, y):
        block_val = int(state.map[y, x])
        try:
            return f"block={BlockType(block_val).name}"
        except ValueError:
            return f"block={block_val}"
    return "free (should be reachable)"


def _bfs_failure_message(
    state: EnvState,
    start: tuple[int, int],
    end: tuple[int, int],
    reserved: set[tuple[int, int]],
    reserved_by: dict[tuple[int, int], str],
    label: str,
) -> str:
    """Build a multi-line diagnostic for a failed BFS route.

    Lists every neighbour of both endpoints with its blocking reason,
    plus an ASCII rendering of the map showing free / reserved /
    blocked tiles so the failure context is self-contained.
    """
    lines = [f"no belt route for {label} from {start} to {end}"]
    for label2, anchor in (("start", start), ("end", end)):
        lines.append(f"  neighbours of {label2} {anchor}:")
        for d, (dx, dy) in _DIR_OFFSETS.items():
            nb = (anchor[0] + dx, anchor[1] + dy)
            reason = _describe_blocked_tile(state, nb, reserved, reserved_by)
            lines.append(f"    {Direction(d).name:5s} {nb}: {reason}")
    h, w = state.map.shape
    lines.append("  map (S=start E=end R=reserved #=blocked .=free):")
    for y in range(h):
        row = []
        for x in range(w):
            pos = (x, y)
            if pos == start:
                ch = "S"
            elif pos == end:
                ch = "E"
            elif pos in reserved:
                ch = "R"
            elif not tile_free(state, x, y):
                ch = "#"
            else:
                ch = "."
            row.append(ch)
        lines.append("    " + " ".join(row))
    return "\n".join(lines)


def _direction_between(src: tuple[int, int], dst: tuple[int, int]) -> int:
    """Return the ``Direction`` from ``src`` to ``dst`` (orthogonal)."""
    dx = dst[0] - src[0]
    dy = dst[1] - src[1]
    if dx == 1:
        return int(Direction.RIGHT)
    if dx == -1:
        return int(Direction.LEFT)
    if dy == 1:
        return int(Direction.DOWN)
    if dy == -1:
        return int(Direction.UP)
    raise ValueError(f"non-adjacent tiles: {src} -> {dst}")


def _belts_along(path: list[tuple[int, int]], dest: tuple[int, int]) -> list[BeltPlan]:
    """Turn a sequence of belt tiles into ``BeltPlan``s with facings.

    Each belt's facing points at the next belt in the chain, or at
    ``dest`` for the last belt.
    """
    plans: list[BeltPlan] = []
    for i, pos in enumerate(path):
        nxt = path[i + 1] if i + 1 < len(path) else dest
        plans.append(BeltPlan(pos=pos, facing=_direction_between(pos, nxt)))
    return plans


# ---------------------------------------------------------------------------
# Top-level planner
# ---------------------------------------------------------------------------


def _invalid(error: str) -> FactoryLayout:
    """Build an empty :class:`FactoryLayout` flagged invalid."""
    return FactoryLayout(
        miners=(),
        assemblers=(),
        arms=(),
        belts=(),
        rocket_tile=(0, 0),
        valid=False,
        error=error,
    )


def _try_layout(
    state: EnvState,
    target: int,
    automated_outputs: list[int],
    ore_items: set[int],
    consumers_per_ore: dict[int, list[int]],
    patch_by_ore: dict[int, OrePatch],
    miner_jobs: list[tuple[int, str, int]],
    base_manual_plans: list[MinerPlan],
    asm_positions: dict[int, tuple[int, int]],
) -> FactoryLayout:
    """Complete the layout given a specific set of assembler positions.

    Returns a :class:`FactoryLayout` with ``valid=True`` on success or
    one of the structured ``valid=False`` errors used by
    :func:`plan_factory` to drive backtracking.
    """
    reserved: set[tuple[int, int]] = set()
    reserved_by: dict[tuple[int, int], str] = {}

    def _reserve(pos: tuple[int, int], label: str) -> None:
        reserved.add(pos)
        reserved_by[pos] = label

    miner_plans: list[MinerPlan] = list(base_manual_plans)
    used_tiles_by_ore: dict[int, set[tuple[int, int]]] = {
        ore: set() for ore in ore_items
    }
    for m in base_manual_plans:
        used_tiles_by_ore[m.source_ore].add(m.pos)
        _reserve(m.pos, f"manual miner ({ItemType(m.source_ore).name})")

    # Reserve the caller-supplied assembler positions.
    assembler_plans: list[AssemblerPlan] = []
    for output in automated_outputs:
        pos = asm_positions[output]
        if pos in reserved or not tile_free(state, pos[0], pos[1]):
            return _invalid(
                f"assembler position {pos} for {ItemType(output).name} "
                "is reserved or not walkable"
            )
        assembler_plans.append(AssemblerPlan(pos=pos, recipe_output=output))
        _reserve(pos, f"assembler ({ItemType(output).name})")

    asm_idx_by_output: dict[int, int] = {
        a.recipe_output: i for i, a in enumerate(assembler_plans)
    }

    # ---- Reserve one arm slot per non-target assembler ----
    # The arm must end up on a side that has both (a) a free adjacent
    # tile for the downstream belt and (b) was not consumed by an
    # input-belt route. Pre-reserve the slot here so belt routing
    # avoids it; the actual ArmPlan is materialised after belts
    # route, using these reservations.
    arm_slots: dict[int, tuple[tuple[int, int], int]] = {}  # asm_idx -> (pos, facing)
    for i, asm in enumerate(assembler_plans):
        if asm.recipe_output == target:
            continue
        chosen: tuple[tuple[int, int], int] | None = None
        for d, (dx, dy) in _DIR_OFFSETS.items():
            cand = (asm.pos[0] + dx, asm.pos[1] + dy)
            if cand in reserved or not tile_free(state, cand[0], cand[1]):
                continue
            # The arm's downstream tile (where its output goes) must
            # also be free — otherwise the output belt can't start.
            downstream = (cand[0] + dx, cand[1] + dy)
            if downstream in reserved or not tile_free(
                state, downstream[0], downstream[1]
            ):
                continue
            chosen = (cand, d)
            break
        if chosen is None:
            return _invalid(
                f"no free tile adjacent to {ItemType(asm.recipe_output).name} "
                f"assembler at {asm.pos} for an arm (need a side with a "
                "free downstream tile too)"
            )
        arm_slots[i] = chosen
        _reserve(chosen[0], f"reserved arm slot ({ItemType(asm.recipe_output).name})")

    # ---- Place factory miners + route their input belts (interleaved) ----
    # For each factory job, try patch tiles in order of distance to
    # the consuming assembler. Commit the first (tile, route) pair
    # that BFS can find. This handles cramped layouts where a fixed
    # tile choice would leave the miner with no route out.
    belt_plans: list[BeltPlan] = []
    for ore, role, consumer in miner_jobs:
        if role != "factory":
            continue
        patch = patch_by_ore[ore]
        asm_idx = asm_idx_by_output.get(consumer)
        if asm_idx is None:
            return _invalid(
                f"factory miner for {ItemType(ore).name} names consumer "
                f"{ItemType(consumer).name} but no assembler plan exists"
            )
        asm_pos = assembler_plans[asm_idx].pos
        candidates = [t for t in patch.tiles if t not in used_tiles_by_ore[ore]]
        candidates.sort(key=lambda t: abs(t[0] - asm_pos[0]) + abs(t[1] - asm_pos[1]))
        chosen_pos: tuple[int, int] | None = None
        chosen_path: list[tuple[int, int]] | None = None
        last_failure: str | None = None
        for cand in candidates:
            # Tentatively reserve the candidate so BFS doesn't route
            # through the miner's own tile, then route.
            reserved.add(cand)
            path = _bfs_belt_route(state, cand, asm_pos, reserved)
            if path is None:
                last_failure = _bfs_failure_message(
                    state,
                    cand,
                    asm_pos,
                    reserved,
                    reserved_by,
                    label=f"{ItemType(ore).name} input belt -> "
                    f"{ItemType(consumer).name} assembler",
                )
                reserved.discard(cand)
                continue
            # Commit.
            chosen_pos = cand
            chosen_path = path
            # Keep the reservation; also label it.
            reserved_by[cand] = f"factory miner ({ItemType(ore).name})"
            break
        if chosen_pos is None or chosen_path is None:
            return _invalid(
                f"could not place {ItemType(ore).name} factory miner feeding "
                f"{ItemType(consumer).name}; last attempt:\n{last_failure}"
            )
        used_tiles_by_ore[ore].add(chosen_pos)
        miner_plans.append(
            MinerPlan(
                pos=chosen_pos,
                facing=int(Direction.RIGHT),
                role="factory",
                source_ore=ore,
                consumer_recipe_output=consumer,
            )
        )
        new_belts = _belts_along(chosen_path, dest=asm_pos)
        belt_plans.extend(new_belts)
        for b in new_belts:
            _reserve(
                b.pos,
                f"input belt ({ItemType(ore).name} -> {ItemType(consumer).name})",
            )

    # ---- Materialise arm plans from the pre-reserved slots ----
    arm_plans: list[ArmPlan] = []
    for i, asm in enumerate(assembler_plans):
        if asm.recipe_output == target:
            continue
        arm_pos, arm_facing = arm_slots[i]
        arm_plans.append(
            ArmPlan(pos=arm_pos, facing=arm_facing, source_assembler_idx=i)
        )
        # The slot was already reserved as "reserved arm slot ..."; the
        # tile stays reserved with the same label, no change needed.

    # ---- Reserve a rocket tile ----
    rocket_asm = next((a for a in assembler_plans if a.recipe_output == target), None)
    if rocket_asm is None:
        return _invalid(f"target {ItemType(target).name} has no assembler plan")
    rocket_tile_opt = _free_tile_near(
        state, rocket_asm.pos[0], rocket_asm.pos[1], reserved
    )
    if rocket_tile_opt is None:
        return _invalid("no free tile near rocket assembler for rocket placement")
    rocket_tile: tuple[int, int] = rocket_tile_opt
    _reserve(rocket_tile, "rocket placement tile")

    # ---- Route output belts (arm-downstream -> target assembler) ----
    for arm in arm_plans:
        src_asm = assembler_plans[arm.source_assembler_idx]
        # Items flow OUT of the arm in arm.facing direction, so the
        # first belt sits at arm.pos + facing offset.
        dx, dy = _DIR_OFFSETS[arm.facing]
        start_pos = (arm.pos[0] + dx, arm.pos[1] + dy)
        dst_asm = assembler_plans[asm_idx_by_output[target]]
        path = _bfs_belt_route(state, start_pos, dst_asm.pos, reserved)
        if path is None:
            # Fall back to starting from the arm's own tile in case
            # the forward step is blocked.
            path = _bfs_belt_route(state, arm.pos, dst_asm.pos, reserved)
        if path is None:
            label = (
                f"{ItemType(src_asm.recipe_output).name} output belt -> "
                f"{ItemType(target).name} assembler"
            )
            return _invalid(
                _bfs_failure_message(
                    state, start_pos, dst_asm.pos, reserved, reserved_by, label
                )
            )
        new_belts = _belts_along(path, dest=dst_asm.pos)
        belt_plans.extend(new_belts)
        for b in new_belts:
            _reserve(
                b.pos,
                f"output belt ({ItemType(src_asm.recipe_output).name})",
            )

    return FactoryLayout(
        miners=tuple(miner_plans),
        assemblers=tuple(assembler_plans),
        arms=tuple(arm_plans),
        belts=tuple(belt_plans),
        rocket_tile=rocket_tile,
        valid=True,
        error=None,
    )


def plan_factory(
    state: EnvState,
    recipe_table: RecipeTable,
    target: int = int(ItemType.ROCKET),
) -> FactoryLayout:
    """Produce a full :class:`FactoryLayout` for the given target.

    Two-pass DAG analysis decides the bill of materials, then a
    backtracking search over assembler placement combinations finds a
    geometry that lets every belt route. Pure-Python; no env actions
    emitted. Returns ``valid=False`` with a structured ``error`` on
    every expected failure so the agent can halt cleanly in Phase 1.
    """
    # ---- Pass 1: DAG walk ----
    automated_outputs, ore_items = _walk_dag(
        recipe_table, target, HAND_CRAFTED_FACTORY_ITEMS
    )
    if not automated_outputs:
        return _invalid(f"no automated recipe chain reaches target {target}")

    consumers_per_ore = _consumers_per_ore(recipe_table, automated_outputs)

    # ---- Pass 2: miner job list ----
    miner_jobs: list[tuple[int, str, int]] = []
    for ore in sorted(ore_items):
        miner_jobs.append((ore, "manual", 0))
        for consumer in consumers_per_ore.get(ore, []):
            miner_jobs.append((ore, "factory", consumer))

    # ---- Read map ----
    patches = find_patches(state)
    patch_by_ore = _patch_by_ore(patches)
    missing = [ore for ore in ore_items if ore not in patch_by_ore]
    if missing:
        return _invalid(
            "ore type(s) missing from map: "
            + ", ".join(ItemType(o).name for o in missing)
        )

    # ---- Place manual miners (same across every backtracking attempt) ----
    base_manual_plans: list[MinerPlan] = []
    manual_used: dict[int, set[tuple[int, int]]] = {ore: set() for ore in ore_items}
    for ore, role, consumer in miner_jobs:
        if role != "manual":
            continue
        patch = patch_by_ore[ore]
        avail = [t for t in patch.tiles if t not in manual_used[ore]]
        if not avail:
            return _invalid(
                f"patch for {ItemType(ore).name} has no free tile for the manual miner"
            )
        pos = avail[0]
        manual_used[ore].add(pos)
        base_manual_plans.append(
            MinerPlan(
                pos=pos,
                facing=int(Direction.RIGHT),
                role="manual",
                source_ore=ore,
                consumer_recipe_output=consumer,
            )
        )

    base_reserved: set[tuple[int, int]] = {m.pos for m in base_manual_plans}

    # ---- Generate candidate assembler positions ----
    # Each automated assembler gets a handful of candidate positions
    # ordered by distance to the centroid of its consuming patches.
    # The product of these lists is searched until one combination
    # yields a complete layout.
    candidates_per_assembler = 16
    min_assembler_separation = 2
    asm_candidates: dict[int, list[tuple[int, int]]] = {}
    for output in automated_outputs:
        consuming_ores = [
            ore for ore, consumers in consumers_per_ore.items() if output in consumers
        ]
        consuming_patches = [patch_by_ore[ore] for ore in consuming_ores]
        if consuming_patches:
            all_tiles = [t for p in consuming_patches for t in p.tiles]
            cx = sum(t[0] for t in all_tiles) // len(all_tiles)
            cy = sum(t[1] for t in all_tiles) // len(all_tiles)
        else:
            h_, w_ = state.map.shape
            cx, cy = w_ // 2, h_ // 2
        asm_candidates[output] = _free_tiles_near(
            state, cx, cy, base_reserved, candidates_per_assembler
        )
        if not asm_candidates[output]:
            return _invalid(
                f"no free tile near ({cx},{cy}) for {ItemType(output).name} assembler"
            )

    # ---- Backtracking search over assembler position combinations ----
    from itertools import product  # noqa: PLC0415

    candidate_lists = [asm_candidates[o] for o in automated_outputs]
    last_error: str | None = None
    attempts = 0
    for combo in product(*candidate_lists):
        # Skip combinations where two assemblers land on the same tile
        # or sit too close together (their input belts would compete).
        if len(set(combo)) != len(combo):
            continue
        too_close = False
        for i in range(len(combo)):
            for j in range(i + 1, len(combo)):
                manhattan = abs(combo[i][0] - combo[j][0]) + abs(
                    combo[i][1] - combo[j][1]
                )
                if manhattan < min_assembler_separation:
                    too_close = True
                    break
            if too_close:
                break
        if too_close:
            continue

        attempts += 1
        asm_positions = dict(zip(automated_outputs, combo, strict=True))
        result = _try_layout(
            state=state,
            target=target,
            automated_outputs=automated_outputs,
            ore_items=ore_items,
            consumers_per_ore=consumers_per_ore,
            patch_by_ore=patch_by_ore,
            miner_jobs=miner_jobs,
            base_manual_plans=base_manual_plans,
            asm_positions=asm_positions,
        )
        if result.valid:
            return result
        last_error = result.error

    return _invalid(
        f"all {attempts} assembler position combinations failed; "
        f"last error:\n{last_error}"
    )
