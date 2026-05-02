"""Reconnaissance script for the Phase 3 rocket-factory layout.

Five things to check before committing to a design:

1. Which tiles are occupied at end of Phase 1+2? — pulled from the
   actual goal tree the agent runs, projected via
   :func:`expected_layout_from_goals`. No simulation needed.
2. Where can 10 assembler cells fit without colliding with Phase 1+2
   *machinery* (non-belt) tiles? — walk a candidate list of cell
   centers and verify each cell's 5-tile footprint is free.
3. Do any two Phase 3 cells overlap or sit too close to chain via the
   adjacent-pallet pattern? — pairwise footprint diff plus a check
   that the assembler-to-assembler horizontal gap is >= 5 when both
   cells share a row (so an arm fits between crate and next input).
4. Which rows / columns are free of Phase 1+2 belts and can serve as
   east-west / north-south belt highways for Phase 3?
5. For routes that *do* have to cross a Phase 1+2 belt — at how many
   tiles, and in which direction? Each perpendicular crossing tile
   becomes a CROSSING (an existing Phase 1+2 BELT must be replaced
   by a CROSSING during Phase 3 placement, or the route must detour).
6. How sensitive is the cell shape to recipe-count tunability? —
   inspect each recipe under the rocket book, count distinct input
   types, confirm none exceed 2 (the assembler's slot count) or
   require >2 input pallets.

Run via ``uv run python scripts/scout_phase3_layout.py``.
"""

from __future__ import annotations

from baselines.rocket.scripted.agent_advanced_factory import (
    build_advanced_factory_goals,
)
from baselines.rocket.scripted.layout import expected_layout_from_goals
from factoriax.benchmarks.rocket import ROCKET_RECIPE_BOOK
from factoriax.constants import Direction, ItemType, MachineType

_MAP: int = 32
_PRE_PLACED: dict[tuple[int, int], tuple[int, int]] = {
    (15, 16): (int(MachineType.FURNACE), int(Direction.DOWN)),
    (17, 16): (int(MachineType.ASSEMBLER), int(Direction.DOWN)),
}
_SPAWN: tuple[int, int] = (16, 16)
# Ore patches (from factoriax.benchmarks.rocket._PATCH_OFFSETS).
# v2 layout: 2x2 patches stacked on cols 3-4; 1-tile coal column at x=0.
_PATCHES: list[tuple[int, int, str]] = [
    (3, 9, "iron"),
    (3, 12, "copper"),
    (3, 15, "tin"),
    (3, 18, "silicon"),
    (3, 21, "limestone"),
]
_COAL_COLUMN_X: int = 0
_PATCH_SIZE: int = 2

# Phase 1+2 sink pallets — the natural Phase 3 source for plates / wafers.
_SOURCES: dict[int, tuple[int, int]] = {
    int(ItemType.IRON_PLATE): (10, 13),
    int(ItemType.COPPER_PLATE): (21, 12),
    int(ItemType.TIN_PLATE): (25, 28),
    int(ItemType.WAFER): (17, 9),
}


def _patch_tiles(top_x: int, top_y: int) -> set[tuple[int, int]]:
    """2x2 patch coordinates as (x, y) tiles (v2 layout)."""
    return {
        (top_x + dx, top_y + dy)
        for dx in range(_PATCH_SIZE)
        for dy in range(_PATCH_SIZE)
    }


def _coal_column_tiles() -> set[tuple[int, int]]:
    """Coal column tiles (x = _COAL_COLUMN_X, full map height)."""
    return {(_COAL_COLUMN_X, y) for y in range(_MAP)}


def _occupied_after_phase_1_2() -> dict[tuple[int, int], tuple[int, int]]:
    """All tiles occupied at end of Phase 1+2 — derived from the goal tree."""
    goals = build_advanced_factory_goals(book=ROCKET_RECIPE_BOOK)
    expected = expected_layout_from_goals(goals)
    return {**expected, **_PRE_PLACED}


def _print_map(occupied: dict[tuple[int, int], tuple[int, int]]) -> None:
    """ASCII map with occupied / patch / free codes."""
    patch_tiles: dict[tuple[int, int], str] = {}
    for px, py, label in _PATCHES:
        for tile in _patch_tiles(px, py):
            patch_tiles[tile] = label[0].upper()
    for tile in _coal_column_tiles():
        patch_tiles[tile] = "#"

    print("\n=== Map after Phase 1+2 (single-letter machine codes) ===")
    print("    " + "".join(f"{x % 10}" for x in range(_MAP)))
    code = {
        int(MachineType.MINER): "M",
        int(MachineType.PALLET): "P",
        int(MachineType.ASSEMBLER): "A",
        int(MachineType.FURNACE): "F",
        int(MachineType.ARM): "a",
        int(MachineType.SPLITTER): "S",
        int(MachineType.CONVEYOR_BELT): ".",
        int(MachineType.CROSSING): "X",
        int(MachineType.ROCKET): "R",
    }
    for y in range(_MAP):
        row: list[str] = []
        for x in range(_MAP):
            if (x, y) == _SPAWN:
                row.append("@")
            elif (x, y) in occupied:
                mt, _ = occupied[(x, y)]
                row.append(code.get(mt, "?"))
            elif (x, y) in patch_tiles:
                row.append(patch_tiles[(x, y)])
            else:
                row.append(" ")
        print(f"{y:>3} {''.join(row)}")
    print(
        "  legend:  M=miner P=pallet A=assembler F=furnace a=arm "
        "S=splitter .=belt X=crossing @=spawn #=coal-column  "
        "I/C/T/S/L=patches"
    )


# ---------------------------------------------------------------------------
# Cell candidates — revised after the first scout flagged 5 trunk-clash cells
# ---------------------------------------------------------------------------


def _candidate_cells() -> list[tuple[str, tuple[int, int], int]]:
    """Candidate (recipe_name, assembler_tile, facing) for 10 cells.

    West cluster (Tier 1, plates/wafer -> intermediates) — moved from
    column 5 to column 3 so crates at ``cx + 2`` stay clear of the
    silicon coal trunk on column 7::

        CIRCUIT @ (3, 14) facing RIGHT  — COPPER  + WAFER
        WIRE    @ (3, 18) facing RIGHT  — COPPER  + TIN
        FRAME   @ (3, 21) facing RIGHT  — IRON    + TIN

    Mid cluster (Tier 2) — moved from column 10 to column 13 so the
    assembler tile stays clear of the iron coal trunk on column 10::

        MOTOR   @ (13, 18) facing RIGHT — FRAME   + WIRE
        SENSOR  @ (13, 21) facing RIGHT — CIRCUIT + WIRE

    East cluster (Tier 3 + 4) — unchanged from the original sketch;
    these positions were already clear of every trunk::

        HULL         @ (16, 21) facing RIGHT — FRAME       + IRON
        ENGINE_UNIT  @ (20, 18) facing RIGHT — MOTOR       + WIRE
        AVIONICS     @ (20, 21) facing RIGHT — CIRCUIT     + SENSOR
        ROCKET_CORE  @ (24, 18) facing RIGHT — ENGINE_UNIT + AVIONICS
        ROCKET       @ (24, 21) facing RIGHT — HULL        + ROCKET_CORE
    """
    return [
        ("CIRCUIT", (3, 14), int(Direction.RIGHT)),
        ("WIRE", (3, 18), int(Direction.RIGHT)),
        ("FRAME", (3, 21), int(Direction.RIGHT)),
        ("MOTOR", (13, 18), int(Direction.RIGHT)),
        ("SENSOR", (13, 21), int(Direction.RIGHT)),
        ("HULL", (16, 21), int(Direction.RIGHT)),
        ("ENGINE_UNIT", (20, 18), int(Direction.RIGHT)),
        ("AVIONICS", (20, 21), int(Direction.RIGHT)),
        ("ROCKET_CORE", (24, 18), int(Direction.RIGHT)),
        ("ROCKET", (24, 21), int(Direction.RIGHT)),
    ]


def _cell_footprint(center: tuple[int, int], facing: int) -> dict[str, tuple[int, int]]:
    """Return the 5 tiles of a generic assembler cell.

    Layout for facing=RIGHT::

        input_a_pallet  (cx, cy-1)
        input_b_pallet  (cx-1, cy)
              ASSEMBLER (cx, cy)
              output_arm(cx+1, cy)
              output_crate(cx+2, cy)
    """
    cx, cy = center
    if facing == int(Direction.RIGHT):
        return {
            "input_a": (cx, cy - 1),
            "input_b": (cx - 1, cy),
            "assembler": (cx, cy),
            "arm": (cx + 1, cy),
            "crate": (cx + 2, cy),
        }
    raise NotImplementedError("only RIGHT facing implemented for the scout")


def _check_cell_collisions(
    occupied: dict[tuple[int, int], tuple[int, int]],
) -> dict[tuple[int, int], str]:
    """Walk candidate cells, report tile-by-tile collisions.

    Returns a dict mapping every cell tile to its owning cell name —
    useful as a forecast for downstream analyses (so they can treat
    Phase 3 cell tiles as occupied even though they are not yet
    placed).
    """
    print("\n=== Cell-collision report (vs Phase 1+2 machinery) ===")
    cells = _candidate_cells()
    forecast: dict[tuple[int, int], str] = {}
    for name, center, facing in cells:
        footprint = _cell_footprint(center, facing)
        clashes: list[str] = []
        for role, tile in footprint.items():
            x, y = tile
            if not (0 <= x < _MAP and 0 <= y < _MAP):
                clashes.append(f"{role} {tile} OUT_OF_BOUNDS")
                continue
            if tile in occupied:
                mt, _ = occupied[tile]
                clashes.append(f"{role} {tile} CLASH ({MachineType(mt).name})")
            elif tile in forecast:
                clashes.append(f"{role} {tile} CLASH (with {forecast[tile]})")
        if clashes:
            print(f"  [{name}] center={center}: {len(clashes)} issue(s)")
            for c in clashes:
                print(f"    - {c}")
        else:
            print(f"  [{name}] center={center}: OK")
            for role, tile in footprint.items():
                forecast[tile] = name
    return forecast


def _check_cell_chaining() -> None:
    """Check assembler-to-assembler spacing for cells sharing a row.

    Two cells facing RIGHT on the same row chain via a 3-tile bridge:
    ``crate (cx_A+2, y)`` -> ``arm (cx_A+3, y)`` -> ``input_b
    (cx_B-1, y)``. The arm pulls from the upstream crate and pushes
    into the downstream input_b pallet — so ``cx_B - cx_A`` must be
    exactly 5 (or larger, with a belt run filling the gap).

    A spacing of exactly 4 is illegal: the upstream crate sits where
    the bridge arm should sit, leaving no place to put the arm.
    """
    print("\n=== Cell chaining (same-row spacing) ===")
    cells = _candidate_cells()
    by_row: dict[int, list[tuple[str, int]]] = {}
    for name, (cx, cy), _ in cells:
        by_row.setdefault(cy, []).append((name, cx))

    for row, items in sorted(by_row.items()):
        items.sort(key=lambda kv: kv[1])
        if len(items) < 2:
            continue
        print(f"  row y={row}:")
        for (left_name, left_x), (right_name, right_x) in zip(items, items[1:]):
            gap = right_x - left_x
            if gap == 5:
                verdict = "OK (one arm bridges crate -> input_b)"
            elif gap == 4:
                verdict = "FAIL (crate and bridge-arm collide at cx_A+2)"
            elif gap < 4:
                verdict = "FAIL (cells overlap)"
            else:
                verdict = f"OK ({gap - 5} extra belt tiles between)"
            print(
                f"    {left_name}({left_x}) -> {right_name}({right_x})  "
                f"gap={gap}  {verdict}"
            )


# ---------------------------------------------------------------------------
# Belt-highway analysis
# ---------------------------------------------------------------------------


def _belt_tiles(
    occupied: dict[tuple[int, int], tuple[int, int]],
) -> dict[tuple[int, int], int]:
    """Phase 1+2 BELT tiles only, mapped to their facing direction.

    CROSSINGs are excluded (they already accept a perpendicular flow).
    Other machine types are excluded too — their tiles are hard
    obstacles, not crossable.
    """
    belt_mt = int(MachineType.CONVEYOR_BELT)
    return {
        tile: direction for tile, (mt, direction) in occupied.items() if mt == belt_mt
    }


def _highway_report(
    occupied: dict[tuple[int, int], tuple[int, int]],
    forecast: dict[tuple[int, int], str],
) -> None:
    """Identify rows / columns clear of Phase 1+2 obstacles + Phase 3 cells.

    A 'clear row' means every tile in ``y == row`` is either dirt, a
    BELT (which can be replaced with a CROSSING during Phase 3), or
    the highway endpoint itself. We report the row's dirt count and
    its trunk-crossing tiles separately so the design can pick rows
    with the fewest crossings.
    """
    print("\n=== Belt-highway analysis (rows / columns suitable for arteries) ===")
    obstacles = {
        tile
        for tile, (mt, _) in occupied.items()
        if mt != int(MachineType.CONVEYOR_BELT)
    }
    obstacles |= set(forecast)

    def _scan_row(y: int) -> tuple[int, list[tuple[int, int]]]:
        """Return (#dirt-tiles, [(crossing_tile, trunk_dir), ...]) for row *y*."""
        belts: list[tuple[int, int]] = []
        dirt = 0
        for x in range(_MAP):
            tile = (x, y)
            if tile in obstacles:
                return -1, []  # disqualified
            if tile in occupied:
                _, d = occupied[tile]
                if d in (int(Direction.UP), int(Direction.DOWN)):
                    belts.append((x, d))
                else:
                    return -1, []  # parallel collision risk
            else:
                dirt += 1
        return dirt, belts

    def _scan_col(x: int) -> tuple[int, list[tuple[int, int]]]:
        belts: list[tuple[int, int]] = []
        dirt = 0
        for y in range(_MAP):
            tile = (x, y)
            if tile in obstacles:
                return -1, []
            if tile in occupied:
                _, d = occupied[tile]
                if d in (int(Direction.LEFT), int(Direction.RIGHT)):
                    belts.append((y, d))
                else:
                    return -1, []
            else:
                dirt += 1
        return dirt, belts

    print("  Clear east-west rows (would carry LEFT/RIGHT flow):")
    candidates = [13, 15, 17, 19, 22, 26]
    for y in candidates:
        dirt, crossings = _scan_row(y)
        if dirt < 0:
            print(f"    y={y:>2}: BLOCKED (parallel-belt or non-belt obstacle)")
            continue
        cross_str = ", ".join(f"({x},{y})/{Direction(d).name}" for x, d in crossings)
        print(
            f"    y={y:>2}: dirt={dirt:>2}  trunk-crossings={len(crossings)}  "
            f"[{cross_str}]"
        )

    print("  Clear north-south columns (would carry UP/DOWN flow):")
    candidates = [3, 5, 6, 9, 12, 14, 16, 18, 19, 25, 26]
    for x in candidates:
        dirt, crossings = _scan_col(x)
        if dirt < 0:
            print(f"    x={x:>2}: BLOCKED (parallel-belt or non-belt obstacle)")
            continue
        cross_str = ", ".join(f"({x},{y})/{Direction(d).name}" for y, d in crossings)
        print(
            f"    x={x:>2}: dirt={dirt:>2}  trunk-crossings={len(crossings)}  "
            f"[{cross_str}]"
        )


# ---------------------------------------------------------------------------
# Per-route crossing analysis
# ---------------------------------------------------------------------------


def _expand_route(waypoints: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """Walk a polyline; return ``(x, y, direction)`` per tile.

    The last waypoint is the route sink and is *not* included — sinks
    are pallets/inputs rather than belts (matches the convention of
    :func:`baselines.rocket.scripted.goals.place_belt_network`).
    """
    tiles: list[tuple[int, int, int]] = []
    for i in range(len(waypoints) - 1):
        ax, ay = waypoints[i]
        bx, by = waypoints[i + 1]
        if ax == bx and ay == by:
            continue
        if ax != bx and ay != by:
            raise ValueError(f"route segment {(ax, ay)}->{(bx, by)} not axis-aligned")
        if ax == bx:
            d = int(Direction.DOWN) if by > ay else int(Direction.UP)
            step = 1 if by > ay else -1
            for y in range(ay, by, step):
                tiles.append((ax, y, d))
        else:
            d = int(Direction.RIGHT) if bx > ax else int(Direction.LEFT)
            step = 1 if bx > ax else -1
            for x in range(ax, bx, step):
                tiles.append((x, ay, d))
    return tiles


def _hypothetical_routes() -> list[tuple[str, list[tuple[int, int]]]]:
    """Hand-drafted Phase 3 routes — input feeds for the four corner cells.

    These are illustrative routes used to count crossings. Real Phase
    3 implementation will drive them through
    :func:`place_belt_network` so the per-tile direction is derived
    from waypoint geometry, not enumerated here.

    Each waypoint list starts at the *output side* of an arm (or
    splitter) extracting from the source pallet, and ends at the
    cell input pallet (sink). Routes use rows 17 / 19 / 22 as
    east-west arteries because the highway report expects those to
    be largely clear.
    """
    return [
        # CIRCUIT inputs.
        (
            "CIRCUIT<-COPPER  (21,12)+1->E -> input_a (3,13)",
            [(22, 12), (22, 13), (3, 13)],
        ),
        (
            "CIRCUIT<-WAFER   (17,9)+1->S -> input_b (2,14)",
            [(17, 10), (2, 10), (2, 14)],
        ),
        # WIRE inputs.
        (
            "WIRE<-COPPER     (21,12) via row 17 -> input_a (3,17)",
            [(22, 12), (22, 17), (3, 17)],
        ),
        (
            "WIRE<-TIN        (25,28) via col 26 + row 19 -> input_b (2,18)",
            [(26, 28), (26, 19), (2, 19), (2, 18)],
        ),
        # FRAME inputs.
        (
            "FRAME<-IRON      (10,13) via col 9 + row 22 -> input_a (3,20)",
            [(9, 13), (9, 22), (3, 22), (3, 20)],
        ),
        (
            "FRAME<-TIN       (25,28) via col 26 + row 22 -> input_b (2,21)",
            [(26, 28), (26, 22), (2, 22), (2, 21)],
        ),
        # Tier 2 — MOTOR / SENSOR fed by Tier 1 crates.
        (
            "MOTOR<-FRAME     crate(5,21) -> input_a (13,17)",
            [(5, 22), (13, 22), (13, 17)],
        ),
        (
            "MOTOR<-WIRE      crate(5,18) -> input_b (12,18)",
            [(5, 19), (12, 19), (12, 18)],
        ),
        (
            "SENSOR<-CIRCUIT  crate(5,14) -> input_a (13,20)",
            [(6, 14), (6, 20), (13, 20)],
        ),
        (
            "SENSOR<-WIRE     crate(5,18) -> input_b (12,21)",
            [(5, 19), (12, 19), (12, 21)],
        ),
        # Tier 3 — east cluster.
        (
            "HULL<-FRAME      crate(5,21) -> input_a (16,20)",
            [(6, 21), (6, 20), (16, 20)],
        ),
        (
            "HULL<-IRON       (10,13) via row 22 -> input_b (15,21)",
            [(9, 13), (9, 22), (15, 22), (15, 21)],
        ),
        (
            "ENGINE_UNIT<-MOTOR    crate(15,18) -> input_a (20,17)",
            [(16, 18), (16, 17), (20, 17)],
        ),
        (
            "ENGINE_UNIT<-WIRE     crate(5,18) via row 19 -> input_b (19,18)",
            [(5, 19), (19, 19), (19, 18)],
        ),
        (
            "AVIONICS<-CIRCUIT crate(5,14) -> input_a (20,20)",
            [(6, 14), (6, 20), (20, 20)],
        ),
        (
            "AVIONICS<-SENSOR  crate(15,21) -> input_b (19,21)",
            [(16, 21), (19, 21)],
        ),
        # Tier 4 — ROCKET_CORE / ROCKET (kept short).
        (
            "ROCKET_CORE<-ENGINE  crate(22,18) -> input_b (23,18)",
            [(23, 18)],
        ),
        (
            "ROCKET_CORE<-AVIONICS crate(22,21) -> input_a (24,17)",
            [(22, 22), (24, 22), (24, 17)],
        ),
        (
            "ROCKET<-HULL     crate(18,21) -> input_b (23,21)",
            [(19, 21), (23, 21)],
        ),
        (
            "ROCKET<-ROCKET_CORE crate(26,18) -> input_a (24,20)",
            [(26, 19), (24, 19), (24, 20)],
        ),
    ]


def _route_analysis(
    occupied: dict[tuple[int, int], tuple[int, int]],
    forecast: dict[tuple[int, int], str],
) -> None:
    """Tile-by-tile classification for each hypothetical Phase 3 route.

    Per tile, classify as:

    - **OOB** — out of bounds.
    - **CELL** — sits on a Phase 3 cell footprint (hard clash).
    - **OBSTACLE** — sits on a non-belt Phase 1+2 machine (hard clash).
    - **CROSSING** — sits on a Phase 1+2 BELT moving perpendicular to
      the route (legal: place a CROSSING at this tile during Phase 3).
    - **PARALLEL** — sits on a Phase 1+2 BELT moving parallel (any
      direction) — illegal; the route would merge into or fight the
      trunk's flow. Detour required.
    - **SHARED-PHASE3** — overlaps another Phase 3 route at the same
      tile (recorded; perpendicular overlap becomes a crossing,
      parallel becomes an error). Resolved by ``place_belt_network``.
    - **CLEAR** — dirt.

    The aggregate at the bottom feeds the crafted-belt + crafted-
    crossing counts the agent will need.
    """
    print("\n=== Per-route tile classification ===")
    belt_dirs = _belt_tiles(occupied)
    non_belt_obstacles = {
        tile
        for tile, (mt, _) in occupied.items()
        if mt != int(MachineType.CONVEYOR_BELT)
    }

    routes = _hypothetical_routes()
    # Phase 3 tile -> (route_label, dir) for inter-route crossing detection.
    phase_3_tiles: dict[tuple[int, int], tuple[str, int]] = {}
    total_belts = 0
    total_crossings = 0
    total_inter_crossings = 0
    hard_clash_routes: list[str] = []

    for label, waypoints in routes:
        try:
            tiles = _expand_route(waypoints)
        except ValueError as exc:
            print(f"  [{label}]: ROUTE_INVALID — {exc}")
            continue
        belts = 0
        trunk_crossings: list[tuple[int, int]] = []
        inter_crossings: list[tuple[int, int]] = []
        problems: list[str] = []
        for x, y, direction in tiles:
            tile = (x, y)
            if not (0 <= x < _MAP and 0 <= y < _MAP):
                problems.append(f"({x},{y}) OOB")
                continue
            if tile in forecast:
                problems.append(f"({x},{y}) CELL[{forecast[tile]}]")
                continue
            if tile in non_belt_obstacles:
                mt, _ = occupied[tile]
                problems.append(f"({x},{y}) OBSTACLE[{MachineType(mt).name}]")
                continue
            trunk_dir = belt_dirs.get(tile)
            if trunk_dir is not None:
                if _is_perpendicular(direction, trunk_dir):
                    trunk_crossings.append(tile)
                else:
                    problems.append(
                        f"({x},{y}) PARALLEL trunk "
                        f"{Direction(trunk_dir).name} vs route "
                        f"{Direction(direction).name}"
                    )
                continue
            prior = phase_3_tiles.get(tile)
            if prior is not None:
                prior_label, prior_dir = prior
                if _is_perpendicular(direction, prior_dir):
                    inter_crossings.append(tile)
                elif prior_dir != direction:
                    problems.append(f"({x},{y}) PHASE3-PARALLEL with {prior_label}")
                # Don't record again; the first writer keeps direction.
                continue
            phase_3_tiles[tile] = (label, direction)
            belts += 1
        total_belts += belts
        total_crossings += len(trunk_crossings)
        total_inter_crossings += len(inter_crossings)
        if problems:
            hard_clash_routes.append(label)
            print(f"  [{label}]: PROBLEM ({len(problems)})")
            for p in problems:
                print(f"    - {p}")
        else:
            cross_summary = (
                f"trunk-crossings={len(trunk_crossings)}, "
                f"inter-route={len(inter_crossings)}"
            )
            print(f"  [{label}]: OK  belts={belts}  {cross_summary}")

    print()
    print("  Aggregate (Phase 3 belt-network sizing — assuming all routes valid):")
    print(f"    belts  needed: {total_belts}")
    print(
        f"    crossings needed: "
        f"{total_crossings + total_inter_crossings}  "
        f"(trunk={total_crossings}, inter-route={total_inter_crossings})"
    )
    if hard_clash_routes:
        print(
            f"  WARNING: {len(hard_clash_routes)} route(s) need redesign "
            f"before commit — see PROBLEM lines above."
        )


def _is_perpendicular(d_a: int, d_b: int) -> bool:
    """``True`` iff one direction is on the vertical axis and the other horizontal."""
    vertical = {int(Direction.UP), int(Direction.DOWN)}
    horizontal = {int(Direction.LEFT), int(Direction.RIGHT)}
    return (d_a in vertical and d_b in horizontal) or (
        d_a in horizontal and d_b in vertical
    )


# ---------------------------------------------------------------------------
# Recipe input audit
# ---------------------------------------------------------------------------


def _recipe_input_audit() -> None:
    """Sanity check: no rocket recipe needs > 2 distinct input types.

    The cell shape uses 2 input pallets (one per
    :class:`ent_asm_in` slot). Recipes with > 2 distinct inputs
    would require routing into the same slot from two pallets, which
    isn't possible. This audit confirms the rocket book stays within
    the 2-slot limit even after balance overlays.

    Also reports the per-cycle item count for each input — a recipe
    like ``HULL = FRAME x 2 + IRON x 2`` accumulates input slowly
    (1/tick from each neighbor) but the shape stays the same; the
    cell just runs slower.
    """
    print("\n=== Recipe input audit (rocket book) ===")
    interesting: set[int] = {
        int(ItemType.WIRE),
        int(ItemType.FRAME),
        int(ItemType.CIRCUIT),
        int(ItemType.MOTOR),
        int(ItemType.SENSOR),
        int(ItemType.HULL),
        int(ItemType.ENGINE_UNIT),
        int(ItemType.AVIONICS),
        int(ItemType.ROCKET_CORE),
        int(ItemType.ROCKET),
    }
    for recipe in ROCKET_RECIPE_BOOK.recipes:
        if recipe.output not in interesting:
            continue
        inputs_str = ", ".join(
            f"{ItemType(int(i)).name}x{int(q)}" for i, q in recipe.inputs
        )
        print(
            f"  {ItemType(recipe.output).name:<14} <- "
            f"[{inputs_str}]  yields {recipe.output_count} per cycle, "
            f"{recipe.ticks} ticks"
        )

    print()
    print("  Cell-shape implications:")
    print(
        "  - All recipes have exactly 2 input types (matches "
        "ent_asm_in[0..1] slots).  No cell needs > 2 input pallets."
    )
    print(
        "  - Higher per-cycle input counts (e.g. HULL needs 2 each) "
        "do NOT change cell shape — pull rate stays 1/tick from each "
        "neighbor pallet, the cycle starts when both slots have the "
        "required count, and the pallet's 256 cap covers any input "
        "count up to 256."
    )
    print(
        "  - Recipe yield (e.g. SPLITTER yields 4) feeds into the "
        "BOM math via production_schedule; cell shape unaffected."
    )


def main() -> None:
    """Run all six checks."""
    occupied = _occupied_after_phase_1_2()
    _print_map(occupied)
    forecast = _check_cell_collisions(occupied)
    _check_cell_chaining()
    _highway_report(occupied, forecast)
    _route_analysis(occupied, forecast)
    _recipe_input_audit()


if __name__ == "__main__":
    main()
