"""High-level goals for the scripted rocket agent.

A :class:`Goal` composes :mod:`skills` to achieve a concrete
in-world outcome (e.g. "have 10 iron ore in inventory", "place a
miner on an ore tile"). Goals are finite-state machines; the
:mod:`planner` layer sequences them through the phases from
``docs/rocket_scripted_agent.md``.

Each goal exposes the same shape as a skill:

    goal.step(view: WorldView) -> (Result, int | None)

Goals may swap out their internal skill when one completes (e.g.
after arriving at a patch, replace ``NavigateAdjacent`` with
``FaceAndInteract`` to mine). They carry enough state to replan
on the fly — for instance if an ore patch depletes, ``MineOre``
picks the next nearest patch without giving up.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from factoriax.constants import Action, Direction, ItemType, MachineType
from factoriax.recipes import RECIPES

from .skills import (
    FaceAndInteract,
    PlaceAt,
    Result,
    Skill,
    StandOnAndAct,
    StepReturn,
)
from .world_model import (
    WorldView,
    place_action,
    withdraw_action,
)

# Signature for "pick a location" callbacks used by :class:`PlaceMachine`.
LocationPredicate = Callable[[WorldView], tuple[int, int] | None]


class Goal:
    """Base class for high-level goals."""

    name: str = "Goal"

    def step(self, view: WorldView) -> StepReturn:  # pragma: no cover - abstract
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Resource-gathering
# ---------------------------------------------------------------------------


class MineOre(Goal):
    """Mine ore until the player holds at least *count* of *item_type*.

    Replans each tick: reselects the nearest ore patch from the current
    observation, so a depleted patch or a new blocker doesn't strand the
    agent.
    """

    name = "MineOre"

    def __init__(self, item_type: int | ItemType, count: int) -> None:
        self.item_type = int(item_type)
        self.count = count
        self._active: Skill | None = None
        self._active_target: tuple[int, int] | None = None

    def step(self, view: WorldView) -> StepReturn:
        if view.player.held(self.item_type) >= self.count:
            return Result.DONE, None

        # Pick the nearest patch by BFS distance, approximated via
        # Manhattan distance from the player.
        px, py = view.player.pos
        tiles = view.ore_tiles(self.item_type)
        if not tiles:
            return Result.FAIL, None
        tiles.sort(key=lambda t: abs(t[0] - px) + abs(t[1] - py))

        # If we already picked a target and it's still productive, keep it.
        if (
            self._active_target is not None
            and view.block_resources[self._active_target[1], self._active_target[0]] > 0
            and self._active is not None
        ):
            result, action = self._active.step(view)
            if result is Result.DONE:
                # Finished one mine cycle. Clear the skill; next tick we
                # start another (same patch, or a new nearest if depleted).
                self._active = None
                # Fall through to create a fresh skill below — don't
                # waste a NOOP when we could emit the next move or mine.
            elif result is Result.FAIL:
                self._active = None
                self._active_target = None
                return Result.RUNNING, int(Action.NOOP)
            else:
                return Result.RUNNING, action

        # Select (or re-select) a patch and start a fresh StandOnAndAct.
        self._active_target = tiles[0]
        self._active = StandOnAndAct(self._active_target, int(Action.MINE))
        return self._active.step(view)


# ---------------------------------------------------------------------------
# Crafting
# ---------------------------------------------------------------------------


_ITEM_TO_CRAFT_ACTION: dict[int, int] = {
    int(ItemType.IRON_PLATE): int(Action.CRAFT_IRON_PLATE),
    int(ItemType.COPPER_PLATE): int(Action.CRAFT_COPPER_PLATE),
    int(ItemType.TIN_PLATE): int(Action.CRAFT_TIN_PLATE),
    int(ItemType.WAFER): int(Action.CRAFT_WAFER),
    int(ItemType.FRAME): int(Action.CRAFT_FRAME),
    int(ItemType.CIRCUIT): int(Action.CRAFT_CIRCUIT),
    int(ItemType.WIRE): int(Action.CRAFT_WIRE),
    int(ItemType.MOTOR): int(Action.CRAFT_MOTOR),
    int(ItemType.SENSOR): int(Action.CRAFT_SENSOR),
    int(ItemType.CONVEYOR_BELT): int(Action.CRAFT_BELT),
    int(ItemType.MINER): int(Action.CRAFT_MINER),
    int(ItemType.ASSEMBLER): int(Action.CRAFT_ASSEMBLER),
    int(ItemType.PALLET): int(Action.CRAFT_PALLET),
    int(ItemType.ARM): int(Action.CRAFT_ARM),
    int(ItemType.FURNACE): int(Action.CRAFT_FURNACE),
    int(ItemType.BASIC_SCIENCE_PACK): int(Action.CRAFT_BASIC_SCIENCE),
    int(ItemType.ADVANCED_SCIENCE_PACK): int(Action.CRAFT_ADV_SCIENCE),
    int(ItemType.ROCKET): int(Action.CRAFT_ROCKET),
}


class CraftItem(Goal):
    """Emit ``CRAFT_<item>`` up to *count* times (or until satisfied).

    The goal is forgiving by design: if ingredients run out partway,
    the agent keeps attempting for a small buffer of extra ticks and
    then moves on. This trades an ideal-world guarantee ("exactly
    *count* produced") for robustness against resource shortages —
    downstream goals that need the item will FAIL on their own and
    the planner will skip them.
    """

    name = "CraftItem"

    def __init__(
        self,
        item_type: int | ItemType,
        count: int,
        extra_attempts: int = 2,
    ) -> None:
        self.item_type = int(item_type)
        if self.item_type not in _ITEM_TO_CRAFT_ACTION:
            raise ValueError(
                f"item_type {ItemType(self.item_type).name} is not craftable",
            )
        self.count = count
        self._attempts_remaining = count + extra_attempts
        self._started_held: int | None = None

    def step(self, view: WorldView) -> StepReturn:
        held = view.player.held(self.item_type)
        if self._started_held is None:
            self._started_held = held
        if held >= self.count:
            return Result.DONE, None
        if self._attempts_remaining <= 0:
            # Out of attempts; either we made some progress or the
            # recipe is unsatisfiable. Either way the planner moves on.
            return Result.DONE, None
        self._attempts_remaining -= 1
        return Result.RUNNING, _ITEM_TO_CRAFT_ACTION[self.item_type]


# ---------------------------------------------------------------------------
# Machine placement
# ---------------------------------------------------------------------------


def _first_ore_tile(
    view: WorldView, item_type: int | ItemType
) -> tuple[int, int] | None:
    """Nearest ore tile of the requested type, or None."""
    tiles = view.ore_tiles(item_type)
    if not tiles:
        return None
    px, py = view.player.pos
    tiles.sort(key=lambda t: abs(t[0] - px) + abs(t[1] - py))
    return tiles[0]


def on_ore(item_type: int | ItemType) -> LocationPredicate:
    """Predicate: pick the nearest ore tile of *item_type*.

    Filters out tiles that already carry a machine so chained
    :class:`PlaceMachine` calls can cover a patch instead of
    repeatedly aiming at the same tile.

    Miners placed on ore tiles auto-extract from them; the engine
    accepts a ``PLACE_MINER`` action that targets an ore tile even
    though the tile isn't "walkable" in the usual sense.
    """

    def picker(view: WorldView) -> tuple[int, int] | None:
        tiles = view.ore_tiles(item_type)
        if not tiles:
            return None
        px, py = view.player.pos
        unoccupied = [t for t in tiles if view.machine_type[t[1], t[0]] == 0]
        if not unoccupied:
            return None
        unoccupied.sort(key=lambda t: abs(t[0] - px) + abs(t[1] - py))
        return unoccupied[0]

    return picker


def free_tile_near_player(
    max_radius: int = 8,
    min_distance: int = 2,
) -> LocationPredicate:
    """Predicate: closest walkable tile that holds no machine.

    Picks a tile at least ``min_distance`` Manhattan away from the
    player and at most ``max_radius``. The minimum distance guards
    against the player boxing themselves in with machines on their
    4-neighbors — after placing, the player's immediate neighbors
    include the targeted tile as a new machine, but the tile they were
    standing on (one-tile-removed from the target) remains walkable.
    Without this buffer, sequential placements on each of the four
    neighbors strand the player on an island.
    """

    def picker(view: WorldView) -> tuple[int, int] | None:
        px, py = view.player.pos
        empty = view.walkable & (view.machine_type == 0)
        empty_np = empty.copy()
        empty_np[py, px] = False
        if not empty_np.any():
            return None
        ys, xs = np.nonzero(empty_np)
        dists = np.abs(xs - px) + np.abs(ys - py)
        band = (dists >= min_distance) & (dists <= max_radius)
        if not band.any():
            # Fall back to any empty tile if the preferred band is
            # unavailable (e.g. map is almost full).
            band = dists <= max_radius
        if not band.any():
            return None
        xs = xs[band]
        ys = ys[band]
        dists = dists[band]
        idx = int(np.argmin(dists))
        return (int(xs[idx]), int(ys[idx]))

    return picker


class PlaceMachine(Goal):
    """Place a machine of *machine_type* at a tile chosen by *predicate*.

    Fails fast if the predicate returns ``None`` (no valid location
    exists right now) or if the player doesn't hold the placeable item.
    """

    name = "PlaceMachine"

    def __init__(
        self,
        machine_type: int | MachineType,
        predicate: LocationPredicate,
    ) -> None:
        self.machine_type = int(machine_type)
        self.item_type = _machine_to_item(self.machine_type)
        self.predicate = predicate
        self._active: FaceAndInteract | None = None
        self._start_count: int | None = None

    def step(self, view: WorldView) -> StepReturn:
        # Detect success: machine count went up, our inventory went down.
        if self._start_count is not None and view.total_machines() > self._start_count:
            return Result.DONE, None

        if view.player.held(self.item_type) < 1:
            return Result.FAIL, None

        if self._active is None:
            target = self.predicate(view)
            if target is None:
                return Result.FAIL, None
            self._start_count = view.total_machines()
            self._active = FaceAndInteract(
                target,
                place_action(self.machine_type),
            )

        result, action = self._active.step(view)
        if result is Result.DONE:
            # Let the next tick verify via the total_machines check.
            self._active = None
            return Result.RUNNING, int(Action.NOOP)
        if result is Result.FAIL:
            self._active = None
            return Result.FAIL, None
        return Result.RUNNING, action


class PlaceMachineAt(Goal):
    """Place a machine at a specific tile facing a specific direction.

    Reuses :class:`PlaceAt` under the hood: navigates the player to
    ``target - unit_vec(facing)``, turns to ``facing``, then emits
    ``PLACE_*``. This is how the factory agent aligns miners so
    their per-tick push lands in the adjacent pallet instead of a
    dead tile.
    """

    name = "PlaceMachineAt"

    def __init__(
        self,
        machine_type: int | MachineType,
        target: tuple[int, int],
        facing: int,
    ) -> None:
        self.machine_type = int(machine_type)
        self.item_type = _machine_to_item(self.machine_type)
        self.target = target
        self.facing = int(facing)
        self._active: PlaceAt | None = None
        self._start_count: int | None = None

    def step(self, view: WorldView) -> StepReturn:
        if self._start_count is not None and view.total_machines() > self._start_count:
            return Result.DONE, None

        if view.player.held(self.item_type) < 1:
            return Result.FAIL, None

        if self._active is None:
            self._start_count = view.total_machines()
            self._active = PlaceAt(
                self.target,
                self.facing,
                place_action(self.machine_type),
            )

        result, action = self._active.step(view)
        if result is Result.DONE:
            self._active = None
            return Result.RUNNING, int(Action.NOOP)
        if result is Result.FAIL:
            self._active = None
            return Result.FAIL, None
        return Result.RUNNING, action


class BeltPath:
    """One axis-aligned belt path in a :func:`place_belt_network` plan.

    Attributes:
        waypoints: ``(x, y)`` tiles in path order. Same shape as the
            input to :func:`place_belt_path` — at least two entries,
            consecutive entries differ in exactly one axis, and the
            last waypoint is the sink (not itself a belt).
        label: Optional human-readable string used in the error
            messages emitted by :func:`place_belt_network`. Bare
            integer indices work too, but a name like ``"iron coal"``
            makes goal-construction-time failures easier to triage.
    """

    __slots__ = ("waypoints", "label")

    def __init__(
        self,
        waypoints: list[tuple[int, int]],
        label: str = "",
    ) -> None:
        self.waypoints = waypoints
        self.label = label


def _belt_path_label(paths: list[BeltPath], idx: int) -> str:
    """Format a path's label for inclusion in a ValueError message."""
    label = paths[idx].label
    return f"'{label}'" if label else f"#{idx}"


# (vert_out_dir, horiz_out_dir) -> packed crossing encoding (1..4).
# Mirrors :data:`factoriax.belts.CROSSING_AXIS_DIRS`. The vertical
# axis maps UP/DOWN flow; horizontal maps LEFT/RIGHT flow.
_CROSSING_ENCODINGS: dict[tuple[int, int], int] = {
    (int(Direction.DOWN), int(Direction.RIGHT)): 1,
    (int(Direction.DOWN), int(Direction.LEFT)): 2,
    (int(Direction.UP), int(Direction.RIGHT)): 3,
    (int(Direction.UP), int(Direction.LEFT)): 4,
}

_VERTICAL_DIRECTIONS = frozenset({int(Direction.UP), int(Direction.DOWN)})


def _expand_belt_path_tiles(
    waypoints: list[tuple[int, int]],
    label: str,
) -> list[tuple[int, int]]:
    """Expand a waypoint list into a per-tile path (sink included).

    Shares the segment-validation rules with :func:`place_belt_path`
    but raises errors prefixed with ``label`` so multi-path callers can
    pinpoint which path is malformed.
    """
    if len(waypoints) < 2:
        raise ValueError(
            f"belt network: path {label} needs >=2 waypoints, got {len(waypoints)}",
        )
    tiles: list[tuple[int, int]] = [waypoints[0]]
    for i in range(len(waypoints) - 1):
        a = waypoints[i]
        b = waypoints[i + 1]
        if a == b:
            raise ValueError(
                f"belt network: degenerate segment {a}->{b} "
                f"(waypoints equal) in path {label}",
            )
        if a[0] != b[0] and a[1] != b[1]:
            raise ValueError(
                f"belt network: segment {a}->{b} is not axis-aligned in path {label}",
            )
        dx = 0 if a[0] == b[0] else (1 if b[0] > a[0] else -1)
        dy = 0 if a[1] == b[1] else (1 if b[1] > a[1] else -1)
        cur = a
        while cur != b:
            cur = (cur[0] + dx, cur[1] + dy)
            tiles.append(cur)
    return tiles


def place_belt_network(
    paths: list[BeltPath],
    *,
    occupied: set[tuple[int, int]] | None = None,
    map_size: tuple[int, int] | None = None,
    start_near: tuple[int, int] | None = None,
) -> list[Goal]:
    """Plan a set of belt paths sharing perpendicular intersection tiles.

    For every tile used by exactly one path, emit a CONVEYOR_BELT.
    For every tile used by two paths whose flow directions are
    perpendicular (one of {UP, DOWN}, one of {LEFT, RIGHT}), emit a
    CROSSING with the encoding implied by ``(vert_dir, horiz_dir)``.

    Any conflict that a CROSSING cannot resolve raises ``ValueError``
    at goal-construction time so the agent author sees the failure
    while building the goal list, not 5000 ticks into the rollout:

    * **parallel collision** — both paths push the *same* direction
      through the tile (items would mix into the same belt).
    * **anti-parallel collision** — the two paths push opposite
      directions on the same axis (a U-turn that no crossing
      encoding represents).
    * **3-way (or more) junction** — a CROSSING supports two axes
      only; T- and X-junctions of three or more paths must be
      decomposed into multiple crossings by hand.
    * **belt passes through another path's sink** — sinks are
      destinations, not pass-throughs; the route would dump items
      into the wrong receiver.

    Placement order is a *proximity-aware* topological walk: at each
    step, among the tiles whose forced stand-tile constraint is
    already satisfied (BELT facing ``d`` needs the tile at ``-d``
    walkable; CROSSING needs at least one of its four neighbours
    walkable), pick the one with the smallest Manhattan distance to a
    moving cursor. The cursor starts at ``start_near`` (or, if
    omitted, at the lex-smallest ready tile) and updates to the most
    recently emitted tile after each placement. The net effect is
    that the agent traces each trunk continuously — placing every
    belt of one path before moving to the next — instead of
    bouncing between trunks because lexicographic ordering visits
    them in column-major waves. Walkable means dirt (any tile not in
    our plan and not in ``occupied``) or an already-emitted
    belt/crossing — both of which the runtime treats as walkable
    for the player.

    The single-path case
    ``place_belt_network([BeltPath(waypoints)])`` is equivalent to
    :func:`place_belt_path(waypoints)`. Callers with a single path
    should keep using :func:`place_belt_path` for clarity.

    Args:
        paths: One :class:`BeltPath` per logical trunk. Order does
            not matter — the topological sort handles dependencies.
        occupied: Tiles already taken by other machines. Crossings
            and belts in our plan must not collide with these.
        map_size: Optional ``(width, height)`` for in-bounds checks
            on every tile (belts, crossings, and sinks).
        start_near: Optional ``(x, y)`` seed for the proximity-aware
            walk. Pass the agent's expected position when this phase
            begins (typically the spawn or the last placement of the
            previous phase) so the first emitted trunk starts close
            to where the agent already is. When ``None``, the cursor
            initialises to the lex-smallest ready tile, which is the
            same starting tile as the previous lexicographic-only
            implementation.

    Returns:
        A flat list of :class:`PlaceMachineAt` goals: one per
        belt/crossing, in placement order. Sinks are *not* emitted —
        they are the responsibility of the caller (a pallet, an
        assembler input, the receiving end of a belt path elsewhere).

    Raises:
        ValueError: any of the conflict categories above, or a path
            self-intersects, or the topological sort cannot find an
            ordering (every remaining tile's stand tile is itself in
            the unplaced set — typically a cycle in the belt graph).
    """
    if not paths:
        return []

    occupied_set = set(occupied or ())
    # tile -> [(path_idx, segment_idx_in_path, direction), ...]
    tile_uses: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    sink_owner: dict[tuple[int, int], int] = {}

    for path_idx, path_obj in enumerate(paths):
        label = _belt_path_label(paths, path_idx)
        tiles = _expand_belt_path_tiles(path_obj.waypoints, label)
        seen_in_path: set[tuple[int, int]] = set()
        for tile in tiles:
            if tile in seen_in_path:
                raise ValueError(
                    f"belt network: path {label} self-intersects at {tile}",
                )
            seen_in_path.add(tile)
        for i, tile in enumerate(tiles[:-1]):
            nx, ny = tiles[i + 1]
            ddx, ddy = nx - tile[0], ny - tile[1]
            if ddx == 1:
                direction = int(Direction.RIGHT)
            elif ddx == -1:
                direction = int(Direction.LEFT)
            elif ddy == 1:
                direction = int(Direction.DOWN)
            else:
                direction = int(Direction.UP)
            tile_uses.setdefault(tile, []).append(
                (path_idx, i, direction),
            )
        sink_tile = tiles[-1]
        if sink_tile in sink_owner:
            raise ValueError(
                f"belt network: tile {sink_tile} is the sink of paths "
                f"{_belt_path_label(paths, sink_owner[sink_tile])} and "
                f"{label} — sinks must be unique",
            )
        sink_owner[sink_tile] = path_idx

    # Sinks cannot be passed through by another path's belt cell.
    for sink_tile, owner_idx in sink_owner.items():
        if sink_tile in tile_uses:
            other = tile_uses[sink_tile][0][0]
            raise ValueError(
                f"belt network: tile {sink_tile} is the sink of path "
                f"{_belt_path_label(paths, owner_idx)} and a belt-cell of "
                f"path {_belt_path_label(paths, other)} — sinks cannot "
                f"be passed through",
            )

    if map_size is not None:
        width, height = map_size
        all_tiles: set[tuple[int, int]] = set(tile_uses) | set(sink_owner)
        for tile in all_tiles:
            x, y = tile
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(
                    f"belt network: tile {tile} is out of map bounds "
                    f"(width={width}, height={height})",
                )
    for tile in tile_uses:
        if tile in occupied_set:
            raise ValueError(
                f"belt network: tile {tile} collides with an occupied tile",
            )

    # Classify every used tile: BELT(direction) or CROSSING(encoding).
    tile_kind: dict[tuple[int, int], tuple[str, int]] = {}
    for tile, uses in tile_uses.items():
        if len(uses) == 1:
            tile_kind[tile] = ("BELT", uses[0][2])
            continue
        if len(uses) == 2:
            d1, d2 = uses[0][2], uses[1][2]
            if d1 == d2:
                raise ValueError(
                    f"belt network: parallel collision at {tile}; paths "
                    f"{_belt_path_label(paths, uses[0][0])} and "
                    f"{_belt_path_label(paths, uses[1][0])} both push "
                    f"{Direction(d1).name} through this tile",
                )
            d1_vert = d1 in _VERTICAL_DIRECTIONS
            d2_vert = d2 in _VERTICAL_DIRECTIONS
            if d1_vert == d2_vert:
                # Same axis but different direction = anti-parallel.
                raise ValueError(
                    f"belt network: anti-parallel collision at {tile}; "
                    f"path {_belt_path_label(paths, uses[0][0])} pushes "
                    f"{Direction(d1).name} and "
                    f"{_belt_path_label(paths, uses[1][0])} pushes "
                    f"{Direction(d2).name}",
                )
            if d1_vert:
                vert_dir, horiz_dir = d1, d2
            else:
                vert_dir, horiz_dir = d2, d1
            tile_kind[tile] = ("CROSSING", _CROSSING_ENCODINGS[(vert_dir, horiz_dir)])
            continue
        labels = [_belt_path_label(paths, u[0]) for u in uses]
        raise ValueError(
            f"belt network: {len(uses)}-way junction at {tile}; "
            f"a CROSSING supports 2 axes only "
            f"(paths involved: {', '.join(labels)})",
        )

    # Proximity-aware topological walk. Each iteration picks the
    # ready tile (forced stand-tile already walkable) closest to a
    # cursor that follows the most recent placement. Ties on distance
    # break lexicographically so emit order stays deterministic.
    direction_offset = {
        int(Direction.UP): (0, -1),
        int(Direction.DOWN): (0, 1),
        int(Direction.LEFT): (-1, 0),
        int(Direction.RIGHT): (1, 0),
    }
    placed: set[tuple[int, int]] = set()
    pending = set(tile_kind)
    emit_order: list[tuple[int, int]] = []

    def _walkable(tile: tuple[int, int]) -> bool:
        if tile in placed:
            return True
        if tile in pending:
            return False
        return tile not in occupied_set

    def _is_ready(tile: tuple[int, int]) -> bool:
        kind, payload = tile_kind[tile]
        if kind == "BELT":
            dx, dy = direction_offset[payload]
            return _walkable((tile[0] - dx, tile[1] - dy))
        return any(
            _walkable((tile[0] + dx, tile[1] + dy))
            for dx, dy in direction_offset.values()
        )

    cursor: tuple[int, int] | None = start_near
    while pending:
        ready = [tile for tile in pending if _is_ready(tile)]
        if not ready:
            stuck = sorted(pending)
            raise ValueError(
                f"belt network: could not order placement; "
                f"{len(pending)} tiles have no walkable stand tile "
                f"(first few: {stuck[:3]}). The belt graph likely "
                f"contains a cycle.",
            )
        if cursor is None:
            next_tile = min(ready)
        else:
            cx, cy = cursor
            next_tile = min(
                ready,
                key=lambda t: (abs(t[0] - cx) + abs(t[1] - cy), t),
            )
        placed.add(next_tile)
        pending.remove(next_tile)
        emit_order.append(next_tile)
        cursor = next_tile

    goals: list[Goal] = []
    for tile in emit_order:
        kind, payload = tile_kind[tile]
        if kind == "BELT":
            goals.append(
                PlaceMachineAt(MachineType.CONVEYOR_BELT, tile, payload),
            )
        else:
            goals.append(
                PlaceMachineAt(MachineType.CROSSING, tile, payload),
            )
    return goals


def belt_network_inventory(paths: list[BeltPath]) -> dict[int, int]:
    """Return the inventory cost of a :func:`place_belt_network` plan.

    Re-runs the path expansion + tile classification and counts BELTs
    and CROSSINGs. Calling this during goal construction also acts as
    a pre-flight: any of the conflict ValueErrors that
    :func:`place_belt_network` would raise propagate out here too, so
    a typo in the path geometry surfaces while writing the bootstrap
    inventory rather than half-way through Phase B.

    Args:
        paths: One :class:`BeltPath` per logical trunk.

    Returns:
        ``{ItemType.CONVEYOR_BELT: n_belts, ItemType.CROSSING:
        n_crossings}``. Keys are omitted when their count is zero so a
        crossing-free plan returns ``{ItemType.CONVEYOR_BELT: n}``
        only.
    """
    goals = place_belt_network(paths)
    n_belts = 0
    n_crossings = 0
    for goal in goals:
        assert isinstance(goal, PlaceMachineAt)
        if goal.machine_type == int(MachineType.CONVEYOR_BELT):
            n_belts += 1
        elif goal.machine_type == int(MachineType.CROSSING):
            n_crossings += 1
    cost: dict[int, int] = {}
    if n_belts:
        cost[int(ItemType.CONVEYOR_BELT)] = n_belts
    if n_crossings:
        cost[int(ItemType.CROSSING)] = n_crossings
    return cost


def place_belt_path(
    waypoints: list[tuple[int, int]],
    *,
    occupied: set[tuple[int, int]] | None = None,
) -> list[Goal]:
    """Lay a chain of belts along an axis-aligned waypoint path.

    Each consecutive pair of waypoints defines an axis-aligned segment;
    the helper enumerates every tile from the first waypoint up to (but
    not including) the last, places a belt on each one with direction
    pointing toward the next tile in the chain, and treats the final
    waypoint as the sink (the receiver of the trunk's items, e.g. a
    coal-buffer pallet placed elsewhere).

    Belt directions are inferred from the path: a belt with the next
    tile to its east faces RIGHT, north → UP, and so on. The belt at a
    corner waypoint (e.g. ``(10, 12)`` between a UP segment and a LEFT
    segment) takes the *outgoing* segment's direction so its push lands
    on the first tile of the next segment.

    Args:
        waypoints: ``(x, y)`` tiles in path order. Must contain at
            least two entries (start and sink). Consecutive entries
            must differ in exactly one axis (segments are pure
            horizontal or vertical).
        occupied: Optional set of tiles already occupied by other
            machines. If any path tile (other than the sink) is in
            this set, the helper raises ``ValueError`` so the caller
            can fix the route before issuing the goals — catches
            collisions at goal-construction time instead of at
            placement time. Belts walked over twice are likewise
            rejected.

    Returns:
        A list of :class:`PlaceMachineAt` goals, one per belt. The
        agent must hold ``len(returned_list)`` ``CONVEYOR_BELT`` items
        before issuing them; per-tile inventory shortage still surfaces
        at runtime via :class:`PlaceMachineAt`'s ``held >= 1`` check.

    Raises:
        ValueError: ``waypoints`` is shorter than 2, a segment is not
            axis-aligned, two consecutive waypoints are equal, the path
            self-intersects, or any non-sink tile collides with the
            ``occupied`` set.
    """
    if len(waypoints) < 2:
        raise ValueError(
            f"place_belt_path needs >=2 waypoints, got {len(waypoints)}",
        )
    blocked = set(occupied or ())

    path: list[tuple[int, int]] = [waypoints[0]]
    for i in range(len(waypoints) - 1):
        a = waypoints[i]
        b = waypoints[i + 1]
        if a == b:
            raise ValueError(f"degenerate segment {a}->{b} (waypoints equal)")
        if a[0] != b[0] and a[1] != b[1]:
            raise ValueError(
                f"segment {a}->{b} is not axis-aligned",
            )
        dx = 0 if a[0] == b[0] else (1 if b[0] > a[0] else -1)
        dy = 0 if a[1] == b[1] else (1 if b[1] > a[1] else -1)
        cur = a
        while cur != b:
            cur = (cur[0] + dx, cur[1] + dy)
            path.append(cur)

    sink = path[-1]
    seen: set[tuple[int, int]] = set()
    goals: list[Goal] = []
    for i, tile in enumerate(path[:-1]):
        if tile in seen:
            raise ValueError(f"path self-intersects at {tile}")
        if tile in blocked:
            raise ValueError(
                f"belt at {tile} collides with occupied tile",
            )
        seen.add(tile)
        nx, ny = path[i + 1]
        ddx, ddy = nx - tile[0], ny - tile[1]
        if ddx == 1:
            direction = Direction.RIGHT
        elif ddx == -1:
            direction = Direction.LEFT
        elif ddy == 1:
            direction = Direction.DOWN
        else:
            direction = Direction.UP
        goals.append(
            PlaceMachineAt(
                MachineType.CONVEYOR_BELT,
                tile,
                int(direction),
            ),
        )
    if sink in seen:
        raise ValueError(f"sink {sink} already on belt path")
    return goals


# Per-item bootstrap cost of one smelter cell built via
# :func:`build_smelter_cell_at`. Two pallets (coal-buffer south + plate
# bus to one side), one cell arm, one furnace; extras are added by the
# kwargs (``with_extractor``, ``output_split``).
_SMELTER_CELL_INVENTORY: tuple[tuple[int, int], ...] = (
    (int(ItemType.PALLET), 2),
    (int(ItemType.ARM), 1),
    (int(ItemType.FURNACE), 1),
)


def smelter_cell_inventory(
    *,
    with_extractor: bool = False,
    output_split: bool = False,
    automation_belt: bool = True,
) -> dict[int, int]:
    """Return the bootstrap inventory cost of one smelter cell.

    Maps :class:`ItemType` integer ids to the count required in the
    player's inventory before issuing the goals from
    :func:`build_smelter_cell_at`. Use this to size the Phase-A craft
    list (e.g. ``ProduceInAssembler(ItemType.PALLET, 2 * num_cells)``)
    so a typo in the per-cell counts surfaces as a one-place change.

    Args:
        with_extractor: When ``True``, accounts for the extra
            extractor arm that :func:`build_smelter_cell_at` places
            when its ``extract_facing`` argument is set — adds one
            more ARM to the cost.
        output_split: When ``True``, the plate_bus PALLET is replaced
            by a SPLITTER feeding a manual-stash PALLET on the cell's
            north output and (by default) an automation BELT on the
            south output. Net per cell: same 2 PALLETs (was
            coal_buffer + plate_bus, now coal_buffer + manual_stash),
            +1 SPLITTER, +1 CONVEYOR_BELT vs the canonical layout.
        automation_belt: Only meaningful with ``output_split=True``.
            When ``False``, the cell helper *does not* emit the
            automation belt at the splitter's south output — the
            caller is responsible for placing whatever consumes the
            splitter's S output (typically a CROSSING emitted by
            :func:`place_belt_network` because a coal trunk shares
            that tile). Drops the +1 CONVEYOR_BELT from the
            inventory.

    Raises:
        ValueError: ``output_split`` and ``with_extractor`` together —
            the splitter's south output already drives the automation
            lane, no extractor is needed. Or ``automation_belt=False``
            without ``output_split=True`` (the kwarg only makes sense
            with the splitter mode).
    """
    if output_split and with_extractor:
        raise ValueError(
            "smelter_cell_inventory: output_split and with_extractor are "
            "mutually exclusive (the splitter replaces the extractor)",
        )
    if not automation_belt and not output_split:
        raise ValueError(
            "smelter_cell_inventory: automation_belt=False is only valid "
            "with output_split=True (it controls whether the splitter's "
            "south-output belt is emitted)",
        )
    cost = dict(_SMELTER_CELL_INVENTORY)
    if with_extractor:
        cost[int(ItemType.ARM)] += 1
    if output_split:
        cost[int(ItemType.SPLITTER)] = 1
        if automation_belt:
            cost[int(ItemType.CONVEYOR_BELT)] = 1
    return cost


_HORIZONTAL_FACINGS: frozenset[int] = frozenset(
    {int(Direction.LEFT), int(Direction.RIGHT)}
)
_DIRECTION_OFFSETS: dict[int, tuple[int, int]] = {
    int(Direction.UP): (0, -1),
    int(Direction.DOWN): (0, 1),
    int(Direction.LEFT): (-1, 0),
    int(Direction.RIGHT): (1, 0),
}


def build_smelter_cell_at(
    furnace_tile: tuple[int, int],
    *,
    facing: int = int(Direction.RIGHT),
    extract_facing: int | None = None,
    output_split: bool = False,
    automation_belt: bool = True,
    occupied: set[tuple[int, int]] | None = None,
    map_size: tuple[int, int] | None = None,
) -> list[Goal]:
    """Place a smelter cell anchored at ``furnace_tile``.

    Layout for the default ``facing=RIGHT`` (the one used by the
    rocket benchmark's iron, tin, and silicon cells)::

           ore_pallet  (fx,   fy-1)  pre-existing, NOT placed here
                furnace(fx,   fy)    facing RIGHT
                  arm  (fx+1, fy)    facing RIGHT
                plate_bus(fx+2, fy)  facing DOWN — output sink
        coal_buffer    (fx,   fy+1)  facing UP   — coal trunk delivers here

    Setting ``facing=LEFT`` mirrors the cell across the y-axis (the
    arm and plate-bus move to the *west* of the furnace). The rocket
    benchmark's copper cell uses this orientation so its plate
    output sits on the same row 11 as iron's, between the two
    patches, instead of east of the copper patch where it would
    require routing belts back across the map.

    When ``extract_facing`` is set, an additional extractor arm is
    woven into the placement order: it is placed *between* the
    coal_buffer and the plate_bus so that its stand tile (= the
    plate_bus's eventual location) is still walkable dirt at the
    moment of placement. Once the plate_bus lands afterward, the
    extractor's "behind" tile becomes the bus and its
    :func:`run_arms` cycle pulls one plate per tick out of the bus
    and pushes onto the tile in ``extract_facing`` from the bus —
    typically the first belt of a downstream plate trunk.

    When ``output_split=True`` the plate_bus PALLET is swapped for a
    SPLITTER at the same tile (facing the cell's ``facing`` so its
    input is the arm-side west/east face). The splitter's two
    perpendicular outputs feed:

    * **manual_stash** — a PALLET to the *north* of the splitter
      (smaller y), where the player-agent withdraws plates for
      hand-crafting.
    * **automation_belt** — a CONVEYOR_BELT to the *south* of the
      splitter (larger y), facing DOWN, that carries plates onward
      to a downstream :func:`place_belt_path` /
      :func:`place_belt_network` route.

    With ``output_split=True``, the cell emits 6 placements instead
    of 4. Placement order: coal_buffer → manual_stash →
    automation_belt → splitter → arm → furnace, so every stand tile
    is dirt or an already-placed walkable belt at the moment of
    placement. ``extract_facing`` is rejected in this mode (the
    splitter is the extractor).

    The ore pallet to the north is assumed to be placed separately
    (Phase A's auto-miner pattern in the rocket benchmark): the
    furnace's :func:`run_assemblers` Phase 0 auto-pulls one ore per
    tick from that pallet and one coal per tick from the coal buffer
    south, so no extra arm orchestration is needed for the inputs.

    The arm is placed *before* the furnace because the arm's stand
    tile (one tile back from the arm in ``facing``) is exactly the
    furnace's eventual tile — at arm-place time that tile must be
    walkable dirt, and once the furnace lands it becomes the arm's
    "behind" tile so :func:`run_arms` can pull plates out of the
    furnace's ``ent_asm_out``.

    Args:
        furnace_tile: ``(fx, fy)`` where the furnace will land. The
            cell-arm and plate-bus tiles are derived in
            ``facing``; the coal-buffer always sits south of the
            furnace.
        facing: ``Direction.RIGHT`` (default) or ``Direction.LEFT``
            — the cell's plate-output side. Other directions raise
            ``ValueError`` (UP/DOWN would put the plate-bus on the
            ore-pallet column or the coal-buffer column, both
            non-walkable in the rocket benchmark layout).
        extract_facing: Optional one of UP/DOWN/LEFT/RIGHT. When
            set, places an extractor arm at
            ``plate_bus + unit(extract_facing)`` facing
            ``extract_facing``, pulling plates out of the bus and
            pushing them onto the tile two further out. Must not be
            the *opposite* of ``facing`` (that would put the
            extractor on top of the cell-arm). Mutually exclusive
            with ``output_split``.
        output_split: When ``True``, swap the plate_bus PALLET for a
            SPLITTER + manual_stash PALLET (north) +
            automation_belt CONVEYOR_BELT (south, suppressible via
            ``automation_belt=False``). Mutually exclusive with
            ``extract_facing``.
        automation_belt: Only meaningful with ``output_split=True``.
            When ``False``, the cell helper does *not* emit the
            CONVEYOR_BELT at the splitter's south output — the caller
            must place whatever consumes the splitter's S output
            (typically a CROSSING placed by
            :func:`place_belt_network` because a coal trunk shares
            that tile). Drops one CONVEYOR_BELT from the inventory.
        occupied: Optional set of tiles already taken. Every cell
            tile (and the extractor when present) is checked.
        map_size: Optional ``(width, height)`` for in-bounds checks.

    Returns:
        4 :class:`PlaceMachineAt` goals by default (5 with
        ``extract_facing``, 5 with ``output_split=True,
        automation_belt=False``, 6 with ``output_split=True``
        default), in placement order. See
        :func:`smelter_cell_inventory` for the matching bootstrap
        inventory.

    Raises:
        ValueError: ``facing`` is not LEFT or RIGHT; ``extract_facing``
            is not a cardinal direction; ``extract_facing`` is the
            opposite of ``facing``; ``extract_facing`` and
            ``output_split`` are both set; ``automation_belt=False``
            without ``output_split=True``; any tile is out of bounds,
            in ``occupied``, or duplicates another piece.
    """
    if output_split and extract_facing is not None:
        raise ValueError(
            "build_smelter_cell_at: output_split is mutually exclusive with "
            "extract_facing (the splitter's south output replaces the "
            "extractor arm)",
        )
    if not automation_belt and not output_split:
        raise ValueError(
            "build_smelter_cell_at: automation_belt=False is only valid "
            "with output_split=True",
        )
    if int(facing) not in _HORIZONTAL_FACINGS:
        raise ValueError(
            f"build_smelter_cell_at facing must be LEFT or RIGHT, got {facing}",
        )
    facing = int(facing)
    side = 1 if facing == int(Direction.RIGHT) else -1

    fx, fy = furnace_tile
    arm_tile = (fx + side, fy)
    plate_bus_tile = (fx + 2 * side, fy)
    coal_buffer_tile = (fx, fy + 1)

    pieces: list[tuple[str, tuple[int, int]]] = [
        ("furnace", furnace_tile),
        ("arm", arm_tile),
        ("plate_bus", plate_bus_tile),
        ("coal_buffer", coal_buffer_tile),
    ]

    manual_stash_tile: tuple[int, int] | None = None
    automation_belt_tile: tuple[int, int] | None = None
    if output_split:
        # plate_bus tile is reused as the SPLITTER tile; the manual
        # stash hangs north of it; the automation belt (when emitted)
        # hangs south.
        manual_stash_tile = (plate_bus_tile[0], plate_bus_tile[1] - 1)
        pieces.append(("manual_stash", manual_stash_tile))
        if automation_belt:
            automation_belt_tile = (plate_bus_tile[0], plate_bus_tile[1] + 1)
            pieces.append(("automation_belt", automation_belt_tile))

    extractor_tile: tuple[int, int] | None = None
    if extract_facing is not None:
        extract_facing_int = int(extract_facing)
        if extract_facing_int not in _DIRECTION_OFFSETS:
            raise ValueError(
                f"extract_facing must be UP/DOWN/LEFT/RIGHT, got {extract_facing}",
            )
        if extract_facing_int == int(Direction.LEFT) and side == 1:
            opposite_of_facing = True
        elif extract_facing_int == int(Direction.RIGHT) and side == -1:
            opposite_of_facing = True
        else:
            opposite_of_facing = False
        if opposite_of_facing:
            raise ValueError(
                f"extract_facing {extract_facing} is the opposite of "
                f"facing {facing}; the extractor would land on the "
                f"cell arm",
            )
        edx, edy = _DIRECTION_OFFSETS[extract_facing_int]
        extractor_tile = (
            plate_bus_tile[0] + edx,
            plate_bus_tile[1] + edy,
        )
        pieces.append(("extractor", extractor_tile))

    if map_size is not None:
        width, height = map_size
        for label, (x, y) in pieces:
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(
                    f"smelter cell {label} tile {(x, y)} is out of "
                    f"map bounds (width={width}, height={height})",
                )

    seen: dict[tuple[int, int], str] = {}
    for label, tile in pieces:
        if tile in seen:
            raise ValueError(
                f"smelter cell {label} tile {tile} duplicates "
                f"{seen[tile]} (check furnace_tile + facing)",
            )
        seen[tile] = label

    if occupied is not None:
        blocked = frozenset(occupied)
        for label, tile in pieces:
            if tile in blocked:
                raise ValueError(
                    f"smelter cell {label} tile {tile} collides with an occupied tile",
                )

    goals: list[Goal] = [
        PlaceMachineAt(
            MachineType.PALLET,
            coal_buffer_tile,
            int(Direction.UP),
        ),
    ]
    if output_split:
        # manual_stash and automation_belt before the splitter so the
        # splitter's stand tile (the arm's eventual location, west or
        # east of it) is still the only walkable adjacency we depend on
        # — the splitter itself never has to stand on either output
        # neighbour. manual_stash facing DOWN and automation_belt
        # facing DOWN share the convention with plate_bus (which faces
        # DOWN today): both have their stand tile on the south side,
        # which is dirt at place time.
        goals.append(
            PlaceMachineAt(
                MachineType.PALLET,
                manual_stash_tile,
                int(Direction.DOWN),
            ),
        )
        if automation_belt:
            goals.append(
                PlaceMachineAt(
                    MachineType.CONVEYOR_BELT,
                    automation_belt_tile,
                    int(Direction.DOWN),
                ),
            )
        goals.append(
            PlaceMachineAt(
                MachineType.SPLITTER,
                plate_bus_tile,
                facing,
            ),
        )
    else:
        if extractor_tile is not None:
            # Extractor arm before the bus pallet — its stand tile is
            # the bus's eventual location, currently dirt.
            goals.append(
                PlaceMachineAt(
                    MachineType.ARM,
                    extractor_tile,
                    int(extract_facing),
                ),
            )
        goals.append(
            PlaceMachineAt(
                MachineType.PALLET,
                plate_bus_tile,
                int(Direction.DOWN),
            ),
        )
    goals.extend(
        [
            PlaceMachineAt(MachineType.ARM, arm_tile, facing),
            PlaceMachineAt(MachineType.FURNACE, furnace_tile, facing),
        ]
    )
    return goals


# Per-item bootstrap cost of one 2-input assembler module placed via
# :func:`build_assembler_module_at`: 1 assembler + 3 pallets
# (input_a, input_b, output) + 1 arm (east-side plate extractor).
_ASSEMBLER_MODULE_INVENTORY: tuple[tuple[int, int], ...] = (
    (int(ItemType.ASSEMBLER), 1),
    (int(ItemType.PALLET), 3),
    (int(ItemType.ARM), 1),
)


def assembler_module_inventory(*, input_b: bool = True) -> dict[int, int]:
    """Return the bootstrap inventory cost of one assembler module.

    Args:
        input_b: When ``True`` (default), accounts for both north and
            west input pallets (3 pallets total: input_a, input_b,
            output). When ``False``, drops one pallet for a 1-input
            module.
    """
    cost = dict(_ASSEMBLER_MODULE_INVENTORY)
    if not input_b:
        cost[int(ItemType.PALLET)] -= 1
    return cost


def build_assembler_module_at(
    center_tile: tuple[int, int],
    *,
    input_b: bool = True,
    occupied: set[tuple[int, int]] | None = None,
    map_size: tuple[int, int] | None = None,
) -> list[Goal]:
    """Place a 1- or 2-input assembler production module.

    Layout (top-down, ``(cx, cy) = center_tile``)::

                       input_a   (cx,   cy-1)  facing DOWN
        input_b Center  Arm  Output  (cx-1..cx+2, cy)
                          .          (cx,   cy+1)  free

    Engine semantics relied on:

    - The assembler at ``center_tile`` auto-pulls from any adjacent
      buffer in :func:`run_assemblers` Phase 0, matching empty /
      same-type input slots. With pallets at the north and west
      neighbours both inputs land in ``ent_asm_in`` without arm
      orchestration; downstream goals can deposit into either pallet
      manually or feed them via belts ending in those tiles.
    - The east-facing arm reads from the assembler's ``ent_asm_out``
      and pushes east into the output pallet's ``ent_buf``.

    Placement order (output -> input_b -> arm -> center -> input_a)
    keeps every stand tile walkable. Input_a is placed last because
    its stand tile (one tile north of input_a) was unaffected by the
    earlier placements; the center is placed *before* input_a since
    the center's stand tile (one tile north = input_a's eventual
    location) must be dirt at center-place time.

    Args:
        center_tile: ``(cx, cy)`` for the assembler. The four other
            module tiles are derived around it.
        input_b: When ``True`` (default), place the west input
            pallet for a 2-input recipe (e.g. CONVEYOR_BELT needs
            IRON_PLATE + COPPER_PLATE). Set ``False`` for 1-input
            recipes; only one input pallet is placed.
        occupied: Optional set of tiles already taken; each module
            tile is checked.
        map_size: Optional ``(width, height)`` for in-bounds checks.

    Returns:
        A list of 4 (1-input) or 5 (2-input) :class:`PlaceMachineAt`
        goals in placement order. See
        :func:`assembler_module_inventory` for the bootstrap cost.

    Raises:
        ValueError: any module tile is out of bounds, in
            ``occupied``, or duplicates another piece.
    """
    cx, cy = center_tile
    input_a_tile = (cx, cy - 1)
    arm_tile = (cx + 1, cy)
    output_tile = (cx + 2, cy)
    input_b_tile = (cx - 1, cy)

    pieces: list[tuple[str, tuple[int, int]]] = [
        ("center", center_tile),
        ("input_a", input_a_tile),
        ("arm", arm_tile),
        ("output", output_tile),
    ]
    if input_b:
        pieces.append(("input_b", input_b_tile))

    if map_size is not None:
        width, height = map_size
        for label, (x, y) in pieces:
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(
                    f"assembler module {label} tile {(x, y)} is out "
                    f"of map bounds (width={width}, height={height})",
                )

    seen: dict[tuple[int, int], str] = {}
    for label, tile in pieces:
        if tile in seen:
            raise ValueError(
                f"assembler module {label} tile {tile} duplicates "
                f"{seen[tile]} (check center_tile)",
            )
        seen[tile] = label

    if occupied is not None:
        blocked = frozenset(occupied)
        for label, tile in pieces:
            if tile in blocked:
                raise ValueError(
                    f"assembler module {label} tile {tile} collides "
                    f"with an occupied tile",
                )

    goals: list[Goal] = [
        PlaceMachineAt(
            MachineType.PALLET,
            output_tile,
            int(Direction.DOWN),
        ),
    ]
    if input_b:
        goals.append(
            PlaceMachineAt(
                MachineType.PALLET,
                input_b_tile,
                int(Direction.LEFT),
            ),
        )
    goals.extend(
        [
            PlaceMachineAt(
                MachineType.ARM,
                arm_tile,
                int(Direction.RIGHT),
            ),
            PlaceMachineAt(
                MachineType.ASSEMBLER,
                center_tile,
                int(Direction.DOWN),
            ),
            PlaceMachineAt(
                MachineType.PALLET,
                input_a_tile,
                int(Direction.DOWN),
            ),
        ]
    )
    return goals


# Per-item bootstrap cost of one ore-extraction node placed via
# :func:`place_ore_node_at`. The pallet is optional (set
# ``with_pallet=False`` for the rocket benchmark's coal node, which
# pushes directly onto a belt instead of a buffer pallet); when
# omitted, only the miner is needed.
_ORE_NODE_INVENTORY_WITH_PALLET: tuple[tuple[int, int], ...] = (
    (int(ItemType.MINER), 1),
    (int(ItemType.PALLET), 1),
)
_ORE_NODE_INVENTORY_MINER_ONLY: tuple[tuple[int, int], ...] = (
    (int(ItemType.MINER), 1),
)


def ore_node_inventory(*, with_pallet: bool = True) -> dict[int, int]:
    """Return the bootstrap inventory cost of one ore-extraction node.

    Maps :class:`ItemType` integer ids to the count required in the
    player's inventory before issuing the goals from
    :func:`place_ore_node_at`. Pass ``with_pallet=False`` for the
    rocket benchmark's coal node, where the miner pushes onto the
    coal trunk's first belt instead of into a holding pallet.
    """
    src = (
        _ORE_NODE_INVENTORY_WITH_PALLET
        if with_pallet
        else _ORE_NODE_INVENTORY_MINER_ONLY
    )
    return dict(src)


def place_ore_node(
    miner_tile: tuple[int, int],
    *,
    direction: int = int(Direction.DOWN),
    with_pallet: bool = True,
    occupied: set[tuple[int, int]] | None = None,
    map_size: tuple[int, int] | None = None,
) -> list[Goal]:
    """Place an auto-mining node: a miner and (optionally) a pallet.

    The miner is placed on ``miner_tile`` facing ``direction``; its
    per-tick output pushes one tile in that direction. When
    ``with_pallet=True`` (default), a pallet is placed on the push
    tile to buffer the output; downstream goals (a smelter cell, a
    withdraw goal, a coal trunk) drain it from there. When
    ``with_pallet=False``, no pallet is placed — useful when the
    miner pushes directly onto a belt, as the rocket benchmark's
    coal node does to feed the iron trunk.

    Bootstrap inventory: ``1 MINER + 1 PALLET`` (or just ``1 MINER``
    when ``with_pallet=False``). See :func:`ore_node_inventory`.

    Placement order: pallet first (if requested), then miner. With
    a DOWN-facing node on a 3x3 ore patch, the pallet's stand tile
    is two tiles south of the patch (dirt) and the miner's stand
    tile is one tile back into the patch interior (walkable ore),
    so placing the pallet first never invalidates the miner's stand
    tile.

    Args:
        miner_tile: ``(x, y)`` for the miner. For the rocket
            benchmark's south-edge pattern this is
            ``(patch_x + 1, patch_y + 2)`` — the south-edge
            centre of a 3x3 patch starting at ``(patch_x, patch_y)``.
        direction: Push direction. Must be one of ``Direction.UP``,
            ``DOWN``, ``LEFT``, ``RIGHT``. The miner faces this
            direction and pushes one tile that way each tick; the
            pallet (if any) sits on that adjacent tile.
        with_pallet: When ``True`` (default), place a pallet on the
            miner's push tile. When ``False``, only place the miner.
        occupied: Optional set of tiles already occupied. The miner
            tile and (when applicable) the pallet tile are checked
            against it; the first collision raises ``ValueError``
            with the offending tile and its role.
        map_size: Optional ``(width, height)``; both placement tiles
            must fit inside ``[0, width) x [0, height)``.

    Returns:
        A list of one or two :class:`PlaceMachineAt` goals
        (pallet then miner if ``with_pallet=True``, else just miner).

    Raises:
        ValueError: ``direction`` is not one of UP/DOWN/LEFT/RIGHT;
            either placement tile is out of ``map_size`` bounds or
            in ``occupied``.
    """
    direction_to_offset = {
        int(Direction.UP): (0, -1),
        int(Direction.DOWN): (0, 1),
        int(Direction.LEFT): (-1, 0),
        int(Direction.RIGHT): (1, 0),
    }
    if int(direction) not in direction_to_offset:
        raise ValueError(
            f"direction must be UP/DOWN/LEFT/RIGHT, got {direction}",
        )
    direction = int(direction)
    dx, dy = direction_to_offset[direction]
    pallet_tile = (miner_tile[0] + dx, miner_tile[1] + dy)

    pieces: list[tuple[str, tuple[int, int]]] = [("miner", miner_tile)]
    if with_pallet:
        pieces.append(("pallet", pallet_tile))

    if map_size is not None:
        width, height = map_size
        for label, (x, y) in pieces:
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(
                    f"ore node {label} tile {(x, y)} is out of "
                    f"map bounds (width={width}, height={height})",
                )

    if occupied is not None:
        blocked = frozenset(occupied)
        for label, tile in pieces:
            if tile in blocked:
                raise ValueError(
                    f"ore node {label} tile {tile} collides with an occupied tile",
                )

    goals: list[Goal] = []
    if with_pallet:
        goals.append(
            PlaceMachineAt(MachineType.PALLET, pallet_tile, direction),
        )
    goals.append(PlaceMachineAt(MachineType.MINER, miner_tile, direction))
    return goals


def wire_coal_feed(
    *,
    miner_tile: tuple[int, int],
    miner_facing: int,
    trunk_waypoints: list[tuple[int, int]],
    occupied: set[tuple[int, int]] | None = None,
    map_size: tuple[int, int] | None = None,
) -> list[Goal]:
    """Place a coal miner and the belt trunk it feeds.

    Combines :func:`place_ore_node` (with no pallet — the miner
    pushes onto the trunk's first belt instead of into a buffer) and
    :func:`place_belt_path` (the trunk itself), in the order that
    keeps every stand tile walkable when the corresponding placement
    happens.

    The miner pushes one tile per tick into ``miner_tile +
    unit(miner_facing)``, which must equal ``trunk_waypoints[0]`` —
    otherwise the miner's output would land on dirt instead of the
    trunk and the helper raises ``ValueError`` so the typo surfaces
    synchronously.

    Placement order: if the trunk's first belt's stand tile equals
    ``miner_tile``, the trunk is placed *before* the miner (placing
    the miner first would block the belt's only legal stand tile).
    Otherwise the miner is placed first so it starts pushing as
    soon as the first belt arrives. The two cases that hit the
    trunk-first branch in the rocket benchmark are tin's
    ``(7, 24) DOWN`` miner with trunk starting at ``(7, 25) DOWN``
    (belt stand = ``(7, 24)``) and silicon's ``(7, 22) UP`` miner
    with trunk starting at ``(7, 21) UP`` (belt stand = ``(7, 22)``).

    Args:
        miner_tile: ``(x, y)`` for the coal miner.
        miner_facing: One of ``Direction.UP/DOWN/LEFT/RIGHT``. The
            miner pushes one tile in this direction each tick.
        trunk_waypoints: Path from the miner's push tile through any
            corners to the sink (typically a coal-buffer pallet
            placed elsewhere). See :func:`place_belt_path` for the
            waypoint semantics.
        occupied: Optional set of tiles already occupied. Both the
            miner tile and every non-sink trunk tile are checked.
        map_size: Optional ``(width, height)`` for in-bounds checks.

    Returns:
        A list of :class:`PlaceMachineAt` goals: one for the miner
        plus ``len(trunk) - 1`` for the belts, in placement order.

    Raises:
        ValueError: ``miner_facing`` is not a cardinal direction;
            ``trunk_waypoints[0]`` does not equal the miner's push
            tile; any tile is out of bounds; any tile collides with
            ``occupied``; or the trunk itself is malformed (see
            :func:`place_belt_path`).
    """
    direction_to_offset = {
        int(Direction.UP): (0, -1),
        int(Direction.DOWN): (0, 1),
        int(Direction.LEFT): (-1, 0),
        int(Direction.RIGHT): (1, 0),
    }
    facing = int(miner_facing)
    if facing not in direction_to_offset:
        raise ValueError(
            f"miner_facing must be UP/DOWN/LEFT/RIGHT, got {miner_facing}",
        )
    if len(trunk_waypoints) < 2:
        raise ValueError(
            "wire_coal_feed needs at least 2 trunk waypoints (start + sink)",
        )

    push_dx, push_dy = direction_to_offset[facing]
    push_tile = (miner_tile[0] + push_dx, miner_tile[1] + push_dy)
    if push_tile != trunk_waypoints[0]:
        raise ValueError(
            f"miner at {miner_tile} facing {miner_facing} pushes onto "
            f"{push_tile}, but trunk starts at {trunk_waypoints[0]}",
        )

    second = trunk_waypoints[1]
    if second[0] == trunk_waypoints[0][0]:
        first_belt_dy = 1 if second[1] > trunk_waypoints[0][1] else -1
        first_belt_dx = 0
    else:
        first_belt_dx = 1 if second[0] > trunk_waypoints[0][0] else -1
        first_belt_dy = 0
    first_belt_stand = (
        trunk_waypoints[0][0] - first_belt_dx,
        trunk_waypoints[0][1] - first_belt_dy,
    )
    miner_blocks_first_belt = first_belt_stand == miner_tile

    # Validate the miner tile against occupied / map bounds eagerly
    # (place_ore_node handles this internally too, but we also need
    # the check before the trunk's collision pass).
    miner_goals = place_ore_node(
        miner_tile,
        direction=facing,
        with_pallet=False,
        occupied=occupied,
        map_size=map_size,
    )

    trunk_occupied: set[tuple[int, int]] = (
        set(occupied) if occupied is not None else set()
    )
    if not miner_blocks_first_belt:
        # The miner is placed before the trunk in this branch, so
        # the trunk must avoid the miner tile. Treat it as occupied
        # for the trunk's collision pass — even when the caller
        # didn't supply an ``occupied`` argument, since the miner
        # itself is a known obstacle once it lands.
        trunk_occupied.add(miner_tile)
    trunk_goals = place_belt_path(
        trunk_waypoints,
        occupied=trunk_occupied,
    )

    if map_size is not None:
        width, height = map_size
        for tile in trunk_waypoints:
            x, y = tile
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(
                    f"coal trunk waypoint {tile} is out of map "
                    f"bounds (width={width}, height={height})",
                )

    if miner_blocks_first_belt:
        return [*trunk_goals, *miner_goals]
    return [*miner_goals, *trunk_goals]


def coal_feed_inventory(
    trunk_waypoints: list[tuple[int, int]],
) -> dict[int, int]:
    """Return the bootstrap inventory cost of one coal feed.

    A coal feed is one MINER plus one CONVEYOR_BELT per non-sink
    tile in the trunk. The trunk's tile count is computed from the
    waypoint list the same way :func:`place_belt_path` does, so this
    helper and the placement helper always agree on the belt count.
    """
    if len(trunk_waypoints) < 2:
        raise ValueError(
            "coal_feed_inventory needs at least 2 waypoints",
        )
    # Total path length in tiles is 1 + sum of segment Manhattan
    # distances (start tile + one tile per step). The last tile is
    # the sink — no belt placed there — so the belt count is exactly
    # the sum of distances.
    belts = sum(
        abs(trunk_waypoints[i + 1][0] - trunk_waypoints[i][0])
        + abs(trunk_waypoints[i + 1][1] - trunk_waypoints[i][1])
        for i in range(len(trunk_waypoints) - 1)
    )
    return {
        int(ItemType.MINER): 1,
        int(ItemType.CONVEYOR_BELT): belts,
    }


def _machine_to_item(machine_type: int) -> int:
    """Item type corresponding to a placeable machine."""
    return {
        int(MachineType.MINER): int(ItemType.MINER),
        int(MachineType.PALLET): int(ItemType.PALLET),
        int(MachineType.CONVEYOR_BELT): int(ItemType.CONVEYOR_BELT),
        int(MachineType.ASSEMBLER): int(ItemType.ASSEMBLER),
        int(MachineType.ARM): int(ItemType.ARM),
        int(MachineType.ROCKET): int(ItemType.ROCKET),
        int(MachineType.FURNACE): int(ItemType.FURNACE),
        int(MachineType.SPLITTER): int(ItemType.SPLITTER),
        int(MachineType.CROSSING): int(ItemType.CROSSING),
    }[machine_type]


# ---------------------------------------------------------------------------
# Interaction with placed machines
# ---------------------------------------------------------------------------


class DepositInto(Goal):
    """Deposit one unit of *item_type* into the nearest *machine_type*.

    Used for loading pallets (to unlock ``pallet_filled``), loading
    assemblers with inputs, or feeding a furnace. The goal succeeds
    after a single successful DEPOSIT; repeat the goal to load more.
    """

    name = "DepositInto"

    def __init__(
        self,
        machine_type: int | MachineType,
        item_type: int | ItemType,
    ) -> None:
        self.machine_type = int(machine_type)
        self.item_type = int(item_type)
        self._active: FaceAndInteract | None = None
        self._start_inv: int | None = None

    def step(self, view: WorldView) -> StepReturn:
        current_inv = view.player.held(self.item_type)

        if self._start_inv is not None and current_inv < self._start_inv:
            # Inventory went down → deposit succeeded.
            return Result.DONE, None

        if current_inv < 1:
            return Result.FAIL, None

        if self._active is None:
            tiles = view.tiles_with_machine(self.machine_type)
            if not tiles:
                return Result.FAIL, None
            px, py = view.player.pos
            tiles.sort(key=lambda t: abs(t[0] - px) + abs(t[1] - py))
            target = tiles[0]
            self._start_inv = current_inv
            from .world_model import deposit_action  # noqa: PLC0415

            self._active = FaceAndInteract(target, deposit_action(self.item_type))

        result, action = self._active.step(view)
        if result is Result.DONE:
            # FaceAndInteract finished emitting its action; next tick will
            # detect the inventory drop and report DONE itself.
            self._active = None
            return Result.RUNNING, int(Action.NOOP)
        if result is Result.FAIL:
            self._active = None
            return Result.FAIL, None
        return Result.RUNNING, action


class DepositIntoAt(Goal):
    """Deposit ``count`` units of *item_type* into a specific pallet tile.

    Differs from :class:`DepositInto` in two ways:

    1. **Targets a specific tile** — the planner controls the bus
       coordinates, so an explicit tile avoids the ambiguity of
       "nearest matching pallet" when multiple pallets exist.
    2. **Bulk** — repeats the face-and-deposit cycle ``count`` times
       or until the player runs out of the item. Each deposit
       transfers one item per tick (engine cap).

    Counting is done via the player's inventory drop, not the
    pallet's current load. When the pallet sits next to a furnace
    the furnace's ``run_assemblers`` Phase 0 drains the pallet at
    one item per tick, so the pallet's buffer count never actually
    grows past zero — but the deposits still flow through. Tracking
    "items I successfully shed" instead of "items in the pallet"
    captures the throughput correctly.

    Args:
        tile: ``(x, y)`` of the target pallet.
        item_type: Item to deposit (used for the
            ``DEPOSIT_<item>`` action selector).
        count: Number of successful deposits to execute. Default 1.
    """

    name = "DepositIntoAt"

    def __init__(
        self,
        tile: tuple[int, int],
        item_type: int | ItemType,
        count: int = 1,
    ) -> None:
        self.tile = tile
        self.item_type = int(item_type)
        self.count = count
        self._active: FaceAndInteract | None = None
        self._start_inv: int | None = None

    def step(self, view: WorldView) -> StepReturn:
        x, y = self.tile
        if view.machine_type[y, x] != int(MachineType.PALLET):
            return Result.FAIL, None

        current_inv = view.player.held(self.item_type)
        if self._start_inv is None:
            self._start_inv = current_inv
        deposited = self._start_inv - current_inv
        if deposited >= self.count:
            return Result.DONE, None

        if current_inv < 1:
            return Result.FAIL, None

        if self._active is not None:
            result, action = self._active.step(view)
            if result is Result.RUNNING:
                return Result.RUNNING, action
            self._active = None
            return Result.RUNNING, int(Action.NOOP)

        from .world_model import deposit_action  # noqa: PLC0415

        self._active = FaceAndInteract(self.tile, deposit_action(self.item_type))
        return self._active.step(view)


class WaitUntil(Goal):
    """Emit ``NOOP`` until *predicate* returns True or the budget expires.

    Useful for waiting on timed game events — ``automated_mining``
    fires one tick after a miner is placed on ore; ``first_assembly``
    fires after the assembler's recipe timer completes.
    """

    name = "WaitUntil"

    def __init__(
        self,
        predicate: Callable[[WorldView], bool],
        max_ticks: int,
    ) -> None:
        self.predicate = predicate
        self.max_ticks = max_ticks
        self._ticks = 0

    def step(self, view: WorldView) -> StepReturn:
        if self.predicate(view):
            return Result.DONE, None
        if self._ticks >= self.max_ticks:
            return Result.FAIL, None
        self._ticks += 1
        return Result.RUNNING, int(Action.NOOP)


# ---------------------------------------------------------------------------
# Machine-based production (furnace / assembler)
# ---------------------------------------------------------------------------


class Wait(Goal):
    """Emit ``NOOP`` for exactly *ticks* then report DONE.

    Used between deposit and withdraw inside a machine production
    cycle so the recipe has time to complete.
    """

    name = "Wait"

    def __init__(self, ticks: int) -> None:
        self.ticks = ticks
        self._elapsed = 0

    def step(self, view: WorldView) -> StepReturn:  # noqa: ARG002
        if self._elapsed >= self.ticks:
            return Result.DONE, None
        self._elapsed += 1
        return Result.RUNNING, int(Action.NOOP)


class WithdrawUntilHeld(Goal):
    """Withdraw from adjacent pallets until the player holds >= count.

    Finds the nearest :class:`MachineType.PALLET` whose buffered item
    matches ``item_type``, navigates adjacent, emits ``WITHDRAW``.
    With the engine's bulk-withdraw semantics (one action transfers
    the whole slot up to inventory capacity), this usually clears a
    pallet in a single visit — a 45-ore pallet drain becomes one
    action instead of 45. Retries with fresh pallet lookups so if the
    first pallet is empty the agent moves to the next.
    """

    name = "WithdrawUntilHeld"

    def __init__(
        self,
        item_type: int | ItemType,
        count: int,
        max_empty_attempts: int = 8,
    ) -> None:
        self.item_type = int(item_type)
        self.count = count
        self.max_empty_attempts = max_empty_attempts
        self._active: FaceAndInteract | None = None
        self._empty_attempts = 0

    def step(self, view: WorldView) -> StepReturn:
        if view.player.held(self.item_type) >= self.count:
            return Result.DONE, None

        if self._active is not None:
            result, action = self._active.step(view)
            if result is Result.RUNNING:
                return Result.RUNNING, action
            self._active = None

        # Find the nearest pallet that currently holds item_type.
        buffered = view.buffered_tiles(self.item_type)
        pallet_tiles = [
            (x, y)
            for (x, y) in buffered
            if view.machine_type[y, x] == int(MachineType.PALLET)
        ]
        if not pallet_tiles:
            self._empty_attempts += 1
            if self._empty_attempts >= self.max_empty_attempts:
                return Result.FAIL, None
            return Result.RUNNING, int(Action.NOOP)

        # Reset empty counter on any productive step.
        self._empty_attempts = 0
        px, py = view.player.pos
        pallet_tiles.sort(key=lambda t: abs(t[0] - px) + abs(t[1] - py))
        target = pallet_tiles[0]
        self._active = FaceAndInteract(target, int(Action.WITHDRAW))
        return self._active.step(view)


class WithdrawFrom(Goal):
    """Withdraw one unit of *item_type* from the nearest *machine_type*.

    Retries up to *max_attempts* times if the first withdraw doesn't
    actually move anything (e.g. recipe hasn't finished yet). Succeeds
    when the player's inventory of *item_type* grows.
    """

    name = "WithdrawFrom"

    def __init__(
        self,
        machine_type: int | MachineType,
        item_type: int | ItemType,
        max_attempts: int = 6,
    ) -> None:
        self.machine_type = int(machine_type)
        self.item_type = int(item_type)
        self.max_attempts = max_attempts
        self._active: FaceAndInteract | None = None
        self._start_inv: int | None = None
        self._attempts = 0

    def step(self, view: WorldView) -> StepReturn:
        current_inv = view.player.held(self.item_type)
        if self._start_inv is not None and current_inv > self._start_inv:
            return Result.DONE, None

        if self._active is not None:
            result, action = self._active.step(view)
            if result is Result.DONE:
                # Emitted the WITHDRAW action; next tick we check inventory.
                self._active = None
                return Result.RUNNING, int(Action.NOOP)
            if result is Result.FAIL:
                self._active = None
                return Result.FAIL, None
            return Result.RUNNING, action

        if self._attempts >= self.max_attempts:
            return Result.FAIL, None

        tiles = view.tiles_with_machine(self.machine_type)
        if not tiles:
            return Result.FAIL, None
        px, py = view.player.pos
        tiles.sort(key=lambda t: abs(t[0] - px) + abs(t[1] - py))
        target = tiles[0]
        if self._start_inv is None:
            self._start_inv = current_inv
        self._attempts += 1
        self._active = FaceAndInteract(target, withdraw_action())
        return self._active.step(view)


_FURNACE_OUTPUTS: frozenset[int] = frozenset(
    {
        int(ItemType.IRON_PLATE),
        int(ItemType.COPPER_PLATE),
        int(ItemType.TIN_PLATE),
        int(ItemType.WAFER),
        int(ItemType.REFRACTORY),
    }
)


def _find_recipe(output_item: int) -> dict | None:
    """Look up the recipe that produces *output_item*."""
    for r in RECIPES:
        if int(r["output"]) == int(output_item):
            return r
    return None


def _default_machine_for(output_item: int) -> int:
    """Decide which machine processes this recipe (furnace vs. assembler)."""
    if int(output_item) in _FURNACE_OUTPUTS:
        return int(MachineType.FURNACE)
    return int(MachineType.ASSEMBLER)


class ProduceInMachine(Goal):
    """Run a recipe on a placed machine until inventory holds *count* outputs.

    One production cycle = deposit every input (respecting per-input
    quantities) → wait for the recipe's ticks → withdraw one output.
    Inputs are not pre-batched: each cycle deposits exactly one
    recipe's worth. This is slower but makes the FSM simple and the
    inventory delta trivial to track.

    Use the :func:`ProduceInFurnace` / :func:`ProduceInAssembler`
    helpers below for the common case — they auto-pick the right
    machine type based on the recipe.
    """

    name = "ProduceInMachine"

    def __init__(
        self,
        output_item: int | ItemType,
        count: int,
        machine_type: int | MachineType | None = None,
    ) -> None:
        self.output_item = int(output_item)
        self.count = count
        recipe = _find_recipe(self.output_item)
        if recipe is None:
            raise ValueError(
                f"no recipe produces {ItemType(self.output_item).name}",
            )
        self.machine_type = (
            int(machine_type)
            if machine_type is not None
            else _default_machine_for(self.output_item)
        )
        self.recipe_inputs: list[tuple[int, int]] = [
            (int(it), int(q)) for it, q in recipe["inputs"]
        ]
        self.wait_ticks: int = int(recipe["ticks"]) + 3

        self._sub: Goal | None = None
        self._phase: str = "deposit"  # "deposit" | "wait" | "withdraw"
        self._deposit_input_idx: int = 0
        self._deposit_count: int = 0
        self._wait_elapsed: int = 0

    def step(self, view: WorldView) -> StepReturn:
        if view.player.held(self.output_item) >= self.count:
            return Result.DONE, None

        if self._sub is not None:
            result, action = self._sub.step(view)
            if result is Result.DONE:
                self._sub = None
                return Result.RUNNING, int(Action.NOOP)
            if result is Result.FAIL:
                self._sub = None
                return Result.FAIL, None
            return Result.RUNNING, action

        if self._phase == "deposit":
            return self._step_deposit(view)
        if self._phase == "wait":
            return self._step_wait()
        if self._phase == "withdraw":
            return self._step_withdraw(view)
        raise AssertionError(f"unknown phase: {self._phase}")

    def _step_deposit(self, view: WorldView) -> StepReturn:
        input_item, required = self.recipe_inputs[self._deposit_input_idx]
        if self._deposit_count < required:
            if view.player.held(input_item) < 1:
                return Result.FAIL, None
            self._deposit_count += 1
            self._sub = DepositInto(self.machine_type, input_item)
            return self._sub.step(view)
        # All copies of this input are deposited; move to next input.
        self._deposit_input_idx += 1
        self._deposit_count = 0
        if self._deposit_input_idx >= len(self.recipe_inputs):
            # All inputs deposited — start the wait.
            self._phase = "wait"
            self._wait_elapsed = 0
        return Result.RUNNING, int(Action.NOOP)

    def _step_wait(self) -> StepReturn:
        if self._wait_elapsed >= self.wait_ticks:
            self._phase = "withdraw"
            return Result.RUNNING, int(Action.NOOP)
        self._wait_elapsed += 1
        return Result.RUNNING, int(Action.NOOP)

    def _step_withdraw(self, view: WorldView) -> StepReturn:
        # Reset the phase before delegating to withdraw; when the
        # withdraw sub-goal finishes we'll loop back to "deposit" for
        # the next production cycle (or exit via the held >= count
        # check at the top of step()).
        self._sub = WithdrawFrom(self.machine_type, self.output_item)
        self._phase = "deposit"
        self._deposit_input_idx = 0
        self._deposit_count = 0
        return self._sub.step(view)


def ProduceInFurnace(  # noqa: N802 - factory mirrors class-style instantiation
    output_item: int | ItemType,
    count: int,
) -> ProduceInMachine:
    """Produce *count* of *output_item* via the nearest furnace."""
    return ProduceInMachine(output_item, count, int(MachineType.FURNACE))


def ProduceInAssembler(  # noqa: N802 - factory mirrors class-style instantiation
    output_item: int | ItemType,
    count: int,
) -> ProduceInMachine:
    """Produce *count* of *output_item* via the nearest assembler."""
    return ProduceInMachine(output_item, count, int(MachineType.ASSEMBLER))


# ---------------------------------------------------------------------------
# Pipelined production across multiple machines
# ---------------------------------------------------------------------------


class PipelinedProduce(Goal):
    """Produce ``count`` of ``output_item`` by rotating through *k* machines.

    Unlike :class:`ProduceInMachine`, which parks the agent at one
    machine during its ``Wait`` phase, this goal keeps *k* machines
    running in parallel. On each tick the agent picks the highest-
    priority action across the k-nearest matching machines:

    1. **Withdraw** any non-empty output slot (slot 2 in the
       observation). Broadened from "only our output_item" so that
       leftovers from a previous goal don't block the deposit gate.
    2. **Deposit** the next missing input, computed per-machine from
       the current physical slot state: slot 0 and slot 1 hold
       assembler/furnace inputs, and the observation surfaces their
       item type + count directly. For each candidate tile, figure
       out which input is under-filled relative to the recipe and
       deposit it — no software ``_deposits`` counter to drift.
    3. Otherwise emit ``NOOP`` — a cycle is cooking and no deposit
       is pending.

    The physical-slot read fully replaces the earlier ``_deposits``
    dict: Phase 0 pulls from adjacent buffers, engine-side cycle
    transitions, and aborted deposits all remain visible through
    the observation, so the FSM can never disagree with the engine.
    """

    name = "PipelinedProduce"

    def __init__(
        self,
        output_item: int | ItemType,
        count: int,
        machine_type: int | MachineType,
        k: int = 3,
    ) -> None:
        self.output_item = int(output_item)
        self.count = count
        self.machine_type = int(machine_type)
        self.k = max(1, k)
        recipe = _find_recipe(self.output_item)
        if recipe is None:
            raise ValueError(
                f"no recipe produces {ItemType(self.output_item).name}",
            )
        # Store recipe inputs aligned with the physical asm_in layout:
        # index 0 → slot 0, index 1 → slot 1 (if present).
        self._recipe_inputs: list[tuple[int, int]] = [
            (int(it), int(qty)) for it, qty in recipe["inputs"]
        ]

        self._active: FaceAndInteract | None = None

    def step(self, view: WorldView) -> StepReturn:
        if view.player.held(self.output_item) >= self.count:
            return Result.DONE, None

        if self._active is not None:
            result, action = self._active.step(view)
            if result is Result.RUNNING:
                return Result.RUNNING, action
            self._active = None
            # Fall through — pick the next action this same tick.

        tiles = view.tiles_with_machine(self.machine_type)
        if not tiles:
            return Result.FAIL, None
        px, py = view.player.pos
        tiles.sort(key=lambda t: abs(t[0] - px) + abs(t[1] - py))
        candidates = tiles[: self.k]

        # Priority 1: withdraw ANY non-empty output slot on a
        # candidate machine. Slot 2 is the uniform output projection.
        for tile in candidates:
            tx, ty = tile
            if int(view.slot2_count[ty, tx]) > 0:
                self._active = FaceAndInteract(tile, int(Action.WITHDRAW))
                return self._active.step(view)

        # Priority 2: deposit the next missing input into the nearest
        # candidate that can accept one. For each candidate, compare
        # the recipe's input requirements against the current physical
        # slot contents (type + count) read straight from the obs.
        for tile in candidates:  # already sorted by distance
            next_item = self._missing_input(view, tile)
            if next_item is None:
                continue
            if view.player.held(next_item) < 1:
                continue
            self._active = FaceAndInteract(
                tile,
                deposit_action_for(next_item),
            )
            return self._active.step(view)

        # Nothing to do this tick: cooking in progress, no output
        # ready yet, no deposits to make. Wait.
        return Result.RUNNING, int(Action.NOOP)

    def _missing_input(
        self,
        view: WorldView,
        tile: tuple[int, int],
    ) -> int | None:
        """Next required-but-missing input item at *tile*, or None.

        Reads the physical slot state from the observation:

        - Slot is empty (count 0, type 0): needs all of ``req_qty``
          deposits of the recipe input type.
        - Slot already matches the recipe type: need
          ``req_qty - slot_count`` more.
        - Slot holds a different type (stale cycle, wrong recipe):
          skip this machine entirely — the engine's type-gate will
          reject our deposit anyway.

        Returns the first input item (in recipe order) that this
        machine still needs, or ``None`` if every slot is at or above
        its quota.
        """
        tx, ty = tile
        slot_types = (view.slot0_type, view.slot1_type)
        slot_counts = (view.slot0_count, view.slot1_count)
        for slot_idx, (req_type, req_qty) in enumerate(self._recipe_inputs):
            have_type = int(slot_types[slot_idx][ty, tx])
            have_count = int(slot_counts[slot_idx][ty, tx])
            if have_type not in (0, req_type):
                return None
            if have_count < req_qty:
                return req_type
        return None


def deposit_action_for(item_type: int) -> int:
    """Resolve an ItemType to its DEPOSIT_* action id."""
    from .world_model import deposit_action

    return deposit_action(item_type)


# ---------------------------------------------------------------------------
# Bootstrap-feedback: withdraw from a known bus pallet, then craft from it
# ---------------------------------------------------------------------------


class WithdrawFromBusAt(Goal):
    """Withdraw from a *specific* pallet tile until inventory is full enough.

    Drains a fixed pallet location — typically a Tier-N bus pallet whose
    coordinates the planner controls. Differs from
    :class:`WithdrawUntilHeld`, which auto-picks the nearest matching
    pallet: in a bus design with one assembler per recipe each item
    has exactly one canonical source pallet, and explicit targeting
    keeps the planner's intent legible.

    If the pallet is mid-production and currently holds fewer items
    than needed, the goal keeps emitting WITHDRAW. It fails only when
    the player's inventory has not advanced for ``max_idle_attempts``
    consecutive ticks — long enough that something upstream is
    genuinely stuck.

    Args:
        tile: ``(x, y)`` of the source pallet.
        item_type: Expected item type in the pallet.
        count: Stop once ``view.player.held(item_type) >= count``.
        max_idle_attempts: Consecutive idle ticks tolerated before
            giving up. 24 covers a full assembler cycle (8 ticks at
            recipe ``ticks=8``) plus a couple of arm transfers, so a
            single missed handoff doesn't FAIL the goal.
    """

    name = "WithdrawFromBusAt"

    def __init__(
        self,
        tile: tuple[int, int],
        item_type: int | ItemType,
        count: int,
        max_idle_attempts: int = 24,
    ) -> None:
        self.tile = tile
        self.item_type = int(item_type)
        self.count = count
        self.max_idle_attempts = max_idle_attempts
        self._active: FaceAndInteract | None = None
        self._idle_attempts = 0
        self._last_held: int | None = None

    def step(self, view: WorldView) -> StepReturn:
        held = view.player.held(self.item_type)
        if held >= self.count:
            return Result.DONE, None

        if self._last_held is not None:
            if held > self._last_held:
                self._idle_attempts = 0
            else:
                self._idle_attempts += 1
        self._last_held = held

        if self._idle_attempts >= self.max_idle_attempts:
            return Result.FAIL, None

        if self._active is not None:
            result, action = self._active.step(view)
            if result is Result.RUNNING:
                return Result.RUNNING, action
            self._active = None
            return Result.RUNNING, int(Action.NOOP)

        x, y = self.tile
        if view.machine_type[y, x] != int(MachineType.PALLET):
            return Result.FAIL, None

        self._active = FaceAndInteract(self.tile, int(Action.WITHDRAW))
        return self._active.step(view)


class CraftFromBus(Goal):
    """Withdraw recipe inputs from named bus pallets, then run them
    through a placed machine.

    Implements the bootstrap-feedback pattern: instead of mining ore
    for every machine the agent needs to build, draw the necessary
    plates / intermediates from already-running bus pallets and run
    them through a furnace or assembler. Once Tier 1 (smelting) is
    online, every later machine — arms, belts, assemblers — comes
    from this goal, which is what makes "ever-increasing chunks"
    actually pay off.

    The goal expands into a sequence of sub-goals:

    1. One :class:`WithdrawFromBusAt` per recipe input, sized to leave
       the player with exactly enough material for ``count`` crafts.
    2. One :class:`ProduceInMachine` cycle for the actual production
       (deposit inputs into the nearest furnace/assembler, wait for
       the recipe ticks, withdraw the output). The rocket benchmark
       masks every ``CRAFT_*`` action, so production must flow
       through a placed machine — hand-crafting in player inventory
       isn't an option.

    Args:
        output_item: Recipe output (e.g. ``ItemType.ARM``).
        count: Number to craft.
        bus_tiles: Mapping ``input_item_id -> (x, y)`` pallet location
            for every input of the recipe. Items not in the recipe
            are ignored; missing recipe inputs raise ``ValueError``.
        machine_type: Override the default machine. ``None`` (the
            default) auto-picks furnace for smelt recipes and
            assembler for everything else, matching
            :func:`ProduceInFurnace` / :func:`ProduceInAssembler`.

    Raises:
        ValueError: If no recipe produces ``output_item`` or if
            ``bus_tiles`` is missing a pallet for one of the recipe's
            inputs.
    """

    name = "CraftFromBus"

    def __init__(
        self,
        output_item: int | ItemType,
        count: int,
        bus_tiles: dict[int | ItemType, tuple[int, int]],
        machine_type: int | MachineType | None = None,
    ) -> None:
        self.output_item = int(output_item)
        self.count = count
        self.bus_tiles = {int(k): v for k, v in bus_tiles.items()}
        recipe = _find_recipe(self.output_item)
        if recipe is None:
            raise ValueError(
                f"no recipe for {ItemType(self.output_item).name}",
            )
        for input_item, _per_craft in recipe["inputs"]:
            if int(input_item) not in self.bus_tiles:
                raise ValueError(
                    f"CraftFromBus({ItemType(self.output_item).name}, "
                    f"count={self.count}) missing bus tile for input "
                    f"{ItemType(int(input_item)).name}",
                )
        self.recipe = recipe
        self.machine_type = machine_type
        self._steps: list[Goal] | None = None
        self._idx = 0

    def _build_steps(self, view: WorldView) -> list[Goal]:
        """Build the sub-goal sequence given the current inventory."""
        # ``view`` is unused now that target_held doesn't depend on
        # current inventory; kept on the signature so the call site
        # in :meth:`step` can pass it without branching.
        del view
        steps: list[Goal] = []
        for input_item, per_craft in self.recipe["inputs"]:
            input_id = int(input_item)
            need_total = self.count * int(per_craft)
            steps.append(
                WithdrawFromBusAt(
                    self.bus_tiles[input_id],
                    input_id,
                    need_total,
                ),
            )
        steps.append(
            ProduceInMachine(
                self.output_item,
                self.count,
                machine_type=self.machine_type,
            ),
        )
        return steps

    def step(self, view: WorldView) -> StepReturn:
        if self._steps is None:
            self._steps = self._build_steps(view)
        if self._idx >= len(self._steps):
            return Result.DONE, None
        result, action = self._steps[self._idx].step(view)
        if result is Result.DONE:
            self._idx += 1
            if self._idx >= len(self._steps):
                return Result.DONE, None
            return Result.RUNNING, int(Action.NOOP)
        if result is Result.FAIL:
            return Result.FAIL, None
        return Result.RUNNING, action


# ---------------------------------------------------------------------------
# Tier 1 — smelter cell (Module 1)
# ---------------------------------------------------------------------------


class BuildSmelterCell(Goal):
    """Build one smelter cell at the south edge of an ore patch.

    Layout (top-down view, ``mx = patch_x + 1`` = patch's center
    column, ``my_se = patch_y_top + 2`` = south edge of a 3x3 patch).
    All coordinates are ``(x, y)``::

           O   O   O                       (patch)
           ... patch_y_top + 0 ...
           O   O   O
           ... patch_y_top + 1 ...
           O   M   O                       (miner ON south-edge ore tile)
           ... my_se ...
           .   P_ore   .                   ore pallet receives miner push
           ... my_se + 1 ...
           .   F     >    P_plate          furnace + east arm + plate pallet
           ... my_se + 2 ...
           .   =coal_trunk_segment=        added by Phase 3

    Engine semantics that the layout relies on:

    - Miner reads ``block_resources`` at its own tile, so it must
      sit ON an ore tile. Direction DOWN means the miner pushes its
      output one tile south — into the ore pallet.
    - Furnace and assembler ``run_assemblers`` auto-pulls from any
      adjacent buffer, matching empty/same input slots by item
      type. With ore pallet north and the coal trunk south, the
      furnace satisfies both inputs without arm orchestration.
    - Arm at ``(mx+1, my_se+2)`` facing EAST has the furnace as its
      "behind" tile and the plate pallet as its "front" tile, so
      each tick it pulls one plate from the furnace's output buffer
      and pushes it into the plate pallet.

    Bootstrap cost (must be in player inventory before this goal
    starts): ``1 MINER + 2 PALLET + 1 FURNACE + 1 ARM``. The order
    of placement — plate-pallet → arm → furnace → ore-pallet → miner
    — keeps every stand tile walkable. The arm is placed *before*
    the furnace because the arm's stand tile (one west of the arm,
    facing east) is the tile that will become the furnace.

    Args:
        patch_x: Top-left column of the 3x3 ore patch.
        patch_y_top: Top-left row of the 3x3 ore patch.
        patch_size: Side length of the patch in tiles. Defaults to 3
            (rocket benchmark). Other sizes shift the miner tile
            accordingly.

    Sub-goals run sequentially via the same Goal-of-goals pattern as
    :class:`CraftFromBus`; any sub-goal failure bubbles up.
    """

    name = "BuildSmelterCell"

    def __init__(
        self,
        patch_x: int,
        patch_y_top: int,
        patch_size: int = 3,
    ) -> None:
        self.patch_x = patch_x
        self.patch_y_top = patch_y_top
        self.patch_size = patch_size
        mx = patch_x + patch_size // 2
        my_se = patch_y_top + patch_size - 1
        self.miner_tile = (mx, my_se)
        self.ore_pallet_tile = (mx, my_se + 1)
        self.furnace_tile = (mx, my_se + 2)
        self.arm_tile = (mx + 1, my_se + 2)
        self.plate_pallet_tile = (mx + 2, my_se + 2)
        self.coal_belt_tile = (mx, my_se + 3)

        self._steps: list[Goal] = [
            PlaceMachineAt(
                MachineType.PALLET,
                self.plate_pallet_tile,
                int(Direction.DOWN),
            ),
            PlaceMachineAt(
                MachineType.ARM,
                self.arm_tile,
                int(Direction.RIGHT),
            ),
            PlaceMachineAt(
                MachineType.FURNACE,
                self.furnace_tile,
                int(Direction.DOWN),
            ),
            PlaceMachineAt(
                MachineType.PALLET,
                self.ore_pallet_tile,
                int(Direction.DOWN),
            ),
            PlaceMachineAt(
                MachineType.MINER,
                self.miner_tile,
                int(Direction.DOWN),
            ),
        ]
        self._idx = 0

    def step(self, view: WorldView) -> StepReturn:
        if self._idx >= len(self._steps):
            return Result.DONE, None
        result, action = self._steps[self._idx].step(view)
        if result is Result.DONE:
            self._idx += 1
            if self._idx >= len(self._steps):
                return Result.DONE, None
            return Result.RUNNING, int(Action.NOOP)
        if result is Result.FAIL:
            return Result.FAIL, None
        return Result.RUNNING, action


# ---------------------------------------------------------------------------
# Tier 1 — coal trunk (Module 2)
# ---------------------------------------------------------------------------


class BuildCoalTrunk(Goal):
    """Lay a coal trunk: feeder arm + chain of belts.

    The trunk consists of one feeder arm pulling from an existing
    coal source pallet, plus a chain of belts that carry the coal
    from the arm's push tile to wherever the planner wants it
    delivered (typically a tile adjacent to a furnace, where the
    furnace's run_assemblers Phase 0 auto-pulls from the belt's
    buffer).

    **Placement-order constraint.** The feeder arm's stand tile is
    ``feeder_arm_tile - unit(feeder_arm_dir)`` — same as the arm's
    "behind" tile, which is exactly where the coal source pallet
    has to live. To place the arm, that tile must be walkable; to
    place the pallet, it must not be. The clean resolution is to
    place the arm *before* the pallet:

    1. Caller has the arm's behind tile as dirt at goal start.
    2. ``BuildCoalTrunk`` places the arm there.
    3. Caller follows up with a pallet placement on the (now-arm's-
       behind) tile via a different stand tile.

    For the rocket benchmark this means the coal-extraction cell
    (miner + pallet on the coal patch) is built *after* the trunk's
    arm, with the pallet's PlaceMachineAt using a non-default facing
    direction so its stand tile sits on the patch instead of on the
    arm.

    Args:
        feeder_arm_tile: ``(x, y)`` for the feeder arm.
        feeder_arm_dir: Facing direction of the feeder arm. The arm
            pulls from its "behind" tile (=
            ``feeder_arm_tile - unit(feeder_arm_dir)``).
        belt_specs: List of ``((x, y), direction)`` for belt segments
            in placement order. Each belt's stand tile must be
            reachable when its turn comes; belts are walkable so
            placing the chain in forward order is safe — the agent
            stands on the previous belt to place the next.

    Bootstrap cost: ``1 ARM + len(belt_specs) BELTs`` in the player's
    inventory before the goal starts.
    """

    name = "BuildCoalTrunk"

    def __init__(
        self,
        feeder_arm_tile: tuple[int, int],
        feeder_arm_dir: int,
        belt_specs: list[tuple[tuple[int, int], int]],
    ) -> None:
        self.feeder_arm_tile = feeder_arm_tile
        self.feeder_arm_dir = int(feeder_arm_dir)
        self.belt_specs = list(belt_specs)
        self._steps: list[Goal] = [
            PlaceMachineAt(
                MachineType.ARM,
                feeder_arm_tile,
                int(feeder_arm_dir),
            ),
        ] + [
            PlaceMachineAt(
                MachineType.CONVEYOR_BELT,
                tile,
                int(direction),
            )
            for tile, direction in belt_specs
        ]
        self._idx = 0

    def step(self, view: WorldView) -> StepReturn:
        if self._idx >= len(self._steps):
            return Result.DONE, None
        result, action = self._steps[self._idx].step(view)
        if result is Result.DONE:
            self._idx += 1
            if self._idx >= len(self._steps):
                return Result.DONE, None
            return Result.RUNNING, int(Action.NOOP)
        if result is Result.FAIL:
            return Result.FAIL, None
        return Result.RUNNING, action


# ---------------------------------------------------------------------------
# Tier 2 — assembler/furnace production module
# ---------------------------------------------------------------------------


class BuildAssemblerModule(Goal):
    """Place a 1- or 2-input production module around a center machine.

    The module is a generic "production cell" used for every tier-2
    intermediate (WIRE, FRAME, WAFER, REFRACTORY) and reused at higher
    tiers. Layout (top-down view, all coordinates ``(x, y)``)::

                          P_in_a            (asm.x,    asm.y - 1)
        P_in_b   Center   Arm     P_out     (asm.x-1)..(asm.x+2, asm.y)
                          .                 (asm.x,    asm.y + 1)  free

    Engine semantics relied on:

    - The center machine (assembler or furnace) auto-pulls one item
      per tick from each adjacent buffer in ``run_assemblers`` Phase 0,
      matching empty/same-type input slots. With pallets at the north
      and west neighbours both inputs land in ``ent_asm_in`` without
      any arm orchestration.
    - The east-facing arm reads from the center machine's
      ``ent_asm_out`` (engine fix in commit 61ff84a) and pushes east
      into the output pallet's ``ent_buf``.
    - For 1-input recipes (REFRACTORY uses COAL only), pass
      ``input_b_tile=None``; only one input pallet is placed and the
      bootstrap cost drops by one PALLET.

    Bootstrap cost in player inventory:

    - 2-input (default): ``1 ASSEMBLER|FURNACE + 3 PALLET + 1 ARM``
    - 1-input: ``1 ASSEMBLER|FURNACE + 2 PALLET + 1 ARM``

    Place order (output -> P_in_b -> arm -> center -> P_in_a) keeps
    every stand tile walkable. The center machine is placed *before*
    P_in_a because the center's stand tile (north, facing DOWN) is
    exactly P_in_a's tile — same trick as
    :class:`BuildSmelterCell` placing the furnace before the ore
    pallet. P_in_b uses facing LEFT so its stand tile is the (still
    empty) center tile.

    Args:
        center_tile: ``(x, y)`` for the assembler or furnace.
        center_machine: :class:`MachineType.ASSEMBLER` or
            :class:`MachineType.FURNACE`.
        input_a_tile: ``(x, y)`` for the north input pallet
            (always required).
        input_b_tile: ``(x, y)`` for the west input pallet, or
            ``None`` for a 1-input recipe.
        output_pallet_tile: ``(x, y)`` for the output pallet
            (east of the arm).
    """

    name = "BuildAssemblerModule"

    def __init__(
        self,
        center_tile: tuple[int, int],
        center_machine: int | MachineType,
        input_a_tile: tuple[int, int],
        input_b_tile: tuple[int, int] | None,
        output_pallet_tile: tuple[int, int],
    ) -> None:
        self.center_tile = center_tile
        self.center_machine = int(center_machine)
        self.input_a_tile = input_a_tile
        self.input_b_tile = input_b_tile
        self.output_pallet_tile = output_pallet_tile
        self.arm_tile = (center_tile[0] + 1, center_tile[1])

        steps: list[Goal] = [
            PlaceMachineAt(
                MachineType.PALLET,
                output_pallet_tile,
                int(Direction.DOWN),
            ),
        ]
        if input_b_tile is not None:
            steps.append(
                PlaceMachineAt(
                    MachineType.PALLET,
                    input_b_tile,
                    int(Direction.LEFT),
                )
            )
        steps.extend(
            [
                PlaceMachineAt(
                    MachineType.ARM,
                    self.arm_tile,
                    int(Direction.RIGHT),
                ),
                PlaceMachineAt(
                    self.center_machine,
                    center_tile,
                    int(Direction.DOWN),
                ),
                PlaceMachineAt(
                    MachineType.PALLET,
                    input_a_tile,
                    int(Direction.DOWN),
                ),
            ]
        )
        self._steps: list[Goal] = steps
        self._idx = 0

    def step(self, view: WorldView) -> StepReturn:
        if self._idx >= len(self._steps):
            return Result.DONE, None
        result, action = self._steps[self._idx].step(view)
        if result is Result.DONE:
            self._idx += 1
            if self._idx >= len(self._steps):
                return Result.DONE, None
            return Result.RUNNING, int(Action.NOOP)
        if result is Result.FAIL:
            return Result.FAIL, None
        return Result.RUNNING, action
