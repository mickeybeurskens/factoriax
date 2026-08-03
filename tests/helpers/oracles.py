"""Navigation primitives for the scripted-oracle scenario tests.

A scripted oracle proves a scenario is solvable: it walks the player to a
target, faces it, and acts. These are the movement blocks that the oracles
share, including a BFS that routes around solid machines. The oracles
themselves stay in their test modules.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from factoriax.engine.constants import Action, BlockType, Direction, ItemType
from factoriax.engine.tables import DIRECTIONS

DIR_TO_MOVE = {
    Direction.UP: Action.UP,
    Direction.DOWN: Action.DOWN,
    Direction.LEFT: Action.LEFT,
    Direction.RIGHT: Action.RIGHT,
}
DIR_TO_FACE = {
    Direction.UP: Action.FACE_UP,
    Direction.DOWN: Action.FACE_DOWN,
    Direction.LEFT: Action.FACE_LEFT,
    Direction.RIGHT: Action.FACE_RIGHT,
}

#: Ore block on the map -> ore item it yields when mined.
BLOCK_TO_ITEM = {
    int(BlockType.IRON): int(ItemType.IRON_ORE),
    int(BlockType.COPPER): int(ItemType.COPPER_ORE),
    int(BlockType.TIN): int(ItemType.TIN_ORE),
    int(BlockType.SILICON): int(ItemType.SILICON),
    int(BlockType.COAL): int(ItemType.COAL),
    int(BlockType.LIMESTONE): int(ItemType.LIMESTONE),
}


def approach(px: int, py: int, tx: int, ty: int) -> int:
    """Greedy move action toward ``(tx, ty)``, longer axis first."""
    dx, dy = tx - px, ty - py
    if abs(dx) >= abs(dy):
        return int(DIR_TO_MOVE[Direction.RIGHT if dx > 0 else Direction.LEFT])
    return int(DIR_TO_MOVE[Direction.DOWN if dy > 0 else Direction.UP])


def face_or_act_adjacent(state, wanted: np.ndarray, act: int) -> int | None:
    """Return ``act`` when facing a wanted tile, FACE when adjacent, else None.

    Parameters
    ----------
    wanted :
        Boolean ``(h, w)`` mask of tiles to act on (mine, deposit, ...).
    act :
        Action to return once the player faces a wanted tile.
    """
    h, w = wanted.shape
    px, py = (int(v) for v in state.player_positions[0])
    for d in DIR_TO_MOVE:
        off = np.asarray(DIRECTIONS[int(d)])
        tx, ty = px + int(off[0]), py + int(off[1])
        if 0 <= tx < w and 0 <= ty < h and wanted[ty, tx]:
            if int(state.player_directions[0]) == int(d):
                return act
            return int(DIR_TO_FACE[d])
    return None


def face_or_mine_adjacent(state, wanted_blocks: set[int]) -> int | None:
    """Return MINE when facing a wanted block, FACE when adjacent, else None."""
    mask = np.isin(np.asarray(state.map), list(wanted_blocks))
    return face_or_act_adjacent(state, mask, int(Action.MINE))


def nearest(state, wanted_blocks: set[int]) -> tuple[int, int] | None:
    """Coordinates of the nearest wanted block, or None."""
    m = np.asarray(state.map)
    px, py = (int(v) for v in state.player_positions[0])
    ys, xs = np.where(np.isin(m, list(wanted_blocks)))
    if len(xs) == 0:
        return None
    dists = np.abs(xs - px) + np.abs(ys - py)
    i = int(np.argmin(dists))
    return int(xs[i]), int(ys[i])


def bfs_step_toward(state, wanted: np.ndarray) -> int:
    """First move of a shortest walkable path to a tile adjacent to
    ``wanted``.

    Machines block movement, and every machine on these maps is solid
    because there are no belts. This is what defeats a greedy walker
    once placed machines appear. BFS routes around them.

    Parameters
    ----------
    wanted :
        Boolean ``(h, w)`` mask of target tiles (stood *next to*, not on).
    """
    m = np.asarray(state.map)
    machines = np.asarray(state.machine_types)
    h, w = m.shape
    walkable = (m != int(BlockType.WATER)) & (machines == 0)
    px, py = (int(v) for v in state.player_positions[0])

    goal = np.zeros_like(wanted)
    for d in DIR_TO_MOVE:
        off = np.asarray(DIRECTIONS[int(d)])
        goal |= np.roll(wanted, (int(off[1]), int(off[0])), axis=(0, 1))
    goal &= walkable

    prev: dict[tuple[int, int], tuple[int, int]] = {}
    seen = {(px, py)}
    queue = deque([(px, py)])
    found = None
    while queue:
        cx, cy = queue.popleft()
        if goal[cy, cx]:
            found = (cx, cy)
            break
        for d in DIR_TO_MOVE:
            off = np.asarray(DIRECTIONS[int(d)])
            nx, ny = cx + int(off[0]), cy + int(off[1])
            if 0 <= nx < w and 0 <= ny < h and walkable[ny, nx]:
                if (nx, ny) not in seen:
                    seen.add((nx, ny))
                    prev[(nx, ny)] = (cx, cy)
                    queue.append((nx, ny))

    assert found is not None, "BFS: no reachable tile adjacent to a target"
    cur = found
    while prev.get(cur, (px, py)) != (px, py) and cur in prev:
        cur = prev[cur]
    return approach(px, py, cur[0], cur[1])


def goto_and_act(state, wanted: np.ndarray, act: int) -> int:
    """One oracle step of walk-to / face / act on a wanted tile.

    Returns ``act`` when already facing a wanted tile, a FACE action
    when merely adjacent, and otherwise the first move of a BFS path.
    Call it once per tick until the act fires.
    """
    action = face_or_act_adjacent(state, wanted, act)
    if action is not None:
        return action
    return bfs_step_toward(state, wanted)
