"""Lookup tables for SPLITTER and CROSSING flow decoding.

The two tile types added on top of the base ``CONVEYOR_BELT`` have
direction encodings that need to be unpacked into per-axis (or per-output)
direction values inside the JIT'd machines pass. Doing the unpacking via
small constant ``jnp`` lookup arrays keeps the per-tick math branchless
and indexable in vectorised code paths.

Both tables are indexed by ``ent_direction`` (an ``int8`` in ``EnvState``).
Index ``0`` is the inactive/NONE direction — both tables return zeros so
inactive entities contribute nothing to neighbour pushes.

SPLITTER conventions
--------------------

A splitter facing direction ``d`` matches the conveyor-belt convention:
items flow *out* in direction ``d`` from a regular belt at the splitter's
``-d`` neighbour. Outputs are the two perpendicular directions:

* facing ``UP`` / ``DOWN``  → outputs ``LEFT`` and ``RIGHT``
* facing ``LEFT`` / ``RIGHT`` → outputs ``UP`` and ``DOWN``

CROSSING conventions
--------------------

A crossing has *two* independent axis flows packed into one ``ent_direction``
byte. Values 1..4 enumerate the four (vertical, horizontal) flow pairs:

* 1 → vertical N→S, horizontal W→E (diagonal ``\\``)
* 2 → vertical N→S, horizontal E→W (diagonal ``/``)
* 3 → vertical S→N, horizontal W→E (diagonal ``/``)
* 4 → vertical S→N, horizontal E→W (diagonal ``\\``)

The diagonal direction is implied by the input-side pairing — the renderer
recovers it from ``CROSSING_DIAGONAL`` rather than reading it back from
the per-axis output directions.
"""

from __future__ import annotations

import jax.numpy as jnp

from factoriax.constants import Direction

# ---------------------------------------------------------------------------
# Splitter — perpendicular output directions per facing
# ---------------------------------------------------------------------------

# Indexed by ``ent_direction``. Each row is the two output Direction values
# the splitter pushes to (perpendicular to its facing). Inactive direction
# (index 0) returns (0, 0) so an unset entry never aliases a real direction.
SPLITTER_PERP_OUTPUTS: jnp.ndarray = jnp.array(
    [
        [0, 0],  # NONE
        [int(Direction.UP), int(Direction.DOWN)],  # facing LEFT  → outputs N + S
        [int(Direction.UP), int(Direction.DOWN)],  # facing RIGHT → outputs N + S
        [int(Direction.LEFT), int(Direction.RIGHT)],  # facing UP   → outputs W + E
        [int(Direction.LEFT), int(Direction.RIGHT)],  # facing DOWN → outputs W + E
    ],
    dtype=jnp.int8,
)


# ---------------------------------------------------------------------------
# Crossing — per-axis output directions
# ---------------------------------------------------------------------------

# Indexed by the crossing's packed ``ent_direction`` (1..4). Returns
# (vertical_output_dir, horizontal_output_dir). The vertical axis pulls
# from the opposite of its output direction (so vert_out=DOWN means the
# axis reads from the N neighbour and writes to the S neighbour); same
# for the horizontal axis.
CROSSING_AXIS_DIRS: jnp.ndarray = jnp.array(
    [
        [0, 0],  # NONE / unset
        [int(Direction.DOWN), int(Direction.RIGHT)],  # 1: N→S + W→E (diagonal \)
        [int(Direction.DOWN), int(Direction.LEFT)],  # 2: N→S + E→W (diagonal /)
        [int(Direction.UP), int(Direction.RIGHT)],  # 3: S→N + W→E (diagonal /)
        [int(Direction.UP), int(Direction.LEFT)],  # 4: S→N + E→W (diagonal \)
    ],
    dtype=jnp.int8,
)


# Visual-only: ``\`` for diagonals NW-SE, ``/`` for NE-SW. The renderer
# uses this to draw the single diagonal line on each crossing tile.
CROSSING_DIAGONAL: tuple[str, ...] = (
    "",  # NONE
    "\\",  # 1: input pair (N, W) — the two inputs are NW corners
    "/",  # 2: input pair (N, E)
    "/",  # 3: input pair (S, W)
    "\\",  # 4: input pair (S, E)
)


# Slot indices in ``ent_asm_in[idx, *]`` for crossing axes.
CROSSING_VERT_SLOT: int = 0
CROSSING_HORIZ_SLOT: int = 1
