"""Decode tables for the direction byte of a ``SPLITTER`` or ``CROSSING`` tile.

``EnvState.ent_direction`` holds one int8 per entity. A conveyor belt reads it
as a plain facing, but a splitter pushes to two sides at once and a crossing
carries two independent flows, so both need the byte unpacked into a pair of
:class:`~factoriax.engine.constants.Direction` values. The tables here do that
unpacking as a constant array indexed by the byte, which keeps the belt tick in
:func:`~factoriax.engine.machines.run_conveyor_belts` branchless and traceable
under :func:`jax.jit`.

Every table reserves index 0 for the value :class:`Direction` leaves unnumbered,
which stored state uses to mean "no direction". Row 0 holds zeros, so an entity
with an unset direction decodes to a pair that matches no side and pushes
nothing.

Splitter encoding
-----------------

A splitter stores the direction items travel through it, the same convention a
conveyor belt uses: a splitter facing ``d`` takes items from the neighbour on
its ``-d`` side. It pushes to the two sides perpendicular to that facing.

* facing ``UP`` or ``DOWN`` pushes to ``LEFT`` and ``RIGHT``
* facing ``LEFT`` or ``RIGHT`` pushes to ``UP`` and ``DOWN``

Crossing encoding
-----------------

A crossing runs a vertical flow and a horizontal flow at the same time and packs
both into the one direction byte. Values 1 to 4 enumerate the four combinations,
written here as the output direction of each axis:

* 1: vertical pushes ``DOWN``, horizontal pushes ``RIGHT``
* 2: vertical pushes ``DOWN``, horizontal pushes ``LEFT``
* 3: vertical pushes ``UP``, horizontal pushes ``RIGHT``
* 4: vertical pushes ``UP``, horizontal pushes ``LEFT``

An axis takes items from the neighbour opposite its output direction, so the two
input sides of a crossing always meet at a corner. That corner and the one
across from it are the ends of the diagonal drawn on the tile.
"""

from __future__ import annotations

import jax.numpy as jnp

from factoriax.engine.constants import Direction

# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------

#: Output sides of a splitter, indexed by ``ent_direction``. Row ``d`` holds the
#: two :class:`~factoriax.engine.constants.Direction` values perpendicular to
#: facing ``d``, which are the sides a splitter facing ``d`` pushes to. Shape
#: ``(5, 2)``, dtype int8. The two entries sit in ascending ``Direction`` order
#: and the order carries no meaning: a splitter treats both sides alike. Row 0
#: is ``(0, 0)``, a pair that matches no direction, so an unset entity never
#: reads as pushing to a real side.
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
# Crossing
# ---------------------------------------------------------------------------

#: Output direction of each crossing axis, indexed by the packed
#: ``ent_direction``. Row ``e`` is ``(vertical_output, horizontal_output)`` for
#: encoding ``e`` in 1..4. Shape ``(5, 2)``, dtype int8. An axis takes items
#: from the neighbour opposite its output direction, so ``vertical_output ==
#: DOWN`` means the vertical flow reads the tile above and writes the tile
#: below. Row 0 is ``(0, 0)``: a crossing with an unset direction moves nothing
#: on either axis.
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


#: Diagonal glyph per crossing encoding, for drawing only. A backslash marks a
#: NW-SE diagonal, a forward slash a NE-SW one. The diagonal runs from the corner
#: where the encoding's two input sides meet to the corner across from it. Index
#: 0 is the empty string, because an unset crossing has no input sides. The
#: simulation never reads this value.
CROSSING_DIAGONAL: tuple[str, ...] = (
    "",  # NONE
    "\\",  # 1: input pair (N, W) — the two inputs are NW corners
    "/",  # 2: input pair (N, E)
    "/",  # 3: input pair (S, W)
    "\\",  # 4: input pair (S, E)
)


#: Slot index of the vertical flow buffer in ``EnvState.ent_asm_in_type`` and
#: ``EnvState.ent_asm_in_count``. A crossing reuses the two assembler input
#: slots as one buffer per axis, which is why the two flows cannot mix: each
#: axis only ever reads and writes its own slot.
CROSSING_VERT_SLOT: int = 0
#: Slot index of the horizontal flow buffer, in the same two arrays as
#: :data:`CROSSING_VERT_SLOT`.
CROSSING_HORIZ_SLOT: int = 1
