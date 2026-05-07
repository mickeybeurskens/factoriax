"""Build the FactoriaX sprite atlas from the procedural drawing code.

Generates ``factoriax/assets/atlas.png`` and the sidecar
``factoriax/assets/atlas.json`` per the layout documented in
``factoriax/assets/atlas.layout.md``. The JAX renderer reads sprites
from this atlas — every visual the play view shows for terrain,
machines, and the player flows through it.

The atlas is sourced from :mod:`factoriax.ui.icons`, which is the
procedural-art module that the editor and play HUD already use. By
funneling both renderers' sprites through the same code we avoid the
play-vs-editor visual divergence the prior flat-color atlas caused.

Output is RGBA. The JAX renderer's :func:`render_map` blends machine
and player layers onto the terrain using the alpha channel, so:

- Block cells are opaque (alpha=255 everywhere) — terrain is the
  ground truth and always paints in full.
- Machine cells keep the transparent corners that
  :func:`render_item_icon` produces, so placed machines read as
  objects sitting on terrain.
- Player and biter cells composite over what's beneath them, so
  walking onto a belt no longer hides the belt.

Directional categories carry one cell per
:class:`~factoriax.constants.Direction`:

- Machines: rows 1-4 hold variants for direction LEFT, RIGHT, UP,
  DOWN respectively. Non-directional machines (``PALLET``,
  ``ASSEMBLER``, ``FURNACE``, ``SCIENCE_LAB``, ``ROCKET``) are
  rendered once and duplicated across all four rows so a uniform
  gather works at render time.
- Player: misc row columns 1-4 hold the same four directions for
  ``player_idx=0``. Column 0 holds the biter sprite.

Usage::

    uv run python scripts/build_atlas.py
    uv run python scripts/build_atlas.py --out custom_dir/

The output is hash-stable across runs; CI verifies this via
``tests/test_atlas_fresh.py``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import orjson

from factoriax.constants import (
    NUM_ITEM_TYPES,
    BlockType,
    Direction,
    ItemType,
    MachineType,
)
from factoriax.jax_renderer import build_digit_atlas
from factoriax.ui.icons import (
    MACHINE_TO_ITEM,
    create_biter_texture,
    create_player_texture,
    get_textures,
    render_item_icon,
)

logger = logging.getLogger(__name__)

CELL_PX: int = 32
NUM_COLS: int = 33  # max(num_block, num_machine, num_item, ...) — driven by ItemType


# Direction order used for the atlas's directional rows / columns.
# Index = atlas row offset for machines (0 = LEFT row, 3 = DOWN row),
# and atlas column offset for the player cells in the misc row
# (1 + index gives the column, leaving col 0 for the biter sprite).
_DIRECTION_ORDER: tuple[Direction, ...] = (
    Direction.LEFT,
    Direction.RIGHT,
    Direction.UP,
    Direction.DOWN,
)
NUM_DIRECTIONS: int = len(_DIRECTION_ORDER)


# Row indices — keep in sync with atlas.layout.md and with the
# constants in factoriax/jax_renderer.py.
ROW_BLOCKS: int = 0
ROW_MACHINES_BASE: int = 1  # rows 1..4 hold one direction each.
ROW_ITEMS: int = ROW_MACHINES_BASE + NUM_DIRECTIONS  # 5
ROW_MISC: int = ROW_ITEMS + 1  # 6
ROW_DIGITS: int = ROW_MISC + 1  # 7
NUM_ROWS: int = ROW_DIGITS + 1  # 8

# Misc row column layout.
COL_MISC_BITER: int = 0
COL_MISC_PLAYER_BASE: int = 1


# Magenta with alpha=255 acts as the "missing sprite" sentinel. The
# alpha is non-zero so a missing entry actually paints, surfacing the
# gap visually rather than silently blending through.
MISSING_RGBA: tuple[int, int, int, int] = (255, 0, 255, 255)


# Directional machines — these get a distinct sprite per direction.
# Non-directional machines (PALLET, ASSEMBLER, FURNACE, SCIENCE_LAB,
# ROCKET, NONE) are rendered once and duplicated across all four
# direction rows.
_DIRECTIONAL_MACHINES: frozenset[MachineType] = frozenset(
    {
        MachineType.CONVEYOR_BELT,
        MachineType.MINER,
        MachineType.ARM,
        MachineType.SPLITTER,
        MachineType.CROSSING,
    }
)


def _drop_alpha(rgba: np.ndarray) -> np.ndarray:
    """Strip alpha from an RGBA cell, returning RGB."""
    return rgba[..., :3].astype(np.uint8)


def _to_rgba(rgb: np.ndarray, alpha: int = 255) -> np.ndarray:
    """Append a constant alpha channel to an RGB cell."""
    h, w = rgb.shape[:2]
    out = np.empty((h, w, 4), dtype=np.uint8)
    out[..., :3] = rgb[..., :3]
    out[..., 3] = alpha
    return out


def _digit_cell(digit: int, digit_atlas: np.ndarray) -> np.ndarray:
    """Render a single digit at the cell's top-left.

    The procedural digit atlas is 3 wide × 5 tall bool mask. We blit
    it as white-on-black at the top-left of a 32x32 cell so the rest
    of the cell is unused space. Returned cell is fully opaque.
    """
    cell = np.zeros((CELL_PX, CELL_PX, 4), dtype=np.uint8)
    cell[..., 3] = 255
    glyph = digit_atlas[digit]  # shape (5, 3) bool
    white = np.array([255, 255, 255, 255], dtype=np.uint8)
    black = np.array([0, 0, 0, 255], dtype=np.uint8)
    cell[: glyph.shape[0], : glyph.shape[1]] = np.where(glyph[:, :, None], white, black)
    return cell


def _ordered_enum_names(enum_cls: type) -> list[str]:
    """Return enum member names sorted by integer value."""
    return [m.name for m in sorted(enum_cls, key=int)]


def _block_cell(block: BlockType, textures: dict[int, np.ndarray]) -> np.ndarray | None:
    """Return the RGBA block sprite for *block*, or ``None`` to leave magenta."""
    tex = textures.get(int(block))
    if tex is None:
        return None
    if tex.shape[-1] == 4:
        # Force opaque — terrain is the ground truth layer.
        out = tex.copy()
        out[..., 3] = 255
        return out
    return _to_rgba(tex)


def _machine_cell(machine: MachineType, direction: Direction) -> np.ndarray | None:
    """Return the RGBA machine sprite for *machine* facing *direction*.

    Non-directional machines ignore *direction*. Returns ``None`` to
    leave the cell magenta when no item maps to this machine
    (e.g. ``MachineType.NONE``); the renderer's NONE row is fully
    transparent so the sentinel never paints in practice.
    """
    if machine == MachineType.NONE:
        # Fully transparent — alpha compositing turns this into a no-op.
        return np.zeros((CELL_PX, CELL_PX, 4), dtype=np.uint8)
    item_id = MACHINE_TO_ITEM.get(int(machine))
    if item_id is None:
        return None
    icon = render_item_icon(item_id, CELL_PX, direction=int(direction))
    if icon.shape[-1] == 3:
        return _to_rgba(icon)
    return icon.astype(np.uint8)


def _item_cell(item: ItemType) -> np.ndarray | None:
    """Return the RGBA item sprite for *item*, or ``None`` to leave magenta.

    Items aren't currently drawn by the world renderer, so the cells
    are reserved for future HUD work.
    """
    if item == ItemType.EMPTY:
        return np.zeros((CELL_PX, CELL_PX, 4), dtype=np.uint8)
    icon = render_item_icon(int(item), CELL_PX, direction=int(Direction.DOWN))
    if icon.shape[-1] == 3:
        return _to_rgba(icon)
    return icon.astype(np.uint8)


def _player_cell(direction: Direction) -> np.ndarray:
    """Return the RGBA player sprite (player 0) facing *direction*."""
    sprite = create_player_texture(
        direction=int(direction),
        player_idx=0,
        is_selected=True,
        size=CELL_PX,
    )
    if sprite.shape[-1] == 3:
        return _to_rgba(sprite)
    return sprite.astype(np.uint8)


def _biter_cell() -> np.ndarray:
    """Return the RGBA biter sprite."""
    sprite = create_biter_texture(CELL_PX)
    if sprite.shape[-1] == 3:
        return _to_rgba(sprite)
    return sprite.astype(np.uint8)


def _build_atlas_array() -> np.ndarray:
    """Construct the (NUM_ROWS * 32, NUM_COLS * 32, 4) uint8 atlas image.

    Cells beyond a category's defined enum values are filled with
    MISSING_RGBA so future enum extensions produce a visible artifact
    rather than silent zeros.
    """
    height = NUM_ROWS * CELL_PX
    width = NUM_COLS * CELL_PX
    atlas = np.empty((height, width, 4), dtype=np.uint8)
    atlas[:, :] = MISSING_RGBA

    def _put(row: int, col: int, cell: np.ndarray) -> None:
        y0, x0 = row * CELL_PX, col * CELL_PX
        atlas[y0 : y0 + CELL_PX, x0 : x0 + CELL_PX] = cell

    block_textures = get_textures(CELL_PX)

    # Row 0: blocks.
    for block in sorted(BlockType, key=int):
        cell = _block_cell(block, block_textures)
        if cell is None:
            continue
        _put(ROW_BLOCKS, int(block), cell)

    # Rows 1..4: machines, one row per direction.
    for d_idx, direction in enumerate(_DIRECTION_ORDER):
        row = ROW_MACHINES_BASE + d_idx
        for machine in sorted(MachineType, key=int):
            # Non-directional machines render with a fixed fallback so
            # all four rows show the same sprite — keeps the gather
            # uniform without forcing the editor to know the difference.
            effective_dir = (
                direction if machine in _DIRECTIONAL_MACHINES else Direction.DOWN
            )
            cell = _machine_cell(machine, effective_dir)
            if cell is None:
                continue
            _put(row, int(machine), cell)

    # Row 5: items.
    for item in sorted(ItemType, key=int):
        cell = _item_cell(item)
        if cell is None:
            continue
        _put(ROW_ITEMS, int(item), cell)

    # Row 6: misc.
    _put(ROW_MISC, COL_MISC_BITER, _biter_cell())
    for d_idx, direction in enumerate(_DIRECTION_ORDER):
        _put(ROW_MISC, COL_MISC_PLAYER_BASE + d_idx, _player_cell(direction))

    # Row 7: digits.
    digit_atlas = np.asarray(build_digit_atlas())
    for digit in range(10):
        _put(ROW_DIGITS, digit, _digit_cell(digit, digit_atlas))

    return atlas


def _build_atlas_json() -> dict:
    """Construct the sidecar JSON describing the atlas layout."""
    direction_axis = [d.name for d in _DIRECTION_ORDER]
    return {
        "cell_px": CELL_PX,
        "rows": NUM_ROWS,
        "cols": NUM_COLS,
        "direction_axis": direction_axis,
        "categories": {
            "blocks": {
                "row": ROW_BLOCKS,
                "names": _ordered_enum_names(BlockType),
                "missing": "magenta",
            },
            "machines": {
                "rows": [ROW_MACHINES_BASE + i for i in range(NUM_DIRECTIONS)],
                "directions": direction_axis,
                "names": _ordered_enum_names(MachineType),
                "missing": "magenta",
                "directional": sorted(m.name for m in _DIRECTIONAL_MACHINES),
            },
            "items": {
                "row": ROW_ITEMS,
                "names": _ordered_enum_names(ItemType),
                "missing": "magenta",
            },
            "misc": {
                "row": ROW_MISC,
                "columns": {
                    "biter": COL_MISC_BITER,
                    **{
                        f"player_{d.name}": COL_MISC_PLAYER_BASE + i
                        for i, d in enumerate(_DIRECTION_ORDER)
                    },
                },
            },
            "digits": {
                "row": ROW_DIGITS,
                "names": [str(d) for d in range(10)],
            },
        },
    }


def build_atlas(out_png: Path, out_json: Path) -> None:
    """Build atlas.png and atlas.json into the given paths.

    The function is deterministic: running it twice into the same
    directory produces byte-identical files. CI verifies this via
    ``tests/test_atlas_fresh.py``.

    Args:
        out_png: Path to write the atlas PNG.
        out_json: Path to write the sidecar JSON.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    atlas = _build_atlas_array()
    iio.imwrite(out_png, atlas, extension=".png")

    payload = _build_atlas_json()
    out_json.write_bytes(
        orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
    )

    # Sanity-check the items row matches NUM_ITEM_TYPES — this catches
    # an enum extension landing without a matching atlas regen.
    assert len(payload["categories"]["items"]["names"]) == NUM_ITEM_TYPES


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "factoriax" / "assets",
        help="Output directory for atlas.png and atlas.json.",
    )
    args = parser.parse_args()
    out_png = args.out / "atlas.png"
    out_json = args.out / "atlas.json"
    build_atlas(out_png, out_json)
    logger.info("Wrote %s and %s", out_png, out_json)
    print(f"Wrote {out_png} ({out_png.stat().st_size} bytes)")
    print(f"Wrote {out_json} ({out_json.stat().st_size} bytes)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
