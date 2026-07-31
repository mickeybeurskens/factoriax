"""Build the Factoriax sprite atlas from the procedural drawing code.

Generates ``factoriax/assets/atlas.png`` and the sidecar
``factoriax/assets/atlas.json`` per the layout documented in
``factoriax/assets/atlas.layout.md``. The JAX renderer reads sprites
from this atlas. Every visual that the play view shows for terrain,
for machines, and for the player flows through it.

The sprites come from :mod:`factoriax.playground.ui.icons`, the drawing
module that the editor and the play HUD already read. Both renderers
therefore take one set of sprites, and the editor cannot drift away from the
play view. The earlier atlas of flat colors did drift.

The atlas is RGBA. :func:`render_map` of the JAX renderer mixes the machine
layer and the player layer onto the terrain, through the alpha channel. Three
rules follow from this:

- Block cells are opaque (alpha=255 everywhere). Terrain is the
  ground truth and always paints in full.
- Machine cells keep the transparent corners that
  :func:`render_item_icon` produces, so placed machines read as
  objects sitting on terrain.
- A player cell and a biter cell mix with the layer under them. A player
  that steps onto a belt therefore does not hide that belt.

Directional categories carry one cell per
:class:`~factoriax.engine.constants.Direction`:

- Machines: rows 1-4 hold variants for direction LEFT, RIGHT, UP,
  DOWN respectively. Non-directional machines (``PALLET``,
  ``ASSEMBLER``, ``FURNACE``, ``SCIENCE_LAB``, ``ROCKET``) are
  rendered once and duplicated across all four rows so a uniform
  gather works at render time.
- Player: misc row col 0 holds the biter. Cols 1..32 hold eight
  players × four directions packed as ``(player_idx, direction)``,
  so player ``p`` direction ``d_idx`` lives at
  ``1 + p * 4 + d_idx``. Players beyond eight wrap modulo eight to
  match :data:`factoriax.playground.ui.icons.PLAYER_COLORS`.

    uv run python -m factoriax.assets.build_atlas
    uv run python -m factoriax.assets.build_atlas --out custom_dir/

Two runs of this module give the same bytes. ``tests/test_atlas_fresh.py``
makes sure of this, and CI runs that test.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import orjson

from factoriax.engine.constants import (
    NUM_ITEM_TYPES,
    BlockType,
    Direction,
    ItemType,
    Machine,
)
from factoriax.engine.renderer import build_digit_atlas
from factoriax.playground.ui.icons import (
    MACHINE_TO_ITEM,
    create_biter_texture,
    create_player_texture,
    get_textures,
    render_item_icon,
)

logger = logging.getLogger(__name__)

CELL_PX: int = 32
# max(num_block, num_machine, num_item, ...). ItemType is the widest row.
NUM_COLS: int = NUM_ITEM_TYPES


# Direction order used for the atlas's directional rows / columns.
# Index = atlas row offset for machines (0 = LEFT row, 3 = DOWN row),
# and atlas column offset for the player cells in the misc row
# (1 + index gives the column, leaving col 0 for the biter sprite).
# Biters were removed from the game. The cell stays, because a cell removed
# here moves every player column and rewrites the committed atlas.
_DIRECTION_ORDER: tuple[Direction, ...] = (
    Direction.LEFT,
    Direction.RIGHT,
    Direction.UP,
    Direction.DOWN,
)
NUM_DIRECTIONS: int = len(_DIRECTION_ORDER)


# Row indices, kept in sync with atlas.layout.md and with the constants in
# factoriax/engine/renderer.py.
ROW_BLOCKS: int = 0
ROW_MACHINES_BASE: int = 1  # rows 1..4 hold one direction each.
ROW_ITEMS: int = ROW_MACHINES_BASE + NUM_DIRECTIONS  # 5
ROW_MISC: int = ROW_ITEMS + 1  # 6
ROW_DIGITS: int = ROW_MISC + 1  # 7
NUM_ROWS: int = ROW_DIGITS + 1  # 8

# Misc row column layout.
COL_MISC_BITER: int = 0
COL_MISC_PLAYER_BASE: int = 1
# Eight players × four directions = 32 player cells, packed as
# (player_idx, direction) starting at COL_MISC_PLAYER_BASE. Player p
# direction d sits at COL_MISC_PLAYER_BASE + p * NUM_DIRECTIONS + d_idx.
# Eight matches the distinct PLAYER_COLORS palette in
# factoriax/ui/icons.py. Players beyond 8 wrap modulo 8.
NUM_PLAYERS: int = 8


# Magenta with alpha=255 acts as the "missing sprite" sentinel. The
# alpha is non-zero so a missing entry actually paints, surfacing the
# gap visually rather than silently blending through.
MISSING_RGBA: tuple[int, int, int, int] = (255, 0, 255, 255)


# Directional machines. These get a distinct sprite per direction.
# Non-directional machines (PALLET, ASSEMBLER, FURNACE, SCIENCE_LAB,
# ROCKET, NONE) are rendered once and duplicated across all four
# direction rows.
_DIRECTIONAL_MACHINES: frozenset[Machine] = frozenset(
    {
        Machine.CONVEYOR_BELT,
        Machine.MINER,
        Machine.ARM,
        Machine.SPLITTER,
        Machine.CROSSING,
    }
)


def _drop_alpha(rgba: np.ndarray) -> np.ndarray:
    """Return one cell without its alpha channel.

    Parameters
    ----------
    rgba
        Cell with four channels.

    Returns
    -------
    numpy.ndarray
        The same cell with three channels, uint8.
    """
    return rgba[..., :3].astype(np.uint8)


def _to_rgba(rgb: np.ndarray, alpha: int = 255) -> np.ndarray:
    """Return one cell with an alpha channel added.

    Parameters
    ----------
    rgb
        Cell with three channels.
    alpha
        Alpha value for every pixel.

    Returns
    -------
    numpy.ndarray
        The same cell with four channels, uint8.
    """
    h, w = rgb.shape[:2]
    out = np.empty((h, w, 4), dtype=np.uint8)
    out[..., :3] = rgb[..., :3]
    out[..., 3] = alpha
    return out


def _digit_cell(digit: int, digit_atlas: np.ndarray) -> np.ndarray:
    """Draw one digit at the top left corner of a cell.

    A digit glyph is a mask 3 pixels wide and 5 pixels tall. The function
    draws it in white on black, at the top left corner. The rest of the cell
    stays black.

    Parameters
    ----------
    digit
        Digit to draw, from 0 to 9.
    digit_atlas
        Glyph mask of each digit.

    Returns
    -------
    numpy.ndarray
        One cell, with an alpha of 255 at every pixel.
    """
    cell = np.zeros((CELL_PX, CELL_PX, 4), dtype=np.uint8)
    cell[..., 3] = 255
    glyph = digit_atlas[digit]  # shape (5, 3) bool
    white = np.array([255, 255, 255, 255], dtype=np.uint8)
    black = np.array([0, 0, 0, 255], dtype=np.uint8)
    cell[: glyph.shape[0], : glyph.shape[1]] = np.where(glyph[:, :, None], white, black)
    return cell


def _ordered_enum_names(enum_cls: type) -> list[str]:
    """Return the name of each member of an enum, in value order.

    The sidecar JSON lists the names in this order, so a reader can find a
    cell by the value of its enum member.

    Parameters
    ----------
    enum_cls
        Enum to read.

    Returns
    -------
    list[str]
        One name for each member, from the lowest value to the highest.
    """
    return [m.name for m in sorted(enum_cls, key=int)]


def _block_cell(block: BlockType, textures: dict[int, np.ndarray]) -> np.ndarray | None:
    """Return the atlas cell of one block type.

    Terrain is the lowest layer, so the cell is opaque at every pixel.

    Parameters
    ----------
    block
        Block type to draw.
    textures
        Texture of each block type, by block value.

    Returns
    -------
    numpy.ndarray or None
        The cell. It is ``None`` when no texture matches the block, and the
        atlas then keeps its magenta marker.
    """
    tex = textures.get(int(block))
    if tex is None:
        return None
    if tex.shape[-1] == 4:
        # Force opaque. Terrain is the ground truth layer.
        out = tex.copy()
        out[..., 3] = 255
        return out
    return _to_rgba(tex)


def _machine_cell(machine: Machine, direction: Direction) -> np.ndarray | None:
    """Return the atlas cell of one machine, at one facing.

    A machine with one look only takes the same cell for all four facings.

    Parameters
    ----------
    machine
        Machine to draw.
    direction
        Facing to draw the machine at.

    Returns
    -------
    numpy.ndarray or None
        The cell. ``Machine.NONE`` gives a cell that is clear at every pixel,
        so it never paints. The result is ``None`` when no item matches the
        machine, and the atlas then keeps its magenta marker.
    """
    if machine == Machine.NONE:
        # Clear at every pixel, so the alpha mix leaves the terrain as it is.
        return np.zeros((CELL_PX, CELL_PX, 4), dtype=np.uint8)
    item_id = MACHINE_TO_ITEM.get(int(machine))
    if item_id is None:
        return None
    icon = render_item_icon(item_id, CELL_PX, direction=int(direction))
    if icon.shape[-1] == 3:
        return _to_rgba(icon)
    return icon.astype(np.uint8)


def _item_cell(item: ItemType) -> np.ndarray | None:
    """Return the atlas cell of one item.

    The world renderer draws no item today. These cells wait for later work
    on the HUD.

    Parameters
    ----------
    item
        Item to draw.

    Returns
    -------
    numpy.ndarray or None
        The cell. ``ItemType.EMPTY`` gives a cell that is clear at every
        pixel.
    """
    if item == ItemType.EMPTY:
        return np.zeros((CELL_PX, CELL_PX, 4), dtype=np.uint8)
    icon = render_item_icon(int(item), CELL_PX, direction=int(Direction.DOWN))
    if icon.shape[-1] == 3:
        return _to_rgba(icon)
    return icon.astype(np.uint8)


def _player_cell(player_idx: int, direction: Direction) -> np.ndarray:
    """Return the atlas cell of one player, at one facing.

    Each player takes its own colors from
    :data:`factoriax.playground.ui.icons.PLAYER_COLORS`.

    Parameters
    ----------
    player_idx
        Index of the player. It selects the colors.
    direction
        Facing to draw the player at.

    Returns
    -------
    numpy.ndarray
        The cell, with four channels.
    """
    sprite = create_player_texture(
        direction=int(direction),
        player_idx=player_idx,
        is_selected=True,
        size=CELL_PX,
    )
    if sprite.shape[-1] == 3:
        return _to_rgba(sprite)
    return sprite.astype(np.uint8)


def _biter_cell() -> np.ndarray:
    """Return the atlas cell of the biter."""
    sprite = create_biter_texture(CELL_PX)
    if sprite.shape[-1] == 3:
        return _to_rgba(sprite)
    return sprite.astype(np.uint8)


def _build_atlas_array() -> np.ndarray:
    """Construct the (NUM_ROWS * 32, NUM_COLS * 32, 4) uint8 atlas image.

    A cell past the last member of a category holds ``MISSING_RGBA``. A new
    enum member therefore shows a magenta cell, and not a clear one.

    Returns
    -------
    numpy.ndarray
        The atlas image, of shape ``(NUM_ROWS * 32, NUM_COLS * 32, 4)``,
        uint8.
    """
    height = NUM_ROWS * CELL_PX
    width = NUM_COLS * CELL_PX
    atlas = np.empty((height, width, 4), dtype=np.uint8)
    atlas[:, :] = MISSING_RGBA

    def _put(row: int, col: int, cell: np.ndarray) -> None:
        """Write one cell into the atlas, at one row and one column."""
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
        for machine in sorted(Machine, key=int):
            # Non-directional machines render with a fixed fallback so
            # all four rows show the same sprite. This keeps the gather
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

    # Row 6: misc. Biter plus per-player directional sprites.
    _put(ROW_MISC, COL_MISC_BITER, _biter_cell())
    for player_idx in range(NUM_PLAYERS):
        for d_idx, direction in enumerate(_DIRECTION_ORDER):
            col = COL_MISC_PLAYER_BASE + player_idx * NUM_DIRECTIONS + d_idx
            _put(ROW_MISC, col, _player_cell(player_idx, direction))

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
                "names": _ordered_enum_names(Machine),
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
                        f"player{p}_{d.name}": (
                            COL_MISC_PLAYER_BASE + p * NUM_DIRECTIONS + i
                        )
                        for p in range(NUM_PLAYERS)
                        for i, d in enumerate(_DIRECTION_ORDER)
                    },
                },
                "num_players": NUM_PLAYERS,
                "directions": direction_axis,
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

    Parameters
    ----------
    out_png :
        Path to write the atlas PNG.
    out_json :
        Path to write the sidecar JSON.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    atlas = _build_atlas_array()
    iio.imwrite(out_png, atlas, extension=".png")

    payload = _build_atlas_json()
    out_json.write_bytes(
        orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
    )

    # Sanity-check the items row matches NUM_ITEM_TYPES. This catches
    # an enum extension that lands without a matching atlas regen.
    assert len(payload["categories"]["items"]["names"]) == NUM_ITEM_TYPES


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent,
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
