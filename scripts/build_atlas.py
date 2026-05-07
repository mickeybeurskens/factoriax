"""Build the FactoriaX sprite atlas from the procedural drawing code.

Generates ``factoriax/assets/atlas.png`` and the sidecar
``factoriax/assets/atlas.json`` per the layout documented in
``factoriax/assets/atlas.layout.md``. The JAX renderer reads sprites
from this atlas — every visual the play view shows for terrain,
machines, and the player flows through it.

The atlas is sourced from :mod:`factoriax.ui.icons`, which is the
procedural-art module that the editor and play HUD already use. By
funneling both renderers' sprites through the same code we avoid the
play-vs-editor visual divergence that motivated the regen. The icons
return RGBA; this script flattens to RGB by compositing each sprite
over a per-category background so transparent regions read sensibly
inside the JAX renderer's hard-overwrite ``render_map`` (no alpha
support there in v1).

Per-category background choices:

- Blocks: textures from :func:`factoriax.ui.icons.get_textures` are
  fully opaque, so the background is irrelevant; we just drop alpha.
- Machines: composite the icon's transparent corners over the legacy
  flat machine-body color (the same value the v1 atlas painted whole
  cells with). Placed machines therefore read as "machine-color block
  with detailed interior" rather than "detailed silhouette on black."
- Items: composite over black. Items aren't drawn by the world
  renderer; the atlas's items row is reserved for HUD work.
- Player / biter: composite over the dirt color (matches the most
  common floor in procedural maps; alternatives like black would
  paint a black square around the player).

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
NUM_ROWS: int = 5
NUM_COLS: int = 33  # max(num_block, num_machine, num_item, ...) — driven by ItemType
MISSING_COLOR: tuple[int, int, int] = (255, 0, 255)  # magenta — flags gaps


# Row indices — keep in sync with atlas.layout.md.
ROW_BLOCKS: int = 0
ROW_MACHINES: int = 1
ROW_ITEMS: int = 2
ROW_MISC: int = 3
ROW_DIGITS: int = 4


# Legacy machine body colors, used as the per-cell background when
# flattening RGBA machine icons to RGB. These match the colors the v1
# atlas painted whole cells with, so the new sprites preserve the
# silhouette of placed machines on the map.
_MACHINE_BODY_BG: dict[MachineType, tuple[int, int, int]] = {
    MachineType.NONE: (0, 0, 0),
    MachineType.MINER: (0, 200, 0),
    MachineType.PALLET: (170, 170, 175),
    MachineType.ASSEMBLER: (160, 80, 200),
    MachineType.CONVEYOR_BELT: (220, 180, 50),
    MachineType.ROCKET: (240, 240, 240),
    MachineType.SCIENCE_LAB: (76, 29, 149),
    MachineType.FURNACE: (180, 90, 50),
    MachineType.ARM: (90, 110, 130),
    MachineType.SPLITTER: (220, 180, 50),
    MachineType.CROSSING: (220, 180, 50),
}

# Player / biter sit on top of terrain; flatten over dirt so the
# transparent margin reads as floor rather than as a black border.
_DIRT_RGB: tuple[int, int, int] = (139, 90, 43)


def _flatten_rgba(rgba: np.ndarray, bg: tuple[int, int, int]) -> np.ndarray:
    """Composite an RGBA cell over a solid RGB background.

    Args:
        rgba: uint8 array of shape ``(H, W, 4)``.
        bg: RGB triple used wherever alpha < 255.

    Returns:
        uint8 array of shape ``(H, W, 3)``.
    """
    if rgba.shape[-1] == 3:
        return rgba.astype(np.uint8)
    rgb = rgba[..., :3].astype(np.float32)
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    bg_arr = np.broadcast_to(np.array(bg, dtype=np.float32), rgb.shape)
    return (rgb * alpha + bg_arr * (1.0 - alpha)).astype(np.uint8)


def _digit_cell(digit: int, digit_atlas: np.ndarray) -> np.ndarray:
    """Render a single digit at the cell's top-left.

    The procedural digit atlas is 3 wide × 5 tall bool mask. We blit
    it as white-on-black at the top-left of a 32x32 cell so the rest
    of the cell is unused space.
    """
    cell = np.zeros((CELL_PX, CELL_PX, 3), dtype=np.uint8)
    glyph = digit_atlas[digit]  # shape (5, 3) bool
    cell[: glyph.shape[0], : glyph.shape[1]] = np.where(
        glyph[:, :, None],
        np.array([255, 255, 255], dtype=np.uint8),
        np.array([0, 0, 0], dtype=np.uint8),
    )
    return cell


def _ordered_enum_names(enum_cls: type) -> list[str]:
    """Return enum member names sorted by integer value."""
    return [m.name for m in sorted(enum_cls, key=int)]


def _block_cell(block: BlockType, textures: dict[int, np.ndarray]) -> np.ndarray | None:
    """Return the RGB block sprite for *block*, or ``None`` to leave magenta."""
    tex = textures.get(int(block))
    if tex is None:
        return None
    return _flatten_rgba(tex, _DIRT_RGB)


def _machine_cell(machine: MachineType) -> np.ndarray | None:
    """Return the RGB machine sprite for *machine*, or ``None`` to leave magenta.

    The cell is the corresponding item's icon (from MACHINE_TO_ITEM)
    composited over the legacy machine body color so transparent
    corners pick up that color rather than black.
    """
    if machine == MachineType.NONE:
        bg = _MACHINE_BODY_BG[MachineType.NONE]
        return np.full((CELL_PX, CELL_PX, 3), bg, dtype=np.uint8)
    item_id = MACHINE_TO_ITEM.get(int(machine))
    if item_id is None:
        return None
    bg = _MACHINE_BODY_BG.get(machine, (0, 0, 0))
    icon = render_item_icon(item_id, CELL_PX, direction=int(Direction.DOWN))
    return _flatten_rgba(icon, bg)


def _item_cell(item: ItemType) -> np.ndarray | None:
    """Return the RGB item sprite for *item*, or ``None`` to leave magenta.

    Items aren't currently drawn by the world renderer, so the
    background choice (black) is purely for the HUD strip and
    debugging atlas inspectors.
    """
    if item == ItemType.EMPTY:
        return np.zeros((CELL_PX, CELL_PX, 3), dtype=np.uint8)
    icon = render_item_icon(int(item), CELL_PX, direction=int(Direction.DOWN))
    return _flatten_rgba(icon, (0, 0, 0))


def _player_cell() -> np.ndarray:
    """Return the RGB player sprite (player 0, facing down)."""
    sprite = create_player_texture(
        direction=int(Direction.DOWN),
        player_idx=0,
        is_selected=True,
        size=CELL_PX,
    )
    return _flatten_rgba(sprite, _DIRT_RGB)


def _biter_cell() -> np.ndarray:
    """Return the RGB biter sprite."""
    sprite = create_biter_texture(CELL_PX)
    return _flatten_rgba(sprite, _DIRT_RGB)


def _build_atlas_array() -> np.ndarray:
    """Construct the (160, 1056, 3) uint8 atlas image.

    Cells beyond a category's defined enum values are filled with
    MISSING_COLOR so future enum extensions produce a visible
    artifact rather than silent zeros.
    """
    height = NUM_ROWS * CELL_PX
    width = NUM_COLS * CELL_PX
    atlas = np.empty((height, width, 3), dtype=np.uint8)
    atlas[:, :] = MISSING_COLOR

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

    # Row 1: machines.
    for machine in sorted(MachineType, key=int):
        cell = _machine_cell(machine)
        if cell is None:
            continue
        _put(ROW_MACHINES, int(machine), cell)

    # Row 2: items.
    for item in sorted(ItemType, key=int):
        cell = _item_cell(item)
        if cell is None:
            continue
        _put(ROW_ITEMS, int(item), cell)

    # Row 3: misc.
    _put(ROW_MISC, 0, _player_cell())
    _put(ROW_MISC, 1, _biter_cell())

    # Row 4: digits.
    digit_atlas = np.asarray(build_digit_atlas())
    for digit in range(10):
        _put(ROW_DIGITS, digit, _digit_cell(digit, digit_atlas))

    return atlas


def _build_atlas_json() -> dict:
    """Construct the sidecar JSON describing the atlas layout."""
    return {
        "cell_px": CELL_PX,
        "rows": NUM_ROWS,
        "cols": NUM_COLS,
        "categories": {
            "blocks": {
                "row": ROW_BLOCKS,
                "names": _ordered_enum_names(BlockType),
                "missing": "magenta",
            },
            "machines": {
                "row": ROW_MACHINES,
                "names": _ordered_enum_names(MachineType),
                "missing": "magenta",
            },
            "items": {
                "row": ROW_ITEMS,
                "names": _ordered_enum_names(ItemType),
                "missing": "magenta",
            },
            "misc": {
                "row": ROW_MISC,
                "names": ["player", "biter"],
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
