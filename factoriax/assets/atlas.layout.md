# Sprite Atlas Layout

Source of truth for `factoriax/assets/atlas.png` and its sidecar
`factoriax/assets/atlas.json`. Both are regenerated from
`scripts/build_atlas.py`; CI verifies they match this layout via
`tests/test_atlas_fresh.py`.

## Grid

The atlas is a uniform grid of 32×32 px **RGBA** cells, packed
**row-major by category, column-by-enum-value**. Indexing a sprite
is two integers: `(row, col)`. The renderer reads them via
`atlas.json`, which the `JaxRenderer` slurps at construction time.

Cell size is fixed at 32×32. Display tile size (`tile_px`) is a
renderer-construction parameter; downsampling from 32×32 to the
display size happens at gather time using nearest-neighbour.
Upsampling (rare; for "zoomed-in" views) uses the same path.

The atlas image dimensions are `(num_rows × 32, num_cols × 32, 4)`
uint8. The alpha channel is meaningful: block cells are fully opaque
(terrain is the ground truth and always paints), while machine and
player cells carry transparent regions so `render_map` can blend
them onto whatever sits beneath them. Walking onto a placed belt no
longer hides the belt because the player sprite's alpha falls
through to the belt underneath.

## Rows

Each row corresponds to one enum category. Cells beyond the enum's
defined values are filled with the row's "missing" color (magenta,
`(255, 0, 255, 255)`) so a future enum extension produces a visible
artifact rather than silent zeros.

| Row | Category               | Source enum                      | Cells | Notes |
| --: | ---------------------- | -------------------------------- | ----: | ----- |
|   0 | blocks                 | `factoriax.constants.BlockType`  | 11    | Sourced from `factoriax.ui.icons.get_textures`. Alpha forced to 255 — terrain is opaque. |
|   1 | machines, dir LEFT     | `factoriax.constants.MachineType`| 11    | Directional machines render with `direction=LEFT`; non-directional machines duplicate the DOWN sprite. |
|   2 | machines, dir RIGHT    | `factoriax.constants.MachineType`| 11    | Same, with `direction=RIGHT`. |
|   3 | machines, dir UP       | `factoriax.constants.MachineType`| 11    | Same, with `direction=UP`. |
|   4 | machines, dir DOWN     | `factoriax.constants.MachineType`| 11    | Same, with `direction=DOWN`. |
|   5 | items                  | `factoriax.constants.ItemType`   | 33    | Sourced from `render_item_icon`. Currently unused by `render_map`; reserved for future HUD work. |
|   6 | misc                   | (manually enumerated)            | 33    | col 0 = biter; cols 1..32 hold 8 players × 4 directions packed as `(player_idx, direction)` starting at col 1, with `col = 1 + player_idx * 4 + direction_idx`. Player slots beyond 8 wrap modulo 8. |
|   7 | digits                 | digits 0-9                       | 10    | Each cell is 32×32; the 3×5 glyph is rendered at the cell's top-left, padded with zeros. Alpha=255. |

Width of the atlas is `max(num_cells_per_row) = 33` (driven by
`ItemType`). Height is `num_rows = 8`. Atlas image:
`(8 × 32, 33 × 32, 4) = (256, 1056, 4)` uint8.

### Direction axis

The directional machine rows (1-4) and the player columns (1-4 of
the misc row) share a fixed axis order: `[LEFT, RIGHT, UP, DOWN]`.
This is also published as `direction_axis` in the sidecar JSON.
Atlas index = `Direction value − 1`, mapping the engine's
`Direction` enum (LEFT=1, RIGHT=2, UP=3, DOWN=4) onto the four
rows / columns.

Directional machines (variants drawn per direction):
`CONVEYOR_BELT`, `MINER`, `ARM`, `SPLITTER`, `CROSSING`. All other
machines (`PALLET`, `ASSEMBLER`, `FURNACE`, `SCIENCE_LAB`,
`ROCKET`, `NONE`) render once with `direction=DOWN` and that sprite
is duplicated across all four direction rows so the renderer's
gather is uniform.

Player slots use the same `[LEFT, RIGHT, UP, DOWN]` axis but are
also keyed by `player_idx`. Eight palettes are baked in (the
distinct entries of `PLAYER_COLORS` in `factoriax/ui/icons.py`);
players 8 and beyond reuse palette 0 onwards via modulo-8.

## Sidecar JSON shape

`factoriax/assets/atlas.json` is a single object:

```json
{
  "cell_px": 32,
  "rows": 8,
  "cols": 33,
  "direction_axis": ["LEFT", "RIGHT", "UP", "DOWN"],
  "categories": {
    "blocks":   {"row": 0, "names": ["INVALID", "OUT_OF_BOUNDS", ...], "missing": "magenta"},
    "machines": {
      "rows": [1, 2, 3, 4],
      "directions": ["LEFT", "RIGHT", "UP", "DOWN"],
      "names": ["NONE", "MINER", ...],
      "directional": ["ARM", "CONVEYOR_BELT", "CROSSING", "MINER", "SPLITTER"],
      "missing": "magenta"
    },
    "items":    {"row": 5, "names": ["EMPTY", "COAL", ...], "missing": "magenta"},
    "misc":     {
      "row": 6,
      "num_players": 8,
      "directions": ["LEFT", "RIGHT", "UP", "DOWN"],
      "columns": {
        "biter": 0,
        "player0_LEFT": 1, "player0_RIGHT": 2, "player0_UP": 3, "player0_DOWN": 4,
        "player1_LEFT": 5, "...": "...", "player7_DOWN": 32
      }
    },
    "digits":   {"row": 7, "names": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]}
  }
}
```

Names within each category appear in enum-value order (so
`names[i]` lives at `(row, i)` for non-directional categories). The
machine row lookup is `rows[direction_axis.index(dir)]`. For v1
the renderer hardcodes the row constants rather than parsing the
JSON, but the JSON is canonical for tooling and tests.

## Determinism

The build script must produce byte-identical output across runs:

- Sort enum members by integer value before iterating.
- Use `np.full(..., dtype=np.uint8)` for backing arrays — no
  uninitialised memory.
- Encode the PNG with `imageio.imwrite(..., compress_level=6)`
  (default) and a fixed metadata block.
- Encode the JSON via `orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)`.

`tests/test_atlas_fresh.py` regenerates both files in a tempdir and
asserts byte-equivalence with the committed copies.

## Future extensions (not in scope yet)

- **Animation frames**: extra columns per cell for sprite cycling
  (e.g. conveyor flow). Renderer reads
  `(row, base_col + state.timestep % num_frames)`.
- **Status overlays**: separate rows for "machine has output",
  "machine starved of input", etc. — visual debugging aids.
- **Glyph atlas**: a richer text atlas to phase out the pygame text
  overlay. Out of scope for v1 by spec decision.
- **Per-player directional sprites beyond 8 slots**: currently the
  misc row caps at 8 distinct palettes. A 9th-or-later player
  recycles palette 0 onwards. Adding a dedicated row block per
  player (or sourcing palettes generatively) would lift the cap.

These are listed so the layout doesn't silently invalidate them.
Adding a row at the bottom is non-breaking; reordering existing
rows would be breaking. Don't reorder.
