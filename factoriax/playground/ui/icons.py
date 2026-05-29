"""Procedural icon and texture generation for play, editor, and menu UI."""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

from factoriax.engine.constants import (
    ITEM_TO_MACHINE,
    BlockType,
    Direction,
    ItemType,
)
from factoriax.playground.ui.theme import BLOCK_PIXEL_SIZE

ASSETS_PATH = Path(__file__).parent.parent / "assets"

ITEM_COLORS: dict[int, tuple[int, int, int]] = {
    ItemType.COAL: (54, 54, 54),
    ItemType.IRON_ORE: (160, 140, 130),
    ItemType.COPPER_ORE: (170, 100, 50),
    ItemType.TIN_ORE: (170, 180, 185),
    ItemType.SILICON: (80, 105, 140),
    ItemType.IRON_PLATE: (192, 192, 192),
    ItemType.COPPER_PLATE: (184, 115, 51),
    ItemType.TIN_PLATE: (200, 200, 190),
    ItemType.WAFER: (80, 90, 140),
    ItemType.FRAME: (140, 150, 165),
    ItemType.CIRCUIT: (40, 160, 80),
    ItemType.WIRE: (200, 140, 60),
    ItemType.MOTOR: (100, 100, 180),
    ItemType.SENSOR: (180, 80, 80),
    ItemType.CONVEYOR_BELT: (220, 180, 50),
    ItemType.MINER: (0, 200, 0),
    ItemType.ASSEMBLER: (160, 80, 200),
    ItemType.PALLET: (170, 170, 175),
    ItemType.ARM: (220, 160, 100),
    ItemType.BASIC_SCIENCE_PACK: (200, 50, 50),
    ItemType.ADVANCED_SCIENCE_PACK: (50, 50, 200),
    ItemType.ROCKET: (240, 240, 240),
    ItemType.FURNACE: (120, 60, 40),
    ItemType.REFRACTORY: (210, 170, 120),
    ItemType.HULL: (150, 160, 170),
    ItemType.ENGINE_UNIT: (220, 140, 60),
    ItemType.AVIONICS: (80, 200, 220),
    ItemType.ROCKET_CORE: (160, 100, 200),
    ItemType.SCIENCE_LAB: (76, 29, 149),
    ItemType.LIMESTONE: (215, 200, 165),
    ItemType.SPLITTER: (240, 195, 70),
    ItemType.CROSSING: (200, 165, 60),
}

# ---------------------------------------------------------------------------
# Block textures
# ---------------------------------------------------------------------------

# Tile colours per BlockType. Single source of truth for the procedural
# textures used by :func:`create_default_textures` (and therefore the
# baked sprite atlas at ``factoriax/assets/atlas.png``) and the paper's
# reset-grid figure, so the engine render and the paper plot agree on
# what each resource looks like.
BLOCK_COLORS: dict[int, tuple[int, int, int]] = {
    int(BlockType.DIRT): (201, 167, 121),
    int(BlockType.WATER): (50, 120, 190),
    int(BlockType.IRON): (111, 138, 166),
    int(BlockType.COPPER): (209, 138, 74),
    int(BlockType.COAL): (43, 43, 43),
    int(BlockType.TIN): (185, 185, 185),
    int(BlockType.SILICON): (111, 90, 138),
    int(BlockType.LIMESTONE): (232, 220, 181),
}

# Ores rendered with a crystal-pattern overlay rather than the default
# blob pattern. Currently SILICON only; the set leaves room for future
# crystalline ores without touching the texture code.
_CRYSTALLINE_ORES: frozenset[int] = frozenset({int(BlockType.SILICON)})

_SOLID_BLOCKS: tuple[int, ...] = (int(BlockType.DIRT), int(BlockType.WATER))
_ORE_BLOCKS: tuple[int, ...] = (
    int(BlockType.IRON),
    int(BlockType.COPPER),
    int(BlockType.COAL),
    int(BlockType.TIN),
    int(BlockType.SILICON),
    int(BlockType.LIMESTONE),
)


def _resize_texture(texture: np.ndarray, size: int) -> np.ndarray:
    """Resize a texture to *size* × *size* using nearest-neighbour sampling.

    Args:
        texture: RGBA array of shape (H, W, 4).
        size: Target side length in pixels.

    Returns:
        RGBA array of shape (size, size, 4).
    """
    src_size = texture.shape[0]
    if src_size == size:
        return texture
    idx = np.round(np.linspace(0, src_size - 1, size)).astype(int)
    resized: np.ndarray = texture[np.ix_(idx, idx)]
    return resized


def _solid_texture(
    size: int,
    color: tuple[int, int, int],
) -> np.ndarray:
    """Create a solid-color RGBA texture.

    Args:
        size: Side length in pixels.
        color: RGB color.

    Returns:
        RGBA uint8 array of shape ``(size, size, 4)``.
    """
    t = np.empty((size, size, 4), dtype=np.uint8)
    t[:, :] = (*color, 255)
    return t


def create_default_textures(size: int = BLOCK_PIXEL_SIZE) -> dict[int, np.ndarray]:
    """Create patterned block textures without touching disk.

    Ore blocks get the same irregular-patches pattern used by their
    inventory icons so the tile, the mined item, and any plate that
    descends from it all share a visual family. Non-ore blocks stay
    solid.

    Args:
        size: Side length of each texture in pixels.

    Returns:
        Dictionary mapping BlockType values to RGBA texture arrays.
    """
    textures: dict[int, np.ndarray] = {
        block_id: _solid_texture(size, BLOCK_COLORS[block_id])
        for block_id in _SOLID_BLOCKS
    }
    for block_id in _ORE_BLOCKS:
        rgb = BLOCK_COLORS[block_id]
        tex = _solid_texture(size, rgb)
        _draw_ore_patches(
            tex, rgb, seed=block_id, crystalline=block_id in _CRYSTALLINE_ORES
        )
        textures[block_id] = tex
    return textures


def load_texture(name: str) -> np.ndarray:
    """Load a texture from the assets directory.

    Args:
        name: Name of the texture file (without extension).

    Returns:
        RGBA numpy array of shape (BLOCK_PIXEL_SIZE, BLOCK_PIXEL_SIZE, 4).

    Raises:
        FileNotFoundError: If the texture file does not exist.
    """
    import imageio.v3 as iio

    path = ASSETS_PATH / f"{name}.png"
    if not path.exists():
        raise FileNotFoundError(f"Texture not found: {path}")
    return iio.imread(path)


def load_all_textures() -> dict[int, np.ndarray]:
    """Load all block textures into a dictionary.

    Returns:
        Dictionary mapping BlockType values to RGBA texture arrays.
    """
    textures: dict[int, np.ndarray] = {}
    texture_names = {
        BlockType.DIRT: "dirt",
        BlockType.WATER: "water",
        BlockType.IRON: "iron",
        BlockType.COPPER: "copper",
        BlockType.COAL: "coal",
        BlockType.TIN: "tin",
        BlockType.SILICON: "silicon",
        BlockType.LIMESTONE: "limestone",
    }
    for block_type, name in texture_names.items():
        textures[int(block_type)] = load_texture(name)
    return textures


@functools.lru_cache(maxsize=8)
def get_textures(size: int = BLOCK_PIXEL_SIZE) -> dict[int, np.ndarray]:
    """Load textures from files, falling back to defaults if not found.

    Results are cached per ``size`` so disk I/O and resizing happen at
    most once per unique block pixel size across the entire process.

    Args:
        size: Required texture side length in pixels.

    Returns:
        Dictionary mapping BlockType values to RGBA texture arrays
    """
    try:
        raw = load_all_textures()
        return {k: _resize_texture(v, size) for k, v in raw.items()}
    except FileNotFoundError:
        return create_default_textures(size)


@functools.lru_cache(maxsize=8)
def build_texture_lookup(size: int) -> np.ndarray:
    """Build a dense texture lookup array indexed by block type.

    Allows tile rendering via a single numpy advanced-index operation
    instead of a Python loop over every map cell.

    Args:
        size: Block pixel size.

    Returns:
        Array of shape (max_block_id + 1, size, size, 4).
    """
    textures = get_textures(size)
    max_id = max(textures.keys())
    lookup = np.zeros((max_id + 1, size, size, 4), dtype=np.uint8)
    default = textures[int(BlockType.DIRT)]
    for i in range(max_id + 1):
        lookup[i] = textures.get(i, default)
    return lookup


# ---------------------------------------------------------------------------
# Player and biter sprites
# ---------------------------------------------------------------------------


PLAYER_COLORS = [
    ([255, 100, 100], [200, 50, 50]),  # Player 0: Red
    ([100, 100, 255], [50, 50, 200]),  # Player 1: Blue
    ([100, 255, 100], [50, 200, 50]),  # Player 2: Green
    ([255, 255, 100], [200, 200, 50]),  # Player 3: Yellow
    ([255, 100, 255], [200, 50, 200]),  # Player 4: Magenta
    ([100, 255, 255], [50, 200, 200]),  # Player 5: Cyan
    ([255, 180, 100], [200, 130, 50]),  # Player 6: Orange
    ([180, 100, 255], [130, 50, 200]),  # Player 7: Purple
    ([180, 255, 180], [130, 200, 130]),  # Player 8: Light green
]


def create_player_texture(
    direction: int = Direction.DOWN,
    player_idx: int = 0,
    is_selected: bool = True,
    size: int = BLOCK_PIXEL_SIZE,
) -> np.ndarray:
    """Create a player texture with directional indicator.

    The player is rendered as a circle with a small triangle indicating
    the direction they are facing. Different players have different colors,
    and the selected player has a highlight ring.

    Args:
        direction: The direction the player is facing (Action enum value)
        player_idx: Index of the player (determines color)
        is_selected: Whether this player is currently selected
        size: Side length of the texture in pixels

    Returns:
        RGBA numpy array of shape (size, size, 4)
    """
    player = np.zeros((size, size, 4), dtype=np.uint8)
    center = size // 2
    radius = size // 3

    color_idx = player_idx % len(PLAYER_COLORS)
    body_color, indicator_color = PLAYER_COLORS[color_idx]

    ys, xs = np.ogrid[:size, :size]
    dist = np.sqrt((xs - center) ** 2 + (ys - center) ** 2)

    player[dist <= radius] = [*body_color, 255]
    if is_selected:
        player[(dist > radius) & (dist <= radius + 2)] = [255, 255, 255, 255]

    indicator_size = max(2, size // 6)

    # Build indicator triangle via vectorised row/column masks.
    rows = np.arange(indicator_size)
    pys: int | np.ndarray
    centers_x: int | np.ndarray
    if direction == Direction.UP:
        pys = 1 + rows
        centers_x = center
    elif direction == Direction.DOWN:
        pys = size - 2 - rows
        centers_x = center
    elif direction == Direction.LEFT:
        pys = center
        centers_x = 1 + rows
    else:  # RIGHT
        pys = center
        centers_x = size - 2 - rows

    for i in range(indicator_size):
        offsets = np.arange(-i, i + 1)
        if direction in (Direction.UP, Direction.DOWN):
            py = int(pys[i])  # type: ignore[index]
            pxs = np.clip(centers_x + offsets, 0, size - 1)
            player[py, pxs] = [*indicator_color, 255]
        else:
            px = int(centers_x[i])  # type: ignore[index]
            pys_clipped = np.clip(pys + offsets, 0, size - 1)
            player[pys_clipped, px] = [*indicator_color, 255]

    return player


def create_player_start_icon(
    player_idx: int = 0, size: int = BLOCK_PIXEL_SIZE
) -> np.ndarray:
    """Create a player start icon as a colored circle with a white X.

    Used in the map editor to mark spawn positions. The circle colour
    comes from ``PLAYER_COLORS[player_idx]``.

    Args:
        player_idx: Player index (determines colour).
        size: Side length of the icon in pixels.

    Returns:
        RGBA numpy array of shape ``(size, size, 4)``.
    """
    icon = np.zeros((size, size, 4), dtype=np.uint8)
    center = size // 2
    radius = max(2, size // 3)
    color_idx = player_idx % len(PLAYER_COLORS)
    body_color = PLAYER_COLORS[color_idx][0]

    ys, xs = np.ogrid[:size, :size]
    dist = np.sqrt((xs - center) ** 2 + (ys - center) ** 2)
    icon[dist <= radius] = [*body_color, 255]

    # Draw a white X inside the circle.
    arm = max(1, radius - 1)
    for d in range(-arm, arm + 1):
        for px, py in ((center + d, center + d), (center + d, center - d)):
            if 0 <= px < size and 0 <= py < size and dist[py, px] <= radius:
                icon[py, px] = [255, 255, 255, 255]

    return icon


BITER_COLOR = (180, 40, 40)


def create_biter_texture(size: int = BLOCK_PIXEL_SIZE) -> np.ndarray:
    """Create a biter texture as a small red circle.

    Args:
        size: Side length of the texture in pixels.

    Returns:
        RGBA numpy array of shape (size, size, 4).
    """
    texture = np.zeros((size, size, 4), dtype=np.uint8)
    center = size // 2
    radius = size // 4
    ys, xs = np.ogrid[:size, :size]
    dist = np.sqrt((xs - center) ** 2 + (ys - center) ** 2)
    texture[dist <= radius] = [*BITER_COLOR, 220]
    # Dark outline.
    texture[(dist > radius) & (dist <= radius + 1)] = [100, 20, 20, 200]
    return texture


# ---------------------------------------------------------------------------
# Item-icon constants
# ---------------------------------------------------------------------------


# Derived from the single source of truth in constants.ITEM_TO_MACHINE.
# New machine types added there automatically appear here.
MACHINE_TO_ITEM: dict[int, int] = {
    int(mt): int(it) for it, mt in ITEM_TO_MACHINE.items()
}

# Dark arrow colour drawn on top of the gold conveyor belt square.
_BELT_ARROW_COLOR: tuple[int, int, int, int] = (60, 50, 10, 255)


# ---------------------------------------------------------------------------
# Belt / miner / arm directional helpers
# ---------------------------------------------------------------------------


def _draw_chevron(
    image: np.ndarray,
    cy: int,
    cx: int,
    size: int,
    direction: int,
) -> None:
    """Draw a single filled chevron arrow into *image*.

    The chevron points in *direction* (Action enum value) and is centred
    on pixel ``(cy, cx)``.  *size* controls the half-width of the arrow.

    Args:
        image: RGBA image array (modified in place).
        cy: Centre row of the chevron.
        cx: Centre column of the chevron.
        size: Half-extent of the arrow in pixels.
        direction: Direction.LEFT / RIGHT / UP / DOWN.
    """
    h, w = image.shape[:2]
    for d in range(-size, size + 1):
        depth = size - abs(d)
        for t in range(depth + 1):
            if direction == Direction.RIGHT:
                py, px = cy + d, cx + t
            elif direction == Direction.LEFT:
                py, px = cy + d, cx - t
            elif direction == Direction.DOWN:
                py, px = cy + t, cx + d
            elif direction == Direction.UP:
                py, px = cy - t, cx + d
            else:
                return
            if 0 <= py < h and 0 <= px < w:
                image[py, px] = _BELT_ARROW_COLOR


def _draw_belt_arrows(
    icon: np.ndarray,
    direction: int,
) -> None:
    """Draw three evenly spaced chevron arrows onto an icon array.

    Args:
        icon: RGBA array of shape ``(size, size, 4)``, modified in place.
        direction: Action enum value for belt facing direction.
    """
    size = icon.shape[0]
    arrow_size = max(1, size // 8)
    mid = size // 2

    if direction in (Direction.LEFT, Direction.RIGHT):
        cy = mid
        for i in range(3):
            cx = size * (1 + 2 * i) // 6
            _draw_chevron(icon, cy, cx, arrow_size, direction)
    elif direction in (Direction.UP, Direction.DOWN):
        cx = mid
        for i in range(3):
            cy = size * (1 + 2 * i) // 6
            _draw_chevron(icon, cy, cx, arrow_size, direction)


_MINER_ARROW_COLOR: tuple[int, int, int, int] = (0, 90, 0, 255)


def _draw_miner_indicator(icon: np.ndarray, direction: int) -> None:
    """Draw a directional output arrow on a miner icon.

    Renders a triangular pointer on the facing edge of the miner so
    players can see which way the miner pushes its output.  The arrow
    is drawn in a darker green that stands out against the bright
    green base.

    Args:
        icon: RGBA array of shape ``(size, size, 4)``, modified
            in place.
        direction: ``Action`` direction the miner faces (output side).
    """
    size = icon.shape[0]
    mid = size // 2
    arrow_len = max(2, size // 4)
    half_w = max(2, size // 4)

    for t in range(arrow_len):
        spread = half_w * (arrow_len - t) // arrow_len
        for s in range(-spread, spread + 1):
            if direction == Direction.RIGHT:
                py, px = mid + s, size - 1 - t
            elif direction == Direction.LEFT:
                py, px = mid + s, t
            elif direction == Direction.DOWN:
                py, px = size - 1 - t, mid + s
            else:  # UP
                py, px = t, mid + s
            if 0 <= py < size and 0 <= px < size:
                icon[py, px] = _MINER_ARROW_COLOR


_ARM_BODY_COLOR: tuple[int, int, int, int] = (180, 130, 70, 255)
_ARM_ARROW_COLOR: tuple[int, int, int, int] = (255, 255, 255, 255)
_ARM_CIRCLE_COLOR: tuple[int, int, int, int] = (80, 60, 40, 255)


def _draw_arm_indicator(icon: np.ndarray, direction: int) -> None:
    """Draw arm icon: circle (pick side) + arrowhead (deposit side).

    The circle marks the source (behind the arm) and the arrow marks
    the destination (facing direction), so the visual clearly shows
    which way items flow.

    Args:
        icon: RGBA array modified in place.
        direction: ``Direction`` value the arm faces (output side).
    """
    s = icon.shape[0]
    mid = s // 2

    # Fill base with arm body color.
    icon[1 : s - 1, 1 : s - 1] = _ARM_BODY_COLOR

    # Arrow on facing edge (output).
    arrow_len = max(2, s // 4)
    half_w = max(2, s // 4)
    for t in range(arrow_len):
        spread = half_w * (arrow_len - t) // arrow_len
        for off in range(-spread, spread + 1):
            if direction == Direction.RIGHT:
                py, px = mid + off, s - 1 - t
            elif direction == Direction.LEFT:
                py, px = mid + off, t
            elif direction == Direction.DOWN:
                py, px = s - 1 - t, mid + off
            else:
                py, px = t, mid + off
            if 0 <= py < s and 0 <= px < s:
                icon[py, px] = _ARM_ARROW_COLOR

    # Circle on back edge (source).
    cr = max(1, s // 6)
    if direction == Direction.RIGHT:
        cy, cx = mid, cr + 1
    elif direction == Direction.LEFT:
        cy, cx = mid, s - cr - 2
    elif direction == Direction.DOWN:
        cy, cx = cr + 1, mid
    else:
        cy, cx = s - cr - 2, mid
    for dy in range(-cr, cr + 1):
        for dx in range(-cr, cr + 1):
            if dy * dy + dx * dx <= cr * cr:
                py, px = cy + dy, cx + dx
                if 0 <= py < s and 0 <= px < s:
                    icon[py, px] = _ARM_CIRCLE_COLOR


# ---------------------------------------------------------------------------
# Shared drawing helpers for ore / plate / item / machine icons
# ---------------------------------------------------------------------------


def _shade(rgb: tuple[int, int, int], delta: int) -> tuple[int, int, int]:
    """Shift an RGB triple toward black (negative) or white (positive).

    Args:
        rgb: Base color.
        delta: Amount to add/subtract from each channel. Clamped to [0, 255].

    Returns:
        Shifted RGB triple.
    """
    return tuple(max(0, min(255, c + delta)) for c in rgb)  # type: ignore[return-value]


def _draw_ore_patches(
    icon: np.ndarray,
    base_rgb: tuple[int, int, int],
    seed: int,
    *,
    crystalline: bool = False,
) -> None:
    """Paint small speckly darker patches onto an ore icon.

    Each "patch" is a scattered cluster of 3-7 small dots rather than a
    smooth circular blob, so ore textures read as grainy rock instead of
    polka dots. Distribution is deterministic for a given ``seed``.

    When ``crystalline`` is True, two to three deliberate bright facet
    dots are placed near the center to signal silicon's glassy nature
    without looking noisy.

    Args:
        icon: RGBA array modified in place.
        base_rgb: Base ore color (already filled into the icon).
        seed: RNG seed controlling patch layout.
        crystalline: Whether to add a few bright facet highlights.
    """
    s = icon.shape[0]
    if s < 4:
        return
    rng = np.random.default_rng(seed)
    dark = _shade(base_rgb, -35)
    deep = _shade(base_rgb, -60)
    light = _shade(base_rgb, 35)

    # Many small clusters rather than a few big blobs.
    n_clusters = max(6, (s * s) // 25)
    cluster_spread = max(1, s // 10)
    for _ in range(n_clusters):
        cy = int(rng.integers(0, s))
        cx = int(rng.integers(0, s))
        color = deep if rng.random() < 0.25 else dark
        # Drop 3-7 individual pixels within a small box around (cy, cx),
        # with probability tapering off at the edges.
        for _ in range(int(rng.integers(3, 8))):
            off_y = int(rng.integers(-cluster_spread, cluster_spread + 1))
            off_x = int(rng.integers(-cluster_spread, cluster_spread + 1))
            # Reject pixels far from center with decreasing probability.
            dist = max(abs(off_y), abs(off_x))
            if rng.random() > 1.0 - (dist / (cluster_spread + 1)) * 0.6:
                py, px = cy + off_y, cx + off_x
                if 0 <= py < s and 0 <= px < s:
                    icon[py, px, :3] = color

    # A couple of subtle bright pinpoints for material catching light.
    for _ in range(max(1, s // 16)):
        gy = int(rng.integers(0, s))
        gx = int(rng.integers(0, s))
        icon[gy, gx, :3] = light

    if crystalline:
        facet = _shade(base_rgb, 80)
        # Two-to-three deliberate facets in a loose diagonal near center.
        cy = cx = s // 2
        facets = [(cy - s // 6, cx - s // 6), (cy, cx), (cy + s // 6, cx + s // 8)]
        for i, (fy, fx) in enumerate(facets):
            if i >= 2 + int(rng.integers(0, 2)):
                break
            if 0 <= fy < s and 0 <= fx < s:
                icon[fy, fx, :3] = facet
                # One neighbor pixel for a tiny plus-shape.
                if fx + 1 < s:
                    icon[fy, fx + 1, :3] = facet


def _draw_plate_shine(
    icon: np.ndarray,
    base_rgb: tuple[int, int, int],
) -> None:
    """Draw a riveted, beveled plate: shine band, edge bevel, corner screws.

    Combines three cues for "manufactured sheet metal":
    - Diagonal bright band across the upper third (light source).
    - 1-pixel bevel: lighter top+left, darker bottom+right.
    - Four small dark screws in the corners, inside the bevel.

    Args:
        icon: RGBA array modified in place.
        base_rgb: Base plate color.
    """
    s = icon.shape[0]
    if s < 4:
        return
    bright = _shade(base_rgb, 45)
    mid = _shade(base_rgb, 22)
    bevel_dark = _shade(base_rgb, -35)
    screw = _shade(base_rgb, -70)

    # Diagonal band near the upper-left, three pixels wide.
    band_len = max(2, (2 * s) // 3)
    for i in range(band_len):
        for t, color in ((-1, mid), (0, bright), (1, mid)):
            y = i + t
            x = i - t
            if 0 <= y < s and 0 <= x < s:
                icon[y, x, :3] = color

    # Bevel: lighter top+left, darker bottom+right (drawn after shine so
    # the outermost ring is clean).
    icon[0, :, :3] = bright
    icon[:, 0, :3] = bright
    icon[s - 1, :, :3] = bevel_dark
    icon[:, s - 1, :3] = bevel_dark

    # Four corner screws (one pixel in each corner, one inside the bevel).
    if s >= 6:
        for cy, cx in ((1, 1), (1, s - 2), (s - 2, 1), (s - 2, s - 2)):
            icon[cy, cx, :3] = screw


def _draw_wafer(
    icon: np.ndarray,
    base_rgb: tuple[int, int, int],
) -> None:
    """Draw a wafer icon: plate shine plus faint concentric arcs.

    The arcs hint at a silicon disc rather than a flat metal sheet.

    Args:
        icon: RGBA array modified in place.
        base_rgb: Base wafer color.
    """
    _draw_plate_shine(icon, base_rgb)
    s = icon.shape[0]
    if s < 8:
        return
    cy = cx = s // 2
    arc = _shade(base_rgb, -25)
    ys, xs = np.ogrid[:s, :s]
    dist = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    for ring_r in (s // 4, s // 3):
        mask = np.abs(dist - ring_r) < 0.6
        icon[mask, :3] = arc


def _draw_wire_helix(
    icon: np.ndarray,
    base_rgb: tuple[int, int, int],
) -> None:
    """Draw twin diagonal strands on a transparent background.

    Two 2-pixel-thick diagonal strands in the wire's base color, with
    a darker center line for depth. Background stays transparent.

    Args:
        icon: RGBA array modified in place (assumed transparent).
        base_rgb: Base wire color.
    """
    s = icon.shape[0]
    if s < 4:
        return
    dark = _shade(base_rgb, -40)
    sep = max(2, s // 4)
    rgba = (*base_rgb, 255)
    dark_rgba = (*dark, 255)
    for i in range(s):
        # Strand 1.
        if 0 <= i < s:
            icon[i, i] = dark_rgba
            if i + 1 < s:
                icon[i, i + 1] = rgba
        # Strand 2 (offset).
        j = i + sep
        if 0 <= j < s:
            icon[i, j] = dark_rgba
            if j - 1 >= 0 and j - 1 < s:
                icon[i, j - 1] = rgba


def _draw_circuit_traces(
    icon: np.ndarray,
    base_rgb: tuple[int, int, int],
) -> None:
    """Draw a PCB tile: green board body with dark traces and bright pads.

    Transparent background, small green board inset from the corners,
    traces and solder pads on top.

    Args:
        icon: RGBA array modified in place (assumed transparent).
        base_rgb: Base circuit (green) color.
    """
    s = icon.shape[0]
    if s < 6:
        return
    board = (*base_rgb, 255)
    dark = (*_shade(base_rgb, -60), 255)
    pad = (*_shade(base_rgb, 70), 255)
    margin = max(1, s // 8)
    icon[margin : s - margin, margin : s - margin] = board
    mid = s // 2
    q = max(margin + 1, s // 4)
    icon[q, margin + 1 : mid + 1] = dark
    icon[q : s - q, mid] = dark
    icon[s - q - 1, mid : s - margin - 1] = dark
    # Two solder pads (2x2 blocks near the trace endpoints).
    icon[q - 1 : q + 1, margin + 1 : margin + 3] = pad
    icon[s - q - 1 : s - q + 1, s - margin - 3 : s - margin - 1] = pad


def _draw_motor(
    icon: np.ndarray,
    base_rgb: tuple[int, int, int],
) -> None:
    """Draw a motor on a transparent background: housing + shaft + rivets.

    Args:
        icon: RGBA array modified in place (assumed transparent).
        base_rgb: Base motor color.
    """
    s = icon.shape[0]
    if s < 6:
        return
    body = (*base_rgb, 255)
    dark = (*_shade(base_rgb, -50), 255)
    bright = (*_shade(base_rgb, 55), 255)
    mid = s // 2
    body_top = s // 4
    body_bot = 3 * s // 4
    body_left = s // 5
    body_right = 3 * s // 5
    # Solid housing body.
    icon[body_top:body_bot, body_left:body_right] = body
    # Dark outline on housing top and bottom.
    icon[body_top, body_left:body_right] = dark
    icon[body_bot - 1, body_left:body_right] = dark
    # Shaft sticking right.
    shaft_y0 = mid - max(1, s // 12)
    shaft_y1 = mid + max(1, s // 12) + 1
    icon[shaft_y0:shaft_y1, body_right : body_right + s // 5] = dark
    # Rivet highlights on the housing.
    for ry in (body_top + 1, body_bot - 2):
        for rx in (body_left + 1, body_right - 2):
            if 0 <= ry < s and 0 <= rx < s:
                icon[ry, rx] = bright


def _draw_sensor_lens(
    icon: np.ndarray,
    base_rgb: tuple[int, int, int],
) -> None:
    """Draw a sensor lens on a transparent background.

    A round body (body color) with a darker iris and bright central glint.

    Args:
        icon: RGBA array modified in place (assumed transparent).
        base_rgb: Base sensor body color.
    """
    s = icon.shape[0]
    if s < 6:
        return
    body = (*base_rgb, 255)
    dark = (*_shade(base_rgb, -55), 255)
    iris = (*_shade(base_rgb, -30), 255)
    bright = (*_shade(base_rgb, 90), 255)
    cy = cx = s // 2
    outer_r = s // 3
    inner_r = max(1, s // 5)
    ys, xs = np.ogrid[:s, :s]
    dist = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    # Body (full disc).
    icon[dist <= outer_r + 1] = body
    # Dark iris ring.
    icon[(dist <= outer_r) & (dist > inner_r)] = dark
    # Iris center.
    icon[dist <= inner_r] = iris
    # Bright glint.
    icon[cy, cx] = bright


def _draw_frame_ibeam(
    icon: np.ndarray,
    base_rgb: tuple[int, int, int],
) -> None:
    """Draw an I-beam silhouette on a transparent background.

    Args:
        icon: RGBA array modified in place (assumed transparent).
        base_rgb: Base frame color.
    """
    s = icon.shape[0]
    if s < 8:
        return
    body = (*base_rgb, 255)
    dark = (*_shade(base_rgb, -45), 255)
    flange_h = max(1, s // 6)
    web_half = max(1, s // 8)
    mid = s // 2
    margin = s // 6
    # Top flange (body color with dark edge).
    icon[margin : margin + flange_h, margin : s - margin] = body
    icon[margin, margin : s - margin] = dark
    icon[margin + flange_h - 1, margin : s - margin] = dark
    # Bottom flange.
    icon[s - margin - flange_h : s - margin, margin : s - margin] = body
    icon[s - margin - flange_h, margin : s - margin] = dark
    icon[s - margin - 1, margin : s - margin] = dark
    # Web.
    icon[margin : s - margin, mid - web_half : mid + web_half + 1] = body
    icon[margin : s - margin, mid - web_half] = dark
    icon[margin : s - margin, mid + web_half] = dark


def _draw_flask(
    icon: np.ndarray,
    base_rgb: tuple[int, int, int],
    *,
    advanced: bool = False,
) -> None:
    """Draw a flask silhouette on a transparent background.

    The bulb is filled with the base color; the neck, rim, and bulb
    outline are drawn in a darker shade. Advanced packs add a central
    glow.

    Args:
        icon: RGBA array modified in place (assumed transparent).
        base_rgb: Base flask color.
        advanced: If True, draw a central glow.
    """
    s = icon.shape[0]
    if s < 8:
        return
    body = (*base_rgb, 255)
    dark = (*_shade(base_rgb, -55), 255)
    mid = s // 2
    neck_top = s // 6
    neck_bot = s // 2
    neck_half = max(1, s // 12)
    # Neck outline in dark, inside body color.
    icon[neck_top:neck_bot, mid - neck_half : mid + neck_half + 1] = dark
    if neck_half >= 1:
        icon[neck_top + 1 : neck_bot, mid - neck_half + 1 : mid + neck_half] = body
    # Bulb.
    body_cy = (3 * s) // 5
    body_r = s // 3
    ys, xs = np.ogrid[:s, :s]
    dist = np.sqrt((ys - body_cy) ** 2 + (xs - mid) ** 2)
    fill = (dist <= body_r - 0.5) & (ys >= neck_bot - 1)
    icon[fill] = body
    ring = (dist <= body_r) & (dist >= body_r - 1.2) & (ys >= neck_bot - 1)
    icon[ring] = dark
    if advanced:
        glow = (255, 230, 120, 255)
        icon[body_cy - 1 : body_cy + 2, mid - 1 : mid + 2] = glow


def _draw_rocket(
    icon: np.ndarray,
    base_rgb: tuple[int, int, int],
) -> None:
    """Draw a rocket silhouette on a transparent background.

    Args:
        icon: RGBA array modified in place (assumed transparent).
        base_rgb: Base rocket body color.
    """
    s = icon.shape[0]
    if s < 8:
        return
    body = (*base_rgb, 255)
    dark = (*_shade(base_rgb, -90), 255)
    exhaust = (255, 210, 80, 255)
    mid = s // 2
    body_w = max(2, s // 5)
    body_top = s // 5
    body_bot = (4 * s) // 5
    # Body (fill + dark edge).
    icon[body_top:body_bot, mid - body_w // 2 : mid - body_w // 2 + body_w] = body
    icon[body_top:body_bot, mid - body_w // 2] = dark
    icon[body_top:body_bot, mid - body_w // 2 + body_w - 1] = dark
    # Conical nose: widest at the base (y just above the body) and tapering
    # to a point at the icon's top edge.
    for t in range(body_top):
        y = body_top - 1 - t
        half = body_w // 2 - t // 2
        if half < 0:
            continue
        icon[y, mid - half : mid + half + 1] = body
        if 0 <= y < s:
            if mid - half >= 0:
                icon[y, mid - half] = dark
            if mid + half < s:
                icon[y, mid + half] = dark
    # Fins.
    fin_y = body_bot - 2
    icon[fin_y : fin_y + 2, max(0, mid - body_w) : mid - body_w // 2] = dark
    icon[fin_y : fin_y + 2, mid + body_w // 2 + 1 : min(s, mid + body_w + 1)] = dark
    # Exhaust spark below body.
    fin_len = max(1, s // 8)
    if body_bot < s:
        icon[body_bot : min(body_bot + fin_len, s), mid] = exhaust


def _draw_assembler_body(
    icon: np.ndarray,
    base_rgb: tuple[int, int, int],
) -> None:
    """Draw an assembler: two side input ports flanking a dark window.

    Args:
        icon: RGBA array modified in place.
        base_rgb: Base assembler (purple) color.
    """
    s = icon.shape[0]
    if s < 6:
        return
    window = _shade(base_rgb, -55)
    port = _shade(base_rgb, 55)
    cy = cx = s // 2
    win_half = max(1, s // 5)
    icon[cy - win_half : cy + win_half + 1, cx - win_half : cx + win_half + 1, :3] = (
        window
    )
    # Ports: two small squares on the left and right edges, vertically centred.
    port_half = max(1, s // 8)
    icon[cy - port_half : cy + port_half + 1, 1 : 2 + port_half, :3] = port
    icon[cy - port_half : cy + port_half + 1, s - 2 - port_half : s - 1, :3] = port


def _draw_furnace_body(
    icon: np.ndarray,
    base_rgb: tuple[int, int, int],
) -> None:
    """Draw a furnace: dark brick body with a glowing central maw.

    Args:
        icon: RGBA array modified in place.
        base_rgb: Base furnace (dark red-brown) color.
    """
    s = icon.shape[0]
    if s < 6:
        return
    body = _shade(base_rgb, -35)
    maw = (200, 110, 40)
    ember = (255, 200, 100)
    # Dark body fill.
    icon[1 : s - 1, 1 : s - 1, :3] = body
    # Maw in the center.
    maw_top = s // 4
    maw_bot = (3 * s) // 4
    maw_left = s // 4
    maw_right = (3 * s) // 4
    icon[maw_top:maw_bot, maw_left:maw_right, :3] = maw
    # Bright ember core.
    cy = cx = s // 2
    core = max(1, s // 8)
    icon[cy - core : cy + core + 1, cx - core : cx + core + 1, :3] = ember


def _draw_science_lab_body(
    icon: np.ndarray,
    base_rgb: tuple[int, int, int],
) -> None:
    """Draw a geodesic dome, top-down.

    Palette C: violet body + lavender ribs + pale apex.

    The silhouette is an inset square rim with four diagonal ribs
    meeting at a central apex, plus four small window panels tucked
    between the ribs. At tile sizes below ~8 px the inner detail
    degrades gracefully — only rim + apex remain.

    Args:
        icon: RGBA array modified in place. Pre-filled with the body
            color by the caller; we overwrite the internal structure.
        base_rgb: Dome body color (palette C ``#4c1d95``).
    """
    s = icon.shape[0]
    if s < 4:
        return
    rib = (196, 181, 253)  # palette C bars: #c4b5fd
    apex = (237, 233, 254)  # palette C apex: #ede9fe
    window = (253, 230, 138)  # warm lit pane, reads as interior light
    body = base_rgb
    # Re-fill the body so the caller's square backdrop becomes the
    # dome interior (inset by 1 from the edge).
    icon[1 : s - 1, 1 : s - 1, :3] = body

    cx = cy = s // 2
    # Four diagonal ribs from the rim to the apex. Draw as a line
    # per rib, thickness 1 at small sizes, 2 at >=12 px.
    t = 1 if s < 12 else 2

    def _line(y0: int, x0: int, y1: int, x1: int) -> None:
        steps = max(abs(y1 - y0), abs(x1 - x0))
        if steps == 0:
            return
        for k in range(steps + 1):
            y = y0 + (y1 - y0) * k // steps
            x = x0 + (x1 - x0) * k // steps
            for dy in range(t):
                for dx in range(t):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < s and 0 <= xx < s:
                        icon[yy, xx, :3] = rib

    # Ribs from the four inner corners to the apex.
    inset = 1
    _line(inset, inset, cy, cx)
    _line(inset, s - 1 - inset, cy, cx)
    _line(s - 1 - inset, inset, cy, cx)
    _line(s - 1 - inset, s - 1 - inset, cy, cx)

    # Four small window panes tucked between ribs (up/down/left/right
    # of the apex), if there's room.
    if s >= 10:
        win_sz = max(1, s // 6)
        offset = max(2, s // 4)
        for oy, ox in (
            (cy - offset, cx - win_sz // 2),
            (cy + offset - win_sz, cx - win_sz // 2),
            (cy - win_sz // 2, cx - offset),
            (cy - win_sz // 2, cx + offset - win_sz),
        ):
            if 0 <= oy < s - win_sz and 0 <= ox < s - win_sz:
                icon[oy : oy + win_sz, ox : ox + win_sz, :3] = window

    # Apex highlight (2x2 or 1x1).
    apex_sz = 2 if s >= 10 else 1
    icon[
        cy - apex_sz // 2 : cy + (apex_sz + 1) // 2,
        cx - apex_sz // 2 : cx + (apex_sz + 1) // 2,
        :3,
    ] = apex


def _draw_pallet_slats(
    icon: np.ndarray,
    base_rgb: tuple[int, int, int],
) -> None:
    """Draw a pallet as a surface with three horizontal dark slat gaps.

    Args:
        icon: RGBA array modified in place.
        base_rgb: Base pallet surface color.
    """
    s = icon.shape[0]
    dark = _shade(base_rgb, -60)
    # Surface was filled by the caller; stripe three gaps across it.
    if s < 5:
        return
    for i in range(3):
        y = (i + 1) * s // 4
        if 0 <= y < s:
            icon[y, 1 : s - 1, :3] = dark


def _draw_machine_frame(icon: np.ndarray) -> None:
    """Bevel every machine with a lighter top-left and darker bottom-right edge.

    Creates a consistent "placed object" look shared by all machines.

    Args:
        icon: RGBA array modified in place.
    """
    s = icon.shape[0]
    if s < 3:
        return
    # Sample the base color from the center so the bevel adapts per machine.
    base = tuple(int(c) for c in icon[s // 2, s // 2, :3])
    light = _shade(base, 45)  # type: ignore[arg-type]
    dark = _shade(base, -55)  # type: ignore[arg-type]
    icon[0, :, :3] = light
    icon[:, 0, :3] = light
    icon[s - 1, :, :3] = dark
    icon[:, s - 1, :3] = dark


def _draw_belt_edges(icon: np.ndarray, direction: int) -> None:
    """Draw dark edge stripes parallel to belt travel direction.

    Args:
        icon: RGBA array modified in place.
        direction: ``Direction`` value the belt faces.
    """
    s = icon.shape[0]
    if s < 4:
        return
    base = tuple(int(c) for c in icon[s // 2, s // 2, :3])
    edge = _shade(base, -55)  # type: ignore[arg-type]
    if direction in (Direction.LEFT, Direction.RIGHT):
        icon[0, :, :3] = edge
        icon[s - 1, :, :3] = edge
    else:
        icon[:, 0, :3] = edge
        icon[:, s - 1, :3] = edge


def _draw_splitter_body(icon: np.ndarray, direction: int) -> None:
    """Draw a T-shape splitter glyph: a single thick line down the input
    axis and two chevrons fanning out to the perpendicular output sides.

    A vertical-facing splitter (``Direction.UP`` / ``Direction.DOWN``)
    splits to LEFT and RIGHT, so the trunk runs vertically and the
    chevrons point W and E. A horizontal-facing splitter does the
    opposite. The trunk is a darker-tone band so the orientation reads
    instantly even at small icon sizes.

    Args:
        icon: RGBA array modified in place.
        direction: ``Direction`` value the splitter faces.
    """
    s = icon.shape[0]
    if s < 6:
        return
    base = tuple(int(c) for c in icon[s // 2, s // 2, :3])
    trunk = _shade(base, -50)  # type: ignore[arg-type]
    mid = s // 2
    half = max(1, s // 8)

    if direction in (Direction.UP, Direction.DOWN):
        # Vertical trunk; outputs LEFT and RIGHT.
        icon[1 : s - 1, mid - half : mid + half + 1, :3] = trunk
        arrow_size = max(1, s // 7)
        _draw_chevron(icon, mid, max(arrow_size, s // 4), arrow_size, Direction.LEFT)
        _draw_chevron(
            icon, mid, s - 1 - max(arrow_size, s // 4), arrow_size, Direction.RIGHT
        )
    else:
        # Horizontal trunk; outputs UP and DOWN.
        icon[mid - half : mid + half + 1, 1 : s - 1, :3] = trunk
        arrow_size = max(1, s // 7)
        _draw_chevron(icon, max(arrow_size, s // 4), mid, arrow_size, Direction.UP)
        _draw_chevron(
            icon, s - 1 - max(arrow_size, s // 4), mid, arrow_size, Direction.DOWN
        )


def _draw_crossing_body(icon: np.ndarray, direction: int) -> None:
    """Draw a single diagonal stripe from the input-corner to the
    output-corner of the crossing.

    The four crossing encodings map (input-pair, output-pair) to a
    diagonal:

    * ``1`` (inputs N+W, outputs S+E) → ``\\`` (NW → SE)
    * ``2`` (inputs N+E, outputs S+W) → ``/`` (NE → SW)
    * ``3`` (inputs S+W, outputs N+E) → ``/`` (SW → NE)
    * ``4`` (inputs S+E, outputs N+W) → ``\\`` (SE → NW)

    The stripe is anti-aliased with a thicker dark band over a thin
    light highlight so the diagonal reads clearly against the base
    fill. Inactive direction (encoding 0) leaves the icon untouched
    so unset crossings render as a solid block — useful for inventory
    icons where the in-flight axes are not yet decided.

    Args:
        icon: RGBA array modified in place.
        direction: Packed crossing direction (1..4); 0 → no-op.
    """
    s = icon.shape[0]
    if s < 6 or direction == 0:
        return
    base = tuple(int(c) for c in icon[s // 2, s // 2, :3])
    dark = _shade(base, -65)  # type: ignore[arg-type]
    light = _shade(base, 55)  # type: ignore[arg-type]
    band = max(1, s // 10)

    # Use the diagonal glyph implied by the encoding.
    backslash = direction in (1, 4)
    coords = np.arange(s)
    for offset in range(-band, band + 1):
        if backslash:
            ys = coords
            xs = coords + offset
        else:
            ys = coords
            xs = (s - 1) - coords + offset
        valid = (xs >= 0) & (xs < s)
        ys, xs = ys[valid], xs[valid]
        color = light if offset == 0 else dark
        icon[ys, xs, :3] = color


def _draw_miner_bore(icon: np.ndarray) -> None:
    """Draw a central dark circular bore on a miner icon.

    Args:
        icon: RGBA array modified in place.
    """
    s = icon.shape[0]
    if s < 5:
        return
    cy = cx = s // 2
    r = max(1, s // 5)
    ys, xs = np.ogrid[:s, :s]
    dist = (ys - cy) ** 2 + (xs - cx) ** 2
    icon[dist <= r * r, :3] = (0, 70, 0)
    if r >= 2:
        icon[dist <= (r - 1) * (r - 1), :3] = (0, 40, 0)


# ItemTypes whose icons are machines placed on the map. Each gets the
# shared top-left highlight / bottom-right shadow frame so placed
# machines read as "built things" against terrain.
_MACHINE_ITEM_TYPES: frozenset[int] = frozenset(int(it) for it in ITEM_TO_MACHINE)

# Ores share the patched-rock texture; silicon additionally gets
# crystalline sparkles on top.
_ORE_ITEMS: dict[int, bool] = {
    int(ItemType.COAL): False,
    int(ItemType.IRON_ORE): False,
    int(ItemType.COPPER_ORE): False,
    int(ItemType.TIN_ORE): False,
    int(ItemType.SILICON): True,
}

_PLATE_ITEMS: frozenset[int] = frozenset(
    {
        int(ItemType.IRON_PLATE),
        int(ItemType.COPPER_PLATE),
        int(ItemType.TIN_PLATE),
    }
)


@functools.lru_cache(maxsize=256)
def render_item_icon(
    item_type: int,
    size: int,
    direction: int | None = None,
) -> np.ndarray:
    """Render a square RGBA icon for an item type.

    This is the single source of truth for how an item looks visually.
    Every UI surface that paints an item — inventory panels, hotbars,
    editor toolbar, menu screens — calls this function so identical
    items always look the same.

    Each item category has a distinct visual language:

    - Ores: irregular darker blobs on a rough base (silicon adds
      crystalline sparkles).
    - Plates: diagonal shine band on a uniform base; wafer adds
      concentric arcs to suggest a disc.
    - Intermediate items: a shape that hints at the object — wire
      helix, circuit traces, motor cylinder, sensor lens, frame I-beam,
      flask, rocket silhouette.
    - Machines: an interior pattern (miner bore, pallet slats, belt
      edge stripes, assembler ports, furnace glowing maw, rocket
      silhouette) plus a shared top-left highlight and bottom-right
      shadow so every placed machine reads as a built object.

    When ``direction`` is ``None`` (e.g. in a menu with no placement
    context), directional indicators default to pointing right.

    Args:
        item_type: ``ItemType`` integer value.
        size: Side length of the returned square in pixels.
        direction: Optional ``Direction`` for directional items.
            Ignored for non-directional items.

    Returns:
        RGBA uint8 array of shape ``(size, size, 4)``.
    """
    rgb = ITEM_COLORS.get(item_type, (128, 128, 128))

    # Shaped items render on a transparent canvas so only the silhouette
    # is visible. Ores, plates, and placed machines keep the full square
    # fill since the base color is itself the material/body of the object.
    shaped_items = {
        int(ItemType.WIRE),
        int(ItemType.CIRCUIT),
        int(ItemType.MOTOR),
        int(ItemType.SENSOR),
        int(ItemType.FRAME),
        int(ItemType.BASIC_SCIENCE_PACK),
        int(ItemType.ADVANCED_SCIENCE_PACK),
    }
    # ROCKET is both a machine (placed on map) and a shaped item; always
    # render as a transparent silhouette because its sprite is iconic.
    shaped_items.add(int(ItemType.ROCKET))

    if item_type in shaped_items:
        icon = np.zeros((size, size, 4), dtype=np.uint8)
    else:
        icon = np.full((size, size, 4), (*rgb, 255), dtype=np.uint8)

    # --- Ores ---
    if item_type in _ORE_ITEMS:
        _draw_ore_patches(
            icon,
            rgb,
            seed=item_type,
            crystalline=_ORE_ITEMS[item_type],
        )
        return icon

    # --- Plates and wafer ---
    if item_type in _PLATE_ITEMS:
        _draw_plate_shine(icon, rgb)
        return icon
    if item_type == int(ItemType.WAFER):
        _draw_wafer(icon, rgb)
        return icon

    # --- Intermediate shaped items (transparent background) ---
    if item_type == int(ItemType.WIRE):
        _draw_wire_helix(icon, rgb)
        return icon
    if item_type == int(ItemType.CIRCUIT):
        _draw_circuit_traces(icon, rgb)
        return icon
    if item_type == int(ItemType.MOTOR):
        _draw_motor(icon, rgb)
        return icon
    if item_type == int(ItemType.SENSOR):
        _draw_sensor_lens(icon, rgb)
        return icon
    if item_type == int(ItemType.FRAME):
        _draw_frame_ibeam(icon, rgb)
        return icon
    if item_type == int(ItemType.BASIC_SCIENCE_PACK):
        _draw_flask(icon, rgb, advanced=False)
        return icon
    if item_type == int(ItemType.ADVANCED_SCIENCE_PACK):
        _draw_flask(icon, rgb, advanced=True)
        return icon
    if item_type == int(ItemType.ROCKET):
        _draw_rocket(icon, rgb)
        return icon

    # --- Machines (solid fill with interior + shared bevel frame) ---
    if item_type in _MACHINE_ITEM_TYPES:
        if item_type == int(ItemType.CONVEYOR_BELT) and size >= 6:
            belt_dir = direction if direction is not None else int(Direction.RIGHT)
            _draw_belt_edges(icon, belt_dir)
            _draw_belt_arrows(icon, belt_dir)
        elif item_type == int(ItemType.MINER) and size >= 6:
            miner_dir = direction if direction is not None else int(Direction.RIGHT)
            _draw_miner_bore(icon)
            _draw_miner_indicator(icon, miner_dir)
        elif item_type == int(ItemType.ARM) and size >= 6:
            arm_dir = direction if direction is not None else int(Direction.RIGHT)
            _draw_arm_indicator(icon, arm_dir)
        elif item_type == int(ItemType.PALLET):
            _draw_pallet_slats(icon, rgb)
        elif item_type == int(ItemType.ASSEMBLER) and size >= 6:
            _draw_assembler_body(icon, rgb)
        elif item_type == int(ItemType.FURNACE) and size >= 6:
            _draw_furnace_body(icon, rgb)
        elif item_type == int(ItemType.SCIENCE_LAB) and size >= 4:
            _draw_science_lab_body(icon, rgb)
        elif item_type == int(ItemType.SPLITTER) and size >= 6:
            split_dir = direction if direction is not None else int(Direction.RIGHT)
            _draw_splitter_body(icon, split_dir)
        elif item_type == int(ItemType.CROSSING) and size >= 6:
            cross_dir = direction if direction is not None else 1
            _draw_crossing_body(icon, cross_dir)
        _draw_machine_frame(icon)
        # Knock the four corner pixels transparent so placed machines
        # read as "objects on terrain" rather than square tiles.
        if size >= 4:
            icon[0, 0, 3] = 0
            icon[0, size - 1, 3] = 0
            icon[size - 1, 0, 3] = 0
            icon[size - 1, size - 1, 3] = 0
        return icon

    return icon
