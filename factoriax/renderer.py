"""Pixel rendering for the FactoriaX environment."""

import functools

import numpy as np

from factoriax.constants import (
    BLOCK_PIXEL_SIZE,
    ITEM_COLORS,
    ITEM_TO_MACHINE,
    MAX_MACHINE_STACK_SIZE,
    NUM_ITEM_TYPES,
    BlockType,
    Direction,
    ItemType,
    MachineType,
    load_all_textures,
)
from factoriax.state import EnvState

INVENTORY_BAR_HEIGHT = 24


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
    solid_colors: dict[int, tuple[int, int, int]] = {
        int(BlockType.DIRT): (139, 90, 43),
        int(BlockType.WATER): (50, 120, 190),
        int(BlockType.NEST): (130, 40, 55),
    }
    ore_colors: dict[int, tuple[tuple[int, int, int], bool]] = {
        int(BlockType.IRON): ((180, 185, 200), False),
        int(BlockType.COPPER): ((200, 120, 45), False),
        int(BlockType.COAL): ((50, 50, 55), False),
        int(BlockType.TIN): ((195, 195, 175), False),
        int(BlockType.SILICON): ((80, 95, 150), True),
    }

    textures: dict[int, np.ndarray] = {
        block_id: _solid_texture(size, rgb) for block_id, rgb in solid_colors.items()
    }
    for block_id, (rgb, crystalline) in ore_colors.items():
        tex = _solid_texture(size, rgb)
        _draw_ore_patches(tex, rgb, seed=block_id, crystalline=crystalline)
        textures[block_id] = tex
    return textures


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


@functools.lru_cache(maxsize=8)
def _get_biter_texture(size: int) -> np.ndarray:
    """Cached biter texture.

    Args:
        size: Block pixel size.

    Returns:
        RGBA numpy array of shape (size, size, 4).
    """
    return create_biter_texture(size)


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


@functools.lru_cache(maxsize=256)
def _get_player_texture(
    direction: int,
    player_idx: int,
    is_selected: bool,
    size: int,
) -> np.ndarray:
    """Cached wrapper around create_player_texture.

    The full set of combinations is small (players × 4 dirs × 2 selected
    states), so every texture is computed at most once per size.

    Args:
        direction: Facing direction (Action enum value).
        player_idx: Player index.
        is_selected: Whether this player is currently selected.
        size: Block pixel size.

    Returns:
        RGBA numpy array of shape (size, size, 4).
    """
    return create_player_texture(direction, player_idx, is_selected, size)


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


def render_inventory_bar(
    state: EnvState,
    width: int,
    selected_item: int = 0,
) -> np.ndarray:
    """Render inventory bar showing the selected player's inventory.

    Each slot corresponds to an ``ItemType`` index. The slot matching
    *selected_item* is highlighted with a white border.

    Args:
        state: Current environment state containing inventory data.
        width: Width of the bar in pixels (should match map render width).
        selected_item: ``ItemType`` index of the currently selected item.

    Returns:
        RGB numpy array of shape (INVENTORY_BAR_HEIGHT, width, 3).
    """
    bar = np.full(
        (INVENTORY_BAR_HEIGHT, width, 3),
        (40, 40, 40),
        dtype=np.uint8,
    )

    slot_width = width // NUM_ITEM_TYPES
    slot_size = min(slot_width - 4, INVENTORY_BAR_HEIGHT - 4)

    selected_player = int(state.selected_player)
    inventory = np.array(state.player_inventory[selected_player])

    for item_idx in range(NUM_ITEM_TYPES):
        x_center = item_idx * slot_width + slot_width // 2
        x_start = x_center - slot_size // 2
        y_start = (INVENTORY_BAR_HEIGHT - slot_size) // 2

        is_selected_slot = item_idx == selected_item
        slot_bg = (100, 100, 100) if is_selected_slot else (60, 60, 60)
        bar[
            y_start : y_start + slot_size,
            x_start : x_start + slot_size,
        ] = slot_bg

        if is_selected_slot:
            bar[y_start, x_start : x_start + slot_size] = (
                255,
                255,
                255,
            )
            bar[
                y_start + slot_size - 1,
                x_start : x_start + slot_size,
            ] = (255, 255, 255)
            bar[y_start : y_start + slot_size, x_start] = (
                255,
                255,
                255,
            )
            bar[
                y_start : y_start + slot_size,
                x_start + slot_size - 1,
            ] = (255, 255, 255)

        count = int(inventory[item_idx])

        if item_idx != 0 and count > 0:
            pad = 2
            color = ITEM_COLORS.get(item_idx, (128, 128, 128))
            bar[
                y_start + pad : y_start + slot_size - pad,
                x_start + pad : x_start + slot_size - pad,
            ] = color

    return bar


# Derived from the single source of truth in constants.ITEM_TO_MACHINE.
# New machine types added there automatically appear here.
MACHINE_TO_ITEM: dict[int, int] = {
    int(mt): int(it) for it, mt in ITEM_TO_MACHINE.items()
}

# Dark arrow colour drawn on top of the gold conveyor belt square.
_BELT_ARROW_COLOR: tuple[int, int, int, int] = (60, 50, 10, 255)


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
# Shared drawing helpers for ore / plate / item / machine textures
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
    """Paint irregular darker blobs (and optional bright glints) onto an icon.

    The patch distribution is deterministic for a given ``seed``, so every
    tile of the same ore type looks identical between frames. When
    ``crystalline`` is True, extra sharp single-pixel highlights are added
    on top (used for silicon).

    Args:
        icon: RGBA array modified in place.
        base_rgb: Base ore color (already filled into the icon).
        seed: RNG seed controlling patch layout.
        crystalline: Whether to add crystalline glint sparkles.
    """
    s = icon.shape[0]
    if s < 4:
        return
    rng = np.random.default_rng(seed)
    dark = _shade(base_rgb, -35)
    deep = _shade(base_rgb, -60)
    light = _shade(base_rgb, 45)

    n_patches = max(4, (s * s) // 40)
    max_r = max(1, s // 7)
    for _ in range(n_patches):
        cy = int(rng.integers(0, s))
        cx = int(rng.integers(0, s))
        r = int(rng.integers(1, max_r + 1))
        color = deep if rng.random() < 0.25 else dark
        ys, xs = np.ogrid[:s, :s]
        # Jitter the radius squared so blobs aren't perfect circles.
        jitter = float(rng.uniform(-1.5, 1.5))
        mask = ((ys - cy) ** 2 + (xs - cx) ** 2) <= (r * r + jitter)
        icon[mask, :3] = color

    # Sparse bright pinpoints (rare single glints).
    n_glints = max(1, s // 16)
    for _ in range(n_glints):
        gy = int(rng.integers(0, s))
        gx = int(rng.integers(0, s))
        icon[gy, gx, :3] = light

    if crystalline:
        n_sparkles = max(3, s // 5)
        sparkle = _shade(base_rgb, 90)
        for _ in range(n_sparkles):
            sy = int(rng.integers(1, s - 1))
            sx = int(rng.integers(1, s - 1))
            icon[sy, sx, :3] = sparkle


def _draw_plate_shine(
    icon: np.ndarray,
    base_rgb: tuple[int, int, int],
) -> None:
    """Draw a diagonal highlight band on a plate icon.

    Creates a 3-pixel-wide bright diagonal slanting from the top-left
    plus subtle corner shadows, so the plate reads as a flat milled
    surface catching the light.

    Args:
        icon: RGBA array modified in place.
        base_rgb: Base plate color.
    """
    s = icon.shape[0]
    if s < 4:
        return
    bright = _shade(base_rgb, 45)
    mid = _shade(base_rgb, 22)
    dark = _shade(base_rgb, -35)

    # Diagonal band near the upper-left, three pixels wide.
    band_len = max(2, (2 * s) // 3)
    for i in range(band_len):
        for t, color in ((-1, mid), (0, bright), (1, mid)):
            y = i + t
            x = i - t
            if 0 <= y < s and 0 <= x < s:
                icon[y, x, :3] = color

    # Soft shadow in the top-left-most corner and bottom-right edge.
    icon[0, 0, :3] = dark
    icon[s - 1, s - 1, :3] = dark
    if s >= 8:
        icon[s - 1, s - 2, :3] = dark
        icon[s - 2, s - 1, :3] = dark


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
    """Draw twin diagonal strands suggesting a coiled wire.

    Args:
        icon: RGBA array modified in place.
        base_rgb: Base wire color.
    """
    s = icon.shape[0]
    dark = _shade(base_rgb, -55)
    sep = max(2, s // 5)
    for i in range(s):
        icon[i, i, :3] = dark
        if i + sep < s:
            icon[i, i + sep, :3] = dark


def _draw_circuit_traces(
    icon: np.ndarray,
    base_rgb: tuple[int, int, int],
) -> None:
    """Draw PCB traces: two right-angle paths meeting a solder pad.

    Args:
        icon: RGBA array modified in place.
        base_rgb: Base circuit (green) color.
    """
    s = icon.shape[0]
    if s < 6:
        return
    dark = _shade(base_rgb, -70)
    pad = _shade(base_rgb, 60)
    mid = s // 2
    q = max(1, s // 4)
    # Horizontal trace near top, vertical spine down to lower-right pad.
    icon[q, 1 : mid + 1, :3] = dark
    icon[q : s - q, mid, :3] = dark
    icon[s - q - 1, mid : s - 1, :3] = dark
    # Two solder pads.
    icon[q - 1 : q + 2, 1:3, :3] = pad
    icon[s - q - 2 : s - q + 1, s - 3 : s - 1, :3] = pad


def _draw_motor(
    icon: np.ndarray,
    base_rgb: tuple[int, int, int],
) -> None:
    """Draw a motor: cylindrical housing with a visible shaft.

    Args:
        icon: RGBA array modified in place.
        base_rgb: Base motor color.
    """
    s = icon.shape[0]
    if s < 6:
        return
    dark = _shade(base_rgb, -55)
    bright = _shade(base_rgb, 55)
    mid = s // 2
    body_top = s // 4
    body_bot = 3 * s // 4
    body_left = s // 5
    body_right = 3 * s // 5
    # Housing outline.
    icon[body_top:body_bot, body_left:body_right, :3] = dark
    # Shaft sticking out to the right.
    shaft_y0 = mid - max(1, s // 12)
    shaft_y1 = mid + max(1, s // 12) + 1
    icon[shaft_y0:shaft_y1, body_right : body_right + s // 5, :3] = dark
    # A couple of rivet highlights on the housing.
    for ry in (body_top + 1, body_bot - 2):
        for rx in (body_left + 1, body_right - 2):
            if 0 <= ry < s and 0 <= rx < s:
                icon[ry, rx, :3] = bright


def _draw_sensor_lens(
    icon: np.ndarray,
    base_rgb: tuple[int, int, int],
) -> None:
    """Draw a sensor as a single large lens with a bright glint.

    Args:
        icon: RGBA array modified in place.
        base_rgb: Base sensor body color.
    """
    s = icon.shape[0]
    if s < 6:
        return
    dark = _shade(base_rgb, -55)
    darker = _shade(base_rgb, -30)
    bright = _shade(base_rgb, 90)
    cy = cx = s // 2
    outer_r = s // 3
    inner_r = max(1, s // 5)
    ys, xs = np.ogrid[:s, :s]
    dist = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    icon[(dist <= outer_r) & (dist > inner_r), :3] = dark
    icon[dist <= inner_r, :3] = darker
    icon[cy, cx, :3] = bright


def _draw_frame_ibeam(
    icon: np.ndarray,
    base_rgb: tuple[int, int, int],
) -> None:
    """Draw a structural frame as an I-beam cross-section.

    Args:
        icon: RGBA array modified in place.
        base_rgb: Base frame color.
    """
    s = icon.shape[0]
    if s < 8:
        return
    dark = _shade(base_rgb, -45)
    flange_h = max(1, s // 6)
    web_half = max(1, s // 8)
    mid = s // 2
    margin = s // 6
    # Top flange.
    icon[margin : margin + flange_h, margin : s - margin, :3] = dark
    # Bottom flange.
    icon[s - margin - flange_h : s - margin, margin : s - margin, :3] = dark
    # Web in the middle.
    icon[margin : s - margin, mid - web_half : mid + web_half + 1, :3] = dark


def _draw_flask(
    icon: np.ndarray,
    base_rgb: tuple[int, int, int],
    *,
    advanced: bool = False,
) -> None:
    """Draw a science-pack flask silhouette.

    The basic and advanced packs share the same flask; the advanced
    variant adds a bright glow in the center of the bulb.

    Args:
        icon: RGBA array modified in place.
        base_rgb: Base flask color.
        advanced: If True, draw a central glow.
    """
    s = icon.shape[0]
    if s < 8:
        return
    dark = _shade(base_rgb, -55)
    bright = _shade(base_rgb, 60)
    mid = s // 2
    neck_top = s // 6
    neck_bot = s // 2
    neck_half = max(1, s // 12)
    # Neck.
    icon[neck_top:neck_bot, mid - neck_half : mid + neck_half + 1, :3] = dark
    # Bulb body: a filled circle sitting low in the icon.
    body_cy = (3 * s) // 5
    body_r = s // 3
    ys, xs = np.ogrid[:s, :s]
    dist = np.sqrt((ys - body_cy) ** 2 + (xs - mid) ** 2)
    ring = (dist <= body_r) & (dist >= body_r - 1.5)
    fill = (dist <= body_r - 1.2) & (ys >= neck_bot - 1)
    icon[ring, :3] = dark
    icon[fill, :3] = bright
    if advanced:
        glow = (255, 230, 120)
        icon[body_cy - 1 : body_cy + 2, mid - 1 : mid + 2, :3] = glow


def _draw_rocket(
    icon: np.ndarray,
    base_rgb: tuple[int, int, int],
) -> None:
    """Draw a rocket silhouette: nose, body, fins, exhaust spark.

    Args:
        icon: RGBA array modified in place.
        base_rgb: Base rocket body color.
    """
    s = icon.shape[0]
    if s < 8:
        return
    dark = _shade(base_rgb, -90)
    exhaust = (255, 210, 80)
    mid = s // 2
    body_w = max(2, s // 5)
    body_top = s // 5
    body_bot = (4 * s) // 5
    # Body.
    icon[body_top:body_bot, mid - body_w // 2 : mid - body_w // 2 + body_w, :3] = dark
    # Conical nose.
    for t in range(body_top):
        y = body_top - 1 - t
        half = body_w // 2 - (body_top - 1 - t) // 2
        if half < 0:
            continue
        icon[y, mid - half : mid + half + 1, :3] = dark
    # Fins.
    fin_y = body_bot - 2
    fin_len = max(1, s // 8)
    icon[fin_y : fin_y + 2, max(0, mid - body_w) : mid - body_w // 2, :3] = dark
    icon[fin_y : fin_y + 2, mid + body_w // 2 + 1 : min(s, mid + body_w + 1), :3] = dark
    # Exhaust spark below body.
    if body_bot < s:
        icon[body_bot : min(body_bot + fin_len, s), mid, :3] = exhaust


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
    Both the map renderer and the menu UI call this function so that
    placed machines and inventory icons are always identical.

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

    # --- Intermediate items ---
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

    # --- Machines (draw interior, then the shared bevel frame) ---
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
        elif item_type == int(ItemType.ROCKET) and size >= 6:
            _draw_rocket(icon, rgb)
        _draw_machine_frame(icon)
        return icon

    return icon


def render_machine_overlays(
    image: np.ndarray,
    state: EnvState,
    block_pixel_size: int,
    frame_tick: int = 0,
) -> None:
    """Draw machine overlays on tiles that have machines.

    Uses :func:`render_item_icon` for each machine so that placed
    machines look identical to their inventory icons.  When
    *frame_tick* is non-zero, active machines pulse and idle machines
    dim.

    Args:
        image: RGBA image to draw on (modified in place)
        state: Current environment state
        block_pixel_size: Size of each block in pixels
        frame_tick: Monotonic frame counter (0 = static)
    """
    machine_size = int(block_pixel_size * 0.6)
    offset = (block_pixel_size - machine_size) // 2

    machine_types = np.array(state.machine_types)
    tile_entity = np.array(state.tile_entity)
    ent_direction = np.array(state.ent_direction)
    ent_power = np.array(state.ent_power)

    ys, xs = np.nonzero(machine_types != MachineType.NONE)
    if ys.size == 0:
        return

    for y, x in zip(ys, xs):
        machine_type = int(machine_types[y, x])
        item_type = MACHINE_TO_ITEM.get(machine_type, int(ItemType.EMPTY))

        eidx = int(tile_entity[y, x])
        direction = int(ent_direction[eidx]) if eidx >= 0 else int(Direction.DOWN)

        icon = render_item_icon(
            item_type,
            machine_size,
            direction,
        )

        if frame_tick > 0:
            if machine_type == int(MachineType.MINER):
                active = is_miner_active(state, int(y), int(x))
            elif machine_type == int(MachineType.ASSEMBLER):
                active = int(ent_power[eidx]) > 0 if eidx >= 0 else False
            else:
                active = True
            icon = apply_activity_tint(icon, active, frame_tick)

        y_start = y * block_pixel_size + offset
        x_start = x * block_pixel_size + offset
        image[
            y_start : y_start + machine_size,
            x_start : x_start + machine_size,
        ] = icon

    draw_belt_cargo(image, state, block_pixel_size)


def render_pixels(
    state: EnvState,
    block_pixel_size: int = BLOCK_PIXEL_SIZE,
    frame_tick: int = 0,
) -> np.ndarray:
    """Render the environment state as an RGB pixel image.

    Renders the map, machines, and all players. The selected player has
    a white highlight ring. Does not include the inventory menu.

    When *frame_tick* is non-zero, world animations are applied: water
    shimmers, active machines pulse, and belt cargo dots slide along
    belts.  The default of 0 produces the same static output as before
    so existing callers are unaffected.

    Args:
        state: Current environment state
        block_pixel_size: Size of each block in pixels
        frame_tick: Monotonic frame counter (0 = static rendering)

    Returns:
        RGB numpy array of the rendered scene
    """
    texture_lookup = build_texture_lookup(block_pixel_size)

    map_array = np.array(state.map)
    map_height, map_width = map_array.shape

    # Clamp unknown block IDs to the DIRT fallback.
    max_id = texture_lookup.shape[0] - 1
    safe_map = np.clip(map_array, 0, max_id)

    # Single numpy index: (H, W, size, size, 4) -> (H*size, W*size, 4)
    tile_textures = texture_lookup[safe_map]
    image = tile_textures.transpose(0, 2, 1, 3, 4).reshape(
        map_height * block_pixel_size, map_width * block_pixel_size, 4
    )
    # Make writable — the reshape may return a view into the read-only cache.
    image = np.array(image)

    if frame_tick > 0:
        animate_water(image, map_array, block_pixel_size, frame_tick)

    render_machine_overlays(image, state, block_pixel_size, frame_tick)

    player_positions = np.array(state.player_positions)
    player_directions = np.array(state.player_directions)
    selected = int(state.selected_player)
    num_players = player_positions.shape[0]

    for player_idx in range(num_players):
        direction = int(player_directions[player_idx])
        is_selected = player_idx == selected
        player_texture = _get_player_texture(
            direction, player_idx, is_selected, block_pixel_size
        )

        px, py = (
            int(player_positions[player_idx, 0]),
            int(player_positions[player_idx, 1]),
        )
        py_start = py * block_pixel_size
        px_start = px * block_pixel_size
        _alpha_blend_inplace(
            image,
            player_texture,
            py_start,
            px_start,
            block_pixel_size,
        )

    return image[:, :, :3]


def _alpha_blend_inplace(
    background: np.ndarray,
    foreground: np.ndarray,
    y_start: int,
    x_start: int,
    size: int,
) -> None:
    """Blend a foreground texture onto the background using alpha compositing.

    Args:
        background: RGBA background image to modify in place
        foreground: RGBA foreground texture
        y_start: Y coordinate of top-left corner
        x_start: X coordinate of top-left corner
        size: Size of the foreground texture
    """
    fg_alpha = foreground[:, :, 3:4].astype(np.float32) / 255.0
    bg_region = background[y_start : y_start + size, x_start : x_start + size]
    blended = foreground[:, :, :3].astype(np.float32) * fg_alpha + bg_region[
        :, :, :3
    ].astype(np.float32) * (1 - fg_alpha)
    bg_region[:, :, :3] = blended.astype(np.uint8)
    bg_region[:, :, 3] = 255


# ---------------------------------------------------------------------------
# Animation helpers
#
# All world-animation logic lives in this section. It is deliberately
# self-contained so it can be extracted into its own module later.
# Every function here takes a `frame_tick` counter and operates on
# plain NumPy arrays.  Nothing in this section touches JAX.
# ---------------------------------------------------------------------------

_TWO_PI: float = 2.0 * np.pi

# Machine activity pulse parameters.
_PULSE_PERIOD: int = 30  # frames for one full sine cycle (~1 s at 30 FPS)
_PULSE_MIN: int = 10
_PULSE_MAX: int = 30
_IDLE_DIM: float = 0.65  # RGB multiplier for idle machines

# Water wave stripe parameters.
_WAVE_STRIPE_COLOR: tuple[int, int, int] = (90, 190, 245)
_WAVE_STRIPE_WIDTH: int = 2  # px thickness of each stripe
_WAVE_PERIOD: int = 60  # frames for stripes to scroll one full cycle
_WAVE_SPACING: int = 5  # px between stripe centers


def animate_water(
    image: np.ndarray,
    map_array: np.ndarray,
    block_pixel_size: int,
    frame_tick: int,
) -> None:
    """Draw synchronized wave stripes across all water tiles.

    Diagonal stripes in a lighter blue scroll steadily across every
    water tile in lockstep, giving the impression of flowing water.
    The stripes tile seamlessly across adjacent water tiles because
    the pattern is computed in global pixel coordinates.

    Args:
        image: RGBA pixel image, modified in place.
        map_array: Integer block-type grid of shape ``(H, W)``.
        block_pixel_size: Tile side length in pixels.
        frame_tick: Monotonic frame counter from the game loop.
    """
    water_ys, water_xs = np.nonzero(map_array == BlockType.WATER)
    if water_ys.size == 0:
        return

    scroll = (frame_tick * _WAVE_SPACING) // _WAVE_PERIOD

    # Local pixel offsets within one tile.
    local_r = np.arange(block_pixel_size)
    local_c = np.arange(block_pixel_size)
    lr, lc = np.meshgrid(local_r, local_c, indexing="ij")

    for idx in range(water_ys.size):
        y, x = int(water_ys[idx]), int(water_xs[idx])
        r0 = y * block_pixel_size
        c0 = x * block_pixel_size

        diag = (r0 + lr) + (c0 + lc) + scroll
        stripe_mask = (diag % _WAVE_SPACING) < _WAVE_STRIPE_WIDTH

        region = image[
            r0 : r0 + block_pixel_size,
            c0 : c0 + block_pixel_size,
        ]
        region[:, :, :3][stripe_mask] = _WAVE_STRIPE_COLOR


def apply_activity_tint(
    icon: np.ndarray,
    active: bool,
    frame_tick: int,
) -> np.ndarray:
    """Return a tinted copy of *icon* based on machine activity.

    Active machines get a pulsing brightness boost.  Idle machines
    are dimmed.  When ``frame_tick`` is 0 the icon is returned
    unchanged so the static renderer path has zero overhead.

    Args:
        icon: Base RGBA icon from :func:`render_item_icon`.
        active: Whether the machine is currently doing work.
        frame_tick: Monotonic frame counter from the game loop.

    Returns:
        A new RGBA array (never mutates the cached *icon*).
    """
    if frame_tick == 0:
        return icon

    result = icon.copy()
    if active:
        phase = _TWO_PI * frame_tick / _PULSE_PERIOD
        boost = int(
            _PULSE_MIN + (_PULSE_MAX - _PULSE_MIN) * (0.5 + 0.5 * np.sin(phase))
        )
        rgb = result[:, :, :3].astype(np.int16) + boost
        np.clip(rgb, 0, 255, out=rgb)
        result[:, :, :3] = rgb.astype(np.uint8)
    else:
        result[:, :, :3] = (result[:, :, :3].astype(np.float32) * _IDLE_DIM).astype(
            np.uint8
        )
    return result


def draw_belt_cargo(
    image: np.ndarray,
    state: EnvState,
    block_pixel_size: int,
) -> None:
    """Draw a static item dot on conveyor belts that hold items.

    Each belt carrying items shows a small coloured square at its
    centre.  The game simulation handles the actual item movement
    between tiles, so the dot just indicates presence.

    Args:
        image: RGBA pixel image, modified in place.
        state: Current environment state.
        block_pixel_size: Tile side length in pixels.
    """
    machine_types = np.array(state.machine_types)
    tile_entity = np.array(state.tile_entity)
    ent_buf_type = np.array(state.ent_buf_type)
    ent_buf_count = np.array(state.ent_buf_count)

    show_cargo = (machine_types == MachineType.CONVEYOR_BELT) | (
        machine_types == MachineType.PALLET
    )
    belt_ys, belt_xs = np.nonzero(show_cargo)
    if belt_ys.size == 0:
        return

    dot_size = max(4, block_pixel_size // 4)
    border = max(2, dot_size // 3)
    outer = dot_size + 2 * border
    half_outer = outer // 2
    mid = block_pixel_size // 2

    for idx in range(belt_ys.size):
        y, x = int(belt_ys[idx]), int(belt_xs[idx])
        eidx = int(tile_entity[y, x])
        if eidx < 0:
            continue
        item_type = int(ent_buf_type[eidx])
        item_count = int(ent_buf_count[eidx])
        if item_type == 0 or item_count <= 0:
            continue

        color = ITEM_COLORS.get(item_type, (128, 128, 128))

        py0 = y * block_pixel_size + mid - half_outer
        px0 = x * block_pixel_size + mid - half_outer

        h, w = image.shape[:2]
        # Dark outline
        oy0 = max(0, py0)
        ox0 = max(0, px0)
        oy1 = min(h, py0 + outer)
        ox1 = min(w, px0 + outer)
        if oy0 < oy1 and ox0 < ox1:
            image[oy0:oy1, ox0:ox1, :3] = (20, 20, 20)
            image[oy0:oy1, ox0:ox1, 3] = 255

        # Inner fill
        iy0 = max(0, py0 + border)
        ix0 = max(0, px0 + border)
        iy1 = min(h, py0 + border + dot_size)
        ix1 = min(w, px0 + border + dot_size)
        if iy0 < iy1 and ix0 < ix1:
            image[iy0:iy1, ix0:ix1, :3] = color
            image[iy0:iy1, ix0:ix1, 3] = 255


def is_miner_active(state: EnvState, y: int, x: int) -> bool:
    """Check whether the miner at ``(y, x)`` is actively mining.

    A miner is active when it has power, the tile below still holds
    resources, and the inventory is not completely full. Fullness is
    checked by summing all item counts in the machine's pouch.

    Args:
        state: Current environment state.
        y: Row of the miner tile.
        x: Column of the miner tile.

    Returns:
        True if the miner is doing work this tick.
    """
    eidx = int(state.tile_entity[y, x])
    if eidx < 0:
        return False
    has_power = int(state.ent_power[eidx]) > 0
    has_resources = int(state.block_resources[y, x]) > 0
    total_count = int(state.ent_buf_count[eidx])
    has_space = total_count < MAX_MACHINE_STACK_SIZE
    return has_power and has_resources and has_space
