"""Pixel rendering for the FactoriaX environment."""

import functools

import numpy as np

from factoriax.constants import (
    BLOCK_PIXEL_SIZE,
    ITEM_COLORS,
    NUM_INVENTORY_SLOTS,
    Action,
    BlockType,
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
    return texture[np.ix_(idx, idx)]


def create_default_textures(size: int = BLOCK_PIXEL_SIZE) -> dict[int, np.ndarray]:
    """Create simple default textures if asset files don't exist.

    Args:
        size: Side length of each texture in pixels.

    Returns:
        Dictionary mapping BlockType values to RGBA texture arrays
    """
    colors: dict[int, tuple[int, int, int]] = {
        int(BlockType.DIRT): (139, 90, 43),
        int(BlockType.WATER): (64, 164, 223),
        int(BlockType.IRON): (192, 192, 192),
        int(BlockType.COPPER): (184, 115, 51),
        int(BlockType.COAL): (54, 54, 54),
    }
    textures: dict[int, np.ndarray] = {}
    for block_id, (r, g, b) in colors.items():
        t = np.empty((size, size, 4), dtype=np.uint8)
        t[:, :] = (r, g, b, 255)
        textures[block_id] = t
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
    direction: int = Action.DOWN,
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
    if direction == Action.UP:
        pys = 1 + rows
        centers_x = center
    elif direction == Action.DOWN:
        pys = size - 2 - rows
        centers_x = center
    elif direction == Action.LEFT:
        pys = center
        centers_x = 1 + rows
    else:  # RIGHT
        pys = center
        centers_x = size - 2 - rows

    for i in range(indicator_size):
        offsets = np.arange(-i, i + 1)
        if direction in (Action.UP, Action.DOWN):
            py = int(pys[i])
            pxs = np.clip(centers_x + offsets, 0, size - 1)
            player[py, pxs] = [*indicator_color, 255]
        else:
            px = int(centers_x[i])
            pys_clipped = np.clip(pys + offsets, 0, size - 1)
            player[pys_clipped, px] = [*indicator_color, 255]

    return player


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


def render_inventory_bar(state: EnvState, width: int) -> np.ndarray:
    """Render inventory bar showing the selected player's inventory.

    The currently selected slot is highlighted with a white border.
    If the player is crafting, a progress indicator is shown.

    Args:
        state: Current environment state containing inventory data
        width: Width of the bar in pixels (should match map render width)

    Returns:
        RGB numpy array of shape (INVENTORY_BAR_HEIGHT, width, 3)
    """
    bar = np.full((INVENTORY_BAR_HEIGHT, width, 3), (40, 40, 40), dtype=np.uint8)

    slot_width = width // NUM_INVENTORY_SLOTS
    slot_size = min(slot_width - 4, INVENTORY_BAR_HEIGHT - 4)

    selected_player = int(state.selected_player)
    selected_slot = int(state.selected_slots[selected_player])
    inventory_items = np.array(state.inventory_items[selected_player])
    inventory_counts = np.array(state.inventory_counts[selected_player])
    craft_progress = int(state.craft_progress[selected_player])

    for slot_idx in range(NUM_INVENTORY_SLOTS):
        x_center = slot_idx * slot_width + slot_width // 2
        x_start = x_center - slot_size // 2
        y_start = (INVENTORY_BAR_HEIGHT - slot_size) // 2

        is_selected_slot = slot_idx == selected_slot
        slot_bg = (100, 100, 100) if is_selected_slot else (60, 60, 60)
        bar[y_start : y_start + slot_size, x_start : x_start + slot_size] = slot_bg

        if is_selected_slot:
            bar[y_start, x_start : x_start + slot_size] = (255, 255, 255)
            bar[y_start + slot_size - 1, x_start : x_start + slot_size] = (
                255,
                255,
                255,
            )
            bar[y_start : y_start + slot_size, x_start] = (255, 255, 255)
            bar[y_start : y_start + slot_size, x_start + slot_size - 1] = (
                255,
                255,
                255,
            )

        item_type = int(inventory_items[slot_idx])
        count = int(inventory_counts[slot_idx])

        if item_type != 0 and count > 0:
            pad = 2
            color = ITEM_COLORS.get(item_type, (128, 128, 128))
            bar[
                y_start + pad : y_start + slot_size - pad,
                x_start + pad : x_start + slot_size - pad,
            ] = color

    if craft_progress > 0:
        indicator_width = 20
        indicator_x = width - indicator_width - 4
        bar[2:6, indicator_x : indicator_x + indicator_width] = (100, 200, 100)

    return bar


MACHINE_TO_ITEM: dict[int, int] = {
    int(MachineType.MINER): int(ItemType.MINER),
    int(MachineType.CHEST): int(ItemType.CHEST),
    int(MachineType.CONVEYOR_BELT): int(ItemType.CONVEYOR_BELT),
    int(MachineType.ARM): int(ItemType.ARM),
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
        direction: Action.LEFT / RIGHT / UP / DOWN.
    """
    h, w = image.shape[:2]
    for d in range(-size, size + 1):
        depth = size - abs(d)
        for t in range(depth + 1):
            if direction == Action.RIGHT:
                py, px = cy + d, cx + t
            elif direction == Action.LEFT:
                py, px = cy + d, cx - t
            elif direction == Action.DOWN:
                py, px = cy + t, cx + d
            elif direction == Action.UP:
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

    if direction in (Action.LEFT, Action.RIGHT):
        cy = mid
        for i in range(3):
            cx = size * (1 + 2 * i) // 6
            _draw_chevron(icon, cy, cx, arrow_size, direction)
    elif direction in (Action.UP, Action.DOWN):
        cx = mid
        for i in range(3):
            cy = size * (1 + 2 * i) // 6
            _draw_chevron(icon, cy, cx, arrow_size, direction)


# Colours for the arm (inserter) direction overlay.
_ARM_LINE_COLOR: tuple[int, int, int, int] = (30, 55, 110, 255)
_ARM_HEAD_COLOR: tuple[int, int, int, int] = (200, 220, 255, 255)
_ARM_TAIL_COLOR: tuple[int, int, int, int] = (40, 70, 140, 255)


def _draw_arm_indicator(icon: np.ndarray, direction: int) -> None:
    """Draw a pick-to-deposit flow indicator on an arm icon.

    Draws a centre line along the arm's facing axis with a small
    circle on the pick (back) end and a chevron arrowhead on the
    deposit (front) end so the player can tell input from output
    at a glance.

    Args:
        icon: RGBA array of shape ``(size, size, 4)``, modified
            in place.
        direction: ``Action`` direction the arm faces (deposit side).
    """
    size = icon.shape[0]
    mid = size // 2
    thickness = max(1, size // 10)
    half_t = thickness // 2

    # Shaft line along the facing axis, inset from edges.
    margin = max(2, size // 6)
    if direction in (Action.LEFT, Action.RIGHT):
        icon[mid - half_t : mid + half_t + 1, margin : size - margin] = (
            _ARM_LINE_COLOR
        )
    else:
        icon[margin : size - margin, mid - half_t : mid + half_t + 1] = (
            _ARM_LINE_COLOR
        )

    # Arrowhead on the deposit (forward) end.
    arrow_size = max(1, size // 8)
    if direction == Action.RIGHT:
        _draw_arm_chevron(icon, mid, size - margin - 1, arrow_size, direction)
    elif direction == Action.LEFT:
        _draw_arm_chevron(icon, mid, margin, arrow_size, direction)
    elif direction == Action.DOWN:
        _draw_arm_chevron(icon, size - margin - 1, mid, arrow_size, direction)
    elif direction == Action.UP:
        _draw_arm_chevron(icon, margin, mid, arrow_size, direction)

    # Small circle on the pick (back) end.
    radius = max(1, size // 8)
    if direction == Action.RIGHT:
        cy, cx = mid, margin
    elif direction == Action.LEFT:
        cy, cx = mid, size - margin - 1
    elif direction == Action.DOWN:
        cy, cx = mid, margin
    else:  # UP
        cy, cx = mid, size - margin - 1
    # Swap for vertical directions — circle is at the opposite end.
    if direction == Action.DOWN:
        cy, cx = margin, mid
    elif direction == Action.UP:
        cy, cx = size - margin - 1, mid

    ys, xs = np.ogrid[:size, :size]
    dist = (xs - cx) ** 2 + (ys - cy) ** 2
    icon[dist <= radius**2] = _ARM_TAIL_COLOR


def _draw_arm_chevron(
    image: np.ndarray,
    cy: int,
    cx: int,
    size: int,
    direction: int,
) -> None:
    """Draw a filled chevron arrowhead for the arm deposit end.

    Same geometry as ``_draw_chevron`` but uses the arm head colour.

    Args:
        image: RGBA image array (modified in place).
        cy: Centre row of the chevron.
        cx: Centre column of the chevron.
        size: Half-extent of the arrow in pixels.
        direction: Action.LEFT / RIGHT / UP / DOWN.
    """
    h, w = image.shape[:2]
    for d in range(-size, size + 1):
        depth = size - abs(d)
        for t in range(depth + 1):
            if direction == Action.RIGHT:
                py, px = cy + d, cx + t
            elif direction == Action.LEFT:
                py, px = cy + d, cx - t
            elif direction == Action.DOWN:
                py, px = cy + t, cx + d
            elif direction == Action.UP:
                py, px = cy - t, cx + d
            else:
                return
            if 0 <= py < h and 0 <= px < w:
                image[py, px] = _ARM_HEAD_COLOR


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
            if direction == Action.RIGHT:
                py, px = mid + s, size - 1 - t
            elif direction == Action.LEFT:
                py, px = mid + s, t
            elif direction == Action.DOWN:
                py, px = size - 1 - t, mid + s
            else:  # UP
                py, px = t, mid + s
            if 0 <= py < size and 0 <= px < size:
                icon[py, px] = _MINER_ARROW_COLOR


@functools.lru_cache(maxsize=128)
def render_item_icon(
    item_type: int,
    size: int,
    direction: int | None = None,
) -> np.ndarray:
    """Render a square RGBA icon for an item type.

    This is the single source of truth for how an item looks visually.
    Both the map renderer and the menu UI call this function so that
    placed machines and inventory icons are always identical.

    Conveyor belts get three chevron arrows overlaid on the base colour.
    Arms get a flow indicator showing pick (circle, back) and deposit
    (arrowhead, front) sides.  When *direction* is ``None`` (e.g. in a
    menu with no placement context), arrows default to pointing right.

    Args:
        item_type: ``ItemType`` integer value.
        size: Side length of the returned square in pixels.
        direction: Optional ``Action`` direction for directional items.
            Ignored for non-directional items.

    Returns:
        RGBA uint8 array of shape ``(size, size, 4)``.
    """
    rgb = ITEM_COLORS.get(item_type, (128, 128, 128))
    icon = np.full((size, size, 4), (*rgb, 255), dtype=np.uint8)

    if item_type == ItemType.CONVEYOR_BELT and size >= 6:
        belt_dir = direction if direction is not None else int(Action.RIGHT)
        _draw_belt_arrows(icon, belt_dir)
    elif item_type == ItemType.ARM and size >= 6:
        arm_dir = direction if direction is not None else int(Action.RIGHT)
        _draw_arm_indicator(icon, arm_dir)
    elif item_type == ItemType.MINER and size >= 6:
        miner_dir = direction if direction is not None else int(Action.RIGHT)
        _draw_miner_indicator(icon, miner_dir)

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
    ys, xs = np.nonzero(machine_types != MachineType.NONE)
    if ys.size == 0:
        return

    directions = np.array(state.machine_direction)

    for y, x in zip(ys, xs):
        machine_type = int(machine_types[y, x])
        item_type = MACHINE_TO_ITEM.get(machine_type, int(ItemType.EMPTY))
        direction = int(directions[y, x])

        icon = render_item_icon(item_type, machine_size, direction)

        if frame_tick > 0:
            if machine_type == int(MachineType.MINER):
                active = is_miner_active(state, y, x)
            elif machine_type == int(MachineType.ARM):
                active = is_arm_active(state, y, x)
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

# Miner output slot index (mirrors machines.py).
_ANIM_MINER_OUTPUT_SLOT: int = 1


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
        boost = int(_PULSE_MIN + (_PULSE_MAX - _PULSE_MIN) * (
            0.5 + 0.5 * np.sin(phase)
        ))
        rgb = result[:, :, :3].astype(np.int16) + boost
        np.clip(rgb, 0, 255, out=rgb)
        result[:, :, :3] = rgb.astype(np.uint8)
    else:
        result[:, :, :3] = (
            result[:, :, :3].astype(np.float32) * _IDLE_DIM
        ).astype(np.uint8)
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
    belt_mask = machine_types == MachineType.CONVEYOR_BELT
    belt_ys, belt_xs = np.nonzero(belt_mask)
    if belt_ys.size == 0:
        return

    inv_items = np.array(state.machine_inventory_items)
    inv_counts = np.array(state.machine_inventory_counts)

    dot_size = max(4, block_pixel_size // 4)
    border = max(2, dot_size // 3)
    outer = dot_size + 2 * border
    half_outer = outer // 2
    mid = block_pixel_size // 2

    for idx in range(belt_ys.size):
        y, x = int(belt_ys[idx]), int(belt_xs[idx])
        if int(inv_counts[y, x, 0]) <= 0:
            continue

        item_type = int(inv_items[y, x, 0])
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
    resources, and the output slot is not completely full.

    Args:
        state: Current environment state.
        y: Row of the miner tile.
        x: Column of the miner tile.

    Returns:
        True if the miner is doing work this tick.
    """
    has_power = int(state.machine_power[y, x]) > 0
    has_resources = int(state.block_resources[y, x]) > 0
    output_count = int(
        state.machine_inventory_counts[y, x, _ANIM_MINER_OUTPUT_SLOT]
    )
    from factoriax.constants import MAX_MACHINE_STACK_SIZE
    has_space = output_count < MAX_MACHINE_STACK_SIZE
    return has_power and has_resources and has_space


def is_arm_active(state: EnvState, y: int, x: int) -> bool:
    """Check whether the arm at ``(y, x)`` is doing work.

    An arm is considered active if its buffer slot holds items
    (mid-transfer).

    Args:
        state: Current environment state.
        y: Row of the arm tile.
        x: Column of the arm tile.

    Returns:
        True if the arm buffer is non-empty.
    """
    return int(state.machine_inventory_counts[y, x, 0]) > 0
