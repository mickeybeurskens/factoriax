"""Pixel rendering for the FactoriaX environment."""

import numpy as np

from factoriax.constants import (
    BLOCK_PIXEL_SIZE,
    ITEM_COLORS,
    NUM_INVENTORY_SLOTS,
    NUM_RECIPES,
    RECIPE_NAMES,
    RECIPES,
    Action,
    BlockType,
    ItemType,
    MachineType,
    load_all_textures,
)
from factoriax.crafting import can_afford_recipe, count_item_in_inventory
from factoriax.state import EnvState

INVENTORY_BAR_HEIGHT = 24


def create_default_textures() -> dict[int, np.ndarray]:
    """Create simple default textures if asset files don't exist.

    Returns:
        Dictionary mapping BlockType values to RGBA texture arrays
    """
    size = BLOCK_PIXEL_SIZE
    textures: dict[int, np.ndarray] = {}

    dirt = np.zeros((size, size, 4), dtype=np.uint8)
    dirt[:, :, 0] = 139
    dirt[:, :, 1] = 90
    dirt[:, :, 2] = 43
    dirt[:, :, 3] = 255
    textures[int(BlockType.DIRT)] = dirt

    water = np.zeros((size, size, 4), dtype=np.uint8)
    water[:, :, 0] = 64
    water[:, :, 1] = 164
    water[:, :, 2] = 223
    water[:, :, 3] = 255
    textures[int(BlockType.WATER)] = water

    iron = np.zeros((size, size, 4), dtype=np.uint8)
    iron[:, :, 0] = 192
    iron[:, :, 1] = 192
    iron[:, :, 2] = 192
    iron[:, :, 3] = 255
    textures[int(BlockType.IRON)] = iron

    copper = np.zeros((size, size, 4), dtype=np.uint8)
    copper[:, :, 0] = 184
    copper[:, :, 1] = 115
    copper[:, :, 2] = 51
    copper[:, :, 3] = 255
    textures[int(BlockType.COPPER)] = copper

    coal = np.zeros((size, size, 4), dtype=np.uint8)
    coal[:, :, 0] = 54
    coal[:, :, 1] = 54
    coal[:, :, 2] = 54
    coal[:, :, 3] = 255
    textures[int(BlockType.COAL)] = coal

    return textures


PLAYER_COLORS = [
    ([255, 100, 100], [200, 50, 50]),    # Player 0: Red
    ([100, 100, 255], [50, 50, 200]),    # Player 1: Blue
    ([100, 255, 100], [50, 200, 50]),    # Player 2: Green
    ([255, 255, 100], [200, 200, 50]),   # Player 3: Yellow
    ([255, 100, 255], [200, 50, 200]),   # Player 4: Magenta
    ([100, 255, 255], [50, 200, 200]),   # Player 5: Cyan
    ([255, 180, 100], [200, 130, 50]),   # Player 6: Orange
    ([180, 100, 255], [130, 50, 200]),   # Player 7: Purple
    ([180, 255, 180], [130, 200, 130]),  # Player 8: Light green
]


def create_player_texture(
    direction: int = Action.DOWN,
    player_idx: int = 0,
    is_selected: bool = True,
) -> np.ndarray:
    """Create a player texture with directional indicator.

    The player is rendered as a circle with a small triangle indicating
    the direction they are facing. Different players have different colors,
    and the selected player has a highlight ring.

    Args:
        direction: The direction the player is facing (Action enum value)
        player_idx: Index of the player (determines color)
        is_selected: Whether this player is currently selected

    Returns:
        RGBA numpy array of shape (BLOCK_PIXEL_SIZE, BLOCK_PIXEL_SIZE, 4)
    """
    size = BLOCK_PIXEL_SIZE
    player = np.zeros((size, size, 4), dtype=np.uint8)
    center = size // 2
    radius = size // 3

    color_idx = player_idx % len(PLAYER_COLORS)
    body_color, indicator_color = PLAYER_COLORS[color_idx]

    for y in range(size):
        for x in range(size):
            dist = ((x - center) ** 2 + (y - center) ** 2) ** 0.5
            if dist <= radius:
                player[y, x] = [*body_color, 255]
            elif is_selected and radius < dist <= radius + 2:
                player[y, x] = [255, 255, 255, 255]

    indicator_size = max(2, size // 6)

    if direction == Action.UP:
        for i in range(indicator_size):
            for j in range(-i, i + 1):
                py = 1 + i
                px = center + j
                if 0 <= px < size and 0 <= py < size:
                    player[py, px] = [*indicator_color, 255]
    elif direction == Action.DOWN:
        for i in range(indicator_size):
            for j in range(-i, i + 1):
                py = size - 2 - i
                px = center + j
                if 0 <= px < size and 0 <= py < size:
                    player[py, px] = [*indicator_color, 255]
    elif direction == Action.LEFT:
        for i in range(indicator_size):
            for j in range(-i, i + 1):
                py = center + j
                px = 1 + i
                if 0 <= px < size and 0 <= py < size:
                    player[py, px] = [*indicator_color, 255]
    elif direction == Action.RIGHT:
        for i in range(indicator_size):
            for j in range(-i, i + 1):
                py = center + j
                px = size - 2 - i
                if 0 <= px < size and 0 <= py < size:
                    player[py, px] = [*indicator_color, 255]

    return player


def get_textures() -> dict[int, np.ndarray]:
    """Load textures from files, falling back to defaults if not found.

    Returns:
        Dictionary mapping BlockType values to RGBA texture arrays
    """
    try:
        return load_all_textures()
    except FileNotFoundError:
        return create_default_textures()


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
    bar = np.zeros((INVENTORY_BAR_HEIGHT, width, 3), dtype=np.uint8)
    bar[:, :] = (40, 40, 40)

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
            bar[y_start + slot_size - 1, x_start : x_start + slot_size] = (255, 255, 255)
            bar[y_start : y_start + slot_size, x_start] = (255, 255, 255)
            bar[y_start : y_start + slot_size, x_start + slot_size - 1] = (255, 255, 255)

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


ITEM_NAMES: dict[int, str] = {
    ItemType.EMPTY: "Empty",
    ItemType.COAL: "Coal",
    ItemType.IRON: "Iron",
    ItemType.COPPER: "Copper",
    ItemType.MINER: "Miner",
}


def render_inventory_menu(
    state: EnvState,
    screen_width: int,
    screen_height: int,
    menu_focus: str = "inventory",
) -> np.ndarray:
    """Render the inventory and crafting menu as an RGBA overlay.

    Creates a semi-transparent menu with two sections:
    - Left: Inventory grid (2x5)
    - Right: Crafting recipes list

    Args:
        state: Current environment state
        screen_width: Total screen width in pixels
        screen_height: Total screen height in pixels
        menu_focus: Which section is focused ("inventory" or "crafting")

    Returns:
        RGBA numpy array that can be alpha-composited over the game render
    """
    menu_width = int(screen_width * 0.75)
    menu_height = int(screen_height * 0.5)
    menu_x = (screen_width - menu_width) // 2
    menu_y = (screen_height - menu_height) // 2

    overlay = np.zeros((screen_height, screen_width, 4), dtype=np.uint8)

    bg_alpha = 217
    inv_section_width = int(menu_width * 0.6)
    craft_section_width = menu_width - inv_section_width

    overlay[menu_y : menu_y + menu_height, menu_x : menu_x + inv_section_width] = (
        40, 40, 40, bg_alpha
    )
    overlay[
        menu_y : menu_y + menu_height,
        menu_x + inv_section_width : menu_x + menu_width
    ] = (55, 55, 55, bg_alpha)

    border_color = (100, 100, 100, 255)
    overlay[menu_y, menu_x : menu_x + menu_width] = border_color
    overlay[menu_y + menu_height - 1, menu_x : menu_x + menu_width] = border_color
    overlay[menu_y : menu_y + menu_height, menu_x] = border_color
    overlay[menu_y : menu_y + menu_height, menu_x + menu_width - 1] = border_color
    overlay[menu_y : menu_y + menu_height, menu_x + inv_section_width] = border_color

    if menu_focus == "inventory":
        highlight = (80, 120, 80, 255)
        overlay[menu_y + 1, menu_x + 1 : menu_x + inv_section_width] = highlight
        overlay[menu_y + 2, menu_x + 1 : menu_x + inv_section_width] = highlight
    else:
        highlight = (80, 120, 80, 255)
        overlay[menu_y + 1, menu_x + inv_section_width + 1 : menu_x + menu_width - 1] = highlight
        overlay[menu_y + 2, menu_x + inv_section_width + 1 : menu_x + menu_width - 1] = highlight

    selected_player = int(state.selected_player)
    selected_slot = int(state.selected_slots[selected_player])
    selected_recipe = int(state.selected_recipes[selected_player])
    craft_progress = int(state.craft_progress[selected_player])
    inventory_items = np.array(state.inventory_items[selected_player])
    inventory_counts = np.array(state.inventory_counts[selected_player])

    padding = 15
    inv_grid_width = inv_section_width - 2 * padding
    inv_grid_height = menu_height - 2 * padding - 25

    cols = 5
    rows = 2
    slot_width = inv_grid_width // cols
    slot_height = inv_grid_height // rows
    slot_size = min(slot_width, slot_height) - 6

    grid_start_x = menu_x + padding
    grid_start_y = menu_y + padding + 10

    for slot_idx in range(NUM_INVENTORY_SLOTS):
        row = slot_idx // cols
        col = slot_idx % cols

        slot_center_x = grid_start_x + col * slot_width + slot_width // 2
        slot_center_y = grid_start_y + row * slot_height + slot_height // 2

        x_start = slot_center_x - slot_size // 2
        y_start = slot_center_y - slot_size // 2

        is_selected = (slot_idx == selected_slot) and (menu_focus == "inventory")
        slot_bg = (100, 100, 100, 255) if is_selected else (60, 60, 60, 255)

        overlay[y_start : y_start + slot_size, x_start : x_start + slot_size] = slot_bg

        if is_selected:
            white = (255, 255, 255, 255)
            overlay[y_start, x_start : x_start + slot_size] = white
            overlay[y_start + slot_size - 1, x_start : x_start + slot_size] = white
            overlay[y_start : y_start + slot_size, x_start] = white
            overlay[y_start : y_start + slot_size, x_start + slot_size - 1] = white

        item_type = int(inventory_items[slot_idx])
        count = int(inventory_counts[slot_idx])

        if item_type != 0 and count > 0:
            item_pad = 3
            rgb = ITEM_COLORS.get(item_type, (128, 128, 128))
            overlay[
                y_start + item_pad : y_start + slot_size - item_pad,
                x_start + item_pad : x_start + slot_size - item_pad,
            ] = (*rgb, 255)

    craft_x = menu_x + inv_section_width + padding
    craft_y = menu_y + padding + 10
    recipe_height = 35

    for recipe_idx in range(NUM_RECIPES):
        recipe = RECIPES[recipe_idx]
        recipe_name = RECIPE_NAMES[recipe_idx]
        is_selected = (recipe_idx == selected_recipe) and (menu_focus == "crafting")
        affordable = bool(can_afford_recipe(state, selected_player, recipe_idx))

        recipe_y = craft_y + recipe_idx * recipe_height

        if is_selected:
            overlay[
                recipe_y : recipe_y + recipe_height - 2,
                craft_x : craft_x + craft_section_width - 2 * padding,
            ] = (80, 80, 80, 255)
            white = (255, 255, 255, 255)
            overlay[recipe_y, craft_x : craft_x + craft_section_width - 2 * padding] = white
            overlay[recipe_y + recipe_height - 3, craft_x : craft_x + craft_section_width - 2 * padding] = white
            overlay[recipe_y : recipe_y + recipe_height - 2, craft_x] = white
            overlay[recipe_y : recipe_y + recipe_height - 2, craft_x + craft_section_width - 2 * padding - 1] = white

        output_item = recipe["output"]
        rgb = ITEM_COLORS.get(output_item, (128, 128, 128))
        icon_size = 16
        icon_y = recipe_y + 4
        icon_x = craft_x + 4
        overlay[icon_y : icon_y + icon_size, icon_x : icon_x + icon_size] = (*rgb, 255)

        inputs = recipe["inputs"]
        input_x = icon_x + icon_size + 8
        for item_type, required in inputs:
            input_rgb = ITEM_COLORS.get(item_type, (128, 128, 128))
            have = int(count_item_in_inventory(state, selected_player, item_type))

            if have >= required:
                input_rgb = tuple(min(255, c + 30) for c in input_rgb)
            else:
                input_rgb = tuple(c // 2 for c in input_rgb)

            input_icon_size = 10
            input_y = recipe_y + 8
            overlay[input_y : input_y + input_icon_size, input_x : input_x + input_icon_size] = (*input_rgb, 255)
            input_x += input_icon_size + 4

        if craft_progress > 0 and recipe_idx == selected_recipe:
            progress_y = recipe_y + recipe_height - 8
            progress_width = craft_section_width - 2 * padding - 10
            progress_filled = int(progress_width * (1 - craft_progress / recipe["ticks"]))
            overlay[progress_y : progress_y + 4, craft_x + 4 : craft_x + 4 + progress_width] = (30, 30, 30, 255)
            if progress_filled > 0:
                overlay[progress_y : progress_y + 4, craft_x + 4 : craft_x + 4 + progress_filled] = (100, 200, 100, 255)

    return overlay


MACHINE_TO_ITEM = {
    MachineType.MINER: ItemType.MINER,
}


def render_machine_overlays(
    image: np.ndarray,
    state: EnvState,
    block_pixel_size: int,
) -> None:
    """Draw machine overlays on tiles that have machines.

    Machines are rendered as smaller squares centered on the tile, using the
    same color as the corresponding item in the inventory. This ensures visual
    consistency between placed machines and inventory items.

    Args:
        image: RGBA image to draw on (modified in place)
        state: Current environment state
        block_pixel_size: Size of each block in pixels
    """
    machine_size = int(block_pixel_size * 0.6)
    offset = (block_pixel_size - machine_size) // 2

    machine_types = np.array(state.machine_types)
    map_height, map_width = machine_types.shape

    for y in range(map_height):
        for x in range(map_width):
            machine_type = int(machine_types[y, x])
            if machine_type != MachineType.NONE:
                item_type = MACHINE_TO_ITEM.get(machine_type, ItemType.EMPTY)
                rgb = ITEM_COLORS.get(item_type, (128, 128, 128))
                color = np.array([*rgb, 255], dtype=np.uint8)

                y_start = y * block_pixel_size + offset
                x_start = x * block_pixel_size + offset
                image[
                    y_start : y_start + machine_size,
                    x_start : x_start + machine_size,
                ] = color


def render_pixels(
    state: EnvState, block_pixel_size: int = BLOCK_PIXEL_SIZE
) -> np.ndarray:
    """Render the environment state as an RGB pixel image.

    Renders the map, machines, and all players. The selected player has
    a white highlight ring. Does not include the inventory menu.

    Args:
        state: Current environment state
        block_pixel_size: Size of each block in pixels

    Returns:
        RGB numpy array of the rendered scene
    """
    textures = get_textures()

    map_array = np.array(state.map)
    map_height, map_width = map_array.shape
    img_height = map_height * block_pixel_size
    img_width = map_width * block_pixel_size
    image = np.zeros((img_height, img_width, 4), dtype=np.uint8)

    for y in range(map_height):
        for x in range(map_width):
            block_type = int(map_array[y, x])
            if block_type in textures:
                texture = textures[block_type]
            else:
                texture = textures[int(BlockType.DIRT)]

            y_start = y * block_pixel_size
            x_start = x * block_pixel_size
            image[
                y_start : y_start + block_pixel_size,
                x_start : x_start + block_pixel_size,
            ] = texture

    render_machine_overlays(image, state, block_pixel_size)

    player_positions = np.array(state.player_positions)
    player_directions = np.array(state.player_directions)
    selected = int(state.selected_player)
    num_players = player_positions.shape[0]

    for player_idx in range(num_players):
        direction = int(player_directions[player_idx])
        is_selected = player_idx == selected
        player_texture = create_player_texture(direction, player_idx, is_selected)

        px, py = int(player_positions[player_idx, 0]), int(player_positions[player_idx, 1])
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
