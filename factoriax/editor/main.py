"""Main event loop for the FactoriaX level editor.

Handles window creation, event dispatch, tool state, and rendering
composition.  Mirrors the structure of ``factoriax.play.main`` but
operates on mutable :class:`~factoriax.editor.state.EditorState`
arrays instead of JAX state.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pygame

from factoriax.constants import (
    BLOCK_MAX_RESOURCES,
    BlockType,
    Direction,
    ItemType,
    MachineType,
)
from factoriax.editor.canvas import (
    Viewport,
    clamp_camera,
    pan,
    render_canvas,
    screen_to_tile,
    zoom,
)
from factoriax.editor.dialogs import (
    FileDialog,
    MachineInspectorDialog,
    NewLevelDialog,
    NumberInputDialog,
    render_help_overlay,
)
from factoriax.editor.inventory_panel import render_inventory_panel
from factoriax.editor.state import (
    EditorState,
    InvTarget,
    ResourceBrush,
    add_biter,
    add_column,
    add_row,
    clear_inventory_slot,
    editor_state_from_level,
    editor_state_to_level,
    erase_block,
    erase_entity,
    erase_machine,
    erase_tile,
    fill_rect_tiles,
    get_inventory_slots,
    get_num_slots,
    new_editor_state,
    remove_column,
    remove_row,
    set_inventory_slot,
    set_machine,
    set_player_position,
    set_tile,
)
from factoriax.editor.toolbar import (
    BLOCK_ITEMS,
    MACHINE_ITEMS,
    MAX_EDITOR_PLAYERS,
    MENU_BAR_HEIGHT,
    STATUS_BAR_HEIGHT,
    TOOL_ERASE,
    TOOL_FILL,
    TOOL_PAINT,
    TOOLBAR_WIDTH,
    get_palette_items_for_machine_slot,
    get_palette_items_for_player,
    render_item_palette,
    render_menu_bar,
    render_status_bar,
    render_toolbar,
)
from factoriax.levels import load_level, save_level
from factoriax.play.main import play_level
from factoriax.ui.compositing import composite_rgba_over_rgb
from factoriax.ui.primitives import ClickRegion, hit_test_regions
from factoriax.ui.window import calculate_window_size

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PAN_SPEED = 1.0
_RES_STEP = 10
_MIN_RES = 0

_DIR_CYCLE = [
    int(Direction.DOWN),
    int(Direction.RIGHT),
    int(Direction.UP),
    int(Direction.LEFT),
]

_CONVEYOR = int(MachineType.CONVEYOR_BELT)

_TOOL_LIST = [TOOL_PAINT, TOOL_FILL, TOOL_ERASE]

_BLOCK_KEYS = {
    pygame.K_1: 0,
    pygame.K_2: 1,
    pygame.K_3: 2,
    pygame.K_4: 3,
    pygame.K_5: 4,
}

_MACHINE_KEYS = {
    pygame.K_6: 0,
    pygame.K_7: 1,
    pygame.K_8: 2,
    pygame.K_9: 3,
    pygame.K_0: 4,
}


# ---------------------------------------------------------------------------
# Tool state dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ToolState:
    """All mutable tool / brush / interaction state for the editor.

    Grouping these in one object keeps the main loop's local namespace
    small and makes it easy to pass the full context to helper functions.

    Attributes:
        tool: Active tool name.
        block: Active ``BlockType`` value.
        machine: Active ``MachineType`` value, or 0 for block mode.
        direction: Global machine placement direction.
        entity: Active entity selection as ``(kind, index)`` or ``None``.
        brush: Resource brush settings.
        show_resources: Whether the resource overlay is visible.
        show_help: Whether the help overlay is visible.
        cursor_tile: Tile coordinate under the cursor, or ``None``.
        painting: ``True`` while left-click drag is active.
        last_paint_tile: Previous tile during a paint drag.
        right_erasing: ``True`` while a right-click erase drag is active.
        tool_before_erase: Tool that was active before right-click erase.
        machine_before_erase: Machine that was selected before right-click erase.
        entity_before_erase: Entity that was selected before right-click erase.
        fill_start: Start tile of a fill-rect drag, or ``None``.
        fill_rect: Current fill-rect selection, or ``None``.
        middle_dragging: ``True`` while middle-button panning is active.
        middle_last: Last raw mouse position during middle-drag.
    """

    tool: str = TOOL_PAINT
    block: int = dataclasses.field(default_factory=lambda: int(BlockType.DIRT))
    machine: int = 0
    direction: int = dataclasses.field(
        default_factory=lambda: int(Direction.DOWN),
    )
    entity: tuple[str, int] | None = None
    brush: ResourceBrush = dataclasses.field(default_factory=ResourceBrush)
    show_resources: bool = False
    show_help: bool = False
    cursor_tile: tuple[int, int] | None = None
    painting: bool = False
    last_paint_tile: tuple[int, int] | None = None
    right_erasing: bool = False
    tool_before_erase: str = TOOL_PAINT
    machine_before_erase: int = 0
    entity_before_erase: tuple[str, int] | None = None
    fill_start: tuple[int, int] | None = None
    fill_rect: tuple[int, int, int, int] | None = None
    middle_dragging: bool = False
    middle_last: tuple[int, int] = (0, 0)
    toolbar_scroll: int = 0
    inventory_mode: bool = False
    inv_target: InvTarget | None = None
    inv_focused_slot: int = 0
    inv_dragging: bool = False
    inv_drag_slot: int = -1

    @property
    def layer(self) -> str:
        """Return the active editing layer name.

        Returns:
            ``"inventory"``, ``"entity"``, ``"machine"``, or
            ``"terrain"``.
        """
        if self.inventory_mode:
            return "inventory"
        if self.entity is not None:
            return "entity"
        return "machine" if self.machine != 0 else "terrain"

    @property
    def brush_name(self) -> str:
        """Return the display name of the active brush."""
        if self.inventory_mode:
            if self.inv_target is not None:
                kind = self.inv_target[0]
                if kind == "player":
                    return f"Player {self.inv_target[1]}"
                return f"Machine ({self.inv_target[1]},{self.inv_target[2]})"
            return "Select target"
        if self.tool == TOOL_ERASE:
            return "Eraser"
        if self.entity is not None:
            kind, idx = self.entity
            if kind == "player":
                return f"Player {idx}"
            return "Biter"
        if self.machine != 0:
            for mid, name in MACHINE_ITEMS:
                if mid == self.machine:
                    return name
        for bid, name in BLOCK_ITEMS:
            if bid == self.block:
                return name
        return "?"

    @property
    def resource_info(self) -> str:
        """Return a status-bar summary of the resource brush."""
        if self.brush.mode == "exact":
            return f"Res:{self.brush.exact_value}"
        return f"Res:{self.brush.range_min}-{self.brush.range_max}"

    def stop_painting(self) -> None:
        """End any active paint or erase drag."""
        self.painting = False
        self.last_paint_tile = None

    def cancel_fill(self) -> None:
        """Cancel an in-progress fill-rect selection."""
        self.fill_start = None
        self.fill_rect = None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _get_palette_items(ts: ToolState, editor: EditorState) -> list[tuple[int, str]]:
    """Return the item palette list appropriate for the current target.

    For player targets every non-empty item is offered. For machine
    targets the list is filtered by slot role. When no target is
    selected an empty list is returned.

    Args:
        ts: Tool state (read for target and focused slot).
        editor: Editor state (read for machine types).

    Returns:
        ``(ItemType_int, display_name)`` pairs for the palette.
    """
    target = ts.inv_target
    if target is None:
        return []
    if target[0] == "player":
        return get_palette_items_for_player()
    mt = int(editor.machine_types[target[2], target[1]])
    return get_palette_items_for_machine_slot(mt, ts.inv_focused_slot)


def _next_direction(current: int) -> int:
    """Cycle to the next direction in clockwise order.

    Args:
        current: Current direction as an ``Action`` integer.

    Returns:
        Next direction value.
    """
    idx = _DIR_CYCLE.index(current) if current in _DIR_CYCLE else 0
    return _DIR_CYCLE[(idx + 1) % 4]


def _direction_from_delta(dx: int, dy: int) -> int | None:
    """Infer a cardinal direction from a tile-space delta.

    Returns ``None`` when the delta is zero (no movement).

    Args:
        dx: Horizontal tile offset (positive = right).
        dy: Vertical tile offset (positive = down).

    Returns:
        ``Action`` direction integer, or ``None``.
    """
    if dx == 0 and dy == 0:
        return None
    if abs(dx) >= abs(dy):
        return int(Direction.RIGHT) if dx > 0 else int(Direction.LEFT)
    return int(Direction.DOWN) if dy > 0 else int(Direction.UP)


def _recalc_layout(
    vp: Viewport,
    window_w: int,
    window_h: int,
) -> tuple[int, int, int]:
    """Derive base dimensions and integer scale from the viewport.

    Args:
        vp: Current viewport.
        window_w: Pygame window width.
        window_h: Pygame window height.

    Returns:
        ``(base_width, base_height, scale)`` tuple.
    """
    bw = TOOLBAR_WIDTH + vp.canvas_w
    bh = MENU_BAR_HEIGHT + vp.canvas_h + STATUS_BAR_HEIGHT
    s = max(1, min(window_w // bw, window_h // bh))
    return bw, bh, s


def _update_viewport(
    editor: EditorState,
    vp: Viewport,
    reset_camera: bool = False,
) -> None:
    """Recalculate tile size so the map fits inside the fixed canvas area.

    The canvas dimensions stay constant (set once at startup from the
    window size). Only ``tile_size`` changes so that the full map is
    visible without scrolling. The zoom can still be adjusted manually
    afterwards.

    Args:
        editor: Current editor state (read-only).
        vp: Viewport to update in place.
        reset_camera: If ``True`` the camera is moved to (0, 0).
    """
    # Pick the largest tile size from the allowed set that fits the map
    # inside the current canvas.
    allowed = [48, 32, 24, 16]
    for ts in allowed:
        if (
            editor.map_width * ts <= vp.canvas_w
            and editor.map_height * ts <= vp.canvas_h
        ):
            vp.tile_size = ts
            break
    else:
        vp.tile_size = allowed[-1]

    if reset_camera:
        vp.camera_x = 0.0
        vp.camera_y = 0.0
    clamp_camera(vp, editor.map_width, editor.map_height)


def run_play_session(state: EditorState, screen: pygame.Surface) -> None:
    """Launch a full play-test session from the editor.

    Converts the editor state to a Level and delegates to
    :func:`~factoriax.play.main.play_level`, which provides the
    complete game UI.  Returns to the editor when the user quits.

    Args:
        state: Current editor state.
        screen: Pygame display surface (reused by the play session).
    """
    level = editor_state_to_level(state)
    num_players = len(level.player_positions) if level.player_positions else 1
    play_level(level, num_players=num_players, screen=screen)
    pygame.display.set_caption("FactoriaX Editor")


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------


def _handle_motion(
    event: pygame.event.Event,
    ts: ToolState,
    editor: EditorState,
    vp: Viewport,
    scale: int,
    rng: np.random.Generator,
) -> None:
    """Process a MOUSEMOTION event: update cursor, paint, pan.

    Args:
        event: The pygame MOUSEMOTION event.
        ts: Tool state (mutated in place).
        editor: Editor state (mutated in place during painting).
        vp: Viewport (mutated in place during panning).
        scale: Current integer display scale.
        rng: Numpy RNG for resource brush sampling.
    """
    mx = event.pos[0] // scale
    my = event.pos[1] // scale
    cx = mx - TOOLBAR_WIDTH
    cy = my - MENU_BAR_HEIGHT

    if ts.inventory_mode:
        half_w = vp.canvas_w // 2
        if cx >= 0 and cx < half_w and cy >= 0 and cy < vp.canvas_h:
            saved_w = vp.canvas_w
            vp.canvas_w = half_w
            ts.cursor_tile = screen_to_tile(vp, cx, cy)
            vp.canvas_w = saved_w
        else:
            ts.cursor_tile = None
    elif cx >= 0 and cy >= 0 and cy < vp.canvas_h:
        ts.cursor_tile = screen_to_tile(vp, cx, cy)
    else:
        ts.cursor_tile = None

    if not ts.inventory_mode and ts.painting and ts.cursor_tile is not None:
        tx, ty = ts.cursor_tile
        if (tx, ty) != ts.last_paint_tile:
            if ts.tool == TOOL_ERASE:
                if ts.right_erasing:
                    if ts.entity_before_erase is not None:
                        erase_entity(editor, tx, ty)
                    elif ts.machine_before_erase != 0:
                        erase_machine(editor, tx, ty)
                    else:
                        erase_block(editor, tx, ty)
                else:
                    erase_tile(editor, tx, ty)
                    erase_entity(editor, tx, ty)
            elif ts.entity is not None:
                kind, idx = ts.entity
                if kind == "player":
                    set_player_position(editor, idx, tx, ty)
                else:
                    add_biter(editor, tx, ty)
            elif ts.machine != 0:
                direction = ts.direction
                if ts.machine == _CONVEYOR and ts.last_paint_tile is not None:
                    drag_dir = _direction_from_delta(
                        tx - ts.last_paint_tile[0],
                        ty - ts.last_paint_tile[1],
                    )
                    if drag_dir is not None:
                        direction = drag_dir
                set_machine(editor, tx, ty, ts.machine, direction)
                ts.last_paint_tile = (tx, ty)
            else:
                set_tile(editor, tx, ty, ts.block, ts.brush, rng)

    if ts.fill_start is not None and ts.cursor_tile is not None:
        ts.fill_rect = (*ts.fill_start, *ts.cursor_tile)

    if ts.middle_dragging:
        dx = (ts.middle_last[0] - event.pos[0]) / scale / vp.tile_size
        dy = (ts.middle_last[1] - event.pos[1]) / scale / vp.tile_size
        pan(vp, dx, dy, editor.map_width, editor.map_height)
        ts.middle_last = event.pos


def _handle_toolbar_click(
    hit: ClickRegion,
    ts: ToolState,
) -> NumberInputDialog | None:
    """Process a toolbar click region hit.

    Args:
        hit: The matched click region.
        ts: Tool state (mutated in place).

    Returns:
        A :class:`NumberInputDialog` if one should be opened, else ``None``.
    """
    if hit.action == "tool":
        ts.tool = _TOOL_LIST[hit.param]
    elif hit.action == "block":
        ts.block = BLOCK_ITEMS[hit.param][0]
        ts.machine = 0
        ts.entity = None
    elif hit.action == "machine":
        ts.machine = MACHINE_ITEMS[hit.param][0]
        ts.entity = None
    elif hit.action == "entity":
        if hit.param < MAX_EDITOR_PLAYERS:
            ts.entity = ("player", hit.param)
        else:
            ts.entity = ("biter", 0)
        ts.machine = 0
    elif hit.action == "toggle_res_mode":
        ts.brush.mode = "range" if ts.brush.mode == "exact" else "exact"
    elif hit.action == "edit_res_exact":
        return NumberInputDialog(
            label=f"Amount (0-{BLOCK_MAX_RESOURCES}):",
            text="",
            max_value=BLOCK_MAX_RESOURCES,
            default=ts.brush.exact_value,
        )
    elif hit.action == "edit_res_min":
        return NumberInputDialog(
            label=f"Min (0-{BLOCK_MAX_RESOURCES}):",
            text="",
            max_value=BLOCK_MAX_RESOURCES,
            default=ts.brush.range_min,
        )
    elif hit.action == "edit_res_max":
        return NumberInputDialog(
            label=f"Max (0-{BLOCK_MAX_RESOURCES}):",
            text="",
            max_value=BLOCK_MAX_RESOURCES,
            default=ts.brush.range_max,
        )
    elif hit.action == "toggle_show_res":
        ts.show_resources = not ts.show_resources
    return None


def _handle_inv_canvas_click(
    tx: int,
    ty: int,
    ts: ToolState,
    editor: EditorState,
) -> None:
    """Select a player or machine on the canvas during inventory mode.

    If the clicked tile contains a player start, the player becomes
    the inventory target. If it contains a machine, the machine
    becomes the target. Otherwise the target is cleared.

    Args:
        tx: Tile x coordinate.
        ty: Tile y coordinate.
        ts: Tool state (mutated in place).
        editor: Editor state (read only).
    """
    if not (0 <= tx < editor.map_width and 0 <= ty < editor.map_height):
        ts.inv_target = None
        return
    for idx, (px, py) in editor.player_positions.items():
        if px == tx and py == ty:
            ts.inv_target = ("player", idx, 0)
            ts.inv_focused_slot = 0
            return
    mt = int(editor.machine_types[ty, tx])
    if mt != int(MachineType.NONE):
        ts.inv_target = ("machine", tx, ty)
        ts.inv_focused_slot = 0
        return
    ts.inv_target = None


def _handle_inv_panel_click(
    hit: ClickRegion,
    ts: ToolState,
    editor: EditorState,
    shift: bool,
) -> None:
    """Handle a click on a slot inside the inventory panel.

    Focusing the clicked slot is the default. Shift-click clears
    the slot instead.

    Args:
        hit: Matched click region (action ``"inv_slot"``).
        ts: Tool state (mutated in place).
        editor: Editor state (mutated in place on shift-clear).
        shift: Whether the shift modifier is held.
    """
    if hit.action != "inv_slot" or ts.inv_target is None:
        return
    slot = hit.param
    if shift:
        clear_inventory_slot(editor, ts.inv_target, slot)
    else:
        ts.inv_focused_slot = slot


def _handle_inv_palette_click(
    hit: ClickRegion,
    ts: ToolState,
    editor: EditorState,
) -> None:
    """Handle a click on an item in the sidebar palette.

    Places the item in the focused slot with count 1, or increments
    the count if the slot already contains the same item type.

    Args:
        hit: Matched click region (action ``"inv_item"``).
        ts: Tool state (read for target and focused slot).
        editor: Editor state (mutated in place).
    """
    if hit.action != "inv_item" or ts.inv_target is None:
        return
    item_type = hit.param
    slots = get_inventory_slots(editor, ts.inv_target)
    slot = ts.inv_focused_slot
    current_item, current_count = slots[slot]
    if current_item == item_type:
        new_count = min(current_count + 1, 64)
        set_inventory_slot(editor, ts.inv_target, slot, item_type, new_count)
    else:
        set_inventory_slot(editor, ts.inv_target, slot, item_type, 1)


def _handle_canvas_click(
    tx: int,
    ty: int,
    ts: ToolState,
    editor: EditorState,
    rng: np.random.Generator,
) -> None:
    """Process a left-click on the canvas.

    Args:
        tx: Tile x coordinate.
        ty: Tile y coordinate.
        ts: Tool state (mutated in place).
        editor: Editor state (mutated in place).
        rng: Numpy RNG for resource brush sampling.
    """
    if ts.tool == TOOL_FILL:
        ts.fill_start = (tx, ty)
        ts.fill_rect = (tx, ty, tx, ty)
    else:
        ts.painting = True
        ts.last_paint_tile = (tx, ty)
        if ts.tool == TOOL_ERASE:
            erase_tile(editor, tx, ty)
            erase_entity(editor, tx, ty)
        elif ts.entity is not None:
            kind, idx = ts.entity
            if kind == "player":
                set_player_position(editor, idx, tx, ty)
            else:
                add_biter(editor, tx, ty)
        elif ts.machine != 0:
            set_machine(editor, tx, ty, ts.machine, ts.direction)
        else:
            set_tile(editor, tx, ty, ts.block, ts.brush, rng)


def _handle_right_click(
    tx: int,
    ty: int,
    ts: ToolState,
    editor: EditorState,
) -> None:
    """Process a right-click on the canvas: layer-aware erase.

    Args:
        tx: Tile x coordinate.
        ty: Tile y coordinate.
        ts: Tool state (mutated in place).
        editor: Editor state (mutated in place).
    """
    ts.right_erasing = True
    ts.tool_before_erase = ts.tool
    ts.machine_before_erase = ts.machine
    ts.entity_before_erase = ts.entity
    ts.tool = TOOL_ERASE
    if ts.entity is not None:
        erase_entity(editor, tx, ty)
    elif ts.machine != 0:
        erase_machine(editor, tx, ty)
    else:
        erase_block(editor, tx, ty)
    ts.painting = True


def _handle_fill_release(
    ts: ToolState,
    editor: EditorState,
    rng: np.random.Generator,
) -> None:
    """Finalise a fill-rect operation on mouse-button release.

    Args:
        ts: Tool state (mutated in place).
        editor: Editor state (mutated in place).
        rng: Numpy RNG for resource brush sampling.
    """
    if ts.fill_start is None or ts.cursor_tile is None:
        return
    tx, ty = ts.cursor_tile
    if ts.entity is not None:
        kind, idx = ts.entity
        if kind == "biter":
            lx = min(ts.fill_start[0], tx)
            ly = min(ts.fill_start[1], ty)
            rx = max(ts.fill_start[0], tx)
            ry = max(ts.fill_start[1], ty)
            for fy in range(max(0, ly), min(editor.map_height, ry + 1)):
                for fx in range(max(0, lx), min(editor.map_width, rx + 1)):
                    add_biter(editor, fx, fy)
        else:
            set_player_position(editor, idx, tx, ty)
        ts.cancel_fill()
        return
    if ts.machine != 0:
        fill_dir = _direction_from_delta(
            tx - ts.fill_start[0],
            ty - ts.fill_start[1],
        )
        if ts.machine == _CONVEYOR and fill_dir is not None:
            direction = fill_dir
        else:
            direction = ts.direction
        lx = min(ts.fill_start[0], tx)
        ly = min(ts.fill_start[1], ty)
        rx = max(ts.fill_start[0], tx)
        ry = max(ts.fill_start[1], ty)
        for fy in range(max(0, ly), min(editor.map_height, ry + 1)):
            for fx in range(max(0, lx), min(editor.map_width, rx + 1)):
                set_machine(editor, fx, fy, ts.machine, direction)
    else:
        fill_rect_tiles(
            editor,
            ts.fill_start[0],
            ts.fill_start[1],
            tx,
            ty,
            ts.block,
            ts.brush,
            rng,
        )
    ts.cancel_fill()


def _handle_inv_keydown(
    key: int,
    ts: ToolState,
    editor: EditorState,
) -> None:
    """Handle keyboard input while inventory mode has a target selected.

    Arrow keys navigate the focused slot. Delete/Backspace clears
    the focused slot. Plus/Minus adjust the stack count.

    Args:
        key: Pygame key constant.
        ts: Tool state (mutated in place).
        editor: Editor state (mutated in place for count changes).
    """
    target = ts.inv_target
    if target is None:
        return
    num_slots = get_num_slots(editor, target)
    if num_slots == 0:
        return

    if key in (pygame.K_RIGHT, pygame.K_DOWN):
        ts.inv_focused_slot = min(ts.inv_focused_slot + 1, num_slots - 1)
    elif key in (pygame.K_LEFT, pygame.K_UP):
        ts.inv_focused_slot = max(ts.inv_focused_slot - 1, 0)
    elif key in (pygame.K_DELETE, pygame.K_BACKSPACE):
        clear_inventory_slot(editor, target, ts.inv_focused_slot)
    elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
        slots = get_inventory_slots(editor, target)
        item_type, count = slots[ts.inv_focused_slot]
        if item_type != int(ItemType.EMPTY) and count < 64:
            set_inventory_slot(
                editor, target, ts.inv_focused_slot, item_type, count + 1
            )
    elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
        slots = get_inventory_slots(editor, target)
        item_type, count = slots[ts.inv_focused_slot]
        if item_type != int(ItemType.EMPTY) and count > 1:
            set_inventory_slot(
                editor, target, ts.inv_focused_slot, item_type, count - 1
            )
        elif count <= 1:
            clear_inventory_slot(editor, target, ts.inv_focused_slot)


def _handle_keydown(
    event: pygame.event.Event,
    ts: ToolState,
    editor: EditorState,
    vp: Viewport,
    screen: pygame.Surface,
    window_w: int,
    window_h: int,
) -> tuple[
    EditorState,
    int,
    int,
    int,
    bool,
    NewLevelDialog | FileDialog | MachineInspectorDialog | None,
]:
    """Process a KEYDOWN event.

    Returns the potentially-replaced editor, updated layout values,
    the running flag, and an optional dialog (new-level, file, or
    machine inspector).

    Args:
        event: Pygame KEYDOWN event.
        ts: Tool state (mutated in place).
        editor: Current editor state.
        vp: Viewport (mutated in place for pan/resize).
        screen: Pygame display surface.
        window_w: Current window width.
        window_h: Current window height.

    Returns:
        ``(editor, base_w, base_h, scale, running, dialog)`` tuple.
    """
    mods = pygame.key.get_mods()
    ctrl = bool(mods & pygame.KMOD_CTRL)
    shift = bool(mods & pygame.KMOD_SHIFT)
    key = event.key

    base_w, base_h, scale = _recalc_layout(vp, window_w, window_h)
    running = True
    dialog: NewLevelDialog | FileDialog | None = None

    if key == pygame.K_ESCAPE:
        if ts.fill_start is not None:
            ts.cancel_fill()
        elif ts.inventory_mode:
            if ts.inv_target is not None:
                ts.inv_target = None
                ts.inv_focused_slot = 0
            else:
                ts.inventory_mode = False
        else:
            running = False

    elif ctrl and key == pygame.K_n:
        dialog = NewLevelDialog()
    elif ctrl and key == pygame.K_o:
        dialog = FileDialog(mode="load")
    elif ctrl and key == pygame.K_s:
        dialog = FileDialog(mode="save", filename_text=editor.name)
    elif key == pygame.K_F5:
        run_play_session(editor, screen)

    elif key == pygame.K_b:
        ts.tool = TOOL_PAINT
    elif key == pygame.K_f:
        ts.tool = TOOL_FILL
    elif key == pygame.K_x:
        ts.tool = TOOL_ERASE

    elif key in _BLOCK_KEYS:
        ts.block = BLOCK_ITEMS[_BLOCK_KEYS[key]][0]
        ts.machine = 0
        ts.entity = None
    elif key in _MACHINE_KEYS:
        ts.machine = MACHINE_ITEMS[_MACHINE_KEYS[key]][0]
        ts.entity = None

    elif key == pygame.K_r:
        _handle_rotate(ts, editor)

    elif key == pygame.K_i:
        insp = _open_inspector(ts, editor)
        if insp is not None:
            return editor, base_w, base_h, scale, running, insp

    elif key == pygame.K_t:
        ts.brush.mode = "range" if ts.brush.mode == "exact" else "exact"
    elif key == pygame.K_v:
        ts.inventory_mode = not ts.inventory_mode
        if not ts.inventory_mode:
            ts.inv_target = None
            ts.inv_focused_slot = 0
    elif key == pygame.K_QUESTION or (key == pygame.K_SLASH and shift):
        ts.show_help = True

    elif ts.inventory_mode and ts.inv_target is not None and not ctrl:
        _handle_inv_keydown(key, ts, editor)

    elif key == pygame.K_RIGHTBRACKET:
        _adjust_resource(ts.brush, +_RES_STEP, shift)
    elif key == pygame.K_LEFTBRACKET:
        _adjust_resource(ts.brush, -_RES_STEP, shift)

    elif ctrl and key in (pygame.K_RIGHT, pygame.K_LEFT, pygame.K_DOWN, pygame.K_UP):
        _handle_resize(key, editor, vp)
        base_w, base_h, scale = _recalc_layout(vp, window_w, window_h)

    elif key == pygame.K_LEFT:
        pan(vp, -_PAN_SPEED, 0, editor.map_width, editor.map_height)
    elif key == pygame.K_RIGHT:
        pan(vp, _PAN_SPEED, 0, editor.map_width, editor.map_height)
    elif key == pygame.K_UP:
        pan(vp, 0, -_PAN_SPEED, 0, editor.map_height)
    elif key == pygame.K_DOWN:
        pan(vp, 0, _PAN_SPEED, editor.map_width, editor.map_height)

    return editor, base_w, base_h, scale, running, dialog


def _handle_rotate(ts: ToolState, editor: EditorState) -> None:
    """Rotate a machine under the cursor, or the global direction.

    Args:
        ts: Tool state (mutated in place).
        editor: Editor state (mutated in place if rotating in-place).
    """
    ct = ts.cursor_tile
    if (
        ct is not None
        and 0 <= ct[0] < editor.map_width
        and 0 <= ct[1] < editor.map_height
        and editor.machine_types[ct[1], ct[0]] != int(MachineType.NONE)
    ):
        r, c = ct[1], ct[0]
        editor.machine_directions[r, c] = _next_direction(
            int(editor.machine_directions[r, c]),
        )
        editor.dirty = True
    else:
        ts.direction = _next_direction(ts.direction)


def _open_inspector(
    ts: ToolState, editor: EditorState
) -> MachineInspectorDialog | None:
    """Try to open a machine inspector at the cursor tile.

    Args:
        ts: Tool state (read for cursor position).
        editor: Editor state (read for machine type).

    Returns:
        A :class:`MachineInspectorDialog` if a machine is under the
        cursor, else ``None``.
    """
    ct = ts.cursor_tile
    if ct is None:
        return None
    x, y = ct
    if not (0 <= x < editor.map_width and 0 <= y < editor.map_height):
        return None
    if editor.machine_types[y, x] == int(MachineType.NONE):
        return None
    return MachineInspectorDialog(
        tile_x=x,
        tile_y=y,
        machine_type=int(editor.machine_types[y, x]),
        inv_items=editor.machine_inventory_items[y, x],
        inv_counts=editor.machine_inventory_counts[y, x],
        selected_recipe=editor.machine_selected_recipe,
        recipe_row=y,
        recipe_col=x,
    )


def _adjust_resource(brush: ResourceBrush, delta: int, shift: bool) -> None:
    """Adjust the resource brush by *delta*, respecting mode.

    Args:
        brush: Resource brush (mutated in place).
        delta: Positive to increase, negative to decrease.
        shift: When ``True`` in range mode, adjusts ``range_min``.
    """
    if shift and brush.mode == "range":
        brush.range_min = max(
            _MIN_RES,
            min(brush.range_min + delta, brush.range_max),
        )
    elif brush.mode == "exact":
        brush.exact_value = max(
            _MIN_RES,
            min(BLOCK_MAX_RESOURCES, brush.exact_value + delta),
        )
    else:
        brush.range_max = max(
            brush.range_min,
            min(BLOCK_MAX_RESOURCES, brush.range_max + delta),
        )


def _handle_resize(
    key: int,
    editor: EditorState,
    vp: Viewport,
) -> None:
    """Add or remove a row/column based on the arrow key pressed.

    Args:
        key: One of ``K_RIGHT``, ``K_LEFT``, ``K_DOWN``, ``K_UP``.
        editor: Editor state (mutated in place).
        vp: Viewport (updated via :func:`_update_viewport`).
    """
    if key == pygame.K_RIGHT:
        add_column(editor)
    elif key == pygame.K_LEFT:
        remove_column(editor)
    elif key == pygame.K_DOWN:
        add_row(editor)
    elif key == pygame.K_UP:
        remove_row(editor)
    _update_viewport(editor, vp)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _validate_inv_target(ts: ToolState, editor: EditorState) -> None:
    """Clear ``inv_target`` if it refers to a deleted player or machine.

    Args:
        ts: Tool state (mutated in place).
        editor: Editor state (read only).
    """
    target = ts.inv_target
    if target is None:
        return
    if target[0] == "player":
        if target[1] not in editor.player_positions:
            ts.inv_target = None
            ts.inv_focused_slot = 0
    else:
        tx, ty = target[1], target[2]
        if (
            tx < 0
            or ty < 0
            or tx >= editor.map_width
            or ty >= editor.map_height
            or int(editor.machine_types[ty, tx]) == int(MachineType.NONE)
        ):
            ts.inv_target = None
            ts.inv_focused_slot = 0


def _blit_canvas_rgba(
    frame: np.ndarray,
    canvas_img: np.ndarray,
    frame_y: int,
    frame_x: int,
    h: int,
    w: int,
) -> None:
    """Alpha-composite an RGBA canvas image onto an RGB frame region.

    Args:
        frame: Destination RGB array (mutated in place).
        canvas_img: Source RGBA canvas image.
        frame_y: Top row in the frame.
        frame_x: Left column in the frame.
        h: Maximum height to composite.
        w: Maximum width to composite.
    """
    canvas_rgb = canvas_img[:, :, :3]
    alpha = canvas_img[:, :, 3:4].astype(np.float32) / 255.0
    bg = frame[frame_y : frame_y + h, frame_x : frame_x + w]
    ch = min(canvas_rgb.shape[0], bg.shape[0])
    cw = min(canvas_rgb.shape[1], bg.shape[1])
    bg[:ch, :cw] = (
        canvas_rgb[:ch, :cw].astype(np.float32) * alpha[:ch, :cw]
        + bg[:ch, :cw].astype(np.float32) * (1.0 - alpha[:ch, :cw])
    ).astype(np.uint8)


def _render_frame(
    editor: EditorState,
    vp: Viewport,
    ts: ToolState,
    base_w: int,
    base_h: int,
    file_dialog: FileDialog | None,
    dialog: NewLevelDialog | None,
    number_dialog: NumberInputDialog | None,
    inspector_dialog: MachineInspectorDialog | None = None,
) -> np.ndarray:
    """Compose the full editor frame from all UI layers.

    When inventory mode is active the canvas area is split: the left
    half shows the map, the right half shows the inventory panel for
    the selected target.  The toolbar is replaced with an item palette.

    Args:
        editor: Current editor state.
        vp: Current viewport.
        ts: Tool state.
        base_w: Base frame width in pixels.
        base_h: Base frame height in pixels.
        file_dialog: Active file save/load dialog, or ``None``.
        dialog: Active new-level dialog, or ``None``.
        number_dialog: Active number-input dialog, or ``None``.
        inspector_dialog: Active machine inspector dialog, or ``None``.

    Returns:
        RGB uint8 array of shape ``(base_h, base_w, 3)``.
    """
    if ts.inventory_mode:
        _validate_inv_target(ts, editor)

    menu_bar, _ = render_menu_bar(base_w)
    canvas_h = vp.canvas_h
    canvas_w = vp.canvas_w

    status_bar = render_status_bar(
        ts.tool,
        ts.brush_name,
        ts.cursor_tile,
        editor.name,
        editor.dirty,
        ts.resource_info,
        base_w,
        layer=ts.layer,
    )

    frame = np.full((base_h, base_w, 3), (30, 30, 30), dtype=np.uint8)
    frame[:MENU_BAR_HEIGHT, :] = menu_bar

    if ts.inventory_mode:
        # Sidebar: item palette instead of normal toolbar.
        palette_items = _get_palette_items(ts, editor)
        palette_img, _ = render_item_palette(palette_items, canvas_h, ts.toolbar_scroll)
        tb_h = min(palette_img.shape[0], canvas_h)
        frame[MENU_BAR_HEIGHT : MENU_BAR_HEIGHT + tb_h, :TOOLBAR_WIDTH] = palette_img[
            :tb_h
        ]

        # Canvas at half width.
        half_w = canvas_w // 2
        saved_w = vp.canvas_w
        vp.canvas_w = half_w
        canvas_img = render_canvas(
            editor, vp, ts.cursor_tile, ts.fill_rect, ts.show_resources
        )
        vp.canvas_w = saved_w
        _blit_canvas_rgba(
            frame, canvas_img, MENU_BAR_HEIGHT, TOOLBAR_WIDTH, canvas_h, half_w
        )

        # Inventory panel on the right half.
        if ts.inv_target is not None:
            inv_w = canvas_w - half_w
            inv_img, _ = render_inventory_panel(
                editor, ts.inv_target, ts.inv_focused_slot, inv_w, canvas_h
            )
            inv_rgb = inv_img[:, :, :3]
            inv_alpha = inv_img[:, :, 3:4].astype(np.float32) / 255.0
            dest = frame[
                MENU_BAR_HEIGHT : MENU_BAR_HEIGHT + canvas_h,
                TOOLBAR_WIDTH + half_w : TOOLBAR_WIDTH + half_w + inv_w,
            ]
            dh = min(inv_rgb.shape[0], dest.shape[0])
            dw = min(inv_rgb.shape[1], dest.shape[1])
            dest[:dh, :dw] = (
                inv_rgb[:dh, :dw].astype(np.float32) * inv_alpha[:dh, :dw]
                + dest[:dh, :dw].astype(np.float32) * (1.0 - inv_alpha[:dh, :dw])
            ).astype(np.uint8)
    else:
        # Normal rendering.
        toolbar, _ = render_toolbar(
            ts.tool,
            ts.block,
            ts.machine,
            ts.direction,
            ts.brush,
            canvas_h,
            ts.show_resources,
            selected_entity=ts.entity,
        )
        tb_visible_h = min(toolbar.shape[0], canvas_h)
        scroll = min(ts.toolbar_scroll, max(0, toolbar.shape[0] - tb_visible_h))
        ts.toolbar_scroll = scroll
        tb_slice = toolbar[scroll : scroll + tb_visible_h]
        frame[
            MENU_BAR_HEIGHT : MENU_BAR_HEIGHT + tb_slice.shape[0],
            :TOOLBAR_WIDTH,
        ] = tb_slice

        canvas_img = render_canvas(
            editor, vp, ts.cursor_tile, ts.fill_rect, ts.show_resources
        )
        _blit_canvas_rgba(
            frame,
            canvas_img,
            MENU_BAR_HEIGHT,
            TOOLBAR_WIDTH,
            canvas_h,
            canvas_w,
        )

    frame[base_h - STATUS_BAR_HEIGHT :, :] = status_bar

    if inspector_dialog is not None:
        composite_rgba_over_rgb(frame, inspector_dialog.render(base_w, base_h))
    if file_dialog is not None:
        composite_rgba_over_rgb(frame, file_dialog.render(base_w, base_h))
    if dialog is not None:
        composite_rgba_over_rgb(frame, dialog.render(base_w, base_h))
    if number_dialog is not None:
        composite_rgba_over_rgb(frame, number_dialog.render(base_w, base_h))
    if ts.show_help:
        composite_rgba_over_rgb(frame, render_help_overlay(base_w, base_h))

    return frame


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _export_benchmark_levels() -> None:
    """Write benchmark levels as JSON files into the levels/ directory.

    Called at editor startup so that benchmark levels are always available
    in the Load dialog. The levels/ directory is gitignored, so these
    files are regenerated each run and never committed.
    """
    from factoriax.benchmarks.skills.fuel_miner import fuel_miner_level
    from factoriax.benchmarks.skills.mining import mining_level
    from factoriax.benchmarks.skills.place_miner import place_miner_level
    from factoriax.editor.dialogs import LEVELS_DIR

    LEVELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, gen in [
        ("mining", mining_level),
        ("place_miner", place_miner_level),
        ("fuel_miner", fuel_miner_level),
    ]:
        path = LEVELS_DIR / f"{name}.json"
        if not path.exists():
            level, _ = gen()
            save_level(level, path)


def main(screen: pygame.Surface | None = None) -> None:
    """Run the FactoriaX level editor.

    Args:
        screen: Existing pygame display surface to reuse.  When
            ``None`` (the default) a new window is created.

    Press ``?`` for a full list of controls.
    """
    _export_benchmark_levels()
    owns_pygame = screen is None
    if owns_pygame:
        pygame.init()

    editor = new_editor_state(15, 15)
    rng = np.random.default_rng(42)
    ts = ToolState()

    vp = Viewport(
        tile_size=32,
        canvas_w=editor.map_width * 32,
        canvas_h=editor.map_height * 32,
    )
    clamp_camera(vp, editor.map_width, editor.map_height)

    base_w, base_h, scale = _recalc_layout(
        vp,
        *calculate_window_size(
            TOOLBAR_WIDTH + vp.canvas_w,
            MENU_BAR_HEIGHT + vp.canvas_h + STATUS_BAR_HEIGHT,
        ),
    )
    window_w, window_h = calculate_window_size(base_w, base_h)
    screen = pygame.display.set_mode(
        (window_w, window_h),
        pygame.RESIZABLE,
    )
    pygame.display.set_caption("FactoriaX Editor")
    clock = pygame.time.Clock()
    base_w, base_h, scale = _recalc_layout(vp, window_w, window_h)

    dialog: NewLevelDialog | None = None
    file_dialog: FileDialog | None = None
    number_dialog: NumberInputDialog | None = None
    number_dialog_target: str = ""
    inspector_dialog: MachineInspectorDialog | None = None

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                continue

            # ---- Modal dialog layers (consume all events) ----

            if inspector_dialog is not None:
                result = inspector_dialog.handle_event(event)
                if result == "close":
                    editor.dirty = True
                    inspector_dialog = None
                continue

            if file_dialog is not None:
                result = file_dialog.handle_event(event)
                if result == "ok":
                    path = file_dialog.get_path()
                    if path is not None:
                        if file_dialog.mode == "save":
                            save_level(
                                editor_state_to_level(editor),
                                path,
                            )
                            editor.dirty = False
                        else:
                            if path.exists():
                                level = load_level(path)
                                editor = editor_state_from_level(level)
                                _update_viewport(
                                    editor,
                                    vp,
                                    reset_camera=True,
                                )
                                base_w, base_h, scale = _recalc_layout(
                                    vp,
                                    window_w,
                                    window_h,
                                )
                    file_dialog = None
                elif result == "cancel":
                    file_dialog = None
                continue

            if dialog is not None:
                result = dialog.handle_event(event)
                if result == "ok":
                    w, h, name = dialog.get_values()
                    editor = new_editor_state(w, h, name)
                    _update_viewport(editor, vp, reset_camera=True)
                    base_w, base_h, scale = _recalc_layout(
                        vp,
                        window_w,
                        window_h,
                    )
                    dialog = None
                elif result == "cancel":
                    dialog = None
                continue

            if number_dialog is not None:
                result = number_dialog.handle_event(event)
                if result == "ok":
                    val = number_dialog.get_value()
                    if number_dialog_target == "exact":
                        ts.brush.exact_value = val
                    elif number_dialog_target == "min":
                        ts.brush.range_min = min(val, ts.brush.range_max)
                    elif number_dialog_target == "max":
                        ts.brush.range_max = max(val, ts.brush.range_min)
                    number_dialog = None
                elif result == "cancel":
                    number_dialog = None
                continue

            if ts.show_help:
                if event.type == pygame.KEYDOWN:
                    ts.show_help = False
                continue

            # ---- Non-modal events ----

            if event.type == pygame.VIDEORESIZE:
                window_w, window_h = event.w, event.h
                vp.canvas_w = window_w - TOOLBAR_WIDTH
                vp.canvas_h = window_h - MENU_BAR_HEIGHT - STATUS_BAR_HEIGHT
                clamp_camera(vp, editor.map_width, editor.map_height)
                base_w, base_h, scale = _recalc_layout(
                    vp,
                    window_w,
                    window_h,
                )

            elif event.type == pygame.MOUSEMOTION:
                _handle_motion(event, ts, editor, vp, scale, rng)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx = event.pos[0] // scale
                my = event.pos[1] // scale

                if event.button == 2:
                    ts.middle_dragging = True
                    ts.middle_last = event.pos
                    continue

                if event.button in (4, 5):
                    if mx < TOOLBAR_WIDTH and my >= MENU_BAR_HEIGHT:
                        step = -20 if event.button == 4 else 20
                        ts.toolbar_scroll = max(0, ts.toolbar_scroll + step)
                    else:
                        cx = mx - TOOLBAR_WIDTH
                        cy = my - MENU_BAR_HEIGHT
                        if cx >= 0 and cy >= 0:
                            zoom(
                                vp,
                                1 if event.button == 4 else -1,
                                cx,
                                cy,
                                editor.map_width,
                                editor.map_height,
                            )
                    continue

                if event.button == 1:
                    if my < MENU_BAR_HEIGHT:
                        _, regions = render_menu_bar(base_w)
                        hit = hit_test_regions(regions, mx, my)
                        if hit is not None:
                            if hit.action == "new":
                                dialog = NewLevelDialog()
                            elif hit.action == "load":
                                file_dialog = FileDialog(mode="load")
                            elif hit.action == "save":
                                file_dialog = FileDialog(
                                    mode="save",
                                    filename_text=editor.name,
                                )
                            elif hit.action == "play":
                                run_play_session(editor, screen)
                        continue

                    if mx < TOOLBAR_WIDTH:
                        if ts.inventory_mode:
                            palette_items = _get_palette_items(ts, editor)
                            _, p_regions = render_item_palette(
                                palette_items,
                                vp.canvas_h,
                                ts.toolbar_scroll,
                            )
                            adjusted = [
                                ClickRegion(
                                    r.x,
                                    r.y + MENU_BAR_HEIGHT,
                                    r.w,
                                    r.h,
                                    r.action,
                                    r.param,
                                )
                                for r in p_regions
                            ]
                            hit = hit_test_regions(adjusted, mx, my)
                            if hit is not None:
                                _handle_inv_palette_click(hit, ts, editor)
                        else:
                            _, tb_regions = render_toolbar(
                                ts.tool,
                                ts.block,
                                ts.machine,
                                ts.direction,
                                ts.brush,
                                vp.canvas_h,
                                ts.show_resources,
                                selected_entity=ts.entity,
                            )
                            adjusted = [
                                ClickRegion(
                                    r.x,
                                    r.y + MENU_BAR_HEIGHT - ts.toolbar_scroll,
                                    r.w,
                                    r.h,
                                    r.action,
                                    r.param,
                                )
                                for r in tb_regions
                            ]
                            hit = hit_test_regions(adjusted, mx, my)
                            if hit is not None:
                                nd = _handle_toolbar_click(hit, ts)
                                if nd is not None:
                                    number_dialog = nd
                                    number_dialog_target = hit.action.replace(
                                        "edit_res_", ""
                                    )
                        continue

                    cx = mx - TOOLBAR_WIDTH
                    cy = my - MENU_BAR_HEIGHT
                    if cx >= 0 and cy >= 0 and cy < vp.canvas_h:
                        if ts.inventory_mode:
                            half_w = vp.canvas_w // 2
                            if cx < half_w:
                                saved_w = vp.canvas_w
                                vp.canvas_w = half_w
                                tx, ty = screen_to_tile(vp, cx, cy)
                                vp.canvas_w = saved_w
                                _handle_inv_canvas_click(tx, ty, ts, editor)
                            elif ts.inv_target is not None:
                                inv_x = cx - half_w
                                inv_w = vp.canvas_w - half_w
                                _, inv_regions = render_inventory_panel(
                                    editor,
                                    ts.inv_target,
                                    ts.inv_focused_slot,
                                    inv_w,
                                    vp.canvas_h,
                                )
                                mods = pygame.key.get_mods()
                                shift = bool(mods & pygame.KMOD_SHIFT)
                                hit = hit_test_regions(inv_regions, inv_x, cy)
                                if hit is not None:
                                    _handle_inv_panel_click(hit, ts, editor, shift)
                        else:
                            tx, ty = screen_to_tile(vp, cx, cy)
                            _handle_canvas_click(tx, ty, ts, editor, rng)

                elif event.button == 3 and not ts.inventory_mode:
                    cx = mx - TOOLBAR_WIDTH
                    cy = my - MENU_BAR_HEIGHT
                    if cx >= 0 and cy >= 0 and cy < vp.canvas_h:
                        tx, ty = screen_to_tile(vp, cx, cy)
                        _handle_right_click(tx, ty, ts, editor)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 2:
                    ts.middle_dragging = False
                elif event.button == 3:
                    if ts.right_erasing:
                        ts.tool = ts.tool_before_erase
                        ts.entity = ts.entity_before_erase
                        ts.right_erasing = False
                        ts.stop_painting()
                elif event.button == 1:
                    _handle_fill_release(ts, editor, rng)
                    ts.stop_painting()

            elif event.type == pygame.KEYDOWN:
                (
                    editor,
                    base_w,
                    base_h,
                    scale,
                    running,
                    new_dialog,
                ) = _handle_keydown(
                    event,
                    ts,
                    editor,
                    vp,
                    screen,
                    window_w,
                    window_h,
                )
                if new_dialog is not None:
                    if isinstance(new_dialog, FileDialog):
                        file_dialog = new_dialog
                    elif isinstance(new_dialog, MachineInspectorDialog):
                        inspector_dialog = new_dialog
                    else:
                        dialog = new_dialog

        # ---- Render ----

        frame = _render_frame(
            editor,
            vp,
            ts,
            base_w,
            base_h,
            file_dialog,
            dialog,
            number_dialog,
            inspector_dialog,
        )
        base_surface = pygame.surfarray.make_surface(
            np.transpose(frame, (1, 0, 2)),
        )
        scaled = pygame.transform.scale(
            base_surface,
            (base_w * scale, base_h * scale),
        )
        screen.fill((0, 0, 0))
        screen.blit(scaled, (0, 0))
        pygame.display.flip()
        clock.tick(30)

    if owns_pygame:
        pygame.quit()
