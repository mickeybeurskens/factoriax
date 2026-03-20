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
    Action,
    BlockType,
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
from factoriax.editor.state import (
    EditorState,
    ResourceBrush,
    add_column,
    add_row,
    editor_state_from_level,
    editor_state_to_level,
    erase_block,
    erase_machine,
    erase_tile,
    fill_rect_tiles,
    new_editor_state,
    remove_column,
    remove_row,
    set_machine,
    set_tile,
)
from factoriax.editor.toolbar import (
    BLOCK_ITEMS,
    MACHINE_ITEMS,
    MENU_BAR_HEIGHT,
    STATUS_BAR_HEIGHT,
    TOOL_ERASE,
    TOOL_FILL,
    TOOL_PAINT,
    TOOLBAR_WIDTH,
    render_menu_bar,
    render_status_bar,
    render_toolbar,
)
from factoriax.levels import load_level, save_level
from factoriax.play.main import (
    calculate_window_size,
    composite_rgba_over_rgb,
    hit_test_regions,
    play_level,
)
from factoriax.play.ui import ClickRegion

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PAN_SPEED = 1.0
_RES_STEP = 10
_MIN_RES = 0

_DIR_CYCLE = [
    int(Action.DOWN),
    int(Action.RIGHT),
    int(Action.UP),
    int(Action.LEFT),
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
        brush: Resource brush settings.
        show_resources: Whether the resource overlay is visible.
        show_help: Whether the help overlay is visible.
        cursor_tile: Tile coordinate under the cursor, or ``None``.
        painting: ``True`` while left-click drag is active.
        last_paint_tile: Previous tile during a paint drag.
        right_erasing: ``True`` while a right-click erase drag is active.
        tool_before_erase: Tool that was active before right-click erase.
        machine_before_erase: Machine that was selected before right-click erase.
        fill_start: Start tile of a fill-rect drag, or ``None``.
        fill_rect: Current fill-rect selection, or ``None``.
        middle_dragging: ``True`` while middle-button panning is active.
        middle_last: Last raw mouse position during middle-drag.
    """

    tool: str = TOOL_PAINT
    block: int = dataclasses.field(default_factory=lambda: int(BlockType.DIRT))
    machine: int = 0
    direction: int = dataclasses.field(
        default_factory=lambda: int(Action.DOWN),
    )
    brush: ResourceBrush = dataclasses.field(default_factory=ResourceBrush)
    show_resources: bool = False
    show_help: bool = False
    cursor_tile: tuple[int, int] | None = None
    painting: bool = False
    last_paint_tile: tuple[int, int] | None = None
    right_erasing: bool = False
    tool_before_erase: str = TOOL_PAINT
    machine_before_erase: int = 0
    fill_start: tuple[int, int] | None = None
    fill_rect: tuple[int, int, int, int] | None = None
    middle_dragging: bool = False
    middle_last: tuple[int, int] = (0, 0)

    @property
    def layer(self) -> str:
        """Return ``"machine"`` or ``"terrain"`` based on brush selection."""
        return "machine" if self.machine != 0 else "terrain"

    @property
    def brush_name(self) -> str:
        """Return the display name of the active brush."""
        if self.tool == TOOL_ERASE:
            return "Eraser"
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
        return int(Action.RIGHT) if dx > 0 else int(Action.LEFT)
    return int(Action.DOWN) if dy > 0 else int(Action.UP)


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
    """Recalculate viewport dimensions after the map size changes.

    Args:
        editor: Current editor state (read-only).
        vp: Viewport to update in place.
        reset_camera: If ``True`` the camera is moved to (0, 0).
    """
    vp.canvas_w = editor.map_width * vp.tile_size
    vp.canvas_h = editor.map_height * vp.tile_size
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
    play_level(level, num_players=1, screen=screen)
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
    if cx >= 0 and cy >= 0 and cy < vp.canvas_h:
        ts.cursor_tile = screen_to_tile(vp, cx, cy)
    else:
        ts.cursor_tile = None

    if ts.painting and ts.cursor_tile is not None:
        tx, ty = ts.cursor_tile
        if (tx, ty) != ts.last_paint_tile:
            if ts.tool == TOOL_ERASE:
                if ts.right_erasing:
                    if ts.machine_before_erase != 0:
                        erase_machine(editor, tx, ty)
                    else:
                        erase_block(editor, tx, ty)
                else:
                    erase_tile(editor, tx, ty)
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
    elif hit.action == "machine":
        ts.machine = MACHINE_ITEMS[hit.param][0]
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
    ts.tool = TOOL_ERASE
    if ts.machine != 0:
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
    elif key in _MACHINE_KEYS:
        ts.machine = MACHINE_ITEMS[_MACHINE_KEYS[key]][0]

    elif key == pygame.K_r:
        _handle_rotate(ts, editor)

    elif key == pygame.K_i:
        insp = _open_inspector(ts, editor)
        if insp is not None:
            return editor, base_w, base_h, scale, running, insp

    elif key == pygame.K_t:
        ts.brush.mode = "range" if ts.brush.mode == "exact" else "exact"
    elif key == pygame.K_v:
        ts.show_resources = not ts.show_resources
    elif key == pygame.K_QUESTION or (key == pygame.K_SLASH and shift):
        ts.show_help = True

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
    menu_bar, _ = render_menu_bar(base_w)
    toolbar, _ = render_toolbar(
        ts.tool,
        ts.block,
        ts.machine,
        ts.direction,
        ts.brush,
        vp.canvas_h,
        ts.show_resources,
    )
    canvas_img = render_canvas(
        editor,
        vp,
        ts.cursor_tile,
        ts.fill_rect,
        ts.show_resources,
    )
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
    tb_h = min(toolbar.shape[0], vp.canvas_h)
    frame[MENU_BAR_HEIGHT : MENU_BAR_HEIGHT + tb_h, :TOOLBAR_WIDTH] = toolbar[:tb_h]

    canvas_rgb = canvas_img[:, :, :3]
    alpha = canvas_img[:, :, 3:4].astype(np.float32) / 255.0
    bg = frame[
        MENU_BAR_HEIGHT : MENU_BAR_HEIGHT + vp.canvas_h,
        TOOLBAR_WIDTH : TOOLBAR_WIDTH + vp.canvas_w,
    ]
    ch = min(canvas_rgb.shape[0], bg.shape[0])
    cw = min(canvas_rgb.shape[1], bg.shape[1])
    bg[:ch, :cw] = (
        canvas_rgb[:ch, :cw].astype(np.float32) * alpha[:ch, :cw]
        + bg[:ch, :cw].astype(np.float32) * (1.0 - alpha[:ch, :cw])
    ).astype(np.uint8)

    frame[base_h - STATUS_BAR_HEIGHT :, :] = status_bar

    if inspector_dialog is not None:
        composite_rgba_over_rgb(
            frame, inspector_dialog.render(base_w, base_h)
        )
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


def main() -> None:
    """Run the FactoriaX level editor.

    Press ``?`` for a full list of controls.
    """
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
                                editor_state_to_level(editor), path,
                            )
                            editor.dirty = False
                        else:
                            if path.exists():
                                level = load_level(path)
                                editor = editor_state_from_level(level)
                                _update_viewport(
                                    editor, vp, reset_camera=True,
                                )
                                base_w, base_h, scale = _recalc_layout(
                                    vp, window_w, window_h,
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
                        _, tb_regions = render_toolbar(
                            ts.tool,
                            ts.block,
                            ts.machine,
                            ts.direction,
                            ts.brush,
                            vp.canvas_h,
                            ts.show_resources,
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
                            for r in tb_regions
                        ]
                        hit = hit_test_regions(adjusted, mx, my)
                        if hit is not None:
                            nd = _handle_toolbar_click(hit, ts)
                            if nd is not None:
                                number_dialog = nd
                                number_dialog_target = hit.action.replace(
                                    "edit_res_",
                                    "",
                                )
                        continue

                    cx = mx - TOOLBAR_WIDTH
                    cy = my - MENU_BAR_HEIGHT
                    if cx >= 0 and cy >= 0 and cy < vp.canvas_h:
                        tx, ty = screen_to_tile(vp, cx, cy)
                        _handle_canvas_click(tx, ty, ts, editor, rng)

                elif event.button == 3:
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

    pygame.quit()
