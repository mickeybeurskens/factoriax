"""Main event loop for the FactoriaX level editor.

Handles window creation, event dispatch, tool state, and rendering
composition.  Mirrors the structure of ``factoriax.play.main`` but
operates on mutable :class:`~factoriax.editor.state.EditorState`
arrays instead of JAX state.
"""

from __future__ import annotations

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
    NewLevelDialog,
    NumberInputDialog,
    ask_load_path,
    ask_save_path,
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


def _brush_name(
    selected_block: int,
    selected_machine: int,
    selected_tool: str,
) -> str:
    """Return the display name of the currently active brush.

    Args:
        selected_block: Active ``BlockType`` value.
        selected_machine: Active ``MachineType`` value, or 0.
        selected_tool: Active tool name.

    Returns:
        Human-readable brush name.
    """
    if selected_tool == TOOL_ERASE:
        return "Eraser"
    if selected_machine != 0:
        for mid, name in MACHINE_ITEMS:
            if mid == selected_machine:
                return name
    for bid, name in BLOCK_ITEMS:
        if bid == selected_block:
            return name
    return "?"


def _resource_info(brush: ResourceBrush) -> str:
    """Return a status-bar summary of the resource brush.

    Args:
        brush: Active resource brush.

    Returns:
        Short string like ``"Res:100"`` or ``"Res:10-50"``.
    """
    if brush.mode == "exact":
        return f"Res:{brush.exact_value}"
    return f"Res:{brush.range_min}-{brush.range_max}"


def run_play_session(
    state: EditorState,
    screen: pygame.Surface,
) -> None:
    """Launch a full play-test session from the editor.

    Converts the editor state to a Level and delegates to
    :func:`~factoriax.play.main.play_level`, which provides the
    complete game UI (inventory, crafting, machine inspection,
    achievements, pause).  Returns to the editor when the user
    quits from the pause menu.

    Args:
        state: Current editor state.
        screen: Pygame display surface (reused by the play session).
    """
    level = editor_state_to_level(state)
    play_level(level, num_players=1, screen=screen)
    pygame.display.set_caption("FactoriaX Editor")


def main() -> None:
    """Run the FactoriaX level editor.

    Controls:
        Left click/drag on canvas: paint with selected brush
        Right click on canvas: erase (DIRT + remove machine)
        Scroll wheel: zoom
        Middle drag / arrow keys: pan
        1-5: select block brush (Dirt, Water, Iron, Copper, Coal)
        6-9: select machine brush (Miner, Chest, Belt, Arm)
        R: rotate machine direction (or rotate existing machine under cursor)
        B: paint tool, F: fill rect tool, X: eraser tool
        T: toggle resource mode (exact / range)
        [ / ]: decrease / increase resource value
        Shift+[ / Shift+]: adjust range_min
        Ctrl+N: new level, Ctrl+O: load, Ctrl+S: save, F5: play
        Escape: cancel operation / close dialog

    Conveyor belts infer direction from drag direction when painting.
    """
    pygame.init()

    editor = new_editor_state(15, 15)
    rng = np.random.default_rng(42)

    base_width = TOOLBAR_WIDTH + editor.map_width * 32
    base_height = MENU_BAR_HEIGHT + editor.map_height * 32 + STATUS_BAR_HEIGHT
    window_width, window_height = calculate_window_size(
        base_width,
        base_height,
    )
    screen = pygame.display.set_mode(
        (window_width, window_height),
        pygame.RESIZABLE,
    )
    pygame.display.set_caption("FactoriaX Editor")
    clock = pygame.time.Clock()

    scale = max(
        1,
        min(window_width // base_width, window_height // base_height),
    )

    vp = Viewport(
        tile_size=32,
        canvas_w=base_width - TOOLBAR_WIDTH,
        canvas_h=base_height - MENU_BAR_HEIGHT - STATUS_BAR_HEIGHT,
    )
    clamp_camera(vp, editor.map_width, editor.map_height)

    selected_tool = TOOL_PAINT
    selected_block = int(BlockType.DIRT)
    selected_machine = 0
    machine_direction = int(Action.DOWN)
    resource_brush = ResourceBrush()

    cursor_tile: tuple[int, int] | None = None
    painting = False
    last_paint_tile: tuple[int, int] | None = None
    right_erasing = False
    tool_before_erase: str = TOOL_PAINT
    machine_before_erase: int = 0
    fill_start: tuple[int, int] | None = None
    fill_rect: tuple[int, int, int, int] | None = None
    middle_dragging = False
    middle_last: tuple[int, int] = (0, 0)

    dialog: NewLevelDialog | None = None
    number_dialog: NumberInputDialog | None = None
    number_dialog_target: str = ""
    show_resources = False
    show_help = False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                continue

            if dialog is not None:
                result = dialog.handle_event(event)
                if result == "ok":
                    w, h, name = dialog.get_values()
                    editor = new_editor_state(w, h, name)
                    _update_layout(editor, vp, reset_camera=True)
                    base_width = TOOLBAR_WIDTH + vp.canvas_w
                    base_height = MENU_BAR_HEIGHT + vp.canvas_h + STATUS_BAR_HEIGHT
                    scale = max(
                        1,
                        min(
                            window_width // base_width,
                            window_height // base_height,
                        ),
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
                        resource_brush.exact_value = val
                    elif number_dialog_target == "min":
                        resource_brush.range_min = min(
                            val,
                            resource_brush.range_max,
                        )
                    elif number_dialog_target == "max":
                        resource_brush.range_max = max(
                            val,
                            resource_brush.range_min,
                        )
                    number_dialog = None
                elif result == "cancel":
                    number_dialog = None
                continue

            if show_help:
                if event.type == pygame.KEYDOWN:
                    show_help = False
                continue

            if event.type == pygame.VIDEORESIZE:
                window_width, window_height = event.w, event.h
                scale = max(
                    1,
                    min(
                        window_width // base_width,
                        window_height // base_height,
                    ),
                )

            elif event.type == pygame.MOUSEMOTION:
                mx = event.pos[0] // scale
                my = event.pos[1] // scale
                cx = mx - TOOLBAR_WIDTH
                cy = my - MENU_BAR_HEIGHT
                if cx >= 0 and cy >= 0 and cy < vp.canvas_h:
                    cursor_tile = screen_to_tile(vp, cx, cy)
                else:
                    cursor_tile = None

                if painting and cursor_tile is not None:
                    tx, ty = cursor_tile
                    on_new_tile = (tx, ty) != last_paint_tile
                    if on_new_tile:
                        if selected_tool == TOOL_ERASE:
                            if right_erasing:
                                if machine_before_erase != 0:
                                    erase_machine(editor, tx, ty)
                                else:
                                    erase_block(editor, tx, ty)
                            else:
                                erase_tile(editor, tx, ty)
                        elif selected_machine != 0:
                            direction = machine_direction
                            if (
                                selected_machine == _CONVEYOR
                                and last_paint_tile is not None
                            ):
                                drag_dir = _direction_from_delta(
                                    tx - last_paint_tile[0],
                                    ty - last_paint_tile[1],
                                )
                                if drag_dir is not None:
                                    direction = drag_dir
                            set_machine(
                                editor,
                                tx,
                                ty,
                                selected_machine,
                                direction,
                            )
                            last_paint_tile = (tx, ty)
                        else:
                            set_tile(
                                editor,
                                tx,
                                ty,
                                selected_block,
                                resource_brush,
                                rng,
                            )

                if fill_start is not None and cursor_tile is not None:
                    fill_rect = (*fill_start, *cursor_tile)

                if middle_dragging:
                    dx = (middle_last[0] - event.pos[0]) / scale / vp.tile_size
                    dy = (middle_last[1] - event.pos[1]) / scale / vp.tile_size
                    pan(vp, dx, dy, editor.map_width, editor.map_height)
                    middle_last = event.pos

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx = event.pos[0] // scale
                my = event.pos[1] // scale

                if event.button == 2:
                    middle_dragging = True
                    middle_last = event.pos
                    continue

                if event.button in (4, 5):
                    cx = mx - TOOLBAR_WIDTH
                    cy = my - MENU_BAR_HEIGHT
                    if cx >= 0 and cy >= 0:
                        direction = 1 if event.button == 4 else -1
                        zoom(
                            vp,
                            direction,
                            cx,
                            cy,
                            editor.map_width,
                            editor.map_height,
                        )
                    continue

                if event.button == 1:
                    if my < MENU_BAR_HEIGHT:
                        _, menu_regions = render_menu_bar(base_width)
                        hit = hit_test_regions(menu_regions, mx, my)
                        if hit is not None:
                            if hit.action == "new":
                                dialog = NewLevelDialog()
                            elif hit.action == "load":
                                path = ask_load_path()
                                if path is not None:
                                    level = load_level(path)
                                    editor = editor_state_from_level(level)
                                    _update_layout(
                                        editor,
                                        vp,
                                        reset_camera=True,
                                    )
                                    base_width = TOOLBAR_WIDTH + vp.canvas_w
                                    base_height = (
                                        MENU_BAR_HEIGHT
                                        + vp.canvas_h
                                        + STATUS_BAR_HEIGHT
                                    )
                                    scale = max(
                                        1,
                                        min(
                                            window_width // base_width,
                                            window_height // base_height,
                                        ),
                                    )
                            elif hit.action == "save":
                                path = ask_save_path(editor.name)
                                if path is not None:
                                    save_level(
                                        editor_state_to_level(editor),
                                        path,
                                    )
                                    editor.dirty = False
                            elif hit.action == "play":
                                run_play_session(editor, screen)
                        continue

                    if mx < TOOLBAR_WIDTH:
                        _, tb_regions = render_toolbar(
                            selected_tool,
                            selected_block,
                            selected_machine,
                            machine_direction,
                            resource_brush,
                            vp.canvas_h,
                            show_resources,
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
                            if hit.action == "tool":
                                selected_tool = [
                                    TOOL_PAINT,
                                    TOOL_FILL,
                                    TOOL_ERASE,
                                ][hit.param]
                            elif hit.action == "block":
                                selected_block = BLOCK_ITEMS[hit.param][0]
                                selected_machine = 0
                            elif hit.action == "machine":
                                selected_machine = MACHINE_ITEMS[hit.param][0]
                            elif hit.action == "toggle_res_mode":
                                resource_brush.mode = (
                                    "range"
                                    if resource_brush.mode == "exact"
                                    else "exact"
                                )
                            elif hit.action == "edit_res_exact":
                                number_dialog = NumberInputDialog(
                                    label=f"Amount (0-{BLOCK_MAX_RESOURCES}):",
                                    text="",
                                    max_value=BLOCK_MAX_RESOURCES,
                                    default=resource_brush.exact_value,
                                )
                                number_dialog_target = "exact"
                            elif hit.action == "edit_res_min":
                                number_dialog = NumberInputDialog(
                                    label=f"Min (0-{BLOCK_MAX_RESOURCES}):",
                                    text="",
                                    max_value=BLOCK_MAX_RESOURCES,
                                    default=resource_brush.range_min,
                                )
                                number_dialog_target = "min"
                            elif hit.action == "edit_res_max":
                                number_dialog = NumberInputDialog(
                                    label=f"Max (0-{BLOCK_MAX_RESOURCES}):",
                                    text="",
                                    max_value=BLOCK_MAX_RESOURCES,
                                    default=resource_brush.range_max,
                                )
                                number_dialog_target = "max"
                            elif hit.action == "toggle_show_res":
                                show_resources = not show_resources
                        continue

                    cx = mx - TOOLBAR_WIDTH
                    cy = my - MENU_BAR_HEIGHT
                    if cx >= 0 and cy >= 0 and cy < vp.canvas_h:
                        tx, ty = screen_to_tile(vp, cx, cy)
                        if selected_tool == TOOL_FILL:
                            fill_start = (tx, ty)
                            fill_rect = (tx, ty, tx, ty)
                        else:
                            painting = True
                            last_paint_tile = (tx, ty)
                            if selected_tool == TOOL_ERASE:
                                erase_tile(editor, tx, ty)
                            elif selected_machine != 0:
                                set_machine(
                                    editor,
                                    tx,
                                    ty,
                                    selected_machine,
                                    machine_direction,
                                )
                            else:
                                set_tile(
                                    editor,
                                    tx,
                                    ty,
                                    selected_block,
                                    resource_brush,
                                    rng,
                                )

                elif event.button == 3:
                    cx = mx - TOOLBAR_WIDTH
                    cy = my - MENU_BAR_HEIGHT
                    if cx >= 0 and cy >= 0 and cy < vp.canvas_h:
                        tx, ty = screen_to_tile(vp, cx, cy)
                        right_erasing = True
                        tool_before_erase = selected_tool
                        machine_before_erase = selected_machine
                        selected_tool = TOOL_ERASE
                        if selected_machine != 0:
                            erase_machine(editor, tx, ty)
                        else:
                            erase_block(editor, tx, ty)
                        painting = True

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 2:
                    middle_dragging = False
                elif event.button == 3:
                    if right_erasing:
                        selected_tool = tool_before_erase
                        right_erasing = False
                        painting = False
                        last_paint_tile = None
                elif event.button == 1:
                    if fill_start is not None and cursor_tile is not None:
                        tx, ty = cursor_tile
                        if selected_machine != 0:
                            fill_dir = _direction_from_delta(
                                tx - fill_start[0],
                                ty - fill_start[1],
                            )
                            if selected_machine == _CONVEYOR and fill_dir is not None:
                                direction = fill_dir
                            else:
                                direction = machine_direction
                            lx = min(fill_start[0], tx)
                            ly = min(fill_start[1], ty)
                            rx = max(fill_start[0], tx)
                            ry = max(fill_start[1], ty)
                            for fy in range(
                                max(0, ly),
                                min(editor.map_height, ry + 1),
                            ):
                                for fx in range(
                                    max(0, lx),
                                    min(editor.map_width, rx + 1),
                                ):
                                    set_machine(
                                        editor,
                                        fx,
                                        fy,
                                        selected_machine,
                                        direction,
                                    )
                        else:
                            fill_rect_tiles(
                                editor,
                                fill_start[0],
                                fill_start[1],
                                tx,
                                ty,
                                selected_block,
                                resource_brush,
                                rng,
                            )
                        fill_start = None
                        fill_rect = None
                    painting = False
                    last_paint_tile = None

            elif event.type == pygame.KEYDOWN:
                mods = pygame.key.get_mods()
                ctrl = mods & pygame.KMOD_CTRL
                shift = mods & pygame.KMOD_SHIFT

                if event.key == pygame.K_ESCAPE:
                    if fill_start is not None:
                        fill_start = None
                        fill_rect = None
                    else:
                        running = False

                elif ctrl and event.key == pygame.K_n:
                    dialog = NewLevelDialog()
                elif ctrl and event.key == pygame.K_o:
                    path = ask_load_path()
                    if path is not None:
                        level = load_level(path)
                        editor = editor_state_from_level(level)
                        _update_layout(editor, vp, reset_camera=True)
                        base_width = TOOLBAR_WIDTH + vp.canvas_w
                        base_height = MENU_BAR_HEIGHT + vp.canvas_h + STATUS_BAR_HEIGHT
                        scale = max(
                            1,
                            min(
                                window_width // base_width,
                                window_height // base_height,
                            ),
                        )
                elif ctrl and event.key == pygame.K_s:
                    path = ask_save_path(editor.name)
                    if path is not None:
                        save_level(editor_state_to_level(editor), path)
                        editor.dirty = False
                elif event.key == pygame.K_F5:
                    run_play_session(editor, screen)

                elif event.key == pygame.K_b:
                    selected_tool = TOOL_PAINT
                elif event.key == pygame.K_f:
                    selected_tool = TOOL_FILL
                elif event.key == pygame.K_x:
                    selected_tool = TOOL_ERASE

                elif event.key == pygame.K_1:
                    selected_block = BLOCK_ITEMS[0][0]
                    selected_machine = 0
                elif event.key == pygame.K_2:
                    selected_block = BLOCK_ITEMS[1][0]
                    selected_machine = 0
                elif event.key == pygame.K_3:
                    selected_block = BLOCK_ITEMS[2][0]
                    selected_machine = 0
                elif event.key == pygame.K_4:
                    selected_block = BLOCK_ITEMS[3][0]
                    selected_machine = 0
                elif event.key == pygame.K_5:
                    selected_block = BLOCK_ITEMS[4][0]
                    selected_machine = 0
                elif event.key == pygame.K_6:
                    selected_machine = MACHINE_ITEMS[0][0]
                elif event.key == pygame.K_7:
                    selected_machine = MACHINE_ITEMS[1][0]
                elif event.key == pygame.K_8:
                    selected_machine = MACHINE_ITEMS[2][0]
                elif event.key == pygame.K_9:
                    selected_machine = MACHINE_ITEMS[3][0]

                elif event.key == pygame.K_r:
                    if (
                        cursor_tile is not None
                        and 0 <= cursor_tile[0] < editor.map_width
                        and 0 <= cursor_tile[1] < editor.map_height
                        and editor.machine_types[cursor_tile[1], cursor_tile[0]]
                        != int(MachineType.NONE)
                    ):
                        cy, cx = cursor_tile[1], cursor_tile[0]
                        editor.machine_directions[cy, cx] = _next_direction(
                            int(editor.machine_directions[cy, cx]),
                        )
                        editor.dirty = True
                    else:
                        machine_direction = _next_direction(
                            machine_direction,
                        )

                elif event.key == pygame.K_t:
                    resource_brush.mode = (
                        "range" if resource_brush.mode == "exact" else "exact"
                    )
                elif event.key == pygame.K_v:
                    show_resources = not show_resources
                elif event.key == pygame.K_QUESTION or (
                    event.key == pygame.K_SLASH and shift
                ):
                    show_help = True

                elif event.key == pygame.K_RIGHTBRACKET:
                    if resource_brush.mode == "exact":
                        resource_brush.exact_value = min(
                            BLOCK_MAX_RESOURCES,
                            resource_brush.exact_value + _RES_STEP,
                        )
                    else:
                        resource_brush.range_max = min(
                            BLOCK_MAX_RESOURCES,
                            resource_brush.range_max + _RES_STEP,
                        )
                elif event.key == pygame.K_LEFTBRACKET:
                    if shift:
                        resource_brush.range_min = max(
                            _MIN_RES,
                            resource_brush.range_min - _RES_STEP,
                        )
                    elif resource_brush.mode == "exact":
                        resource_brush.exact_value = max(
                            _MIN_RES,
                            resource_brush.exact_value - _RES_STEP,
                        )
                    else:
                        resource_brush.range_max = max(
                            resource_brush.range_min,
                            resource_brush.range_max - _RES_STEP,
                        )

                elif ctrl and event.key == pygame.K_RIGHT:
                    add_column(editor)
                    _update_layout(editor, vp)
                    base_width = TOOLBAR_WIDTH + vp.canvas_w
                    base_height = (
                        MENU_BAR_HEIGHT + vp.canvas_h + STATUS_BAR_HEIGHT
                    )
                    scale = max(
                        1,
                        min(
                            window_width // base_width,
                            window_height // base_height,
                        ),
                    )
                elif ctrl and event.key == pygame.K_LEFT:
                    remove_column(editor)
                    _update_layout(editor, vp)
                    base_width = TOOLBAR_WIDTH + vp.canvas_w
                    base_height = (
                        MENU_BAR_HEIGHT + vp.canvas_h + STATUS_BAR_HEIGHT
                    )
                    scale = max(
                        1,
                        min(
                            window_width // base_width,
                            window_height // base_height,
                        ),
                    )
                elif ctrl and event.key == pygame.K_DOWN:
                    add_row(editor)
                    _update_layout(editor, vp)
                    base_width = TOOLBAR_WIDTH + vp.canvas_w
                    base_height = (
                        MENU_BAR_HEIGHT + vp.canvas_h + STATUS_BAR_HEIGHT
                    )
                    scale = max(
                        1,
                        min(
                            window_width // base_width,
                            window_height // base_height,
                        ),
                    )
                elif ctrl and event.key == pygame.K_UP:
                    remove_row(editor)
                    _update_layout(editor, vp)
                    base_width = TOOLBAR_WIDTH + vp.canvas_w
                    base_height = (
                        MENU_BAR_HEIGHT + vp.canvas_h + STATUS_BAR_HEIGHT
                    )
                    scale = max(
                        1,
                        min(
                            window_width // base_width,
                            window_height // base_height,
                        ),
                    )
                elif event.key == pygame.K_LEFT:
                    pan(
                        vp,
                        -_PAN_SPEED,
                        0,
                        editor.map_width,
                        editor.map_height,
                    )
                elif event.key == pygame.K_RIGHT:
                    pan(
                        vp,
                        _PAN_SPEED,
                        0,
                        editor.map_width,
                        editor.map_height,
                    )
                elif event.key == pygame.K_UP:
                    pan(vp, 0, -_PAN_SPEED, 0, editor.map_height)
                elif event.key == pygame.K_DOWN:
                    pan(
                        vp,
                        0,
                        _PAN_SPEED,
                        editor.map_width,
                        editor.map_height,
                    )

        menu_bar, _ = render_menu_bar(base_width)
        toolbar, _ = render_toolbar(
            selected_tool,
            selected_block,
            selected_machine,
            machine_direction,
            resource_brush,
            vp.canvas_h,
            show_resources,
        )

        canvas_img = render_canvas(
            editor,
            vp,
            cursor_tile,
            fill_rect,
            show_resources,
        )

        bn = _brush_name(selected_block, selected_machine, selected_tool)
        ri = _resource_info(resource_brush)
        active_layer = "machine" if selected_machine != 0 else "terrain"
        status_bar = render_status_bar(
            selected_tool,
            bn,
            cursor_tile,
            editor.name,
            editor.dirty,
            ri,
            base_width,
            layer=active_layer,
        )

        frame = np.full(
            (base_height, base_width, 3),
            (30, 30, 30),
            dtype=np.uint8,
        )
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
        blended = (
            canvas_rgb[:ch, :cw].astype(np.float32) * alpha[:ch, :cw]
            + bg[:ch, :cw].astype(np.float32) * (1.0 - alpha[:ch, :cw])
        ).astype(np.uint8)
        bg[:ch, :cw] = blended

        frame[base_height - STATUS_BAR_HEIGHT :, :] = status_bar

        if dialog is not None:
            dialog_overlay = dialog.render(base_width, base_height)
            frame = composite_rgba_over_rgb(frame, dialog_overlay)

        if number_dialog is not None:
            num_overlay = number_dialog.render(base_width, base_height)
            frame = composite_rgba_over_rgb(frame, num_overlay)

        if show_help:
            help_overlay = render_help_overlay(base_width, base_height)
            frame = composite_rgba_over_rgb(frame, help_overlay)

        base_surface = pygame.surfarray.make_surface(
            np.transpose(frame, (1, 0, 2)),
        )
        scaled = pygame.transform.scale(
            base_surface,
            (base_width * scale, base_height * scale),
        )
        screen.fill((0, 0, 0))
        screen.blit(scaled, (0, 0))
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


def _update_layout(
    editor: EditorState,
    vp: Viewport,
    reset_camera: bool = False,
) -> None:
    """Recalculate viewport dimensions after the map size changes.

    Args:
        editor: Current editor state (read-only).
        vp: Viewport to update in place.
        reset_camera: If ``True`` the camera is moved to (0, 0).
            Otherwise the current position is preserved and clamped.
    """
    vp.canvas_w = editor.map_width * vp.tile_size
    vp.canvas_h = editor.map_height * vp.tile_size
    if reset_camera:
        vp.camera_x = 0.0
        vp.camera_y = 0.0
    clamp_camera(vp, editor.map_width, editor.map_height)
