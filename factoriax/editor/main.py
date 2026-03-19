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
    ask_load_path,
    ask_save_path,
)
from factoriax.editor.state import (
    EditorState,
    ResourceBrush,
    editor_state_from_level,
    editor_state_to_level,
    erase_tile,
    fill_rect_tiles,
    new_editor_state,
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
)
from factoriax.play.ui import ClickRegion

_PAN_SPEED = 1.0
_RES_STEP = 10
_MIN_RES = 0


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
    window_width: int,
    window_height: int,
) -> None:
    """Launch a play-test session from the editor.

    Converts the editor state to a Level, resets the environment, and
    runs a simplified play loop.  Returns to the editor when the user
    presses Escape.

    Args:
        state: Current editor state.
        screen: Pygame display surface.
        window_width: Window width in pixels.
        window_height: Window height in pixels.
    """
    import jax
    import jax.numpy as jnp
    from jax import random

    from factoriax.envs.factoriax_env import make_factoriax_env
    from factoriax.renderer import render_pixels
    from factoriax.state import EnvParams

    level = editor_state_to_level(state)
    env, _ = make_factoriax_env()
    params = EnvParams(
        map_width=level.map_width,
        map_height=level.map_height,
        num_players=1,
    )

    obs, env_state = env.reset_from_level(level, params)
    step_fn = jax.jit(env.step_env)

    rng = random.PRNGKey(0)
    rng, warmup_key = random.split(rng)
    step_fn(warmup_key, env_state, jnp.int32(Action.NOOP), params)[
        0
    ].block_until_ready()

    key_to_action = {
        pygame.K_a: Action.LEFT,
        pygame.K_d: Action.RIGHT,
        pygame.K_w: Action.UP,
        pygame.K_s: Action.DOWN,
        pygame.K_SPACE: Action.MINE,
        pygame.K_e: Action.PLACE,
    }

    clock = pygame.time.Clock()
    base_w = params.map_width * 32
    base_h = params.map_height * 32
    scale = max(1, min(window_width // base_w, window_height // base_h))

    running = True
    while running:
        action = Action.NOOP
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in key_to_action:
                    action = key_to_action[event.key]

        if not running:
            break

        if action != Action.NOOP:
            rng, step_key = random.split(rng)
            obs, env_state, reward, done, info = step_fn(
                step_key, env_state, jnp.int32(action), params
            )
            if done:
                obs, env_state = env.reset_from_level(level, params)

        pixels = render_pixels(env_state)
        base_surface = pygame.surfarray.make_surface(np.transpose(pixels, (1, 0, 2)))
        scaled = pygame.transform.scale(base_surface, (base_w * scale, base_h * scale))
        screen.fill((0, 0, 0))
        sx = (window_width - base_w * scale) // 2
        sy = (window_height - base_h * scale) // 2
        screen.blit(scaled, (sx, sy))
        pygame.display.flip()
        clock.tick(30)


def main() -> None:
    """Run the FactoriaX level editor.

    Controls:
        Left click/drag on canvas: paint with selected brush
        Right click on canvas: erase (DIRT + remove machine)
        Scroll wheel: zoom
        Middle drag / arrow keys: pan
        1-5: select block brush (Dirt, Water, Iron, Copper, Coal)
        6-9: select machine brush (Miner, Chest, Belt, Arm)
        R: rotate machine direction
        B: paint tool, F: fill rect tool, X: eraser tool
        T: toggle resource mode (exact / range)
        [ / ]: decrease / increase resource value
        Shift+[ / Shift+]: adjust range_min
        Ctrl+N: new level, Ctrl+O: load, Ctrl+S: save, F5: play
        Escape: cancel operation / close dialog
    """
    pygame.init()

    editor = new_editor_state(15, 15)
    rng = np.random.default_rng(42)

    base_width = TOOLBAR_WIDTH + editor.map_width * 32
    base_height = MENU_BAR_HEIGHT + editor.map_height * 32 + STATUS_BAR_HEIGHT
    window_width, window_height = calculate_window_size(base_width, base_height)
    screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
    pygame.display.set_caption("FactoriaX Editor")
    clock = pygame.time.Clock()

    scale = max(1, min(window_width // base_width, window_height // base_height))

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
    fill_start: tuple[int, int] | None = None
    fill_rect: tuple[int, int, int, int] | None = None
    middle_dragging = False
    middle_last: tuple[int, int] = (0, 0)

    dialog: NewLevelDialog | None = None

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
                    _update_layout(editor, vp)
                    base_width = TOOLBAR_WIDTH + vp.canvas_w
                    base_height = MENU_BAR_HEIGHT + vp.canvas_h + STATUS_BAR_HEIGHT
                    scale = max(
                        1, min(window_width // base_width, window_height // base_height)
                    )
                    dialog = None
                elif result == "cancel":
                    dialog = None
                continue

            if event.type == pygame.VIDEORESIZE:
                window_width, window_height = event.w, event.h
                scale = max(
                    1, min(window_width // base_width, window_height // base_height)
                )

            elif event.type == pygame.MOUSEMOTION:
                mx, my = event.pos[0] // scale, event.pos[1] // scale
                cx = mx - TOOLBAR_WIDTH
                cy = my - MENU_BAR_HEIGHT
                if cx >= 0 and cy >= 0 and cy < vp.canvas_h:
                    cursor_tile = screen_to_tile(vp, cx, cy)
                else:
                    cursor_tile = None

                if painting and cursor_tile is not None:
                    tx, ty = cursor_tile
                    if selected_tool == TOOL_ERASE:
                        erase_tile(editor, tx, ty)
                    elif selected_machine != 0:
                        set_machine(editor, tx, ty, selected_machine, machine_direction)
                    else:
                        set_tile(editor, tx, ty, selected_block, resource_brush, rng)

                if fill_start is not None and cursor_tile is not None:
                    fill_rect = (*fill_start, *cursor_tile)

                if middle_dragging:
                    dx = (middle_last[0] - event.pos[0]) / scale / vp.tile_size
                    dy = (middle_last[1] - event.pos[1]) / scale / vp.tile_size
                    pan(vp, dx, dy, editor.map_width, editor.map_height)
                    middle_last = event.pos

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos[0] // scale, event.pos[1] // scale

                if event.button == 2:
                    middle_dragging = True
                    middle_last = event.pos
                    continue

                if event.button in (4, 5):
                    cx = mx - TOOLBAR_WIDTH
                    cy = my - MENU_BAR_HEIGHT
                    if cx >= 0 and cy >= 0:
                        direction = 1 if event.button == 4 else -1
                        zoom(vp, direction, cx, cy, editor.map_width, editor.map_height)
                    continue

                if event.button == 1:
                    if my < MENU_BAR_HEIGHT:
                        menu_bar, menu_regions = render_menu_bar(base_width)
                        hit = hit_test_regions(menu_regions, mx, my)
                        if hit is not None:
                            _handle_menu_action(
                                hit.action,
                                editor,
                                vp,
                                screen,
                                window_width,
                                window_height,
                            )
                            if hit.action == "new":
                                dialog = NewLevelDialog()
                            elif hit.action == "load":
                                path = ask_load_path()
                                if path is not None:
                                    level = load_level(path)
                                    editor = editor_state_from_level(level)
                                    _update_layout(editor, vp)
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
                                    save_level(editor_state_to_level(editor), path)
                                    editor.dirty = False
                            elif hit.action == "play":
                                run_play_session(
                                    editor, screen, window_width, window_height
                                )
                        continue

                    if mx < TOOLBAR_WIDTH:
                        _, tb_regions = render_toolbar(
                            selected_tool,
                            selected_block,
                            selected_machine,
                            machine_direction,
                            resource_brush,
                            vp.canvas_h,
                        )
                        adjusted = [
                            ClickRegion(
                                r.x, r.y + MENU_BAR_HEIGHT, r.w, r.h, r.action, r.param
                            )
                            for r in tb_regions
                        ]
                        hit = hit_test_regions(adjusted, mx, my)
                        if hit is not None:
                            if hit.action == "tool":
                                selected_tool = [TOOL_PAINT, TOOL_FILL, TOOL_ERASE][
                                    hit.param
                                ]
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
                            if selected_tool == TOOL_ERASE:
                                erase_tile(editor, tx, ty)
                            elif selected_machine != 0:
                                set_machine(
                                    editor, tx, ty, selected_machine, machine_direction
                                )
                            else:
                                set_tile(
                                    editor, tx, ty, selected_block, resource_brush, rng
                                )

                elif event.button == 3:
                    cx = mx - TOOLBAR_WIDTH
                    cy = my - MENU_BAR_HEIGHT
                    if cx >= 0 and cy >= 0 and cy < vp.canvas_h:
                        tx, ty = screen_to_tile(vp, cx, cy)
                        erase_tile(editor, tx, ty)
                        painting = True
                        selected_tool = TOOL_ERASE

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 2:
                    middle_dragging = False
                elif event.button == 1:
                    if fill_start is not None and cursor_tile is not None:
                        tx, ty = cursor_tile
                        if selected_machine != 0:
                            lx = min(fill_start[0], tx)
                            ly = min(fill_start[1], ty)
                            rx = max(fill_start[0], tx)
                            ry = max(fill_start[1], ty)
                            for fy in range(max(0, ly), min(editor.map_height, ry + 1)):
                                for fx in range(
                                    max(0, lx), min(editor.map_width, rx + 1)
                                ):
                                    set_machine(
                                        editor,
                                        fx,
                                        fy,
                                        selected_machine,
                                        machine_direction,
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
                        _update_layout(editor, vp)
                        base_width = TOOLBAR_WIDTH + vp.canvas_w
                        base_height = MENU_BAR_HEIGHT + vp.canvas_h + STATUS_BAR_HEIGHT
                        scale = max(
                            1,
                            min(
                                window_width // base_width, window_height // base_height
                            ),
                        )
                elif ctrl and event.key == pygame.K_s:
                    path = ask_save_path(editor.name)
                    if path is not None:
                        save_level(editor_state_to_level(editor), path)
                        editor.dirty = False
                elif event.key == pygame.K_F5:
                    run_play_session(editor, screen, window_width, window_height)

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
                    dirs = [
                        int(Action.DOWN),
                        int(Action.RIGHT),
                        int(Action.UP),
                        int(Action.LEFT),
                    ]
                    idx = (
                        dirs.index(machine_direction)
                        if machine_direction in dirs
                        else 0
                    )
                    machine_direction = dirs[(idx + 1) % 4]

                elif event.key == pygame.K_t:
                    resource_brush.mode = (
                        "range" if resource_brush.mode == "exact" else "exact"
                    )

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
                            _MIN_RES, resource_brush.range_min - _RES_STEP
                        )
                    elif resource_brush.mode == "exact":
                        resource_brush.exact_value = max(
                            _MIN_RES, resource_brush.exact_value - _RES_STEP
                        )
                    else:
                        resource_brush.range_max = max(
                            resource_brush.range_min,
                            resource_brush.range_max - _RES_STEP,
                        )

                elif event.key == pygame.K_LEFT:
                    pan(vp, -_PAN_SPEED, 0, editor.map_width, editor.map_height)
                elif event.key == pygame.K_RIGHT:
                    pan(vp, _PAN_SPEED, 0, editor.map_width, editor.map_height)
                elif event.key == pygame.K_UP:
                    pan(vp, 0, -_PAN_SPEED, 0, editor.map_height)
                elif event.key == pygame.K_DOWN:
                    pan(vp, 0, _PAN_SPEED, editor.map_width, editor.map_height)

        menu_bar, _ = render_menu_bar(base_width)
        toolbar, _ = render_toolbar(
            selected_tool,
            selected_block,
            selected_machine,
            machine_direction,
            resource_brush,
            vp.canvas_h,
        )

        canvas_img = render_canvas(editor, vp, cursor_tile, fill_rect)

        bn = _brush_name(selected_block, selected_machine, selected_tool)
        ri = _resource_info(resource_brush)
        status_bar = render_status_bar(
            selected_tool,
            bn,
            cursor_tile,
            editor.name,
            editor.dirty,
            ri,
            base_width,
        )

        frame = np.full((base_height, base_width, 3), (30, 30, 30), dtype=np.uint8)
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

        base_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        scaled = pygame.transform.scale(
            base_surface, (base_width * scale, base_height * scale)
        )
        screen.fill((0, 0, 0))
        screen.blit(scaled, (0, 0))
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


def _update_layout(editor: EditorState, vp: Viewport) -> None:
    """Recalculate viewport dimensions after the map size changes.

    Args:
        editor: Current editor state (read-only).
        vp: Viewport to update in place.
    """
    vp.canvas_w = editor.map_width * vp.tile_size
    vp.canvas_h = editor.map_height * vp.tile_size
    vp.camera_x = 0.0
    vp.camera_y = 0.0
    clamp_camera(vp, editor.map_width, editor.map_height)


def _handle_menu_action(
    action: str,
    editor: EditorState,
    vp: Viewport,
    screen: pygame.Surface,
    window_width: int,
    window_height: int,
) -> None:
    """Placeholder for menu actions handled inline in the event loop.

    Args:
        action: Menu action name.
        editor: Current editor state.
        vp: Current viewport.
        screen: Pygame display surface.
        window_width: Window width.
        window_height: Window height.
    """
