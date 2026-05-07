"""Controls and display settings screen for FactoriaX.

Tabbed view of keyboard and controller bindings, plus a Display section
with fullscreen and UI scale toggles. Click a binding to enter listening
mode and press a new key/button to rebind. Bindings are mutated on the
provided ``PlayerConfig`` in place.

Size note: ``run_controls_menu`` is a single pygame event loop with
three concerns (keyboard rebind, controller rebind, display config)
gated by a tab selector; their state and event handling are tightly
interleaved, so factoring out sub-flows would cross-import each
flow's state back into this module. The file sits just over the
spec's 800-line ceiling for the same reason ``play/ui.py`` does:
the screen is one coherent unit.
"""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from factoriax.config import (
    PlayerAction,
    PlayerConfig,
    build_controller_lookup,
    build_key_lookup,
    default_controller,
    default_keyboard,
    resolve_event,
)
from factoriax.play.launch_screen import _confirm_scale_change
from factoriax.ui import theme as _theme
from factoriax.ui.fonts import get_pixel_font
from factoriax.ui.forms import (
    BASE_BTN_H,
    BASE_BTN_W,
    BASE_CHECKBOX_SIZE,
    BASE_INPUT_H,
    BASE_INPUT_W,
    BASE_ROW_GAP,
    BASE_ROW_H,
    BASE_SCROLL_STEP,
    BASE_SECTION_GAP,
    BASE_SECTION_HEADER_H,
    BASE_SECTION_PAD_TOP,
    BASE_SECTION_RULE_H,
    BASE_SIDE_PAD,
    BASE_TOP_BAR_H,
    BG,
    FPS,
    GOLD,
    INPUT_BG,
    INPUT_BORDER,
    INPUT_TEXT,
    LABEL_COLOR,
    SECTION_RULE,
    draw_button,
    draw_checkbox,
    draw_scrollbar,
)
from factoriax.ui.scaling import ScaledCanvas
from factoriax.ui.window import auto_ui_scale, calculate_window_size

# -- Controls display helper ----------------------------------------------


_REBIND_ACTIONS: list[tuple[str | None, list[tuple[str, str]]]] = []
"""Grouped actions for the rebinding UI: ``(category, [(action, label)])``."""


def _init_rebind_actions() -> list[tuple[str | None, list[tuple[str, str]]]]:
    """Build the grouped action list on first access.

    Returns:
        List of ``(category_label, [(action_key, display_label)])``.
    """
    from factoriax.config import PlayerAction

    return [
        (
            "Movement",
            [
                (PlayerAction.MOVE_UP, "Move Up"),
                (PlayerAction.MOVE_DOWN, "Move Down"),
                (PlayerAction.MOVE_LEFT, "Move Left"),
                (PlayerAction.MOVE_RIGHT, "Move Right"),
            ],
        ),
        (
            "Actions",
            [
                (PlayerAction.MINE, "Mine"),
                (PlayerAction.INTERACT, "Interact"),
                (PlayerAction.ROTATE, "Rotate"),
            ],
        ),
        (
            "Menus",
            [
                (PlayerAction.OPEN_INVENTORY, "Inventory"),
                (PlayerAction.OPEN_ACHIEVEMENTS, "Achievements"),
                (PlayerAction.OPEN_MACHINE, "Inspect Machine"),
                (PlayerAction.OPEN_HELP, "Help"),
                (PlayerAction.TOGGLE_HOTBAR, "Hotbar Page"),
                (PlayerAction.CONFIRM, "Confirm"),
            ],
        ),
    ]


# Friendly display names for controller input strings.
_CONTROLLER_DISPLAY: dict[str, str] = {
    "BUTTON_0": "A / Cross",
    "BUTTON_1": "B / Circle",
    "BUTTON_2": "X / Square",
    "BUTTON_3": "Y / Triangle",
    "BUTTON_4": "LB",
    "BUTTON_5": "RB",
    "BUTTON_6": "Select",
    "BUTTON_7": "Start",
    "BUTTON_8": "L3",
    "BUTTON_9": "R3",
    "AXIS_0_NEG": "L-Stick Left",
    "AXIS_0_POS": "L-Stick Right",
    "AXIS_1_NEG": "L-Stick Up",
    "AXIS_1_POS": "L-Stick Down",
    "HAT_0_UP": "D-pad Up",
    "HAT_0_DOWN": "D-pad Down",
    "HAT_0_LEFT": "D-pad Left",
    "HAT_0_RIGHT": "D-pad Right",
}


def _format_key_display(names: list[str]) -> str:
    """Format keyboard binding names for display.

    Args:
        names: Raw binding names (e.g. ``["K_w", "K_UP"]``).

    Returns:
        Human-readable string (e.g. ``"w, UP"``).
    """
    if not names:
        return "-"
    parts: list[str] = []
    for n in names:
        # "SHIFT+K_1" -> "Shift+1", "K_w" -> "w", "K_UP" -> "UP"
        pieces = n.split("+")
        formatted = []
        for p in pieces:
            if p.startswith("K_"):
                formatted.append(p[2:])
            else:
                formatted.append(p.capitalize())
        parts.append("+".join(formatted))
    return ", ".join(parts)


def _format_controller_display(names: list[str]) -> str:
    """Format controller binding names for display.

    Args:
        names: Raw binding names (e.g. ``["BUTTON_0"]``).

    Returns:
        Human-readable string (e.g. ``"A / Cross"``).
    """
    if not names:
        return "-"
    return ", ".join(_CONTROLLER_DISPLAY.get(n, n) for n in names)


def _format_binding(names: list[str], device: str) -> str:
    """Format binding names for the active device tab.

    Args:
        names: Raw binding name list from config.
        device: ``"keyboard"`` or ``"controller"``.

    Returns:
        Human-readable display string.
    """
    if device == "keyboard":
        return _format_key_display(names)
    return _format_controller_display(names)


@dataclass
class _RebindState:
    """Tracks which tab is active and whether we're listening for input."""

    active_tab: str = "keyboard"
    listening_action: str | None = None
    focused_row: int = 0


def _controls_section_height(
    n_rows: int,
    n_categories: int,
    s: int,
) -> int:
    """Compute the pixel height of the controls section.

    Args:
        n_rows: Number of binding rows.
        n_categories: Number of category sub-headers.
        s: UI scale factor.

    Returns:
        Total height in pixels.
    """
    section_header_h = BASE_SECTION_HEADER_H * s
    section_rule_h = BASE_SECTION_RULE_H * s
    section_gap = BASE_SECTION_GAP * s
    row_h = BASE_ROW_H * s
    row_gap = BASE_ROW_GAP * s
    cat_h = 20 * s
    btn_h = BASE_BTN_H * s
    pad = BASE_SECTION_PAD_TOP * s
    return (
        pad
        + section_header_h
        + row_h
        + row_gap  # tab bar
        + section_rule_h
        + section_gap
        + n_categories * (cat_h + row_gap)
        + n_rows * (row_h + row_gap)
        + btn_h
        + row_gap  # reset button
        + pad
    )


def run_controls_menu(
    screen: pygame.Surface,
    config: PlayerConfig,
) -> tuple[bool, int]:
    """Show key/controller bindings with rebinding and display settings.

    Displays a tabbed view of keyboard and controller bindings. Each
    binding is clickable: click to enter listening mode, then press the
    desired key or button to rebind it. The Display section at the
    bottom provides fullscreen and UI scale toggles.

    Bindings are mutated in place on *config*. The caller should
    persist the config after this function returns.

    Args:
        screen: Pygame display surface.
        config: Full player configuration (bindings mutated in place).

    Returns:
        Tuple of ``(fullscreen, ui_scale)`` with possibly updated
        display settings.
    """
    from factoriax.config import (
        Bindings,
        controller_event_to_name,
        event_to_key_name,
    )

    global _REBIND_ACTIONS  # noqa: PLW0603
    if not _REBIND_ACTIONS:
        _REBIND_ACTIONS = _init_rebind_actions()

    s = _theme.UI_SCALE
    canvas = ScaledCanvas(1024, s, screen)
    sw, sh = canvas.width, canvas.height

    # Scaled layout constants.
    top_bar_h = BASE_TOP_BAR_H * s
    section_pad_top = BASE_SECTION_PAD_TOP * s
    section_header_h = BASE_SECTION_HEADER_H * s
    section_rule_h = BASE_SECTION_RULE_H * s
    section_gap = BASE_SECTION_GAP * s
    row_h = BASE_ROW_H * s
    row_gap = BASE_ROW_GAP * s
    side_pad = BASE_SIDE_PAD * s
    input_w = BASE_INPUT_W * s
    input_h = BASE_INPUT_H * s
    checkbox_size = BASE_CHECKBOX_SIZE * s
    btn_w = BASE_BTN_W * s
    btn_h = BASE_BTN_H * s
    scroll_step = BASE_SCROLL_STEP * s
    cat_h = 20 * s  # category sub-header height
    tab_gap = 24 * s

    back_rect = pygame.Rect(side_pad, 8 * s, btn_w, btn_h)
    reset_btn_w = 200 * s

    clock = pygame.time.Clock()

    fullscreen = config.fullscreen
    ui_scale = config.ui_scale
    rs = _RebindState()
    scroll_offset = 0

    font_header = get_pixel_font(32 * s)
    font_label = get_pixel_font(28 * s)
    font_input = get_pixel_font(28 * s)
    font_btn = get_pixel_font(28 * s)
    font_section = get_pixel_font(32 * s)
    font_cat = get_pixel_font(20 * s)

    # Count rows and categories for height calculation.
    n_rows = sum(len(acts) for _, acts in _REBIND_ACTIONS)
    n_cats = len(_REBIND_ACTIONS)

    controls_h = _controls_section_height(n_rows, n_cats, s)
    display_h = (
        section_header_h
        + section_rule_h
        + section_gap
        + 2 * (row_h + row_gap)
        + section_pad_top
    )
    content_h = controls_h + display_h

    def _active_bindings() -> Bindings:
        if rs.active_tab == "keyboard":
            return config.keyboard
        return config.controller

    # Flat list of action keys for focused_row indexing.
    flat_actions: list[str] = [
        action_key for _, actions in _REBIND_ACTIONS for action_key, _ in actions
    ]

    while True:
        scrollable_area = sh - top_bar_h
        max_scroll = max(0, content_h - scrollable_area)
        scroll_offset = max(0, min(scroll_offset, max_scroll))

        # -- Events ---------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return fullscreen, ui_scale

            if event.type == pygame.VIDEORESIZE:
                canvas.handle_resize(event.w, event.h)

            # --- Listening mode: capture input -----------------------
            if rs.listening_action is not None:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        rs.listening_action = None
                    elif rs.active_tab == "keyboard":
                        mods = pygame.key.get_mods()
                        name = event_to_key_name(event.key, mods)
                        bindings = _active_bindings()
                        bindings[rs.listening_action] = [name]
                        rs.listening_action = None
                elif rs.active_tab == "controller":
                    if event.type in (
                        pygame.JOYBUTTONDOWN,
                        pygame.JOYHATMOTION,
                        pygame.JOYAXISMOTION,
                    ):
                        ctrl_name = controller_event_to_name(event)
                        if ctrl_name is not None:
                            bindings = _active_bindings()
                            bindings[rs.listening_action] = [ctrl_name]
                            rs.listening_action = None
                continue  # Consume all events while listening.

            # --- Normal mode -----------------------------------------
            if event.type == pygame.MOUSEWHEEL:
                scroll_offset -= event.y * scroll_step
                scroll_offset = max(
                    0,
                    min(scroll_offset, max_scroll),
                )

            # Keyboard/controller navigation via configured bindings.
            _kb = build_key_lookup(default_keyboard())
            _cl = build_controller_lookup(default_controller())
            nav = resolve_event(event, _kb, _cl)

            _nav_up = PlayerAction.NAV_UP in nav
            _nav_down = PlayerAction.NAV_DOWN in nav
            _nav_confirm = PlayerAction.CONFIRM in nav
            _nav_back = PlayerAction.BACK in nav
            _nav_left = PlayerAction.NAV_LEFT in nav
            _nav_right = PlayerAction.NAV_RIGHT in nav

            # Escape is hardcoded (not in bindings) so handle it too.
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                _nav_back = True

            if _nav_up and flat_actions:
                rs.focused_row = (rs.focused_row - 1) % len(flat_actions)
                # Auto-scroll to keep focused row visible.
                row_y = rs.focused_row * (row_h + row_gap)
                if row_y < scroll_offset:
                    scroll_offset = row_y
                elif row_y + row_h > scroll_offset + scrollable_area:
                    scroll_offset = row_y + row_h - scrollable_area
            elif _nav_down and flat_actions:
                rs.focused_row = (rs.focused_row + 1) % len(flat_actions)
                row_y = rs.focused_row * (row_h + row_gap)
                if row_y < scroll_offset:
                    scroll_offset = row_y
                elif row_y + row_h > scroll_offset + scrollable_area:
                    scroll_offset = row_y + row_h - scrollable_area
            elif _nav_confirm and flat_actions:
                rs.listening_action = flat_actions[rs.focused_row]
            elif _nav_back:
                return fullscreen, ui_scale
            elif _nav_left:
                rs.active_tab = "keyboard"
            elif _nav_right:
                rs.active_tab = "controller"

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = canvas.to_canvas(*event.pos)

                if back_rect.collidepoint(mx, my):
                    return fullscreen, ui_scale

                # Tab buttons (recompute positions to match draw).
                tab_cy = top_bar_h + section_pad_top - scroll_offset + section_header_h
                kb_surf = font_label.render(
                    "Keyboard",
                    False,
                    GOLD,
                )
                kb_rect = pygame.Rect(
                    side_pad,
                    tab_cy,
                    kb_surf.get_width() + 16 * s,
                    row_h,
                )
                ctrl_surf = font_label.render(
                    "Controller",
                    False,
                    GOLD,
                )
                ctrl_rect = pygame.Rect(
                    kb_rect.right + tab_gap,
                    tab_cy,
                    ctrl_surf.get_width() + 16 * s,
                    row_h,
                )
                if kb_rect.collidepoint(mx, my):
                    rs.active_tab = "keyboard"
                elif ctrl_rect.collidepoint(mx, my):
                    rs.active_tab = "controller"

                # Binding boxes.
                bind_cy = tab_cy + row_h + row_gap + section_rule_h + section_gap
                box_x = sw - side_pad - input_w
                for cat_label, actions in _REBIND_ACTIONS:
                    if cat_label is not None:
                        bind_cy += cat_h + row_gap
                    for action_key, _label in actions:
                        box_y = bind_cy + (row_h - input_h) // 2
                        box_rect = pygame.Rect(
                            box_x,
                            box_y,
                            input_w,
                            input_h,
                        )
                        if box_rect.collidepoint(mx, my):
                            rs.listening_action = action_key
                        bind_cy += row_h + row_gap

                # Reset button.
                reset_cy = bind_cy + row_gap
                reset_rect = pygame.Rect(
                    (sw - reset_btn_w) // 2,
                    reset_cy,
                    reset_btn_w,
                    btn_h,
                )
                if reset_rect.collidepoint(mx, my):
                    if rs.active_tab == "keyboard":
                        config.keyboard = default_keyboard()
                    else:
                        config.controller = default_controller()

                # Display: fullscreen checkbox.
                disp_cy = (
                    reset_cy
                    + btn_h
                    + row_gap
                    + section_pad_top
                    + section_header_h
                    + section_rule_h
                    + section_gap
                )
                cb_x = side_pad
                cb_y = disp_cy + (row_h - checkbox_size) // 2
                cb_lbl = font_label.render(
                    "Fullscreen",
                    False,
                    LABEL_COLOR,
                )
                cb_hit = pygame.Rect(
                    cb_x,
                    cb_y,
                    checkbox_size + 8 + cb_lbl.get_width(),
                    checkbox_size,
                )
                if cb_hit.collidepoint(mx, my):
                    fullscreen = not fullscreen
                    if fullscreen:
                        screen = pygame.display.set_mode(
                            (0, 0),
                            pygame.FULLSCREEN,
                        )
                    else:
                        cpx = 1024 * s
                        w, h = calculate_window_size(cpx, cpx)
                        screen = pygame.display.set_mode((w, h))
                    canvas.handle_resize(*screen.get_size())

                # Display: UI scale buttons.
                scale_cy = disp_cy + row_h + row_gap
                scale_lbl = font_label.render(
                    "UI Scale",
                    False,
                    LABEL_COLOR,
                )
                sbtn_w = 60 * s
                sbtn_h2 = row_h - 4 * s
                sbtn_x0 = side_pad + scale_lbl.get_width() + 16 * s
                sbtn_y = scale_cy + (row_h - sbtn_h2) // 2
                for si in range(4):
                    bx = sbtn_x0 + si * (sbtn_w + 4 * s)
                    if pygame.Rect(
                        bx,
                        sbtn_y,
                        sbtn_w,
                        sbtn_h2,
                    ).collidepoint(mx, my):
                        if si != ui_scale:
                            confirmed = _confirm_scale_change(
                                screen,
                                si,
                            )
                            if confirmed:
                                ui_scale = si
                            cur_s = ui_scale if ui_scale > 0 else auto_ui_scale()
                            _theme.apply_scale(cur_s)
                            return fullscreen, ui_scale
                        break

            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return fullscreen, ui_scale

        # -- Drawing --------------------------------------------------
        surf = canvas.surface
        mouse_pos = canvas.to_canvas(*pygame.mouse.get_pos())
        surf.fill(BG)

        draw_button(
            surf,
            back_rect,
            "< Back",
            font_btn,
            back_rect.collidepoint(mouse_pos),
        )

        title_surf = font_header.render("Controls", False, GOLD)
        title_x = (sw - title_surf.get_width()) // 2
        title_y = 8 * s + (btn_h - title_surf.get_height()) // 2
        surf.blit(title_surf, (title_x, title_y))

        content_clip = pygame.Rect(0, top_bar_h, sw, sh - top_bar_h)
        surf.set_clip(content_clip)

        cy = top_bar_h + section_pad_top - scroll_offset

        # -- Controls section header ----------------------------------
        sec_surf = font_section.render("Bindings", False, GOLD)
        surf.blit(sec_surf, (side_pad, cy))
        cy += section_header_h

        # Tab bar.
        for tab_name in ("Keyboard", "Controller"):
            tab_key = tab_name.lower()
            is_active_tab = rs.active_tab == tab_key
            color = GOLD if is_active_tab else LABEL_COLOR
            tab_surf = font_label.render(tab_name, False, color)
            tab_x = (
                side_pad
                if tab_name == "Keyboard"
                else (
                    side_pad
                    + font_label.render("Keyboard", False, GOLD).get_width()
                    + 16 * s
                    + tab_gap
                )
            )
            tab_y = cy + (row_h - tab_surf.get_height()) // 2
            surf.blit(tab_surf, (tab_x, tab_y))
            if is_active_tab:
                underline_y = cy + row_h - 2 * s
                pygame.draw.line(
                    surf,
                    GOLD,
                    (tab_x, underline_y),
                    (tab_x + tab_surf.get_width(), underline_y),
                    2,
                )
        cy += row_h + row_gap

        # Rule below tabs.
        pygame.draw.line(
            surf,
            SECTION_RULE,
            (side_pad, cy),
            (sw - side_pad, cy),
        )
        cy += section_rule_h + section_gap

        # Binding rows.
        bindings = _active_bindings()
        box_x = sw - side_pad - input_w
        flat_idx = 0
        for cat_label, actions in _REBIND_ACTIONS:
            if cat_label is not None:
                cat_surf = font_cat.render(
                    cat_label,
                    False,
                    GOLD,
                )
                cat_y = cy + (cat_h - cat_surf.get_height()) // 2
                surf.blit(cat_surf, (side_pad, cat_y))
                cy += cat_h + row_gap

            for action_key, label in actions:
                is_focused = flat_idx == rs.focused_row
                flat_idx += 1

                # Label.
                lbl_color = GOLD if is_focused else LABEL_COLOR
                lbl_surf = font_label.render(
                    label,
                    False,
                    lbl_color,
                )
                lbl_y = cy + (row_h - lbl_surf.get_height()) // 2
                surf.blit(lbl_surf, (side_pad, lbl_y))

                # Binding box.
                box_y = cy + (row_h - input_h) // 2
                box_rect = pygame.Rect(
                    box_x,
                    box_y,
                    input_w,
                    input_h,
                )
                is_listening = rs.listening_action == action_key
                is_hovered = box_rect.collidepoint(mouse_pos)

                pygame.draw.rect(
                    surf,
                    INPUT_BG,
                    box_rect,
                )
                if is_listening:
                    border_c = GOLD
                elif is_focused:
                    border_c = LABEL_COLOR
                elif is_hovered:
                    border_c = LABEL_COLOR
                else:
                    border_c = INPUT_BORDER
                pygame.draw.rect(surf, border_c, box_rect, 1)

                if is_listening:
                    prompt = (
                        "Press key..."
                        if rs.active_tab == "keyboard"
                        else "Press input..."
                    )
                    txt_surf = font_input.render(
                        prompt,
                        False,
                        GOLD,
                    )
                else:
                    names = bindings.get(action_key, [])
                    display = _format_binding(
                        names,
                        rs.active_tab,
                    )
                    txt_surf = font_input.render(
                        display,
                        False,
                        INPUT_TEXT,
                    )

                # Clip text to box width.
                max_tw = input_w - 8
                if txt_surf.get_width() > max_tw:
                    txt_surf = txt_surf.subsurface(
                        txt_surf.get_width() - max_tw,
                        0,
                        max_tw,
                        txt_surf.get_height(),
                    )
                surf.blit(
                    txt_surf,
                    (
                        box_x + 4,
                        box_y + (input_h - txt_surf.get_height()) // 2,
                    ),
                )

                cy += row_h + row_gap

        # Reset button.
        cy += row_gap
        reset_rect = pygame.Rect(
            (sw - reset_btn_w) // 2,
            cy,
            reset_btn_w,
            btn_h,
        )
        draw_button(
            surf,
            reset_rect,
            "Reset to Defaults",
            font_btn,
            reset_rect.collidepoint(mouse_pos),
        )
        cy += btn_h + row_gap + section_pad_top

        # -- Display section ------------------------------------------
        sec_surf = font_section.render("Display", False, GOLD)
        surf.blit(sec_surf, (side_pad, cy))
        cy += section_header_h
        pygame.draw.line(
            surf,
            SECTION_RULE,
            (side_pad, cy),
            (sw - side_pad, cy),
        )
        cy += section_rule_h + section_gap

        cb_x = side_pad
        cb_y = cy + (row_h - checkbox_size) // 2
        draw_checkbox(
            surf,
            cb_x,
            cb_y,
            checkbox_size,
            fullscreen,
            pygame.Rect(
                cb_x,
                cb_y,
                checkbox_size,
                checkbox_size,
            ).collidepoint(mouse_pos),
        )
        lbl_surf = font_label.render(
            "Fullscreen",
            False,
            LABEL_COLOR,
        )
        surf.blit(
            lbl_surf,
            (
                cb_x + checkbox_size + 8,
                cy + (row_h - lbl_surf.get_height()) // 2,
            ),
        )
        cy += row_h + row_gap

        # UI Scale selector.
        scale_labels = ["Auto", "1x", "2x", "3x"]
        scale_lbl = font_label.render(
            "UI Scale",
            False,
            LABEL_COLOR,
        )
        surf.blit(
            scale_lbl,
            (
                side_pad,
                cy + (row_h - scale_lbl.get_height()) // 2,
            ),
        )
        sbtn_w = 60 * s
        sbtn_h2 = row_h - 4 * s
        sbtn_x0 = side_pad + scale_lbl.get_width() + 16 * s
        sbtn_y = cy + (row_h - sbtn_h2) // 2
        for si, sl in enumerate(scale_labels):
            bx = sbtn_x0 + si * (sbtn_w + 4 * s)
            is_active_scale = si == ui_scale
            bg = (80, 75, 50) if is_active_scale else (40, 40, 45)
            pygame.draw.rect(
                surf,
                bg,
                (bx, sbtn_y, sbtn_w, sbtn_h2),
            )
            border_c = GOLD if is_active_scale else (70, 70, 70)
            pygame.draw.rect(
                surf,
                border_c,
                (bx, sbtn_y, sbtn_w, sbtn_h2),
                2,
            )
            st = font_label.render(sl, False, LABEL_COLOR)
            surf.blit(
                st,
                (
                    bx + (sbtn_w - st.get_width()) // 2,
                    sbtn_y + (sbtn_h2 - st.get_height()) // 2,
                ),
            )
        cy += row_h + row_gap + section_pad_top

        surf.set_clip(None)
        draw_scrollbar(
            surf,
            scroll_offset,
            max_scroll,
            content_h,
            top_bar_h,
        )

        canvas.present(screen)
        clock.tick(FPS)
