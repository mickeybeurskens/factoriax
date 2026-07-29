"""Reusable game UI: menus, hotbar, click regions, and event dispatch.

Extracted from :mod:`factoriax.playground.play.main` so that both the play loop
and the debugger can share the same input handling and menu rendering
without duplicating code. GameUI owns a :class:`PlayState` internally
and translates pygame events into :class:`GameUIResult` values that
the caller can act on.

GameUI does NOT step the environment, manage the pygame window, or
handle recording. Those responsibilities remain with the caller.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pygame

from factoriax.engine.actions import (
    ITEM_TO_CRAFT_ACTION,
    ITEM_TO_DEPOSIT_ACTION,
    ITEM_TO_PLACE_ACTION,
    NO_ACTION,
)
from factoriax.engine.constants import (
    NUM_ITEM_TYPES,
    PLACEABLE_ITEM_LIST,
    ROTATE_BASE,
    Action,
    Direction,
    ItemType,
    Machine,
)
from factoriax.engine.jax_renderer import JaxRenderer
from factoriax.engine.state import EnvParams, EnvState
from factoriax.playground.config import (
    ControllerLookup,
    KeyLookup,
    PlayerAction,
    resolve_controller_button,
    resolve_controller_hat,
    resolve_key,
)
from factoriax.playground.play.achievements import FREE_PLAY_ACHIEVEMENTS
from factoriax.playground.play.play_state import PlayState
from factoriax.playground.play.ui import (
    _entity_inventory,
    _hotbar_h,  # noqa: F401 — re-export
    render_achievement_menu,
    render_help_overlay,
    render_hotbar,
    render_info_panel,
    render_inventory_menu,
    render_machine_menu,
    render_pause_menu,
    render_victory_screen,
)
from factoriax.playground.ui.compositing import composite_rgba_over_rgb
from factoriax.playground.ui.primitives import ClickRegion, hit_test_regions

# Module-level renderer cache keyed by tile size. Play windows resize
# the render at runtime; cache instances so each tile size pays the
# atlas-build + JIT-compile cost once.
_RENDERER_CACHE: dict[int, JaxRenderer] = {}


def _renderer_for(tile_px: int) -> JaxRenderer:
    """

    Parameters
    ----------
    tile_px : int :

    tile_px: int :


    Returns
    -------


    """
    renderer = _RENDERER_CACHE.get(tile_px)
    if renderer is None:
        renderer = JaxRenderer(tile_px=tile_px)
        _RENDERER_CACHE[tile_px] = renderer
    return renderer


_PLACEABLE_ITEM_SET: frozenset[int] = frozenset(PLACEABLE_ITEM_LIST)

# Direction -> next direction one step clockwise, indexed by Direction
# value (1-indexed: LEFT=1..DOWN=4; index 0 is unused). Drives the
# rotate-machine play action; lives here as it is play-UI-only wiring.
TURN_RIGHT_MAP = jnp.array([0, 3, 4, 2, 1], dtype=jnp.int32)

# Machines the play hotbar offers, in palette order (SPLITTER / CROSSING are
# intentionally not hand-placeable from the UI). The palette is the policy;
# the PLACE action ids come from factoriax.engine.actions so they track the enum.
_PLACE_PALETTE: tuple[ItemType, ...] = (
    ItemType.MINER,
    ItemType.PALLET,
    ItemType.CONVEYOR_BELT,
    ItemType.ASSEMBLER,
    ItemType.ARM,
    ItemType.ROCKET,
    ItemType.FURNACE,
    ItemType.SCIENCE_LAB,
)
_ITEM_TO_PLACE_ACTION: dict[int, int] = {
    int(it): int(ITEM_TO_PLACE_ACTION[it]) for it in _PLACE_PALETTE
}

# Maps PlayerAction movement names to (Direction, move_Action, face_Action).
_MOVE_TO_DIR: dict[str, tuple[int, int, int]] = {
    PlayerAction.MOVE_UP: (Direction.UP, int(Action.UP), int(Action.FACE_UP)),
    PlayerAction.MOVE_DOWN: (
        Direction.DOWN,
        int(Action.DOWN),
        int(Action.FACE_DOWN),
    ),
    PlayerAction.MOVE_LEFT: (
        Direction.LEFT,
        int(Action.LEFT),
        int(Action.FACE_LEFT),
    ),
    PlayerAction.MOVE_RIGHT: (
        Direction.RIGHT,
        int(Action.RIGHT),
        int(Action.FACE_RIGHT),
    ),
}

_SLOT_ACTIONS: dict[str, int] = {
    PlayerAction.SLOT_1: int(ItemType.MINER),
    PlayerAction.SLOT_2: int(ItemType.PALLET),
    PlayerAction.SLOT_3: int(ItemType.CONVEYOR_BELT),
    PlayerAction.SLOT_4: int(ItemType.ASSEMBLER),
    PlayerAction.SLOT_5: int(ItemType.ARM),
    PlayerAction.SLOT_6: int(ItemType.ROCKET),
}

_PLAYER_ACTIONS: dict[str, int] = {
    PlayerAction.SELECT_PLAYER_1: 0,
    PlayerAction.SELECT_PLAYER_2: 1,
    PlayerAction.SELECT_PLAYER_3: 2,
    PlayerAction.SELECT_PLAYER_4: 3,
    PlayerAction.SELECT_PLAYER_5: 4,
    PlayerAction.SELECT_PLAYER_6: 5,
    PlayerAction.SELECT_PLAYER_7: 6,
    PlayerAction.SELECT_PLAYER_8: 7,
    PlayerAction.SELECT_PLAYER_9: 8,
}


@dataclasses.dataclass
class GameUIResult:
    """Result of processing a single pygame event through GameUI."""

    action: int | None = None
    state: EnvState | None = None
    quit: bool = False
    reset: bool = False


def _tile_in_front(state: EnvState, player_idx: int) -> tuple[int, int]:
    """Return the (x, y) tile immediately in front of a player.

    Parameters
    ----------
    state :
        Current environment state
    player_idx :
        Index of the player
    state : EnvState :

    player_idx : int :

    state: EnvState :

    player_idx: int :


    Returns
    -------


    """
    pos = np.array(state.player_positions[player_idx])
    direction = int(state.player_directions[player_idx])
    offsets: dict[int, tuple[int, int]] = {
        int(Direction.LEFT): (-1, 0),
        int(Direction.RIGHT): (1, 0),
        int(Direction.UP): (0, -1),
        int(Direction.DOWN): (0, 1),
    }
    dx, dy = offsets.get(direction, (0, 0))
    return int(pos[0]) + dx, int(pos[1]) + dy


class GameUI:
    """Reusable game UI component for menus, hotbar, and input dispatch.

    Owns a :class:`PlayState` internally. Callers feed it pygame events
    and the current :class:`EnvState`; it returns rendered overlays and
    action intents via :class:`GameUIResult`.

    Parameters
    ----------

    Parameters
    ----------
    kb_lookup :
        Key lookup table from
    welcome_open :
        Whether to show the welcome screen initially

    Returns
    -------


    """

    def __init__(
        self,
        params: EnvParams,
        kb_lookup: KeyLookup,
        *,
        ctrl_lookup: ControllerLookup | None = None,
        welcome_open: bool = True,
    ) -> None:
        self._ps = PlayState(welcome_open=welcome_open)
        self._kb_lookup = kb_lookup
        self._ctrl_lookup = ctrl_lookup
        self._params = params
        self._win_ox = 0
        self._win_oy = 0
        self._win_scale = 1

    @property
    def play_state(self) -> PlayState:
        """Access the internal PlayState for reading UI flags."""
        return self._ps

    def set_window_transform(
        self,
        win_ox: int,
        win_oy: int,
        win_scale: int,
    ) -> None:
        """Update the window-to-canvas coordinate transform.

        Parameters
        ----------
        win_ox :
            Window X offset of the canvas origin.
        win_oy :
            Window Y offset of the canvas origin.
        win_scale :
            Integer scale factor from canvas to window pixels.
        win_ox : int :

        win_oy : int :

        win_scale : int :

        win_ox: int :

        win_oy: int :

        win_scale: int :


        Returns
        -------


        """
        self._win_ox = win_ox
        self._win_oy = win_oy
        self._win_scale = win_scale

    def has_menu_open(self) -> bool:
        """ """
        ps = self._ps
        return (
            ps.inventory_open
            or ps.achievement_open
            or ps.machine_open
            or ps.pause_open
            or ps.help_open
        )

    def handle_event(
        self,
        event: pygame.event.Event,
        state: EnvState,
    ) -> GameUIResult:
        """Process a single pygame event and return the result.

        The caller is responsible for acting on the result: stepping
        the environment if ``result.action`` is set, resetting if
        ``result.reset`` is True, and quitting if ``result.quit``
        is True.

        Parameters
        ----------
        event :
            Pygame event to process.
        state :
            Current environment state.
        event : pygame.event.Event :

        state : EnvState :

        event: pygame.event.Event :

        state: EnvState :


        Returns
        -------


        """
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self._handle_click(event, state)
        if event.type == pygame.KEYDOWN:
            return self._handle_keydown(event, state)
        if event.type in (
            pygame.JOYBUTTONDOWN,
            pygame.JOYHATMOTION,
        ):
            return self._handle_controller_event(event, state)
        return GameUIResult(state=state)

    def update_hover(self, state: EnvState) -> None:
        """Update the tile highlight to the tile the player faces.

        Call once per frame after event processing.

        Parameters
        ----------
        state :
            Current environment state.
        state : EnvState :

        state: EnvState :


        Returns
        -------


        """
        ftx, fty = _tile_in_front(state, int(state.selected_player))
        map_h = int(state.map.shape[0])
        map_w = int(state.map.shape[1])
        if 0 <= ftx < map_w and 0 <= fty < map_h:
            self._ps.hover_tile_x = ftx
            self._ps.hover_tile_y = fty
        else:
            self._ps.hover_tile_x = -1
            self._ps.hover_tile_y = -1

    def render_frame(
        self,
        state: EnvState,
        ui_w: int,
        ui_h: int,
        tile_px: int,
        world_ox: int,
        world_oy: int,
        achievements: jnp.ndarray | None = None,
    ) -> tuple[np.ndarray, list[ClickRegion]]:
        """Build one UI frame with all active overlays.

        Renders the game world and composites all menu overlays on top.

        Parameters
        ----------
        state :
            Current environment state.
        ui_w :
            UI canvas width.
        ui_h :
            UI canvas height.
        tile_px :
            Tile pixel size.
        world_ox :
            World X offset within canvas.
        world_oy :
            World Y offset within canvas.
        achievements :
            Achievement flags from the wrapper state.
            Required when the achievement menu is open.
        state : EnvState :

        ui_w : int :

        ui_h : int :

        tile_px : int :

        world_ox : int :

        world_oy : int :

        achievements : jnp.ndarray | None :
            (Default value = None)
        state: EnvState :

        ui_w: int :

        ui_h: int :

        tile_px: int :

        world_ox: int :

        world_oy: int :

        achievements: jnp.ndarray | None :
             (Default value = None)

        Returns
        -------


        """
        ps = self._ps
        pixels = np.asarray(_renderer_for(tile_px).jit_render_map(state))
        click_regions: list[ClickRegion] = []

        ui_frame = np.zeros((ui_h, ui_w, 3), dtype=np.uint8)
        ph, pw = pixels.shape[:2]
        ui_frame[world_oy : world_oy + ph, world_ox : world_ox + pw] = pixels

        # Tile highlight.
        htx, hty = ps.hover_tile_x, ps.hover_tile_y
        if htx >= 0 and hty >= 0:
            hx = world_ox + htx * tile_px
            hy = world_oy + hty * tile_px
            hx2 = hx + tile_px
            hy2 = hy + tile_px
            highlight = np.array([255, 255, 255], dtype=np.uint8)
            for row in (hy, hy2 - 1):
                if 0 <= row < ui_h:
                    c0, c1 = max(0, hx), min(ui_w, hx2)
                    ui_frame[row, c0:c1] = ui_frame[row, c0:c1] // 2 + highlight // 2
            for col in (hx, hx2 - 1):
                if 0 <= col < ui_w:
                    r0, r1 = max(0, hy), min(ui_h, hy2)
                    ui_frame[r0:r1, col] = ui_frame[r0:r1, col] // 2 + highlight // 2

        hotbar_overlay, hotbar_regions = render_hotbar(
            state,
            ui_w,
            ui_h,
            ps.selected_item,
            ps.frame_tick,
        )
        composite_rgba_over_rgb(ui_frame, hotbar_overlay)
        click_regions.extend(hotbar_regions)

        info_overlay = render_info_panel(state, ui_w, ui_h, htx, hty)
        composite_rgba_over_rgb(ui_frame, info_overlay)

        if ps.machine_open:
            machine_overlay, machine_regions = render_machine_menu(
                state,
                self._params,
                ui_w,
                ui_h,
                ps.machine_tx,
                ps.machine_ty,
                ps.machine_panel_active,
                ps.selected_item,
                ps.focused_machine_item,
            )
            composite_rgba_over_rgb(ui_frame, machine_overlay)
            click_regions.extend(machine_regions)

        if ps.inventory_open:
            menu_overlay, inv_regions = render_inventory_menu(
                state,
                self._params,
                ui_w,
                ui_h,
                ps.menu_focus,
                ps.held_item,
                ps.selected_recipe,
                ps.selected_item,
            )
            composite_rgba_over_rgb(ui_frame, menu_overlay)
            click_regions.extend(inv_regions)

        if ps.achievement_open and achievements is not None:
            ach_overlay = render_achievement_menu(
                np.asarray(achievements),
                ui_w,
                ui_h,
                ps.achievement_scroll,
                ps.achievement_selection,
            )
            composite_rgba_over_rgb(ui_frame, ach_overlay)

        if ps.pause_open:
            pause_overlay, pause_regions = render_pause_menu(
                ui_w,
                ui_h,
                ps.pause_selection,
            )
            composite_rgba_over_rgb(ui_frame, pause_overlay)
            click_regions.extend(pause_regions)

        if ps.help_open:
            composite_rgba_over_rgb(ui_frame, render_help_overlay(ui_w, ui_h))

        if ps.victory_open:
            composite_rgba_over_rgb(
                ui_frame,
                render_victory_screen(ui_w, ui_h),
            )

        return ui_frame, click_regions

    # ------------------------------------------------------------------
    # Private event handlers
    # ------------------------------------------------------------------

    def _to_canvas(self, window_x: int, window_y: int) -> tuple[int, int]:
        """Transform window pixel coordinates to canvas coordinates.

        Parameters
        ----------
        window_x : int :

        window_y : int :

        window_x: int :

        window_y: int :


        Returns
        -------


        """
        return (
            (window_x - self._win_ox) // self._win_scale,
            (window_y - self._win_oy) // self._win_scale,
        )

    def _handle_click(
        self,
        event: pygame.event.Event,
        state: EnvState,
    ) -> GameUIResult:
        """Process a mouse click against the current click regions.

        Parameters
        ----------
        event : pygame.event.Event :

        state : EnvState :

        event: pygame.event.Event :

        state: EnvState :


        Returns
        -------


        """
        ps = self._ps
        action: int | None = None
        quit_flag = False
        reset_flag = False

        base_x, base_y = self._to_canvas(event.pos[0], event.pos[1])
        hit = hit_test_regions(ps.click_regions, base_x, base_y)

        if hit is not None:
            if hit.action == "select_slot":
                ps.selected_item = hit.param
                if ps.machine_open:
                    ps.machine_panel_active = False
            elif hit.action == "cycle_prev":
                items = PLACEABLE_ITEM_LIST
                idx = items.index(ps.selected_item) if ps.selected_item in items else 0
                ps.selected_item = items[(idx - 1) % len(items)]
            elif hit.action == "cycle_next":
                items = PLACEABLE_ITEM_LIST
                idx = items.index(ps.selected_item) if ps.selected_item in items else -1
                ps.selected_item = items[(idx + 1) % len(items)]
            elif hit.action == "place_selected":
                place_action = _ITEM_TO_PLACE_ACTION.get(
                    ps.selected_item,
                )
                if place_action is not None:
                    action = place_action
            elif hit.action == "select_recipe":
                ps.selected_recipe = hit.param
            elif hit.action == "select_machine_slot":
                ps.focused_machine_item = hit.param
                ps.machine_panel_active = True
            elif hit.action == "pause_option":
                if hit.param == 0:
                    ps.pause_open = False
                elif hit.param == 1:
                    ps.help_open = True
                    ps.pause_open = False
                elif hit.param == 2:
                    ps.pause_open = False
                    reset_flag = True
                else:
                    quit_flag = True

        return GameUIResult(
            action=action,
            state=state,
            quit=quit_flag,
            reset=reset_flag,
        )

    def _handle_keydown(
        self,
        event: pygame.event.Event,
        state: EnvState,
    ) -> GameUIResult:
        """Dispatch a KEYDOWN event to the appropriate context handler.

        Parameters
        ----------
        event : pygame.event.Event :

        state : EnvState :

        event: pygame.event.Event :

        state: EnvState :


        Returns
        -------


        """
        ps = self._ps

        if ps.help_open:
            ps.help_open = False
            return GameUIResult(state=state)

        if event.key == pygame.K_ESCAPE:
            return self._dispatch_actions(
                frozenset(),
                state,
                is_escape=True,
            )

        mods = pygame.key.get_mods()
        actions = resolve_key(self._kb_lookup, event.key, mods)
        if not actions:
            return GameUIResult(state=state)

        return self._dispatch_actions(actions, state)

    def _dispatch_actions(
        self,
        actions: frozenset[str],
        state: EnvState,
        *,
        is_escape: bool = False,
    ) -> GameUIResult:
        """Route a set of player actions through the context dispatch.

        Shared by keyboard and controller event handlers. The
        ``is_escape`` flag triggers the universal back/pause logic
        (closing the topmost open menu, or opening the pause menu).

        Parameters
        ----------
        actions :
            Resolved player action names.
        state :
            Current environment state.
        is_escape :
            Whether to apply escape/back logic.
        actions : frozenset[str] :

        state : EnvState :

        * :

        is_escape : bool :
            (Default value = False)
        actions: frozenset[str] :

        state: EnvState :

        is_escape: bool :
             (Default value = False)

        Returns
        -------


        """
        ps = self._ps
        action: int | None = None
        quit_flag = False
        reset_flag = False

        if is_escape:
            if ps.pause_open:
                ps.pause_open = False
            elif ps.inventory_open:
                ps.inventory_open = False
                ps.held_item = None
            elif ps.achievement_open:
                ps.achievement_open = False
            elif ps.machine_open:
                ps.machine_open = False
            else:
                ps.pause_open = True
                ps.pause_selection = 0
            return GameUIResult(state=state)

        if ps.pause_open:
            state, reset_flag, quit_flag = self._handle_pause_keys(
                actions,
                state,
            )
        elif PlayerAction.OPEN_MACHINE in actions:
            state = self._handle_machine_toggle(state)
        elif ps.machine_open:
            state, action = self._handle_machine_keys(actions, state)
        elif PlayerAction.TOGGLE_HOTBAR in actions:
            state = self._handle_assembler_recipe_or_hotbar(state)
        elif PlayerAction.OPEN_INVENTORY in actions:
            ps.inventory_open = not ps.inventory_open
            if ps.inventory_open:
                ps.achievement_open = False
                ps.machine_open = False
        elif PlayerAction.OPEN_ACHIEVEMENTS in actions:
            ps.achievement_open = not ps.achievement_open
            if ps.achievement_open:
                ps.inventory_open = False
                ps.achievement_scroll = 0
                ps.achievement_selection = 0
        elif ps.achievement_open:
            self._handle_achievement_keys(actions)
        elif ps.inventory_open:
            action = self._handle_crafting_nav(actions)
        else:
            state, action = self._handle_world_keys(actions, state)

        return GameUIResult(
            action=action,
            state=state,
            quit=quit_flag,
            reset=reset_flag,
        )

    def _handle_controller_event(
        self,
        event: pygame.event.Event,
        state: EnvState,
    ) -> GameUIResult:
        """Dispatch a controller button or hat event.

        Parameters
        ----------
        event : pygame.event.Event :

        state : EnvState :

        event: pygame.event.Event :

        state: EnvState :


        Returns
        -------


        """
        if self._ctrl_lookup is None:
            return GameUIResult(state=state)

        ps = self._ps
        lookup = self._ctrl_lookup

        if ps.help_open:
            if event.type == pygame.JOYBUTTONDOWN:
                ps.help_open = False
            return GameUIResult(state=state)

        actions: frozenset[str] = frozenset()
        if event.type == pygame.JOYBUTTONDOWN:
            actions = resolve_controller_button(lookup, event.button)
        elif event.type == pygame.JOYHATMOTION:
            actions = resolve_controller_hat(
                lookup,
                event.hat,
                event.value,
            )

        if not actions:
            return GameUIResult(state=state)

        is_escape = PlayerAction.BACK in actions
        return self._dispatch_actions(actions, state, is_escape=is_escape)

    def _handle_pause_keys(
        self,
        actions: frozenset[str],
        state: EnvState,
    ) -> tuple[EnvState, bool, bool]:
        """Handle keys while pause menu is open.

        Parameters
        ----------
        actions : frozenset[str] :

        state : EnvState :

        actions: frozenset[str] :

        state: EnvState :


        Returns
        -------


        """
        ps = self._ps
        reset_flag = False
        quit_flag = False
        if PlayerAction.NAV_UP in actions:
            ps.pause_selection = max(0, ps.pause_selection - 1)
        elif PlayerAction.NAV_DOWN in actions:
            ps.pause_selection = min(3, ps.pause_selection + 1)
        elif PlayerAction.CONFIRM in actions:
            if ps.pause_selection == 0:
                ps.pause_open = False
            elif ps.pause_selection == 1:
                ps.help_open = True
                ps.pause_open = False
            elif ps.pause_selection == 2:
                ps.pause_open = False
                reset_flag = True
            else:
                quit_flag = True
        return state, reset_flag, quit_flag

    def _handle_machine_toggle(self, state: EnvState) -> EnvState:
        """Toggle the machine inspection menu.

        Parameters
        ----------
        state : EnvState :

        state: EnvState :


        Returns
        -------


        """
        ps = self._ps
        if ps.machine_open:
            ps.machine_open = False
        else:
            selected_player = int(state.selected_player)
            tx, ty = _tile_in_front(state, selected_player)
            map_h, map_w = state.map.shape
            if (
                0 <= tx < map_w
                and 0 <= ty < map_h
                and int(state.machine_types[ty, tx]) != int(Machine.NONE)
            ):
                ps.machine_tx, ps.machine_ty = tx, ty
                ps.machine_open = True
                ps.machine_panel_active = True
                ps.inventory_open = False
                ps.achievement_open = False
                ps.pause_open = False
        return state

    def _handle_machine_keys(
        self,
        actions: frozenset[str],
        state: EnvState,
    ) -> tuple[EnvState, int | None]:
        """Handle keys while machine menu is open.

        Parameters
        ----------
        actions : frozenset[str] :

        state : EnvState :

        actions: frozenset[str] :

        state: EnvState :


        Returns
        -------


        """
        ps = self._ps
        action: int | None = None
        if PlayerAction.NAV_UP in actions or PlayerAction.NAV_DOWN in actions:
            ps.machine_panel_active = not ps.machine_panel_active
        elif PlayerAction.NAV_LEFT in actions or PlayerAction.NAV_RIGHT in actions:
            is_left = PlayerAction.NAV_LEFT in actions
            if ps.machine_panel_active:
                # Cycle focused_machine_item through non-empty types
                # in the machine's inventory.
                machine_inv = _entity_inventory(
                    state,
                    ps.machine_ty,
                    ps.machine_tx,
                )
                active = [
                    i for i in range(1, NUM_ITEM_TYPES) if int(machine_inv[i]) > 0
                ]
                if active and ps.focused_machine_item in active:
                    idx = active.index(ps.focused_machine_item)
                    idx = (idx + (-1 if is_left else 1)) % len(active)
                    ps.focused_machine_item = active[idx]
                elif active:
                    ps.focused_machine_item = active[0]
            else:
                delta = -1 if is_left else 1
                current = ps.selected_item
                new_item = ((current - 1 + delta) % (NUM_ITEM_TYPES - 1)) + 1
                ps.selected_item = new_item
        elif PlayerAction.CONFIRM in actions:
            if ps.machine_panel_active:
                # Engine withdraws whatever is in the target machine's
                # output slot — item focus is purely a UI affordance.
                action = int(Action.WITHDRAW)
            else:
                # Deposit the selected item (the guard only skips EMPTY).
                deposit = int(ITEM_TO_DEPOSIT_ACTION[ps.selected_item])
                if deposit != NO_ACTION:
                    action = deposit
        elif PlayerAction.CYCLE_RECIPE in actions:
            state = self._handle_assembler_recipe_or_hotbar(state)
        return state, action

    def _handle_assembler_recipe_or_hotbar(
        self,
        state: EnvState,
    ) -> EnvState:
        """Handle Q key (no-op, assemblers auto-detect recipes).

        Retained as a stub so the dispatch table entry and key binding
        continue to resolve without error.

        Parameters
        ----------
        state : EnvState :

        state: EnvState :


        Returns
        -------


        """
        return state

    def _handle_achievement_keys(
        self,
        actions: frozenset[str],
    ) -> None:
        """Handle keys in the achievement menu.

        Parameters
        ----------
        actions : frozenset[str] :

        actions: frozenset[str] :


        Returns
        -------


        """
        ps = self._ps
        row_h = 36
        if PlayerAction.NAV_UP in actions:
            ps.achievement_selection = max(0, ps.achievement_selection - 1)
        elif PlayerAction.NAV_DOWN in actions:
            ps.achievement_selection = min(
                len(FREE_PLAY_ACHIEVEMENTS) - 1,
                ps.achievement_selection + 1,
            )
        sel_top = ps.achievement_selection * row_h
        sel_bot = sel_top + row_h
        if sel_top < ps.achievement_scroll:
            ps.achievement_scroll = sel_top
        elif sel_bot > ps.achievement_scroll + 8 * row_h:
            ps.achievement_scroll = sel_bot - 8 * row_h

    def _handle_crafting_nav(
        self,
        actions: frozenset[str],
    ) -> int | None:
        """Handle keys in the crafting panel.

        Parameters
        ----------
        actions : frozenset[str] :

        actions: frozenset[str] :


        Returns
        -------


        """
        ps = self._ps
        table = self._params.recipe_table
        num_recipes = int(table.outputs.shape[0])
        if PlayerAction.NAV_UP in actions:
            ps.selected_recipe = (ps.selected_recipe - 1) % num_recipes
        elif PlayerAction.NAV_DOWN in actions:
            ps.selected_recipe = (ps.selected_recipe + 1) % num_recipes
        elif PlayerAction.CONFIRM in actions:
            # Same order the craft panel lists.
            output = int(table.outputs[ps.selected_recipe])
            return int(ITEM_TO_CRAFT_ACTION[output])
        return None

    def _handle_world_keys(
        self,
        actions: frozenset[str],
        state: EnvState,
    ) -> tuple[EnvState, int | None]:
        """Handle keys in the world (no menu open).

        Parameters
        ----------
        actions : frozenset[str] :

        state : EnvState :

        actions: frozenset[str] :

        state: EnvState :


        Returns
        -------


        """
        action: int | None = None
        params = self._params

        if PlayerAction.INTERACT in actions:
            action = _handle_world_interact(
                state,
                self._ps.selected_item,
            )
        elif PlayerAction.ROTATE in actions:
            action = _rotate_action_for_tile(state)
        elif PlayerAction.MINE in actions:
            action = int(Action.MINE)
        else:
            # Player selection.
            player_match = actions & _PLAYER_ACTIONS.keys()
            if player_match:
                player_name = next(iter(player_match))
                player_idx = _PLAYER_ACTIONS[player_name]
                if player_idx < state.player_positions.shape[0]:
                    state = state.replace(selected_player=player_idx)
                return state, action

            # Item type selection via number keys.
            slot_match = actions & _SLOT_ACTIONS.keys()
            if slot_match:
                slot_name = next(iter(slot_match))
                item_type = _SLOT_ACTIONS[slot_name]
                if item_type < NUM_ITEM_TYPES:
                    self._ps.selected_item = item_type
                return state, action

            # Movement (face-then-move).
            for move_action, (want_dir, move_act, face_act) in _MOVE_TO_DIR.items():
                if move_action in actions:
                    sel = int(state.selected_player)
                    facing = int(state.player_directions[sel])
                    action = move_act if facing == want_dir else face_act
                    break

        return state, action


def _rotate_action_for_tile(state: EnvState) -> int:
    """Compute the ROTATE_* action to cycle a machine clockwise.

    Looks up the machine in front of the player, reads its current
    direction, advances it one step clockwise via TURN_RIGHT_MAP,
    and returns the corresponding absolute ROTATE_* action.

    Parameters
    ----------
    state :
        Current environment state.
    state : EnvState :

    state: EnvState :


    Returns
    -------


    """
    selected_player = int(state.selected_player)
    tx, ty = _tile_in_front(state, selected_player)
    map_h, map_w = state.map.shape
    if not (0 <= tx < map_w and 0 <= ty < map_h):
        return int(Action.NOOP)
    if int(state.machine_types[ty, tx]) == int(Machine.NONE):
        return int(Action.NOOP)
    eidx = int(state.tile_entity[ty, tx])
    if eidx < 0:
        return int(Action.NOOP)
    cur_dir = int(state.ent_direction[eidx])
    next_dir = int(TURN_RIGHT_MAP[cur_dir])
    # Direction values are 1-indexed (LEFT=1..DOWN=4),
    # ROTATE_* offsets are 0-indexed (LEFT=0..DOWN=3).
    return ROTATE_BASE + next_dir - 1


def _handle_world_interact(
    state: EnvState,
    selected_item: int,
) -> int | None:
    """Determine action for the interact key (pickup or place).

    If the tile in front of the player has a machine, returns PICKUP.
    Otherwise, returns the placement action for the currently selected
    machine, or ``None`` if no placeable machine is selected.

    Parameters
    ----------
    state :
        Current environment state.
    selected_item :
        Currently selected item type from the hotbar.
    state : EnvState :

    selected_item : int :

    state: EnvState :

    selected_item: int :


    Returns
    -------


    """
    selected_player = int(state.selected_player)
    tx, ty = _tile_in_front(state, selected_player)
    map_h, map_w = state.map.shape
    has_machine = (
        0 <= tx < map_w
        and 0 <= ty < map_h
        and int(state.machine_types[ty, tx]) != int(Machine.NONE)
    )
    if has_machine:
        return int(Action.PICKUP)
    return _ITEM_TO_PLACE_ACTION.get(selected_item)
