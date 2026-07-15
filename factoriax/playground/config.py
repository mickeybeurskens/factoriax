"""Player configuration: persistent settings and remappable input bindings.

Stores environment parameters and input bindings (keyboard + controller)
in a JSON file at the project root. The binding system uses a flat map of
:class:`PlayerAction` names to physical key names. A thin resolver
converts these into O(1) lookup dicts at startup.

The separation of concerns is:

- **What** a key does = the binding map (data, configurable).
- **When** it applies = context logic in the play loop (code).
- **How** it executes = handler functions (code).

This means the same physical key (e.g. W) can map to both ``move_up``
and ``nav_up``. The play loop checks which one is relevant based on
which menu is open.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import orjson
import pygame

from factoriax.engine.state import EnvParams

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("player_config.json")

# Modifier flag constants used in key lookup tuples.
_MOD_NONE = 0
_MOD_SHIFT = pygame.KMOD_SHIFT
_MOD_CTRL = pygame.KMOD_CTRL
_MOD_MASK = _MOD_SHIFT | _MOD_CTRL


# ---------------------------------------------------------------------------
# PlayerAction
# ---------------------------------------------------------------------------


class PlayerAction(StrEnum):
    """Every player intention, decoupled from pygame keys and JAX actions.

    Gameplay actions matter in the world. Navigation actions matter in
    menus and as toggle keys. Both groups share the same binding map
    so a physical key can map to one action from each group.
    """

    # -- Gameplay (world context) -------------------------------------------
    MOVE_UP = "move_up"
    MOVE_DOWN = "move_down"
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"
    MINE = "mine"
    INTERACT = "interact"
    ROTATE = "rotate"

    # Direct slot selection (1-indexed to match keyboard labels).
    SLOT_1 = "slot_1"
    SLOT_2 = "slot_2"
    SLOT_3 = "slot_3"
    SLOT_4 = "slot_4"
    SLOT_5 = "slot_5"
    SLOT_6 = "slot_6"
    SLOT_7 = "slot_7"
    SLOT_8 = "slot_8"
    SLOT_9 = "slot_9"
    SLOT_10 = "slot_10"

    # Player selection in multiplayer.
    SELECT_PLAYER_1 = "select_player_1"
    SELECT_PLAYER_2 = "select_player_2"
    SELECT_PLAYER_3 = "select_player_3"
    SELECT_PLAYER_4 = "select_player_4"
    SELECT_PLAYER_5 = "select_player_5"
    SELECT_PLAYER_6 = "select_player_6"
    SELECT_PLAYER_7 = "select_player_7"
    SELECT_PLAYER_8 = "select_player_8"
    SELECT_PLAYER_9 = "select_player_9"

    # -- Navigation (menus and toggles) -------------------------------------
    NAV_UP = "nav_up"
    NAV_DOWN = "nav_down"
    NAV_LEFT = "nav_left"
    NAV_RIGHT = "nav_right"
    CONFIRM = "confirm"
    BACK = "back"
    QUIT = "quit"
    CLEAR_BINDING = "clear_binding"

    OPEN_INVENTORY = "open_inventory"
    OPEN_ACHIEVEMENTS = "open_achievements"
    OPEN_MACHINE = "open_machine"
    OPEN_HELP = "open_help"
    TOGGLE_HOTBAR = "toggle_hotbar"
    CYCLE_RECIPE = "cycle_recipe"


# ---------------------------------------------------------------------------
# Bindings type and PlayerConfig
# ---------------------------------------------------------------------------

# action name -> list of physical key names (e.g. "K_w", "SHIFT+K_1")
Bindings = dict[str, list[str]]

# Reverse lookup: (modifier_flags, pygame_key_int) -> frozenset of actions
KeyLookup = dict[tuple[int, int], frozenset[str]]


@dataclass
class PlayerConfig:
    """Persistent player configuration."""

    env_params: dict[str, int | float] = field(default_factory=dict)
    seed: int = 42
    keyboard: Bindings = field(default_factory=dict)
    controller: Bindings = field(default_factory=dict)
    fullscreen: bool = False
    ui_scale: int = 0


# ---------------------------------------------------------------------------
# Default bindings
# ---------------------------------------------------------------------------


def default_keyboard() -> Bindings:
    """Return default keyboard bindings matching the original hardcoded keys."""
    return {
        # Gameplay: movement
        PlayerAction.MOVE_UP: ["K_w", "K_UP"],
        PlayerAction.MOVE_DOWN: ["K_s", "K_DOWN"],
        PlayerAction.MOVE_LEFT: ["K_a", "K_LEFT"],
        PlayerAction.MOVE_RIGHT: ["K_d", "K_RIGHT"],
        # Gameplay: actions
        PlayerAction.MINE: ["K_SPACE"],
        PlayerAction.INTERACT: ["K_e"],
        PlayerAction.ROTATE: ["K_r"],
        # Gameplay: slots (0-indexed internally, 1-indexed names)
        PlayerAction.SLOT_1: ["K_1"],
        PlayerAction.SLOT_2: ["K_2"],
        PlayerAction.SLOT_3: ["K_3"],
        PlayerAction.SLOT_4: ["K_4"],
        PlayerAction.SLOT_5: ["K_5"],
        PlayerAction.SLOT_6: ["K_6"],
        PlayerAction.SLOT_7: ["K_7"],
        PlayerAction.SLOT_8: ["K_8"],
        PlayerAction.SLOT_9: ["SHIFT+K_1"],
        PlayerAction.SLOT_10: ["SHIFT+K_2"],
        # Gameplay: player selection
        PlayerAction.SELECT_PLAYER_1: ["CTRL+K_1"],
        PlayerAction.SELECT_PLAYER_2: ["CTRL+K_2"],
        PlayerAction.SELECT_PLAYER_3: ["CTRL+K_3"],
        PlayerAction.SELECT_PLAYER_4: ["CTRL+K_4"],
        PlayerAction.SELECT_PLAYER_5: ["CTRL+K_5"],
        PlayerAction.SELECT_PLAYER_6: ["CTRL+K_6"],
        PlayerAction.SELECT_PLAYER_7: ["CTRL+K_7"],
        PlayerAction.SELECT_PLAYER_8: ["CTRL+K_8"],
        PlayerAction.SELECT_PLAYER_9: ["CTRL+K_9"],
        # Navigation: menu movement (same physical keys as gameplay movement)
        PlayerAction.NAV_UP: ["K_w", "K_UP"],
        PlayerAction.NAV_DOWN: ["K_s", "K_DOWN"],
        PlayerAction.NAV_LEFT: ["K_a", "K_LEFT"],
        PlayerAction.NAV_RIGHT: ["K_d", "K_RIGHT"],
        PlayerAction.CONFIRM: ["K_RETURN", "K_e"],
        PlayerAction.BACK: ["K_BACKSPACE"],
        PlayerAction.QUIT: ["K_ESCAPE"],
        PlayerAction.CLEAR_BINDING: ["K_DELETE"],
        # Navigation: toggles
        PlayerAction.OPEN_INVENTORY: ["K_i"],
        PlayerAction.OPEN_ACHIEVEMENTS: ["K_p"],
        PlayerAction.OPEN_MACHINE: ["K_f"],
        PlayerAction.OPEN_HELP: ["K_QUESTION", "SHIFT+K_SLASH"],
        PlayerAction.TOGGLE_HOTBAR: ["K_q"],
        PlayerAction.CYCLE_RECIPE: ["K_q"],
    }


def default_controller() -> Bindings:
    """Return default controller bindings for a standard gamepad.

    Button numbering follows SDL2 game controller layout:
    0=A/Cross, 1=B/Circle, 2=X/Square, 3=Y/Triangle,
    4=LB, 5=RB, 6=Back/Select, 7=Start, 8=L3, 9=R3.

    Axis convention: AXIS_{n}_POS for positive deflection,
    AXIS_{n}_NEG for negative. Left stick = axes 0 (X) and 1 (Y).
    D-pad = HAT_0_{UP,DOWN,LEFT,RIGHT}.
    """
    return {
        # Movement via left stick
        PlayerAction.MOVE_UP: ["AXIS_1_NEG"],
        PlayerAction.MOVE_DOWN: ["AXIS_1_POS"],
        PlayerAction.MOVE_LEFT: ["AXIS_0_NEG"],
        PlayerAction.MOVE_RIGHT: ["AXIS_0_POS"],
        # Actions
        PlayerAction.MINE: ["BUTTON_2"],
        PlayerAction.INTERACT: ["BUTTON_0"],
        PlayerAction.ROTATE: ["BUTTON_3"],
        # Navigation via D-pad
        PlayerAction.NAV_UP: ["HAT_0_UP"],
        PlayerAction.NAV_DOWN: ["HAT_0_DOWN"],
        PlayerAction.NAV_LEFT: ["HAT_0_LEFT"],
        PlayerAction.NAV_RIGHT: ["HAT_0_RIGHT"],
        PlayerAction.CONFIRM: ["BUTTON_0"],
        PlayerAction.BACK: ["BUTTON_1"],
        PlayerAction.QUIT: [],
        PlayerAction.CLEAR_BINDING: [],
        # Toggles
        PlayerAction.OPEN_INVENTORY: ["BUTTON_3"],
        PlayerAction.TOGGLE_HOTBAR: ["BUTTON_4"],
        PlayerAction.CYCLE_RECIPE: ["BUTTON_4"],
        PlayerAction.OPEN_MACHINE: ["BUTTON_9"],
        PlayerAction.OPEN_ACHIEVEMENTS: [],
        PlayerAction.OPEN_HELP: ["BUTTON_6"],
    }


# ---------------------------------------------------------------------------
# Key name parsing
# ---------------------------------------------------------------------------


def _parse_key_name(name: str) -> tuple[int, int]:
    """Parse a key name like ``"K_w"`` or ``"SHIFT+K_1"`` into (mods, key).

    Parameters
    ----------
    name :
        Key name string from the binding config.
    """
    parts = name.split("+")
    mods = _MOD_NONE
    key_part = parts[-1]

    for mod_str in parts[:-1]:
        upper = mod_str.strip().upper()
        if upper == "SHIFT":
            mods |= _MOD_SHIFT
        elif upper == "CTRL":
            mods |= _MOD_CTRL
        else:
            raise ValueError(f"Unknown modifier: {mod_str!r} in {name!r}")

    key_int = getattr(pygame, key_part.strip(), None)
    if key_int is None:
        raise ValueError(f"Unknown pygame key: {key_part!r} in {name!r}")

    return mods, key_int


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


def build_key_lookup(bindings: Bindings) -> KeyLookup:
    """Build a reverse lookup dict from bindings.

    Maps ``(modifier_flags, pygame_key_int)`` to the frozenset of
    :class:`PlayerAction` names bound to that key combination.

    Parameters
    ----------
    bindings :
        Action name to key name list mapping.
    """
    tmp: dict[tuple[int, int], set[str]] = {}
    for action, keys in bindings.items():
        for key_name in keys:
            try:
                mod_key = _parse_key_name(key_name)
            except ValueError:
                logger.error("Skipping invalid key binding: %s -> %s", action, key_name)
                continue
            tmp.setdefault(mod_key, set()).add(action)
    return {k: frozenset(v) for k, v in tmp.items()}


def resolve_key(lookup: KeyLookup, key: int, mods: int = 0) -> frozenset[str]:
    """Look up a key press and return all matching player actions.

    Tries the exact modifier combination first. If no match is found
    and modifiers were held, falls back to the bare (unmodified) key.

    Parameters
    ----------
    lookup :
        Reverse lookup dict from :func:`build_key_lookup`.
    key :
        Pygame key constant (e.g. ``pygame.K_w``).
    mods :
        Pygame modifier bitmask from ``pygame.key.get_mods()``.
    """
    normalized = mods & _MOD_MASK
    if normalized:
        result = lookup.get((normalized, key))
        if result:
            return result
    return lookup.get((_MOD_NONE, key), frozenset())


# ---------------------------------------------------------------------------
# Controller resolver
# ---------------------------------------------------------------------------

# Joystick deadzone threshold for axis-to-action conversion.
_AXIS_DEADZONE: float = 0.3

# Reverse lookup: controller input name -> frozenset of actions.
ControllerLookup = dict[str, frozenset[str]]


def build_controller_lookup(bindings: Bindings) -> ControllerLookup:
    """Build a reverse lookup dict from controller bindings.

    Maps input name strings (``"BUTTON_0"``, ``"HAT_0_UP"``,
    ``"AXIS_1_NEG"``) to the frozenset of action names bound to
    that input.

    Parameters
    ----------
    bindings :
        Action name to input name list mapping.
    """
    tmp: dict[str, set[str]] = {}
    for action, inputs in bindings.items():
        for input_name in inputs:
            tmp.setdefault(input_name, set()).add(action)
    return {k: frozenset(v) for k, v in tmp.items()}


def resolve_controller_button(
    lookup: ControllerLookup,
    button: int,
) -> frozenset[str]:
    """Resolve a controller button press to player actions.

    Parameters
    ----------
    lookup :
        Controller lookup from :func:`build_controller_lookup`.
    button :
        Button index from the pygame event.
    """
    return lookup.get(f"BUTTON_{button}", frozenset())


def resolve_controller_hat(
    lookup: ControllerLookup,
    hat: int,
    value: tuple[int, int],
) -> frozenset[str]:
    """Resolve a controller hat/d-pad event to player actions.

    A hat value of ``(0, 0)`` (centered) produces no actions. Non-zero
    components are mapped to direction names and unioned.

    Parameters
    ----------
    lookup :
        Controller lookup from :func:`build_controller_lookup`.
    hat :
        Hat index from the pygame event.
    value :
        ``(x, y)`` hat position from the pygame event.

    int] :
    """
    x, y = value
    result: frozenset[str] = frozenset()
    if x < 0:
        result = result | lookup.get(f"HAT_{hat}_LEFT", frozenset())
    elif x > 0:
        result = result | lookup.get(f"HAT_{hat}_RIGHT", frozenset())
    if y > 0:
        result = result | lookup.get(f"HAT_{hat}_UP", frozenset())
    elif y < 0:
        result = result | lookup.get(f"HAT_{hat}_DOWN", frozenset())
    return result


def resolve_controller_axis(
    lookup: ControllerLookup,
    axis: int,
    value: float,
) -> frozenset[str]:
    """Resolve a controller axis value to player actions.

    Values within the deadzone (``+/-_AXIS_DEADZONE``) produce no
    actions. Beyond the deadzone, the positive or negative direction
    name is looked up.

    Parameters
    ----------
    lookup :
        Controller lookup from :func:`build_controller_lookup`.
    axis :
        Axis index.
    value :
        Current axis value (``-1.0`` to ``1.0``).
    """
    if value > _AXIS_DEADZONE:
        return lookup.get(f"AXIS_{axis}_POS", frozenset())
    if value < -_AXIS_DEADZONE:
        return lookup.get(f"AXIS_{axis}_NEG", frozenset())
    return frozenset()


def resolve_event(
    event: pygame.event.Event,
    kb_lookup: KeyLookup,
    ctrl_lookup: ControllerLookup | None = None,
) -> frozenset[str]:
    """Resolve any input event to player actions.

    Handles KEYDOWN, JOYBUTTONDOWN, and JOYHATMOTION events through
    the appropriate lookup. Returns an empty frozenset for unrecognised
    event types. Useful in menus that need the same navigation as the
    in-game UI without duplicating resolution logic.

    Parameters
    ----------
    event :
        Pygame event.
    kb_lookup :
        Keyboard reverse lookup.
    ctrl_lookup :
        Controller reverse lookup (may be ``None``).
    """
    if event.type == pygame.KEYDOWN:
        mods = pygame.key.get_mods()
        return resolve_key(kb_lookup, event.key, mods)
    if ctrl_lookup is None:
        return frozenset()
    if event.type == pygame.JOYBUTTONDOWN:
        return resolve_controller_button(ctrl_lookup, event.button)
    if event.type == pygame.JOYHATMOTION:
        return resolve_controller_hat(
            ctrl_lookup,
            event.hat,
            event.value,
        )
    return frozenset()


# ---------------------------------------------------------------------------
# Input name formatting (for rebinding UI)
# ---------------------------------------------------------------------------

# Reverse mapping from pygame key int to attribute name (e.g. 119 -> "K_w").
_KEY_INT_TO_NAME: dict[int, str] = {
    getattr(pygame, attr): attr for attr in dir(pygame) if attr.startswith("K_")
}


def event_to_key_name(key: int, mods: int) -> str:
    """Format a key press as a binding name string.

    Inverse of :func:`_parse_key_name`. Converts a pygame key constant
    and modifier bitmask into the ``"K_w"`` / ``"SHIFT+K_1"`` format
    used in the bindings config.

    Parameters
    ----------
    key :
        Pygame key constant (e.g. ``pygame.K_w``).
    mods :
        Pygame modifier bitmask from ``pygame.key.get_mods()``.
    """
    key_part = _KEY_INT_TO_NAME.get(key, f"K_{key}")
    parts: list[str] = []
    normalized = mods & _MOD_MASK
    if normalized & _MOD_CTRL:
        parts.append("CTRL")
    if normalized & _MOD_SHIFT:
        parts.append("SHIFT")
    parts.append(key_part)
    return "+".join(parts)


def controller_event_to_name(event: pygame.event.Event) -> str | None:
    """Format a controller event as a binding name string.

    Handles ``JOYBUTTONDOWN``, ``JOYHATMOTION``, and
    ``JOYAXISMOTION`` events. Returns ``None`` for hat center
    position or axis values within the deadzone.

    Parameters
    ----------
    event :
        Pygame joystick event.
    """
    if event.type == pygame.JOYBUTTONDOWN:
        return f"BUTTON_{event.button}"
    if event.type == pygame.JOYHATMOTION:
        x, y = event.value
        if x < 0:
            return f"HAT_{event.hat}_LEFT"
        if x > 0:
            return f"HAT_{event.hat}_RIGHT"
        if y > 0:
            return f"HAT_{event.hat}_UP"
        if y < 0:
            return f"HAT_{event.hat}_DOWN"
        return None
    if event.type == pygame.JOYAXISMOTION:
        if event.value > _AXIS_DEADZONE:
            return f"AXIS_{event.axis}_POS"
        if event.value < -_AXIS_DEADZONE:
            return f"AXIS_{event.axis}_NEG"
        return None
    return None


# ---------------------------------------------------------------------------
# EnvParams conversion
# ---------------------------------------------------------------------------

# Field names shared between EnvParams and the config dict.
_ENV_PARAM_FIELDS: tuple[str, ...] = (
    "max_timesteps",
    "water_probability",
    "iron_probability",
    "copper_probability",
    "coal_probability",
    "tin_probability",
    "silicon_probability",
    "base_resources",
    "miner_mining_rate",
    "player_mining_yield",
)


def env_params_to_dict(params: EnvParams) -> dict[str, int | float]:
    """Convert an EnvParams instance to a plain dict."""
    return {name: getattr(params, name) for name in _ENV_PARAM_FIELDS}


def config_to_env_params(config: PlayerConfig) -> EnvParams:
    """Build an EnvParams from the config's env_params dict.

    Missing or invalid fields fall back to EnvParams defaults.

    Parameters
    ----------
    config :
        Player configuration.
    """
    defaults = EnvParams()
    kwargs: dict[str, int | float] = {}
    for name in _ENV_PARAM_FIELDS:
        value = config.env_params.get(name)
        if value is not None:
            kwargs[name] = value
        else:
            kwargs[name] = getattr(defaults, name)
    return EnvParams(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------


def _merge_bindings(saved: Bindings, defaults: Bindings) -> Bindings:
    """Merge saved bindings with defaults, adding missing actions.

    Saved bindings take priority. Actions present in defaults but
    absent from saved are added with their default keys.

    Parameters
    ----------
    saved :
        Bindings loaded from disk.
    defaults :
        Default bindings.
    """
    merged = dict(defaults)
    merged.update(saved)
    return merged


def load_config(path: Path = CONFIG_PATH) -> PlayerConfig:
    """Load player config from disk, falling back to defaults.

    If the file does not exist or is malformed, returns a config with
    all default values. Missing fields are filled from defaults.

    Parameters
    ----------
    path :
        Path to the JSON config file.
    """
    defaults_kb = default_keyboard()
    defaults_ctrl = default_controller()
    defaults_env = env_params_to_dict(EnvParams())

    if not path.exists():
        return PlayerConfig(
            env_params=defaults_env,
            keyboard=defaults_kb,
            controller=defaults_ctrl,
            fullscreen=False,
            ui_scale=0,
        )

    try:
        raw = orjson.loads(path.read_bytes())
    except (orjson.JSONDecodeError, OSError) as exc:
        logger.error("Failed to read config %s: %s", path, exc)
        return PlayerConfig(
            env_params=defaults_env,
            keyboard=defaults_kb,
            controller=defaults_ctrl,
            fullscreen=False,
            ui_scale=0,
        )

    env = dict(defaults_env)
    env.update(raw.get("env_params", {}))

    keyboard = _merge_bindings(raw.get("keyboard", {}), defaults_kb)
    controller = _merge_bindings(raw.get("controller", {}), defaults_ctrl)

    display = raw.get("display", {})
    fullscreen = bool(display.get("fullscreen", False))
    ui_scale = int(display.get("ui_scale", 0))
    seed = int(raw.get("seed", 42))

    return PlayerConfig(
        env_params=env,
        seed=seed,
        keyboard=keyboard,
        controller=controller,
        fullscreen=fullscreen,
        ui_scale=ui_scale,
    )


def _stringify_bindings(bindings: Bindings) -> dict[str, list[str]]:
    """Ensure binding keys are plain strings for JSON serialization.

    Parameters
    ----------
    bindings :
        Binding map (keys may be StrEnum members).
    """
    return {str(k): v for k, v in bindings.items()}


def save_config(config: PlayerConfig, path: Path = CONFIG_PATH) -> None:
    """Write player config to disk as formatted JSON.

    Parameters
    ----------
    config :
        Player configuration to persist.
    path :
        Destination file path.
    """
    data = {
        "env_params": config.env_params,
        "seed": int(config.seed),
        "keyboard": _stringify_bindings(config.keyboard),
        "controller": _stringify_bindings(config.controller),
        "display": {
            "fullscreen": config.fullscreen,
            "ui_scale": config.ui_scale,
        },
    }
    try:
        path.write_bytes(
            orjson.dumps(data, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
        )
    except OSError as exc:
        logger.error("Failed to write config %s: %s", path, exc)
