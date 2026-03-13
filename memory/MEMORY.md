# FactoriaX Project Memory

## Architecture

- `factoriax/` — all source code
- `factoriax/envs/factoriax_env.py` — JAX RL environment
- `factoriax/renderer.py` — pure numpy pixel rendering (NO pygame); shared by player and RL
- `factoriax/play_factoriax.py` — pygame-based interactive player entry point
- `factoriax/player_ui.py` — player-only UI (pygame + text); NOT imported by RL code
- `factoriax/achievements.py` — JAX-compatible achievement system
- `factoriax/state.py` — EnvState / EnvParams flax structs
- `factoriax/constants.py` — BlockType, ItemType, MachineType, Action enums, BLOCK_PIXEL_SIZE=16

## Key Constraints

- `renderer.py` must stay pure numpy — RL bots import it, no pygame allowed there
- Player-specific UI (fonts, text, menus) lives exclusively in `player_ui.py`
- Uses JAX/flax for GPU-accelerated sim; state is immutable flax struct

## Player Controls (play_factoriax.py)

WASD=move, Space=mine, I=inventory menu, P=achievement menu, C=craft,
E=place machine, Tab/[=cycle slots/recipes, Left/Right=menu focus, 1-9=select player,
R=reset, Q/Escape=quit. I and P are mutually exclusive (opening one closes the other).

## Font / Text Rendering

- `player_ui.get_pixel_font(size)` — cached via lru_cache; tries terminus first
- `_render_text_rgba(text, font, color)` — renders to RGBA numpy array, antialias=False
- `_blit_rgba(overlay, src, y, x)` — alpha-composites with bounds clipping

## player_ui.py Structure

Style constants at top (_PANEL_BG, _BORDER, _BORDER_PX, _FOCUS_STRIP, _HEADER_H, _SEP_H,
_FONT_HEADER=13, _FONT_BODY=10, _AFFORD_COLOR, _CANNOT_AFFORD_COLOR, _ITEM_NAMES).

Public API:
- `get_pixel_font(size)` — lru_cache, uses terminus then monospace fallback
- `draw_panel(overlay, x, y, w, h, *, bg, border, border_px)` — base for all menus
- `render_achievement_menu(state, w, h)` — P key menu
- `render_inventory_menu(state, w, h, menu_focus)` — I key menu (moved from renderer.py)

Private helpers: `_render_text_rgba`, `_blit_rgba`, `_draw_section_header`.

## Rendering Pipeline (player)

1. `render_pixels(state)` → RGB numpy (renderer.py, shared)
2. Optional: `render_inventory_menu(...)` → RGBA overlay (player_ui.py, player-only)
3. Optional: `render_achievement_menu(...)` → RGBA overlay (player_ui.py, player-only)
4. `composite_rgba_over_rgb(pixels, overlay)` → RGB
5. pygame surface → scale → blit

## renderer.py (RL-safe)

Only contains: create_default_textures, PLAYER_COLORS, create_player_texture,
get_textures, render_inventory_bar, MACHINE_TO_ITEM, render_machine_overlays,
render_pixels, _alpha_blend_inplace.
render_inventory_menu was REMOVED — it now lives in player_ui.py.
