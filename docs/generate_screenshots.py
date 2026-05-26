"""Generate screenshots for the README."""

import os
from pathlib import Path

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import numpy as np
import pygame
from PIL import Image

pygame.init()

from factoriax.editor.canvas import Viewport  # noqa: E402
from factoriax.editor.main import (  # noqa: E402
    MENU_BAR_HEIGHT,
    TOOLBAR_WIDTH,
    ToolState,
    _render_frame,
)
from factoriax.editor.state import ResourceBrush, editor_state_from_level  # noqa: E402
from factoriax.editor.toolbar import STATUS_BAR_HEIGHT, TOOL_PAINT  # noqa: E402
from factoriax.engine.constants import Action, BlockType  # noqa: E402
from factoriax.engine.jax_renderer import JaxRenderer  # noqa: E402
from factoriax.engine.levels import build_state, load_level  # noqa: E402
from factoriax.engine.state import EnvParams  # noqa: E402

MEDIA_DIR = Path("docs/media")
ROCKET_LEVEL = Path("levels/rocket.lvl.json")


def render_factory() -> None:
    """Render the rocket level as a gameplay screenshot."""
    level = load_level(ROCKET_LEVEL)
    params = EnvParams(
        map_width=level.block_map.shape[1],
        map_height=level.block_map.shape[0],
        num_players=1,
        max_timesteps=1000,
    )
    state = build_state(level, params)
    pixels = np.asarray(JaxRenderer(tile_px=32).jit_render_map(state))
    img = Image.fromarray(pixels)
    img = img.resize((img.width * 2, img.height * 2), Image.NEAREST)
    img.save(MEDIA_DIR / "factory.png")
    print(f"Saved factory.png ({img.width}x{img.height})")


def render_editor() -> None:
    """Render the full editor UI with toolbar, menu bar, and status bar."""
    level = load_level(ROCKET_LEVEL)
    es = editor_state_from_level(level)

    tile_px = 48
    canvas_w = es.map_width * tile_px
    canvas_h = es.map_height * tile_px
    base_w = TOOLBAR_WIDTH + canvas_w
    base_h = MENU_BAR_HEIGHT + canvas_h + STATUS_BAR_HEIGHT

    vp = Viewport(
        tile_size=tile_px,
        camera_x=0.0,
        camera_y=0.0,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
    )

    ts = ToolState(
        tool=TOOL_PAINT,
        block=int(BlockType.IRON),
        machine=0,
        direction=int(Action.DOWN),
        brush=ResourceBrush(mode="exact", exact_value=1000),
        show_resources=True,
        cursor_tile=(8, 7),
    )

    frame = _render_frame(
        es,
        vp,
        ts,
        base_w,
        base_h,
        file_dialog=None,
        dialog=None,
        number_dialog=None,
    )

    img = Image.fromarray(frame)
    img.save(MEDIA_DIR / "editor.png")
    print(f"Saved editor.png ({img.width}x{img.height})")


def main() -> None:
    """Generate all README screenshots."""
    render_factory()
    render_editor()


if __name__ == "__main__":
    main()
