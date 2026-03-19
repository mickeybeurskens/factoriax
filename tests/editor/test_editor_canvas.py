"""Tests for canvas rendering, especially after resize operations."""

import numpy as np
import pygame
import pytest

from factoriax.constants import Action, BlockType, MachineType
from factoriax.editor.canvas import Viewport, clamp_camera, render_canvas
from factoriax.editor.state import (
    ResourceBrush,
    add_column,
    add_row,
    new_editor_state,
    remove_column,
    remove_row,
    set_machine,
    set_tile,
)


@pytest.fixture(autouse=True)
def _init_pygame() -> None:
    """Ensure pygame is initialised for font rendering."""
    pygame.init()
    yield  # type: ignore[misc]
    pygame.quit()


def _make_vp(state: object) -> Viewport:
    """Build a viewport matching the editor state dimensions."""
    from factoriax.editor.state import EditorState

    es: EditorState = state  # type: ignore[assignment]
    vp = Viewport(
        tile_size=32,
        canvas_w=es.map_width * 32,
        canvas_h=es.map_height * 32,
    )
    clamp_camera(vp, es.map_width, es.map_height)
    return vp


class TestRenderAfterResize:
    """Canvas rendering must not crash after add/remove row/column."""

    def test_render_after_add_column(self) -> None:
        state = new_editor_state(5, 5)
        add_column(state)
        vp = _make_vp(state)
        img = render_canvas(state, vp, None)
        assert img.shape == (vp.canvas_h, vp.canvas_w, 4)

    def test_render_after_add_row(self) -> None:
        state = new_editor_state(5, 5)
        add_row(state)
        vp = _make_vp(state)
        img = render_canvas(state, vp, None)
        assert img.shape == (vp.canvas_h, vp.canvas_w, 4)

    def test_render_after_remove_column(self) -> None:
        state = new_editor_state(5, 5)
        remove_column(state)
        vp = _make_vp(state)
        img = render_canvas(state, vp, None)
        assert img.shape == (vp.canvas_h, vp.canvas_w, 4)

    def test_render_after_remove_row(self) -> None:
        state = new_editor_state(5, 5)
        remove_row(state)
        vp = _make_vp(state)
        img = render_canvas(state, vp, None)
        assert img.shape == (vp.canvas_h, vp.canvas_w, 4)

    def test_render_after_many_resizes(self) -> None:
        state = new_editor_state(5, 5)
        for _ in range(20):
            add_column(state)
            add_row(state)
        vp = _make_vp(state)
        img = render_canvas(state, vp, None)
        assert img.shape == (vp.canvas_h, vp.canvas_w, 4)

    def test_render_with_resources_after_resize(self) -> None:
        state = new_editor_state(5, 5)
        rng = np.random.default_rng(0)
        brush = ResourceBrush(mode="exact", exact_value=42)
        set_tile(state, 2, 2, int(BlockType.COAL), brush, rng)
        add_column(state)
        add_row(state)
        vp = _make_vp(state)
        img = render_canvas(state, vp, None, show_resources=True)
        assert img.shape == (vp.canvas_h, vp.canvas_w, 4)

    def test_render_with_machines_after_resize(self) -> None:
        state = new_editor_state(5, 5)
        set_machine(state, 2, 2, int(MachineType.MINER), int(Action.DOWN))
        add_column(state)
        add_row(state)
        vp = _make_vp(state)
        img = render_canvas(state, vp, None)
        assert img.shape == (vp.canvas_h, vp.canvas_w, 4)

    def test_render_cursor_in_new_area(self) -> None:
        state = new_editor_state(5, 5)
        add_column(state)
        add_row(state)
        vp = _make_vp(state)
        img = render_canvas(state, vp, (5, 5))
        assert img.shape == (vp.canvas_h, vp.canvas_w, 4)

    def test_render_selection_spanning_resize(self) -> None:
        state = new_editor_state(5, 5)
        add_column(state)
        add_row(state)
        vp = _make_vp(state)
        img = render_canvas(state, vp, None, selection_rect=(0, 0, 5, 5))
        assert img.shape == (vp.canvas_h, vp.canvas_w, 4)

    def test_shrink_to_minimum(self) -> None:
        state = new_editor_state(3, 3)
        remove_column(state)
        remove_column(state)
        remove_row(state)
        remove_row(state)
        vp = _make_vp(state)
        img = render_canvas(state, vp, None)
        assert img.shape == (32, 32, 4)
        assert state.map_width == 1
        assert state.map_height == 1
