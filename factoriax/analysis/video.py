"""Frame composition and MP4 encoding for episode rollouts.

Used by both the PPO eval pipeline and the scripted-agent scenarios
to produce wandb-friendly map+inventory videos. Keep this module
free of training-loop dependencies — only :class:`EnvState` is
needed.
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from factoriax.analysis.inventory import render_inventory_panel
from factoriax.engine.jax_renderer import JaxRenderer
from factoriax.engine.state import EnvState

# JaxRenderer holds device-resident atlases for one tile size. Cache
# instances per ``block_pixel_size`` so the per-call cost is just a
# JIT-compiled gather, not an atlas rebuild.
_RENDERER_CACHE: dict[int, JaxRenderer] = {}


def _get_renderer(block_pixel_size: int) -> JaxRenderer:
    """

    Parameters
    ----------
    block_pixel_size: int :


    Returns
    -------
    type


    """
    renderer = _RENDERER_CACHE.get(block_pixel_size)
    if renderer is None:
        renderer = JaxRenderer(tile_px=block_pixel_size)
        _RENDERER_CACHE[block_pixel_size] = renderer
    return renderer


@contextlib.contextmanager
def _suppress_fork_warning() -> Iterator[None]:
    """Silence imageio/FFMPEG's harmless ``os.fork()`` RuntimeWarning.

    JAX initializes a thread pool eagerly, and Python warns when a
    multithreaded process forks (which is what imageio does to spawn
    its FFMPEG worker). The warning is not actionable here — the
    fork happens in a child that immediately ``exec``s ffmpeg.

    Parameters
    ----------

    Returns
    -------

    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="os.fork()",
        )
        yield


INV_PANEL_WIDTH: int = 192


def compose_frame_with_inventory(
    state: EnvState,
    *,
    block_pixel_size: int = 16,
    inv_panel_width: int = INV_PANEL_WIDTH,
) -> np.ndarray:
    """Return ``map_render | inventory_panel`` concatenated horizontally.

    The inventory panel mirrors the side panel the agent debugger
    displays in its top-right quadrant, rendered at the same height as
    the map render so the two stitch together cleanly.

    Parameters
    ----------
    state :
        The environment state to render
    block_pixel_size :
        Tile size in the map render
    block :
        Defaults to 16
    inv_panel_width :
        Width of the inventory side
    state: EnvState :

    * :

    block_pixel_size: int :
         (Default value = 16)
    inv_panel_width: int :
         (Default value = INV_PANEL_WIDTH)

    Returns
    -------
    type
        RGB ``uint8`` array of shape
        ``(map_h, map_w + inv_panel_width, 3)``.

    """
    renderer = _get_renderer(block_pixel_size)
    map_img = np.asarray(renderer.jit_render_map(state))
    panel_h = int(map_img.shape[0])
    inv_vec = np.asarray(state.player_inventory[int(state.selected_player)])
    panel = render_inventory_panel(
        inv_vec,
        width=inv_panel_width,
        height=panel_h,
    )
    return np.concatenate([map_img, panel], axis=1)


def write_video(path: Any, frames: list[np.ndarray], fps: int) -> None:
    """Encode *frames* to an MP4 at *path* using imageio / FFMPEG.

    Buffers all frames as a single ``uint8`` array before encoding.
    Use :func:`write_video_streaming` instead when ``len(frames)`` or
    the per-frame size would push peak memory beyond a few hundred MB.

    Parameters
    ----------
    path :
        Destination ``.mp4`` path. Parent directories are created.
    frames :
        List of RGB ``uint8`` arrays of identical shape.
    fps :
        Output frame rate.
    path: Any :

    frames: list[np.ndarray] :

    fps: int :


    Returns
    -------

    Raises
    ------
    ImportError
        If ``imageio[ffmpeg]`` is not installed.

    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    import imageio.v3 as iio  # noqa: PLC0415

    with _suppress_fork_warning():
        iio.imwrite(
            str(out),
            np.stack([f.astype(np.uint8) for f in frames]),
            plugin="FFMPEG",
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
        )


def write_video_streaming(
    path: Any,
    frames: Iterable[np.ndarray],
    fps: int,
) -> int:
    """Stream-encode *frames* to an MP4 one frame at a time.

    Memory stays bounded to a single frame, which matters for long
    scripted episodes that buffer ~6000 frames at ~700x512x3 bytes
    each (~6 GB of raw RGB if buffered).

    Parameters
    ----------
    path :
        Destination ``.mp4`` path. Parent directories are created.
    frames :
        Iterable of RGB ``uint8`` frames; consumed lazily so a
        generator that renders on demand is the intended use.
    fps :
        Output frame rate.
    path: Any :

    frames: Iterable[np.ndarray] :

    fps: int :


    Returns
    -------

        Number of frames written.

    Raises
    ------
    ImportError
        If ``imageio[ffmpeg]`` is not installed.

    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    import imageio.v2 as iio  # noqa: PLC0415

    with _suppress_fork_warning():
        writer = iio.get_writer(
            str(out),
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
        )
        written = 0
        try:
            for frame in frames:
                writer.append_data(np.asarray(frame, dtype=np.uint8))
                written += 1
        finally:
            writer.close()
    return written
