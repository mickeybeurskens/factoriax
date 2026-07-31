"""Build video frames from env states and encode them to MP4.

A frame is the map render with the inventory panel of the selected
player beside it. The PPO eval pipeline and the scripted-agent
scenarios both use these frames.

Two encoders are available.
:func:`write_video` holds every frame in memory at once and is the
simpler call. :func:`write_video_streaming` holds one frame at a time
and is the one to use for a long episode.

The module reads :class:`EnvState` and nothing else from the training
side. Keep it that way, so a scenario runner can import it without
pulling in a training loop.
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from factoriax.analysis.inventory import render_inventory_panel
from factoriax.engine.renderer import JaxRenderer
from factoriax.engine.state import EnvState

# JaxRenderer holds device-resident atlases for one tile size. Cache
# instances per ``block_pixel_size`` so the per-call cost is just a
# JIT-compiled gather, not an atlas rebuild.
_RENDERER_CACHE: dict[int, JaxRenderer] = {}


def _get_renderer(block_pixel_size: int) -> JaxRenderer:
    """Return the renderer for one tile size, from the cache.

    Parameters
    ----------
    block_pixel_size :
        The width of one map tile in pixels.

    Returns
    -------
    JaxRenderer
        A renderer whose atlases are already on the device. Building
        one is expensive, so a repeat call for the same size returns
        the same object.

    Notes
    -----
    The cache never drops an entry. Each entry holds device memory for
    its atlases, so a caller that sweeps many tile sizes keeps them all
    resident.
    """
    renderer = _RENDERER_CACHE.get(block_pixel_size)
    if renderer is None:
        renderer = JaxRenderer(tile_px=block_pixel_size)
        _RENDERER_CACHE[block_pixel_size] = renderer
    return renderer


@contextlib.contextmanager
def _suppress_fork_warning() -> Iterator[None]:
    """Silence imageio/FFMPEG's harmless ``os.fork()`` RuntimeWarning.

    JAX starts a thread pool as soon as it is imported. Python warns
    when a process with several threads forks, and a fork is how
    imageio starts its FFMPEG worker.

    The warning needs no action here. The fork happens in a child that
    replaces itself with ffmpeg at once, so none of the JAX threads
    carry over.

    Yields
    ------
    None
        Inside the block, that one warning is filtered. Every other
        warning behaves as before, and the filter is removed on exit.
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

    The panel is the same one the agent debugger shows in its top
    right. It is drawn at the height of the map render, so the two
    join with no gap.

    The panel shows the inventory of ``state.selected_player`` only. In
    a game with several players, the other inventories do not appear.

    Parameters
    ----------
    state :
        The state to draw.
    block_pixel_size :
        The width of one map tile in pixels. This sets the size of the
        map render, and therefore of the whole frame.
    inv_panel_width :
        The width of the inventory panel in pixels. A panel under 22
        pixels tall holds no rows, but the height comes from the map
        render and is far above that in practice.

    Returns
    -------
    numpy.ndarray
        RGB uint8 of shape ``(map_h, map_w + inv_panel_width, 3)``.
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

    The function stacks every frame into one array before it encodes.
    Peak memory therefore holds the whole episode twice over. Use
    :func:`write_video_streaming` when that total passes a few hundred
    megabytes.

    Parameters
    ----------
    path :
        Where to write the ``.mp4``. Missing parent directories are
        created.
    frames :
        RGB uint8 frames. Every frame must have the same shape, and
        both side lengths must be even for the ``yuv420p`` format.
    fps :
        Frames per second in the output.

    Raises
    ------
    ImportError
        When ``imageio[ffmpeg]`` is absent. The import is inside the
        function, so a caller that never writes a video does not need
        the dependency.
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

    Memory holds one frame at a time. A long scripted episode can run
    to 6000 frames of about 700 by 512 pixels. Buffered, that is near
    6 GB of raw RGB, so streaming is the only workable choice.

    Parameters
    ----------
    path :
        Where to write the ``.mp4``. Missing parent directories are
        created.
    frames :
        RGB uint8 frames, read one at a time. A generator that renders
        each frame on demand is the intended argument, because it
        keeps the whole episode out of memory.
    fps :
        Frames per second in the output.

    Returns
    -------
    int
        How many frames were written.

    Raises
    ------
    ImportError
        When ``imageio[ffmpeg]`` is absent.

    Notes
    -----
    The writer is closed even when a frame raises, so a partial file
    is left behind and not a locked one.
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
