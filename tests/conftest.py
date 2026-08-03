"""Suite-wide fixtures and headless setup.

The two environment pins below run at import, before any test module loads.
They must stay at module scope. Everything else in this file is a fixture.

Shared env fixtures:

- ``state_factory`` in this file. Builds an :class:`EnvState` from grid-shaped
  arguments. See ``tests/helpers/states.py``.
- ``pygame_display`` in this file. A display surface and a font subsystem. It
  is autouse under the directories that render, and nowhere else.
- ``canonical_env_8x8_1p`` in this file. An 8x8 single-player
  ``FactoriaxEnv()`` with a JIT-compiled step. Returns
  ``(env, params, jit_step_fn, state)``.
- ``make_env(achievement_fn)`` in ``tests/test_achievement_engine.py``. An 8x8
  single-player env with a custom ``achievement_fn``.

JIT cache thrash drives the wall time of this suite. Each new
``FactoriaxEnv(...)`` plus ``jax.jit(env.step_env)`` pair costs about 7 seconds
of XLA compile. Consume a shared env fixture before you build one. If the test
asserts the behaviour of one specific map shape, build your own env and add a
one-line comment that names the assertion.
"""

# Force headless rendering for every test in the suite. Setting these
# BEFORE pygame is imported anywhere is what keeps test runs from
# popping a window on the developer's desktop. Any conftest that later
# calls pygame.display.init() will get the dummy driver, which is
# side-effect-free but still supports Surface.blit and font rendering.
import os
from collections.abc import Callable
from typing import Any

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# Pin JAX to the CPU backend for unit tests. The suite is dominated by
# small ops (single-tile states, MAX_ACHIEVEMENTS-sized masks) where
# CUDA autotuning costs far more than the kernels themselves. On a GPU
# host the full suite takes ~110s. On CPU it takes ~70s. Benchmarks or
# scripts that legitimately need GPU can override this by exporting
# ``JAX_PLATFORMS`` before invoking pytest.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import pytest  # noqa: E402
from jax import random  # noqa: E402

from factoriax.engine.envs.base import FactoriaxEnv  # noqa: E402
from factoriax.engine.state import EnvParams, EnvState  # noqa: E402
from tests.helpers.states import make_state  # noqa: E402


@pytest.fixture(scope="session")
def pygame_display() -> None:
    """Initialise the pygame display and font subsystems once per session.

    This fixture is not autouse. Only the directories that render request it,
    through an autouse fixture of their own. Most of the suite never touches
    pygame, and an unconditional display init taxes every one of those tests.

    The SDL drivers are pinned to ``dummy`` at module scope above, so the call
    has no side effect on the desktop. The surface from ``set_mode`` is what
    :class:`factoriax.playground.ui.scaling.ScaledCanvas` reads through
    ``pygame.display.get_surface()``.
    """
    import pygame

    pygame.display.init()
    pygame.display.set_mode((800, 600))
    pygame.font.init()


@pytest.fixture(scope="session")
def canonical_env_8x8_1p() -> tuple[FactoriaxEnv, EnvParams, Any, Any]:
    """Return an 8x8 single-player env, its JIT step, and a reset state.

    Session scope means the XLA compile of ``env.step_env`` runs one time,
    not one time per test.

    A consumer must not replace ``initial_state`` in place. To step further,
    pass a different state forward locally.

    Returns
    -------
    tuple
        ``(env, params, jit_step_fn, initial_state)``. The state is the
        post-reset state at ``timestep == 0`` from ``random.PRNGKey(0)``. A
        test steps *from* it and cannot mutate it, because a JAX pytree is
        immutable by construction.
    """
    env = FactoriaxEnv(map_width=8, map_height=8)
    params = EnvParams()
    _, initial_state = env.reset_env(random.PRNGKey(0), params)
    jit_step_fn = jax.jit(env.step_env)
    return env, params, jit_step_fn, initial_state


@pytest.fixture
def state_factory() -> Callable[..., EnvState]:
    """Return :func:`tests.helpers.states.make_state`.

    Only ``world_map`` is required. Every other field has a default. Machine
    state goes in as ``(H, W)``-shaped grids, and the factory packs it into
    the engine's flat ``ent_*`` entity arrays.
    """
    return make_state
