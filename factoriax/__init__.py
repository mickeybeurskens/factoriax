"""Factoriax: a JAX grid environment for reinforcement learning research.

The package holds four parts. :mod:`factoriax.engine` is the simulation, the
environments, and the wrappers. :mod:`factoriax.analysis` reads a recorded
rollout. :mod:`factoriax.playground` is the human interface, which is the
game and the level editor. :mod:`factoriax.assets` builds the sprite atlas.

Build an environment with :func:`factoriax.make.env_from_name`. Start the
playground with ``python -m factoriax.playground``.

This module holds no imports, so a training run pulls in the engine only, and
never pygame. Import from the defining module instead.
"""
