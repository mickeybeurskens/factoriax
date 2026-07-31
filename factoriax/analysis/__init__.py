"""Tools for reading and drawing recorded episodes.

:class:`~factoriax.analysis.trajectory.Trajectory` is the shared input
format. Every plot in this package takes one.

Importing this package imports the five submodules listed in
``__all__``, and therefore matplotlib. The other modules
(``build_progression``, ``categories``, ``curriculum_strip``,
``graph_layout``, ``inventory``, ``recipe_graph``, ``recorder``, and
``utils``) are not re-exported here. Import them by their full path.
"""

from . import actions, eval, milestones, state, video
from .trajectory import Trajectory

__all__ = [
    "Trajectory",
    "actions",
    "eval",
    "milestones",
    "state",
    "video",
]
