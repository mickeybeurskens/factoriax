"""RL Trajectory Inspector for FactoriaX.

Interactive step-by-step episode replay with reward charts, action
timelines, and player state panels. Launch via::

    python -m factoriax.inspector path/to/trajectory.npz
"""

from factoriax.inspector.main import main

__all__ = ["main"]
