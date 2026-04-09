"""Entry point for ``python -m factoriax.agentdebugger``.

Opens a trajectory replay window when given a ``.npz`` file::

    python -m factoriax.agentdebugger rollout.npz
    python -m factoriax.agentdebugger rollout.npz --level levels/mine.json
"""

import sys

from factoriax.agentdebugger.main import Debugger

if len(sys.argv) < 2:
    print(
        "Usage: python -m factoriax.agentdebugger <trajectory.npz>"
        " [--level <level.json>]\n"
        "\n"
        "If the trajectory contains full state data (recorded with\n"
        "states_to_trajectory), the game world renders automatically.\n"
        "Otherwise, provide --level to replay actions on a level.\n"
        "\n"
        "Example:\n"
        "  uv run python -m factoriax.agentdebugger rollout.npz\n"
        "  uv run python -m factoriax.agentdebugger rollout.npz"
        " --level levels/mine_resources.json"
    )
    sys.exit(1)

traj_path = sys.argv[1]
level_path = None
if "--level" in sys.argv:
    idx = sys.argv.index("--level")
    if idx + 1 < len(sys.argv):
        level_path = sys.argv[idx + 1]

debugger = Debugger.from_trajectory(traj_path, level_path=level_path)
debugger.run()
