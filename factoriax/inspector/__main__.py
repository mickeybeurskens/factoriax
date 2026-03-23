"""Entry point for ``python -m factoriax.inspector``."""

import sys

from factoriax.inspector.main import main

if len(sys.argv) < 2:
    print(
        "Usage: python -m factoriax.inspector <trajectory.npz>"
        " --level <level.json>\n"
        "\n"
        "Example:\n"
        "  uv run python -m factoriax.inspector rollout.npz"
        " --level levels/mine_resources.json\n"
        "\n"
        "The --level flag is required to render the game world.\n"
        "Available levels in levels/:\n"
        "  mine_resources.json   craft_chests.json   fill_chest.json\n"
        "  (or any .json exported from the editor)"
    )
    sys.exit(1)

traj_path = sys.argv[1]
level_path = None
if "--level" in sys.argv:
    idx = sys.argv.index("--level")
    if idx + 1 < len(sys.argv):
        level_path = sys.argv[idx + 1]

if level_path is None:
    print(
        "WARNING: No --level provided. The game world will not be"
        " rendered.\n"
        "  Add: --level levels/mine_resources.json\n"
    )

main(traj_path, level_path=level_path)
