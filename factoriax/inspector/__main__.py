"""Entry point for ``python -m factoriax.inspector``."""

import sys

from factoriax.inspector.main import main

if len(sys.argv) < 2:
    print(
        "Usage: python -m factoriax.inspector <trajectory.npz> [--level <level.json>]"
    )
    sys.exit(1)

traj_path = sys.argv[1]
level_path = None
if "--level" in sys.argv:
    idx = sys.argv.index("--level")
    if idx + 1 < len(sys.argv):
        level_path = sys.argv[idx + 1]

main(traj_path, level_path=level_path)
