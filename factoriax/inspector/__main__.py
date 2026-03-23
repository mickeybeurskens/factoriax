"""Entry point for ``python -m factoriax.inspector``."""

import sys

from factoriax.inspector.main import main

if len(sys.argv) < 2:
    print("Usage: python -m factoriax.inspector <trajectory.npz>")
    sys.exit(1)

main(sys.argv[1])
