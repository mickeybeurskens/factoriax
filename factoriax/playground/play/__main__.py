"""Entry point for ``python -m factoriax.playground.play``.

Delegates to the main menu entry point so all human-facing launches
go through the same title screen.
"""

from factoriax.__main__ import _run

if __name__ == "__main__":
    _run()
