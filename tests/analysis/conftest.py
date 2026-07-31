"""Select a headless matplotlib backend for the analysis render tests.

pytest imports every ``conftest`` in a directory before the test
modules beside it. This module therefore runs before anything here
imports ``pyplot``. A call to ``matplotlib.use`` takes effect only
before that import.

Agg is the backend that writes files. It needs no display. On a machine
without a desktop session, ``plt.subplots`` and ``savefig`` therefore
do not ask for a window and fail.
"""

import matplotlib

matplotlib.use("Agg")
