"""Force a headless matplotlib backend for the analysis render tests.

Loaded by pytest before the test modules import ``matplotlib.pyplot``, so
figure creation and ``savefig`` never reach for a GUI backend.
"""

import matplotlib

matplotlib.use("Agg")
