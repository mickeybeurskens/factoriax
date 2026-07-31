"""The environments, the wrappers, and the scenario registry.

- ``base.py`` holds :class:`FactoriaxEnv`, the core gymnax-compatible
  environment.
- ``wrappers.py`` holds :class:`AutoResetWrapper`,
  :class:`ActionMaskWrapper`, and :class:`LogWrapper`.
- ``registry.py`` holds the scenario list and the :func:`make` factory.
- ``easy_rocket.py`` and ``rocket.py`` hold two scenario definitions.
"""
