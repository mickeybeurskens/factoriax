"""The level editor.

The editor reads and writes a :class:`~factoriax.engine.levels.Level`. It
holds the level in an :class:`~factoriax.playground.editor.state.EditorState`
while the user edits it.

Start the editor with ``python -m factoriax.playground.editor``.

This module holds no imports, so the engine import path never pulls in pygame.
Import from the defining module instead.
"""
