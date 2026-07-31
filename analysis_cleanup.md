# Analysis module duplicate code cleanup

- [x] `_resolve_player_actions` re-implemented inline in `milestones.py:first_action_timestep`. Moved to `utils.py`. Both `actions.py` and `milestones.py` import from there
- [x] `if ax is None: fig, ax = plt.subplots(...) else: fig = ax.figure`. Replaced 15 sites with `resolve_ax(ax, figsize)` from `utils.py`
- [ ] `_darken` defined separately in `recipe_graph.py:117` and `curriculum_strip.py:60`. Same purpose, different algorithms
- [ ] Luminance-based text color selection (`0.2126 * r + 0.7152 * g + 0.0722 * b`) in `recipe_graph.py:_text_palette` and `curriculum_strip.py:_text_color`
- [ ] `out_path.parent.mkdir / fig.savefig / plt.close(fig) / return out_path` pattern in `build_progression.py`, `recipe_graph.py`, `curriculum_strip.py`
- [ ] `_pretty` (`recipe_graph.py:171`) and `_item_label` (`inventory.py:48`) both do `name.replace("_", " ").title()`
- [ ] `if action_labels is None: action_labels = DEFAULT_ACTION_LABELS[:num_actions]` repeated in 6+ functions in `actions.py`
