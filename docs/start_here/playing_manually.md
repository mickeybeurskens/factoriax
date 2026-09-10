# Playing A Game Manually

Factoriax has a human playable interface, called the playground. 
It contains a human playable version of Factoriax and a level editor.

## Opening The Playground

To open the playground, run this command:

```bash
uv run python -m factoriax.playground
```

If you installed Factoriax with pip, run this command instead:

```bash
python -m factoriax.playground
```

The launcher shows four entries:

- Play starts a new game. You choose the environment parameters and the seed.
  The engine generates the world from its general terrain sampler.
- Editor opens the level editor. Use it to author custom maps.
- Settings changes the display, the UI scale, fullscreen mode, and the key and
  controller bindings. Factoriax writes the changes to `player_config.json`,
  in the directory that you started it from.
- Quit closes Factoriax.

## Playing The Game

You control a player on a grid. You mine ore, craft machines from that ore,
place the machines, and let them run.

Press `?` in the game for the full list of control option. This table gives the default bindings.

| Group | Key | Action |
| --- | --- | --- |
| Movement | `WASD` or the arrow keys | Move the player |
| Movement | `Ctrl+1` to `Ctrl+9` | Switch the active player |
| Actions | `Space` | Mine the ore under the player |
| Actions | `E` | Place a machine, or pick one up |
| Actions | `R` | Rotate the machine in front |
| Actions | `1` to `8` | Select inventory slot 1 to 8 |
| Actions | `Shift+1` to `Shift+2` | Select inventory slot 9 or 10 |
| Crafting | `I` | Open the crafting menu |
| Crafting | `W` or `S` | Select a recipe |
| Crafting | `E` | Craft the selected recipe |
| Machines | `F` | Inspect the machine in front |
| Machines | `W` or `S` | Switch panel |
| Machines | `A` or `D` | Select an item |
| Machines | `E` | Transfer items |
| Other | `P` | Open the achievements |
| Other | `Escape` | Close a menu, or pause the game |

You can change every binding, for the keyboard and for a controller. Open the
Settings page of the launcher to do this. A standard gamepad works without
setup. 

## The Level Editor

To open the editor, select it in the menu or run this command:

```bash
uv run python -m factoriax.playground.editor
```

Paint terrain with the left mouse button. Erase with the right mouse button.
Press `?` in the editor for the full list of button options.

| Group | Key | Action |
| --- | --- | --- |
| Drawing | Left click or drag | Paint a tile or a machine |
| Drawing | Right click or drag | Erase, one layer at a time |
| Drawing | `X` | Eraser tool, which clears both layers |
| Drawing | `B` or `F` | Paint tool, or fill rectangle tool |
| Drawing | `1` to `5` | Terrain: dirt, water, iron, copper, coal |
| Drawing | `6` to `0` | Select a machine from the palette |
| Drawing | `R` | Rotate the machine under the cursor |
| Drawing | `I` | Inspect the inventory of a machine |
| Resources | `T` | Switch between exact mode and range mode |
| Resources | `V` | Turn the resource overlay on or off |
| Resources | `[` and `]` | Adjust the resource value |
| View | Scroll wheel | Zoom |
| View | Middle drag, or the arrow keys | Pan |
| Map | `Ctrl` and an arrow key | Add or remove a row or a column |
| Map | `Ctrl+N`, `Ctrl+O`, `Ctrl+S` | New map, load a map, save the map |
| Map | `F5` | Play-test the current map |
| Map | `Escape` | Quit the editor |

You can load levels in the level editor, or by using the Python API:

```python
from pathlib import Path

from factoriax.engine.levels import load_level

level = load_level(Path("levels/my_level.json"))
```

