# Factoriax

> "The Factory must grow!"

Factoriax is a GPU accelerated factory building simulator for reinforcement learning research in the style of the game [Factorio](https://www.factorio.com/), written in JAX.

---

```{image} _static/title_image.png
:alt: A Factoriax factory: miners on ore patches, belts that carry ore to furnaces and assemblers, and a player among them
:align: center
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Home

Home <self>
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Start here

start_here/index
start_here/getting_started
start_here/playing_manually
```

% Hidden for the release. To bring a section back, uncomment its block here
% and drop the matching entry from `exclude_patterns` in conf.py.
%
% Stub pages from the "Start here" toctree above:
%
% start_here/moving_player
% start_here/speeding_up
% start_here/saving_runs
% start_here/plotting_results
%
% ```{toctree}
% :maxdepth: 1
% :hidden:
% :caption: Training models
%
% training/index
% training/train_ppo_mining
% training/plotting_results
% ```
%
% ```{toctree}
% :maxdepth: 1
% :hidden:
% :caption: Understanding Factoriax
%
% understanding/index
% understanding/state_and_step
% understanding/observation_space
% understanding/action_design
% understanding/scenarios_and_curriculum
% ```
%
% ```{toctree}
% :maxdepth: 1
% :hidden:
% :caption: Modifying environments
%
% modifying/index
% modifying/change_observations
% modifying/change_actions
% modifying/change_textures
% modifying/adding_mechanics
% modifying/rewriting_base_mechanics
% ```

```{toctree}
:maxdepth: 2
:hidden:
:caption: API documentation

api/index
```

## Why Factoriax?

Factorio is quite a cool game with an enormous amount of emergent gameplay complexity, and it is interesting to use it as a benchmark to evaluate AI systems.
However, even though the game is famous for being [carefully optimized](https://www.factorio.com/blog/post/fff-421), training Reinforcement Learning agents on the base game is very slow.
Factoriax is basically a simplified version of the core Factorio game mechanics that can run millions of game ticks per second on higher end GPUs (as of writing), while also running quickly enough to train models on local laptop GPUs.
This allows researchers to test their RL ideas in Factoriax at scale, while allowing students to get familiar with RL concepts without the need for enormous GPU clusters.

And honestly, Factorio is just a lot of fun. 
Even though the Factoriax implementation of the original game mechanics is quite minimal, I hope it does some justice to the original game.

## Using The Docs
The documentation is divided in different sections:

- {doc}`start_here/index`: A tutorial to get you up and running with Factoriax and JAX as soon as possible.
- {doc}`api/index`: A reference for understanding the code in the repository.

% Hidden for the release along with their sections:
%
% - {doc}`training/index`: Guides on how to use Factoriax to train your own reinforcement learning models.
% - {doc}`modifying/index`: Guides on how to create your own Factoriax environments and extend basic functionality.
% - {doc}`understanding/index`: A more in depth discussion of the design decisions behind Factoriax. Useful to improve your understanding more broadly.

Remember:
> The factory must grow!

