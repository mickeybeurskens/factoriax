"""The environment object a training loop talks to, and the hooks that shape it.

:class:`FactoriaxEnv` wraps the engine in the gymnax interface. ``reset_env``
builds a world, ``step_env`` advances it, and the two space methods describe
what an agent sends and what it receives.

The engine itself knows nothing about a task. It has no reward, no goal, and no
end except the end of the time budget. An author builds a scenario with
functions that the constructor takes, and not with a subclass. Those functions
turn the same simulation into mining practice or into a rocket launch:

``achievement_fn``
    Reads a state and returns the achievement bits that hold in it.
``reward_fn``
    Scores a step. Without it, every step scores zero.
``done_fn``
    Ends an episode early. The environment ORs it with the timeout and never
    replaces the timeout.
``terrain_fn`` and ``level``
    Decide which world a reset produces.
``step_hooks`` and ``reset_hooks``
    Change a state after the engine finishes with it.

The constructor captures each one. Two environments that differ in one of them
are therefore two JIT cache entries, and each one pays for its own compile.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces  # type: ignore[import-untyped]

from factoriax.engine.constants import NUM_ACTIONS
from factoriax.engine.levels import Level, build_state, generate_terrain, initial_state
from factoriax.engine.observations import (
    NUM_PLAYER_SCALARS,
    NUM_SPATIAL_CHANNELS,
    OBSERVATIONS,
)
from factoriax.engine.state import EnvParams, EnvState
from factoriax.engine.step import factoriax_step, is_game_over

# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

AchievementFn = Callable[[EnvState], jax.Array]
StepHook = Callable[[jax.Array, EnvState, EnvParams], EnvState]
ResetHook = Callable[[jax.Array, EnvState, EnvParams], EnvState]
TerrainFn = Callable[[jax.Array, EnvParams], jax.Array]
RewardFn = Callable[[EnvState, EnvState, EnvParams], jax.Array]
DoneFn = Callable[[EnvState, EnvParams], jax.Array]


def achievement_hook(condition_fn: AchievementFn) -> StepHook:
    """Wrap an achievement condition into a step hook that latches its bits.

    The latch makes an achievement a milestone and not a state flag. Once a bit
    is set, it stays set for the rest of the episode, even after the condition
    stops holding. :func:`factoriax.engine.rewards.achievement_reward`
    therefore pays for it exactly one time.

    Parameters
    ----------
    condition_fn
        Reads a state and returns the bits that hold now. Shape
        ``(MAX_ACHIEVEMENTS,)``, bool.

    Returns
    -------
    StepHook
        A hook that folds those bits into ``achievements_unlocked`` with OR.
        The hook ignores its key argument and its params argument.
    """

    def hook(key: jax.Array, state: EnvState, params: EnvParams) -> EnvState:
        del key, params
        return state.replace(
            achievements_unlocked=state.achievements_unlocked | condition_fn(state),
        )

    return hook


class FactoriaxEnv(environment.Environment[EnvState, EnvParams]):  # type: ignore[misc]
    """The JAX grid environment of Factoriax.

    The class advances the world state, which holds the terrain, the machines,
    and the player inventories. It also evaluates an optional
    achievement-condition function in each step. The reward and the other parts
    that shape a policy belong in the gymnax wrappers around this environment.
    :class:`factoriax.engine.renderer.JaxRenderer` draws the pixels. This class
    exposes no render method.

    The constructor captures ``achievement_fn``, and :meth:`step_env` folds its
    result into ``state.achievements_unlocked``. Two different functions give
    two JIT cache entries, through ``static_argnames=("self",)`` on
    :meth:`step`. Pass ``None`` to skip that pass, which then costs nothing.

    ``terrain_fn`` and ``level`` decide what a reset builds, in that order.
    With a ``terrain_fn``, :meth:`reset_env` calls it with the PRNG key, and
    the scenario owns the world generation. Without one, a ``level`` of
    ``None`` means a generated world from the key, and a
    :class:`~factoriax.engine.levels.Level` goes through
    :func:`~factoriax.engine.levels.build_state`, which ignores the key for the
    layout. The ``reset_hooks`` then change the new state on every reset path,
    for example to fill a player inventory. They take the same arguments as
    ``step_hooks``. ``reset_env(key, params)`` is the one reset method, as
    gymnax requires.

    Parameters
    ----------
    achievement_fn
        Reads a state and returns ``bool[MAX_ACHIEVEMENTS]``. The environment
        evaluates it in every step, and a True bit latches into
        ``state.achievements_unlocked``. ``None`` skips the pass.
    level
        Fixed :class:`Level` for :meth:`reset_env` to build. ``None`` means a
        world generated from the PRNG key.
    terrain_fn
        Builds a terrain grid from the key. It wins over ``level``, so a
        scenario that supplies both gets generated terrain and no placed
        machines.
    step_hooks
        Hooks that run in order after every step. Each one takes
        ``(key, state, params)`` and returns a state.
    reset_hooks
        Hooks that run in order after every reset, with the same arguments. Use
        them to fill a player inventory or to place the machines of a scenario.
    reward_fn
        Scores each step from ``(prev_state, new_state, params)``. ``None``
        means that every step returns 0.0.
    done_fn
        Ends an episode from ``(state, params)``. The environment ORs it with
        the timeout, so it can end an episode early but can never extend one.
    obs
        Key into ``OBSERVATIONS`` that names the profile and the view, for
        example ``"x_ray_global"``. An unknown key raises ``ValueError``.
    obs_radius
        Half-width of the window for a ``local`` observation. A global view
        ignores it.
    map_width
        Map width in tiles. The environment ignores it when ``level`` supplies
        the map.
    map_height
        Map height in tiles. The environment ignores it when ``level`` supplies
        the map.
    num_players
        Number of players in the world. Only ``selected_player`` acts.
    max_machines
        Number of entity slots to allocate. A value of 0 asks for a size from
        the map area. The value is fixed for the episode, and a map with every
        slot in use refuses a new placement.
    """

    def __init__(
        self,
        achievement_fn: AchievementFn | None = None,
        level: Level | None = None,
        terrain_fn: TerrainFn | None = None,
        step_hooks: tuple[StepHook, ...] = (),
        reset_hooks: tuple[ResetHook, ...] = (),
        reward_fn: RewardFn | None = None,
        done_fn: DoneFn | None = None,
        obs: str = "x_ray_global",
        obs_radius: int = 7,
        map_width: int = 32,
        map_height: int = 32,
        num_players: int = 1,
        max_machines: int = 0,
    ) -> None:
        """Initialize the environment."""
        super().__init__()
        if obs not in OBSERVATIONS:
            raise ValueError(
                f"unknown obs {obs!r}; valid choices: {sorted(OBSERVATIONS)}"
            )
        self._achievement_fn = achievement_fn
        self._level = level
        self._terrain_fn = terrain_fn
        self._reward_fn = reward_fn
        hooks = tuple(step_hooks)
        if achievement_fn is not None:
            hooks = hooks + (achievement_hook(achievement_fn),)
        self._step_hooks = hooks
        self._reset_hooks = tuple(reset_hooks)
        self._done_fn = done_fn
        self.obs = obs
        self.obs_radius = int(obs_radius)
        self._obs_profile = "superficial" if obs.startswith("superficial") else "x_ray"
        self._obs_is_local = obs.endswith("_local")
        self.num_players: int = int(num_players)
        self.max_machines: int = int(max_machines)
        if level is not None:
            self.map_width: int = level.map_width
            self.map_height: int = level.map_height
        else:
            self.map_width = map_width
            self.map_height = map_height

    @property
    def default_params(self) -> EnvParams:
        """Return the default parameters of this environment."""
        return EnvParams()

    @partial(jax.jit, static_argnames=("self",))
    def step(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | jax.Array,
        params: EnvParams | None = None,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[str, Any]]:
        """Step the environment, with no auto-reset.

        This method overrides the gymnax ``step()``. That method calls
        ``reset_env`` in every tick for its auto-reset. It therefore runs the
        whole terrain generation in every step, even when the episode is not
        done, and about triples the cost of a step. This method calls
        ``step_env`` directly instead.

        Use :class:`~factoriax.engine.envs.wrappers.AutoResetWrapper` when you
        need auto-reset for a ``lax.scan`` training loop.

        Parameters
        ----------
        key
            PRNG key. The method passes it to the hooks. The step itself is
            deterministic.
        state
            State to advance.
        action
            Action for the selected player.
        params
            Environment parameters. ``None`` takes ``default_params``.

        Returns
        -------
        tuple
            ``(obs, new_state, reward, done, info)``. There is no auto-reset.
            When ``done`` is True, the state in the result is the final state
            of the episode, and not the start of a new one.
        """
        if params is None:
            params = self.default_params
        return self.step_env(key, state, action, params)

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[str, Any]]:
        """Run one environment step.

        The method runs the game rules. It then evaluates the optional
        achievement condition function and folds the result into
        ``state.achievements_unlocked`` with OR. It skips that pass when the
        constructor received no ``achievement_fn``. The JIT trace resolves the
        ``is None`` test, so that path costs no branch in a step.

        The order matters. The engine runs first, then the ``step_hooks``, then
        the reward against the state after the hooks, then the end test, then
        the observation. A hook can therefore change the value of a step, and
        ``reward_fn`` sees the state that the agent observes.

        Parameters
        ----------
        key
            PRNG key. The method passes it to the hooks.
        state
            State to advance.
        action
            Action for the selected player. The method casts it to int32.
        params
            Environment parameters.

        Returns
        -------
        tuple
            ``(obs, new_state, reward, done, info)``. ``reward`` is 0.0 when
            the constructor received no ``reward_fn``, and ``info`` is always
            empty.
        """
        action_arr = jnp.int32(action)
        new_state = factoriax_step(key, state, action_arr, params)
        for hook in self._step_hooks:
            new_state = hook(key, new_state, params)
        reward = (
            self._reward_fn(state, new_state, params)
            if self._reward_fn is not None
            else jnp.float32(0.0)
        )
        done = self.is_terminal(new_state, params)
        obs = self.get_obs(new_state, params)
        info: dict[str, Any] = {}
        return obs, new_state, reward, done, info

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Reset the environment to a start state.

        There are three ways to build a world, and the method tries them in
        this order: ``terrain_fn``, then a bound ``level``, then the terrain
        generator. With a ``level``, the method calls
        :func:`~factoriax.engine.levels.build_state`, which ignores the key for
        the layout.

        CAUTION: A scenario that supplies both a ``terrain_fn`` and a ``level``
        gets the terrain and loses the machines of the level. Few scenarios
        want that.

        The ``reset_hooks`` run after the build, on every path.

        Parameters
        ----------
        key
            PRNG key for the world generation. A bound ``level`` ignores it for
            the layout, but the method still passes it to the hooks.
        params
            Environment parameters.

        Returns
        -------
        tuple
            ``(obs, state)``, with ``timestep`` at 0.
        """
        if self._terrain_fn is not None:
            world_map = self._terrain_fn(key, params)
            state = initial_state(
                world_map, params, self.num_players, self.max_machines
            )
        elif self._level is None:
            world_map = generate_terrain(key, params, self.map_height, self.map_width)
            state = initial_state(
                world_map, params, self.num_players, self.max_machines
            )
        else:
            state = build_state(self._level, self.num_players, self.max_machines)
        for hook in self._reset_hooks:
            state = hook(key, state, params)
        obs = self.get_obs(state, params)
        return obs, state

    def get_obs(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Build the observation of the selected player.

        The method always observes ``state.selected_player``. With more than
        one player, the result is therefore the view of one agent, and not a
        stack with one view for each player.

        Parameters
        ----------
        state
            State to observe.
        params
            Environment parameters.

        Returns
        -------
        jax.Array
            1-D float32. :meth:`observation_space` gives its length.
        """
        fn = OBSERVATIONS[self.obs]
        if self._obs_is_local:
            return fn(state, params, state.selected_player, radius=self.obs_radius)
        return fn(state, params, state.selected_player)

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Report whether the episode is at its end.

        The timeout always applies. The method ORs a ``done_fn`` on top of it,
        so a scenario can end an episode early. No scenario can make an episode
        run longer than ``params.max_timesteps``.

        Parameters
        ----------
        state
            State to test.
        params
            Supplies ``max_timesteps``.

        Returns
        -------
        jax.Array
            Scalar bool.
        """
        over = is_game_over(state, params)
        if self._done_fn is not None:
            over = over | self._done_fn(state, params)
        return over

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        """Describe the action space, which holds every action of the engine.

        The size is ``NUM_ACTIONS`` for every scenario. A scenario whose recipe
        book cannot make an item still exposes the ``CRAFT_`` action of that
        item. That action does nothing, and it is not an error. A wrapper masks
        the actions that a scenario cannot use.

        Parameters
        ----------
        params
            The method does not read this argument. It is present for the
            gymnax signature.

        Returns
        -------
        spaces.Discrete
            A space of size ``NUM_ACTIONS``.
        """
        return spaces.Discrete(NUM_ACTIONS)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Describe the observation space of the selected variant.

        The bounds are 0.0 to 1.0, and every channel scales to stay inside
        them. This includes the x_ray facing readouts, which divide each count
        by a value equal to or larger than the ``MACHINE_MAX_STACK`` of every
        machine.

        Parameters
        ----------
        params
            The method does not read this argument. The shape comes from the
            constructor arguments, so it is fixed for the life of the
            environment.

        Returns
        -------
        spaces.Box
            1-D float32, shape ``(obs_size,)``.
        """
        if self._obs_is_local:
            side = 2 * self.obs_radius + 1
            tiles = side * side
        else:
            tiles = self.map_width * self.map_height
        obs_size = (
            NUM_SPATIAL_CHANNELS[self._obs_profile] * tiles
            + NUM_PLAYER_SCALARS[self._obs_profile]
        )
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_size,),
            dtype=jnp.float32,
        )
