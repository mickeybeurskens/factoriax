"""Interactive play script for FactoriaX using pygame."""

import jax
import numpy as np
import pygame
from jax import random

from factoriax.constants import BLOCK_PIXEL_SIZE, Action
from factoriax.envs.factoriax_env import make_factoriax_env
from factoriax.renderer import INVENTORY_BAR_HEIGHT, render_pixels


def main() -> None:
    """Run the interactive FactoriaX game.

    Controls:
        WASD: Move the selected player
        Space: Mine at current position
        C: Start crafting selected recipe
        E: Place machine from selected inventory slot
        Tab: Next inventory slot
        [: Previous inventory slot
        T: Next recipe
        G: Previous recipe
        1-9: Select player (if that many players exist)
        R: Reset the game
        Q/Escape: Quit
    """
    pygame.init()

    env, params = make_factoriax_env()
    window_width = params.map_width * BLOCK_PIXEL_SIZE
    window_height = params.map_height * BLOCK_PIXEL_SIZE + INVENTORY_BAR_HEIGHT
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("FactoriaX")
    clock = pygame.time.Clock()

    rng = random.PRNGKey(42)
    rng, reset_key = random.split(rng)
    obs, state = env.reset_env(reset_key, params)

    step_fn = jax.jit(env.step_env)

    key_to_action = {
        pygame.K_a: Action.LEFT,
        pygame.K_d: Action.RIGHT,
        pygame.K_w: Action.UP,
        pygame.K_s: Action.DOWN,
        pygame.K_SPACE: Action.MINE,
        pygame.K_c: Action.CRAFT,
        pygame.K_e: Action.PLACE,
        pygame.K_TAB: Action.NEXT_SLOT,
        pygame.K_LEFTBRACKET: Action.PREV_SLOT,
        pygame.K_t: Action.NEXT_RECIPE,
        pygame.K_g: Action.PREV_RECIPE,
    }

    key_to_player = {
        pygame.K_1: 0,
        pygame.K_2: 1,
        pygame.K_3: 2,
        pygame.K_4: 3,
        pygame.K_5: 4,
        pygame.K_6: 5,
        pygame.K_7: 6,
        pygame.K_8: 7,
        pygame.K_9: 8,
    }

    running = True
    while running:
        action = Action.NOOP

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    rng, reset_key = random.split(rng)
                    obs, state = env.reset_env(reset_key, params)
                elif event.key in key_to_player:
                    player_idx = key_to_player[event.key]
                    if player_idx < params.num_players:
                        state = state.replace(selected_player=player_idx)
                elif event.key in key_to_action:
                    action = key_to_action[event.key]

        if action != Action.NOOP:
            rng, step_key = random.split(rng)
            obs, state, reward, done, info = step_fn(step_key, state, action, params)

            if done:
                rng, reset_key = random.split(rng)
                obs, state = env.reset_env(reset_key, params)

        pixels = render_pixels(state)
        surface = pygame.surfarray.make_surface(np.transpose(pixels, (1, 0, 2)))
        screen.blit(surface, (0, 0))
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
