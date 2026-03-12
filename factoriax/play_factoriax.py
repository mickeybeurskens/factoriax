"""Interactive play script for FactoriaX using pygame."""

import jax
import numpy as np
import pygame
from jax import random

from factoriax.constants import BLOCK_PIXEL_SIZE, Action
from factoriax.envs.factoriax_env import make_factoriax_env
from factoriax.renderer import render_pixels


def main() -> None:
    """Run the interactive FactoriaX game."""
    pygame.init()

    env, params = make_factoriax_env()
    window_width = params.map_width * BLOCK_PIXEL_SIZE
    window_height = params.map_height * BLOCK_PIXEL_SIZE
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
