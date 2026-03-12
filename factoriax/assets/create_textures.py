"""Script to generate simple texture assets."""

import numpy as np

SIZE = 16


def create_dirt_texture() -> np.ndarray:
    """Create a dirt texture with slight variation."""
    np.random.seed(42)
    base = np.array([139, 90, 43], dtype=np.uint8)
    texture = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
    for y in range(SIZE):
        for x in range(SIZE):
            variation = np.random.randint(-10, 11, 3)
            color = np.clip(base.astype(np.int32) + variation, 0, 255)
            texture[y, x, :3] = color
            texture[y, x, 3] = 255
    return texture


def create_water_texture() -> np.ndarray:
    """Create a water texture with wave-like variation."""
    np.random.seed(43)
    base = np.array([64, 164, 223], dtype=np.uint8)
    texture = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
    for y in range(SIZE):
        for x in range(SIZE):
            wave = int(8 * np.sin(x * 0.5 + y * 0.3))
            variation = np.array([wave, wave // 2, wave // 4])
            color = np.clip(base.astype(np.int32) + variation, 0, 255)
            texture[y, x, :3] = color
            texture[y, x, 3] = 255
    return texture


def create_player_texture() -> np.ndarray:
    """Create a simple player sprite."""
    texture = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
    center = SIZE // 2
    for y in range(SIZE):
        for x in range(SIZE):
            dist = ((x - center) ** 2 + (y - center) ** 2) ** 0.5
            if dist <= SIZE // 3:
                texture[y, x] = [255, 100, 100, 255]
            elif dist <= SIZE // 3 + 1:
                texture[y, x] = [200, 60, 60, 255]
    return texture


if __name__ == "__main__":
    from pathlib import Path

    import imageio.v3 as iio

    assets_dir = Path(__file__).parent
    iio.imwrite(assets_dir / "dirt.png", create_dirt_texture())
    iio.imwrite(assets_dir / "water.png", create_water_texture())
    iio.imwrite(assets_dir / "player.png", create_player_texture())
    print("Textures created successfully")
