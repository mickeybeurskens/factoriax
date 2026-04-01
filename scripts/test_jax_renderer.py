"""Tests for the pure-JAX parallel renderer.

Validates that the JAX renderer produces correct output across single
and batched environments, and that it composes properly with jit and vmap.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from factoriax.constants import (
    ITEM_COLORS,
    BlockType,
    Direction,
    ItemType,
    MachineType,
)
from factoriax.envs.factoriax_env import FactoriaXEnv
from factoriax.levels import LevelBuilder, build_state
from factoriax.renderer import render_pixels
from factoriax.state import EnvParams, EnvState
from factoriax.jax_renderer import (
    INV_HEIGHT,
    JaxRenderer,
    SLOT_BG,
    SLOT_BG_SELECTED,
    build_block_atlas,
    build_digit_atlas,
    build_item_color_atlas,
    build_machine_atlas,
    build_player_sprite,
    render_hud,
    render_inventory_strip,
    render_map,
    render_map_with_inventory,
)
from scripts.jax_render_benchmark import (
    extract_single_state,
    make_batched_envs,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TILE_PX = 8
MAP_SIZE = 8


@pytest.fixture()
def params() -> EnvParams:
    """Small environment params for fast tests."""
    return EnvParams(
        map_width=MAP_SIZE,
        map_height=MAP_SIZE,
        num_players=1,
        max_timesteps=100,
        nest_probability=0.0,
        max_biters=1,
    )


@pytest.fixture()
def block_atlas() -> jnp.ndarray:
    """Block texture atlas."""
    return build_block_atlas(TILE_PX)


@pytest.fixture()
def machine_atlas() -> jnp.ndarray:
    """Machine texture atlas."""
    return build_machine_atlas(TILE_PX)


@pytest.fixture()
def player_sprite() -> jnp.ndarray:
    """Player sprite."""
    return build_player_sprite(TILE_PX)


@pytest.fixture()
def single_state(params: EnvParams) -> EnvState:
    """A single non-batched environment state."""
    env = FactoriaXEnv()
    _, state = env.reset_env(jax.random.key(0), params)
    return state


# ---------------------------------------------------------------------------
# Atlas construction tests
# ---------------------------------------------------------------------------


class TestAtlasConstruction:
    """Tests for texture atlas building."""

    def test_block_atlas_shape(self, block_atlas: jnp.ndarray) -> None:
        """Block atlas has correct shape and covers all BlockType values."""
        num_types = max(int(b) for b in BlockType) + 1
        assert block_atlas.shape == (num_types, TILE_PX, TILE_PX, 3)
        assert block_atlas.dtype == jnp.uint8

    def test_machine_atlas_shape(self, machine_atlas: jnp.ndarray) -> None:
        """Machine atlas has correct shape and covers all MachineType values."""
        num_types = max(int(m) for m in MachineType) + 1
        assert machine_atlas.shape == (num_types, TILE_PX, TILE_PX, 3)
        assert machine_atlas.dtype == jnp.uint8

    def test_player_sprite_shape(self, player_sprite: jnp.ndarray) -> None:
        """Player sprite is a single tile-sized RGB image."""
        assert player_sprite.shape == (TILE_PX, TILE_PX, 3)
        assert player_sprite.dtype == jnp.uint8

    def test_block_atlas_dirt_not_black(
        self, block_atlas: jnp.ndarray
    ) -> None:
        """Dirt tile should have non-zero pixels (brown color)."""
        dirt_tile = block_atlas[int(BlockType.DIRT)]
        assert jnp.any(dirt_tile > 0)

    def test_machine_atlas_none_is_black(
        self, machine_atlas: jnp.ndarray
    ) -> None:
        """NONE machine type should be all-black (used as transparent)."""
        none_tile = machine_atlas[int(MachineType.NONE)]
        assert jnp.all(none_tile == 0)


# ---------------------------------------------------------------------------
# Single-state rendering tests
# ---------------------------------------------------------------------------


class TestSingleRender:
    """Tests for rendering a single environment state."""

    def test_output_shape(
        self,
        single_state: EnvState,
        block_atlas: jnp.ndarray,
        machine_atlas: jnp.ndarray,
        player_sprite: jnp.ndarray,
    ) -> None:
        """Output image has the expected pixel dimensions."""
        img = render_map(
            single_state, block_atlas, machine_atlas, player_sprite
        )
        expected_h = MAP_SIZE * TILE_PX
        expected_w = MAP_SIZE * TILE_PX
        assert img.shape == (expected_h, expected_w, 3)

    def test_output_dtype(
        self,
        single_state: EnvState,
        block_atlas: jnp.ndarray,
        machine_atlas: jnp.ndarray,
        player_sprite: jnp.ndarray,
    ) -> None:
        """Output should be uint8 RGB."""
        img = render_map(
            single_state, block_atlas, machine_atlas, player_sprite
        )
        assert img.dtype == jnp.uint8

    def test_not_all_black(
        self,
        single_state: EnvState,
        block_atlas: jnp.ndarray,
        machine_atlas: jnp.ndarray,
        player_sprite: jnp.ndarray,
    ) -> None:
        """Rendered image should contain visible content, not all zeros."""
        img = render_map(
            single_state, block_atlas, machine_atlas, player_sprite
        )
        assert jnp.any(img > 0)

    def test_jit_produces_same_output(
        self,
        single_state: EnvState,
        block_atlas: jnp.ndarray,
        machine_atlas: jnp.ndarray,
        player_sprite: jnp.ndarray,
    ) -> None:
        """JIT-compiled render should produce identical output."""
        img_eager = render_map(
            single_state, block_atlas, machine_atlas, player_sprite
        )
        img_jit = jax.jit(render_map)(
            single_state, block_atlas, machine_atlas, player_sprite
        )
        np.testing.assert_array_equal(np.array(img_eager), np.array(img_jit))

    def test_deterministic(
        self,
        single_state: EnvState,
        block_atlas: jnp.ndarray,
        machine_atlas: jnp.ndarray,
        player_sprite: jnp.ndarray,
    ) -> None:
        """Two calls with the same state produce identical images."""
        img1 = render_map(
            single_state, block_atlas, machine_atlas, player_sprite
        )
        img2 = render_map(
            single_state, block_atlas, machine_atlas, player_sprite
        )
        np.testing.assert_array_equal(np.array(img1), np.array(img2))


# ---------------------------------------------------------------------------
# Terrain layer tests
# ---------------------------------------------------------------------------


class TestTerrainRendering:
    """Tests that terrain tiles map to the correct atlas colors."""

    def test_uniform_dirt_map(
        self,
        block_atlas: jnp.ndarray,
        machine_atlas: jnp.ndarray,
        player_sprite: jnp.ndarray,
    ) -> None:
        """A map of all-dirt should render as a uniform brown image.

        We build a level with no machines, no ores, and check that
        all non-player pixels match the dirt atlas color.
        """
        level = LevelBuilder(4, 4).build("all_dirt")
        p = EnvParams(
            map_width=4,
            map_height=4,
            num_players=1,
            max_timesteps=10,
            max_biters=1,
        )
        state = build_state(level, p)

        atlas_4 = build_block_atlas(TILE_PX)
        matlas_4 = build_machine_atlas(TILE_PX)

        img = np.array(
            render_map(state, atlas_4, matlas_4, player_sprite)
        )

        # The dirt color from the atlas
        dirt_color = np.array(atlas_4[int(BlockType.DIRT), 0, 0, :])

        # Player occupies one tile at center (2, 2). Check a corner tile
        # that's definitely just dirt.
        tile_region = img[0:TILE_PX, 0:TILE_PX, :]
        np.testing.assert_array_equal(
            tile_region,
            np.broadcast_to(dirt_color, tile_region.shape),
        )

    def test_water_tile_color(
        self,
        machine_atlas: jnp.ndarray,
        player_sprite: jnp.ndarray,
    ) -> None:
        """A water tile should render with the water atlas color."""
        level = (
            LevelBuilder(4, 4)
            .fill_rect(0, 0, 1, 1, BlockType.WATER)
            .set_player_position(3, 3)
            .build("water_corner")
        )
        p = EnvParams(
            map_width=4,
            map_height=4,
            num_players=1,
            max_timesteps=10,
            max_biters=1,
        )
        state = build_state(level, p)

        atlas = build_block_atlas(TILE_PX)
        img = np.array(render_map(state, atlas, machine_atlas, player_sprite))

        water_color = np.array(atlas[int(BlockType.WATER), 0, 0, :])
        tile_region = img[0:TILE_PX, 0:TILE_PX, :]
        np.testing.assert_array_equal(
            tile_region,
            np.broadcast_to(water_color, tile_region.shape),
        )


# ---------------------------------------------------------------------------
# Machine overlay tests
# ---------------------------------------------------------------------------


class TestMachineOverlay:
    """Tests that machine overlays appear on tiles with machines."""

    def test_machine_changes_pixels(
        self,
        block_atlas: jnp.ndarray,
        machine_atlas: jnp.ndarray,
        player_sprite: jnp.ndarray,
    ) -> None:
        """Placing a machine should change the rendered pixels on that tile."""
        level_bare = LevelBuilder(4, 4).set_player_position(0, 0).build("bare")
        level_machine = (
            LevelBuilder(4, 4)
            .place_machine(3, 3, int(MachineType.MINER), int(Direction.DOWN))
            .set_player_position(0, 0)
            .build("with_miner")
        )
        p = EnvParams(
            map_width=4,
            map_height=4,
            num_players=1,
            max_timesteps=10,
            max_biters=1,
        )
        state_bare = build_state(level_bare, p)
        state_machine = build_state(level_machine, p)

        img_bare = np.array(
            render_map(state_bare, block_atlas, machine_atlas, player_sprite)
        )
        img_machine = np.array(
            render_map(
                state_machine, block_atlas, machine_atlas, player_sprite
            )
        )

        # The tile at (3, 3) should differ
        y0 = 3 * TILE_PX
        x0 = 3 * TILE_PX
        tile_bare = img_bare[y0 : y0 + TILE_PX, x0 : x0 + TILE_PX]
        tile_machine = img_machine[y0 : y0 + TILE_PX, x0 : x0 + TILE_PX]
        assert not np.array_equal(tile_bare, tile_machine)

    def test_no_machine_tile_unchanged(
        self,
        block_atlas: jnp.ndarray,
        machine_atlas: jnp.ndarray,
        player_sprite: jnp.ndarray,
    ) -> None:
        """Tiles without machines should render the same as bare terrain."""
        level_bare = LevelBuilder(4, 4).set_player_position(0, 0).build("bare")
        level_machine = (
            LevelBuilder(4, 4)
            .place_machine(3, 3, int(MachineType.CHEST), int(Direction.DOWN))
            .set_player_position(0, 0)
            .build("with_chest")
        )
        p = EnvParams(
            map_width=4,
            map_height=4,
            num_players=1,
            max_timesteps=10,
            max_biters=1,
        )
        state_bare = build_state(level_bare, p)
        state_machine = build_state(level_machine, p)

        img_bare = np.array(
            render_map(state_bare, block_atlas, machine_atlas, player_sprite)
        )
        img_machine = np.array(
            render_map(
                state_machine, block_atlas, machine_atlas, player_sprite
            )
        )

        # Tile (1, 1) has no machine in either case, should be identical
        y0 = 1 * TILE_PX
        x0 = 1 * TILE_PX
        tile_bare = img_bare[y0 : y0 + TILE_PX, x0 : x0 + TILE_PX]
        tile_machine = img_machine[y0 : y0 + TILE_PX, x0 : x0 + TILE_PX]
        np.testing.assert_array_equal(tile_bare, tile_machine)


# ---------------------------------------------------------------------------
# Player sprite tests
# ---------------------------------------------------------------------------


class TestPlayerSprite:
    """Tests that player sprites are stamped at the correct positions."""

    def test_player_visible_at_position(
        self,
        block_atlas: jnp.ndarray,
        machine_atlas: jnp.ndarray,
        player_sprite: jnp.ndarray,
    ) -> None:
        """The player sprite should appear at the player's tile position."""
        level = LevelBuilder(8, 8).set_player_position(2, 3).build("player")
        p = EnvParams(
            map_width=8,
            map_height=8,
            num_players=1,
            max_timesteps=10,
            max_biters=1,
        )
        state = build_state(level, p)
        img = np.array(
            render_map(state, block_atlas, machine_atlas, player_sprite)
        )

        # The player tile should differ from the raw terrain tile
        y0 = 3 * TILE_PX  # row = y
        x0 = 2 * TILE_PX  # col = x
        player_tile = img[y0 : y0 + TILE_PX, x0 : x0 + TILE_PX]
        dirt_color = np.array(block_atlas[int(BlockType.DIRT), 0, 0, :])
        dirt_tile = np.broadcast_to(dirt_color, player_tile.shape)
        assert not np.array_equal(player_tile, dirt_tile)

    def test_player_at_different_position(
        self,
        block_atlas: jnp.ndarray,
        machine_atlas: jnp.ndarray,
        player_sprite: jnp.ndarray,
    ) -> None:
        """Moving the player should change which tile has the sprite."""
        level_a = LevelBuilder(8, 8).set_player_position(1, 1).build("a")
        level_b = LevelBuilder(8, 8).set_player_position(6, 6).build("b")
        p = EnvParams(
            map_width=8,
            map_height=8,
            num_players=1,
            max_timesteps=10,
            max_biters=1,
        )
        state_a = build_state(level_a, p)
        state_b = build_state(level_b, p)

        img_a = np.array(
            render_map(state_a, block_atlas, machine_atlas, player_sprite)
        )
        img_b = np.array(
            render_map(state_b, block_atlas, machine_atlas, player_sprite)
        )

        # Tile (1,1) should differ between the two renders
        y0 = 1 * TILE_PX
        x0 = 1 * TILE_PX
        assert not np.array_equal(
            img_a[y0 : y0 + TILE_PX, x0 : x0 + TILE_PX],
            img_b[y0 : y0 + TILE_PX, x0 : x0 + TILE_PX],
        )


# ---------------------------------------------------------------------------
# Batched (vmap) rendering tests
# ---------------------------------------------------------------------------


class TestVmapRender:
    """Tests for vmapped batch rendering."""

    def test_vmap_output_shape(
        self,
        params: EnvParams,
        block_atlas: jnp.ndarray,
        machine_atlas: jnp.ndarray,
        player_sprite: jnp.ndarray,
    ) -> None:
        """Vmapped render should add a batch dimension to the output."""
        n = 4
        _, states = make_batched_envs(n, params)
        vmap_render = jax.jit(
            jax.vmap(render_map, in_axes=(0, None, None, None))
        )
        imgs = vmap_render(states, block_atlas, machine_atlas, player_sprite)
        expected = (n, MAP_SIZE * TILE_PX, MAP_SIZE * TILE_PX, 3)
        assert imgs.shape == expected

    def test_vmap_matches_individual(
        self,
        params: EnvParams,
        block_atlas: jnp.ndarray,
        machine_atlas: jnp.ndarray,
        player_sprite: jnp.ndarray,
    ) -> None:
        """Each slice of vmapped output should match individual render."""
        n = 3
        _, states = make_batched_envs(n, params)
        vmap_render = jax.jit(
            jax.vmap(render_map, in_axes=(0, None, None, None))
        )
        batch_imgs = np.array(
            vmap_render(states, block_atlas, machine_atlas, player_sprite)
        )

        for i in range(n):
            single = extract_single_state(states, i)
            individual = np.array(
                jax.jit(render_map)(
                    single, block_atlas, machine_atlas, player_sprite
                )
            )
            np.testing.assert_array_equal(batch_imgs[i], individual)

    def test_different_envs_different_images(
        self,
        params: EnvParams,
        block_atlas: jnp.ndarray,
        machine_atlas: jnp.ndarray,
        player_sprite: jnp.ndarray,
    ) -> None:
        """Different env states should produce different rendered frames."""
        _, states = make_batched_envs(2, params)
        vmap_render = jax.jit(
            jax.vmap(render_map, in_axes=(0, None, None, None))
        )
        imgs = np.array(
            vmap_render(states, block_atlas, machine_atlas, player_sprite)
        )
        # With different random seeds, terrain should differ
        assert not np.array_equal(imgs[0], imgs[1])


# ---------------------------------------------------------------------------
# Inventory rendering tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def item_colors() -> jnp.ndarray:
    """Item color atlas."""
    return build_item_color_atlas()


@pytest.fixture()
def digit_atlas() -> jnp.ndarray:
    """Digit bitmap atlas."""
    return build_digit_atlas()


class TestInventoryStrip:
    """Tests for the inventory strip renderer."""

    def test_strip_shape(
        self,
        single_state: EnvState,
        item_colors: jnp.ndarray,
        digit_atlas: jnp.ndarray,
    ) -> None:
        """Inventory strip should be INV_HEIGHT x map_width_px x 3."""
        img_width = MAP_SIZE * TILE_PX
        strip = render_inventory_strip(
            single_state, item_colors, digit_atlas, img_width
        )
        assert strip.shape == (INV_HEIGHT, img_width, 3)
        assert strip.dtype == jnp.uint8

    def test_empty_slots_are_dark(
        self,
        single_state: EnvState,
        item_colors: jnp.ndarray,
        digit_atlas: jnp.ndarray,
    ) -> None:
        """Empty inventory should produce a mostly dark strip."""
        img_width = MAP_SIZE * TILE_PX
        strip = np.array(
            render_inventory_strip(
                single_state, item_colors, digit_atlas, img_width
            )
        )
        # Average brightness should be low (close to SLOT_BG = 40).
        assert strip.mean() < 100

    def test_nonempty_slot_has_item_color(
        self,
        block_atlas: jnp.ndarray,
        machine_atlas: jnp.ndarray,
        player_sprite: jnp.ndarray,
        item_colors: jnp.ndarray,
        digit_atlas: jnp.ndarray,
    ) -> None:
        """A slot with items should contain the item's color."""
        # Build a level where the player starts with iron.
        level = (
            LevelBuilder(MAP_SIZE, MAP_SIZE)
            .build("with_iron")
        )
        level.player_inventory = [(int(ItemType.IRON), 10)]
        p = EnvParams(
            map_width=MAP_SIZE,
            map_height=MAP_SIZE,
            num_players=1,
            max_timesteps=10,
            max_biters=1,
        )
        state = build_state(level, p)

        img_width = MAP_SIZE * TILE_PX
        strip = np.array(
            render_inventory_strip(state, item_colors, digit_atlas, img_width)
        )

        # The iron color should appear somewhere in the strip.
        iron_color = np.array(ITEM_COLORS[int(ItemType.IRON)])
        has_iron = np.all(strip == iron_color, axis=-1)
        assert has_iron.any()

    def test_selected_slot_brighter(
        self,
        single_state: EnvState,
        item_colors: jnp.ndarray,
        digit_atlas: jnp.ndarray,
    ) -> None:
        """Selected slot background should be brighter than unselected."""
        img_width = MAP_SIZE * TILE_PX
        strip = np.array(
            render_inventory_strip(
                single_state, item_colors, digit_atlas, img_width
            )
        )
        slot_width = img_width // 10
        selected = int(single_state.selected_slots[0])

        # Check that the selected slot region is brighter on average.
        sel_region = strip[:, selected * slot_width:(selected + 1) * slot_width]
        # Pick an unselected slot.
        other = (selected + 1) % 10
        other_region = strip[:, other * slot_width:(other + 1) * slot_width]
        assert sel_region.mean() > other_region.mean()


class TestWithInventoryRender:
    """Tests for the combined map + inventory renderer."""

    def test_output_shape(
        self,
        single_state: EnvState,
        block_atlas: jnp.ndarray,
        machine_atlas: jnp.ndarray,
        player_sprite: jnp.ndarray,
        item_colors: jnp.ndarray,
        digit_atlas: jnp.ndarray,
    ) -> None:
        """Output should be map height + INV_HEIGHT."""
        img = render_map_with_inventory(
            single_state, block_atlas, machine_atlas, player_sprite,
            item_colors, digit_atlas,
        )
        expected_h = MAP_SIZE * TILE_PX + INV_HEIGHT
        expected_w = MAP_SIZE * TILE_PX
        assert img.shape == (expected_h, expected_w, 3)

    def test_vmap_with_inventory(
        self,
        params: EnvParams,
        block_atlas: jnp.ndarray,
        machine_atlas: jnp.ndarray,
        player_sprite: jnp.ndarray,
        item_colors: jnp.ndarray,
        digit_atlas: jnp.ndarray,
    ) -> None:
        """Vmapped inventory render should produce correct batch shape."""
        n = 3
        _, states = make_batched_envs(n, params)
        vmap_render = jax.jit(
            jax.vmap(
                render_map_with_inventory,
                in_axes=(0, None, None, None, None, None),
            )
        )
        imgs = vmap_render(
            states, block_atlas, machine_atlas, player_sprite,
            item_colors, digit_atlas,
        )
        expected_h = MAP_SIZE * TILE_PX + INV_HEIGHT
        expected_w = MAP_SIZE * TILE_PX
        assert imgs.shape == (n, expected_h, expected_w, 3)


# ---------------------------------------------------------------------------
# Full HUD rendering tests
# ---------------------------------------------------------------------------


class TestFullHUD:
    """Tests for the 4-quadrant HUD renderer."""

    def test_output_shape(
        self,
        single_state: EnvState,
        block_atlas: jnp.ndarray,
        machine_atlas: jnp.ndarray,
        player_sprite: jnp.ndarray,
        item_colors: jnp.ndarray,
        digit_atlas: jnp.ndarray,
    ) -> None:
        """HUD output should be 2x map height (map + HUD panel)."""
        img = render_hud(
            single_state, block_atlas, machine_atlas, player_sprite,
            item_colors, digit_atlas,
        )
        map_px = MAP_SIZE * TILE_PX
        assert img.shape == (2 * map_px, map_px, 3)
        assert img.dtype == jnp.uint8

    def test_hud_has_content(
        self,
        block_atlas: jnp.ndarray,
        machine_atlas: jnp.ndarray,
        player_sprite: jnp.ndarray,
        item_colors: jnp.ndarray,
        digit_atlas: jnp.ndarray,
    ) -> None:
        """HUD panel with items should not be uniformly dark."""
        level = LevelBuilder(MAP_SIZE, MAP_SIZE).build("hud_content")
        level.player_inventory = [(int(ItemType.IRON), 42)]
        p = EnvParams(
            map_width=MAP_SIZE,
            map_height=MAP_SIZE,
            num_players=1,
            max_timesteps=10,
            max_biters=1,
        )
        state = build_state(level, p)
        img = np.array(
            render_hud(
                state, block_atlas, machine_atlas, player_sprite,
                item_colors, digit_atlas,
            )
        )
        # The bottom half (HUD) should have some bright pixels.
        map_px = MAP_SIZE * TILE_PX
        hud_region = img[map_px:, :, :]
        assert hud_region.max() > 50

    def test_vmap_hud(
        self,
        params: EnvParams,
        block_atlas: jnp.ndarray,
        machine_atlas: jnp.ndarray,
        player_sprite: jnp.ndarray,
        item_colors: jnp.ndarray,
        digit_atlas: jnp.ndarray,
    ) -> None:
        """Vmapped full HUD render should produce correct batch shape."""
        n = 2
        _, states = make_batched_envs(n, params)
        vmap_render = jax.jit(
            jax.vmap(
                render_hud,
                in_axes=(0, None, None, None, None, None),
            )
        )
        imgs = vmap_render(
            states, block_atlas, machine_atlas, player_sprite,
            item_colors, digit_atlas,
        )
        map_px = MAP_SIZE * TILE_PX
        assert imgs.shape == (n, 2 * map_px, map_px, 3)
