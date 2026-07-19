"""Static terrain: one elevation field from which mountains, rivers and forest
all follow.

The design intent is that these are not three pasted-on rules but three
consequences of a single model:

    mountains -- the elevation field itself, a gaussian ridge along a meandering
                 centerline, with peaks and passes modulated along its length
    rivers    -- the steepest-descent paths of that field, traced from sources
                 near the crest down to the antipodal sea
    forest    -- what grows at mid elevation within reach of water

Everything here is computed **once** in `build()` and then closed over by the
jitted step, so it is constant-folded and never rides in the `lax.scan` carry.
Nothing in this module runs per step.

Torus care: the ridge centerline uses an integer wavenumber so it meets itself
across the seam, and every distance goes through `_wrap`.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import Config


class Terrain(NamedTuple):
    height: jax.Array      # f32 [n_cells]   elevation, ~[-basin_depth, ridge_height]
    grad_x: jax.Array      # f32 [n_cells]   d(height)/dx per *world unit*
    grad_y: jax.Array      # f32 [n_cells]
    water_dist: jax.Array  # f32 [n_cells]   distance to nearest river/sea, world units
    forest: jax.Array      # f32 [n_cells]   canopy density in [0, 1]
    rock: jax.Array        # f32 [n_cells]   bare-rock fraction in [0, 1]
    capacity: jax.Array    # f32 [n_cells]   plant carrying capacity (replaces plant_max)
    rivers: jax.Array      # f32 [n_rivers, river_steps, 2]  traced polylines


def _wrap(d: jax.Array, size: float) -> jax.Array:
    """Shortest signed offset on a periodic axis."""
    return (d + size / 2.0) % size - size / 2.0


def ridge_center_y(x: jax.Array, cfg: Config) -> jax.Array:
    """The meandering centerline of the range: y as a function of x."""
    phase = 2.0 * jnp.pi * cfg.ridge_wavenumber * x / cfg.world_size
    return cfg.ridge_base_y + cfg.ridge_amplitude * jnp.sin(phase)


def height_at(pos: jax.Array, cfg: Config) -> jax.Array:
    """Analytic elevation at `[N, 2]` continuous positions.

    Two superposed terms:
      * a sharp gaussian ridge -- the mountains, steep and local
      * a gentle regional fall from the range to the antipodal band -- without
        it the plains would be perfectly flat and rivers would stall the moment
        they left the mountain flank
    """
    x, y = pos[:, 0], pos[:, 1]
    d = jnp.abs(_wrap(y - ridge_center_y(x, cfg), cfg.world_size))

    # peaks and passes along the length of the range
    peak_phase = 2.0 * jnp.pi * cfg.ridge_peak_wavenumber * x / cfg.world_size
    h_local = cfg.ridge_height * (
        1.0 - cfg.ridge_peak_depth * (0.5 + 0.5 * jnp.cos(peak_phase))
    )
    ridge = h_local * jnp.exp(-(d ** 2) / (2.0 * cfg.ridge_sigma ** 2))

    # regional drainage: 0 at the crest, -basin_depth at the antipode, flat-ish
    # in between so the plains stay plains
    t = (1.0 - jnp.cos(jnp.pi * d / cfg.half_world)) * 0.5
    basin = cfg.basin_depth * t ** cfg.basin_power

    return ridge - basin


def _cell_centers(cfg: Config) -> jax.Array:
    """World-space centre of every plant-grid cell, in cell index order iy*g+ix."""
    g = cfg.grid
    c = (jnp.arange(g) + 0.5) * cfg.cell_size
    ix, iy = jnp.meshgrid(c, c, indexing="xy")     # [g, g] with row = iy
    return jnp.stack([ix.reshape(-1), iy.reshape(-1)], axis=1)


def _grad_at(pos: jax.Array, cfg: Config) -> jax.Array:
    """Analytic-ish gradient by central difference in world space, `[N, 2]`.

    Used while tracing rivers, where we need the slope at arbitrary continuous
    points rather than at cell centres.
    """
    e = cfg.cell_size * 0.5
    ex = jnp.array([e, 0.0])
    ey = jnp.array([0.0, e])
    hx1 = height_at(jnp.mod(pos + ex, cfg.world_size), cfg)
    hx0 = height_at(jnp.mod(pos - ex, cfg.world_size), cfg)
    hy1 = height_at(jnp.mod(pos + ey, cfg.world_size), cfg)
    hy0 = height_at(jnp.mod(pos - ey, cfg.world_size), cfg)
    return jnp.stack([(hx1 - hx0) / (2 * e), (hy1 - hy0) / (2 * e)], axis=1)


def trace_rivers(cfg: Config) -> jax.Array:
    """Follow steepest descent from `n_rivers` sources near the crest.

    Returns `[n_rivers, river_steps, 2]`. Sources are spread along the range and
    placed alternately on either flank, so rivers drain both halves of the world
    rather than all running to the same side.

    Fixed step count and fixed step length keep the shape static, so this stays
    jit-friendly even though it only ever runs once.
    """
    k = jnp.arange(cfg.n_rivers)
    src_x = (k + 0.5) * cfg.world_size / cfg.n_rivers
    side = jnp.where(k % 2 == 0, 1.0, -1.0)          # alternate flanks
    # start just off the crest: on the crest itself the cross-ridge gradient is
    # zero by symmetry and descent has no direction to pick
    src_y = ridge_center_y(src_x, cfg) + side * cfg.ridge_sigma * 0.5
    start = jnp.stack([src_x, jnp.mod(src_y, cfg.world_size)], axis=1)

    def step(p, _):
        g = _grad_at(p, cfg)
        n = jnp.linalg.norm(g, axis=1, keepdims=True)
        direction = -g / jnp.maximum(n, 1e-8)
        nxt = jnp.mod(p + direction * cfg.river_step_len, cfg.world_size)
        # Freeze a river once it reaches the sea or runs out of slope. Without
        # this it keeps spending its step budget wandering the flat sea bottom,
        # where the gradient is numerical noise -- which shows up as the river
        # flowing *uphill*. Freezing keeps the shape static (no early exit) while
        # making every recorded step a genuine descent.
        done = (height_at(p, cfg) < cfg.sea_level) | (n[:, 0] < 1e-6)
        p = jnp.where(done[:, None], p, nxt)
        return p, p

    _, path = jax.lax.scan(step, start, None, length=cfg.river_steps)
    return jnp.transpose(path, (1, 0, 2))            # [n_rivers, river_steps, 2]


def _dist_to_rivers(points: jax.Array, rivers: jax.Array, cfg: Config) -> jax.Array:
    """Min torus distance from each of `[N, 2]` points to any traced river point.

    Chunked one river at a time so the full [N, n_rivers*river_steps] pair matrix
    never materialises, and so the chunking divides exactly however `n_rivers`
    and `river_steps` are configured.
    """
    best = jnp.full((points.shape[0],), jnp.inf)

    def body(i, best):
        rp = rivers[i]                                            # [river_steps, 2]
        d = _wrap(points[:, None, :] - rp[None, :, :], cfg.world_size)
        dist = jnp.sqrt(jnp.sum(d * d, axis=2) + 1e-12)           # [N, river_steps]
        return jnp.minimum(best, jnp.min(dist, axis=1))

    return jax.lax.fori_loop(0, rivers.shape[0], body, best)


def build(cfg: Config) -> Terrain:
    """Compute every static terrain field once."""
    centers = _cell_centers(cfg)
    height = height_at(centers, cfg)

    # world-unit gradient on the grid, via the torus central difference that
    # ecology.gradient already implements for scalar fields
    from .ecology import gradient
    gx_cell, gy_cell = gradient(height, cfg)
    grad_x = gx_cell / cfg.cell_size
    grad_y = gy_cell / cfg.cell_size

    rivers = trace_rivers(cfg)
    d_river = _dist_to_rivers(centers, rivers, cfg)

    # Open water is the antipodal lowland; treat any sub-sea-level cell as being
    # at distance zero from water, so drinking and the water sense read the sea
    # and the rivers through one field.
    is_sea = height < cfg.sea_level
    water_dist = jnp.where(is_sea, 0.0, d_river)

    # forest: a gaussian band in elevation, fading away from water
    elev_band = jnp.exp(
        -((height - cfg.forest_elev) ** 2) / (2.0 * cfg.forest_elev_sigma ** 2)
    )
    water_prox = jnp.exp(-water_dist / cfg.forest_water_scale)
    forest = elev_band * water_prox
    forest = jnp.where(is_sea, 0.0, forest)           # no canopy on open water

    # bare rock on the peaks
    t = jnp.clip((height - cfg.rock_h0) / (cfg.rock_h1 - cfg.rock_h0), 0.0, 1.0)
    rock = t * t * (3.0 - 2.0 * t)                    # smoothstep

    fertility = cfg.grass_base + cfg.forest_bonus * forest
    capacity = cfg.plant_max * fertility * (1.0 - rock)
    capacity = jnp.where(is_sea, 0.0, capacity)       # nothing grazes open water

    return Terrain(
        height=height, grad_x=grad_x, grad_y=grad_y, water_dist=water_dist,
        forest=forest, rock=rock, capacity=capacity, rivers=rivers,
    )
