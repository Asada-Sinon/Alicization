"""Directional retina vision (M1). Each agent sees the world in `retina_sectors`
angular wedges around its own heading. Per sector it gets four channels:

    food      -- plant energy sampled a short way ahead in that direction
    prey      -- nearest more-herbivorous neighbour (something it could eat)
    predator  -- nearest more-carnivorous neighbour (something that could eat it)
    water     -- proximity of the stream sampled a short way ahead

plus its own energy, diet, and water level. Prey/predator are *relative* to the
viewer's own diet, so one brain reads the same retina whether it's grazer or
hunter. This replaces the old scalar gradient 'smell'.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import Config
from .ecology import stream_dist
from .state import WorldState, pos_to_cell


def sense(state: WorldState, nbr: jax.Array, delta: jax.Array, dist: jax.Array,
          valid: jax.Array, cfg: Config) -> jax.Array:
    """Build brain inputs [n_max, in_dim] = [food(R), prey(R), predator(R),
    energy, diet].  nbr/delta/dist/valid come from spatial.geometry.
    """
    n = cfg.n_max
    R = cfg.retina_sectors
    two_pi = 2.0 * jnp.pi
    width = two_pi / R

    # --- agent channels: bin neighbours into sectors relative to own heading ---
    safe = jnp.clip(nbr, 0, n - 1)
    diet_j = state.diet[safe]                                  # [n, M]
    di = state.diet[:, None]                                   # [n, 1]
    bearing = jnp.arctan2(delta[..., 1], delta[..., 0]) - state.heading[:, None]
    bearing = jnp.mod(bearing, two_pi)                         # [n, M]
    sector = jnp.clip(jnp.floor(bearing / width).astype(jnp.int32), 0, R - 1)
    closeness = jnp.clip(1.0 - dist / cfg.vision_radius, 0.0, 1.0) * valid
    prey_val = closeness * jnp.maximum(di - diet_j, 0.0)       # j more herbivorous
    pred_val = closeness * jnp.maximum(diet_j - di, 0.0)       # j more carnivorous

    prey_cols, pred_cols = [], []
    for s in range(R):
        m = sector == s
        prey_cols.append(jnp.max(jnp.where(m, prey_val, 0.0), axis=1))
        pred_cols.append(jnp.max(jnp.where(m, pred_val, 0.0), axis=1))
    prey = jnp.stack(prey_cols, axis=1)                        # [n, R]
    pred = jnp.stack(pred_cols, axis=1)                        # [n, R]

    # --- food/water channels: sample a point ahead in each sector direction ---
    ang = state.heading[:, None] + (jnp.arange(R)[None, :] + 0.5) * width  # [n, R]
    offset = jnp.stack([jnp.cos(ang), jnp.sin(ang)], axis=2) * cfg.food_sample_dist
    sample = jnp.mod(state.pos[:, None, :] + offset, cfg.world_size)       # [n, R, 2]
    cells = pos_to_cell(sample.reshape(-1, 2), cfg).reshape(n, R)
    food = state.plant[cells] / cfg.plant_max                  # [n, R]

    sd = stream_dist(sample.reshape(-1, 2), cfg).reshape(n, R)
    water_ch = jnp.clip(1.0 - sd / cfg.stream_half_width, 0.0, 1.0)  # [n, R]

    energy = jnp.tanh(state.energy / cfg.energy_scale)[:, None]
    own_water = jnp.tanh(state.water / cfg.water_scale)[:, None]
    return jnp.concatenate(
        [food, prey, pred, water_ch, energy, di, own_water], axis=1
    )                                                            # [n, 4R+3]
