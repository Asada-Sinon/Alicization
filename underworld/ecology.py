"""The plant layer: a scalar energy field on the coarse grid.

Cheap and embarrassingly parallel -- agents read the cell under them, and the
field regrows logistically toward its carrying capacity. The spatial gradient of
this field is what agents 'smell' in M0.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import Config
from .state import WorldState, pos_to_cell


def gradient(field: jax.Array, cfg: Config):
    """Central-difference gradient of a flat [n_cells] field on the torus.

    Returns (gx, gy) as flat [n_cells] arrays, matching cell indexing iy*grid+ix.
    Used for both the plant field and the prey-density field.
    """
    p = field.reshape(cfg.grid, cfg.grid)          # [iy, ix]
    gx = 0.5 * (jnp.roll(p, -1, axis=1) - jnp.roll(p, 1, axis=1))  # d/dx along columns
    gy = 0.5 * (jnp.roll(p, -1, axis=0) - jnp.roll(p, 1, axis=0))  # d/dy along rows
    return gx.reshape(-1), gy.reshape(-1)


def stream_dist(pos: jax.Array, cfg: Config) -> jax.Array:
    """Distance from `[N, 2]` positions to a meandering stream centerline: a sine
    wave in y as a function of x, `stream_wavenumber` full periods across
    `world_size` so it tiles seamlessly on the torus. Both axes wrap.
    """
    two_pi = 2.0 * jnp.pi
    phase = two_pi * cfg.stream_wavenumber * pos[:, 0] / cfg.world_size
    center_y = cfg.stream_base_y + cfg.stream_amplitude * jnp.sin(phase)
    dy = jnp.abs(pos[:, 1] - center_y) % cfg.world_size
    dy = jnp.minimum(dy, cfg.world_size - dy)          # torus-wrapped vertical gap
    return dy


def in_stream(pos: jax.Array, cfg: Config) -> jax.Array:
    return stream_dist(pos, cfg) < cfg.stream_half_width


def regrow(plant: jax.Array, cfg: Config) -> jax.Array:
    """Logistic regrowth plus a small spontaneous baseline (so cells recover)."""
    growth = cfg.regrow_rate * plant * (1.0 - plant / cfg.plant_max)
    plant = plant + growth + cfg.regrow_baseline
    return jnp.clip(plant, 0.0, cfg.plant_max)


def prey_field(state: WorldState, cfg: Config) -> jax.Array:
    """Herbivore biomass per cell -- what carnivores hunt. Weighted by how
    herbivorous (vulnerable) each agent is and how much energy it carries.
    """
    cell = pos_to_cell(state.pos, cfg)
    mass = state.alive * (1.0 - state.diet) ** 2 * jnp.maximum(state.energy, 0.0)
    return jnp.zeros(cfg.n_cells).at[cell].add(mass)
