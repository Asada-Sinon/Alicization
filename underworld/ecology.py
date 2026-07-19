"""The plant layer: a scalar energy field on the coarse grid.

Cheap and embarrassingly parallel -- agents read the cell under them, and the
field regrows logistically toward a *per-cell* carrying capacity supplied by the
terrain (forest is fertile, bare rock and open water grow nothing).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import Config


def gradient(field: jax.Array, cfg: Config):
    """Central-difference gradient of a flat [n_cells] field on the torus.

    Returns (gx, gy) as flat [n_cells] arrays, matching cell indexing iy*grid+ix.
    Note these are per-*cell* differences: divide by `cfg.cell_size` for a slope
    in world units. Used by `terrain.build` for the elevation gradient.
    """
    p = field.reshape(cfg.grid, cfg.grid)          # [iy, ix]
    gx = 0.5 * (jnp.roll(p, -1, axis=1) - jnp.roll(p, 1, axis=1))  # d/dx along columns
    gy = 0.5 * (jnp.roll(p, -1, axis=0) - jnp.roll(p, 1, axis=0))  # d/dy along rows
    return gx.reshape(-1), gy.reshape(-1)


def regrow(plant: jax.Array, capacity: jax.Array, cfg: Config) -> jax.Array:
    """Logistic regrowth toward a per-cell carrying capacity, plus a small
    spontaneous baseline so grazed-out cells can recover.

    Same logistic form as before; `capacity` is simply a field now instead of the
    scalar `plant_max`. The baseline is scaled by capacity too -- otherwise bare
    rock and open water would sprout grass out of nothing.
    """
    safe = jnp.maximum(capacity, 1e-6)
    growth = cfg.regrow_rate * plant * (1.0 - plant / safe)
    baseline = cfg.regrow_baseline * (capacity / cfg.plant_max)
    plant = plant + growth + baseline
    return jnp.clip(plant, 0.0, capacity)
