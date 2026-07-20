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


def regrow(field: jax.Array, capacity: jax.Array, rate: float, baseline_rate: float,
           ref_max: float) -> jax.Array:
    """Logistic regrowth toward a per-cell carrying capacity, plus a small
    spontaneous baseline so grazed-out cells can recover.

    `capacity` is a field rather than a scalar. The baseline is scaled by
    capacity too -- otherwise bare rock and open water would sprout out of
    nothing. Parameterised on (rate, baseline_rate, ref_max) so the grass and
    fruit layers share one implementation: they differ only in how fast they come
    back, and fruit's slowness is exactly what makes a remembered patch worth
    something.
    """
    safe = jnp.maximum(capacity, 1e-6)
    growth = rate * field * (1.0 - field / safe)
    baseline = baseline_rate * (capacity / ref_max)
    field = field + growth + baseline
    return jnp.clip(field, 0.0, capacity)
