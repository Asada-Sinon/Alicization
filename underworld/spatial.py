"""Fixed-capacity spatial neighbour index (the M1 foundation).

Agents are binned onto a coarse `sense_grid`. Each cell keeps up to `k_neighbors`
agent indices in a `[n_sense_cells, K]` table; each agent then reads its 3x3 cell
block to get a bounded `[n, 9K]` candidate list. Everything is static-shaped and
jit-friendly: the per-cell slot of each agent is its rank within the cell, found
with an argsort + a cumulative-max trick (no dynamic loops).

This one index powers both directional vision (sensors.py) and neighbour-based
predation (dynamics.py).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import Config
from .state import WorldState, pos_to_cell

# 3x3 cell block offsets.
_OFFSETS = [(-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 0), (0, 1),
            (1, -1), (1, 0), (1, 1)]


def build_table(state: WorldState, cfg: Config) -> jax.Array:
    """Return `[n_sense_cells, K]` of agent indices per cell (-1 = empty)."""
    n = cfg.n_max
    g = cfg.sense_grid
    K = cfg.k_neighbors
    n_cells = cfg.n_sense_cells

    cell = pos_to_cell(state.pos, cfg, grid=g)
    # Dead agents go to a dump cell so they never appear as neighbours.
    cell = jnp.where(state.alive, cell, n_cells).astype(jnp.int32)

    order = jnp.argsort(cell)                 # stable: agents grouped by cell
    sorted_cell = cell[order]
    idxs = jnp.arange(n)
    is_new = jnp.concatenate([jnp.array([True]), sorted_cell[1:] != sorted_cell[:-1]])
    seg_start = jax.lax.cummax(jnp.where(is_new, idxs, 0))
    rank = idxs - seg_start                   # 0-based position within the cell

    # Scatter into a table with an extra dump row (dead) and dump col (overflow).
    rows = sorted_cell
    cols = jnp.minimum(rank, K)               # rank >= K -> overflow dump column
    table = jnp.full((n_cells + 1, K + 1), -1, dtype=jnp.int32)
    table = table.at[rows, cols].set(order.astype(jnp.int32))
    return table[:n_cells, :K]


def gather_neighbors(state: WorldState, table: jax.Array, cfg: Config) -> jax.Array:
    """Return `[n, 9K]` candidate neighbour indices from each agent's 3x3 block."""
    g = cfg.sense_grid
    cell = pos_to_cell(state.pos, cfg, grid=g)
    cx = cell % g
    cy = cell // g
    cols = []
    for dx, dy in _OFFSETS:
        ncell = ((cy + dy) % g) * g + ((cx + dx) % g)   # [n]
        cols.append(table[ncell])                        # [n, K]
    return jnp.concatenate(cols, axis=1)                 # [n, 9K]


def geometry(state: WorldState, nbr: jax.Array, cfg: Config):
    """Relative geometry of candidates. nbr: [n, M] indices (-1 empty).

    Returns (delta [n,M,2] torus-wrapped, dist [n,M], valid [n,M]).
    """
    safe = jnp.clip(nbr, 0, cfg.n_max - 1)
    npos = state.pos[safe]                               # [n, M, 2]
    d = npos - state.pos[:, None, :]
    half = cfg.world_size / 2.0
    d = (d + half) % cfg.world_size - half               # shortest torus vector
    dist = jnp.sqrt(jnp.sum(d * d, axis=2) + 1e-9)
    self_idx = jnp.arange(cfg.n_max)[:, None]
    valid = (nbr >= 0) & (nbr != self_idx)
    return d, dist, valid
