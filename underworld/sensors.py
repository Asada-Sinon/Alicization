"""Directional retina vision (M1). Each agent sees the world in `retina_sectors`
angular wedges around its own heading. Per sector it gets six channels:

    food      -- plant energy sampled a short way ahead in that direction
    prey      -- nearest more-herbivorous neighbour (something it could eat)
    predator  -- nearest more-carnivorous neighbour (something that could eat it)
    water     -- proximity of a river or the sea, sampled a short way ahead
    slope     -- elevation *relative to the agent's own*, sampled ahead
    peer      -- nearest neighbour of *similar* diet, regardless of who could
                 eat whom

plus its own energy, diet, and water level. Prey/predator are *relative* to the
viewer's own diet, so one brain reads the same retina whether it's grazer or
hunter. Slope is likewise relative rather than absolute height: what an agent can
act on is "is it uphill that way", which is exactly what it pays climb_cost for.

Prey/predator are diet-*difference* signals, so for two agents at nearly the
same diet both are ~0 -- conspecifics of similar diet are mutually invisible.
That silently blocked every form of social behaviour (following, group cohesion,
juveniles learning water locations from adults) because the input layer simply
carried no signal to learn from, no matter how the brain evolved. `peer` is
diet-*similarity*, the complementary construction, so it is maximal exactly
where prey/pred both vanish.

Under dense canopy the effective vision radius shrinks (`forest_occlusion`) while
`attack_range` does not -- short sight with unchanged reach is what makes forest
genuine ambush cover.

Optionally (`los_occlusion_enabled`, default off), a candidate within vision
range is still invisible if the terrain between it and the observer rises above
the straight line between their two heights -- a mountain blocks sight, docs/
three_d.md S5.1. This only zeroes `closeness` for prey/pred/peer; it adds no
retina channel and does not change `in_dim`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from . import memory
from .config import Config
from .state import WorldState, pos_to_cell


def sense(state: WorldState, nbr: jax.Array, delta: jax.Array, dist: jax.Array,
          valid: jax.Array, terrain, cfg: Config) -> jax.Array:
    """Build brain inputs [n_max, in_dim] = [food(R), prey(R), predator(R),
    water(R), slope(R), energy, diet, own_water].  nbr/delta/dist/valid come from
    spatial.geometry.
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

    # Canopy at the *viewer's* cell shortens how far it can see. This only ever
    # shrinks the radius, so it stays inside what the neighbour table already
    # gathered -- no risk of needing candidates the 3x3 block never collected.
    own_cell = pos_to_cell(state.pos, cfg)
    vision = cfg.vision_radius * (1.0 - cfg.forest_occlusion * terrain.forest[own_cell])
    closeness = jnp.clip(1.0 - dist / vision[:, None], 0.0, 1.0) * valid

    # Terrain line-of-sight (docs/three_d.md S5.1): a candidate otherwise inside
    # vision range is invisible if a mountain sits between it and the observer.
    # `los_occlusion_enabled` is a Python bool baked into the jit (Config is
    # closed over, not traced), so this whole block is absent from the trace --
    # not merely zeroed -- when off, matching the `trample_impact`-style
    # ablation convention: the mechanism doesn't exist until switched on.
    if cfg.los_occlusion_enabled:
        # `delta` (from spatial.geometry) is already the shortest torus vector
        # observer->candidate, so interior points on the line are obtained by
        # scaling it, and re-wrapping is the *only* place the torus needs
        # handling -- never recompute from absolute positions.
        cand_pos = jnp.mod(state.pos[:, None, :] + delta, cfg.world_size)  # [n, M, 2]
        m = cand_pos.shape[1]
        cand_h = terrain.height[pos_to_cell(cand_pos.reshape(-1, 2), cfg)].reshape(n, m)
        own_h = terrain.height[own_cell]                                   # [n]

        s = cfg.los_samples
        frac = jnp.arange(1, s + 1, dtype=jnp.float32) / (s + 1)           # [S]
        sample_pos = jnp.mod(
            state.pos[:, None, None, :]
            + delta[:, :, None, :] * frac[None, None, :, None],
            cfg.world_size,
        )                                                                  # [n, M, S, 2]
        sample_h = terrain.height[
            pos_to_cell(sample_pos.reshape(-1, 2), cfg)
        ].reshape(n, m, s)

        # A straight line-of-sight would see the linear interpolation between
        # the two endpoint heights; a sample rising more than `los_margin`
        # above that interpolation is a ridge poking through the sightline.
        interp_h = own_h[:, None, None] + (cand_h - own_h[:, None])[:, :, None] * frac[None, None, :]
        blocked = jnp.any(sample_h - interp_h > cfg.los_margin, axis=2)     # [n, M]
        closeness = jnp.where(blocked, 0.0, closeness)

    prey_val = closeness * jnp.maximum(di - diet_j, 0.0)       # j more herbivorous
    pred_val = closeness * jnp.maximum(diet_j - di, 0.0)       # j more carnivorous
    # Similarity rather than difference: 1.0 at diet_j == di, falling off to 0
    # at the maximum possible gap (diet is a sigmoid output, so that gap is 1.0).
    # This is what prey_val/pred_val cannot express by construction -- they are
    # built from a signed difference, so they are both exactly zero at the same
    # point where this is at its peak.
    # `peer_channel_enabled` is a Python bool baked into the jit (Config is
    # closed over, not traced), so this is a compile-time zero, not a per-step
    # branch. It is an ablation arm's control switch: the channel stays present
    # in `in_dim` (so a genome trained with it on still loads against an
    # ablation run) but contributes no signal when off.
    peer_scale = 1.0 if cfg.peer_channel_enabled else 0.0
    peer_val = closeness * (1.0 - jnp.abs(di - diet_j)) * peer_scale

    prey_cols, pred_cols, peer_cols = [], [], []
    for s in range(R):
        m = sector == s
        prey_cols.append(jnp.max(jnp.where(m, prey_val, 0.0), axis=1))
        pred_cols.append(jnp.max(jnp.where(m, pred_val, 0.0), axis=1))
        peer_cols.append(jnp.max(jnp.where(m, peer_val, 0.0), axis=1))
    prey = jnp.stack(prey_cols, axis=1)                        # [n, R]
    pred = jnp.stack(pred_cols, axis=1)                        # [n, R]
    peer = jnp.stack(peer_cols, axis=1)                        # [n, R]

    # --- field channels: sample a point ahead in each sector direction ---
    ang = state.heading[:, None] + (jnp.arange(R)[None, :] + 0.5) * width  # [n, R]
    offset = jnp.stack([jnp.cos(ang), jnp.sin(ang)], axis=2) * cfg.food_sample_dist
    sample = jnp.mod(state.pos[:, None, :] + offset, cfg.world_size)       # [n, R, 2]
    cells = pos_to_cell(sample.reshape(-1, 2), cfg).reshape(n, R)
    # Fruit rides the existing food channel weighted by what it is actually
    # worth to eat, rather than claiming a sixth retina channel. Both are food;
    # what an agent needs to see is edible energy ahead. Keeping them in one
    # channel also leaves `in_dim` -- and therefore every evolved genome --
    # untouched by this layer.
    edible = state.plant[cells] + cfg.fruit_energy * state.fruit[cells]
    food = edible / cfg.plant_max                              # [n, R]

    wd = terrain.water_dist[cells]                             # [n, R]
    water_ch = jnp.clip(1.0 - wd / cfg.vision_radius, 0.0, 1.0)

    # Slope: elevation ahead relative to here. Scaled so a full ridge-height
    # difference saturates tanh rather than sitting in its linear toe.
    dh = terrain.height[cells] - terrain.height[own_cell][:, None]
    slope = jnp.tanh(dh / (0.25 * cfg.ridge_height))            # [n, R]

    energy = jnp.tanh(state.energy / cfg.energy_scale)[:, None]
    own_water = jnp.tanh(state.water / cfg.water_scale)[:, None]
    # Memory goes on the *end* of the vector: `server/app.py:_build_detail`
    # slices the retina channels by leading offset, and appending keeps those
    # slices valid. `peer` is appended after memory for the same reason --
    # it was added later than the original five retina channels, and putting
    # it here rather than interleaved keeps every existing `li[a*r:b*r]` slice
    # in `server/app.py` correct without renumbering.
    mem = memory.encode(state.memory, state.heading, cfg)        # [n, 4*slots]
    return jnp.concatenate(
        [food, prey, pred, water_ch, slope, energy, di, own_water, mem, peer],
        axis=1,
    )                                                    # [n, 5R+3+4*slots+R]
