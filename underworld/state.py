"""The world state: one big pytree of fixed-shape arrays.

Everything about every (potential) agent is a row in these `[n_max, ...]` arrays.
Living vs. dead is tracked purely by the boolean `alive` mask, so births and
deaths never resize an array -- they just flip mask bits and scatter into slots.
`WorldState` is a NamedTuple, which JAX automatically treats as a pytree.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import Config


class WorldState(NamedTuple):
    alive: jax.Array       # bool   [n_max]
    pos: jax.Array         # f32    [n_max, 2]   position in world units
    heading: jax.Array     # f32    [n_max]      facing angle (radians)
    vel: jax.Array         # f32    [n_max, 2]   last velocity (for render/metrics)
    energy: jax.Array      # f32    [n_max]
    water: jax.Array       # f32    [n_max]      hydration (separate from energy)
    age: jax.Array         # f32    [n_max]      steps lived
    genome: jax.Array      # f32    [n_max, genome_size]
    hue: jax.Array         # f32    [n_max]      lineage colour in [0, 1)
    diet: jax.Array        # f32    [n_max]      0 herbivore .. 1 carnivore (cached)
    generation: jax.Array  # f32    [n_max]      lineage depth
    last_food: jax.Array   # f32    [n_max]      plant energy gained last step
    last_meat: jax.Array   # f32    [n_max]      predation energy gained last step
    last_damage: jax.Array # f32    [n_max]      energy lost to predators last step
    last_drink: jax.Array  # f32    [n_max]      water gained last step (stream + meat)
    hidden: jax.Array      # f32    [n_max, hidden]     recurrent brain state (memory)
    last_input: jax.Array  # f32    [n_max, in_dim]     last retina input (inspector)
    last_output: jax.Array # f32    [n_max, out_dim]    last brain output (inspector)
    plant: jax.Array       # f32    [n_cells]    plant energy field (flattened grid)


def diet_of(genome: jax.Array, cfg: Config) -> jax.Array:
    """Map the diet gene to [0, 1]: 0 = pure herbivore, 1 = pure carnivore."""
    return jax.nn.sigmoid(genome[:, cfg.diet_index])


def init_state(cfg: Config, key: jax.Array) -> WorldState:
    k_pos, k_head, k_gen, k_hue, k_plant, k_carn = jax.random.split(key, 6)
    n = cfg.n_max

    alive = jnp.arange(n) < cfg.n_init
    pos = jax.random.uniform(k_pos, (n, 2), maxval=cfg.world_size)
    heading = jax.random.uniform(k_head, (n,), maxval=2.0 * jnp.pi)
    vel = jnp.zeros((n, 2))
    energy = jnp.where(alive, cfg.energy_init, 0.0)
    water = jnp.where(alive, cfg.water_init, 0.0)
    age = jnp.zeros(n)

    genome = jax.random.normal(k_gen, (n, cfg.genome_size)) * cfg.genome_init_scale
    # Seed the diet gene *bimodally*: a herbivore cluster (sigmoid ~0.08) and a
    # carnivore cluster (sigmoid ~0.88), so two distinct types exist from step 0.
    r = jax.random.uniform(k_carn, (n,))
    diet_gene = jnp.where(r < cfg.carnivore_init_frac, 2.0, -2.5)
    diet_gene = diet_gene + 0.3 * genome[:, cfg.diet_index]  # a little spread
    genome = genome.at[:, cfg.diet_index].set(diet_gene)

    hue = jax.random.uniform(k_hue, (n,))
    diet = diet_of(genome, cfg)
    zeros = jnp.zeros(n)

    # Patchy initial plant field: random per-cell energy around the mean.
    plant = jax.random.uniform(
        k_plant, (cfg.n_cells,), minval=0.0, maxval=2.0 * cfg.plant_init
    )
    plant = jnp.clip(plant, 0.0, cfg.plant_max)

    return WorldState(
        alive=alive, pos=pos, heading=heading, vel=vel, energy=energy, water=water,
        age=age, genome=genome, hue=hue, diet=diet, generation=zeros,
        last_food=zeros, last_meat=zeros, last_damage=zeros, last_drink=zeros,
        hidden=jnp.zeros((n, cfg.hidden)),
        last_input=jnp.zeros((n, cfg.in_dim)),
        last_output=jnp.zeros((n, cfg.out_dim)),
        plant=plant,
    )


def pos_to_cell(pos: jax.Array, cfg: Config, grid: int | None = None) -> jax.Array:
    """Map [N, 2] world positions to flat cell indices on a `grid`x`grid` torus
    (defaults to the plant grid). Cell index is iy*grid+ix.
    """
    g = cfg.grid if grid is None else grid
    ij = jnp.floor(pos / (cfg.world_size / g)).astype(jnp.int32) % g  # [N,2] (ix,iy)
    ix, iy = ij[:, 0], ij[:, 1]
    return iy * g + ix
