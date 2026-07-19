"""Per-step telemetry. Kept as a NamedTuple of scalars so `lax.scan` can stack a
whole run into time series for plotting / emergence checks.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import Config
from .state import WorldState, pos_to_cell


class Metrics(NamedTuple):
    population: jax.Array
    mean_energy: jax.Array
    mean_age: jax.Array
    plant_total: jax.Array
    mean_diet: jax.Array        # 0 herbivore .. 1 carnivore
    carnivore_frac: jax.Array   # fraction of living agents with diet > 0.5
    diet_std: jax.Array         # spread of diet (high => bimodal herb/carn split)
    carn_speed: jax.Array       # mean |vel| of carnivores (diet > 0.65): ambush vs pursuit tell
    herb_speed: jax.Array       # mean |vel| of herbivores (diet < 0.35)
    mean_water: jax.Array       # hydration level, separate resource from energy
    mean_elevation: jax.Array   # mean terrain height under the population
    forest_frac: jax.Array      # fraction of the population standing under canopy


def compute(state: WorldState, terrain, cfg: Config) -> Metrics:
    alive = state.alive.astype(jnp.float32)
    pop = jnp.sum(alive)
    denom = jnp.maximum(pop, 1.0)
    cell = pos_to_cell(state.pos, cfg)

    mean_diet = jnp.sum(state.diet * alive) / denom
    diet_var = jnp.sum(((state.diet - mean_diet) ** 2) * alive) / denom

    speed = jnp.linalg.norm(state.vel, axis=1)
    is_carn = (state.diet > 0.65) * alive
    is_herb = (state.diet < 0.35) * alive
    carn_n = jnp.maximum(jnp.sum(is_carn), 1.0)
    herb_n = jnp.maximum(jnp.sum(is_herb), 1.0)

    return Metrics(
        population=pop,
        mean_energy=jnp.sum(state.energy * alive) / denom,
        mean_age=jnp.sum(state.age * alive) / denom,
        plant_total=jnp.sum(state.plant),
        mean_diet=mean_diet,
        carnivore_frac=jnp.sum((state.diet > 0.5) * alive) / denom,
        diet_std=jnp.sqrt(diet_var),
        carn_speed=jnp.sum(speed * is_carn) / carn_n,
        herb_speed=jnp.sum(speed * is_herb) / herb_n,
        mean_water=jnp.sum(state.water * alive) / denom,
        mean_elevation=jnp.sum(terrain.height[cell] * alive) / denom,
        forest_frac=jnp.sum((terrain.forest[cell] > 0.5) * alive) / denom,
    )
