"""Per-step telemetry. Kept as a NamedTuple of scalars so `lax.scan` can stack a
whole run into time series for plotting / emergence checks.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import Config
from .state import WorldState, invest_of, pos_to_cell, size_of


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
    # Spatial occupancy relative to water. Appended, never inserted --
    # `scripts/run_headless.py` indexes this tuple positionally.
    herb_water_dist: jax.Array  # mean distance-to-water under herbivores
    carn_water_dist: jax.Array  # mean distance-to-water under carnivores
    water_bound_frac: jax.Array # fraction standing in the drinkable band
    inland_frac: jax.Array      # fraction beyond the water sensor's reach entirely
    fruit_total: jax.Array      # standing fruit crop, the canopy's high-value layer
    # Per-offspring investment, the quantity/quality dial. The spread and the
    # diet correlation matter more than the mean: a population splitting into
    # cheap-and-many versus few-and-provisioned shows up as spread, and a mean
    # sitting still can hide it entirely.
    mean_invest: jax.Array      # mean evolved fraction of energy given per birth
    invest_std: jax.Array       # spread (high => the population is not of one mind)
    invest_diet_corr: jax.Array # do carnivores provision differently from grazers?
    # Death toll split by cause, as counts, and the summed age of the dead in
    # each class (see reproduction.Deaths). Any claim that some trait could be
    # selected for has to start here: a trait that reduces predation risk is
    # worth nothing if predation is not killing anyone. Sum over a run, then
    # divide -- per-step fractions would weight a two-death step like a
    # two-hundred-death one, and mean age is a ratio of two sums.
    death_predation: jax.Array
    death_starvation: jax.Array
    death_thirst: jax.Array
    death_senescence: jax.Array
    deathage_predation: jax.Array
    deathage_starvation: jax.Array
    deathage_thirst: jax.Array
    deathage_senescence: jax.Array
    # Body-size gene (docs/biology.md S8.2's water-economy design). The mean is
    # the whole falsifiable prediction: does selection actually move it, or does
    # it just sit at the neutral 1.0 starting point?
    mean_size: jax.Array
    size_std: jax.Array


def compute(state: WorldState, terrain, deaths, cfg: Config) -> Metrics:
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

    # How far the population actually lives from water. The retina's water
    # channel dies at `vision_radius` and each sector samples `food_sample_dist`
    # ahead, so past their sum an agent has no directional water signal at all --
    # that boundary is what `inland_frac` counts against.
    wd = terrain.water_dist[cell]
    sensor_reach = cfg.vision_radius + cfg.food_sample_dist

    # Investment gene, over the living only. The correlation is computed the
    # long way rather than with jnp.corrcoef because that would weight the dead
    # slots equally; `denom` is already the living count.
    invest = invest_of(state.genome, cfg)
    mean_invest = jnp.sum(invest * alive) / denom

    size = size_of(state.genome, cfg)
    mean_size = jnp.sum(size * alive) / denom
    size_var = jnp.sum(((size - mean_size) ** 2) * alive) / denom
    d_inv = (invest - mean_invest) * alive
    d_diet = (state.diet - mean_diet) * alive
    invest_var = jnp.sum(d_inv * d_inv) / denom
    cov = jnp.sum(d_inv * d_diet) / denom
    corr = cov / jnp.maximum(jnp.sqrt(invest_var * diet_var), 1e-8)

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
        herb_water_dist=jnp.sum(wd * is_herb) / herb_n,
        carn_water_dist=jnp.sum(wd * is_carn) / carn_n,
        water_bound_frac=jnp.sum((wd < cfg.river_half_width) * alive) / denom,
        inland_frac=jnp.sum((wd > sensor_reach) * alive) / denom,
        fruit_total=jnp.sum(state.fruit),
        mean_invest=mean_invest,
        invest_std=jnp.sqrt(invest_var),
        invest_diet_corr=corr,
        death_predation=deaths.predation.astype(jnp.float32),
        death_starvation=deaths.starvation.astype(jnp.float32),
        death_thirst=deaths.thirst.astype(jnp.float32),
        death_senescence=deaths.senescence.astype(jnp.float32),
        deathage_predation=deaths.age_predation.astype(jnp.float32),
        deathage_starvation=deaths.age_starvation.astype(jnp.float32),
        deathage_thirst=deaths.age_thirst.astype(jnp.float32),
        deathage_senescence=deaths.age_senescence.astype(jnp.float32),
        mean_size=mean_size,
        size_std=jnp.sqrt(size_var),
    )
