"""Per-step telemetry. Kept as a NamedTuple of scalars so `lax.scan` can stack a
whole run into time series for plotting / emergence checks.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import Config
from .state import (WorldState, armor_of, attack_range_of, escape_of, invest_of,
                    pos_to_cell, size_of, spike_of)


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
    # G-matrix off-diagonals: does selection couple body size to the other two
    # evolving traits? Computed the same long way as `invest_diet_corr` (Pearson
    # over the living only, `denom` is the living count) so all three trait-pair
    # correlations share one estimator. Appended, never inserted.
    size_diet_corr: jax.Array   # do carnivores carry a different body size?
    size_invest_corr: jax.Array # is size allometrically tied to per-birth investment?
    # Red-queen co-evolution (docs/attack_range_redqueen.md). The *lineage-split*
    # means are the whole point: a real arms race shows attack climbing among
    # carnivores AND escape climbing among herbivores, in step. Population means
    # blur that, so `carn_attack`/`herb_escape` -- the functional carriers -- are
    # reported alongside. Appended, never inserted (run_headless reads by name).
    mean_attack: jax.Array      # population mean of the evolved attack-range gene
    attack_std: jax.Array       # spread -- a live arms race keeps this from collapsing
    carn_attack: jax.Array      # carnivore-lineage mean (diet > 0.65): the real reach
    mean_escape: jax.Array      # population mean of the evolved escape gene
    escape_std: jax.Array
    herb_escape: jax.Array      # herbivore-lineage mean (diet < 0.35): the real evasion
    hunt_success: jax.Array     # fraction of carnivores that landed a bite this step --
    #                             the predation-success rate whose fall as prey escape
    #                             rises is the arms race's signature
    # Day-night clock (docs/day_night.md): the global phase in [0, 1) at this step,
    # 0/1 = midnight, 0.5 = midday. Stacked per-step by the scan so the diel
    # commuting test can bin `carn_water_dist` into day-half vs night-half offline
    # -- a static mean averages the two and would hide a perfect commute. Constant 0
    # when day_length=0. Appended, never inserted (run_headless reads by name).
    phase: jax.Array
    # Morphological defences (docs/trait_defense_catalog.md,
    # docs/trait_addition_feasibility.md). Like the red-queen genes, the herbivore-
    # lineage mean is the functional carrier (prey are the ones defending), and the
    # carnivore-lineage mean is the control that should stay ~0 if the (1-diet) tax
    # gate is working. Appended, never inserted (run_headless reads by name).
    mean_armor: jax.Array       # population mean of the evolved armour gene
    armor_std: jax.Array        # spread -- collapsed near 0 is the "no defence" signature
    herb_armor: jax.Array       # herbivore-lineage mean (diet < 0.35): the real armour
    carn_armor: jax.Array       # carnivore-lineage mean: diet-gate control, expect ~0
    mean_spike: jax.Array
    spike_std: jax.Array
    herb_spike: jax.Array       # herbivore-lineage mean (defensive envenomation)
    carn_spike: jax.Array       # carnivore-lineage mean (offensive bite bonus)
    mean_venom: jax.Array       # mean active envenomation debuff over the living --
    #                             shows the herbivore->carnivore retaliation firing
    carrion_total: jax.Array    # standing corpse mass (docs/multispecies_feasibility.md
    #                             §4); 0 when carrion_enabled is off


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
    d_size = (size - mean_size) * alive
    invest_var = jnp.sum(d_inv * d_inv) / denom
    cov = jnp.sum(d_inv * d_diet) / denom
    corr = cov / jnp.maximum(jnp.sqrt(invest_var * diet_var), 1e-8)
    # G-matrix off-diagonals involving size, same estimator as `corr` above.
    cov_sd = jnp.sum(d_size * d_diet) / denom
    cov_si = jnp.sum(d_size * d_inv) / denom
    size_diet_corr = cov_sd / jnp.maximum(jnp.sqrt(size_var * diet_var), 1e-8)
    size_invest_corr = cov_si / jnp.maximum(jnp.sqrt(size_var * invest_var), 1e-8)

    # Red-queen genes: population mean/spread plus the lineage that actually uses
    # each (carnivores hunt, herbivores flee), computed with the same is_carn/is_herb
    # masks as carn_speed/herb_speed.
    attack = attack_range_of(state.genome, cfg)
    escape = escape_of(state.genome, cfg)
    mean_attack = jnp.sum(attack * alive) / denom
    attack_var = jnp.sum(((attack - mean_attack) ** 2) * alive) / denom
    mean_escape = jnp.sum(escape * alive) / denom
    escape_var = jnp.sum(((escape - mean_escape) ** 2) * alive) / denom
    carn_attack = jnp.sum(attack * is_carn) / carn_n
    herb_escape = jnp.sum(escape * is_herb) / herb_n
    hunt_success = jnp.sum((state.last_meat > 0.0) * is_carn) / carn_n

    # Morphological defences: population mean/spread plus the lineage split, same
    # is_carn/is_herb masks as the red-queen genes above.
    armor = armor_of(state.genome, cfg)
    spike = spike_of(state.genome, cfg)
    mean_armor = jnp.sum(armor * alive) / denom
    armor_var = jnp.sum(((armor - mean_armor) ** 2) * alive) / denom
    mean_spike = jnp.sum(spike * alive) / denom
    spike_var = jnp.sum(((spike - mean_spike) ** 2) * alive) / denom
    herb_armor = jnp.sum(armor * is_herb) / herb_n
    carn_armor = jnp.sum(armor * is_carn) / carn_n
    herb_spike = jnp.sum(spike * is_herb) / herb_n
    carn_spike = jnp.sum(spike * is_carn) / carn_n

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
        size_diet_corr=size_diet_corr,
        size_invest_corr=size_invest_corr,
        mean_attack=mean_attack,
        attack_std=jnp.sqrt(attack_var),
        carn_attack=carn_attack,
        mean_escape=mean_escape,
        escape_std=jnp.sqrt(escape_var),
        herb_escape=herb_escape,
        hunt_success=hunt_success,
        phase=state.phase,
        mean_armor=mean_armor,
        armor_std=jnp.sqrt(armor_var),
        herb_armor=herb_armor,
        carn_armor=carn_armor,
        mean_spike=mean_spike,
        spike_std=jnp.sqrt(spike_var),
        herb_spike=herb_spike,
        carn_spike=carn_spike,
        mean_venom=jnp.sum(state.venom * alive) / denom,
        carrion_total=jnp.sum(state.carrion),
    )
