"""Physics & feeding: turn brain outputs into motion, then let agents graze,
hunt, and pay their metabolic bill. Feeding interactions are resolved per plant
cell (cell size ~ a bite radius), which stays cheap and fully vectorized.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import Config
from .state import WorldState, pos_to_cell


def act(state: WorldState, outputs: jax.Array, terrain, cfg: Config):
    """Apply brain outputs. Returns (new_state, thrust, climb) where thrust is in
    [0, 1] and `climb` is the elevation *gained* this step (>= 0), which
    `metabolize` charges for.

    Dense canopy slows movement: forest is food-rich but hard to cross, so it's a
    genuine trade-off rather than a free bonus.
    """
    turn = outputs[:, 0] * cfg.max_turn                 # [-max_turn, max_turn]
    thrust = 0.5 * (outputs[:, 1] + 1.0)                # [0, 1]

    cell = pos_to_cell(state.pos, cfg)
    slow = 1.0 - cfg.forest_slow * terrain.forest[cell]

    heading = jnp.mod(state.heading + turn * cfg.dt, 2.0 * jnp.pi)
    speed = thrust * cfg.max_speed * slow
    vel = jnp.stack([jnp.cos(heading), jnp.sin(heading)], axis=1) * speed[:, None]
    vel = vel * state.alive[:, None]                    # the dead don't drift
    pos = jnp.mod(state.pos + vel * cfg.dt, cfg.world_size)

    # Elevation gained along the displacement, from the terrain gradient at the
    # cell we left. Uphill only -- descending is free, but must never *produce*
    # energy or agents would farm the slope by running downhill forever.
    dh = (terrain.grad_x[cell] * vel[:, 0] + terrain.grad_y[cell] * vel[:, 1]) * cfg.dt
    climb = jnp.maximum(dh, 0.0) * state.alive

    return state._replace(heading=heading, vel=vel, pos=pos), thrust, climb


def graze(state: WorldState, cfg: Config):
    """Herbivory: agents drain the plant field under them, fairly sharing a
    cell's yield. Carnivores (high diet) barely graze. Returns (energy, plant,
    food_gain, water_gain).

    Forage carries water. Real herbivores draw much of their intake preformed
    from what they eat -- enough that species on moist forage can go without
    drinking entirely -- so a diet of nothing but dry plants was what pinned this
    population to the riverbank. The subsidy rides on `gain`, which is already
    the post-competition share, so it self-limits exactly where it should: a
    crowd grazing one cell down gets proportionally less water from it, and a
    stripped cell yields none at all.
    """
    cell = pos_to_cell(state.pos, cfg)
    # Grazing skill falls off *steeply* with diet ((1-diet)^6): even a
    # mostly-carnivorous omnivore gets almost nothing from plants, closing off
    # the "camp in one spot, graze a trickle, ambush whatever wanders past" niche
    # that a gentler taper still permitted. Above `carn_graze_cutoff`, grazing is
    # hard-zeroed -- true carnivores must hunt or starve, no plant fallback at
    # all. This both enforces a herbivore-majority pyramid and damps the
    # predator-prey oscillation.
    herbivory = jnp.where(
        state.diet > cfg.carn_graze_cutoff, 0.0, (1.0 - state.diet) ** 6
    )
    demand = cfg.eat_rate * herbivory * state.alive            # [n_max]

    demand_per_cell = jnp.zeros(cfg.n_cells).at[cell].add(demand)
    removed_per_cell = jnp.minimum(demand_per_cell, state.plant)
    frac = jnp.where(demand_per_cell > 0, removed_per_cell / demand_per_cell, 0.0)
    gain = demand * frac[cell] * cfg.eat_efficiency

    energy = state.energy + gain
    plant = state.plant - removed_per_cell
    return energy, plant, gain, gain * cfg.forage_water_frac


def eat_fruit(state: WorldState, cfg: Config):
    """Foraging the canopy's high-value layer. Structurally identical to `graze`
    -- same per-cell demand pool, same fair share, same herbivory taper -- but on
    a field that is scarce, concentrated and slow to come back.

    Returns (energy, fruit, fruit_gain, water_gain). Fruit carries water at the
    same rate as grass; it is plant tissue either way.
    """
    cell = pos_to_cell(state.pos, cfg)
    herbivory = jnp.where(
        state.diet > cfg.carn_graze_cutoff, 0.0, (1.0 - state.diet) ** 6
    )
    demand = cfg.fruit_eat_rate * herbivory * state.alive

    demand_per_cell = jnp.zeros(cfg.n_cells).at[cell].add(demand)
    removed_per_cell = jnp.minimum(demand_per_cell, state.fruit)
    frac = jnp.where(demand_per_cell > 0, removed_per_cell / demand_per_cell, 0.0)
    taken = demand * frac[cell]
    gain = taken * cfg.fruit_energy * cfg.eat_efficiency

    energy = state.energy + gain
    fruit = state.fruit - removed_per_cell
    return energy, fruit, gain, taken * cfg.forage_water_frac


def drink(state: WorldState, terrain, cfg: Config):
    """Hydration: standing in a river or at the sea refills water, uncapped by any
    shared demand pool (flowing water isn't meaningfully depleted at this scale,
    only spatially constrained). Returns (water, drink_gain).

    This is now a lookup into the precomputed `water_dist` field rather than the
    old analytic sine -- cheaper per step, and it works for the traced rivers and
    the sea alike.
    """
    cell = pos_to_cell(state.pos, cfg)
    in_water = terrain.water_dist[cell] < cfg.river_half_width
    gain = jnp.where(in_water, cfg.drink_rate, 0.0) * state.alive
    water = jnp.minimum(state.water + gain, cfg.water_max)
    return water, gain


def predation(state: WorldState, nbr: jax.Array, dist: jax.Array,
              valid: jax.Array, cfg: Config):
    """Neighbour-based carnivory. Each agent attacks the *nearest eligible prey*
    among its neighbours -- an agent that is `diet_delta` more herbivorous than
    itself, within `attack_range`. Damage is scatter-added onto prey, capped by
    the prey's energy, then redistributed to the attackers (trophic loss via
    `pred_efficiency`). A bite also draws a proportional share of the prey's
    *water* (capped by what it has), redistributed the same way -- a kill
    hydrates as well as feeds. Because a carnivore genuinely needs a weaker
    neighbour to eat, an all-carnivore population starves -- enforcing a
    herbivore-majority pyramid. Returns (energy, meat_gain, damage_taken, water,
    water_gain, water_damage_taken).
    """
    n = cfg.n_max
    safe = jnp.clip(nbr, 0, n - 1)
    d_i = state.diet[:, None]
    d_j = state.diet[safe]
    e_j = jnp.maximum(state.energy[safe], 0.0)
    eligible = valid & (d_i - d_j > cfg.diet_delta) & (dist < cfg.attack_range) & (e_j > 0.0)

    BIG = 1e9
    d_masked = jnp.where(eligible, dist, BIG)
    best = jnp.argmin(d_masked, axis=1)                            # [n]
    has_target = jnp.take_along_axis(d_masked, best[:, None], axis=1)[:, 0] < BIG
    target = jnp.take_along_axis(nbr, best[:, None], axis=1)[:, 0]  # [n]
    target = jnp.where(has_target, target, n)                      # n = dump slot

    dmg = jnp.where(has_target, cfg.pred_rate * state.diet, 0.0) * state.alive
    wanted = jnp.zeros(n + 1).at[target].add(dmg)                  # per-prey demand
    prey_e = jnp.concatenate([jnp.maximum(state.energy, 0.0), jnp.zeros(1)])
    removed = jnp.minimum(wanted, prey_e)
    scale = jnp.where(wanted > 0, removed / wanted, 0.0)           # attacker payout frac

    meat_gain = dmg * scale[target] * cfg.pred_efficiency          # [n]
    damage = removed[:n]                                           # [n]
    energy = state.energy - damage + meat_gain

    # Same bite events, but drawing against the prey's *water* pool instead.
    water_dmg = dmg * cfg.meat_water_frac
    water_wanted = jnp.zeros(n + 1).at[target].add(water_dmg)
    prey_w = jnp.concatenate([jnp.maximum(state.water, 0.0), jnp.zeros(1)])
    water_removed = jnp.minimum(water_wanted, prey_w)
    water_scale = jnp.where(water_wanted > 0, water_removed / water_wanted, 0.0)

    water_gain = water_dmg * water_scale[target] * cfg.pred_efficiency
    water_damage = water_removed[:n]
    water = state.water - water_damage + water_gain

    return energy, meat_gain, damage, water, water_gain, water_damage


def metabolize(energy: jax.Array, thrust: jax.Array, diet: jax.Array,
               climb: jax.Array, alive: jax.Array, cfg: Config) -> jax.Array:
    """Charge the per-step energy cost of existing, moving, climbing, and (for
    carnivores) hunting. The diet-scaled term makes pure carnivory unsustainable
    when prey is scarce -- the feedback that keeps herbivores and carnivores
    coexisting. The climb term is work against gravity, proportional to elevation
    gained, which is what makes the range a soft barrier rather than a wall.
    """
    cost = (cfg.base_cost + cfg.move_cost * thrust + cfg.carn_cost * diet) * alive
    return energy - cost - cfg.climb_cost * climb


def thirst(water: jax.Array, thrust: jax.Array, alive: jax.Array,
           cfg: Config) -> jax.Array:
    """Charge the per-step water cost of existing and moving (panting/sweating).
    Symmetric across diet -- both herbivores and carnivores need to visit the
    stream; carnivores get a bonus top-up from every successful kill instead.
    """
    cost = (cfg.base_water_cost + cfg.move_water_cost * thrust) * alive
    return water - cost
