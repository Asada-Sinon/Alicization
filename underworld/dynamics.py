"""Physics & feeding: turn brain outputs into motion, then let agents graze,
hunt, and pay their metabolic bill. Feeding interactions are resolved per plant
cell (cell size ~ a bite radius), which stays cheap and fully vectorized.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import Config
from .ecology import in_stream
from .state import WorldState, pos_to_cell


def act(state: WorldState, outputs: jax.Array, cfg: Config):
    """Apply brain outputs. Returns (new_state, thrust) where thrust in [0, 1]."""
    turn = outputs[:, 0] * cfg.max_turn                 # [-max_turn, max_turn]
    thrust = 0.5 * (outputs[:, 1] + 1.0)                # [0, 1]

    heading = jnp.mod(state.heading + turn * cfg.dt, 2.0 * jnp.pi)
    speed = thrust * cfg.max_speed
    vel = jnp.stack([jnp.cos(heading), jnp.sin(heading)], axis=1) * speed[:, None]
    vel = vel * state.alive[:, None]                    # the dead don't drift
    pos = jnp.mod(state.pos + vel * cfg.dt, cfg.world_size)

    return state._replace(heading=heading, vel=vel, pos=pos), thrust


def graze(state: WorldState, cfg: Config):
    """Herbivory: agents drain the plant field under them, fairly sharing a
    cell's yield. Carnivores (high diet) barely graze. Returns (energy, plant,
    food_gain).
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
    return energy, plant, gain


def drink(state: WorldState, cfg: Config):
    """Hydration: standing in the stream refills water, uncapped by any shared
    demand pool (a flowing stream isn't meaningfully depleted at this scale, only
    spatially constrained). Returns (water, drink_gain).
    """
    gain = jnp.where(in_stream(state.pos, cfg), cfg.drink_rate, 0.0) * state.alive
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
               alive: jax.Array, cfg: Config) -> jax.Array:
    """Charge the per-step energy cost of existing, moving, and (for carnivores)
    hunting. The diet-scaled term makes pure carnivory unsustainable when prey is
    scarce -- the feedback that keeps herbivores and carnivores coexisting.
    """
    cost = (cfg.base_cost + cfg.move_cost * thrust + cfg.carn_cost * diet) * alive
    return energy - cost


def thirst(water: jax.Array, thrust: jax.Array, alive: jax.Array,
           cfg: Config) -> jax.Array:
    """Charge the per-step water cost of existing and moving (panting/sweating).
    Symmetric across diet -- both herbivores and carnivores need to visit the
    stream; carnivores get a bonus top-up from every successful kill instead.
    """
    cost = (cfg.base_water_cost + cfg.move_water_cost * thrust) * alive
    return water - cost
