"""Physics & feeding: turn brain outputs into motion, then let agents graze,
hunt, and pay their metabolic bill. Feeding interactions are resolved per plant
cell (cell size ~ a bite radius), which stays cheap and fully vectorized.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import Config
from .state import WorldState, attack_range_of, escape_of, pos_to_cell


def _herbivory(diet: jax.Array, cfg: Config) -> jax.Array:
    """Grazing skill as a function of diet, shared by `graze` and `eat_fruit` so
    the plant and fruit layers taper identically. The rationale for the taper's
    shape -- the steep `(1-diet)^6` and the hard `carn_graze_cutoff` -- lives on
    the call site in `graze`.
    """
    return jnp.where(diet > cfg.carn_graze_cutoff, 0.0, (1.0 - diet) ** 6)


def act(state: WorldState, outputs: jax.Array, terrain, cfg: Config):
    """Apply brain outputs. Returns (new_state, thrust, climb) where thrust is in
    [0, 1] and `climb` is the elevation *gained* this step (>= 0), which
    `metabolize` charges for.

    Dense canopy slows movement: forest is food-rich but hard to cross, so it's a
    genuine trade-off rather than a free bonus.

    Recent foot traffic (`state.trample`, from the *end* of the previous step --
    see `step.py`) cancels part of that canopy penalty: a cell repeatedly walked
    has its undergrowth worn down, so the same route gets cheaper to cross again.
    This is the sign-corrected Stage 0 niche-construction mechanism (see
    `trample_path_gain` in config.py and docs/biology.md SS11.1) -- the earlier
    version eroded food capacity instead, which measurably turned out to be a
    *dispersing* negative feedback, not a path-forming one. `trample_path_gain`
    defaults to 0.0, making `path_relief` identically zero and this identical to
    the pre-existing behaviour.
    """
    turn = outputs[:, 0] * cfg.max_turn                 # [-max_turn, max_turn]
    thrust = 0.5 * (outputs[:, 1] + 1.0)                # [0, 1]

    cell = pos_to_cell(state.pos, cfg)
    path_relief = jnp.clip(cfg.trample_path_gain * state.trample[cell], 0.0, 1.0)
    slow = 1.0 - cfg.forest_slow * terrain.forest[cell] * (1.0 - path_relief)

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
    herbivory = _herbivory(state.diet, cfg)
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
    herbivory = _herbivory(state.diet, cfg)
    demand = cfg.fruit_eat_rate * herbivory * state.alive

    demand_per_cell = jnp.zeros(cfg.n_cells).at[cell].add(demand)
    removed_per_cell = jnp.minimum(demand_per_cell, state.fruit)
    frac = jnp.where(demand_per_cell > 0, removed_per_cell / demand_per_cell, 0.0)
    taken = demand * frac[cell]
    gain = taken * cfg.fruit_energy * cfg.eat_efficiency

    energy = state.energy + gain
    fruit = state.fruit - removed_per_cell
    return energy, fruit, gain, taken * cfg.forage_water_frac


def drink(state: WorldState, terrain, cfg: Config, size: jax.Array):
    """Hydration: standing in a river or at the sea refills water, uncapped by any
    shared demand pool (flowing water isn't meaningfully depleted at this scale,
    only spatially constrained). Returns (water, drink_gain).

    This is now a lookup into the precomputed `water_dist` field rather than the
    old analytic sine -- cheaper per step, and it works for the traced rivers and
    the sea alike.

    `water_max` scales with `size` at exponent 1.0 -- water storage is a volume,
    so it scales isometrically with body size, unlike the sub-linear metabolic
    scaling in `metabolize`/`thirst` below.
    """
    cell = pos_to_cell(state.pos, cfg)
    in_water = terrain.water_dist[cell] < cfg.river_half_width
    gain = jnp.where(in_water, cfg.drink_rate, 0.0) * state.alive
    water = jnp.minimum(state.water + gain, cfg.water_max * size)
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

    # Effective bite reach, the red-queen contest (docs/attack_range_redqueen.md):
    # the attacker's own heritable reach, minus the *prey's* heritable evasion. Both
    # flags are Python bools baked into the jit (Config is closed over, not traced),
    # so an off arm compiles the constant/no-op branch away entirely -- the same
    # convention as `peer_channel_enabled`, keeping every arm genome-compatible.
    if cfg.attack_range_heritable:
        reach = attack_range_of(state.genome, cfg)[:, None]        # [n, 1]
    else:
        reach = cfg.attack_range
    if cfg.prey_escape_enabled:
        reach = reach - escape_of(state.genome, cfg)[safe]         # [n, M]
    eligible = valid & (d_i - d_j > cfg.diet_delta) & (dist < reach) & (e_j > 0.0)

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
               climb: jax.Array, alive: jax.Array, cfg: Config,
               size: jax.Array, attack_range: jax.Array | None = None,
               escape: jax.Array | None = None) -> jax.Array:
    """Charge the per-step energy cost of existing, moving, climbing, and (for
    carnivores) hunting. The diet-scaled term makes pure carnivory unsustainable
    when prey is scarce -- the feedback that keeps herbivores and carnivores
    coexisting. The climb term is work against gravity, proportional to elevation
    gained, which is what makes the range a soft barrier rather than a wall.

    The base/move/hunt cost scales with `size` at Kleiber's exponent (0.75):
    metabolic rate is sub-linear in body mass. `size` deliberately does not
    touch `eat_rate`/`fruit_eat_rate` (intake) or `attack_range`/`diet_delta`
    (predation) -- if intake also scaled up, size would run away to its upper
    bound with no countervailing cost. Climb cost is left unscaled: it is not
    part of this design's claim and there is no measurement backing an exponent
    for it yet.

    The red-queen taxes (`attack_range`/`escape`, docs/attack_range_redqueen.md)
    ride the ENERGY ledger deliberately -- never `thirst`'s water ledger, which
    would censor the trait through the juvenile-thirst bottleneck exactly as it
    reversed the body-size gene (docs/trait_roadmap.md §5). Attack is charged only
    above the 6.0 baseline and scaled by `diet` (so drifting herbivores pay ~0);
    escape is scaled by `1-diet` (so drifting carnivores pay ~0). Both are optional
    so the pre-gene call signature (and the metabolize unit test) still works; each
    is gated by its ablation flag so a control arm's neutral-drift gene levies
    nothing.
    """
    cost = (cfg.base_cost + cfg.move_cost * thrust + cfg.carn_cost * diet)
    cost = cost * (size ** 0.75) * alive
    tax = jnp.zeros_like(cost)
    if cfg.attack_range_heritable and attack_range is not None:
        tax = tax + cfg.attack_cost * jnp.maximum(attack_range - cfg.attack_range, 0.0) * diet
    if cfg.prey_escape_enabled and escape is not None:
        tax = tax + cfg.escape_cost * escape * (1.0 - diet)
    return energy - cost - cfg.climb_cost * climb - tax * alive


def thirst(water: jax.Array, thrust: jax.Array, alive: jax.Array,
           cfg: Config, size: jax.Array, light: jax.Array | None = None) -> jax.Array:
    """Charge the per-step water cost of existing and moving (panting/sweating).
    Symmetric across diet -- both herbivores and carnivores need to visit the
    stream; carnivores get a bonus top-up from every successful kill instead.

    Scales with `size` at 0.75, the same Kleiber exponent as `metabolize` --
    water loss, like energy loss, is metabolic/surface-area driven rather than a
    simple volume effect (that's `water_max` in `drink`, which scales at 1.0).

    The day-night heat term (docs/day_night.md): midday heat raises evaporative
    loss, so water cost is scaled by (1 + heat_water_amp*light) where `light` is
    0 at midnight and 1 at midday. `light` is optional (default None) so the
    pre-clock call signature and the thirst unit test still work unchanged; it is
    only passed by `step.py` when `day_length > 0`, matching the compile-time
    gating of `metabolize`'s red-queen taxes.
    """
    cost = (cfg.base_water_cost + cfg.move_water_cost * thrust)
    if light is not None:
        cost = cost * (1.0 + cfg.heat_water_amp * light)
    cost = cost * (size ** 0.75) * alive
    return water - cost
