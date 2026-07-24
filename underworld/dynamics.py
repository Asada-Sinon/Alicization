"""Physics & feeding: turn brain outputs into motion, then let agents graze,
hunt, and pay their metabolic bill. Feeding interactions are resolved per plant
cell (cell size ~ a bite radius), which stays cheap and fully vectorized.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import Config
from .state import (WorldState, armor_of, attack_range_of, escape_of, pos_to_cell,
                    spike_of)


def _herbivory(diet: jax.Array, cfg: Config) -> jax.Array:
    """Grazing skill as a function of diet, shared by `graze` and `eat_fruit` so
    the plant and fruit layers taper identically. The rationale for the taper's
    shape -- the steep `(1-diet)^6` and the hard `carn_graze_cutoff` -- lives on
    the call site in `graze`.
    """
    return jnp.where(diet > cfg.carn_graze_cutoff, 0.0, (1.0 - diet) ** 6)


def _forage_heat_scale(state: WorldState, cfg: Config) -> jax.Array:
    """Midday-heat foraging penalty for `graze`/`eat_fruit` (docs/day_night.md §6,
    the water-neutral Phase-2 substrate). Returns a multiplier on intake demand:
    1 at night, (1 - forage_heat) at midday. `light` is 0 at midnight, 1 at midday.
    Only ever called inside a `cfg.day_length > 0` branch, so when the clock is off
    this code is absent from the trace and grazing is bit-exact the pre-clock kernel.
    """
    light = 0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * state.phase))
    return jnp.clip(1.0 - cfg.forage_heat * light, 0.0, None)


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
    # Envenomation (docs/trait_defense_landing.md §7) saps speed: a carnivore that bit
    # a spiked prey last step drags. `venom` is 0 for everyone when spike_heritable is
    # off (or nobody has been bitten by a spiked prey), so this is a no-op there.
    venom_slow = 1.0 - cfg.venom_slow * jnp.clip(state.venom, 0.0, 1.0)
    speed = thrust * cfg.max_speed * slow * venom_slow
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
    if cfg.day_length > 0:
        demand = demand * _forage_heat_scale(state, cfg)

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
    if cfg.day_length > 0:
        demand = demand * _forage_heat_scale(state, cfg)

    demand_per_cell = jnp.zeros(cfg.n_cells).at[cell].add(demand)
    removed_per_cell = jnp.minimum(demand_per_cell, state.fruit)
    frac = jnp.where(demand_per_cell > 0, removed_per_cell / demand_per_cell, 0.0)
    taken = demand * frac[cell]
    gain = taken * cfg.fruit_energy * cfg.eat_efficiency

    energy = state.energy + gain
    fruit = state.fruit - removed_per_cell
    return energy, fruit, gain, taken * cfg.forage_water_frac


def scavenge(state: WorldState, cfg: Config):
    """Carnivores eat carrion at their cell -- a second meat source besides live prey
    (docs/multispecies_feasibility.md §4). Structurally identical to `graze`: a per-cell
    demand pool, fair share of what the cell holds -- but the skill scales with `diet`
    (only meat-eaters scavenge; herbivores draw ~0) and it drains the `carrion` field
    instead of the plant field. A corpse hydrates like a kill (`meat_water_frac`).
    Only called when `carrion_enabled`. Returns (energy, carrion, scav_gain, water_gain).
    """
    cell = pos_to_cell(state.pos, cfg)
    demand = cfg.carrion_eat_rate * state.diet * state.alive          # carnivores scavenge
    demand_per_cell = jnp.zeros(cfg.n_cells).at[cell].add(demand)
    removed_per_cell = jnp.minimum(demand_per_cell, state.carrion)
    frac = jnp.where(demand_per_cell > 0, removed_per_cell / demand_per_cell, 0.0)
    taken = demand * frac[cell]
    gain = taken * cfg.carrion_energy
    energy = state.energy + gain
    carrion = state.carrion - removed_per_cell
    return energy, carrion, gain, gain * cfg.meat_water_frac


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
              valid: jax.Array, cfg: Config, light: jax.Array | None = None):
    """Neighbour-based carnivory. Each agent attacks the *nearest eligible prey*
    among its neighbours -- an agent that is `diet_delta` more herbivorous than
    itself, within `attack_range`. Damage is scatter-added onto prey, capped by
    the prey's energy, then redistributed to the attackers (trophic loss via
    `pred_efficiency`). A bite also draws a proportional share of the prey's
    *water* (capped by what it has), redistributed the same way -- a kill
    hydrates as well as feeds. Because a carnivore genuinely needs a weaker
    neighbour to eat, an all-carnivore population starves -- enforcing a
    herbivore-majority pyramid. Returns (energy, meat_gain, damage_taken, water,
    water_gain, water_damage_taken, venom_deposit) -- the last is per-attacker
    envenomation from spiked prey, added to the biter's `venom` field in step.py.
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
    # Nocturnal predation boost (docs/day_night.md §4): predators reach farther in
    # the dark. `light` is 0 at midnight, 1 at midday, so the boost peaks at night.
    # Clipped to the sense cell so a boosted bite can never exceed what the 3x3
    # neighbour block gathered (the same ceiling `attack_max` is guarded against).
    # `light is not None` only when day_length>0; otherwise this is absent and
    # predation is bit-exact the pre-clock kernel.
    if light is not None:
        reach = reach * (1.0 + cfg.pred_night_amp * (1.0 - light))
        reach = jnp.minimum(reach, cfg.world_size / cfg.sense_grid)
    eligible = valid & (d_i - d_j > cfg.diet_delta) & (dist < reach) & (e_j > 0.0)

    BIG = 1e9
    d_masked = jnp.where(eligible, dist, BIG)
    best = jnp.argmin(d_masked, axis=1)                            # [n]
    has_target = jnp.take_along_axis(d_masked, best[:, None], axis=1)[:, 0] < BIG
    target = jnp.take_along_axis(nbr, best[:, None], axis=1)[:, 0]  # [n]
    target = jnp.where(has_target, target, n)                      # n = dump slot

    base_dmg = jnp.where(has_target, cfg.pred_rate * state.diet, 0.0) * state.alive
    # Carnivore OFFENSE (docs/trait_defense_landing.md §7): the attacker's own spikes
    # add bite damage, which is what lets it punch through evolved prey armour. gene=0
    # -> x1, so a fresh population bites exactly as the pre-spike kernel did.
    if cfg.spike_heritable:
        dmg = base_dmg * (1.0 + cfg.spike_offense_gain * spike_of(state.genome, cfg))
    else:
        dmg = base_dmg
    wanted = jnp.zeros(n + 1).at[target].add(dmg)                  # per-prey demand
    # Prey armour (docs/trait_defense_catalog.md) negates a fraction of each bite.
    # Scaling the per-prey *demand* -- not just the damage taken -- keeps predation
    # conservative: the attacker's payout `scale` falls in step, so meat_gain stays
    # <= damage (test_predation_energy_not_created). The dump slot (n) keeps full
    # retention; nothing targets it with dmg>0 anyway.
    if cfg.armor_heritable:
        retain = jnp.concatenate([1.0 - armor_of(state.genome, cfg), jnp.ones(1)])
        wanted = wanted * retain
    prey_e = jnp.concatenate([jnp.maximum(state.energy, 0.0), jnp.zeros(1)])
    removed = jnp.minimum(wanted, prey_e)
    scale = jnp.where(wanted > 0, removed / wanted, 0.0)           # attacker payout frac

    meat_gain = dmg * scale[target] * cfg.pred_efficiency          # [n]
    damage = removed[:n]                                           # [n]
    energy = state.energy - damage + meat_gain

    # Herbivore DEFENSE (docs/trait_defense_landing.md §7): a bitten prey's spikes
    # ENVENOM the attacker -- a debuff deposited on the biter that decays over the
    # following steps (read by `act`/`metabolize`, decayed in `step.py`), NOT an
    # instant energy reflect. Non-lethal here; its bite still lands. Zero for
    # non-attackers (dmg=0) and the dump slot (spike 0), so no extra mask is needed.
    if cfg.spike_heritable:
        prey_spike = jnp.concatenate([spike_of(state.genome, cfg), jnp.zeros(1)])
        venom_deposit = jnp.where(dmg > 0.0, prey_spike[target] * cfg.spike_venom_gain, 0.0)
    else:
        venom_deposit = jnp.zeros(n)

    # Same bite events, but drawing against the prey's *water* pool instead.
    water_dmg = dmg * cfg.meat_water_frac
    water_wanted = jnp.zeros(n + 1).at[target].add(water_dmg)
    prey_w = jnp.concatenate([jnp.maximum(state.water, 0.0), jnp.zeros(1)])
    water_removed = jnp.minimum(water_wanted, prey_w)
    water_scale = jnp.where(water_wanted > 0, water_removed / water_wanted, 0.0)

    water_gain = water_dmg * water_scale[target] * cfg.pred_efficiency
    water_damage = water_removed[:n]
    water = state.water - water_damage + water_gain

    return energy, meat_gain, damage, water, water_gain, water_damage, venom_deposit


def metabolize(energy: jax.Array, thrust: jax.Array, diet: jax.Array,
               climb: jax.Array, alive: jax.Array, cfg: Config,
               size: jax.Array, attack_range: jax.Array | None = None,
               escape: jax.Array | None = None, armor: jax.Array | None = None,
               spike: jax.Array | None = None, venom: jax.Array | None = None) -> jax.Array:
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
    # Morphological defences (docs/trait_defense_catalog.md): armour and spikes pay an
    # energy upkeep scaled by (1-diet), so drifting carnivores (who are rarely prey)
    # pay ~0. Same energy-ledger discipline as the red-queen taxes above -- never
    # thirst (docs/trait_addition_feasibility.md §B.2).
    if cfg.armor_heritable and armor is not None:
        tax = tax + cfg.armor_cost * armor * (1.0 - diet)
    # Spike tax is UNIVERSAL now (docs/trait_defense_landing.md §7): spikes are
    # dual-use (carnivore offense / herbivore defense), so both lineages grow and pay
    # for them -- no (1-diet) gate.
    if cfg.spike_heritable and spike is not None:
        tax = tax + cfg.spike_cost * spike
    energy = energy - cost - cfg.climb_cost * climb - tax * alive
    # Envenomation energy drain: the ongoing poison cost on a bitten carnivore. Zero
    # when venom is 0 (spike off, or nobody envenomed), so a no-op there.
    if venom is not None:
        energy = energy - cfg.venom_drain * jnp.clip(venom, 0.0, 1.0) * alive
    return energy


def thirst(water: jax.Array, thrust: jax.Array, alive: jax.Array,
           cfg: Config, size: jax.Array, light: jax.Array | None = None) -> jax.Array:
    """Charge the per-step water cost of existing and moving (panting/sweating).
    Symmetric across diet -- both herbivores and carnivores need to visit the
    stream; carnivores get a bonus top-up from every successful kill instead.

    Scales with `size` at 0.75, the same Kleiber exponent as `metabolize` --
    water loss, like energy loss, is metabolic/surface-area driven rather than a
    simple volume effect (that's `water_max` in `drink`, which scales at 1.0).

    The day-night heat term (docs/day_night.md) taxes ACTIVITY, not rest: only the
    movement (panting) component is scaled by midday heat -- `move_water_cost *
    thrust * (1 + heat_water_amp*light)`, with `light` 0 at midnight and 1 at
    midday. The resting `base_water_cost` is left alone, so an agent can sit or
    drink near water at midday for its ordinary cost and pays the heat penalty only
    when it forages or travels. That makes "forage in the cool night, rest by the
    water at midday" a survivable commute rather than a death tax: the seed-0 probe
    showed a flat multiplier on base+move drove thirst mortality +18-21pp over the
    60.1% baseline and culled the far-from-water population at midday (an artifact
    that faked a day/night distance gap), docs/day_night.md §4. `light` is optional
    (default None) so the pre-clock call signature and the thirst unit test still
    work unchanged and bit-exact; it is only passed by `step.py` when
    `day_length > 0`, matching the compile-time gating of `metabolize`'s taxes.
    """
    cost = (cfg.base_water_cost + cfg.move_water_cost * thrust)
    if light is not None:
        # Extra panting cost from midday heat, ADDED onto the movement term only
        # (base rest cost untouched). Written as an addition rather than folding a
        # (1+heat*light) factor into the line above so the light-None path is the
        # byte-identical float expression the pre-clock kernel ran -- this world is
        # chaotic enough that even a 1-ULP rounding change from restructuring the
        # expression cascades into a visibly different smoke population, so the
        # day_length=0 no-op must reuse the exact original ops, not merely an
        # algebraically-equal rewrite.
        cost = cost + cfg.move_water_cost * thrust * cfg.heat_water_amp * light
    cost = cost * (size ** 0.75) * alive
    return water - cost
