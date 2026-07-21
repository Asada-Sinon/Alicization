"""The Cardinal loop: one pure `world_step`, jit-compiled and closed over a
`Config`. `build_step` returns a `(state, key) -> (state, Metrics)` function;
`scan_steps` runs many of them with `lax.scan` for headless fast-forward (FLA).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from . import (brain, dynamics, ecology, memory, metrics, reproduction, sensors,
               spatial)
from . import terrain as terrain_mod
from .config import Config
from .state import WorldState, diet_of, init_state, pos_to_cell


def build_step(cfg: Config, terrain):
    """Return a jitted single world-step for this configuration.

    `terrain` is static for the whole run, so it is closed over here rather than
    carried in `WorldState` -- that keeps it out of the `lax.scan` carry instead
    of copying several [n_cells] fields every single step.
    """

    def world_step(state: WorldState, key: jax.Array):
        # 1. see (retina) -> 2. think (recurrent) -> 3. act
        table = spatial.build_table(state, cfg)
        nbr = spatial.gather_neighbors(state, table, cfg)
        delta, dist, valid = spatial.geometry(state, nbr, cfg)
        inputs = sensors.sense(state, nbr, delta, dist, valid, terrain, cfg)
        outputs, hidden = brain.forward(state.genome, inputs, state.hidden, cfg)
        state = state._replace(hidden=hidden, last_input=inputs, last_output=outputs)
        state, thrust, climb = dynamics.act(state, outputs, terrain, cfg)

        # 3b. carry the memory vectors across the movement just made, so a slot
        # written below records an offset of ~0 from where the agent now stands.
        key, k_drift = jax.random.split(key)
        state = state._replace(
            memory=memory.advance(state.memory, state.vel * cfg.dt, k_drift, cfg))

        # 4. graze/drink, then hunt (neighbours re-indexed after moving), then metabolism
        energy, plant, food_gain, forage_water = dynamics.graze(state, cfg)
        state = state._replace(energy=energy, plant=plant)
        energy, fruit, fruit_gain, fruit_water = dynamics.eat_fruit(state, cfg)
        water, drink_gain = dynamics.drink(state, terrain, cfg)
        # Forage water lands inside the same cap as drinking, so eating can
        # top a tank up but never overfill one past what a river would.
        water = jnp.minimum(water + forage_water + fruit_water, cfg.water_max)
        state = state._replace(energy=energy, water=water, fruit=fruit)

        # 4b. remember where that came from. Writes are triggered by the eating
        # itself, not by a brain decision -- the agent does not choose to
        # memorise, it simply has been somewhere worth returning to.
        mem = memory.write(state.memory, 0, cfg.memory_water_slots,
                           drink_gain > 0.0, cfg)
        mem = memory.write(mem, cfg.memory_water_slots, cfg.memory_slots,
                           fruit_gain > 0.0, cfg)
        state = state._replace(memory=mem)
        table2 = spatial.build_table(state, cfg)
        nbr2 = spatial.gather_neighbors(state, table2, cfg)
        _d2, dist2, valid2 = spatial.geometry(state, nbr2, cfg)
        energy, meat_gain, damage, water, meat_water_gain, water_damage = \
            dynamics.predation(state, nbr2, dist2, valid2, cfg)
        energy = dynamics.metabolize(
            energy, thrust, state.diet, climb, state.alive, cfg)
        water = dynamics.thirst(water, thrust, state.alive, cfg)
        state = state._replace(
            energy=energy, water=water, age=state.age + state.alive,
            last_food=food_gain + fruit_gain, last_meat=meat_gain, last_damage=damage,
            last_drink=drink_gain + meat_water_gain + forage_water + fruit_water,
        )

        # 5. death -> 6. birth
        state, deaths = reproduction.cull(state, water_damage, cfg)
        state = reproduction.reproduce(state, key, cfg)

        # 7a. passive trampling: a niche-construction feedback with zero genome
        # cost (docs/TODO.md priority 3, Stage 0). Reuses the same per-cell
        # scatter-add idiom as `dynamics.graze`'s `demand_per_cell` -- agents
        # deposit onto a [n_cells] field just by standing there, no brain
        # output involved. Computed on the post-movement, post-birth/death
        # population, same cell grid `ecology.regrow` operates on below.
        cell = pos_to_cell(state.pos, cfg)
        occupancy = jnp.zeros(cfg.n_cells).at[cell].add(state.alive.astype(jnp.float32))
        trample = jnp.clip(
            state.trample * cfg.trample_decay + occupancy * cfg.trample_rate,
            0.0, 1.0)
        # Erodes the *grass* carrying capacity only -- feet wear down the herb
        # layer they walk on, not canopy fruit hanging overhead. Default
        # trample_impact=0.0 makes this identically `terrain.capacity`, so the
        # trample field's own dynamics have no effect on the rest of the sim
        # unless an ablation arm turns it on.
        effective_capacity = jnp.clip(
            terrain.capacity * (1.0 - trample * cfg.trample_impact), 0.0, None)

        # 7b. plants regrow; refresh the cached diet for the whole population
        state = state._replace(
            trample=trample,
            plant=ecology.regrow(state.plant, effective_capacity, cfg.regrow_rate,
                                 cfg.regrow_baseline, cfg.plant_max),
            fruit=ecology.regrow(state.fruit, terrain.fruit_capacity,
                                 cfg.fruit_regrow_rate, cfg.fruit_regrow_baseline,
                                 cfg.fruit_max),
            diet=diet_of(state.genome, cfg),
        )

        return state, metrics.compute(state, terrain, deaths, cfg)

    return jax.jit(world_step)


def make_scan(step_fn):
    """Wrap a step into a scan body carrying (state, key). Returns
    `(state, key, n_steps) -> (state, key, stacked_metrics)`."""

    def body(carry, _):
        state, key = carry
        key, sub = jax.random.split(key)
        state, m = step_fn(state, sub)
        return (state, key), m

    @functools.partial(jax.jit, static_argnums=(2,))
    def scan_steps(state, key, n_steps):
        (state, key), ms = jax.lax.scan(body, (state, key), None, length=n_steps)
        return state, key, ms

    return scan_steps


def new_world(cfg: Config):
    """Convenience: build terrain + a fresh state + step fn + scan fn for a config.

    Returns `(state, key, step_fn, scan_fn, terrain)`. Terrain is deterministic
    from `cfg` alone (no RNG), so it is identical across resets of the same run.
    """
    key = jax.random.PRNGKey(cfg.seed)
    key, sub = jax.random.split(key)
    terrain = terrain_mod.build(cfg)
    state = init_state(cfg, sub, terrain)
    step_fn = build_step(cfg, terrain)
    scan_fn = make_scan(step_fn)
    return state, key, step_fn, scan_fn, terrain
