"""The Cardinal loop: one pure `world_step`, jit-compiled and closed over a
`Config`. `build_step` returns a `(state, key) -> (state, Metrics)` function;
`scan_steps` runs many of them with `lax.scan` for headless fast-forward (FLA).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from . import brain, dynamics, ecology, metrics, reproduction, sensors, spatial
from .config import Config
from .state import WorldState, diet_of, init_state


def build_step(cfg: Config):
    """Return a jitted single world-step for this configuration."""

    def world_step(state: WorldState, key: jax.Array):
        # 1. see (retina) -> 2. think (recurrent) -> 3. act
        table = spatial.build_table(state, cfg)
        nbr = spatial.gather_neighbors(state, table, cfg)
        delta, dist, valid = spatial.geometry(state, nbr, cfg)
        inputs = sensors.sense(state, nbr, delta, dist, valid, cfg)
        outputs, hidden = brain.forward(state.genome, inputs, state.hidden, cfg)
        state = state._replace(hidden=hidden, last_input=inputs, last_output=outputs)
        state, thrust = dynamics.act(state, outputs, cfg)

        # 4. graze/drink, then hunt (neighbours re-indexed after moving), then metabolism
        energy, plant, food_gain = dynamics.graze(state, cfg)
        water, drink_gain = dynamics.drink(state, cfg)
        state = state._replace(energy=energy, water=water, plant=plant)
        table2 = spatial.build_table(state, cfg)
        nbr2 = spatial.gather_neighbors(state, table2, cfg)
        _d2, dist2, valid2 = spatial.geometry(state, nbr2, cfg)
        energy, meat_gain, damage, water, meat_water_gain, water_damage = \
            dynamics.predation(state, nbr2, dist2, valid2, cfg)
        energy = dynamics.metabolize(energy, thrust, state.diet, state.alive, cfg)
        water = dynamics.thirst(water, thrust, state.alive, cfg)
        state = state._replace(
            energy=energy, water=water, age=state.age + state.alive,
            last_food=food_gain, last_meat=meat_gain, last_damage=damage,
            last_drink=drink_gain + meat_water_gain,
        )

        # 5. death -> 6. birth
        state = reproduction.cull(state, cfg)
        state = reproduction.reproduce(state, key, cfg)

        # 7. plants regrow; refresh the cached diet for the whole population
        state = state._replace(
            plant=ecology.regrow(state.plant, cfg),
            diet=diet_of(state.genome, cfg),
        )

        return state, metrics.compute(state, cfg)

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
    """Convenience: build a fresh state + step fn + scan fn for a config."""
    key = jax.random.PRNGKey(cfg.seed)
    key, sub = jax.random.split(key)
    state = init_state(cfg, sub)
    step_fn = build_step(cfg)
    scan_fn = make_scan(step_fn)
    return state, key, step_fn, scan_fn
