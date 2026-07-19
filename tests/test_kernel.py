"""M0 sanity tests: shapes, invariants, determinism, and the birth/death
slot bookkeeping. Run with: .venv/bin/python -m pytest
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from underworld import Config, new_world
from underworld import terrain as terrain_mod
from underworld.state import init_state


# Small, fast config for tests.
def tiny_cfg(**kw):
    # A small *world*, not just a small grid: terrain lengths are fractions of
    # world_size, but the plant cell must stay comparable to river_half_width or
    # no cell centre ever registers as water and everything dies of thirst.
    base = dict(n_max=256, n_init=64, world_size=128.0, grid=32,
                sense_grid=6, seed=1)
    base.update(kw)
    return Config(**base)


def run(cfg, n_steps, key_seed=0):
    state, key, step_fn, scan_fn, _terrain = new_world(cfg)
    state, key, ms = scan_fn(state, key, n_steps)
    return state, ms


def test_shapes():
    cfg = tiny_cfg()
    key = jax.random.PRNGKey(0)
    s = init_state(cfg, key, terrain_mod.build(cfg))
    assert s.alive.shape == (cfg.n_max,)
    assert s.pos.shape == (cfg.n_max, 2)
    assert s.genome.shape == (cfg.n_max, cfg.genome_size)
    assert s.water.shape == (cfg.n_max,)
    assert s.plant.shape == (cfg.n_cells,)
    assert int(jnp.sum(s.alive)) == cfg.n_init


def test_no_nans_and_invariants():
    cfg = tiny_cfg()
    state, ms = run(cfg, 300)
    for name, arr in state._asdict().items():
        assert bool(jnp.all(jnp.isfinite(arr))), f"non-finite in {name}"
    # Living agents must have positive energy/water (dead are culled each step).
    living_energy = state.energy[state.alive]
    assert bool(jnp.all(living_energy > 0.0))
    living_water = state.water[state.alive]
    assert bool(jnp.all(living_water > 0.0))
    # Plant field stays within [0, carrying capacity].
    assert float(jnp.min(state.plant)) >= 0.0
    assert float(jnp.max(state.plant)) <= cfg.plant_max + 1e-4
    # Positions stay on the torus.
    assert float(jnp.min(state.pos)) >= 0.0
    assert float(jnp.max(state.pos)) < cfg.world_size


def test_population_bounded():
    cfg = tiny_cfg()
    state, ms = run(cfg, 300)
    pop = np.asarray(ms.population)
    assert pop.max() <= cfg.n_max
    assert pop.min() >= 0


def test_determinism():
    # GPU atomic scatter-adds (per-cell feeding sums) are not bit-reproducible,
    # so exact equality can drift over long horizons. Over a short horizon the
    # life/death structure is identical and values match to tolerance. (For full
    # bit-determinism, run with XLA_FLAGS=--xla_gpu_deterministic_ops=true.)
    cfg = tiny_cfg()
    s1, m1 = run(cfg, 15)
    s2, m2 = run(cfg, 15)
    assert bool(jnp.array_equal(s1.alive, s2.alive))
    assert bool(jnp.array_equal(m1.population, m2.population))
    assert bool(jnp.allclose(s1.genome, s2.genome, atol=1e-4))
    assert bool(jnp.allclose(s1.energy, s2.energy, atol=1e-2))


def test_different_seed_diverges():
    s1, _ = run(tiny_cfg(seed=1), 200)
    s2, _ = run(tiny_cfg(seed=2), 200)
    assert not bool(jnp.allclose(s1.pos, s2.pos))


def test_neighbor_index():
    """Hand-checked: clustered agents see each other, distant ones don't."""
    from underworld import spatial
    cfg = tiny_cfg(n_max=64, n_init=3, sense_grid=4, k_neighbors=8)
    s = init_state(cfg, jax.random.PRNGKey(0), terrain_mod.build(cfg))
    pos = np.array(s.pos)
    pos[0] = [10.0, 10.0]
    pos[1] = [12.0, 11.0]          # ~2 units from agent 0
    pos[2] = [200.0, 200.0]        # far away
    s = s._replace(pos=jnp.array(pos), alive=jnp.arange(cfg.n_max) < 3)

    table = spatial.build_table(s, cfg)
    nbr = spatial.gather_neighbors(s, table, cfg)
    _d, dist, valid = spatial.geometry(s, nbr, cfg)

    row, v, dd = np.array(nbr[0]), np.array(valid[0]), np.array(dist[0])
    close = {int(row[k]) for k in range(row.size) if v[k] and dd[k] < cfg.vision_radius}
    assert 1 in close          # nearby agent is found
    assert 2 not in close       # distant agent is outside vision radius
    assert 0 not in close       # self excluded


def test_diet_in_range():
    cfg = tiny_cfg()
    state, ms = run(cfg, 200)
    d = state.diet[state.alive]
    assert float(jnp.min(d)) >= 0.0 and float(jnp.max(d)) <= 1.0


def test_predation_energy_not_created():
    """Predation must not create net energy or water: gains <= losses (trophic
    loss), for both resources."""
    cfg = tiny_cfg()
    from underworld import dynamics, spatial
    state, _ = run(cfg, 50)
    table = spatial.build_table(state, cfg)
    nbr = spatial.gather_neighbors(state, table, cfg)
    _d, dist, valid = spatial.geometry(state, nbr, cfg)
    _e, meat_gain, damage, _w, water_gain, water_damage = \
        dynamics.predation(state, nbr, dist, valid, cfg)
    assert float(jnp.sum(meat_gain)) <= float(jnp.sum(damage)) + 1e-3
    assert float(jnp.sum(water_gain)) <= float(jnp.sum(water_damage)) + 1e-3


def test_alive_energy_consistency():
    """Dead slots should not accumulate positive energy or run the brain."""
    cfg = tiny_cfg()
    state, ms = run(cfg, 200)
    dead = ~state.alive
    # Culled/empty slots that were never (re)born stay at zero-ish; at minimum
    # they must not hold reproducible energy above the threshold.
    assert bool(jnp.all(state.energy[dead] <= cfg.repro_threshold))
