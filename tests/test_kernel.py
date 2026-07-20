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
from underworld import dynamics, memory, reproduction
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
    assert s.fruit.shape == (cfg.n_cells,)
    assert s.memory.shape == (cfg.n_max, cfg.memory_slots, 3)
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


def test_memory_is_not_heritable():
    """A newborn starts with an empty map, and the parent keeps its own.

    Genes cross generations; memory does not. This guards both halves: a child
    that woke up knowing where water is would be Lamarckian, and a parent whose
    slots got clobbered by giving birth would lose a lifetime of learning to the
    permutation-scatter writing over the wrong rows.
    """
    cfg = tiny_cfg(n_max=64, n_init=8)
    key = jax.random.PRNGKey(0)
    terrain = terrain_mod.build(cfg)
    state = init_state(cfg, key, terrain)

    # One rich parent with a known slot; nobody else can breed.
    known = jnp.array([12.0, -5.0, 1.0])
    memory = jnp.zeros_like(state.memory).at[0, 0].set(known)
    energy = jnp.where(jnp.arange(cfg.n_max) == 0, cfg.repro_threshold + 5.0, 1.0)
    state = state._replace(
        alive=jnp.arange(cfg.n_max) < 4, memory=memory, energy=energy)

    child = reproduction.reproduce(state, jax.random.PRNGKey(1), cfg)
    born = np.asarray(child.alive & ~state.alive)
    assert born.sum() == 1, "expected exactly one birth"
    i = int(np.argmax(born))

    assert np.all(np.asarray(child.memory[i]) == 0.0), (
        f"newborn woke up with memory {np.asarray(child.memory[i, 0])} -- "
        f"memory must be acquired, not inherited")
    assert np.allclose(np.asarray(child.memory[0, 0]), np.asarray(known)), (
        "the parent lost its own slots to the birth")


def test_memory_write_replaces_exactly_one_slot():
    """The weakest-slot write must touch one column and leave the rest alone."""
    cfg = tiny_cfg()
    n, k = 8, cfg.memory_slots
    mem = jnp.tile(jnp.array([3.0, 4.0, 0.9]), (n, k, 1))
    mem = mem.at[:, 1, 2].set(0.1)                     # slot 1 is the weakest
    should = jnp.arange(n) < 3                         # only the first three write

    out = memory.write(mem, 0, cfg.memory_water_slots, should, cfg)
    changed = np.asarray(jnp.any(out != mem, axis=2))  # [n, k]

    assert np.all(changed[:3].sum(axis=1) == 1), "writers must change one slot"
    assert not changed[3:].any(), "non-writers must be untouched"
    assert np.all(changed[:3, 1]), "the weakest slot should be the one replaced"
    assert np.allclose(np.asarray(out[:3, 1]), [0.0, 0.0, 1.0])


def test_memory_tracks_position():
    """Walking away from a remembered point must grow the vector back to it.

    With drift disabled this is exact dead reckoning, so it isolates the
    bookkeeping from the noise model.
    """
    cfg = tiny_cfg(memory_drift=0.0)
    n, k = 4, cfg.memory_slots
    mem = jnp.zeros((n, k, 3)).at[:, 0, 2].set(1.0)    # standing on the memory
    disp = jnp.tile(jnp.array([1.0, 0.0]), (n, 1))     # walk +x, one unit a step

    key = jax.random.PRNGKey(0)
    for _ in range(10):
        mem = memory.advance(mem, disp, key, cfg)

    # Ten steps east means the remembered point is now ten units west.
    assert np.allclose(np.asarray(mem[:, 0, :2]), [[-10.0, 0.0]] * n, atol=1e-4)
    assert float(mem[0, 0, 2]) == pytest.approx(cfg.memory_decay ** 10, rel=1e-5)


def test_memory_vector_bounded():
    """Slots stay shortest-path vectors on the torus, however long the run."""
    cfg = tiny_cfg()
    state, _ms = run(cfg, 400)
    mem = np.asarray(state.memory)
    assert np.all(np.isfinite(mem))
    assert np.all(np.abs(mem[..., :2]) <= cfg.half_world + 1e-3)
    assert mem[..., 2].min() >= 0.0 and mem[..., 2].max() <= 1.0 + 1e-6


def test_fruit_field_bounded():
    """Fruit stays inside its own capacity, which is a different field from the
    plant capacity -- a copy-paste of the grass clamp would pass everywhere
    except the cells that matter."""
    cfg = tiny_cfg()
    state, _ms = run(cfg, 300)
    _s, _k, _sf, _sc, terrain = new_world(cfg)
    fcap = np.asarray(terrain.fruit_capacity)
    fruit = np.asarray(state.fruit)
    assert np.all(np.isfinite(fruit))
    assert fruit.min() >= 0.0
    assert np.all(fruit <= fcap + 1e-4)


def test_forage_water_cannot_replace_drinking():
    """At equilibrium plant density, grazing subsidises thirst but never covers
    it -- water stays a spatial constraint, not a rate one.

    Stated against the *equilibrium* field on purpose. The stronger claim (that
    grazing can never sustain an agent) is false at any useful `forage_water_frac`:
    a forager crossing virgin ground strips far more than a standing one and does
    go net-positive. That is the correct ecology -- inland self-sufficiency is a
    low-density privilege that vanishes as the interior fills up and draws the
    field down -- so the invariant is written where the population actually lives.
    """
    cfg = tiny_cfg()
    state, _ms = run(cfg, 600)

    herb = (state.alive & (state.diet < 0.35)).astype(jnp.float32)
    n = jnp.maximum(jnp.sum(herb), 1.0)
    food = float(jnp.sum(state.last_food * herb) / n)
    thrust = 0.5 * (state.last_output[:, 1] + 1.0)
    mean_thrust = float(jnp.sum(thrust * herb) / n)

    subsidy = food * cfg.forage_water_frac
    cost = cfg.base_water_cost + cfg.move_water_cost * mean_thrust
    assert subsidy < 0.6 * cost, (
        f"forage water {subsidy:.4f}/step covers {100*subsidy/cost:.0f}% of the "
        f"{cost:.4f}/step thirst cost -- the river has stopped mattering")


def test_forage_water_not_created_for_the_dead():
    """Grazing must not hydrate culled slots -- `demand` is gated on `alive`."""
    cfg = tiny_cfg()
    state, _ms = run(cfg, 200)
    _e, _p, gain, water_gain = dynamics.graze(state, cfg)
    dead = ~state.alive
    assert float(jnp.max(jnp.abs(gain[dead]))) == 0.0
    assert float(jnp.max(jnp.abs(water_gain[dead]))) == 0.0


def test_metrics_water_bounded():
    """The spatial-occupancy metrics must stay interpretable.

    These are the readouts the river-camping work is measured against, so a
    silently out-of-range value would invalidate the comparison rather than
    fail loudly somewhere else.
    """
    cfg = tiny_cfg()
    _state, ms = run(cfg, 200)
    m = {k: np.asarray(v) for k, v in ms._asdict().items()}

    for name in ("herb_water_dist", "carn_water_dist",
                 "water_bound_frac", "inland_frac"):
        assert np.all(np.isfinite(m[name])), f"non-finite in {name}"

    for name in ("water_bound_frac", "inland_frac"):
        assert np.all(m[name] >= 0.0) and np.all(m[name] <= 1.0), name

    # A distance on the torus cannot exceed the half-diagonal.
    limit = cfg.half_world * np.sqrt(2.0) + 1e-4
    assert np.all(m["herb_water_dist"] <= limit)
    assert np.all(m["carn_water_dist"] <= limit)

    # The two fractions partition opposite ends of the same axis: an agent
    # cannot be both inside the drinkable band and beyond the sensor's reach.
    assert np.all(m["water_bound_frac"] + m["inland_frac"] <= 1.0 + 1e-4)
