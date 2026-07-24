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
from underworld import genome as genome_mod
from underworld import terrain as terrain_mod
from underworld.state import (armor_of, attack_range_of, escape_of, init_state,
                              invest_of, size_of, spike_of)


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
    assert s.trample.shape == (cfg.n_cells,)
    assert int(jnp.sum(s.alive)) == cfg.n_init
    assert float(jnp.max(jnp.abs(s.trample))) == 0.0  # no one has walked yet


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
    # Trample field is bounded in [0, 1] by construction (clipped every step).
    assert float(jnp.min(state.trample)) >= 0.0
    assert float(jnp.max(state.trample)) <= 1.0 + 1e-6
    # Positions stay on the torus.
    assert float(jnp.min(state.pos)) >= 0.0
    assert float(jnp.max(state.pos)) < cfg.world_size


def test_layer_can_be_switched_off():
    """A zeroed resource layer is a control arm, not a crash.

    `fruit_max=0` is how "is the fruit layer doing anything?" gets answered, so
    the kernel has to survive a capacity field that is identically zero. It did
    not: `regrow`'s baseline divided by `ref_max`, giving 0/0, and the NaN spread
    through energy into position -- the run still printed a flat population and a
    plausible-looking table, which is the failure mode worth a test.
    """
    cfg = tiny_cfg(fruit_max=0.0)
    state, ms = run(cfg, 300)
    for name, arr in state._asdict().items():
        assert bool(jnp.all(jnp.isfinite(arr))), f"non-finite in {name}"
    assert float(jnp.max(state.fruit)) == 0.0
    assert int(jnp.sum(state.alive)) > 0


def test_trample_default_is_truly_off():
    """`trample_impact` defaults to 0.0 (docs/TODO.md priority 3, Stage 0), the
    reverse of `fruit_max`'s "0 disables" convention -- this mechanism doesn't
    exist unless an ablation arm explicitly turns it on. The field itself still
    accumulates every step (it costs nothing not to), so the real invariant is
    that its *dynamics* (rate, decay) must have zero effect on the rest of the
    sim when impact=0 -- proving it is inert, not merely small. Two configs
    with wildly different trample_rate/trample_decay but the same seed and
    trample_impact=0.0 must produce the same world, even though their trample
    fields themselves differ.
    """
    cfg_a = tiny_cfg(trample_rate=0.01, trample_decay=0.99)
    cfg_b = tiny_cfg(trample_rate=0.5, trample_decay=0.5)
    assert cfg_a.trample_impact == 0.0 and cfg_b.trample_impact == 0.0

    state_a, ms_a = run(cfg_a, 300)
    state_b, ms_b = run(cfg_b, 300)

    # The trample fields really did evolve differently under the two settings.
    assert not bool(jnp.allclose(state_a.trample, state_b.trample))
    # ...but nothing downstream of `effective_capacity` did: same life/death
    # structure and matching plant/energy fields (tolerance matches
    # test_determinism's allowance for GPU scatter-add reordering).
    assert bool(jnp.array_equal(state_a.alive, state_b.alive))
    assert bool(jnp.array_equal(ms_a.population, ms_b.population))
    assert bool(jnp.allclose(state_a.plant, state_b.plant, atol=1e-4))
    assert bool(jnp.allclose(state_a.energy, state_b.energy, atol=1e-2))


def test_trample_impact_erodes_plant_capacity():
    """When actually turned on, trample must erode what a cell can grow.

    Tested directly against `ecology.regrow` with a fixed trample field rather
    than by running the full sim: over a long horizon, two arms with different
    capacity from step 1 onward diverge chaotically in population and
    position (much like two different seeds), so comparing aggregate `plant`
    totals after a run is confounded by that divergence, not just by the
    mechanism under test.
    """
    from underworld import ecology
    cfg = tiny_cfg(trample_impact=0.5)
    capacity = jnp.full((cfg.n_cells,), cfg.plant_max)
    trample = jnp.ones((cfg.n_cells,))  # maximum trample everywhere
    effective = jnp.clip(capacity * (1.0 - trample * cfg.trample_impact), 0.0, None)
    field = jnp.full((cfg.n_cells,), cfg.plant_max)  # start saturated at capacity
    grown_on = ecology.regrow(field, effective, cfg.regrow_rate,
                              cfg.regrow_baseline, cfg.plant_max)
    grown_off = ecology.regrow(field, capacity, cfg.regrow_rate,
                               cfg.regrow_baseline, cfg.plant_max)
    assert bool(jnp.all(grown_on < grown_off))


def test_trample_path_gain_default_is_truly_off():
    """`trample_path_gain` defaults to 0.0 -- the sign-corrected companion to
    `trample_impact` from docs/biology.md SS11.1 (real trail formation lowers
    movement cost; it does not erode food capacity, which was measured to be a
    *dispersing* feedback instead). Same backward-compatibility invariant as
    `test_trample_default_is_truly_off`: trample_rate/trample_decay may differ,
    but with trample_path_gain=0.0 `dynamics.act` never reads the field, so the
    two runs must be identical downstream even though the trample fields
    themselves differ.
    """
    cfg_a = tiny_cfg(trample_rate=0.01, trample_decay=0.99)
    cfg_b = tiny_cfg(trample_rate=0.5, trample_decay=0.5)
    assert cfg_a.trample_path_gain == 0.0 and cfg_b.trample_path_gain == 0.0

    state_a, ms_a = run(cfg_a, 300)
    state_b, ms_b = run(cfg_b, 300)

    # The trample fields really did evolve differently under the two settings.
    assert not bool(jnp.allclose(state_a.trample, state_b.trample))
    # ...but nothing downstream of `path_relief` did.
    assert bool(jnp.array_equal(state_a.alive, state_b.alive))
    assert bool(jnp.array_equal(ms_a.population, ms_b.population))
    assert bool(jnp.allclose(state_a.pos, state_b.pos, atol=1e-3))
    assert bool(jnp.allclose(state_a.energy, state_b.energy, atol=1e-2))


def test_trample_path_gain_speeds_up_forest_movement():
    """When turned on, a trampled forest cell must cost less speed than an
    untrampled one -- tested directly against `dynamics.act` rather than
    end-to-end, for the same reason `test_trample_impact_erodes_plant_capacity`
    is: comparing full-sim aggregates after divergence would confound the
    mechanism under test with chaotic drift between runs.
    """
    cfg = tiny_cfg(trample_path_gain=0.5, forest_slow=0.25)
    key = jax.random.PRNGKey(0)
    terrain = terrain_mod.build(cfg)
    state = init_state(cfg, key, terrain)

    # Force every living agent into the cell with the densest canopy, at full
    # thrust, heading fixed, so any speed difference comes only from trample.
    forest_cell = int(jnp.argmax(terrain.forest))
    assert float(terrain.forest[forest_cell]) > 0.0
    fx = (forest_cell % cfg.grid + 0.5) * cfg.cell_size
    fy = (forest_cell // cfg.grid + 0.5) * cfg.cell_size
    state = state._replace(
        pos=jnp.tile(jnp.array([fx, fy]), (cfg.n_max, 1)),
        heading=jnp.zeros(cfg.n_max),
    )
    outputs = jnp.tile(jnp.array([0.0, 1.0]), (cfg.n_max, 1))  # turn=0, thrust=1

    trampled = state._replace(trample=jnp.ones(cfg.n_cells))
    untrampled = state._replace(trample=jnp.zeros(cfg.n_cells))

    state_t, _, _ = dynamics.act(trampled, outputs, terrain, cfg)
    state_u, _, _ = dynamics.act(untrampled, outputs, terrain, cfg)

    alive = state.alive
    speed_t = jnp.linalg.norm(state_t.vel[alive], axis=1)
    speed_u = jnp.linalg.norm(state_u.vel[alive], axis=1)
    assert bool(jnp.all(speed_t > speed_u))


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
    _e, meat_gain, damage, _w, water_gain, water_damage, _v = \
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


def test_scavenge_feeds_carnivores_not_herbivores():
    """dynamics.scavenge: a carnivore standing on carrion gains energy from it; a
    herbivore on the same carrion gains ~0 (diet-gated). Carrion is consumed.
    (docs/multispecies_feasibility.md §4)"""
    from underworld.state import diet_of, pos_to_cell
    cfg = tiny_cfg(n_max=2, n_init=2)
    terrain = terrain_mod.build(cfg)
    state = init_state(cfg, jax.random.PRNGKey(0), terrain)
    genome = state.genome.at[0, cfg.diet_index].set(6.0).at[1, cfg.diet_index].set(-6.0)
    pos = jnp.zeros((2, 2))                                   # both on the same cell
    carrion = jnp.zeros(cfg.n_cells).at[pos_to_cell(pos, cfg)[0]].set(5.0)
    state = state._replace(genome=genome, diet=diet_of(genome, cfg), pos=pos,
                           alive=jnp.array([True, True]), energy=jnp.array([10.0, 10.0]),
                           carrion=carrion)
    energy, carr, gain, _w = dynamics.scavenge(state, cfg)
    assert float(gain[0]) > 0.0                              # carnivore scavenges
    assert float(gain[0]) > float(gain[1]) * 5.0             # far more than the herbivore
    assert float(energy[0]) > 10.0                           # carnivore gained energy
    assert float(jnp.sum(carr)) < 5.0                        # carrion was consumed


def test_carrion_off_by_default_stays_zero():
    """With carrion_enabled False (default) no corpse is deposited and nothing scavenges,
    so the field stays identically 0 -- the world is bit-exact the pre-carrion kernel."""
    cfg = tiny_cfg()
    state, _ms = run(cfg, 50)
    assert float(jnp.sum(state.carrion)) == 0.0, "carrion must stay 0 when disabled"


def test_density_dependent_reproduction_suppresses_crowded_births():
    """With density_repro_penalty>0 a crowded cell raises the energy bar to breed, so
    the same breeders produce fewer offspring than with the penalty off
    (docs/herbivore_overpopulation.md L6). Default penalty 0 is a pure-energy gate."""
    from underworld.state import pos_to_cell
    cfg_off = tiny_cfg(n_max=64, n_init=16)                          # penalty 0
    cfg_on = tiny_cfg(n_max=64, n_init=16, density_repro_penalty=2.0,
                      density_repro_cap=4.0)
    terrain = terrain_mod.build(cfg_off)
    state = init_state(cfg_off, jax.random.PRNGKey(0), terrain)
    # 16 breeders, energy just above the base threshold, all packed into one cell.
    alive = jnp.arange(cfg_off.n_max) < 16
    energy = jnp.where(alive, cfg_off.repro_threshold + 4.0, 0.5)
    pos = jnp.where(alive[:, None], jnp.zeros((cfg_off.n_max, 2)), state.pos)
    state = state._replace(alive=alive, energy=energy, pos=pos)
    crowd = jnp.zeros(cfg_off.n_cells).at[pos_to_cell(state.pos, cfg_off)].add(
        state.alive.astype(jnp.float32))                            # ~16 in the packed cell

    key = jax.random.PRNGKey(1)
    born_off = int(jnp.sum(reproduction.reproduce(state, key, cfg_off).alive) - 16)
    born_on = int(jnp.sum(
        reproduction.reproduce(state, key, cfg_on, crowd).alive) - 16)
    assert born_off > 0, "the off arm should breed (energy is above threshold)"
    # crowd 16, cap 4 -> crowding clips to 1 -> eff_threshold = 16*(1+2*1) = 48 > energy 20
    assert born_on == 0, "a saturated cell must suppress births under the penalty"


def test_investment_gene_controls_the_handover():
    """The energy a child receives must follow its parent's gene, not a constant.

    Two parents identical but for the investment gene must hand over different
    amounts. Without this the gene is inert and every downstream measurement of
    "evolved investment" would be measuring drift in a gene nothing reads.
    """
    cfg = tiny_cfg(n_max=64, n_init=8)
    key = jax.random.PRNGKey(0)
    terrain = terrain_mod.build(cfg)
    state = init_state(cfg, key, terrain)

    # Two breeders, same energy, opposite investment genes; nobody else alive.
    genome = state.genome.at[0, cfg.invest_index].set(-4.0)   # ~invest_min
    genome = genome.at[1, cfg.invest_index].set(4.0)          # ~invest_min+span
    energy = jnp.where(jnp.arange(cfg.n_max) < 2, cfg.repro_threshold + 5.0, 0.5)
    state = state._replace(
        alive=jnp.arange(cfg.n_max) < 2, genome=genome, energy=energy)

    frac = np.asarray(invest_of(state.genome, cfg))
    assert frac[0] == pytest.approx(cfg.invest_min, abs=0.02)
    assert frac[1] == pytest.approx(cfg.invest_min + cfg.invest_span, abs=0.02)

    child = reproduction.reproduce(state, jax.random.PRNGKey(1), cfg)
    born = np.asarray(child.alive & ~state.alive)
    assert born.sum() == 2, "expected both parents to breed"

    # Each child's energy should be its own parent's energy times that parent's
    # fraction. Match children to parents by which fraction the amount implies.
    parent_energy = float(cfg.repro_threshold + 5.0)
    got = sorted(float(child.energy[i]) for i in np.flatnonzero(born))
    assert got[0] == pytest.approx(parent_energy * frac[0], rel=0.02)
    assert got[1] == pytest.approx(parent_energy * frac[1], rel=0.02)


def test_investment_gene_is_bounded_and_heritable():
    """The gene stays inside [invest_min, invest_min+span] however extreme the
    underlying value, and crossover never blends it between parents."""
    cfg = tiny_cfg()
    wild = jnp.linspace(-50.0, 50.0, cfg.n_max)
    genome = jnp.zeros((cfg.n_max, cfg.genome_size)).at[:, cfg.invest_index].set(wild)
    frac = np.asarray(invest_of(genome, cfg))
    assert frac.min() >= cfg.invest_min - 1e-6
    assert frac.max() <= cfg.invest_min + cfg.invest_span + 1e-6

    # Diet is exempt from crossover; investment deliberately is NOT. Exempting a
    # gene makes it non-recombining and therefore permanently linked to every
    # other exempt gene, which would let `invest_diet_corr` climb by hitchhiking
    # instead of by selection -- corrupting the one metric that exists to detect
    # whether carnivores genuinely provision differently.
    a = jnp.zeros((cfg.n_max, cfg.genome_size)).at[:, cfg.invest_index].set(-3.0)
    b = jnp.ones((cfg.n_max, cfg.genome_size)).at[:, cfg.invest_index].set(3.0)
    mixed = np.asarray(genome_mod.crossover(a, b, jax.random.PRNGKey(0), cfg))
    assert np.all(mixed[:, cfg.diet_index] == 0.0), "diet must come from parent A"
    inv = mixed[:, cfg.invest_index]
    assert set(np.unique(inv)) == {-3.0, 3.0}, "investment must recombine"
    assert 0.2 < float((inv == 3.0).mean()) < 0.8, "recombination looks biased"
    # ...and brain genes mix too, or crossover is doing nothing at all.
    assert 0.0 < float(np.mean(mixed[:, :cfg.brain_params])) < 1.0


def test_size_gene_is_bounded_and_exempt_from_crossover():
    """The size gene stays inside [size_min, size_min+span] however extreme the
    underlying value, and (like diet, unlike investment) never recombines --
    the brain is not adapted to size differences yet, but the exemption is
    already in place for the day something is (see genome.crossover)."""
    cfg = tiny_cfg()
    wild = jnp.linspace(-50.0, 50.0, cfg.n_max)
    genome = jnp.zeros((cfg.n_max, cfg.genome_size)).at[:, cfg.size_index].set(wild)
    size = np.asarray(size_of(genome, cfg))
    assert size.min() >= cfg.size_min - 1e-6
    assert size.max() <= cfg.size_min + cfg.size_span + 1e-6
    # A gene of 0 must sigmoid to the old, unscaled behaviour (size == 1.0),
    # or a fresh population would start already off-baseline.
    neutral = float(size_of(jnp.zeros((1, cfg.genome_size)), cfg)[0])
    assert neutral == pytest.approx(1.0, abs=1e-4)

    a = jnp.zeros((cfg.n_max, cfg.genome_size)).at[:, cfg.size_index].set(-3.0)
    b = jnp.ones((cfg.n_max, cfg.genome_size)).at[:, cfg.size_index].set(3.0)
    mixed = np.asarray(genome_mod.crossover(a, b, jax.random.PRNGKey(0), cfg))
    assert np.all(mixed[:, cfg.size_index] == -3.0), "size must come from parent A"
    assert np.all(mixed[:, cfg.diet_index] == 0.0), "diet must still come from parent A"


def test_size_gene_scales_metabolism_and_water_capacity():
    """Bigger bodies cost more per step (Kleiber, ^0.75) but hold more water
    (a volume, ^1.0) -- these are two different exponents and must not collapse
    to the same scaling, or the whole water-economy design in
    docs/biology.md S8.2 is measuring nothing."""
    cfg = tiny_cfg()
    alive = jnp.array([True, True])
    thrust = jnp.array([0.5, 0.5])
    diet = jnp.array([0.0, 0.0])
    climb = jnp.array([0.0, 0.0])
    small, big = 0.5, 1.5

    e_small = dynamics.metabolize(
        jnp.array([10.0, 10.0]), thrust, diet, climb, alive, cfg,
        jnp.array([small, small]))[0]
    e_big = dynamics.metabolize(
        jnp.array([10.0, 10.0]), thrust, diet, climb, alive, cfg,
        jnp.array([big, big]))[0]
    cost_small = 10.0 - float(e_small)
    cost_big = 10.0 - float(e_big)
    assert cost_big > cost_small, "a bigger body must cost more to run"
    ratio = cost_big / cost_small
    assert ratio == pytest.approx((big / small) ** 0.75, rel=0.05)

    w_small = dynamics.thirst(jnp.array([10.0, 10.0]), thrust, alive, cfg,
                               jnp.array([small, small]))[0]
    w_big = dynamics.thirst(jnp.array([10.0, 10.0]), thrust, alive, cfg,
                             jnp.array([big, big]))[0]
    wcost_small = 10.0 - float(w_small)
    wcost_big = 10.0 - float(w_big)
    assert wcost_big / wcost_small == pytest.approx((big / small) ** 0.75, rel=0.05)

    # water_max scales isometrically (^1.0), a different exponent from the
    # ^0.75 cost above -- collapsing the two would mean size buys nothing.
    state, key, step_fn, _scan_fn, terrain = new_world(cfg)
    st = state._replace(
        water=jnp.full((cfg.n_max,), 100.0),  # already over any plausible cap
        alive=jnp.ones((cfg.n_max,), dtype=bool),
    )
    size = jnp.full((cfg.n_max,), big)
    capped, _ = dynamics.drink(st, terrain, cfg, size)
    assert float(jnp.max(capped)) == pytest.approx(cfg.water_max * big, rel=1e-4)


def test_attack_and_escape_genes_neutral_start_and_bounds():
    """The red-queen pair (docs/attack_range_redqueen.md) must start neutral so a
    fresh population reproduces the pre-gene world: attack_range_of at gene 0 equals
    the old `attack_range` constant (6.0), escape_of at gene 0 is exactly 0 (no
    evasion). Both stay inside their declared ranges however extreme the gene."""
    cfg = tiny_cfg()
    wild = jnp.linspace(-50.0, 50.0, cfg.n_max)
    g_atk = jnp.zeros((cfg.n_max, cfg.genome_size)).at[:, cfg.attack_index].set(wild)
    atk = np.asarray(attack_range_of(g_atk, cfg))
    assert atk.min() >= cfg.attack_min - 1e-6
    assert atk.max() <= cfg.attack_max + 1e-6
    neutral_atk = float(attack_range_of(jnp.zeros((1, cfg.genome_size)), cfg)[0])
    assert neutral_atk == pytest.approx(cfg.attack_range, abs=1e-4)

    g_esc = jnp.zeros((cfg.n_max, cfg.genome_size)).at[:, cfg.escape_index].set(wild)
    esc = np.asarray(escape_of(g_esc, cfg))
    assert esc.min() >= -1e-6
    assert esc.max() <= cfg.escape_span * 0.5 + 1e-6
    neutral_esc = float(escape_of(jnp.zeros((1, cfg.genome_size)), cfg)[0])
    assert neutral_esc == pytest.approx(0.0, abs=1e-6)


def test_red_queen_taxes_hit_energy_not_water():
    """The attack/escape taxes MUST ride the energy ledger, never thirst
    (docs/trait_roadmap.md §5): a water-ledger cost would recreate the body-size
    gene's juvenile-thirst censoring. A big-reach carnivore and a high-escape
    herbivore both lose extra *energy* in metabolize; neither loses extra water in
    thirst."""
    cfg = tiny_cfg()
    alive = jnp.array([True, True])
    thrust = jnp.array([0.5, 0.5])
    climb = jnp.array([0.0, 0.0])
    size = jnp.array([1.0, 1.0])
    e0 = jnp.array([10.0, 10.0])

    # A carnivore at max reach vs at the neutral baseline: extra energy only.
    diet_c = jnp.array([1.0, 1.0])
    reach_hi = jnp.array([cfg.attack_max, cfg.attack_max])
    reach_base = jnp.array([cfg.attack_range, cfg.attack_range])
    esc0 = jnp.array([0.0, 0.0])
    e_hi = float(dynamics.metabolize(e0, thrust, diet_c, climb, alive, cfg, size,
                                     reach_hi, esc0)[0])
    e_base = float(dynamics.metabolize(e0, thrust, diet_c, climb, alive, cfg, size,
                                       reach_base, esc0)[0])
    assert e_base - e_hi == pytest.approx(
        cfg.attack_cost * (cfg.attack_max - cfg.attack_range), rel=1e-4)

    # A herbivore with high escape pays an energy tax scaled by (1-diet).
    diet_h = jnp.array([0.0, 0.0])
    esc_hi = jnp.array([5.0, 5.0])
    e_esc = float(dynamics.metabolize(e0, thrust, diet_h, climb, alive, cfg, size,
                                      reach_base, esc_hi)[0])
    e_noesc = float(dynamics.metabolize(e0, thrust, diet_h, climb, alive, cfg, size,
                                        reach_base, esc0)[0])
    assert e_noesc - e_esc == pytest.approx(cfg.escape_cost * 5.0, rel=1e-4)

    # Thirst must be untouched by either gene -- it takes neither argument.
    w0 = jnp.array([10.0, 10.0])
    w = dynamics.thirst(w0, thrust, alive, cfg, size)
    assert float(w[0]) == pytest.approx(
        10.0 - (cfg.base_water_cost + cfg.move_water_cost * 0.5) * (1.0 ** 0.75),
        rel=1e-4)


def test_prey_escape_shrinks_effective_attack_reach():
    """A prey's escape gene shortens the attacker's *effective* reach: a bite that
    lands on a zero-escape prey misses the same prey once it evolves enough evasion
    (docs/attack_range_redqueen.md)."""
    cfg = tiny_cfg(n_max=2, n_init=2)
    terrain = terrain_mod.build(cfg)
    state = init_state(cfg, jax.random.PRNGKey(0), terrain)

    # Agent 0 carnivore at neutral reach (6.0); agent 1 herbivore prey at dist 5.0.
    genome = state.genome.at[:, cfg.attack_index].set(0.0)   # reach 6.0
    genome = genome.at[0, cfg.diet_index].set(6.0)           # sigmoid -> ~1 carnivore
    genome = genome.at[1, cfg.diet_index].set(-6.0)          # sigmoid -> ~0 herbivore
    from underworld.state import diet_of
    state = state._replace(
        genome=genome, diet=diet_of(genome, cfg),
        energy=jnp.array([10.0, 10.0]), water=jnp.array([10.0, 10.0]),
        alive=jnp.array([True, True]))
    nbr = jnp.array([[1], [0]])
    dist = jnp.array([[5.0], [5.0]])
    valid = jnp.array([[True], [True]])

    # No evasion -> the bite lands.
    st0 = state._replace(genome=state.genome.at[1, cfg.escape_index].set(0.0))
    _, meat0, _, _, _, _, _ = dynamics.predation(st0, nbr, dist, valid, cfg)
    assert float(meat0[0]) > 0.0

    # High evasion (escape ~5.9, effective reach ~0.1) -> the same bite misses.
    st1 = state._replace(genome=state.genome.at[1, cfg.escape_index].set(6.0))
    _, meat1, _, _, _, _, _ = dynamics.predation(st1, nbr, dist, valid, cfg)
    assert float(meat1[0]) == 0.0


def test_defence_genes_neutral_start_and_bounds():
    """The morphological defences (docs/trait_defense_catalog.md) must start neutral
    so a fresh population reproduces the pre-gene world: armor_of and spike_of at
    gene 0 are exactly 0 (no defence is ever seeded, only evolved), and both stay
    inside [0, span/2] however extreme the gene."""
    cfg = tiny_cfg()
    wild = jnp.linspace(-50.0, 50.0, cfg.n_max)
    for of, idx, span in ((armor_of, cfg.armor_index, cfg.armor_span),
                          (spike_of, cfg.spike_index, cfg.spike_span)):
        g = jnp.zeros((cfg.n_max, cfg.genome_size)).at[:, idx].set(wild)
        v = np.asarray(of(g, cfg))
        assert v.min() >= -1e-6
        assert v.max() <= span * 0.5 + 1e-6
        neutral = float(of(jnp.zeros((1, cfg.genome_size)), cfg)[0])
        assert neutral == pytest.approx(0.0, abs=1e-6)


def test_defence_taxes_hit_energy_not_water():
    """The armour/spike upkeep and the venom drain MUST ride the energy ledger, never
    thirst (docs/trait_addition_feasibility.md §B.2): a water-ledger cost would recreate
    the body-size gene's juvenile-thirst censoring. Armour is (1-diet)-gated (only the
    hunted pay); the spike tax is UNIVERSAL (both lineages grow spikes, offensive for
    carnivores / defensive for herbivores -- docs/trait_defense_landing.md §7); none of
    the three touch water in thirst (which takes none of them)."""
    cfg = tiny_cfg()
    alive = jnp.array([True, True])
    thrust = jnp.array([0.5, 0.5])
    climb = jnp.array([0.0, 0.0])
    size = jnp.array([1.0, 1.0])
    e0 = jnp.array([10.0, 10.0])
    diet_h = jnp.array([0.0, 0.0])
    base = jnp.array([cfg.attack_range, cfg.attack_range])
    z = jnp.array([0.0, 0.0])

    # Armour tax = armor_cost * armor * (1-diet), on top of the neutral baseline.
    arm = jnp.array([0.4, 0.4])
    e_arm = float(dynamics.metabolize(e0, thrust, diet_h, climb, alive, cfg, size,
                                      base, z, arm, z)[0])
    e_noarm = float(dynamics.metabolize(e0, thrust, diet_h, climb, alive, cfg, size,
                                        base, z, z, z)[0])
    assert e_noarm - e_arm == pytest.approx(cfg.armor_cost * 0.4, rel=1e-4)

    # Spike tax = spike_cost * spike (universal -- same for a herbivore here).
    spk = jnp.array([0.4, 0.4])
    e_spk = float(dynamics.metabolize(e0, thrust, diet_h, climb, alive, cfg, size,
                                      base, z, z, spk)[0])
    assert e_noarm - e_spk == pytest.approx(cfg.spike_cost * 0.4, rel=1e-4)

    # A carnivore (diet 1) pays ~0 for ARMOUR (still (1-diet)-gated), but DOES pay the
    # UNIVERSAL spike tax for its offensive spikes.
    diet_c = jnp.array([1.0, 1.0])
    e_carn_arm = float(dynamics.metabolize(e0, thrust, diet_c, climb, alive, cfg, size,
                                           base, z, arm, z)[0])
    e_carn_none = float(dynamics.metabolize(e0, thrust, diet_c, climb, alive, cfg, size,
                                            base, z, z, z)[0])
    assert e_carn_none - e_carn_arm == pytest.approx(0.0, abs=1e-6)   # no armour tax
    e_carn_spk = float(dynamics.metabolize(e0, thrust, diet_c, climb, alive, cfg, size,
                                           base, z, z, spk)[0])
    assert e_carn_none - e_carn_spk == pytest.approx(cfg.spike_cost * 0.4, rel=1e-4)

    # Venom drain also rides the energy ledger, scaled by clip(venom, 0, 1).
    ven = jnp.array([1.0, 1.0])
    e_ven = float(dynamics.metabolize(e0, thrust, diet_h, climb, alive, cfg, size,
                                      base, z, z, z, ven)[0])
    assert e_noarm - e_ven == pytest.approx(cfg.venom_drain * 1.0, rel=1e-4)

    # Thirst is untouched by any defence gene or venom -- it takes none of them.
    w = dynamics.thirst(jnp.array([10.0, 10.0]), thrust, alive, cfg, size)
    assert float(w[0]) == pytest.approx(
        10.0 - (cfg.base_water_cost + cfg.move_water_cost * 0.5) * (1.0 ** 0.75),
        rel=1e-4)


def test_armor_reduces_bite_and_spike_hurts_attacker():
    """Armour cuts the energy a prey loses to a bite; a carnivore's own spikes make its
    bite hit HARDER (offense); a prey's spikes ENVENOM the attacker (a deposit onto the
    biter, not an instant reflect) -- all relative to the same undefended baseline, and
    predation must still never create energy (docs/trait_defense_landing.md §7)."""
    cfg = tiny_cfg(n_max=2, n_init=2)
    terrain = terrain_mod.build(cfg)
    state = init_state(cfg, jax.random.PRNGKey(0), terrain)

    # Agent 0 carnivore, agent 1 herbivore prey at dist 3.0 well inside reach 6.0.
    from underworld.state import diet_of
    genome = state.genome.at[:, cfg.attack_index].set(0.0)
    genome = genome.at[:, cfg.escape_index].set(0.0)
    genome = genome.at[0, cfg.diet_index].set(6.0)      # carnivore
    genome = genome.at[1, cfg.diet_index].set(-6.0)     # herbivore
    state = state._replace(
        genome=genome, diet=diet_of(genome, cfg),
        energy=jnp.array([10.0, 10.0]), water=jnp.array([10.0, 10.0]),
        alive=jnp.array([True, True]))
    nbr = jnp.array([[1], [0]])
    dist = jnp.array([[3.0], [3.0]])
    valid = jnp.array([[True], [True]])

    # Baseline: no defence, and the attacker carnivore has no offensive spike.
    g0 = state.genome.at[:, cfg.spike_index].set(0.0).at[1, cfg.armor_index].set(0.0)
    e0, meat0, dmg0, _, _, _, ven0 = dynamics.predation(state._replace(genome=g0),
                                                        nbr, dist, valid, cfg)
    assert float(dmg0[1]) > 0.0                          # the prey took a bite
    assert float(jnp.sum(meat0)) <= float(jnp.sum(dmg0)) + 1e-4  # no energy created
    assert float(ven0[0]) == 0.0                         # no spikes -> no venom

    # Armour: the prey loses strictly less energy to the same bite.
    g_arm = g0.at[1, cfg.armor_index].set(6.0)             # armor ~0.5
    _, _, dmg_arm, _, _, _, _ = dynamics.predation(state._replace(genome=g_arm),
                                                   nbr, dist, valid, cfg)
    assert float(dmg_arm[1]) < float(dmg0[1])

    # Carnivore OFFENSE: the attacker's (agent 0) own spikes make its bite hit harder.
    g_off = g0.at[0, cfg.spike_index].set(6.0)             # attacker spike ~0.5
    _, _, dmg_off, _, _, _, _ = dynamics.predation(state._replace(genome=g_off),
                                                   nbr, dist, valid, cfg)
    assert float(dmg_off[1]) > float(dmg0[1])             # spiked carnivore bites harder

    # Herbivore DEFENSE: a spiked prey (agent 1) envenoms its attacker (venom_deposit,
    # the 7th return) -- non-lethal, the bite still lands, and no energy is created.
    g_def = g0.at[1, cfg.spike_index].set(6.0)             # prey spike ~0.5
    e_d, meat_d, dmg_d, _, _, _, ven_d = dynamics.predation(state._replace(genome=g_def),
                                                            nbr, dist, valid, cfg)
    assert float(ven_d[0]) > 0.0                          # attacker gets envenomed
    assert float(jnp.sum(meat_d)) <= float(jnp.sum(dmg_d)) + 1e-4


def test_venom_slows_movement():
    """An envenomed agent (venom>0) moves slower than an identical un-envenomed one
    under the same thrust -- the herbivore->carnivore retaliation debuff acting in
    `act` (docs/trait_defense_landing.md §7)."""
    cfg = tiny_cfg(n_max=2, n_init=2)
    terrain = terrain_mod.build(cfg)
    state = init_state(cfg, jax.random.PRNGKey(0), terrain)
    # Same cell (so identical forest slow) and heading; the only difference is venom.
    state = state._replace(
        pos=jnp.zeros((2, 2)), heading=jnp.zeros(2), alive=jnp.array([True, True]),
        venom=jnp.array([0.0, 1.0]))
    outputs = jnp.zeros((2, cfg.out_dim)).at[:, 1].set(1.0)   # full forward thrust
    moved, _thrust, _climb = dynamics.act(state, outputs, terrain, cfg)
    v_clean = float(jnp.linalg.norm(moved.vel[0]))
    v_venom = float(jnp.linalg.norm(moved.vel[1]))
    assert v_venom < v_clean
    assert v_venom == pytest.approx(v_clean * (1.0 - cfg.venom_slow), rel=1e-3)


def test_defence_genes_recombine_unlike_size():
    """Armour and spike feed predation but not the sensorimotor loop, so like escape
    (and unlike size/diet) they are NOT crossover-exempt -- they recombine freely,
    keeping the G-matrix estimator honest (docs/trait_defense_catalog.md)."""
    cfg = tiny_cfg()
    a = jnp.zeros((cfg.n_max, cfg.genome_size))
    a = a.at[:, cfg.armor_index].set(-3.0).at[:, cfg.spike_index].set(-3.0)
    b = jnp.ones((cfg.n_max, cfg.genome_size))
    b = b.at[:, cfg.armor_index].set(3.0).at[:, cfg.spike_index].set(3.0)
    mixed = np.asarray(genome_mod.crossover(a, b, jax.random.PRNGKey(0), cfg))
    # At least some agents must have taken the armour/spike gene from parent B --
    # i.e. the columns are not frozen to parent A the way size/diet are.
    assert np.any(mixed[:, cfg.armor_index] == 3.0), "armour must recombine"
    assert np.any(mixed[:, cfg.spike_index] == 3.0), "spike must recombine"


def test_child_water_investment_clamped_to_own_tank():
    """A large parent's absolute water transfer must not exceed what a
    small-size child's own tank (`water_max * size`) can hold -- otherwise the
    excess is silently lost the moment `drink`'s cap is next applied, which
    would make `invest_frac` lie about how much the child actually received."""
    cfg = tiny_cfg(n_max=64, n_init=8)
    key = jax.random.PRNGKey(0)
    terrain = terrain_mod.build(cfg)
    state = init_state(cfg, key, terrain)

    # One breeder: huge investment fraction, and a genome that will mutate the
    # child toward a small size (we can't force the child's exact genome since
    # mutation is stochastic, but we CAN force the parent's water high enough
    # that even a size-1.0 child's cap would be exceeded without clamping).
    genome = state.genome.at[0, cfg.invest_index].set(4.0)     # near invest_max
    genome = genome.at[0, cfg.size_index].set(0.0)             # parent size 1.0
    alive = jnp.arange(cfg.n_max) < 1
    energy = jnp.where(alive, cfg.repro_threshold + 5.0, 0.5)
    water = jnp.where(alive, cfg.water_max * 5.0, 0.1)  # far more than any child's cap
    state = state._replace(alive=alive, genome=genome, energy=energy, water=water)

    child = reproduction.reproduce(state, jax.random.PRNGKey(1), cfg)
    born = np.asarray(child.alive & ~state.alive)
    assert born.sum() == 1
    i = int(np.flatnonzero(born)[0])
    child_size = float(size_of(child.genome[i:i + 1], cfg)[0])
    assert float(child.water[i]) <= cfg.water_max * child_size + 1e-4


def test_water_lactation_floor_is_a_noop_at_default():
    """`water_lactation_floor_frac` defaults to 0.0, which must not change a
    single float versus the old `water = parent_water * invest_frac` formula
    -- this is the golden-band invariant the whole mechanism rests on."""
    cfg = tiny_cfg(n_max=64, n_init=8)
    key = jax.random.PRNGKey(0)
    terrain = terrain_mod.build(cfg)
    state = init_state(cfg, key, terrain)

    genome = state.genome.at[0, cfg.invest_index].set(-1.0)  # low, not floor-clamped
    alive = jnp.arange(cfg.n_max) < 1
    energy = jnp.where(alive, cfg.repro_threshold + 5.0, 0.5)
    water = jnp.where(alive, 6.0, 0.1)
    state = state._replace(alive=alive, genome=genome, energy=energy, water=water)

    child = reproduction.reproduce(state, jax.random.PRNGKey(1), cfg)
    frac = float(invest_of(state.genome, cfg)[0])
    born = np.asarray(child.alive & ~state.alive)
    i = int(np.flatnonzero(born)[0])
    assert float(child.water[i]) == pytest.approx(6.0 * frac, rel=1e-5)


def test_water_lactation_floor_decouples_water_from_energy_investment():
    """The whole point of the mechanism (docs/water_fix_provisioning.md): a
    parent with a LOW invest_frac must still hand over a floored water
    fraction once `water_lactation_floor_frac` is raised, while its ENERGY
    handover stays exactly `invest_frac`-proportional -- unlike raising
    `invest_min` itself (docs/water_system.md arm_B), which moves both
    resources together because they share one gene."""
    cfg = tiny_cfg(n_max=64, n_init=8, water_lactation_floor_frac=0.6)
    key = jax.random.PRNGKey(0)
    terrain = terrain_mod.build(cfg)
    state = init_state(cfg, key, terrain)

    genome = state.genome.at[0, cfg.invest_index].set(-4.0)  # ~invest_min = 0.2
    alive = jnp.arange(cfg.n_max) < 1
    energy = jnp.where(alive, cfg.repro_threshold + 5.0, 0.5)
    water = jnp.where(alive, 6.0, 0.1)
    state = state._replace(alive=alive, genome=genome, energy=energy, water=water)

    frac = float(invest_of(state.genome, cfg)[0])
    assert frac == pytest.approx(cfg.invest_min, abs=0.02)

    child = reproduction.reproduce(state, jax.random.PRNGKey(1), cfg)
    born = np.asarray(child.alive & ~state.alive)
    i = int(np.flatnonzero(born)[0])
    parent_energy = float(cfg.repro_threshold + 5.0)
    # Energy handover still follows invest_frac alone -- the floor never
    # touches it.
    assert float(child.energy[i]) == pytest.approx(parent_energy * frac, rel=0.02)
    # Water handover is floored to 0.6 of the parent's water, well above what
    # the ~0.2 investment gene alone would have given (1.2 vs 3.6).
    assert float(child.water[i]) == pytest.approx(6.0 * 0.6, rel=1e-5)
    assert float(child.water[i]) > 6.0 * frac * 1.5


def test_peer_channel_reveals_similar_diet_neighbours():
    """`prey`/`pred` are diet-*difference* channels and are both exactly zero
    for two agents of near-identical diet -- conspecifics were mutually
    invisible. `peer` is the diet-*similarity* channel added to fix that; it
    must be near its maximum exactly where prey/pred vanish."""
    cfg = tiny_cfg(n_max=8, n_init=2, vision_radius=40.0)
    state, key, step_fn, _scan_fn, terrain = new_world(cfg)
    # Two same-diet agents standing on top of each other.
    pos = state.pos.at[0].set(jnp.array([cfg.world_size / 2, cfg.world_size / 2]))
    pos = pos.at[1].set(jnp.array([cfg.world_size / 2, cfg.world_size / 2]))
    state = state._replace(pos=pos, diet=jnp.full((cfg.n_max,), 0.5))
    state, _ms = step_fn(state, key)

    r = cfg.retina_sectors
    peer_off = 5 * r + 3 + 4 * cfg.memory_slots
    li = np.asarray(state.last_input[0])
    assert float(np.max(li[r:2 * r])) < 1e-3, "prey channel should be ~0 for same diet"
    assert float(np.max(li[2 * r:3 * r])) < 1e-3, "pred channel should be ~0 for same diet"
    assert float(np.max(li[peer_off:peer_off + r])) > 0.5, \
        "peer channel should fire strongly for an adjacent same-diet neighbour"

def test_diet_bimodal_init_can_be_switched_off():
    """docs/biology.md §10.1: the herbivore/carnivore split starts from a
    bimodal founder seeding by default. `diet_bimodal_init=False` is the
    ablation arm -- founders should start from one neutral cluster instead."""
    cfg = tiny_cfg(n_max=4000, n_init=4000, carnivore_init_frac=0.5)
    key = jax.random.PRNGKey(0)
    state = init_state(cfg, key, terrain_mod.build(cfg))
    diet_gene = state.genome[:, cfg.diet_index]
    mid_frac = float(jnp.mean((diet_gene > -0.5) & (diet_gene < 0.5)))
    assert mid_frac < 0.1, f"expected a bimodal split, got {mid_frac:.2f} in the middle"

    cfg2 = tiny_cfg(n_max=4000, n_init=4000, carnivore_init_frac=0.5,
                     diet_bimodal_init=False)
    state2 = init_state(cfg2, key, terrain_mod.build(cfg2))
    diet_gene2 = state2.genome[:, cfg2.diet_index]
    mid_frac2 = float(jnp.mean((diet_gene2 > -0.5) & (diet_gene2 < 0.5)))
    assert mid_frac2 > 0.5, f"expected one neutral cluster, got {mid_frac2:.2f} in the middle"


def test_diet_crossover_exempt_can_be_switched_off():
    """docs/biology.md §10.1: diet is exempt from crossover by default (always
    taken from parent A). `diet_crossover_exempt=False` should let it recombine
    like any other gene."""
    cfg_default = tiny_cfg()
    cfg_ablated = tiny_cfg(diet_crossover_exempt=False)
    a = jnp.zeros((cfg_default.n_max, cfg_default.genome_size)).at[:, cfg_default.diet_index].set(-3.0)
    b = jnp.ones((cfg_default.n_max, cfg_default.genome_size)).at[:, cfg_default.diet_index].set(3.0)

    mixed_default = np.asarray(genome_mod.crossover(a, b, jax.random.PRNGKey(0), cfg_default))
    assert np.all(mixed_default[:, cfg_default.diet_index] == -3.0), \
        "default: diet must always come from parent A"

    mixed_ablated = np.asarray(genome_mod.crossover(a, b, jax.random.PRNGKey(0), cfg_ablated))
    diet_col = mixed_ablated[:, cfg_ablated.diet_index]
    assert set(np.unique(diet_col)) == {-3.0, 3.0}, "diet must recombine when exemption is off"
    assert 0.2 < float((diet_col == 3.0).mean()) < 0.8, "recombination looks biased"


def test_diet_mutation_asymmetric_can_be_switched_off():
    """docs/biology.md §10.1: diet mutates at a slower `diet_mutation_sigma` by
    default. `diet_mutation_asymmetric=False` should make it mutate at the same
    rate as brain genes."""
    cfg_default = tiny_cfg()
    cfg_symmetric = tiny_cfg(diet_mutation_asymmetric=False)
    genome = jnp.zeros((20000, cfg_default.genome_size))
    key = jax.random.PRNGKey(0)

    mutated_default = genome_mod.mutate(genome, key, cfg_default)
    sd_default = float(jnp.std(mutated_default[:, cfg_default.diet_index]))
    assert sd_default == pytest.approx(cfg_default.diet_mutation_sigma, rel=0.05)

    mutated_symmetric = genome_mod.mutate(genome, key, cfg_symmetric)
    sd_symmetric = float(jnp.std(mutated_symmetric[:, cfg_symmetric.diet_index]))
    brain_sd = float(jnp.std(mutated_symmetric[:, 0]))
    assert sd_symmetric == pytest.approx(cfg_default.mutation_sigma, rel=0.05)
    assert sd_symmetric == pytest.approx(brain_sd, rel=0.05), \
        "diet should mutate at the same rate as a brain gene when symmetric"


def test_assortative_mating_can_be_switched_off():
    """docs/biology.md §10.1/§10.5: partners are sorted by diet (assortative)
    by default, so paired agents have near-identical diet. `assortative_mating
    =False` should pair reproducers uniformly at random instead -- the switch
    that theory (Dieckmann & Doebeli 1999) says is what *maintains* a branch,
    so it is tested separately from the other three diet switches."""
    n = 2000
    want = jnp.ones(n, dtype=bool)
    diet = jnp.linspace(0.0, 1.0, n)
    key = jax.random.PRNGKey(0)

    cfg_assort = tiny_cfg(n_max=n, n_init=n)
    partner_assort = reproduction._assortative_mate(want, diet, cfg_assort, key)
    diff_assort = float(jnp.mean(jnp.abs(diet - diet[partner_assort])))

    cfg_random = tiny_cfg(n_max=n, n_init=n, assortative_mating=False)
    partner_random = reproduction._assortative_mate(want, diet, cfg_random, key)
    diff_random = float(jnp.mean(jnp.abs(diet - diet[partner_random])))

    assert diff_assort < 0.05, "assortative mating should pair near-identical diets"
    assert diff_random > diff_assort * 5, "random pairing should not track diet"


def test_peer_channel_can_be_disabled():
    """`peer_channel_enabled=False` is the ablation arm for the peer channel:
    there is no pre-channel population to compare against (it's part of the
    sensory layer), so the control has to be a forced-zero flag rather than a
    removed input. `in_dim`/`genome_size` must stay identical to the enabled
    config, or the two arms would not be genome-compatible."""
    cfg_on = tiny_cfg(n_max=8, n_init=2, vision_radius=40.0)
    cfg_off = tiny_cfg(n_max=8, n_init=2, vision_radius=40.0,
                        peer_channel_enabled=False)
    assert cfg_off.in_dim == cfg_on.in_dim
    assert cfg_off.genome_size == cfg_on.genome_size

    state, key, step_fn, _scan_fn, terrain = new_world(cfg_off)
    pos = state.pos.at[0].set(jnp.array([cfg_off.world_size / 2, cfg_off.world_size / 2]))
    pos = pos.at[1].set(jnp.array([cfg_off.world_size / 2, cfg_off.world_size / 2]))
    state = state._replace(pos=pos, diet=jnp.full((cfg_off.n_max,), 0.5))
    state, _ms = step_fn(state, key)

    r = cfg_off.retina_sectors
    peer_off = 5 * r + 3 + 4 * cfg_off.memory_slots
    li = np.asarray(state.last_input[0])
    assert float(np.max(li[peer_off:peer_off + r])) < 1e-6, \
        "peer channel must be forced to zero when peer_channel_enabled=False"


def test_los_occlusion_default_is_truly_off():
    """`los_occlusion_enabled` defaults to False (docs/three_d.md S5.1), the
    same convention as `trample_impact`: the mechanism doesn't exist unless an
    ablation arm explicitly turns it on. Unlike the trample tests, there is no
    always-on field to show "differs but doesn't matter" -- when the flag is
    off, `sensors.sense` must skip the whole line-of-sight block at trace time
    (see the `if cfg.los_occlusion_enabled:` guard), so varying `los_samples`/
    `los_margin` while leaving the flag off must produce a bit-identical run,
    not merely a numerically close one.
    """
    from underworld import spatial, sensors

    cfg_a = tiny_cfg(los_samples=3, los_margin=0.01)
    cfg_b = tiny_cfg(los_samples=8, los_margin=5.0)
    assert cfg_a.los_occlusion_enabled is False
    assert cfg_b.los_occlusion_enabled is False

    state_a, ms_a = run(cfg_a, 300)
    state_b, ms_b = run(cfg_b, 300)
    assert bool(jnp.array_equal(state_a.alive, state_b.alive))
    assert bool(jnp.array_equal(ms_a.population, ms_b.population))
    assert bool(jnp.array_equal(state_a.pos, state_b.pos))
    assert bool(jnp.array_equal(state_a.energy, state_b.energy))
    assert bool(jnp.array_equal(state_a.last_input, state_b.last_input))

    # Also check directly against sensors.sense with occlusion physically
    # present in the terrain: with the flag off it must still be ignored.
    cfg_on = tiny_cfg(n_max=8, n_init=2, world_size=128.0, grid=32,
                       sense_grid=3, vision_radius=40.0, forest_occlusion=0.0,
                       los_occlusion_enabled=True)
    cfg_off = tiny_cfg(n_max=8, n_init=2, world_size=128.0, grid=32,
                        sense_grid=3, vision_radius=40.0, forest_occlusion=0.0,
                        los_occlusion_enabled=False)
    key = jax.random.PRNGKey(0)
    terrain = terrain_mod.build(cfg_on)
    state = init_state(cfg_on, key, terrain)
    pos = state.pos.at[0].set(jnp.array([20.0, 64.0]))
    pos = pos.at[1].set(jnp.array([50.0, 64.0]))
    state = state._replace(pos=pos, diet=jnp.full((cfg_on.n_max,), 0.5))

    g = cfg_on.grid
    height = jnp.zeros((g, g)).at[16, 7:10].set(5.0).reshape(-1)
    mountain = terrain._replace(height=height)

    table = spatial.build_table(state, cfg_on)
    nbr = spatial.gather_neighbors(state, table, cfg_on)
    delta, dist, valid = spatial.geometry(state, nbr, cfg_on)

    inputs_off = sensors.sense(state, nbr, delta, dist, valid, mountain, cfg_off)
    r = cfg_off.retina_sectors
    peer_off = 5 * r + 3 + 4 * cfg_off.memory_slots
    li_off = np.asarray(inputs_off[0])
    assert float(np.max(li_off[peer_off:peer_off + r])) > 0.1, \
        "a mountain in the terrain must have zero effect when the flag is off"


def test_los_occlusion_blocks_view_through_mountain():
    """docs/three_d.md S5.1: with `los_occlusion_enabled=True`, a candidate
    otherwise inside vision range must become invisible if a tall ridge sits
    on the line between it and the observer -- checked directly against
    `sensors.sense` output (same technique as
    `test_peer_channel_reveals_similar_diet_neighbours`), not end-to-end.
    """
    from underworld import spatial, sensors

    # sense_grid must satisfy cell_size >= vision_radius (world_size / sense_grid
    # >= vision_radius) or the 3x3 neighbour block can miss a candidate this far
    # away -- see the config comment on `sense_grid`.
    cfg = tiny_cfg(n_max=8, n_init=2, world_size=128.0, grid=32, sense_grid=3,
                   vision_radius=40.0, forest_occlusion=0.0,
                   los_occlusion_enabled=True)
    key = jax.random.PRNGKey(0)
    terrain = terrain_mod.build(cfg)
    state = init_state(cfg, key, terrain)

    # Observer and candidate 30 world units apart (< vision_radius=40, so
    # visible on flat ground), same y so the ridge sits squarely between them.
    pos = state.pos.at[0].set(jnp.array([20.0, 64.0]))
    pos = pos.at[1].set(jnp.array([50.0, 64.0]))
    state = state._replace(pos=pos, diet=jnp.full((cfg.n_max,), 0.5))

    r = cfg.retina_sectors
    peer_off = 5 * r + 3 + 4 * cfg.memory_slots

    table = spatial.build_table(state, cfg)
    nbr = spatial.gather_neighbors(state, table, cfg)
    delta, dist, valid = spatial.geometry(state, nbr, cfg)
    # Both endpoint cells (x=20 -> cell 5, x=50 -> cell 12) are flat (height 0);
    # a tall, narrow ridge sits at cells x in [7, 9] on the same row -- squarely
    # between the two agents and well clear of both endpoints.
    g = cfg.grid
    mountain_height = jnp.zeros((g, g)).at[16, 7:10].set(5.0).reshape(-1)
    mountain = terrain._replace(height=mountain_height)

    inputs = sensors.sense(state, nbr, delta, dist, valid, mountain, cfg)
    li = np.asarray(inputs[0])
    assert float(np.max(li[peer_off:peer_off + r])) < 1e-6, \
        "candidate behind the ridge must be invisible (peer channel forced to 0)"

    # Same positions, same distance, but flat terrain: the candidate must
    # remain visible -- guards against an occlusion criterion so strict it
    # blocks everyone regardless of terrain.
    flat = terrain._replace(height=jnp.zeros_like(terrain.height))
    inputs_flat = sensors.sense(state, nbr, delta, dist, valid, flat, cfg)
    li_flat = np.asarray(inputs_flat[0])
    assert float(np.max(li_flat[peer_off:peer_off + r])) > 0.1, \
        "same distance, no mountain in the way -- must not be falsely occluded"


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


def test_death_causes_partition_the_toll():
    """The four death counts must be a genuine partition of the deaths.

    They are mutually exclusive by construction, so the real risk is the other
    direction: a death that satisfies none of the four (and so vanishes) or an
    off-by-one against the actual drop in `alive`. This checks against the truth
    -- the number of alive bits `cull` clears -- rather than against itself.
    """
    cfg = tiny_cfg()
    state, key, _step, _scan, terrain = new_world(cfg)

    # Drive a step by hand so pre-cull state and the cull result are both in
    # scope. Half the population is starved and a quarter dehydrated outright so
    # the toll is large enough to be worth counting.
    n = cfg.n_max
    energy = jnp.where(jnp.arange(n) % 2 == 0, -1.0, state.energy)
    water = jnp.where(jnp.arange(n) % 4 == 1, -1.0, state.water)
    # Some of the starved were bitten hard enough this step that the bite is
    # what did it; others were not bitten at all.
    last_damage = jnp.where(jnp.arange(n) % 6 == 0, 5.0, 0.0)
    water_damage = jnp.where(jnp.arange(n) % 12 == 1, 5.0, 0.0)
    pre = state._replace(energy=energy, water=water, last_damage=last_damage)

    post, deaths = reproduction.cull(pre, water_damage, cfg)

    n_died = int(jnp.sum(pre.alive) - jnp.sum(post.alive))
    total = int(deaths.predation + deaths.starvation + deaths.thirst
                + deaths.senescence)
    assert total == n_died, f"{total} counted vs {n_died} actually culled"
    assert n_died > 0, "test set up no deaths at all"
    # Age sums must be attributed to the same partition, or mean-age-at-death
    # is computed against the wrong denominator.
    age_total = float(deaths.age_predation + deaths.age_starvation
                      + deaths.age_thirst + deaths.age_senescence)
    expected = float(jnp.sum(pre.age * (pre.alive & ~post.alive)))
    assert abs(age_total - expected) < 1e-3, f"{age_total} vs {expected}"
    # The counterfactual arm must actually fire, or the predation count is
    # vacuously correct and this test proves nothing about it.
    assert int(deaths.predation) > 0
    # The dead never come back to life.
    assert bool(jnp.all(post.alive <= pre.alive))


def test_water_deficit_buffer_defaults_to_old_instant_death():
    """cfg.water_deficit_buffer=0.0 (the default) must reproduce the exact old
    `water <= 0.0` rule -- docs/water_fix_buffer.md's whole golden-band argument
    rests on this being a true no-op at the default value."""
    cfg = tiny_cfg()  # water_deficit_buffer=0.0 by default
    state, key, _step, _scan, terrain = new_world(cfg)
    n = cfg.n_max
    water = jnp.where(jnp.arange(n) % 3 == 0, 0.0, 1.0)          # exactly zero
    water = jnp.where(jnp.arange(n) % 3 == 1, -0.5, water)       # negative
    pre = state._replace(water=water)
    _post, deaths = reproduction.cull(pre, jnp.zeros(n), cfg)
    exactly_zero = jnp.sum(pre.alive & (jnp.arange(n) % 3 == 0))
    negative = jnp.sum(pre.alive & (jnp.arange(n) % 3 == 1))
    # Both "exactly zero" and "negative" must die when the buffer is 0 -- only
    # strictly positive water survives, same as the pre-existing rule.
    assert int(deaths.thirst) >= int(exactly_zero) + int(negative)


def test_water_deficit_buffer_delays_death_within_tolerance():
    """With a positive buffer, water may run negative down to -buffer without
    counting as dehydration; past -buffer it still kills, same as before. This
    is the mechanism docs/water_fix_buffer.md proposes: a mild sign-flip from
    real mammals' tolerance for a double-digit-percent water deficit."""
    cfg = tiny_cfg(water_deficit_buffer=2.0)
    state, key, _step, _scan, terrain = new_world(cfg)
    n = cfg.n_max
    # Three bands: comfortably positive, in the tolerated deficit, past it.
    water = jnp.where(jnp.arange(n) % 3 == 0, 1.0, -1.0)         # ok / in-buffer
    water = jnp.where(jnp.arange(n) % 3 == 2, -3.0, water)       # past the buffer
    pre = state._replace(water=water)
    post, deaths = reproduction.cull(pre, jnp.zeros(n), cfg)

    in_buffer = pre.alive & (jnp.arange(n) % 3 == 1)   # water == -1.0, buffer == 2.0
    past_buffer = pre.alive & (jnp.arange(n) % 3 == 2)  # water == -3.0
    # In-buffer agents must survive this cull...
    assert bool(jnp.all(post.alive[in_buffer] == pre.alive[in_buffer]))
    # ...but agents past the buffer must still die.
    assert bool(jnp.all(~post.alive[past_buffer]))
    assert int(deaths.thirst) >= int(jnp.sum(past_buffer))
    # Living water may now sit below zero (down to -buffer) without being dead --
    # the invariant relaxes from "living implies water > 0" to "living implies
    # water > -buffer", which is the whole point of the mechanism.
    living_water = post.water[post.alive]
    assert bool(jnp.all(living_water > -cfg.water_deficit_buffer - 1e-4))


def test_reproduce_does_not_conjure_water_from_a_deficit_parent():
    """A parent with negative water (possible only when water_deficit_buffer > 0
    -- see docs/water_fix_buffer.md) must not (a) gain water by "investing" a
    negative fraction into a child, or (b) hand the child negative starting
    water. Both would be free water conjured from a negative number, on top of
    a child born already dehydrated -- neither is defensible."""
    cfg = tiny_cfg(n_max=64, n_init=8, water_deficit_buffer=3.0)
    key = jax.random.PRNGKey(0)
    terrain = terrain_mod.build(cfg)
    state = init_state(cfg, key, terrain)

    genome = state.genome.at[0, cfg.invest_index].set(4.0)  # near invest_max
    alive = jnp.arange(cfg.n_max) < 1
    energy = jnp.where(alive, cfg.repro_threshold + 5.0, 0.5)
    water = jnp.where(alive, -1.0, 0.1)   # parent in deficit but still alive
    state = state._replace(alive=alive, genome=genome, energy=energy, water=water)

    child = reproduction.reproduce(state, jax.random.PRNGKey(1), cfg)
    born = np.asarray(child.alive & ~state.alive)
    assert born.sum() == 1
    i = int(np.flatnonzero(born)[0])
    assert float(child.water[i]) >= 0.0, "child must not start already dehydrated"
    # The parent's own water must not have gone *up* from "investing" a
    # negative fraction of a negative balance.
    assert float(child.water[0]) <= float(state.water[0]) + 1e-4


# ---------------------------------------------------------------------------
# Coverage-gap tests (test_coverage_audit.md). Everything below was added to
# close holes the existing suite left open: the red-queen ablation flags'
# no-op equivalence, the spatial neighbour index's silent overflow/dead-agent
# dropping, the reproduction permutation-scatter's conservation invariants,
# the memory partition boundary, predation target selection, and the trait
# genome layout. Unit-level and deterministic (no scatter-add reordering), so
# they assert exact equality where the whole-sim tests can only assert bands.
# ---------------------------------------------------------------------------


def _two_agent_predation_state(cfg, dist=5.0, prey_escape_gene=0.0):
    """A carnivore (agent 0) and a herbivore prey (agent 1) `dist` apart, both
    at neutral attack reach (attack gene 0 -> 6.0). Shared setup for the
    predation-reach tests below."""
    from underworld.state import diet_of
    terrain = terrain_mod.build(cfg)
    state = init_state(cfg, jax.random.PRNGKey(0), terrain)
    genome = state.genome.at[:, cfg.attack_index].set(0.0)      # reach 6.0
    genome = genome.at[0, cfg.diet_index].set(6.0)             # carnivore
    genome = genome.at[1, cfg.diet_index].set(-6.0)            # herbivore prey
    genome = genome.at[1, cfg.escape_index].set(prey_escape_gene)
    state = state._replace(
        genome=genome, diet=diet_of(genome, cfg),
        energy=jnp.array([10.0, 10.0]), water=jnp.array([10.0, 10.0]),
        alive=jnp.array([True, True]))
    nbr = jnp.array([[1], [0]])
    d = jnp.array([[dist], [dist]])
    valid = jnp.array([[True], [True]])
    return state, nbr, d, valid


def test_attack_flag_off_reproduces_neutral_gene_predation():
    """`attack_range_heritable`/`prey_escape_enabled` are compile-time ablation
    flags. Their whole contract is that a *neutral* genome under the flags-on
    world (attack gene 0 -> reach 6.0, escape gene 0 -> 0) is bit-for-bit the
    flags-off world (fixed `cfg.attack_range`, no escape term). If it were not,
    every ablation arm would silently be measuring a different baseline, not the
    same world with the mechanism switched out. Checked directly on `predation`,
    whose branch on the flags is exactly what this guards."""
    cfg_on = tiny_cfg(n_max=2, n_init=2)
    cfg_off = tiny_cfg(n_max=2, n_init=2, attack_range_heritable=False,
                       prey_escape_enabled=False)
    # Genome-compatible: the flags never touch trait_dim/genome_size.
    assert cfg_on.genome_size == cfg_off.genome_size
    state, nbr, d, valid = _two_agent_predation_state(cfg_on, dist=5.0)

    out_on = dynamics.predation(state, nbr, d, valid, cfg_on)
    out_off = dynamics.predation(state, nbr, d, valid, cfg_off)
    for a, b in zip(out_on, out_off):
        assert bool(jnp.allclose(a, b, atol=1e-6)), "ablation arm moved the baseline"
    # ...and the bite must actually have landed, or the equivalence is vacuous.
    assert float(out_on[1][0]) > 0.0, "neutral carnivore should bite prey at dist 5 < 6"


def test_metabolize_flags_off_ignore_reach_and_escape_args():
    """With both red-queen flags off, `metabolize` must levy zero attack/escape
    tax however extreme the reach/escape passed in -- the off-branch compiles the
    tax away. This is the metabolize half of the ablation contract, complementing
    the predation half above."""
    cfg = tiny_cfg(attack_range_heritable=False, prey_escape_enabled=False)
    e0 = jnp.array([10.0, 10.0])
    thrust = jnp.array([0.5, 0.5])
    diet = jnp.array([1.0, 0.0])
    climb = jnp.array([0.0, 0.0])
    alive = jnp.array([True, True])
    size = jnp.array([1.0, 1.0])

    base = dynamics.metabolize(e0, thrust, diet, climb, alive, cfg, size)
    taxed = dynamics.metabolize(e0, thrust, diet, climb, alive, cfg, size,
                                jnp.array([1e3, 1e3]), jnp.array([1e3, 1e3]))
    assert bool(jnp.allclose(base, taxed, atol=1e-6)), \
        "flags-off metabolize must ignore the reach/escape arguments entirely"


def test_prey_escape_effective_reach_is_attack_minus_escape():
    """The red-queen arithmetic itself: a bite lands iff `dist < attack - escape`.
    With the attacker at neutral reach (6.0) and a prey carrying a known escape
    gene, the crossover distance must sit exactly at `6.0 - escape_of(prey)` --
    hit just inside it, miss just outside. `test_prey_escape_shrinks_effective_
    attack_reach` only checks the two extremes; this pins the boundary to the
    subtraction."""
    cfg = tiny_cfg(n_max=2, n_init=2)
    esc = float(escape_of(
        jnp.zeros((1, cfg.genome_size)).at[:, cfg.escape_index].set(1.0), cfg)[0])
    assert esc > 0.5, "test needs a prey escape big enough to move the boundary"
    effective = cfg.attack_range - esc

    st_in, nbr, _d, valid = _two_agent_predation_state(
        cfg, dist=effective - 0.15, prey_escape_gene=1.0)
    d_in = jnp.array([[effective - 0.15], [effective - 0.15]])
    _, meat_in, _, _, _, _, _ = dynamics.predation(st_in, nbr, d_in, valid, cfg)
    assert float(meat_in[0]) > 0.0, "bite just inside the effective reach must land"

    d_out = jnp.array([[effective + 0.15], [effective + 0.15]])
    _, meat_out, _, _, _, _, _ = dynamics.predation(st_in, nbr, d_out, valid, cfg)
    assert float(meat_out[0]) == 0.0, "bite just outside the effective reach must miss"


def test_predation_hits_nearest_eligible_prey_only():
    """Each attacker bites the *nearest eligible* prey among its neighbours, not
    all of them (docs: `argmin` over eligible distances). A carnivore with two
    herbivore neighbours inside reach must damage only the closer one."""
    from underworld.state import diet_of
    cfg = tiny_cfg(n_max=3, n_init=3)
    terrain = terrain_mod.build(cfg)
    state = init_state(cfg, jax.random.PRNGKey(0), terrain)
    genome = state.genome.at[:, cfg.attack_index].set(0.0)     # reach 6.0
    genome = genome.at[0, cfg.diet_index].set(6.0)            # carnivore
    genome = genome.at[1, cfg.diet_index].set(-6.0)           # herbivore
    genome = genome.at[2, cfg.diet_index].set(-6.0)           # herbivore
    state = state._replace(
        genome=genome, diet=diet_of(genome, cfg),
        energy=jnp.full(3, 10.0), water=jnp.full(3, 10.0),
        alive=jnp.ones(3, dtype=bool))
    # Agent 0 sees prey 1 at dist 3 and prey 2 at dist 5 (both < reach 6).
    nbr = jnp.array([[1, 2], [0, 0], [0, 0]])
    dist = jnp.array([[3.0, 5.0], [3.0, 3.0], [5.0, 5.0]])
    valid = jnp.array([[True, True], [True, False], [True, False]])

    _e, meat, damage, _w, _wg, _wd, _v = dynamics.predation(state, nbr, dist, valid, cfg)
    assert float(damage[1]) > 0.0, "the nearer prey must be bitten"
    assert float(damage[2]) == 0.0, "the farther prey must be untouched (only nearest)"
    assert float(meat[0]) > 0.0


def test_predation_respects_diet_delta_threshold():
    """A neighbour is prey only if it is *strictly more* than `diet_delta` more
    herbivorous. A neighbour whose diet gap sits just under the threshold is not
    eligible and takes no damage -- the guard that stops near-identical diets
    (conspecifics) from eating each other."""
    cfg = tiny_cfg(n_max=2, n_init=2)
    terrain = terrain_mod.build(cfg)
    state = init_state(cfg, jax.random.PRNGKey(0), terrain)
    # Diets differ by less than diet_delta: agent 0 is the (weak) attacker.
    diet = jnp.array([0.5, 0.5 - cfg.diet_delta * 0.5])
    state = state._replace(
        diet=diet, energy=jnp.array([10.0, 10.0]), water=jnp.array([10.0, 10.0]),
        genome=state.genome.at[:, cfg.attack_index].set(0.0),
        alive=jnp.array([True, True]))
    nbr = jnp.array([[1], [0]])
    dist = jnp.array([[3.0], [3.0]])
    valid = jnp.array([[True], [True]])
    _e, meat, damage, _w, _wg, _wd, _v = dynamics.predation(state, nbr, dist, valid, cfg)
    assert float(damage[1]) == 0.0, "sub-threshold diet gap must not be edible"
    assert float(meat[0]) == 0.0


def test_neighbor_table_drops_overflow_beyond_k():
    """A cell holding more than `k_neighbors` agents keeps only K of them; the
    overflow lands in the dump column that `build_table` slices off, so those
    agents become invisible to both vision and predation (CLAUDE.md: 'overflow
    beyond k_neighbors ... silently dropped'). Regression guard: pile K+4 agents
    into one cell and assert exactly K distinct indices survive in the table."""
    from underworld import spatial
    K = 8
    cfg = tiny_cfg(n_max=64, n_init=64, sense_grid=4, k_neighbors=K)
    state = init_state(cfg, jax.random.PRNGKey(0), terrain_mod.build(cfg))
    n_here = K + 4
    # First n_here agents share one exact position (one cell); the rest are dead.
    spot = jnp.array([10.0, 10.0])
    pos = jnp.where((jnp.arange(cfg.n_max) < n_here)[:, None], spot, state.pos)
    state = state._replace(pos=pos, alive=jnp.arange(cfg.n_max) < n_here)

    table = np.asarray(spatial.build_table(state, cfg))
    survivors = set(int(v) for v in np.unique(table) if v >= 0)
    assert len(survivors) == K, f"expected {K} survivors, got {len(survivors)}"
    assert survivors == set(range(K)), "the first K by cell-rank must be the ones kept"


def test_neighbor_table_excludes_the_dead():
    """Dead agents are routed to a dump *row* (`n_sense_cells`) that `build_table`
    slices off, so a corpse never surfaces as anyone's neighbour -- the other half
    of the silent-drop contract from `test_neighbor_table_drops_overflow_beyond_k`."""
    from underworld import spatial
    cfg = tiny_cfg(n_max=32, n_init=32, sense_grid=4, k_neighbors=8)
    state = init_state(cfg, jax.random.PRNGKey(0), terrain_mod.build(cfg))
    # Only agents 0 and 1 alive, co-located; everyone else dead but positioned
    # on top of them (so it is aliveness, not distance, that must exclude them).
    spot = jnp.array([10.0, 10.0])
    pos = jnp.tile(spot, (cfg.n_max, 1))
    state = state._replace(pos=pos, alive=jnp.arange(cfg.n_max) < 2)

    table = np.asarray(spatial.build_table(state, cfg))
    survivors = set(int(v) for v in np.unique(table) if v >= 0)
    assert survivors == {0, 1}, f"dead agents leaked into the table: {survivors}"


def test_reproduce_conserves_energy_and_writes_each_slot_once():
    """The permutation-scatter's core invariants (CLAUDE.md 'writes each index
    exactly once'):

      * births equal `min(wanters, free)`; no living agent is ever culled by
        reproduce (it only ever adds alive bits);
      * newborns land only in previously-free slots;
      * living non-parents are byte-for-byte untouched (the no-op write-back);
      * total energy is *conserved* -- a parent pays exactly what its child
        receives, so the sum over the whole array is unchanged.
    """
    cfg = tiny_cfg(n_max=64, n_init=8)
    state = init_state(cfg, jax.random.PRNGKey(0), terrain_mod.build(cfg))
    # Agents 0..4 are rich breeders; 5..7 alive but too poor to want; rest dead.
    idx = jnp.arange(cfg.n_max)
    energy = jnp.where(idx < 5, cfg.repro_threshold + 5.0,
                       jnp.where(idx < 8, 1.0, 0.0))
    state = state._replace(alive=idx < 8, energy=energy)
    want = int(jnp.sum(state.alive & (state.energy > cfg.repro_threshold)))
    assert want == 5

    e_before = float(jnp.sum(state.energy))
    child = reproduction.reproduce(state, jax.random.PRNGKey(1), cfg)

    born = np.asarray(child.alive & ~state.alive)
    assert int(born.sum()) == 5, "births must equal min(wanters, free)"
    assert bool(jnp.all(child.alive[:8])), "reproduce must never cull a living agent"
    assert int(jnp.sum(child.alive)) == 13
    assert np.all(np.flatnonzero(born) >= 8), "newborns must occupy only free slots"
    # Living non-parents (5,6,7) must be untouched by the scatter.
    assert np.allclose(np.asarray(child.energy[5:8]), 1.0)
    # Energy is conserved: parents' investment == childrens' endowment, exactly.
    assert float(jnp.sum(child.energy)) == pytest.approx(e_before, rel=1e-5)


def test_reproduce_bounded_by_free_slots_not_wanters():
    """When would-be parents outnumber free slots, only `free`-many births
    happen and not one living agent is overwritten -- the `n_birth = min(...)`
    clamp that keeps the fixed-shape array from overflowing."""
    cfg = tiny_cfg(n_max=8, n_init=6)
    state = init_state(cfg, jax.random.PRNGKey(0), terrain_mod.build(cfg))
    # 6 alive, all rich (all want); only 2 free slots.
    state = state._replace(
        alive=jnp.arange(8) < 6,
        energy=jnp.where(jnp.arange(8) < 6, cfg.repro_threshold + 5.0, 0.0))
    child = reproduction.reproduce(state, jax.random.PRNGKey(1), cfg)
    born = int(jnp.sum(child.alive & ~state.alive))
    assert born == 2, f"only 2 free slots -> 2 births, got {born}"
    assert bool(jnp.all(child.alive[:6])), "no living parent may be overwritten"
    assert int(jnp.sum(child.alive)) == 8


def test_memory_write_respects_partition_boundary():
    """Slots are partitioned by position: `[0, water_slots)` water, the rest
    fruit. A fruit write must never disturb a water slot, and vice versa --
    otherwise a fruit sighting would clobber a remembered river. Complements
    `test_memory_write_replaces_exactly_one_slot`, which only exercised the
    water partition."""
    cfg = tiny_cfg()
    n = 6
    w, k = cfg.memory_water_slots, cfg.memory_slots
    assert 0 < w < k, "test assumes both partitions are non-empty"
    # Every slot occupied; the last fruit slot is the weakest fruit slot.
    mem = jnp.tile(jnp.array([3.0, 4.0, 0.9]), (n, k, 1))
    mem = mem.at[:, k - 1, 2].set(0.1)
    should = jnp.arange(n) < 3

    out = memory.write(mem, w, k, should, cfg)          # write into the FRUIT range
    changed = np.asarray(jnp.any(out != mem, axis=2))   # [n, k]
    # Water slots [0, w) untouched for absolutely everyone.
    assert not changed[:, :w].any(), "a fruit write leaked into the water partition"
    # Writers changed exactly one fruit slot (the weakest, k-1); others none.
    assert np.all(changed[:3, w:].sum(axis=1) == 1)
    assert np.all(changed[:3, k - 1]), "the weakest fruit slot should be replaced"
    assert not changed[3:].any(), "non-writers must be untouched"


def test_memory_encode_bearing_is_egocentric():
    """`encode` reports each slot's bearing *relative to the agent's heading*, the
    same egocentric convention the retina uses. A slot lying straight ahead of an
    agent -- whatever its absolute compass direction -- must encode as sin~0,
    cos~1; one directly behind as sin~0, cos~-1."""
    cfg = tiny_cfg()
    k = cfg.memory_slots
    heading = jnp.array([0.5, 2.3])                     # two arbitrary headings
    # Slot 0 points exactly along each agent's own heading (dead ahead).
    ahead = jnp.stack([jnp.cos(heading), jnp.sin(heading)], axis=1) * 20.0
    mem = jnp.zeros((2, k, 3))
    mem = mem.at[:, 0, :2].set(ahead).at[:, 0, 2].set(1.0)
    feats = np.asarray(memory.encode(mem, heading, cfg))  # [2, 4k]
    # slot 0 layout: [sin(bearing), cos(bearing), tanh(dist), strength]
    for a in range(2):
        assert feats[a, 0] == pytest.approx(0.0, abs=1e-5), "dead-ahead sin ~ 0"
        assert feats[a, 1] == pytest.approx(1.0, abs=1e-5), "dead-ahead cos ~ 1"


def test_ecology_regrow_clips_and_recovers():
    """`regrow` invariants isolated from the full sim: it clips a field above its
    per-cell capacity back down, a zero-capacity cell stays exactly zero (the
    0/0 baseline guard that the `fruit_max=0` ablation depends on), and a
    grazed-out cell with positive capacity recovers via the spontaneous
    baseline even though the logistic term is zero there."""
    from underworld import ecology
    cap = jnp.array([5.0, 0.0, 5.0])
    field = jnp.array([9.0, 0.0, 0.0])                  # over-cap, dead-cell, grazed-out
    out = np.asarray(ecology.regrow(field, cap, 0.06, 0.02, 5.0))
    assert np.all(np.isfinite(out))
    assert out[0] <= 5.0 + 1e-6, "a field above capacity must be clipped down"
    assert out[1] == 0.0, "a zero-capacity cell must stay exactly zero (0/0 guard)"
    assert out[2] > 0.0, "a grazed-out cell with capacity must recover via baseline"
    assert out[2] <= 5.0 + 1e-6


def test_trait_gene_indices_are_distinct_and_in_range():
    """The trait genes are appended after the brain block at fixed offsets; a
    collision (two traits sharing a column) or an out-of-range index would let one
    gene silently overwrite another with no test firing anywhere. Guards the genome
    layout the whole trait-evolution programme rests on."""
    cfg = Config()
    idxs = [cfg.diet_index, cfg.invest_index, cfg.size_index,
            cfg.attack_index, cfg.escape_index, cfg.armor_index, cfg.spike_index]
    assert len(set(idxs)) == len(idxs), "trait gene indices collide"
    assert min(idxs) == cfg.brain_params, "traits must start right after the brain block"
    assert max(idxs) < cfg.genome_size, "a trait index falls outside the genome"
    assert cfg.genome_size == cfg.brain_params + cfg.trait_dim
    assert len(idxs) == cfg.trait_dim, "trait_dim disagrees with the number of trait genes"
    # in_dim must match the documented channel formula (5 retina + 3 scalars +
    # 4/slot memory + 1 peer retina channel).
    assert cfg.in_dim == 6 * cfg.retina_sectors + 3 + 4 * cfg.memory_slots
