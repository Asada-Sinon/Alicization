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
from underworld.state import init_state, invest_of, size_of


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
