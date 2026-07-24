"""The world state: one big pytree of fixed-shape arrays.

Everything about every (potential) agent is a row in these `[n_max, ...]` arrays.
Living vs. dead is tracked purely by the boolean `alive` mask, so births and
deaths never resize an array -- they just flip mask bits and scatter into slots.
`WorldState` is a NamedTuple, which JAX automatically treats as a pytree.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import Config


class WorldState(NamedTuple):
    alive: jax.Array       # bool   [n_max]
    pos: jax.Array         # f32    [n_max, 2]   position in world units
    heading: jax.Array     # f32    [n_max]      facing angle (radians)
    vel: jax.Array         # f32    [n_max, 2]   last velocity (for render/metrics)
    energy: jax.Array      # f32    [n_max]
    water: jax.Array       # f32    [n_max]      hydration (separate from energy)
    age: jax.Array         # f32    [n_max]      steps lived
    genome: jax.Array      # f32    [n_max, genome_size]
    hue: jax.Array         # f32    [n_max]      lineage colour in [0, 1)
    diet: jax.Array        # f32    [n_max]      0 herbivore .. 1 carnivore (cached)
    generation: jax.Array  # f32    [n_max]      lineage depth
    last_food: jax.Array   # f32    [n_max]      plant energy gained last step
    last_meat: jax.Array   # f32    [n_max]      predation energy gained last step
    last_damage: jax.Array # f32    [n_max]      energy lost to predators last step
    last_drink: jax.Array  # f32    [n_max]      water gained last step (stream + meat)
    hidden: jax.Array      # f32    [n_max, hidden]     recurrent brain state (memory)
    last_input: jax.Array  # f32    [n_max, in_dim]     last retina input (inspector)
    last_output: jax.Array # f32    [n_max, out_dim]    last brain output (inspector)
    plant: jax.Array       # f32    [n_cells]    plant energy field (flattened grid)
    fruit: jax.Array       # f32    [n_cells]    fruit field: patchy, canopy-only
    memory: jax.Array      # f32 [n_max, slots, 3]  (dx, dy, strength) -- see memory.py
    trample: jax.Array     # f32    [n_cells]    recent foot traffic, in [0, 1] --
    #                                              passive niche construction (Stage 0,
    #                                              see docs/TODO.md priority 3). A
    #                                              per-*cell* field like plant/fruit,
    #                                              not per-agent, so reproduction.place()
    #                                              needs no change for it.
    fear: jax.Array        # f32    [n_cells]    landscape of fear: a lagged, decaying
    #                                              trace of where carnivores have lurked
    #                                              (docs/landscape_of_fear.md S3.2). Same
    #                                              per-cell shape and update idiom as
    #                                              trample; folded into the pred retina
    #                                              channel (sensors.sense) so prey can
    #                                              learn to avoid a predator's camping
    #                                              ground. Default off (fear_rate=0):
    #                                              stays identically zero, so the fold is
    #                                              a no-op -- same convention as trample.
    phase: jax.Array       # f32    scalar (0-d)  day-night clock in [0, 1): 0/1 = midnight,
    #                                              0.5 = midday (docs/day_night.md). Advanced
    #                                              by 1/day_length each step, wrapped mod 1;
    #                                              read a step later (step-start) to drive the
    #                                              retina-darkening and thirst-heat folds, the
    #                                              same "deposit-then-read-next-step" idiom as
    #                                              trample/fear. A *scalar* leaf: it changes no
    #                                              [n_max]/genome shape, so founder RNG is
    #                                              untouched and day_length=0 (never advanced)
    #                                              reproduces the pre-clock baseline bit-exact.


def diet_of(genome: jax.Array, cfg: Config) -> jax.Array:
    """Map the diet gene to [0, 1]: 0 = pure herbivore, 1 = pure carnivore."""
    return jax.nn.sigmoid(genome[:, cfg.diet_index])


def invest_of(genome: jax.Array, cfg: Config) -> jax.Array:
    """Fraction of a parent's energy and water handed to one offspring.

    The quantity/quality dial: a low value buys many cheap offspring, a high one
    buys few well-provisioned ones. It needs no cost term because it is pure
    allocation -- there is no "more is better" direction to run away in, which is
    what makes it the safest possible first evolvable trait.

    Not cached on `WorldState` like `diet` is: it is read once per birth and once
    per metrics pass, both of which already hold the genome, so caching it would
    add an [n_max] array to the scan carry for nothing.
    """
    return cfg.invest_min + cfg.invest_span * jax.nn.sigmoid(genome[:, cfg.invest_index])


def size_of(genome: jax.Array, cfg: Config) -> jax.Array:
    """Map the size gene to [size_min, size_min+size_span]; 1.0 at gene=0.

    Like `invest_of`, not cached on `WorldState`: it is only ever read as a
    per-agent scalar in `dynamics.drink/metabolize/thirst` (never broadcast
    across the neighbour axis the way `diet` is in sensing/predation), so
    recomputing it from the genome each time it's needed is cheap and avoids
    growing the `lax.scan` carry.
    """
    return cfg.size_min + cfg.size_span * jax.nn.sigmoid(genome[:, cfg.size_index])


def attack_range_of(genome: jax.Array, cfg: Config) -> jax.Array:
    """Map the attack gene to [attack_min, attack_min+attack_span]; 6.0 at gene=0.

    The predator's bite reach (docs/attack_range_redqueen.md). A gene of 0 sigmoids
    to 0.5 -> `cfg.attack_range` (6.0), the old fixed constant, so a fresh population
    starts at the previous behaviour. Read per-agent inside `dynamics.predation`
    (attacker reach) and `dynamics.metabolize` (its energy tax); like `size_of` it is
    recomputed from the genome rather than cached on `WorldState`.
    """
    return cfg.attack_min + cfg.attack_span * jax.nn.sigmoid(genome[:, cfg.attack_index])


def escape_of(genome: jax.Array, cfg: Config) -> jax.Array:
    """Map the escape gene to [0, escape_span/2]; 0 at gene=0.

    The prey's evasion (docs/attack_range_redqueen.md), subtracted from an attacker's
    reach to give the *effective* range against this prey. Neutral (gene 0) is exactly
    0 -- a fresh population has no evasion, so any escape is evolved, not seeded. Uses
    `sigmoid(gene) - 0.5` clipped at 0 (i.e. only the positive half of the logistic
    counts), keeping the trait one-sided: a gene can only ever *buy* evasion, never a
    negative one, and pays `escape_cost` in proportion to what it buys.
    """
    return cfg.escape_span * jnp.clip(jax.nn.sigmoid(genome[:, cfg.escape_index]) - 0.5,
                                      0.0, None)


def armor_of(genome: jax.Array, cfg: Config) -> jax.Array:
    """Map the armour gene to [0, armor_span/2]; 0 at gene=0.

    The prey's damage reduction (docs/trait_defense_catalog.md): the fraction of an
    incoming bite's energy damage negated. One-sided like `escape_of` -- gene 0 is
    exactly 0 (a fresh population has no armour), and the gene can only ever *buy*
    defence, paying `armor_cost` in proportion. Read per-prey inside
    `dynamics.predation` and taxed on the energy ledger in `dynamics.metabolize`
    (never thirst -- docs/trait_addition_feasibility.md §B.2).
    """
    return cfg.armor_span * jnp.clip(jax.nn.sigmoid(genome[:, cfg.armor_index]) - 0.5,
                                     0.0, None)


def spike_of(genome: jax.Array, cfg: Config) -> jax.Array:
    """Map the spike gene to [0, spike_span/2]; 0 at gene=0.

    The prey's retaliation (docs/trait_defense_catalog.md): energy reflected back onto
    an attacker in proportion to the damage the attacker's bite deals. One-sided like
    `escape_of`; gene 0 is exactly 0. Read per-prey inside `dynamics.predation`
    (reflected onto the biter) and taxed on the energy ledger in `dynamics.metabolize`.
    """
    return cfg.spike_span * jnp.clip(jax.nn.sigmoid(genome[:, cfg.spike_index]) - 0.5,
                                     0.0, None)


def init_state(cfg: Config, key: jax.Array, terrain) -> WorldState:
    k_pos, k_head, k_gen, k_hue, k_plant, k_carn, k_rej = jax.random.split(key, 7)
    n = cfg.n_max

    alive = jnp.arange(n) < cfg.n_init

    # Founders are placed by rejection sampling against terrain fertility, so
    # nobody starts stranded on bare rock or in open water with nothing to eat.
    # Fixed number of proposals keeps the shape static.
    def propose(i, acc):
        pos_i, best_i = acc
        k = jax.random.fold_in(k_rej, i)
        cand = jax.random.uniform(k, (n, 2), maxval=cfg.world_size)
        score = terrain.capacity[pos_to_cell(cand, cfg)]
        take = score > best_i
        return (jnp.where(take[:, None], cand, pos_i),
                jnp.where(take, score, best_i))

    pos0 = jax.random.uniform(k_pos, (n, 2), maxval=cfg.world_size)
    pos, _ = jax.lax.fori_loop(
        0, 6, propose, (pos0, terrain.capacity[pos_to_cell(pos0, cfg)])
    )
    heading = jax.random.uniform(k_head, (n,), maxval=2.0 * jnp.pi)
    vel = jnp.zeros((n, 2))
    energy = jnp.where(alive, cfg.energy_init, 0.0)
    water = jnp.where(alive, cfg.water_init, 0.0)
    age = jnp.zeros(n)

    genome = jax.random.normal(k_gen, (n, cfg.genome_size)) * cfg.genome_init_scale
    # Seed the diet gene *bimodally*: a herbivore cluster (sigmoid ~0.08) and a
    # carnivore cluster (sigmoid ~0.88), so two distinct types exist from step 0.
    # `cfg.diet_bimodal_init` is a compile-time flag (Config is baked into the
    # jit), so branching on it in plain Python is fine -- see docs/biology.md
    # §10.1: this is one of four layers that hold the split apart, and the
    # ablation arm needs founders to start from a single neutral cluster
    # instead, so any bimodality later is evolved rather than seeded.
    r = jax.random.uniform(k_carn, (n,))
    if cfg.diet_bimodal_init:
        diet_center = jnp.where(r < cfg.carnivore_init_frac, 2.0, -2.5)
    else:
        diet_center = jnp.zeros_like(r)
    diet_gene = diet_center + 0.3 * genome[:, cfg.diet_index]  # a little spread
    genome = genome.at[:, cfg.diet_index].set(diet_gene)

    hue = jax.random.uniform(k_hue, (n,))
    diet = diet_of(genome, cfg)
    zeros = jnp.zeros(n)

    # Patchy initial plant field, scaled by what each cell can actually support:
    # forest starts lush, plains moderate, rock and sea bare.
    plant = jax.random.uniform(
        k_plant, (cfg.n_cells,), minval=0.0, maxval=2.0 * cfg.plant_init
    )
    plant = jnp.minimum(plant * (terrain.capacity / cfg.plant_max), terrain.capacity)
    # Fruit starts at capacity: it regrows an order of magnitude slower than
    # grass, so seeding it low would just mean no fruit exists for thousands of
    # steps and the first measurements would be of a world without it.
    fruit = terrain.fruit_capacity

    return WorldState(
        alive=alive, pos=pos, heading=heading, vel=vel, energy=energy, water=water,
        age=age, genome=genome, hue=hue, diet=diet, generation=zeros,
        last_food=zeros, last_meat=zeros, last_damage=zeros, last_drink=zeros,
        hidden=jnp.zeros((n, cfg.hidden)),
        last_input=jnp.zeros((n, cfg.in_dim)),
        last_output=jnp.zeros((n, cfg.out_dim)),
        plant=plant,
        fruit=fruit,
        # Founders know nothing. Strength 0 reads as "empty slot" and is the
        # natural argmin target, so the first drink each agent takes fills a slot
        # rather than overwriting one.
        memory=jnp.zeros((n, cfg.memory_slots, 3)),
        # No one has walked anywhere yet.
        trample=jnp.zeros(cfg.n_cells),
        # No predator has lurked anywhere yet.
        fear=jnp.zeros(cfg.n_cells),
        # The clock starts at midnight (phase 0). Stays here forever when
        # day_length=0, so the diel folds are bit-exact no-ops.
        phase=jnp.zeros(()),
    )


def pos_to_cell(pos: jax.Array, cfg: Config, grid: int | None = None) -> jax.Array:
    """Map [N, 2] world positions to flat cell indices on a `grid`x`grid` torus
    (defaults to the plant grid). Cell index is iy*grid+ix.
    """
    g = cfg.grid if grid is None else grid
    ij = jnp.floor(pos / (cfg.world_size / g)).astype(jnp.int32) % g  # [N,2] (ix,iy)
    ix, iy = ij[:, 0], ij[:, 1]
    return iy * g + ix
