"""Death and birth on fixed-capacity arrays.

The whole trick: never resize. `cull` clears `alive` bits; `reproduce` matches
parents (who have enough energy) to freed slots and scatters children in. Both
the parent list and the slot list are argsort permutations of `[0, n_max)`, so
every `.at[idx].set(...)` writes each index exactly once -- non-births write the
slot's existing value (a no-op), which keeps everything static-shaped and jit
friendly.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import Config
from .genome import crossover, mutate
from .state import WorldState, invest_of, size_of


class Deaths(NamedTuple):
    """Per-step death counts split by cause, plus the summed age of the dead in
    each class.

    Counts rather than fractions so a run can be summed and divided once at the
    end; per-step fractions would weight a step with two deaths the same as a
    step with two hundred. Age arrives as a *sum* for the same reason -- divide
    by the matching count after summing the run to get mean age at death, which
    is what separates "juveniles die before they learn the map" from "adults
    misjudge an excursion". Those two want different mechanisms.
    """
    predation: jax.Array
    starvation: jax.Array
    thirst: jax.Array
    senescence: jax.Array
    age_predation: jax.Array
    age_starvation: jax.Array
    age_thirst: jax.Array
    age_senescence: jax.Array


def cull(state: WorldState, water_damage: jax.Array, cfg: Config):
    """Starvation, dehydration, and old age. Freed slots become available for
    births. Returns `(state, Deaths)`.

    Nothing here kills an agent *directly by predation* -- a bitten agent dies of
    energy (or water) hitting zero like any other. So predation is attributed
    **counterfactually**: a death counts as predation when the damage taken this
    very step is what pushed the pool below zero, i.e. the agent would still have
    a positive pool without it. That deliberately excludes the slow case where
    repeated bites bled an agent down over many steps and metabolism finished the
    job -- those land in `starvation`, which makes this a *lower bound* on how
    much predation matters. A bite draws water as well as energy, hence
    `water_damage`, which unlike `last_damage` is not carried on the state.

    Causes are made mutually exclusive by priority (predation > starvation >
    thirst > old age) so the four counts sum to the death toll and can be read
    as a partition.

    `cfg.water_deficit_buffer` (docs/water_fix_buffer.md) lets `water` run
    negative down to `-water_deficit_buffer` before it counts as dehydration --
    real mammals tolerate a double-digit-percent water deficit before death, not
    a hard zero. Default 0.0 makes this identical to the old `water <= 0.0`
    test.
    """
    starved = state.energy <= 0.0
    parched = state.water <= -cfg.water_deficit_buffer
    aged = state.age > cfg.max_age
    died = state.alive & (starved | parched | aged)

    fatal_bite = (starved & (state.energy + state.last_damage > 0.0)) | \
                 (parched & (state.water + water_damage > -cfg.water_deficit_buffer))
    predation = died & fatal_bite
    starvation = died & starved & ~predation
    thirst = died & parched & ~predation & ~starvation
    senescence = died & ~predation & ~starvation & ~thirst

    deaths = Deaths(
        predation=jnp.sum(predation),
        starvation=jnp.sum(starvation),
        thirst=jnp.sum(thirst),
        senescence=jnp.sum(senescence),
        age_predation=jnp.sum(state.age * predation),
        age_starvation=jnp.sum(state.age * starvation),
        age_thirst=jnp.sum(state.age * thirst),
        age_senescence=jnp.sum(state.age * senescence),
    )
    return state._replace(alive=state.alive & (~died)), deaths


def _assortative_mate(want: jax.Array, diet: jax.Array, cfg: Config,
                       key: jax.Array) -> jax.Array:
    """For every agent, find another *wanting-to-reproduce* agent with a similar
    diet to serve as a second genetic parent -- assortative by diet so crossover
    mixes brain genes within a species rather than between herbivores and
    carnivores. Falls back to pairing an agent with itself (crossover becomes a
    no-op, i.e. the old asexual clone) when there's no one else to pair with this
    step (0 or an odd number of reproducers).

    `cfg.assortative_mating=False` is the ablation arm (docs/biology.md
    §10.1/§10.5): wanters are ranked by an independent uniform draw instead of
    diet, so pairing is uniformly random among reproducers rather than
    diet-sorted. Dieckmann & Doebeli (1999) is why this one is tested
    separately from the other three diet-speciation switches -- theory says
    assortative mating *maintains* an evolved branch rather than merely seeding
    one, so it should be ablated only after checking whether a branch forms at
    all without the other three.
    """
    n = cfg.n_max
    rank_key = diet if cfg.assortative_mating else jax.random.uniform(key, (n,))
    order = jnp.argsort(jnp.where(want, rank_key, jnp.inf))  # wanters first
    n_want = jnp.sum(want)
    swap = jnp.arange(n) ^ 1                              # pairs (0,1) (2,3) ...
    swap = jnp.where(swap < n_want, swap, jnp.arange(n))  # odd one out -> self
    partner_by_rank = order[swap]
    rank_of = jnp.argsort(order)                          # inverse permutation
    return partner_by_rank[rank_of]


def reproduce(state: WorldState, key: jax.Array, cfg: Config) -> WorldState:
    alive = state.alive
    want = alive & (state.energy > cfg.repro_threshold)
    free = ~alive

    n_birth = jnp.minimum(jnp.sum(want), jnp.sum(free))
    k = jnp.arange(cfg.n_max)
    is_birth = k < n_birth                       # first n_birth (parent, slot) pairs

    parent_idx = jnp.argsort(~want)              # wanters first (stable)
    slot_idx = jnp.argsort(~free)                # free slots first (stable)

    k_gen, k_cross, k_pos, k_head, k_hue, k_mate = jax.random.split(key, 6)

    # --- build child values for every k (only the is_birth ones are used) ---
    mate_idx = _assortative_mate(want, state.diet, cfg, k_mate)[parent_idx]
    crossed = crossover(state.genome[parent_idx], state.genome[mate_idx], k_cross, cfg)
    child_genome = mutate(crossed, k_gen, cfg)
    # How much to hand over is the parent's own gene, not a global constant.
    # Energy still follows that gene alone. Water can additionally be topped
    # up by a lactation floor (docs/water_fix_provisioning.md) that is NOT
    # part of invest_frac's own gene-bounded range -- it is a Config constant,
    # not a second gene, so it cannot be bred back down the way raising
    # invest_min itself was measured to be absorbed by evolution
    # (docs/water_system.md SS2.3/3.3, arm_B). Clipped to [0, 1] so a
    # misconfigured floor above 1.0 can never demand more water than the
    # parent has; at the default 0.0 this is exactly the old shared-fraction
    # behaviour (max(invest_frac, 0.0) == invest_frac).
    invest_frac = invest_of(state.genome, cfg)[parent_idx]
    invest = state.energy[parent_idx] * invest_frac
    water_frac = jnp.clip(jnp.maximum(invest_frac, cfg.water_lactation_floor_frac), 0.0, 1.0)
    # With `water_deficit_buffer > 0` a living parent's water can be negative
    # (in deficit but not yet dead). Clamp to 0 before taking a fraction of it,
    # or a deficit parent would hand a child *negative* starting water (born
    # already dehydrated) while the parent's own water balance would rise --
    # free water conjured from a negative number. At the default buffer=0,
    # `alive` already implies `water > 0` (cull runs first each step), so this
    # is a no-op there. The two knobs compose: the lactation floor sets the
    # fraction, this clamp guards the base it multiplies.
    water_invest = jnp.maximum(state.water[parent_idx], 0.0) * water_frac
    # A small-`size` child cannot hold more water than its own tank -- without
    # this, a large parent's absolute transfer could exceed a small-genotype
    # child's `water_max * size`, and the excess would simply vanish into
    # nowhere at the first `drink`-side clamp next step. Uses `child_genome`,
    # not `state.size` (which doesn't exist -- see `state.size_of`), since the
    # child's size is a property of its own, just-built genome.
    child_size = size_of(child_genome, cfg)
    water_invest = jnp.minimum(water_invest, cfg.water_max * child_size)
    offset = jax.random.uniform(
        k_pos, (cfg.n_max, 2), minval=-cfg.spawn_radius, maxval=cfg.spawn_radius
    )
    child_pos = jnp.mod(state.pos[parent_idx] + offset, cfg.world_size)
    child_heading = jax.random.uniform(k_head, (cfg.n_max,), maxval=2.0 * jnp.pi)
    hue_drift = jax.random.normal(k_hue, (cfg.n_max,)) * cfg.hue_drift
    child_hue = jnp.mod(state.hue[parent_idx] + hue_drift, 1.0)
    child_gen = state.generation[parent_idx] + 1.0

    # --- parents pay the energy/water they invest in the child ---
    energy = state.energy.at[parent_idx].add(jnp.where(is_birth, -invest, 0.0))
    water = state.water.at[parent_idx].add(jnp.where(is_birth, -water_invest, 0.0))

    # --- scatter children into freed slots (permutation write, mask by is_birth) ---
    def place(field, child_vals):
        keep = field[slot_idx]
        expand = is_birth.reshape((-1,) + (1,) * (field.ndim - 1))
        new_at_slot = jnp.where(expand, child_vals, keep)
        return field.at[slot_idx].set(new_at_slot)

    zeros = jnp.zeros(cfg.n_max)
    alive = place(alive, jnp.ones(cfg.n_max, dtype=bool))
    energy = place(energy, invest)
    water = place(water, water_invest)
    genome = place(state.genome, child_genome)
    pos = place(state.pos, child_pos)
    heading = place(state.heading, child_heading)
    hue = place(state.hue, child_hue)
    age = place(state.age, zeros)
    vel = place(state.vel, jnp.zeros((cfg.n_max, 2)))
    generation = place(state.generation, child_gen)
    last_food = place(state.last_food, zeros)
    last_meat = place(state.last_meat, zeros)
    last_damage = place(state.last_damage, zeros)
    last_drink = place(state.last_drink, zeros)
    hidden = place(state.hidden, jnp.zeros((cfg.n_max, cfg.hidden)))
    # Memory is NOT inherited -- newborns start with an empty map and have to
    # learn the world themselves. Genes are the heritable channel; memory is
    # acquired within a lifetime. An earlier version copied the parent's slots
    # at birth, which is Lamarckian; the argument against it stands on its own
    # and the burden was on the mechanism. The ablation was *underpowered*, not
    # null: n=6 paired, +0.020 mean inland_frac, SD 0.031 -> p=0.175 at 25%
    # power, but equivalence-bounded below 0.05 (TOST p=0.032). `place` handles
    # the [n_max, slots, 3] rank without modification -- `expand` is generic.
    memory = place(state.memory, jnp.zeros_like(state.memory))
    last_input = place(state.last_input, jnp.zeros((cfg.n_max, cfg.in_dim)))
    last_output = place(state.last_output, jnp.zeros((cfg.n_max, cfg.out_dim)))

    return state._replace(
        alive=alive, energy=energy, water=water, genome=genome, pos=pos,
        heading=heading, hue=hue, age=age, vel=vel, generation=generation,
        last_food=last_food, last_meat=last_meat, last_damage=last_damage,
        last_drink=last_drink, hidden=hidden, last_input=last_input,
        last_output=last_output, memory=memory,
    )
