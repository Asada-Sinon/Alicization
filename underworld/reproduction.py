"""Death and birth on fixed-capacity arrays.

The whole trick: never resize. `cull` clears `alive` bits; `reproduce` matches
parents (who have enough energy) to freed slots and scatters children in. Both
the parent list and the slot list are argsort permutations of `[0, n_max)`, so
every `.at[idx].set(...)` writes each index exactly once -- non-births write the
slot's existing value (a no-op), which keeps everything static-shaped and jit
friendly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import Config
from .genome import crossover, mutate
from .state import WorldState, invest_of


def cull(state: WorldState, cfg: Config) -> WorldState:
    """Starvation, dehydration, and old age. Freed slots become available for
    births."""
    died = state.alive & (
        (state.energy <= 0.0) | (state.water <= 0.0) | (state.age > cfg.max_age)
    )
    return state._replace(alive=state.alive & (~died))


def _assortative_mate(want: jax.Array, diet: jax.Array, cfg: Config) -> jax.Array:
    """For every agent, find another *wanting-to-reproduce* agent with a similar
    diet to serve as a second genetic parent -- assortative by diet so crossover
    mixes brain genes within a species rather than between herbivores and
    carnivores. Falls back to pairing an agent with itself (crossover becomes a
    no-op, i.e. the old asexual clone) when there's no one else to pair with this
    step (0 or an odd number of reproducers).
    """
    n = cfg.n_max
    order = jnp.argsort(jnp.where(want, diet, jnp.inf))  # wanters first, by diet
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

    k_gen, k_cross, k_pos, k_head, k_hue = jax.random.split(key, 5)

    # --- build child values for every k (only the is_birth ones are used) ---
    mate_idx = _assortative_mate(want, state.diet, cfg)[parent_idx]
    crossed = crossover(state.genome[parent_idx], state.genome[mate_idx], k_cross, cfg)
    child_genome = mutate(crossed, k_gen, cfg)
    # How much to hand over is the parent's own gene, not a global constant.
    # Energy and water use the same fraction: provisioning is provisioning.
    invest_frac = invest_of(state.genome, cfg)[parent_idx]
    invest = state.energy[parent_idx] * invest_frac
    water_invest = state.water[parent_idx] * invest_frac
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
