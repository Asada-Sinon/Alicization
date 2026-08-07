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
from .state import (WorldState, forage_pref_of, invest_of, mate_forage_of, pos_to_cell,
                    size_of)


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
                       key: jax.Array, forage: jax.Array | None = None,
                       mate_w: jax.Array | None = None) -> jax.Array:
    """For every agent, find another *wanting-to-reproduce* agent with a similar
    diet to serve as a second genetic parent -- assortative by diet so crossover
    mixes brain genes within a species rather than between herbivores and
    carnivores. Falls back to pairing an agent with itself (crossover becomes a
    no-op, i.e. the old asexual clone) when there's no one else to pair with this
    step (0 or an odd number of reproducers).

    `cfg.assortative_mating=False` is the ablation arm (docs/trait_evolution.md
    §10.1/§10.5 -- the material moved there when biology.md was split, and the old
    `docs/biology.md` pointer this docstring used to carry no longer resolves):
    wanters are ranked by an independent uniform draw instead of diet, so pairing
    is uniformly random among reproducers rather than diet-sorted. Dieckmann &
    Doebeli (1999) is why this one is tested separately from the other three
    diet-speciation switches -- theory says assortative mating *maintains* an
    evolved branch rather than merely seeding one, so it should be ablated only
    after checking whether a branch forms at all without the other three.

    `cfg.mate_forage_weight > 0` additionally sorts by the grass<->fruit gene, so
    the two foraging ecotypes preferentially breed within themselves
    (docs/multispecies_program.md §20). Three things about that branch:

    - The `w == 0` path is the *old code*, reached by a compile-time `if` because
      `Config` is a jit constant. That is load-bearing, not tidiness: the composite
      key `2*cls + ...` puts carnivores in [2,3), where float32's ULP is 2^-22
      against 2^-24 down in [0.5,1) -- two bits of precision lost, so carnivores
      whose diets differ by under ~2.4e-7 would *tie*, and `jnp.argsort` is stable
      (ties break by index). That silently changes who pairs with whom, ~1e-3 of
      steps, ~100 times over a 100k-step run. Order-equivalence is not enough here.
    - `2.0 * cls` does NOT make cross-class pairing impossible -- rank-adjacent
      pairing still crosses at the boundary when the herbivore count is odd (0 or 1
      pair, ~0.5/step), exactly as sorting by `diet` alone already does. What it
      buys is that a large `w` cannot shuffle carnivores *into* the herbivore run,
      which would be strictly worse than today.
    - `forage` is only read when the knob is on, so the default arm pays nothing.
    """
    n = cfg.n_max
    rank_key = diet if cfg.assortative_mating else jax.random.uniform(key, (n,))
    _mate_forage_on = cfg.mate_forage_heritable or cfg.mate_forage_weight > 0.0
    if _mate_forage_on and cfg.assortative_mating:
        # R19 (§22): `w` becomes PER-AGENT when the gene is on. Everything below is
        # unchanged -- broadcasting a [n] array where a scalar used to be is the whole
        # difference, so the `w == 0` agent still sorts by diet exactly as before.
        #
        # ⚠️ **§22.3 said "use the mean of the two parents"; that is not implementable
        # here and the deviation is recorded in §22.3b.** The rank key is what DECIDES
        # who pairs with whom, so it must be computable before any pairing exists --
        # a parental mean would need the pairing it is supposed to produce. Using each
        # agent's own `w` keeps the interaction two-sided anyway: whether A and B end
        # up rank-adjacent depends on BOTH keys, so a w=0 agent queueing by diet and a
        # w=0.5 agent queueing by forage_pref are exactly what drives them apart.
        w = mate_w if cfg.mate_forage_heritable else cfg.mate_forage_weight
        cls = (diet >= 0.5).astype(diet.dtype)     # 0 = herbivore, 1 = carnivore
        # **诊断对照（§22.4d）**：把排序轴换成每步重抽的随机数。它保留「按 `w` 的强度
        # 改变排队依据」这件事本身，但**切断该轴与任何可遗传量的关联**。
        # 用途：跑前探针发现 `forage_tradeoff=0` 的 H2 对照臂**也在涨**，而它按设计
        # 不该涨。若换成随机轴后涨幅消失 ⇒ 涨的是「w 高的个体都按同一条可遗传轴排队、
        # 因而互相靠近」这个 by-construction 的正反馈，与隔离的适应度收益无关。
        axis = (jax.random.uniform(key, (n,)) if cfg.mate_forage_random_axis else forage)
        if cfg.mate_forage_heritable:
            # **等方差混合（R19 必需，§22.4e）。** 直接写 `(1−w)·diet + w·axis` 有一个
            # by-construction 的假阳性：食草者内部 `diet` 只散布 **0.035**，而 `axis`
            # 散布 0.3–1.0，**于是 `w` 越大，这个体的 key 被甩得越远**。排序之后
            # `w≈0` 的个体全挤在中间、高 `w` 的个体落在两端——**两端只有其他高 `w` 个体，
            # 它们因此互相配对、后代继承高 `w`，形成与隔离收益无关的正反馈。**
            # 实测（§22.4d 的四臂探针，20k 步）：即使把 axis 换成**纯随机数**，
            # `mean gene` 仍涨 +0.267（MDE 才 0.176）；只有关掉同型交配才不涨（−0.065）。
            # 秩变换把两个轴都压成 [0,1] 上的均匀分布 ⇒ `w` 不再改变 key 的散布。
            # ⚠️ 光做秩变换**还不够**，实测会把假信号翻成反向的（−0.25）：
            # 两个**独立**轴线性混合时 `Var((1−w)X + wY) = ((1−w)² + w²)·Var`，
            # 它在 w=0 时是 1、w=0.5 时只有 0.5——**高 w 的个体反而挤在队列中间，
            # 于是换成低 w 的个体落在两端互相配对，把 w 压下去。**
            # 「线性混合两个独立轴」这个形式**本身就做不到方差对 w 恒定**，
            # 必须再除以 `sqrt((1−w)² + w²)`。除完之后 w=0 仍恰好是 `rank(diet)`、
            # w=1 仍恰好是 `rank(axis)`，语义不变；key 的上界 1/√0.5≈1.41 < 2，
            # 所以 `2·cls` 仍然分得开食草者与食肉者。
            def _rank01(x):
                return jnp.argsort(jnp.argsort(x)).astype(jnp.float32) / max(n - 1.0, 1.0)
            norm = jnp.sqrt(jnp.square(1.0 - w) + jnp.square(w))
            rank_key = 2.0 * cls + ((1.0 - w) * _rank01(diet) + w * _rank01(axis)) / norm
        else:
            # R17 的常量路径**原样不动**：所有个体共用一个 `w`，个体间没有散布差异，
            # 上面那个副产物按构造不存在，所以它的已发表结论不受影响。
            rank_key = 2.0 * cls + (1.0 - w) * diet + w * axis
    order = jnp.argsort(jnp.where(want, rank_key, jnp.inf))  # wanters first
    n_want = jnp.sum(want)
    swap = jnp.arange(n) ^ 1                              # pairs (0,1) (2,3) ...
    swap = jnp.where(swap < n_want, swap, jnp.arange(n))  # odd one out -> self
    partner_by_rank = order[swap]
    rank_of = jnp.argsort(order)                          # inverse permutation
    return partner_by_rank[rank_of]


def reproduce(state: WorldState, key: jax.Array, cfg: Config,
              crowd: jax.Array | None = None) -> WorldState:
    alive = state.alive
    # Density-dependent reproduction (docs/herbivore_overpopulation.md L6): a crowded
    # cell raises the energy bar to breed, so dense patches self-limit. `crowd` is the
    # per-cell agent count from step.py. Default off (penalty 0) compiles this away,
    # leaving `want` bit-exact the old pure-energy gate. Herbivores form the dense
    # crowds, so this throttles them far more than the sparse carnivores -- targeting
    # the density problem without touching water/plant carrying capacity.
    threshold = cfg.repro_threshold
    if cfg.density_repro_penalty > 0.0 and crowd is not None:
        crowding = jnp.clip(crowd[pos_to_cell(state.pos, cfg)] / cfg.density_repro_cap,
                            0.0, 1.0)
        threshold = cfg.repro_threshold * (1.0 + cfg.density_repro_penalty * crowding)
    want = alive & (state.energy > threshold)
    free = ~alive

    n_birth = jnp.minimum(jnp.sum(want), jnp.sum(free))
    k = jnp.arange(cfg.n_max)
    is_birth = k < n_birth                       # first n_birth (parent, slot) pairs

    parent_idx = jnp.argsort(~want)              # wanters first (stable)
    slot_idx = jnp.argsort(~free)                # free slots first (stable)

    k_gen, k_cross, k_pos, k_head, k_hue, k_mate = jax.random.split(key, 6)

    # --- build child values for every k (only the is_birth ones are used) ---
    _mate_forage_on = cfg.mate_forage_heritable or cfg.mate_forage_weight > 0.0
    forage = forage_pref_of(state.genome, cfg) if _mate_forage_on else None
    # R19: per-agent assortment strength. Only read when the gene is on, so the
    # default arm and R17's constant arm both pay nothing.
    mate_w = mate_forage_of(state.genome, cfg) if cfg.mate_forage_heritable else None
    mate_idx = _assortative_mate(want, state.diet, cfg, k_mate, forage, mate_w)[parent_idx]
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
    # Envenomation is not inherited -- a newborn starts clean, whatever debuff the
    # slot's previous occupant carried (docs/trait_defense_landing.md §7).
    venom = place(state.venom, zeros)
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
        last_drink=last_drink, venom=venom, hidden=hidden, last_input=last_input,
        last_output=last_output, memory=memory,
    )
