"""Long-term spatial memory: where an agent has found water and fruit before.

The retina is short-sighted by design -- the water channel dies at
`vision_radius`, and over half the map lies beyond even that plus the sector
sample offset. An agent standing in the interior gets *identical* (all-zero)
water input whether the nearest river is 40 units north or 140 south. That is
not a sensing bug to be widened away; real animals cannot see past the horizon
either. What real animals have instead is memory, and that is what this supplies.

Two tiers, matching two well-studied mechanisms:

  * short term -- the recurrent hidden state in `brain.py`, already there. This
    is the working register: what just happened, where I was heading.
  * long term -- the slots here. Desert ants navigate by path integration,
    maintaining a continuously updated "home vector" of distance and direction
    back to a remembered point; elephant matriarchs recall water sources not
    visited in decades, and herds led by older matriarchs survive droughts that
    kill calves in herds led by younger ones. Crucially, the insect literature
    shows this does *not* require a cognitive map -- a decentralised set of
    vectors suffices, which is also what a fixed-shape jitted kernel can carry.

A slot is `(dx, dy, strength)`: a *relative* offset from the agent to the
remembered point, plus a confidence that decays. Relative rather than absolute is
the load-bearing choice: the torus never has to be reasoned about twice, since
each step subtracts the displacement and re-wraps to the shortest path.

Slots are partitioned by *position*, not by a type tag: `[0, water_slots)` hold
water, the rest hold fruit. The brain then reads a fixed meaning from each input
group instead of having to learn to decode a type bit, and each slot costs one
input less.

**Memory is not heritable.** Genes are the channel that crosses generations;
memory is acquired within one lifetime and dies with its holder. A newborn gets
empty slots and has to find water itself. An earlier version copied the parent's
slots at birth -- Lamarckian, and removed on that argument alone. The ablation
that tested it was *underpowered* (n=6 paired, +0.020 mean inland_frac, p=0.175
at 25% power), not a demonstrated null; equivalence bounds any effect below 0.05
(TOST p=0.032). See `reproduction.py` and `docs/conventions.md` §4.
Anything resembling a mother teaching a calf would have to be *learned*, by
juveniles following adults and drinking for themselves, which the write path here
already supports; it is not something to hand over at birth.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import Config


def advance(memory: jax.Array, disp: jax.Array, key: jax.Array,
            cfg: Config) -> jax.Array:
    """Carry the slots one step: subtract this step's displacement, decay
    confidence, and accumulate dead-reckoning error.

    Must run *after* movement and *before* any write, so that a slot written this
    step records an offset of ~0 from where the agent actually stands.

    The drift is deliberately mild. Random-walk error grows as sqrt(n), so over a
    200-step journey it accumulates to only a few world units -- about right for
    ant-grade path integration. Forgetting is the decay's job, not the drift's;
    cranking `memory_drift` up to force forgetting would instead just make the
    vectors noisy while they are still fresh.
    """
    v = memory[..., :2] - disp[:, None, :]
    step_len = jnp.linalg.norm(disp, axis=1)[:, None, None]
    v = v + jax.random.normal(key, v.shape) * cfg.memory_drift * step_len
    half = cfg.half_world
    v = (v + half) % cfg.world_size - half     # keep it the shortest-path vector
    s = memory[..., 2] * cfg.memory_decay
    return jnp.concatenate([v, s[..., None]], axis=2)


def write(memory: jax.Array, lo: int, hi: int, should_write: jax.Array,
          cfg: Config) -> jax.Array:
    """Record 'here' into the weakest slot of the partition `[lo, hi)`.

    Pure `where` rather than a scatter: `argmin` over a static slice picks one
    column, `one_hot` turns it into a mask, and the write is a select. Nothing
    here is a dynamic index or an atomic, so this stays jittable and -- unlike
    the per-cell scatter-adds elsewhere in the step -- adds no nondeterminism.

    An empty slot has strength 0, which makes it the natural `argmin` target, so
    unused slots fill before established ones are overwritten and no separate
    validity mask is needed.
    """
    k = hi - lo
    strength = jax.lax.dynamic_slice_in_dim(memory[..., 2], lo, k, axis=1)
    weakest = jnp.argmin(strength, axis=1)                        # [n]
    target = jax.nn.one_hot(weakest, k) > 0                       # [n, k]
    hit = target & should_write[:, None]

    fresh = jnp.zeros((memory.shape[0], k, 3)).at[..., 2].set(1.0)
    part = jax.lax.dynamic_slice_in_dim(memory, lo, k, axis=1)
    part = jnp.where(hit[..., None], fresh, part)
    return jax.lax.dynamic_update_slice_in_dim(memory, part, lo, axis=1)


def encode(memory: jax.Array, heading: jax.Array, cfg: Config) -> jax.Array:
    """Slots -> brain inputs, `[n, k, 3]` -> `[n, 4k]`.

    Four numbers per slot: the bearing as sin/cos, the distance squashed, and the
    confidence. The bearing is taken relative to the agent's own heading, the
    same egocentric convention `sensors.sense` uses for the retina, so "left"
    means the same thing to memory and to vision.

    `memory_dist_scale` sets where `tanh` is responsive: at the median
    distance-to-water it sits mid-range, and it saturates around the point where
    the trip stops being survivable -- which is the range over which the
    difference actually matters to the decision.
    """
    dx, dy, s = memory[..., 0], memory[..., 1], memory[..., 2]
    bearing = jnp.arctan2(dy, dx) - heading[:, None]
    dist = jnp.sqrt(dx * dx + dy * dy)
    feats = jnp.stack(
        [jnp.sin(bearing), jnp.cos(bearing),
         jnp.tanh(dist / cfg.memory_dist_scale), s],
        axis=2,
    )                                                    # [n, k, 4]
    return feats.reshape(memory.shape[0], -1)
