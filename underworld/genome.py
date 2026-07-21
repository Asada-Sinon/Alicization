"""Genome operations: gaussian mutation and (asexual-triggered) crossover of a
flattened brain-weights vector. Full sexual reproduction (mate-seeking courtship
behaviour, two-parent energy cost) is a later milestone; `crossover` here just
recombines two existing parents' genes at birth so useful sub-behaviours can mix
faster than mutation alone can find them.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import Config


def mutate(genome: jax.Array, key: jax.Array, cfg: Config) -> jax.Array:
    """Add gaussian noise to a batch of genomes: [N, G] -> [N, G]. Brain genes
    mutate at `mutation_sigma`; trait genes mutate slower.

    The asymmetry does two jobs at once. It keeps herbivore and carnivore
    lineages distinct rather than blurring toward omnivore, which is what it was
    originally tuned for. And it means the body drifts slower than the brain can
    track it -- a controller is rarely far from the body it is controlling, which
    is the standing problem with co-evolving morphology and control.

    `cfg.diet_mutation_asymmetric=False` is the ablation arm for the first half
    of that claim (docs/biology.md §10.1): diet then mutates at the same
    `mutation_sigma` as every brain gene instead of the slower `diet_mutation_sigma`.
    """
    sigma = jnp.full((cfg.genome_size,), cfg.mutation_sigma)
    diet_sigma = cfg.diet_mutation_sigma if cfg.diet_mutation_asymmetric else cfg.mutation_sigma
    sigma = sigma.at[cfg.diet_index].set(diet_sigma)
    sigma = sigma.at[cfg.invest_index].set(cfg.invest_mutation_sigma)
    sigma = sigma.at[cfg.size_index].set(cfg.size_mutation_sigma)
    return genome + jax.random.normal(key, genome.shape) * sigma


def crossover(genome_a: jax.Array, genome_b: jax.Array, key: jax.Array,
              cfg: Config) -> jax.Array:
    """Uniform crossover: each gene independently comes from parent A or B.

    **Only the diet gene is exempt**, always taken from parent A, so mixing two
    parents' brains never blends their heritable diet type back toward the
    omnivore middle.

    **Size is also exempt**, for the second of the two reasons given below.

    The investment gene deliberately does *not* get that protection, and the
    reasoning is worth recording because the opposite is tempting. Exempting a
    gene makes it non-recombining, so it stays linked to every other exempt gene
    and correlations between them can accumulate by hitchhiking rather than by
    selection -- which would corrupt `invest_diet_corr`, the very metric that
    exists to detect whether carnivores really do provision differently. The two
    reasons to exempt a gene are that it defines a discrete type worth keeping
    distinct (diet) or that the brain is adapted to it and an intermediate value
    would mismatch (this is that future body-size gene: `dynamics.act` and
    everything sensory is unaffected by `size` today, but the moment a future
    change makes speed or sensor reach depend on it, a body from a crossover
    step that mismatches either parent's control policy is exactly the
    co-evolution failure mode this exemption exists to avoid -- cheaper to
    exempt now than to migrate every evolved genome later). Investment is
    neither: it is read once at birth and never enters the sensorimotor loop at
    all.

    `cfg.diet_crossover_exempt=False` is the ablation arm for the *diet*
    exemption (docs/biology.md §10.1): diet then recombines like any other gene,
    letting two parents of different types produce an intermediate-diet child.
    Size has no matching switch -- its exemption is about controller/body
    mismatch, which is not what §10.1 is testing.
    """
    take_b = jax.random.bernoulli(key, 0.5, genome_a.shape)
    if cfg.diet_crossover_exempt:
        take_b = take_b.at[:, cfg.diet_index].set(False)
    take_b = take_b.at[:, cfg.size_index].set(False)
    return jnp.where(take_b, genome_b, genome_a)
