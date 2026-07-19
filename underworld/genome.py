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
    mutate at `mutation_sigma`; the diet gene mutates far slower so herbivore and
    carnivore lineages stay distinct rather than blurring together.
    """
    sigma = jnp.full((cfg.genome_size,), cfg.mutation_sigma)
    sigma = sigma.at[cfg.diet_index].set(cfg.diet_mutation_sigma)
    return genome + jax.random.normal(key, genome.shape) * sigma


def crossover(genome_a: jax.Array, genome_b: jax.Array, key: jax.Array,
              cfg: Config) -> jax.Array:
    """Uniform crossover: each brain gene independently comes from parent A or B.
    The diet gene always comes from parent A untouched, so mixing two parents'
    brains never blends their heritable diet type (which would blur the
    herbivore/carnivore split back toward the omnivore middle).
    """
    take_b = jax.random.bernoulli(key, 0.5, genome_a.shape)
    take_b = take_b.at[:, cfg.diet_index].set(False)
    return jnp.where(take_b, genome_b, genome_a)
