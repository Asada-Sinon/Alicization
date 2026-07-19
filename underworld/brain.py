"""The 'fluctlight' (摇光): a fixed-topology *recurrent* net whose weights live
in the genome. The whole population is stepped as one batch of einsums -- no
per-agent Python loop, no gradient training. Brains change only through mutation
and recombination. The recurrent hidden state gives each agent memory: its
decision depends not just on what it senses now but on its own internal state
carried across steps.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import Config


def split_params(genome: jax.Array, cfg: Config):
    """Slice a [N, genome_size] genome into batched recurrent-net tensors."""
    i, h, o = cfg.in_dim, cfg.hidden, cfg.out_dim
    n = genome.shape[0]
    a = i * h
    b = a + h * h
    c = b + h
    d = c + h * o
    w_in = genome[:, 0:a].reshape(n, i, h)
    w_rec = genome[:, a:b].reshape(n, h, h)
    b_h = genome[:, b:c]
    w_out = genome[:, c:d].reshape(n, h, o)
    b_out = genome[:, d:d + o]
    return w_in, w_rec, b_h, w_out, b_out


def forward(genome: jax.Array, inputs: jax.Array, hidden: jax.Array, cfg: Config):
    """One recurrent step for the whole population.

    inputs [N, in_dim], hidden [N, hidden] -> (outputs [N, out_dim] in [-1,1],
    new_hidden [N, hidden]).
    """
    w_in, w_rec, b_h, w_out, b_out = split_params(genome, cfg)
    h_new = jnp.tanh(
        jnp.einsum("ni,nih->nh", inputs, w_in)
        + jnp.einsum("nk,nkh->nh", hidden, w_rec)
        + b_h
    )
    out = jnp.tanh(jnp.einsum("nh,nho->no", h_new, w_out) + b_out)
    return out, h_new
