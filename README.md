# Underworld

A GPU-native 2D artificial-life sandbox, inspired by the artificial *fluctlights*
(人工摇光) of *SAO: Alicization*. Creatures aren't scripted — each carries a tiny
evolvable neural "brain" (its 摇光), sees the world through a retina, decides, and
survives or dies. Behavior **emerges** from selection, bottom-up.

Built to run on GPU from the first line: the whole world is tensors, the whole
population is stepped as one batched op, and it's meant to scale from a single
RTX 4090 to multi-GPU.

## What's in it

The full loop is up and stable: **JAX sim core → websocket → WebGL dashboard**.

**World.** Continuous 2D torus (256²) with a regrowing plant-energy field on a 64²
grid, plus a meandering stream that is the only source of water. Fixed-capacity
population (`n_max`) behind an `alive` mask; births scatter into freed slots, arrays
are never resized.

**Brain (摇光).** A fixed-topology *recurrent* net whose weights are the genome —
16 hidden units carrying memory across steps, outputs `[turn, thrust]`. No
backprop anywhere: brains change only by mutation and recombination.

**Senses.** A directional retina of 8 angular wedges around the agent's heading,
4 channels each — food, prey, predator, water — plus its own energy, diet and
hydration. Prey/predator are read *relative to the viewer's own diet*, so one brain
works whether it's grazing or hunting.

**Food web.** A heritable diet gene (0 = herbivore, 1 = carnivore), seeded bimodally
so two distinct types exist from step 0. Carnivores bite a specific weaker
*neighbour* (not a mean field), with trophic loss on the transfer; true carnivores
can't subsist on plants at all. A kill hydrates as well as feeds.

**Genetics.** Mutation plus uniform crossover of brain genes, with mates matched
assortatively by diet so recombination mixes within a species rather than blurring
the herbivore/carnivore split back to omnivore.

**Dashboard.** Live WebGL view of the world, time-series telemetry, and a per-agent
inspector that shows what the selected creature is actually seeing (its retina) and
the state of its recurrent memory.

Deferred: true two-parent-cost sexual reproduction, multi-GPU `shard_map`,
checkpointing.

## Setup

```bash
uv venv --python 3.10
uv pip install "jax[cuda12]" numpy fastapi "uvicorn[standard]" websockets pyyaml chex pytest
```

The package is intentionally **not installed** — run from the repo root.

## Run

Headless evolution (the FLA fast-forward path; prints metrics and asserts that
stable coexistence emerges):

```bash
.venv/bin/python scripts/run_headless.py            # 4000 steps
.venv/bin/python scripts/run_headless.py 30000 500  # total_steps, chunk
```

Live dashboard (opens a browser at http://localhost:8000):

```bash
.venv/bin/python scripts/run_live.py
# remote box: .venv/bin/python scripts/run_live.py --host 0.0.0.0 --no-open
```

Tests:

```bash
.venv/bin/python -m pytest
```

> The live server holds the whole GPU — stop it before running tests, or JAX
> fails with `CUDA_ERROR_OUT_OF_MEMORY`.

## Typical run

On one RTX 4090, ~1950 steps/s with the current config. A healthy run settles into
a stable pyramid rather than a collapse:

```
   step     pop   energy   water     age     plant   diet  dietSD  carn%  carnV  herbV
   1000     206     9.26    7.01   208.2      7130   0.25   0.320  20.4%   4.68   5.47
   2000     223     9.56    6.82   201.4      6841   0.21   0.287  15.7%   4.18   6.24
   3000     214     9.09    7.11   254.9      6995   0.24   0.316  19.6%   4.60   6.66
```

What to watch: `pop` holding rather than trending to zero, `carn%` oscillating in a
band instead of decaying to the floor, `dietSD` staying high (~0.3 = the two diets
are still cleanly separated, not blurring into omnivores), and `carnV` climbing
toward `herbV` — that last one means carnivores are evolving *active pursuit*
instead of sitting still and ambushing.

## Layout

```
underworld/   JAX sim core (config, state, spatial, sensors, brain, genome,
              dynamics, ecology, reproduction, step, metrics)
server/       FastAPI + websocket, binary snapshot protocol
web/          WebGL dashboard (index.html, render.js, main.js) — no build step
scripts/      run_headless.py, run_live.py
tests/        kernel sanity, invariants, determinism, neighbour index
```

See `CLAUDE.md` for architecture details, the per-step pipeline order, and the
cross-file contracts that break silently.

## Design notes

- **Everything is tensors.** `WorldState` is a pytree of `[n_max, ...]` arrays;
  `world_step` is a pure function, `jax.jit`-compiled and closed over a static
  `Config`. Fast-forward wraps it in `lax.scan`.
- **Fixed shapes.** Population is capped at `n_max`; life/death is a boolean mask;
  births are permutation-scatter into free slots. This is what keeps the step
  jittable and multi-GPU-shardable later.
- **No gradient training.** Brains evolve. Nothing backprops.
- **Sim ⟂ view.** The core runs headless; the dashboard is a subscriber that can
  attach and detach without disturbing a run.
- **Ecology numbers are empirical.** The constants in `config.py` carry comments
  recording what was tried and why — several plausible-looking values were tested
  and rejected for driving carnivores extinct over long runs.
