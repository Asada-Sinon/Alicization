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

**World.** Continuous 2D torus (512²) with a regrowing plant-energy field on a 128²
grid. Fixed-capacity population (`n_max`) behind an `alive` mask; births scatter into
freed slots, arrays are never resized.

**Terrain.** A mountain range, rivers, and forest — all derived from a single
elevation field rather than pasted on separately. The range is a gaussian ridge
along a meandering centerline, with peaks and passes modulated along its length;
rivers are the steepest-descent paths of that field, traced from sources near the
crest down to the sea; forest grows at mid elevation within reach of water. Climbing
costs energy (uphill only), canopy slows movement and shortens sight without
shortening a predator's reach, and bare rock grows nothing. So *where* an agent is
becomes a decision worth evolving, not just *who* it chases.

**Brain (摇光).** A fixed-topology *recurrent* net whose weights are the genome —
16 hidden units carrying memory across steps, outputs `[turn, thrust]`. No
backprop anywhere: brains change only by mutation and recombination.

**Senses.** A directional retina of 8 angular wedges around the agent's heading,
5 channels each — food, prey, predator, water, slope — plus its own energy, diet and
hydration. Prey/predator are read *relative to the viewer's own diet*, and slope is
read relative to the agent's own elevation, so one brain works whether it's grazing
or hunting, uphill or down.

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

On one RTX 4090, ~975 steps/s with the current config. A healthy run settles into
a stable pyramid rather than a collapse:

```
   step     pop   energy   water     age     plant   diet  dietSD  carn%  carnV  herbV
   5000     606     9.12    7.62   359.4     16034   0.24   0.308  18.5%   2.47   5.93
  15000     657     9.46    7.29   344.0     15575   0.28   0.334  23.4%   1.25   6.51
  30000     687     9.29    7.32   427.8     15335   0.30   0.345  25.6%   1.08   6.77
```

What to watch: `pop` holding rather than trending to zero, `carn%` oscillating in a
band instead of decaying to the floor, and `dietSD` staying high (~0.3 = the two
diets are still cleanly separated, not blurring into omnivores).

`carnV` versus `herbV` is the ambush-vs-pursuit tell. In the old flat world
carnivores evolved toward active pursuit and `carnV` climbed toward `herbV`. With
forest cover in the world they go the other way — `carnV` settles near 1 while
prey run at ~6.7 — because sitting in cover where prey cannot see you beats chasing
them. That is the forest doing what forest does, not a regression.

## Layout

```
underworld/   JAX sim core (config, state, terrain, spatial, sensors, brain,
              genome, dynamics, ecology, reproduction, step, metrics)
server/       FastAPI + websocket, binary snapshot protocol
web/          WebGL dashboard (index.html, render.js, main.js) — no build step
scripts/      run_headless.py, run_live.py
tests/        kernel sanity, invariants, determinism, neighbour index, terrain
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
