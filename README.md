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
6 channels each — food, prey, predator, water, slope, peer — plus its own energy,
diet and hydration, plus four numbers per long-term memory slot. Prey/predator are
read *relative to the viewer's own diet*, and slope relative to the agent's own
elevation, so one brain works whether it's grazing or hunting, uphill or down.
`peer` is the complementary construction — diet *similarity* rather than difference
— because prey/predator are both exactly zero between two agents of the same diet,
which left conspecifics mutually invisible and blocked every form of social
behaviour. Total input width is 67; the genome is 1381 floats.

**Memory.** Two tiers. The short one is the recurrent hidden state; the long one is
four slots of `(dx, dy, strength)` — two for water, two for fruit — stored as
vectors *relative to the holder* and re-wrapped every step, so the torus is reasoned
about once. Newborns start empty: genes cross generations, memory does not.

**Food web.** A heritable diet gene (0 = herbivore, 1 = carnivore), seeded bimodally
so two distinct types exist from step 0. Carnivores bite a specific weaker
*neighbour* (not a mean field), with trophic loss on the transfer; true carnivores
can't subsist on plants at all. A kill hydrates as well as feeds.

**Genetics.** Mutation plus uniform crossover of brain genes, with mates matched
assortatively by diet so recombination mixes within a species rather than blurring
the herbivore/carnivore split back to omnivore. Three traits are heritable besides
the brain: `diet`, per-offspring `invest`ment, and body `size`. Each of the four
layers that hold the diet split apart is an independently ablatable switch — which
is how we found out the split is **not** a pure product of selection (see
`docs/experiments.md`).

**Dashboard.** Live WebGL view of the world, time-series telemetry, and a per-agent
inspector that shows what the selected creature is actually seeing (its retina) and
the state of its recurrent memory.

Deferred: true two-parent-cost sexual reproduction, multi-GPU `shard_map`,
checkpointing. **There is no checkpoint migration**, so any change to `in_dim` or
`genome_size` restarts every evolved brain from random — which is why such changes
get batched into a single deliberate population invalidation rather than made one
at a time.

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

> **`CUDA_ERROR_OUT_OF_MEMORY` here is almost always a preallocation artifact.**
> JAX grabs 75% of the card up front, so the *second* process to start fails even
> though a full-size run's real peak is **918 MiB** (measured, `n_max=16384`).
> Prefix with `XLA_PYTHON_CLIENT_PREALLOCATE=false` and a dozen runs — plus a live
> dashboard — coexist happily on one card. Without it you are serialising jobs for
> no reason.

## Typical run

On one RTX 4090, ~440–500 steps/s at the current config (`n_max=16384`, `in_dim=67`).
A healthy run settles into a stable pyramid rather than a collapse:

```
   step     pop   energy   water     age     plant   diet  dietSD  carn%  carnV  herbV
   5000     998     9.88    5.39   233.6     12134   0.18   0.258  12.0%   3.20   5.89
  15000    1195     9.74    4.10   281.8      9940   0.23   0.346  22.0%   1.90   6.99
  30000    1245     9.75    3.92   355.0      9576   0.26   0.371  25.5%   1.29   7.16
```

What to watch: `pop` holding rather than trending to zero, `carn%` oscillating in a
band instead of decaying to the floor, and `dietSD` staying high (~0.35 = the two
diets are still cleanly separated, not blurring into omnivores).

`carnV` versus `herbV` is the ambush-vs-pursuit tell. In the old flat world
carnivores evolved toward active pursuit and `carnV` climbed toward `herbV`. With
forest cover in the world they go the other way — `carnV` settles near 1.3 while
prey run at ~7.2 — because sitting in cover where prey cannot see you beats chasing
them. That is the forest doing what forest does, not a regression.

The run also prints a **death-cause breakdown**, which is the single most useful
number in the whole system:

```
  thirst       81.9%   mean age    46.7 steps
  predation    13.2%   mean age   120.3 steps
  starvation    4.8%   mean age   415.8 steps
  senescence    0.1%   mean age  3001.0 steps
```

Four fifths of all deaths are juveniles dying of thirst before they have learned
where the water is. That one bottleneck **censors every other selection pressure**
in the model, and it is why several biologically sensible traits have measured as
inert or actively harmful — see `docs/mortality.md`.

## Layout

```
underworld/   JAX sim core (config, state, terrain, spatial, sensors, brain,
              genome, memory, dynamics, ecology, reproduction, step, metrics)
server/       FastAPI + websocket, binary snapshot protocol
web/          WebGL dashboard (index.html, render.js, main.js) — no build step
scripts/      run_headless.py, run_live.py
tests/        kernel sanity, invariants, determinism, neighbour index, terrain
docs/         research notes — see below
```

See `CLAUDE.md` for architecture details, the per-step pipeline order, and the
cross-file contracts that break silently.

## Research notes

This project has a rule: **an investigation whose output is a conclusion rather
than code has to be written down under `docs/`, or it did not happen.** (It was
adopted after a full 3D feasibility study was completed, relayed verbally, acted
on by nobody, and lost.) Every claim is tagged by how it was established —
published fact, measured here, where it lands in the code, or an untested
proposal — and negative results are kept rather than quietly dropped.

| | |
| --- | --- |
| `docs/TODO.md` | Start here. Task queue and index; carries no argument of its own. |
| `docs/biology.md` | Real-world ecology and behaviour the design leans on: piosphere effect, edible biomass, preformed water, spatial memory, the Weismann barrier, Type III survivorship, why parental care evolves. |
| `docs/mortality.md` | Death-cause decomposition and competing-risks censoring. The denominator for every other document. |
| `docs/experiments.md` | Mechanisms that were built, measured, and **did not work** — the fruit layer, both signs of the trampling feedback, dismantling the diet split. The most load-bearing file here: it is what stops the same idea being re-attempted. |
| `docs/trait_evolution.md` | What can and cannot evolve today (1378 of 1381 genome floats are brain weights; only 3 are body), the evolution of evolvability, the open-endedness ceiling, and why the body-size gene was falsified in the wrong direction. |
| `docs/three_d.md` | Going 3D: render-only vs discrete strata vs true 3D, with measured benchmarks and a from-scratch encounter-rate derivation validated against three independent prototypes. |
| `docs/carnivore_riparian.md` | Why predators camp on the rivers, why the "too many herbivores" intuition is backwards, and five candidate ecological retunes. |

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
