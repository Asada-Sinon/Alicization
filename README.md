# Underworld

A GPU-native 2D artificial-life sandbox, inspired by the artificial *fluctlights*
(人工摇光) of *SAO: Alicization*. Creatures aren't scripted — each carries a tiny
evolvable neural "brain" (its 摇光), senses the world, decides, and survives or
dies. Behavior **emerges** from selection, bottom-up.

Built to run on GPU from the first line: the whole world is tensors, the whole
population is stepped as one batched op, and it's meant to scale from a single
RTX 4090 to multi-GPU.

## Status — M0 (vertical slice)

The end-to-end pipeline is up: **JAX sim core → websocket → WebGL dashboard**.

- Continuous 2D torus world with a regrowing plant-energy field.
- Fixed-capacity population (`n_max`) with an `alive` mask; births scatter into
  freed slots, no array resizing.
- Each agent: an MLP brain (weights = genome), foraging via the local food
  gradient, energy metabolism, death by starvation/old-age, asexual reproduction
  with mutation, lineage colour.
- Headless fast-forward via `lax.scan` (FLA), and a live browser dashboard.

Coming next (see `../.claude/plans/alicization-playful-bee.md`): M1 recurrent
brains + retina vision + brain inspector, M2 predator-prey food web, M3 sexual
reproduction, M4 multi-GPU + checkpoints.

## Setup

```bash
uv venv --python 3.10
uv pip install "jax[cuda12]" numpy fastapi "uvicorn[standard]" websockets pyyaml chex pytest
```

## Run

Headless evolution (prints metrics, checks that foraging emerges):

```bash
.venv/bin/python scripts/run_headless.py            # 4000 steps
.venv/bin/python scripts/run_headless.py 8000 200   # longer
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

## Layout

```
underworld/   JAX sim core (config, state, brain, genome, sensors, dynamics,
              ecology, reproduction, step, metrics)
server/       FastAPI + websocket, binary snapshot protocol
web/          WebGL dashboard (index.html, render.js, main.js)
scripts/      run_headless.py, run_live.py
tests/        kernel sanity + determinism tests
```

## Design notes

- **Everything is tensors.** `WorldState` is a pytree of `[n_max, ...]` arrays;
  `world_step` is a pure function, `jax.jit`-compiled, and closed over a static
  `Config`. Fast-forward wraps it in `lax.scan`.
- **Fixed shapes.** Population is capped at `n_max`; life/death is a boolean
  mask; births are permutation-scatter into free slots. This is what keeps the
  step jittable and multi-GPU-shardable later.
- **No gradient training.** Brains evolve (mutation now, crossover later), they
  don't backprop.
- **Sim ⟂ view.** The core runs headless; the dashboard is just a subscriber and
  can attach/detach without disturbing the run.
