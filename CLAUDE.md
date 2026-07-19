# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**Underworld** — a GPU-native 2D artificial-life sandbox inspired by the artificial
fluctlights (人工摇光) of *SAO: Alicization*. Creatures are not scripted: each carries an
evolvable recurrent neural brain (its 摇光), senses through a retina, and lives or dies.
Behaviour emerges from selection. Themed naming is deliberate — world = Underworld,
brain = 摇光/fluctlight, time acceleration = FLA.

## Commands

The package is **not pip-installed**. Run everything from the repo root; scripts do
`sys.path.insert(0, ".")`.

```bash
.venv/bin/python -m pytest                                   # all kernel tests
.venv/bin/python -m pytest tests/test_kernel.py::test_determinism   # one test
.venv/bin/python scripts/run_headless.py                     # 4000 steps, prints metrics
.venv/bin/python scripts/run_headless.py 30000 500           # total_steps, chunk
.venv/bin/python scripts/run_live.py                         # dashboard at :8000
.venv/bin/python scripts/run_live.py --host 0.0.0.0 --no-open  # remote box
node --check web/main.js && node --check web/render.js       # no JS build step
```

**The live server holds the whole GPU.** Running `pytest` while `run_live.py` is up
fails with `CUDA_ERROR_OUT_OF_MEMORY`, and the server itself can get OOM-killed
(exit 137). Kill the server before running tests.

There is no linter, formatter, or JS toolchain configured. `web/` is plain
ES5-flavoured JS served statically by FastAPI — no bundler, no `node_modules`.

## Git

Remote is `origin` → `git@github-asada:Asada-Sinon/Alicization.git`, branch `main`
(tracking `origin/main`). Note `github-asada` is an **SSH host alias** from
`~/.ssh/config`, not a real hostname — don't "correct" it to `github.com`, and if a
push fails with a host-resolution error the alias is missing, not the URL.

Commit as you finish each coherent piece of work rather than letting changes pile up;
push to `origin main` once the relevant checks below pass. Prefer several small
commits over one large one.

Before committing, make sure the work actually holds:

```bash
pkill -f "[s]cripts/run_live.py"    # the server holds the GPU; brackets stop
                                    # the pattern matching its own shell
.venv/bin/python -m pytest
node --check web/main.js && node --check web/render.js   # if web/ changed
```

Wire-format or shader changes need more than tests — see the contracts section below;
verify them against a running server or a screenshot before committing.

Keep out of the repo: screenshots, scratch scripts, `node_modules`, and anything
under `outputs/ checkpoints/ runs/`. `.gitignore` covers `*.log` and the usual
Python/venv noise, but scratch work belongs in the session scratchpad, not here.

## Architecture

### The kernel is one jitted pure function

`underworld/step.py` `build_step(cfg)` closes over a frozen `Config` and returns a
`jax.jit`-ed `world_step(state, key) -> (state, Metrics)`. `make_scan` wraps it in
`lax.scan` for headless fast-forward. `new_world(cfg)` gives you
`(state, key, step_fn, scan_fn)`.

Per-step order matters and is easy to get wrong when editing:

1. `spatial.build_table` → `gather_neighbors` → `geometry` (neighbour index)
2. `sensors.sense` (retina) → `brain.forward` (recurrent) → `dynamics.act` (move)
3. `dynamics.graze` + `drink`
4. **neighbour index is rebuilt** — predation must see post-movement positions
5. `dynamics.predation` → `metabolize` → `thirst`
6. `reproduction.cull` (death) → `reproduce` (birth)
7. `ecology.regrow`, and `diet` is re-cached from the genome

### Everything is fixed-shape tensors

`WorldState` (`state.py`) is a `NamedTuple` pytree of `[n_max, ...]` arrays. Life and
death are the boolean `alive` mask only — **arrays are never resized**. Births use a
permutation-scatter idiom (`reproduction.reproduce`): parents and free slots are both
`argsort` permutations of `[0, n_max)`, so every `.at[idx].set(...)` writes each index
exactly once and non-births write back the existing value as a no-op. This is what keeps
the step jittable and shardable later. Preserve this pattern; do not introduce
boolean-indexed or dynamically-shaped operations into the step.

`spatial.py` uses the same discipline: agents bin onto `sense_grid`, rank within a cell
comes from `argsort` + `lax.cummax` (no dynamic loops), dead agents go to a dump cell and
overflow beyond `k_neighbors` to a dump column. One index feeds both vision and predation.

### Config is baked into the jit

`Config` is a frozen dataclass treated as compile-time constant. Anything that changes an
array *shape* (`n_max`, `grid`, `retina_sectors`, `hidden`, `trait_dim`) must live there,
and changing it requires a fresh `new_world`. Several fields are **derived properties**,
not stored: `in_dim = 4*retina_sectors + 3`, `brain_params`, `diet_index`, `genome_size`.

Consequence worth knowing before touching sensors or the brain: adding a retina channel
or resizing `hidden` changes `genome_size`, which **invalidates the entire evolved
population** — there is no checkpoint migration, brains restart from random.

### Sim ⟂ view

`server/app.py` runs the sim in an executor thread (JAX releases the GIL during device
work) and pushes the latest snapshot over a websocket. Slow clients just miss frames —
backpressure for free. The dashboard can attach/detach without disturbing a run.

## Cross-file contracts (these break silently)

**Binary protocol.** `server/protocol.py` defines a packed header + agents + plant field.
Three places must change together:
- the `_HEADER` struct format string and the `encode()` pack call
- `HEADER_BYTES` in `web/main.js`
- every `dv.getFloat32(offset)` in `main.js` `parse()` **after** the insertion point

Currently v4: 72-byte header. New metrics can be added without touching `server/app.py` —
`encode()` reads from a dict built by `metrics._asdict()`, so any field on `Metrics` is
already available by name. Verify wire changes against a live server, not by reading;
a bad offset produces plausible-looking wrong numbers, not an error.

**Species colours are duplicated in three files** and must match exactly:
`web/render.js` shader constants (`vec3` literals), `web/index.html` `:root` custom
properties, and the `C` object in `web/main.js`. Herbivore `#9e52eb` = `vec3(0.62,0.32,0.92)`,
carnivore `#f24038` = `vec3(0.95,0.25,0.22)`. They were out of sync once already.

**Stream geometry** is procedural, never stored: `ecology.stream_dist` computes it in
Python/JAX and the `PLANT_FS` fragment shader recomputes the same sine formula. The four
stream params ship in the header every frame so the client never hard-codes `Config`.

## Working on this codebase

**Ecology parameters are empirically tuned, not arbitrary.** `config.py` carries long
comments recording what was tried and why values are where they are (`plant_max`,
`regrow_baseline`, `attack_range`, `carn_cost`, `n_init`). Several plausible-looking
changes have been tested and rejected because they drove carnivores extinct over 20k+
steps. Don't retune casually; validate with a long `run_headless.py` run and watch
`carn%`, `dietSD`, and `pop` for collapse, not just the first few hundred steps.

**Verify GPU and canvas output by looking at it.** A shader bug that made the plant field
render as a flat saturated slab shipped undetected for a long time because it was only
ever reasoned about. Screenshots (headless chromium with
`--use-gl=angle --use-angle=swiftshader --enable-unsafe-swiftshader`) catch what code
review does not.

**Determinism is not bit-exact on GPU.** Per-cell scatter-adds are atomic and reorder, so
`test_determinism` asserts identical life/death structure plus value tolerance over a
short horizon. For true bit-determinism run with
`XLA_FLAGS=--xla_gpu_deterministic_ops=true`.

**Known dead code:** `ecology.gradient` and `ecology.prey_field` are unused leftovers
from the pre-retina scalar "smell" sensing.

**`README.md` is stale** — it describes the M0 vertical slice (MLP brain, food-gradient
sensing, asexual reproduction). The code has since gained a recurrent brain, retina
vision, predator-prey with neighbour-based predation, a water/thirst mechanic with a
meandering stream, and genetic crossover. Trust the source over the README.
