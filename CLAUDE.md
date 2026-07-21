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

**`CUDA_ERROR_OUT_OF_MEMORY` here is almost always a preallocation artifact, not
a real shortage.** JAX grabs 75% of the card up front (18.5 GB of 24 GB), so the
*second* process to start fails even though a full-size run's real peak is
**918 MiB** (measured, `n_max=16384`). Set

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python ...
```

and a dozen runs coexist happily — which is how the multi-seed sweeps get run in
parallel. Without it you are serialising jobs for no reason, and a live
`run_live.py` will appear to "hold the whole GPU" (it does not; it holds the
preallocation).

There is no linter, formatter, or JS toolchain configured. `web/` is plain
ES5-flavoured JS served statically by FastAPI — no bundler, no `node_modules`.

## Git

Remote is `origin` → `git@github-asada:Asada-Sinon/Alicization.git`, branch `main`
(tracking `origin/main`). Note `github-asada` is an **SSH host alias** from
`~/.ssh/config`, not a real hostname — don't "correct" it to `github.com`, and if a
push fails with a host-resolution error the alias is missing, not the URL.

Commit as you finish each coherent piece of work rather than letting changes pile up.
**Once the checks below pass, push to `origin main` without asking** — the push is
part of finishing the work, not a separate decision needing sign-off.

**Split by reason for change, not by feature.** A useful test: if the commit message
needs the words "also" or "while I was there", it should have been two commits. One
past commit bundled a new resource layer, a wire-protocol bump, and a shader change
across 14 files — that should have been three, because a later reader looking for
"when did the protocol change" has to read a message about fruit ecology to find it.

**Commit messages are written in Chinese** — both the subject line and the body.
This is the project's convention; follow it even when the conversation is in
English. Two things stay as they are:

- the `Co-Authored-By:` trailer, which is parsed by tooling, not read as prose
- identifiers quoted from the code (`carn_cost`, `world_step`, `UNTR`, file paths),
  which are names, not description — don't translate them

The bar for the body is the same as it would be in English: say *why* the change
was made and what was measured, not just what was touched.

```
为地形系统增加山脉、河流与森林

将世界扩大到 512²，让"待在哪里"本身成为值得演化的决策……

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

Before committing, make sure the work actually holds:

```bash
pkill -f "[s]cripts/run_live.py"    # brackets stop the pattern matching its
                                    # own shell. Only needed if you did NOT
                                    # set PREALLOCATE=false above.
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python -m pytest
node --check web/main.js && node --check web/render.js   # if web/ changed
```

The full suite takes several minutes — give it a long timeout rather than
assuming it hung.

Wire-format or shader changes need more than tests — see the contracts section below;
verify them against a running server or a screenshot before committing.

Keep out of the repo: screenshots, scratch scripts, `node_modules`, and anything
under `outputs/ checkpoints/ runs/`. `.gitignore` covers `*.log` and the usual
Python/venv noise, but scratch work belongs in the session scratchpad, not here.

## Research lands in `docs/`, or it did not happen

**Any investigation whose output is a conclusion rather than code MUST be written
to a file under `docs/` and committed before the task is considered done.** A
report that exists only in conversation is lost at the next context compaction —
this has already happened once: a full feasibility study on moving to 3D was
completed, verbally relayed, acted on by nobody, and then vanished, so the same
question had to be asked again from scratch weeks later.

- One topic, one file: `docs/<topic>.md`. Long is fine. These are allowed — and
  encouraged — to read like literature reviews: full prose, real citations
  (author, year, venue), tables, derivations, dead ends. They are not prompts
  and should not be compressed into bullet-point instructions.
- **Mark every claim by how it was established.** `docs/biology.md` uses
  `[现实]` (published fact), `[本世界实测]` (measured here), `[对应]` (where it
  lands in the code), `[提案，非结论]` (proposal, not a finding). Reuse that
  vocabulary. State plainly when a source could not be verified rather than
  laundering it into confident prose.
- Add a one-line pointer in `docs/TODO.md` so the next session finds it without
  reading everything.
- Negative and inconclusive results are the point, not an embarrassment — most
  of §9, §11, §12 and §13 of `docs/biology.md` are failures, and they are the
  most load-bearing sections in the file because they stop the same idea being
  re-attempted.

## Architecture

### The kernel is one jitted pure function

`underworld/step.py` `build_step(cfg)` closes over a frozen `Config` and returns a
`jax.jit`-ed `world_step(state, key) -> (state, Metrics)`. `make_scan` wraps it in
`lax.scan` for headless fast-forward. `new_world(cfg)` gives you
`(state, key, step_fn, scan_fn)`.

Per-step order matters and is easy to get wrong when editing:

1. `spatial.build_table` → `gather_neighbors` → `geometry` (neighbour index)
2. `sensors.sense` (retina + memory) → `brain.forward` (recurrent) → `dynamics.act` (move)
3. `memory.advance` — **after the move, before any write**, so a slot recorded
   below reads as ~0 offset from where the agent actually stands
4. `dynamics.graze` + `eat_fruit` + `drink`
5. `memory.write` ×2 (water slots on `drink_gain > 0`, fruit on `fruit_gain > 0`)
6. **neighbour index is rebuilt** — predation must see post-movement positions
7. `dynamics.predation` → `metabolize` → `thirst`
8. `reproduction.cull` (death) → `reproduce` (birth) — memory inheritance rides
   through `place()`
9. `ecology.regrow` ×2 (grass, fruit), and `diet` is re-cached from the genome

### Terrain is static, and derived from one elevation field

`underworld/terrain.py` builds every map field **once** in `build(cfg)`:
mountains are a gaussian ridge along a meandering centerline; rivers are the
steepest-descent paths of that field traced with `lax.scan` from sources near the
crest; forest is what grows at mid elevation within reach of water; the plant
carrying capacity is derived from forest and bare rock. They are three
consequences of one model, not three pasted-on rules.

`new_world` returns `(state, key, step_fn, scan_fn, terrain)` and the terrain is
**closed over** by `build_step`, not stored in `WorldState` — putting it in the
state would copy several `[n_cells]` fields through every `lax.scan` step for no
reason. Nothing in `terrain.py` runs per step.

Two scaling rules the config enforces:

- **World-scale lengths are fractions of `world_size`** (`ridge_sigma_frac`,
  `ridge_amp_frac`, …) so changing the map size keeps the geography proportionate.
  Agent-scale lengths (`vision_radius`, `river_half_width`, `attack_range`) stay
  absolute — those are set by the creatures, not the map.
- **`sense_grid` must scale with `world_size`** so a sense cell stays ≥
  `vision_radius`. Too small and agents outside the 3×3 block are invisible; too
  large and cells exceed `k_neighbors` and the overflow is silently dropped from
  both vision and predation. `test_sense_cell_covers_vision_radius` guards this.

Also note the plant grid cell must stay comparable to `river_half_width`: water is
sampled at cell centres, so a coarse grid on a large world can leave *no* cell
registering as water, and everything dies of thirst.

### Memory is two tiers, and neither is inherited

`memory.py` holds `[n_max, memory_slots, 3]` slots of `(dx, dy, strength)`. The
short tier is the recurrent hidden state in `brain.py`; the long tier is these
slots. **The vectors are relative to the holder, not absolute coordinates** — each
step subtracts the displacement and re-wraps to shortest-path, so the torus is
reasoned about once. Never recompute a slot from absolute positions.

Slots are **partitioned by position, not tagged**: `[0, memory_water_slots)` is
water, the rest fruit. The brain reads a fixed meaning per input group, and each
slot costs one input less. Writes use `argmin` → `one_hot` → `where` rather than
`.at[].set()`, so nothing here is a dynamic index or an atomic — unlike the
per-cell scatter-adds, this adds no nondeterminism. Strength 0 means "empty" and
is the natural `argmin` target, so no validity mask is needed.

Newborns get **empty** slots. Genes cross generations; memory is acquired within a
lifetime and dies with its holder — copying a parent's slots at birth is Lamarckian.
Removed in `cbe434d`; don't reintroduce it.

An earlier version of this paragraph said the copy "was also measured to do
nothing." **That was an overclaim and is worth keeping as a cautionary note.** The
data (n=6 paired, mean `inland_frac` +0.020, paired SD 0.031) gives p=0.175 with a
95% interval of [−0.013, +0.053] and **25% power** — a null was the most likely
outcome even if the true effect were exactly the +0.020 observed. What the data
does support, by equivalence test, is the narrower claim that any effect is
**smaller than 0.05** (TOST p=0.032). Removing the mechanism was still right,
because the Lamarckian argument stands on its own and the burden was on the
mechanism — but "underpowered, bounded below 0.05" is the honest description, not
"does nothing."

The legitimate route to cross-generational knowledge is social learning — juveniles
following adults and drinking for themselves, which `memory.write` already
supports since it only asks whether you are at water, not how you got there.
`reproduction.place` needs no change for the rank-3 field either way — its
`expand` is already generic.

That empty start is a real handicap: measured hazard is **17.7× higher** in the
first 50 steps of life than in old age, and 63% of all deaths happen there. See
`docs/biology.md` §6.

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
not stored: `in_dim = 5*retina_sectors + 3 + 4*memory_slots`, `brain_params`,
`diet_index`, `genome_size`, `memory_slots`.

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

Currently v6: 68-byte header, then agents, then the `plant` and `fruit` u8 planes.
New metrics can be added without touching `server/app.py` — `encode()` reads from a
dict built by `metrics._asdict()`, so any field on `Metrics` is already available by
name. Verify wire changes against a live server, not by reading; a bad offset
produces plausible-looking wrong numbers, not an error.

**Append, never insert.** v6 grew the header (`fruit_total` last, at offset 64) and
added a second grid plane after `plant`, and *no existing client offset moved* —
which is the only reason it was a four-line change. Putting either one further up
would have silently shifted every read after it. Same discipline applies to
`Metrics`: `scripts/run_headless.py` now reads it by name, but `protocol.encode`'s
pack call is still positional.

**Terrain travels in its own one-shot message** (`encode_terrain`, magic `UNTR`:
12-byte header + three `grid²` u8 planes), sent on connect and after a reset, not
per frame. `main.js` tells the two message types apart **by magic bytes, not by
length** — length would break the moment the grid changed. The client uploads it
as an RGB texture and the shader samples the height plane for relief, so no
world-generation formula is duplicated in GLSL any more. (This replaced an earlier
contract where the stream's sine formula had to be kept in sync between
`ecology.py` and `PLANT_FS`.)

**Species colours are duplicated in three files** and must match exactly:
`web/render.js` shader constants (`vec3` literals), `web/index.html` `:root` custom
properties, and the `C` object in `web/main.js`. Herbivore `#9e52eb` = `vec3(0.62,0.32,0.92)`,
carnivore `#f24038` = `vec3(0.95,0.25,0.22)`. They were out of sync once already.

## Working on this codebase

**Ecology parameters are empirically tuned, not arbitrary.** `config.py` carries long
comments recording what was tried and why values are where they are (`plant_max`,
`regrow_baseline`, `attack_range`, `carn_cost`, `n_init`). Several plausible-looking
changes have been tested and rejected because they drove carnivores extinct over 20k+
steps. Don't retune casually; validate with a long `run_headless.py` run and watch
`carn%`, `dietSD`, and `pop` for collapse, not just the first few hundred steps.

**Never tune ecology on a single run.** Carnivore survival is a near-threshold
stochastic process, and run-to-run variance is larger than most parameter effects.
A config that looked clearly best on one seed scored 0% carnivores on all four
seeds tested, while one that looked clearly worst averaged 2%+ — the entire
first-pass conclusion was noise.

**Six seeds paired, or five per arm unpaired — three is below the floor.** This
supersedes an earlier "≥3 seeds" rule here, which was arithmetically incapable of
producing a result: the *smallest attainable* two-sided p is 0.10 for a 3-vs-3
Mann-Whitney and 0.25 for a 3-pair signed-rank test, whatever the data. n=4 per
arm unpaired and n=6 paired are the first sample sizes where the test can reach
0.05 at all. Measured between-seed SD here is `inland_frac` ±0.027 and
`carnivore_frac` ±0.012, so detecting a 0.02 shift in `inland_frac` at 80% power
needs ~21 paired seeds — budget accordingly, or state plainly that the run was
underpowered.

Report the per-seed numbers, not just the mean; `--json` already emits them.
Prefer Mann-Whitney (or paired Wilcoxon) with an effect size and a bootstrap
interval over a bare mean, and do not Bonferroni-correct — report every p-value
you computed instead.

**Seeds vary the founders, not the world.** `terrain.build(cfg)` uses no RNG at
all, so every seed runs on the *same map*. Any spatial claim (`inland_frac`,
`water_bound_frac`, the distance metrics) therefore generalises to *this river
system*, not to rivers in general — that is pseudoreplication, and the honest fix
is to vary `ridge_wavenumber` / `ridge_amplitude` / `ridge_base_y` and the river
sources across a terrain seed, then treat terrain and founders as crossed factors.
Note a rigid translation is not enough: the world is a torus, so shifting the map
changes nothing.

**Spatial metrics need a null before they mean anything.** `inland_frac = 0.30` is
not "low" until you know what random placement gives. Computed from the terrain
alone: uniform over all cells → 0.556, over habitable cells → 0.675,
capacity-weighted → 0.650. So the population sits ~0.35 *below* chance, which is
the actual finding; and an effect of +0.020 is 5.8% of that gap, which is the
honest way to size it.

**`n_init` has a sweet spot that is narrow in both directions.** Seed near the
equilibrium and the carnivore founder pool is too small to survive its own
stochasticity; seed far above it and the initial die-off does the killing (a 7×
crash wiped carnivores on every seed). Roughly 3× the expected equilibrium works.

**Verify GPU and canvas output by looking at it.** A shader bug that made the plant field
render as a flat saturated slab shipped undetected for a long time because it was only
ever reasoned about. Screenshots (headless chromium with
`--use-gl=angle --use-angle=swiftshader --enable-unsafe-swiftshader`) catch what code
review does not.

**Determinism is not bit-exact on GPU.** Per-cell scatter-adds are atomic and reorder, so
`test_determinism` asserts identical life/death structure plus value tolerance over a
short horizon. For true bit-determinism run with
`XLA_FLAGS=--xla_gpu_deterministic_ops=true`.

**Dead code note:** `ecology.prey_field` and the sine-stream helpers are gone;
`ecology.gradient` is live again as the terrain slope operator.

**`README.md` is stale** — it describes the M0 vertical slice (MLP brain, food-gradient
sensing, asexual reproduction). The code has since gained a recurrent brain, retina
vision, predator-prey with neighbour-based predation, a water/thirst mechanic with a
meandering stream, and genetic crossover. Trust the source over the README.
