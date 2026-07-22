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

**Every python invocation needs `XLA_PYTHON_CLIENT_PREALLOCATE=false`.** Without
it JAX preallocates 75% of the card and the next process to start dies with a
fake `CUDA_ERROR_OUT_OF_MEMORY`; the real peak is 918 MiB. A `PreToolUse` hook
enforces this, so a missing prefix is refused rather than debugged. Why it
matters: `docs/conventions.md` §1.

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python scripts/check.py   # ~14s
.venv/bin/python scripts/check.py --contracts                 # ~0.2s, no JAX
.venv/bin/python scripts/check.py --full                      # + pytest, ~3min
.venv/bin/python -m pytest                                    # all kernel tests
.venv/bin/python -m pytest tests/test_kernel.py::test_determinism   # one test
.venv/bin/python scripts/run_headless.py                      # 4000 steps, prints metrics
.venv/bin/python scripts/run_headless.py 30000 500            # total_steps, chunk
.venv/bin/python scripts/run_live.py                          # dashboard at :8000
.venv/bin/python scripts/run_live.py --host 0.0.0.0 --no-open   # remote box
node --check web/main.js && node --check web/render.js        # no JS build step
```

There is no linter, formatter, or JS toolchain configured. `web/` is plain
ES5-flavoured JS served statically by FastAPI — no bundler, no `node_modules`.

## Verification

`scripts/check.py` is the fast loop, and it exists because there was nothing
between "it looks right" and a three-minute test suite. Run it after every
change; **paste its output rather than asserting the work is done.**

- **tier 1 (`--contracts`, 0.2s, no JAX)** — the cross-file contracts below,
  checked mechanically: the wire format against every offset in `web/main.js`,
  the species colours across the three files that duplicate them, the config
  scaling rules, syntax. A `PostToolUse` hook runs this after every source edit,
  so a broken contract comes back immediately instead of at the next commit.
- **tier 2 (default, 14s)** — a 2048-agent, 200-step world: no NaN, shapes
  intact, fields inside their capacities, plus a **golden band** on ten metrics
  in `scripts/golden.json`. That band is what catches a config change that
  silently moves the population — a 3% nudge to `eat_rate` trips it.
- **tier 3 (`--full`)** — everything plus `pytest`. This is the pre-commit run.

If a change is *meant* to move the golden numbers, re-record with `--bless` and
say why in the commit message. **Never widen the bands to make a failure go
away** — a band widened for that reason is a check deleted. Band sizing and the
measurement behind it: `docs/conventions.md` §9.

## Git

Remote is `origin` → `git@github-asada:Asada-Sinon/Alicization.git`, branch `main`
(tracking `origin/main`). Note `github-asada` is an **SSH host alias** from
`~/.ssh/config`, not a real hostname — don't "correct" it to `github.com`, and if a
push fails with a host-resolution error the alias is missing, not the URL.

Commit as you finish each coherent piece of work rather than letting changes pile up.
**Once the checks below pass, push to `origin main` without asking** — the push is
part of finishing the work, not a separate decision needing sign-off.

**Split by reason for change, not by feature.** A useful test: if the commit
message needs the words "also" or "while I was there", it should have been two
commits. The 14-file commit that taught this: `docs/conventions.md` §2.

**Commit messages are written in Chinese** — subject and body — even when the
conversation is in English. Two things stay as they are: the `Co-Authored-By:`
trailer (parsed by tooling, not read as prose) and identifiers quoted from the
code (`carn_cost`, `world_step`, `UNTR`, file paths — those are names, not
description). Say *why* the change was made and what was measured. Example and
full rationale: `docs/conventions.md` §2.

Before committing, make sure the work actually holds:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python scripts/check.py --full
```

That is `check.py` plus the whole suite; it takes several minutes — give it a
long timeout rather than assuming it hung.

Wire-format and shader changes need more than tests. `check.py --contracts`
covers the protocol offsets and the duplicated colours mechanically, but a
shader still has to be *looked at* — verify against a running server or a
screenshot before committing (`docs/conventions.md` §10).

Keep out of the repo: screenshots, scratch scripts, `node_modules`, and anything
under `outputs/ checkpoints/ runs/`. `.gitignore` covers `*.log` and the usual
Python/venv noise, but scratch work belongs in the session scratchpad, not here.

## Research lands in `docs/`, or it did not happen

**Any investigation whose output is a conclusion rather than code MUST be written
to a file under `docs/` and committed before the task is considered done.** A
report that exists only in conversation is lost at the next context compaction,
and that has already cost this project a full 3D feasibility study once
(`docs/conventions.md` §3).

- One topic, one file: `docs/<topic>.md`. Long is fine — these should read like
  literature reviews (full prose, real citations, tables, derivations, dead
  ends), not like prompts.
- **Mark every claim by how it was established**: `[现实]` published fact,
  `[本世界实测]` measured here, `[对应]` where it lands in the code,
  `[提案，非结论]` proposal. Say plainly when a source could not be verified
  rather than laundering it into confident prose.
- Add a one-line pointer in `docs/TODO.md` so the next session finds it.
- Negative and inconclusive results are the point, not an embarrassment. They
  are what stop the same idea being re-attempted.

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
Removed in `cbe434d`; don't reintroduce it. The ablation that tested this was
underpowered and an earlier note here overclaimed it as "measured to do nothing";
the honest reading, and why removal was still right, is in
`docs/conventions.md` §4.

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

`check.py --contracts` now verifies the mechanical half of this section — header
size, every f32/u32 offset, the `encode()` arity, the terrain magic, and the
duplicated colours — in 0.2s, and a hook runs it after every edit. What follows
is the design discipline the checker cannot express: *why* these are shaped the
way they are, and which changes are safe.

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

**Species colours are duplicated in three files** — `web/render.js` shader `vec3`
literals, `web/index.html` `:root` properties, the `C` object in `web/main.js` —
and were out of sync once already. Herbivore `#9e52eb`, carnivore `#f24038`; the
checker holds all three in agreement, so add any new duplicated colour to
`check_species_colours` rather than to a comment. (`--plant: #1a8033` in
`index.html` still claims to equal a `render.js` constant that no longer exists;
the plant colour is now folded into the terrain shading and has no single
literal to match.)

## Working on this codebase

**Ecology parameters are empirically tuned, not arbitrary.** `config.py` carries
long comments recording what was tried and why (`plant_max`, `regrow_baseline`,
`attack_range`, `carn_cost`, `n_init`). Several plausible-looking changes were
tested and rejected because they drove carnivores extinct over 20k+ steps.
`check.py`'s golden band now trips on a 3% nudge, which tells you a change had an
effect — it does not tell you the effect is good. Validate with a long
`run_headless.py` run and watch `carn%`, `dietSD` and `pop` for collapse, not
just the first few hundred steps. What has already been tried and rejected:
`docs/conventions.md` §8.

**Never tune ecology on a single run.** Carnivore survival is near-threshold and
run-to-run variance exceeds most parameter effects: a config that looked clearly
best on one seed scored 0% carnivores on all four seeds tested, while one that
looked clearly worst averaged 2%+. The entire first-pass conclusion was noise.

**Six seeds paired, or five per arm unpaired — three is below the floor**, because
a 3-vs-3 test cannot reach p=0.05 whatever the data. Report per-seed numbers, not
just the mean (`--json` emits them). Prefer Mann-Whitney or paired Wilcoxon with
an effect size and a bootstrap interval, and do not Bonferroni-correct — report
every p-value you computed. The arithmetic, and the ~21 paired seeds a 0.02
`inland_frac` shift actually needs: `docs/conventions.md` §5.

**Seeds vary the founders, not the world.** `terrain.build(cfg)` uses no RNG, so
every seed runs on the *same map*. Any spatial claim therefore generalises to
*this river system*, not to rivers in general — that is pseudoreplication. Vary
`ridge_wavenumber` / `ridge_amplitude` / `ridge_base_y` and the river sources
across a terrain seed and cross the two factors (`docs/conventions.md` §6).

**Spatial metrics need a null before they mean anything.** `inland_frac = 0.30` is
not "low" until you know that random placement gives 0.556–0.675 (computed from
the terrain in `docs/conventions.md` §7). The population sits ~0.35 *below*
chance — that is the finding, and it is what any effect size should be sized
against.

**Verify GPU and canvas output by looking at it.** A shader bug that rendered the
plant field as a flat saturated slab shipped undetected for a long time because it
was only ever reasoned about. Screenshots (headless chromium with
`--use-gl=angle --use-angle=swiftshader --enable-unsafe-swiftshader`) catch what
code review does not.

**Determinism is not bit-exact on GPU.** Per-cell scatter-adds are atomic and
reorder, so `test_determinism` asserts identical life/death structure plus value
tolerance over a short horizon. For true bit-determinism run with
`XLA_FLAGS=--xla_gpu_deterministic_ops=true`. Measured drift on the `check.py`
smoke config is nevertheless 0.000% over five runs — see `docs/conventions.md` §9
before touching the golden bands.

**Dead code note:** `ecology.prey_field` and the sine-stream helpers are gone;
`ecology.gradient` is live again as the terrain slope operator.

**`README.md` is stale** — it describes the M0 vertical slice (MLP brain, food-gradient
sensing, asexual reproduction). The code has since gained a recurrent brain, retina
vision, predator-prey with neighbour-based predation, a water/thirst mechanic with a
meandering stream, and genetic crossover. Trust the source over the README.
