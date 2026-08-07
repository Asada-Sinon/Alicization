"""The fast verification loop: what to run after every code change.

The full `pytest` suite is thorough but takes minutes, and a 20000-step
`run_headless.py` takes longer still. Between them there was nothing -- which
meant that after editing `sensors.py` or `config.py` the cheapest available
signal was "it looks right", and "looks right" is not a signal. This script
fills that gap: it is meant to run in seconds and to fail loudly.

Three tiers, cheapest first, each a superset of the one before:

    python scripts/check.py --contracts   # no JAX at all, well under a second
    python scripts/check.py               # + a small live world (the default)
    python scripts/check.py --full        # + the whole pytest suite

Tier 1 is the cross-file contracts `CLAUDE.md` warns "break silently" -- the
wire format against its three JavaScript readers, and the species colours that
are duplicated across three files and have been out of sync once already. Those
are exactly the failures that no Python test can see and that a code review
reliably misses, and checking them costs a regex.

Tier 2 builds a *small* world and steps it. It is not an experiment and proves
nothing about ecology -- it proves the kernel still runs, the shapes still line
up, and nothing has gone NaN. It also compares a handful of metrics against
`scripts/golden.json`, which is what catches the change that silently halves the
population.

    XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python scripts/check.py

Exit code is 0 only if every check passed. Nothing here writes to the repo
except `--bless`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import py_compile
import re
import struct
import subprocess
import sys
import time

# Allow running from the repo root without installing the package.
sys.path.insert(0, ".")

ROOT = pathlib.Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "scripts" / "golden.json"

# The smoke world. Only the *agent* arrays shrink: `world_size`, `grid` and
# `sense_grid` stay at their real values because the config enforces scaling
# relations between them (a sense cell must stay >= vision_radius; a plant cell
# must stay comparable to river_half_width), and a checker that quietly violated
# those would be testing a world that cannot exist. Shrinking `n_max` alone
# changes no geometry -- it just makes the compile and the step cheap.
SMOKE = dict(n_max=2048, n_init=400)
SMOKE_STEPS = 200
SMOKE_SEED = 0

# Metrics compared against golden, and the relative band each must stay inside.
#
# These were sized by measurement, not by guessing. Five identical runs of the
# smoke config on this box drifted **0.000%** on every metric below -- the
# atomic scatter-add reordering that `test_determinism` has to tolerate does not
# show up at 200 steps on 2048 agents. So the bands are not set by noise; they
# are set by *granularity*, which is the thing that actually limits them:
#
#   population 832  =>  one agent is 0.12% of it
#   carnivore_frac  =>  one agent is 5.6% of it (18 carnivores of 832)
#
# A band has to clear the quantum of one agent or it fails on a change that
# means nothing, so `carnivore_frac` gets the widest one here despite being the
# most stable in absolute terms. Everything else is set an order of magnitude
# above the measured drift and still far below any real behavioural change.
#
# If these start flaking, re-measure before widening -- a band widened to make a
# failure go away is a check deleted.
GOLDEN_BANDS = {
    "population": 0.02,       # ~17 agents
    "mean_energy": 0.03,
    "mean_water": 0.03,
    "mean_age": 0.03,
    "plant_total": 0.02,
    "fruit_total": 0.02,
    "mean_diet": 0.03,
    "carnivore_frac": 0.15,   # ~2.7 agents -- granularity-bound, not noise-bound
    "mean_size": 0.01,        # population means over 832 agents; very stable
    "mean_invest": 0.01,
}


class Report:
    """Collects pass/fail so one run reports every problem, not just the first."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0

    def ok(self, what: str) -> None:
        self.checks += 1
        print(f"  \033[32mok\033[0m   {what}")

    def fail(self, what: str, detail: str) -> None:
        self.checks += 1
        self.failures.append(f"{what}: {detail}")
        print(f"  \033[31mFAIL\033[0m {what}\n       {detail}")

    def expect(self, cond: bool, what: str, detail: str = "") -> bool:
        if cond:
            self.ok(what)
        else:
            self.fail(what, detail)
        return cond


# --------------------------------------------------------------------------
# Tier 1: contracts. No JAX, no GPU, no world.
# --------------------------------------------------------------------------

def check_syntax(r: Report) -> None:
    """Every Python file parses, and every JS file parses if node is around.

    Cheap, and it means a typo in a file the smoke world does not import still
    gets caught before it reaches a commit.
    """
    bad = []
    for p in sorted(ROOT.glob("**/*.py")):
        if any(part in (".venv", ".claude", "__pycache__") for part in p.parts):
            continue
        try:
            py_compile.compile(str(p), doraise=True, cfile=str(p) + "c")
        except py_compile.PyCompileError as e:
            bad.append(f"{p.relative_to(ROOT)}: {e.msg.splitlines()[-1]}")
        finally:
            pathlib.Path(str(p) + "c").unlink(missing_ok=True)
    r.expect(not bad, "python files parse", "; ".join(bad))

    js = sorted((ROOT / "web").glob("*.js"))
    try:
        bad_js = [str(p.relative_to(ROOT)) for p in js
                  if subprocess.run(["node", "--check", str(p)],
                                    capture_output=True).returncode != 0]
        r.expect(not bad_js, "web/*.js parse", "; ".join(bad_js))
    except FileNotFoundError:
        print("  skip node --check (node not on PATH)")


def check_wire_protocol(r: Report) -> None:
    """`server/protocol.py` against every offset in `web/main.js`.

    `CLAUDE.md`: "a bad offset produces plausible-looking wrong numbers, not an
    error". That is precisely why this belongs in a checker rather than in a
    reviewer's head. Three things must agree, and all three are derived here
    from the struct format string rather than written down twice:

      - the header size against the client's `HEADER_BYTES`
      - the *number* of float fields against the client's `getFloat32` reads
      - each field's byte offset against the offset the client reads it from

    The last one is the whole point: appending a field is safe, inserting one
    shifts every read after it and nothing errors.
    """
    from server import protocol

    main_js = (ROOT / "web" / "main.js").read_text()

    m = re.search(r"HEADER_BYTES\s*=\s*(\d+)", main_js)
    if not r.expect(m is not None, "main.js declares HEADER_BYTES", "not found"):
        return
    js_bytes = int(m.group(1))
    r.expect(protocol._HEADER.size == js_bytes,
             f"header size agrees ({protocol._HEADER.size} bytes)",
             f"protocol.py packs {protocol._HEADER.size}, main.js reads {js_bytes}")

    # Walk the format string to get the authoritative (offset, kind) list.
    offsets, off = [], 0
    for kind, size in _walk_format(protocol._HEADER.format.lstrip("<>!=")):
        offsets.append((off, kind))
        off += size

    # Read only the snapshot parser: `parseTerrain` has its own offsets against
    # a different header and would otherwise pollute the comparison.
    parse = main_js
    if "function parse" in parse:
        parse = parse[parse.index("function parse"):]
    if "function parseTerrain" in parse:
        parse = parse[:parse.index("function parseTerrain")]
    js_f32 = [int(x) for x in re.findall(r"getFloat32\((\d+),", parse)]
    js_u32 = [int(x) for x in re.findall(r"getUint32\((\d+),", parse)]

    py_f32 = [o for o, k in offsets if k == "f"]
    py_u32 = [o for o, k in offsets if k in "iI"]

    r.expect(sorted(js_f32) == py_f32,
             f"header f32 offsets agree ({len(py_f32)} fields)",
             f"protocol.py {py_f32}\n       main.js    {sorted(js_f32)}")
    r.expect(sorted(js_u32) == py_u32,
             f"header u32 offsets agree ({len(py_u32)} fields)",
             f"protocol.py {py_u32}\n       main.js    {sorted(js_u32)}")

    # `encode()`'s pack call is still positional (CLAUDE.md flags this), so an
    # arity mismatch is a live risk: struct.pack would raise at runtime, but
    # only when a client actually connects, which may be long after the commit.
    src = (ROOT / "server" / "protocol.py").read_text()
    # Anchored so it does not match `_TERRAIN_HEADER.pack(`, which contains
    # `_HEADER.pack(` as a substring and takes three arguments.
    pack = re.search(r"(?<![A-Z])_HEADER\.pack\(", src)
    n_args = _count_args(src, pack.end() - 1) if pack else -1
    r.expect(n_args == len(offsets),
             f"encode() passes {len(offsets)} values to a {len(offsets)}-field header",
             f"format has {len(offsets)} fields, pack call passes {n_args}")

    # The terrain message is told apart by magic bytes, not length -- so the
    # magic the client compares against has to be the magic Python sends.
    r.expect(protocol._TERRAIN_HEADER.size == 12,
             "terrain header is 12 bytes",
             f"got {protocol._TERRAIN_HEADER.size}")
    magic_js = re.search(
        r"magic\[0\]\s*===\s*(\d+).*?magic\[1\]\s*===\s*(\d+).*?"
        r"magic\[2\]\s*===\s*(\d+).*?magic\[3\]\s*===\s*(\d+)", main_js, re.S)
    if magic_js:
        got = bytes(int(g) for g in magic_js.groups())
        r.expect(got == protocol.TERRAIN_MAGIC,
                 f"terrain magic agrees ({protocol.TERRAIN_MAGIC!r})",
                 f"protocol.py {protocol.TERRAIN_MAGIC!r}, main.js {got!r}")


def _count_args(src: str, open_paren: int) -> int:
    """Arity of the call whose '(' is at `open_paren`.

    Depth-aware, because the arguments here are `float(metrics.get("x", 0.0))`
    and a naive comma count reads 30 where there are 17; and trailing-comma
    aware, because the call has one and it is not an argument.
    """
    depth, args, pending = 0, 0, False
    for ch in src[open_paren:]:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                return args + (1 if pending else 0)
        elif ch == "," and depth == 1:
            args, pending = args + 1, False
        elif depth >= 1 and not ch.isspace():
            pending = True
    return -1


def _walk_format(fmt: str):
    """Yield (kind, size) per field of a struct format, expanding counts."""
    i = 0
    while i < len(fmt):
        if fmt[i] == " ":
            i += 1
            continue
        n = ""
        while fmt[i].isdigit():
            n += fmt[i]
            i += 1
        kind, count = fmt[i], int(n or 1)
        i += 1
        if kind == "s":                      # one field of `count` bytes
            yield kind, count
        else:                                # `count` fields of one item each
            for _ in range(count):
                yield kind, struct.calcsize(kind)


def check_agent_stride(r: Report) -> None:
    """**per-agent record 的宽度**，在三个文件之间。

    `check_wire_protocol` 查的全是 header；**per-agent 那一段此前完全没人管**。
    它坏掉的表现是：`protocol.encode` 每个 agent 写 K 个 float 而 `main.js` 按 K-1 读，
    于是**从第二个 agent 起每一个字段都错位**——画面照样画出来，颜色/大小/位置全是别人的。
    **看起来合理的错画面，不是异常**，正是 `CLAUDE.md` 说的该由检查器守的那一类。

    三处必须自洽：

      - `protocol.py` 的 `agents = np.empty((n, K), ...)`
      - `web/main.js` 的 `STRIDE = K`
      - `web/render.js` 里每个 `vertexAttribPointer(..., stride, offset)` 的最大 offset
        必须落在 K 个 float 之内（stride 本身由 `STRIDE_FLOATS * 4` 算出，不重复写死）
    """
    src = (ROOT / "server" / "protocol.py").read_text()
    main_js = (ROOT / "web" / "main.js").read_text()
    render_js = (ROOT / "web" / "render.js").read_text()

    m = re.search(r"agents\s*=\s*np\.empty\(\(\s*n\s*,\s*(\d+)\s*\)", src)
    if not r.expect(m is not None, "protocol.py declares the agent record width",
                    "no `agents = np.empty((n, K)` found"):
        return
    py_k = int(m.group(1))

    m = re.search(r"STRIDE\s*=\s*(\d+)", main_js)
    if not r.expect(m is not None, "main.js declares STRIDE", "not found"):
        return
    js_k = int(m.group(1))

    r.expect(py_k == js_k,
             f"agent record width agrees ({py_k} floats)",
             f"protocol.py writes {py_k}, main.js reads {js_k}")

    # `protocol.encode` 逐列赋值，列号必须恰好覆盖 0..K-1，一个不多一个不少。
    cols = sorted({int(c) for c in re.findall(r"agents\[:,\s*(\d+)\]\s*=", src)})
    r.expect(cols == list(range(py_k)),
             f"encode() fills every agent column (0..{py_k - 1})",
             f"filled {cols}, expected {list(range(py_k))}")

    # shader 的属性必须整个落在记录之内。**要查末字节，不是首字节**：
    # 一个 `vec2` 属性从 offset 36 起会读到 40，而 `36 < 40` 会让只查首字节的版本
    # 静默通过、实际越界读到下一个 agent 的 x。v10 把相邻的 diet+energy 并成 vec2,
    # 正是这一步第一次让这个洞可被触发——所以这里捕获**分量数**。
    ptrs = re.findall(
        r"vertexAttribPointer\(\s*[^,]+,\s*(\d+)\s*,\s*gl\.FLOAT\s*,"
        r"\s*\w+\s*,\s*stride\s*,\s*(\d+)\s*\)", render_js)
    worst = max((int(o) + 4 * int(c) for c, o in ptrs), default=-1)
    r.expect(bool(ptrs) and worst <= py_k * 4,
             f"render.js agent attributes fit in the record (last byte {worst} <= {py_k * 4})",
             f"an attribute ends at {worst}, past {py_k} floats ({py_k * 4} bytes)")

    # **锚点不能只靠 `pt.`**：某条 pointer 写成别的变量名就会从 `ptrs` 里掉出去，
    # max 被低估而检查静默通过。改锚 `stride` 字面量，并与实际启用的属性数交叉核对。
    # 减 1 是 `a_h`，它走另一个 buffer（`stride` 参数为 0），不在这条记录里。
    enabled = render_js.count("enableVertexAttribArray(pt.")
    r.expect(len(ptrs) == enabled - 1,
             f"every enabled agent attribute is accounted for ({len(ptrs)} of {enabled - 1})",
             f"{enabled - 1} attributes enabled from the agent buffer but "
             f"{len(ptrs)} pointer calls matched -- one is written in a form the "
             f"regex misses, so the bounds check above is reading an incomplete set")


def check_ecotype_bins(r: Report) -> None:
    """dashboard 的觅食直方图必须与**分析脚本**用同一套分箱。

    面板画的是 `forage_pref` 的分布，而 `docs/multispecies_feasibility.md` §17.9/§18
    的判决也是在这个分布上做的。**两边分箱不一样，屏幕上的东西就不是判决里的那个东西**
    ——而且不会有任何报错，只会得到一张看起来合理的图。

    只钉**箱数**：那是能可靠正则化的部分。分箱**公式**（端点 clip 而非丢弃）由
    `tests/test_neutral_null.py:test_hist_of_counts_every_individual` 在 Python 端守，
    JS 端不测（本仓库不跑 node 之外的 JS 工具）。
    """
    traj = (ROOT / "explorations" / "20260804-readouts" / "trajectory.py").read_text()
    main_js = (ROOT / "web" / "main.js").read_text()
    m = re.search(r"BINS\s*=\s*np\.linspace\(\s*0(?:\.0)?\s*,\s*1(?:\.0)?\s*,\s*(\d+)\s*\)", traj)
    j = re.search(r"EC_BINS\s*=\s*(\d+)", main_js)
    if not r.expect(m is not None and j is not None,
                    "both sides declare the forage binning",
                    f"trajectory.py BINS found={m is not None}, main.js EC_BINS found={j is not None}"):
        return
    py_bins = int(m.group(1)) - 1          # linspace(0,1,N) 给 N-1 个箱
    js_bins = int(j.group(1))
    r.expect(py_bins == js_bins,
             f"dashboard forage histogram matches the analysis binning ({js_bins} bins)",
             f"trajectory.py has {py_bins} bins, main.js draws {js_bins}")


def check_species_colours(r: Report) -> None:
    """Herbivore and carnivore colours, duplicated across three files.

    `CLAUDE.md`: "They were out of sync once already." The shader carries them
    as float `vec3` literals rounded to two decimals, so the comparison is
    against the hex rounded the same way -- exact equality on the rounded value,
    not a loose tolerance, or a genuinely different colour could slip through.
    """
    main_js = (ROOT / "web" / "main.js").read_text()
    render_js = (ROOT / "web" / "render.js").read_text()
    index_html = (ROOT / "web" / "index.html").read_text()

    for name in ("herb", "carn"):
        m_js = re.search(rf"\b{name}:\s*\"(#[0-9a-fA-F]{{6}})\"", main_js)
        m_css = re.search(rf"--{name}:\s*(#[0-9a-fA-F]{{6}})", index_html)
        m_gl = re.search(rf"vec3\s+{name}\s*=\s*vec3\(([^)]+)\)", render_js)
        if not r.expect(all((m_js, m_css, m_gl)), f"{name} colour found in all 3 files",
                        f"main.js={bool(m_js)} index.html={bool(m_css)} render.js={bool(m_gl)}"):
            continue
        hexes = {m_js.group(1).lower(), m_css.group(1).lower()}
        if not r.expect(len(hexes) == 1, f"{name} hex agrees between main.js and index.html",
                        f"{m_js.group(1)} vs {m_css.group(1)}"):
            continue
        h = m_js.group(1).lstrip("#")
        want = tuple(round(int(h[i:i + 2], 16) / 255.0, 2) for i in (0, 2, 4))
        got = tuple(round(float(x), 2) for x in m_gl.group(1).split(","))
        r.expect(want == got, f"{name} shader vec3 matches {m_js.group(1)}",
                 f"hex rounds to vec3{want}, render.js has vec3{got}")


def check_config_invariants(r: Report) -> None:
    """Shape relations that would otherwise only surface as a jit error deep in
    a run, plus the two scaling rules `config.py` documents but cannot enforce.
    """
    from underworld import Config

    cfg = Config()
    # Not a tautology even though `in_dim` is derived: it pins the composition
    # the docstring claims against the one the property computes. Adding a
    # retina channel makes this fail until the number here is updated too, which
    # is the point -- that edit invalidates every evolved population (genome_size
    # moves), so it should cost a deliberate second edit rather than sliding
    # through as a one-character change.
    r.expect(cfg.in_dim == 6 * cfg.retina_sectors + 3 + 4 * cfg.memory_slots,
             f"in_dim = {cfg.in_dim} matches its stated composition "
             f"(6 x {cfg.retina_sectors} sectors + 3 + 4 x {cfg.memory_slots} slots)",
             f"in_dim={cfg.in_dim}, sectors={cfg.retina_sectors}, "
             f"memory_slots={cfg.memory_slots} -- if a channel was added on "
             f"purpose, update this line and note that genome_size moved")
    r.expect(cfg.genome_size == cfg.brain_params + cfg.trait_dim,
             f"genome_size = {cfg.genome_size} = brain_params + trait_dim")
    r.expect(cfg.escape_index < cfg.genome_size,
             "trait gene indices are inside the genome",
             f"escape_index={cfg.escape_index}, genome_size={cfg.genome_size}")
    # Day-night clock (docs/day_night.md): day_length=0 is the disabled default
    # (compile-time no-op); a negative period is meaningless, and the night vision
    # multiplier must stay a fraction so darkness only ever shrinks inter-agent
    # sight, never extends it past the daytime radius.
    r.expect(cfg.day_length >= 0,
             f"day_length = {cfg.day_length} is non-negative (0 disables the clock)",
             f"day_length={cfg.day_length} must be >= 0")
    r.expect(0.0 <= cfg.night_vision_floor <= 1.0,
             f"night_vision_floor = {cfg.night_vision_floor} is a fraction in [0, 1]",
             f"night_vision_floor={cfg.night_vision_floor} must be in [0, 1]")

    sense_cell = cfg.world_size / cfg.sense_grid
    r.expect(sense_cell >= cfg.vision_radius,
             f"sense cell ({sense_cell:.1f}) covers vision_radius ({cfg.vision_radius})",
             "agents outside the 3x3 block become invisible to vision and predation")
    # The evolvable attack range obeys the same ceiling as vision: a bite beyond the
    # sense cell reaches prey the neighbour table never gathered, so the extra reach
    # would fail silently (docs/attack_range_redqueen.md). Guard the *max* the gene
    # can express, not the neutral value.
    r.expect(sense_cell >= cfg.attack_max,
             f"sense cell ({sense_cell:.1f}) covers max attack_range ({cfg.attack_max})",
             "evolved long-reach predators would bite prey outside the gathered 3x3 block")
    # Water is sampled at cell centres, so a river of width 2*half_width is only
    # guaranteed to land on a sample if a cell is no wider than that -- coarser
    # and a river can thread between centres, no cell reads as drinkable, and
    # everything dies of thirst. This is the necessary condition, cheap to
    # state; whether *this* river system actually registers is measured for real
    # in tier 2 (`drinkable cells exist`), which is the check that matters.
    r.expect(cfg.cell_size <= 2 * cfg.river_half_width,
             f"plant cell ({cfg.cell_size:.1f}) can resolve a river "
             f"({2 * cfg.river_half_width:.1f} wide)",
             "a river can fall between cell centres and leave no drinkable cell")


# --------------------------------------------------------------------------
# Tier 2: a small live world.
# --------------------------------------------------------------------------

def run_smoke(r: Report, bless: bool) -> None:
    import dataclasses

    import jax
    import numpy as np

    from underworld import Config, new_world

    cfg = dataclasses.replace(Config(), seed=SMOKE_SEED, **SMOKE)
    t0 = time.time()
    state, key, _step, scan_fn, terrain = new_world(cfg)
    # **预热一次并丢弃。** 这不是性能优化，是**消除一个 6.88% 的方差源**：
    # XLA 在进程内第一次执行时做 autotuning（挑 kernel），选出来的归约顺序与之后
    # 不同，而逐格 scatter-add 是原子操作、对顺序敏感。实测同一配置连跑 6 次，
    # `fruit_total` 第 1 次是 40.598、第 2–6 次全是 43.563——**首次运行是个离群值，
    # 而 `check.py` 每次都是全新进程，测的永远是那个离群值**。
    # `conventions.md` §9 记的「五次运行漂移 0.000%」是在同一进程内测的，
    # 因此**恰好错过了这个方差源**；golden 的 ±2% band 也就定标在了一个
    # 比真实散布窄 3 倍的尺度上。
    # 预热之后测量总是落在「已 autotuned」的状态，band 才有意义。
    _ws, _wk, _wm = scan_fn(state, key, 1)
    jax.block_until_ready(_wm)
    state, key, ms = scan_fn(state, key, SMOKE_STEPS)
    jax.block_until_ready(ms)
    print(f"  ({SMOKE_STEPS} steps on n_max={cfg.n_max} in {time.time() - t0:.1f}s "
          f"incl. compile, device {jax.devices()[0].platform})")

    # --- nothing is NaN or Inf ---------------------------------------------
    bad = [f for f, v in zip(state._fields, state)
           if v.dtype != bool and not bool(np.isfinite(np.asarray(v)).all())]
    r.expect(not bad, "no NaN/Inf anywhere in WorldState", f"non-finite: {bad}")

    metrics = {k: float(np.asarray(v)[-1]) for k, v in ms._asdict().items()}
    bad_m = [k for k, v in metrics.items() if not np.isfinite(v)]
    r.expect(not bad_m, "no NaN/Inf in Metrics", f"non-finite: {bad_m}")

    # --- shapes still line up ----------------------------------------------
    r.expect(state.last_input.shape == (cfg.n_max, cfg.in_dim),
             f"last_input is [n_max, in_dim] = {state.last_input.shape}",
             f"expected {(cfg.n_max, cfg.in_dim)}, got {state.last_input.shape}")
    r.expect(state.genome.shape == (cfg.n_max, cfg.genome_size),
             f"genome is [n_max, genome_size] = {state.genome.shape}",
             f"expected {(cfg.n_max, cfg.genome_size)}")
    r.expect(state.memory.shape == (cfg.n_max, cfg.memory_slots, 3),
             f"memory is [n_max, slots, 3] = {state.memory.shape}")

    # --- the population is still there --------------------------------------
    alive = np.asarray(state.alive)
    pop = int(alive.sum())
    r.expect(pop > 0, f"population survived ({pop} alive)",
             "everything died inside 200 steps -- the kernel, not the ecology")
    r.expect(pop <= cfg.n_max, "population within n_max", f"{pop} > {cfg.n_max}")

    # --- per-agent quantities stay in their declared ranges ------------------
    # Checked over the living only: dead slots are not zeroed on death, they are
    # simply masked out, so a dead row carrying stale values is correct.
    def live(x):
        return np.asarray(x)[alive]

    r.expect(bool((live(state.energy) >= 0).all()), "no living agent has negative energy",
             f"min energy {float(live(state.energy).min()):.3f}")
    r.expect(bool((live(state.water) >= 0).all()), "no living agent has negative water",
             f"min water {float(live(state.water).min()):.3f}")
    d = live(state.diet)
    r.expect(bool(((d >= 0) & (d <= 1)).all()), "diet stays in [0, 1]",
             f"range [{d.min():.3f}, {d.max():.3f}]")
    pos = live(state.pos)
    r.expect(bool(((pos >= 0) & (pos < cfg.world_size)).all()),
             "positions stay on the torus",
             f"range [{pos.min():.2f}, {pos.max():.2f}] vs world_size {cfg.world_size}")
    strength = live(state.memory)[:, :, 2]
    r.expect(bool(((strength >= 0) & (strength <= 1)).all()),
             "memory strength stays in [0, 1]",
             f"range [{strength.min():.3f}, {strength.max():.3f}]")
    off = np.abs(live(state.memory)[:, :, :2])
    r.expect(bool((off <= cfg.half_world + 1e-3).all()),
             "memory offsets stay within half a world",
             f"max |offset| {off.max():.2f} > half_world {cfg.half_world}")

    # --- fields stay under the capacities terrain derived for them -----------
    plant, fruit = np.asarray(state.plant), np.asarray(state.fruit)
    cap, fcap = np.asarray(terrain.capacity), np.asarray(terrain.fruit_capacity)
    r.expect(bool((plant <= cap + 1e-3).all()) and bool((plant >= -1e-6).all()),
             "plant field within [0, capacity]",
             f"max excess {float((plant - cap).max()):.4f}")
    r.expect(bool((fruit <= fcap + 1e-3).all()) and bool((fruit >= -1e-6).all()),
             "fruit field within [0, fruit_capacity]",
             f"max excess {float((fruit - fcap).max()):.4f}")

    # --- the world is actually habitable ------------------------------------
    # The measured version of the plant-cell rule in tier 1: not "the geometry
    # permits a river to register" but "this river system does register". A
    # world where no cell reads as drinkable kills everything by thirst, and it
    # would do so while every shape check above still passed.
    drinkable = int((np.asarray(terrain.water_dist) < cfg.river_half_width).sum())
    r.expect(drinkable > 0, f"drinkable cells exist ({drinkable} of {cfg.n_cells})",
             "no cell registers as water -- the rivers fell between cell centres")

    # --- deaths partition the toll ------------------------------------------
    # Every death has exactly one cause; a cause added to `reproduction.cull`
    # without being added to `Metrics` shows up here as a shortfall.
    stacked = {k: float(np.asarray(v).sum()) for k, v in ms._asdict().items()}
    toll = sum(stacked[f"death_{c}"] for c in
               ("predation", "starvation", "thirst", "senescence"))
    r.expect(toll > 0, f"deaths were recorded ({toll:.0f} over the run)",
             "no deaths at all in 200 steps -- cull may have stopped firing")

    # --- golden band ---------------------------------------------------------
    watched = {k: metrics[k] for k in GOLDEN_BANDS if k in metrics}
    if bless:
        GOLDEN.write_text(json.dumps(
            {"_comment": "Regenerate with `scripts/check.py --bless`. Do this only "
                         "when a change is *meant* to move these numbers, and say "
                         "so in the commit message.",
             "config": {"steps": SMOKE_STEPS, "seed": SMOKE_SEED, **SMOKE},
             "metrics": watched}, indent=2) + "\n")
        print(f"  blessed {GOLDEN.relative_to(ROOT)} with {len(watched)} metrics")
        return

    if not GOLDEN.exists():
        print(f"  skip golden band ({GOLDEN.relative_to(ROOT)} missing -- "
              f"run with --bless once to create it)")
        return

    g = json.loads(GOLDEN.read_text())
    if g.get("config") != {"steps": SMOKE_STEPS, "seed": SMOKE_SEED, **SMOKE}:
        r.fail("golden was recorded on this smoke config",
               f"golden: {g.get('config')}\n       now:    "
               f"{{'steps': {SMOKE_STEPS}, 'seed': {SMOKE_SEED}, **{SMOKE}}}")
        return
    drift = []
    for k, want in g["metrics"].items():
        got, band = watched.get(k), GOLDEN_BANDS[k]
        if got is None:
            continue
        span = max(abs(want) * band, 1e-6)
        if abs(got - want) > span:
            pct = 100 * (got - want) / want if want else float("inf")
            drift.append(f"{k}: {want:.4g} -> {got:.4g} ({pct:+.1f}%, band ±{100*band:.0f}%)")
    r.expect(not drift,
             f"golden band held for {len(g['metrics'])} metrics",
             "\n       ".join(drift) +
             "\n       If this change was *supposed* to move these, re-bless with "
             "`--bless` and say why in the commit message. Do NOT widen "
             "GOLDEN_BANDS to make it pass.")


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--contracts", action="store_true",
                    help="tier 1 only: no JAX, no world, well under a second")
    ap.add_argument("--full", action="store_true",
                    help="also run the whole pytest suite (minutes)")
    ap.add_argument("--bless", action="store_true",
                    help="rewrite scripts/golden.json from this run")
    args = ap.parse_args()

    r = Report()
    t0 = time.time()

    print("contracts")
    check_syntax(r)
    check_wire_protocol(r)
    check_agent_stride(r)
    check_ecotype_bins(r)
    check_species_colours(r)
    check_config_invariants(r)

    if not args.contracts:
        print("smoke world")
        run_smoke(r, args.bless)

    if args.full:
        print("pytest")
        rc = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT).returncode
        r.expect(rc == 0, "pytest suite", f"exit code {rc}")

    dt = time.time() - t0
    print()
    if r.failures:
        print(f"\033[31mFAILED\033[0m {len(r.failures)}/{r.checks} checks in {dt:.1f}s")
        for f in r.failures:
            print(f"  - {f.splitlines()[0]}")
        return 1
    print(f"\033[32mPASSED\033[0m {r.checks} checks in {dt:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
