"""Fast-forward evolution with no visualization -- the FLA (fluctlight
acceleration) mode. Runs many steps via lax.scan, prints time-series metrics,
and checks for the M0 emergence signal: agents evolve to move toward food.

Usage:
    .venv/bin/python scripts/run_headless.py [total_steps] [chunk] [--seed N] [--json]
                                             [--set FIELD=VALUE ...]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time

# Allow running from the repo root without installing the package.
sys.path.insert(0, ".")

import jax
import numpy as np

from underworld import Config, new_world


def parse_overrides(pairs) -> dict:
    """Turn `--set name=value` strings into typed kwargs for `dataclasses.replace`.

    Ablations are how most claims here get falsified ("is the fruit layer doing
    anything?" is answered by a `fruit_max=0` arm, not by reading the code), so
    the control arm has to be reachable without editing `config.py` -- editing it
    would make the two arms two different working trees. Values are coerced to
    the field's declared type so `--set fruit_max=0` gives a float and
    `--set n_init=200` an int, not strings that silently poison the jit.
    """
    types = {f.name: f.type for f in dataclasses.fields(Config)}
    out = {}
    for p in pairs or ():
        name, _, raw = p.partition("=")
        if name not in types:
            raise SystemExit(f"--set: no Config field named {name!r}")
        t = types[name]
        # Annotations may be strings under `from __future__ import annotations`.
        t = {"int": int, "float": float, "bool": bool}.get(t, t) if isinstance(t, str) else t
        out[name] = (raw not in ("0", "False", "false")) if t is bool else t(raw)
    return out


def main(total_steps: int = 4000, chunk: int = 200, seed: int = 0,
         as_json: bool = False, overrides: dict | None = None) -> None:
    cfg = dataclasses.replace(Config(), seed=seed, **(overrides or {}))
    print(f"device: {jax.devices()[0]}")
    print(f"genome_size={cfg.genome_size}  n_max={cfg.n_max}  n_cells={cfg.n_cells}"
          f"  seed={cfg.seed}")

    state, key, _step, scan_fn, _terrain = new_world(cfg)

    # Warm up / compile on a single chunk and time it.
    t0 = time.time()
    state, key, ms = scan_fn(state, key, chunk)
    jax.block_until_ready(ms)
    compile_t = time.time() - t0
    print(f"first chunk (compile+run {chunk} steps): {compile_t:.2f}s")

    print(f"\n{'step':>7} {'pop':>7} {'energy':>8} {'water':>7} {'age':>7} "
          f"{'plant':>9} {'diet':>6} {'dietSD':>7} {'carn%':>6} "
          f"{'carnV':>6} {'herbV':>6} {'hWD':>6} {'cWD':>6} {'wBnd':>6} "
          f"{'inl':>6} {'steps/s':>9}")

    def report(step, ms, dt):
        # ms fields are stacked over the chunk; show the last value. Read by
        # name -- appending a metric must never shift an existing column.
        r = {k: float(np.asarray(v)[-1]) for k, v in ms._asdict().items()}
        sps = chunk / dt if dt > 0 else float("nan")
        print(f"{step:>7} {r['population']:>7.0f} {r['mean_energy']:>8.2f} "
              f"{r['mean_water']:>7.2f} {r['mean_age']:>7.1f} "
              f"{r['plant_total']:>9.0f} {r['mean_diet']:>6.2f} "
              f"{r['diet_std']:>7.3f} {100 * r['carnivore_frac']:>5.1f}% "
              f"{r['carn_speed']:>6.2f} {r['herb_speed']:>6.2f} "
              f"{r['herb_water_dist']:>6.1f} {r['carn_water_dist']:>6.1f} "
              f"{r['water_bound_frac']:>6.2f} {r['inland_frac']:>6.2f} {sps:>9.0f}")
        return r

    # Death counts are per-step and must be *summed* over the whole run, not
    # sampled at the chunk boundary like the state metrics above. Mean age at
    # death is likewise a ratio of two run-long sums, not a mean of per-step
    # means -- the latter would let a step with one death outvote a step with
    # two hundred.
    CAUSES = ("predation", "starvation", "thirst", "senescence")
    ACC = tuple(f"death_{c}" for c in CAUSES) + tuple(f"deathage_{c}" for c in CAUSES)
    toll = {k: 0.0 for k in ACC}

    def tally(ms):
        for k in ACC:
            toll[k] += float(np.asarray(getattr(ms, k)).sum())

    first = report(chunk, ms, compile_t)
    tally(ms)
    carn = [first["carnivore_frac"]]
    pops = [first["population"]]
    row = first

    done = chunk
    while done < total_steps:
        t0 = time.time()
        state, key, ms = scan_fn(state, key, chunk)
        jax.block_until_ready(ms)
        dt = time.time() - t0
        done += chunk
        row = report(done, ms, dt)
        tally(ms)
        carn.append(row["carnivore_frac"])
        pops.append(row["population"])
        if row["population"] < 1:
            print("!! population collapsed to zero")
            break

    total_deaths = max(sum(toll[f"death_{c}"] for c in CAUSES), 1.0)
    # Suffixed so these never collide with the same-named per-step counts that
    # `row` carries into the JSON line below.
    summary = {}
    for c in CAUSES:
        n = toll[f"death_{c}"]
        summary[f"death_{c}_frac"] = n / total_deaths
        summary[f"death_{c}_age"] = toll[f"deathage_{c}"] / max(n, 1.0)
    print(f"\ndeaths by cause (whole run, n={total_deaths:.0f}), "
          f"share and mean age at death:")
    for c in CAUSES:
        print(f"  {c:<11} {100 * summary[f'death_{c}_frac']:>5.1f}%   "
              f"mean age {summary[f'death_{c}_age']:>7.1f} steps   "
              f"(n={toll[f'death_{c}']:.0f})")

    # Emergence check: a stable ecosystem where herbivores AND carnivores coexist.
    late_carn = float(np.mean(carn[-max(1, len(carn) // 4):]))
    min_pop = float(np.min(pops))
    print(f"\nlate carnivore fraction={late_carn:.2f}  min population={min_pop:.0f}")
    if as_json:
        # One line per run so a multi-seed sweep can be aggregated without
        # re-parsing the table above.
        print("JSON " + json.dumps({"seed": seed, "steps": done,
                                    "late_carn": late_carn, "min_pop": min_pop,
                                    **row, **summary,
                                    "total_deaths": total_deaths,
                                    "overrides": overrides or {}}))
    coexist = 0.05 < late_carn < 0.95
    survived = min_pop >= 1
    if coexist and survived:
        print("PASS: stable predator/herbivore coexistence emerged.")
    elif survived:
        print(f"(survived but not a clear split: carnivores={late_carn:.0%} "
              f"-- tune diet/predation params)")
    else:
        print("(population went extinct -- tune params)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("total_steps", nargs="?", type=int, default=4000)
    ap.add_argument("chunk", nargs="?", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0,
                    help="cfg.seed -- reseeds the founder population; the "
                         "terrain is derived from cfg alone and does not move")
    ap.add_argument("--json", action="store_true",
                    help="emit a final JSON line for multi-seed aggregation")
    ap.add_argument("--set", action="append", metavar="FIELD=VALUE", dest="sets",
                    help="override a Config field, e.g. --set fruit_max=0. "
                         "Repeatable. For ablation arms -- keeps both arms on "
                         "one working tree")
    args = ap.parse_args()
    main(args.total_steps, args.chunk, args.seed, args.json,
         parse_overrides(args.sets))
