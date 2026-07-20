"""Fast-forward evolution with no visualization -- the FLA (fluctlight
acceleration) mode. Runs many steps via lax.scan, prints time-series metrics,
and checks for the M0 emergence signal: agents evolve to move toward food.

Usage:
    .venv/bin/python scripts/run_headless.py [total_steps] [chunk] [--seed N] [--json]
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


def main(total_steps: int = 4000, chunk: int = 200, seed: int = 0,
         as_json: bool = False) -> None:
    cfg = dataclasses.replace(Config(), seed=seed)
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

    first = report(chunk, ms, compile_t)
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
        carn.append(row["carnivore_frac"])
        pops.append(row["population"])
        if row["population"] < 1:
            print("!! population collapsed to zero")
            break

    # Emergence check: a stable ecosystem where herbivores AND carnivores coexist.
    late_carn = float(np.mean(carn[-max(1, len(carn) // 4):]))
    min_pop = float(np.min(pops))
    print(f"\nlate carnivore fraction={late_carn:.2f}  min population={min_pop:.0f}")
    if as_json:
        # One line per run so a multi-seed sweep can be aggregated without
        # re-parsing the table above.
        print("JSON " + json.dumps({"seed": seed, "steps": done,
                                    "late_carn": late_carn, "min_pop": min_pop,
                                    **row}))
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
    args = ap.parse_args()
    main(args.total_steps, args.chunk, args.seed, args.json)
