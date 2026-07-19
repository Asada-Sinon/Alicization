"""Fast-forward evolution with no visualization -- the FLA (fluctlight
acceleration) mode. Runs many steps via lax.scan, prints time-series metrics,
and checks for the M0 emergence signal: agents evolve to move toward food.

Usage:
    .venv/bin/python scripts/run_headless.py [total_steps] [chunk]
"""

from __future__ import annotations

import sys
import time

# Allow running from the repo root without installing the package.
sys.path.insert(0, ".")

import jax
import numpy as np

from underworld import Config, new_world


def main(total_steps: int = 4000, chunk: int = 200) -> None:
    cfg = Config()
    print(f"device: {jax.devices()[0]}")
    print(f"genome_size={cfg.genome_size}  n_max={cfg.n_max}  n_cells={cfg.n_cells}")

    state, key, _step, scan_fn, _terrain = new_world(cfg)

    # Warm up / compile on a single chunk and time it.
    t0 = time.time()
    state, key, ms = scan_fn(state, key, chunk)
    jax.block_until_ready(ms)
    compile_t = time.time() - t0
    print(f"first chunk (compile+run {chunk} steps): {compile_t:.2f}s")

    print(f"\n{'step':>7} {'pop':>7} {'energy':>8} {'water':>7} {'age':>7} "
          f"{'plant':>9} {'diet':>6} {'dietSD':>7} {'carn%':>6} "
          f"{'carnV':>6} {'herbV':>6} {'steps/s':>9}")

    def report(step, ms, dt):
        # ms fields are stacked over the chunk; show the last value.
        row = tuple(float(np.asarray(x)[-1]) for x in ms)
        sps = chunk / dt if dt > 0 else float("nan")
        print(f"{step:>7} {row[0]:>7.0f} {row[1]:>8.2f} {row[9]:>7.2f} {row[2]:>7.1f} "
              f"{row[3]:>9.0f} {row[4]:>6.2f} {row[6]:>7.3f} {100 * row[5]:>5.1f}% "
              f"{row[7]:>6.2f} {row[8]:>6.2f} {sps:>9.0f}")
        return row

    first = report(chunk, ms, compile_t)
    carn = [first[5]]
    pops = [first[0]]

    done = chunk
    while done < total_steps:
        t0 = time.time()
        state, key, ms = scan_fn(state, key, chunk)
        jax.block_until_ready(ms)
        dt = time.time() - t0
        done += chunk
        row = report(done, ms, dt)
        carn.append(row[5])
        pops.append(row[0])
        if row[0] < 1:
            print("!! population collapsed to zero")
            break

    # Emergence check: a stable ecosystem where herbivores AND carnivores coexist.
    late_carn = float(np.mean(carn[-max(1, len(carn) // 4):]))
    min_pop = float(np.min(pops))
    print(f"\nlate carnivore fraction={late_carn:.2f}  min population={min_pop:.0f}")
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
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    main(total, chunk)
