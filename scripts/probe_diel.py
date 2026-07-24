"""Phase-binned diel probe: does the day-night cycle make the SAME population sit
at a different distance from water by time of day? That intra-run day-vs-night gap
is the commuting signal a static `carn_water_dist` mean averages away (docs/
day_night.md §4, the disease docs/landscape_of_fear.md §6 named: a mean cannot tell
"shorter stay" from "never visits").

The scan already stacks every Metrics field per step; we keep the full
`carn_water_dist`/`herb_water_dist`/speed series plus `phase`, drop the early
transient, then split the remaining steps into a day-half (light>0.5, i.e. phase in
(0.25,0.75), midday-centred) and a night-half, and compare.

    .venv/bin/python scripts/probe_diel.py [steps] --seed N [--set FIELD=VALUE ...]

THIS IS A PROBE. Single-seed numbers here are for range-finding params only --
conventions.md forbids drawing an ecological conclusion from one seed. The 6-seed
paired protocol (docs/day_night.md §4) is what decides anything.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys

sys.path.insert(0, ".")

import jax
import numpy as np

from underworld import Config, new_world
from scripts.run_headless import parse_overrides


def main(steps: int = 10000, seed: int = 0, overrides: dict | None = None,
         warmup_frac: float = 0.5) -> None:
    cfg = dataclasses.replace(Config(), seed=seed, **(overrides or {}))
    if cfg.day_length <= 0:
        raise SystemExit("probe_diel needs day_length>0 (the whole point is the "
                         "diel split); pass --set day_length=400")
    print(f"device: {jax.devices()[0]}  seed={seed}  steps={steps}  "
          f"day_length={cfg.day_length}  heat_water_amp={cfg.heat_water_amp}  "
          f"night_vision_floor={cfg.night_vision_floor}")

    state, key, _step, scan_fn, _terrain = new_world(cfg)
    state, key, ms = scan_fn(state, key, steps)
    jax.block_until_ready(ms)

    d = {k: np.asarray(v) for k, v in ms._asdict().items()}
    phase = d["phase"]                              # [steps]
    light = 0.5 * (1.0 - np.cos(2.0 * np.pi * phase))

    # Drop the transient: the population equilibrates from n_init over the first
    # chunk, and its spatial distribution with it.
    lo = int(steps * warmup_frac)
    sl = slice(lo, steps)
    is_day = light[sl] > 0.5                        # midday-centred half
    is_night = ~is_day

    def split(field):
        v = d[field][sl]
        day = float(v[is_day].mean()) if is_day.any() else float("nan")
        night = float(v[is_night].mean()) if is_night.any() else float("nan")
        return day, night

    print(f"\nlate window: steps {lo}..{steps} "
          f"({is_day.sum()} day-half, {is_night.sum()} night-half)")
    print(f"{'metric':>18} {'day':>8} {'night':>8} {'night-day':>10}")
    out = {}
    for field in ("carn_water_dist", "herb_water_dist", "water_bound_frac",
                  "carn_speed", "herb_speed", "carnivore_frac", "population",
                  "hunt_success"):
        day, night = split(field)
        out[f"{field}_day"] = day
        out[f"{field}_night"] = night
        print(f"{field:>18} {day:>8.3f} {night:>8.3f} {night - day:>+10.3f}")

    # The headline: does the carnivore sit further from water in one half than the
    # other? Positive night-day = carnivores are FURTHER from water at night (they
    # commute inland in the dark). Negative = they hug the river more at night.
    cwd_gap = out["carn_water_dist_night"] - out["carn_water_dist_day"]
    print(f"\ncarn_water_dist night-day gap = {cwd_gap:+.3f}  "
          f"(the diel commuting signal; sign tells which half they leave the river)")

    # Hard-watch: the whole-run thirst mortality (docs/day_night.md §4). Heat that
    # is too strong kills far-from-water agents at midday -- which both (a) violates
    # the juvenile-thirst constraint and (b) manufactures a day/night distance gap by
    # CULLING rather than COMMUTING (watch it against the day/night population swing).
    causes = ("predation", "starvation", "thirst", "senescence")
    tolls = {c: float(d[f"death_{c}"].sum()) for c in causes}
    tot = max(sum(tolls.values()), 1.0)
    thirst_age = (float(d["deathage_thirst"].sum()) / max(tolls["thirst"], 1.0))
    pop_swing = out["population_night"] - out["population_day"]
    print(f"whole-run thirst deaths = {100*tolls['thirst']/tot:.1f}%  "
          f"(mean age {thirst_age:.1f})   |   day/night population swing = "
          f"{pop_swing:+.0f}  (large swing => distance gap may be culling, not commuting)")
    out["death_thirst_frac"] = tolls["thirst"] / tot
    out["death_thirst_age"] = thirst_age
    out["pop_swing_night_minus_day"] = pop_swing

    print("JSON " + json.dumps({"seed": seed, "steps": steps,
                                "day_length": cfg.day_length,
                                "cwd_night_minus_day": cwd_gap,
                                **out, "overrides": overrides or {}}))
    print("\n[PROBE -- single seed, range-finding only. No conclusion until the "
          "6-seed paired protocol, docs/day_night.md §4.]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="?", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup-frac", type=float, default=0.5,
                    help="fraction of steps dropped as transient before the split")
    ap.add_argument("--set", action="append", metavar="FIELD=VALUE", dest="sets")
    args = ap.parse_args()
    main(args.steps, args.seed, parse_overrides(args.sets), args.warmup_frac)
