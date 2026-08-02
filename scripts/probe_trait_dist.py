"""Trait-gene DISTRIBUTION probe: is a trait splitting the population in two, or
just spreading out?

`Metrics` only carries the mean and the standard deviation of each trait gene, and
those two numbers cannot tell character displacement from plain drift -- a
population that splits into two groups and a population that merely gets noisier
both raise the SD. The falsifiable prediction for resource partitioning
(docs/multispecies_feasibility.md §9, docs/multispecies_program.md S2) is about the
SHAPE of the distribution, so the shape has to leave the kernel.

    .venv/bin/python scripts/probe_trait_dist.py [steps] --seed N \
        --trait forage_pref [--lineage herb] [--set FIELD=VALUE ...] [--dump PATH]

Emits a `JSON` line (same convention as run_headless) plus, with --dump, the raw
per-agent values as .npy so any other test can be run on them later.

## Why a bootstrap likelihood-ratio test and not Hartigan's dip

The pre-registered plan said "dip test". This environment has no `diptest` package,
and hand-porting Hartigan's AS 217 (the alternating GCM/LCM fit) is exactly the kind
of change that fails silently: a subtly wrong port returns a plausible number, not an
error. So the shape test here is a parametric bootstrap likelihood-ratio test of one
Gaussian component against two (McLachlan 1987) -- every piece of it (the EM fit, the
LR statistic, the bootstrap null) is checkable against synthetic data with known
ground truth, and `tests/test_kernel.py` does check it.

It is also the better-matched test for this particular question. Dip is a general
test against ANY departure from unimodality; the alternative we actually care about
is specifically "two groups", which is what a two-component mixture models. Reported
alongside is the bimodality coefficient, a cheap closed-form descriptive.

## The absolute p-value here is INVALID. Use the paired between-arm comparison.

Every shape test above assumes iid draws. An evolved population is not a sample --
it is a genealogy, and kin share gene values, so the distribution is lumpy with
family clusters whatever the ecology is doing. The effective sample size is the
number of independent lineages, not the number of agents, and with n~2000 agents the
bootstrap null (which resamples iid) is far too narrow.

Measured, not argued (2000 steps, seed 0, herb carriers):

    forage_tradeoff=0.5 (gene ACTIVE)   LR=212.8  n=2107  p=0.0200 (floor)
    forage_tradeoff=0.0 (gene INERT)    LR= 65.9  n=1666  p=0.0200 (floor)

The neutral control -- where the gene is compile-time disconnected from the world and
can do literally nothing -- also rejects unimodality at the floor. So `blrt_p` is
reported for completeness but must never be read as "this arm is bimodal".

What IS usable is `blrt_lr_per_n` compared BETWEEN PAIRED ARMS: kin structure inflates
both arms, so the difference still carries signal (0.101 vs 0.040 above). LR is divided
by n because 2*(ll2-ll1) grows with sample size and the arms have different carrier
counts -- comparing raw LR across arms would be comparing population sizes.

THIS IS A MEASUREMENT TOOL, not a conclusion. One seed decides nothing here either
(conventions.md §5) -- run it over the 6-seed paired protocol and compare arms.
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
from underworld import state as state_mod
from scripts.run_headless import parse_overrides


# Trait name -> the decoder in underworld/state.py. Reading the raw genome column
# would be wrong: every trait goes through its own squashing function (forage_pref
# is a two-sided sigmoid, escape/armor/spike are one-sided), and the shape of the
# distribution is not preserved by those maps.
TRAITS = {
    "diet": "diet_of", "invest": "invest_of", "size": "size_of",
    "attack": "attack_range_of", "escape": "escape_of",
    "armor": "armor_of", "spike": "spike_of", "forage_pref": "forage_pref_of",
}


# --- shape statistics ------------------------------------------------------

def bimodality_coefficient(x: np.ndarray) -> float:
    """BC = (skew^2 + 1) / (kurt + 3(n-1)^2/((n-2)(n-3))). Above ~0.555 (the value
    for a uniform distribution) is conventionally read as "leaning bimodal". Cheap
    and closed-form, but it is only a descriptive -- it has no null distribution
    attached, which is why the BLRT below is the actual test."""
    n = len(x)
    if n < 4:
        return float("nan")
    m = x.mean()
    s = x.std()
    if s <= 0:
        return float("nan")
    z = (x - m) / s
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4)) - 3.0            # excess kurtosis
    corr = 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return (skew ** 2 + 1.0) / (kurt + corr)


def _normal_loglik(x: np.ndarray) -> float:
    """Log-likelihood of the single-Gaussian MLE fit."""
    n = len(x)
    var = max(float(np.var(x)), 1e-12)
    return -0.5 * n * (np.log(2 * np.pi * var) + 1.0)


def _gmm2_loglik(x: np.ndarray, iters: int = 300, restarts: int = 5,
                 rng: np.random.Generator | None = None) -> float:
    """Log-likelihood of the best two-component 1-D Gaussian mixture found by EM.

    Multiple restarts because EM is local: a single median split converges to a
    trivial "one wide component plus one spike" solution often enough to matter.
    Variances are floored -- an unfloored 1-D mixture likelihood is unbounded (a
    component collapsing onto a single point sends it to +inf), which would make the
    LR statistic meaningless rather than merely noisy.
    """
    rng = rng or np.random.default_rng(0)
    n = len(x)
    spread = max(float(np.std(x)), 1e-9)
    vfloor = (spread * 1e-3) ** 2
    best = -np.inf
    for r in range(restarts):
        if r == 0:
            mu = np.array([np.quantile(x, 0.25), np.quantile(x, 0.75)])
        else:
            mu = rng.choice(x, size=2, replace=False).astype(float)
            if mu[0] == mu[1]:
                mu = mu + np.array([-spread, spread]) * 0.5
        var = np.full(2, max(spread ** 2, vfloor))
        w = np.array([0.5, 0.5])
        ll = -np.inf
        for _ in range(iters):
            # E step, in log space so a far-out point cannot underflow both
            # components to zero and produce a 0/0 responsibility.
            lp = (np.log(np.maximum(w, 1e-300))[None, :]
                  - 0.5 * np.log(2 * np.pi * var)[None, :]
                  - 0.5 * (x[:, None] - mu[None, :]) ** 2 / var[None, :])
            mx = lp.max(axis=1, keepdims=True)
            lse = mx[:, 0] + np.log(np.exp(lp - mx).sum(axis=1))
            new_ll = float(lse.sum())
            resp = np.exp(lp - lse[:, None])
            # M step
            nk = resp.sum(axis=0)
            if np.any(nk < 1e-9):
                break
            w = nk / n
            mu = (resp * x[:, None]).sum(axis=0) / nk
            var = np.maximum((resp * (x[:, None] - mu[None, :]) ** 2).sum(axis=0) / nk,
                             vfloor)
            if new_ll - ll < 1e-9:
                ll = new_ll
                break
            ll = new_ll
        best = max(best, ll)
    return float(best)


def blrt_two_components(x: np.ndarray, n_boot: int = 199, seed: int = 0) -> dict:
    """Parametric bootstrap likelihood-ratio test: one Gaussian vs two.

    The LR statistic 2*(ll2 - ll1) does NOT have a chi-square null here -- the
    one-component model sits on the boundary of the two-component parameter space,
    so the regularity conditions fail. The null is therefore simulated: resample
    from the fitted single Gaussian and recompute the statistic (McLachlan 1987).

    p is the standard (1 + #{boot >= observed}) / (n_boot + 1) so it can never be
    exactly 0 -- with n_boot=199 the floor is p=0.005.

    READ THE MODULE DOCSTRING BEFORE USING `p`: on an evolved population it is
    invalid (kin structure breaks iid and the inert-gene control also floors it).
    `lr_per_n` compared between paired arms is the statistic that carries signal.
    """
    rng = np.random.default_rng(seed)
    ll1 = _normal_loglik(x)
    ll2 = _gmm2_loglik(x, rng=rng)
    lr = 2.0 * (ll2 - ll1)
    mu, sd = float(np.mean(x)), float(np.std(x))
    n = len(x)
    ge = 0
    for _ in range(n_boot):
        xb = rng.normal(mu, max(sd, 1e-12), size=n)
        lrb = 2.0 * (_gmm2_loglik(xb, rng=rng) - _normal_loglik(xb))
        ge += int(lrb >= lr)
    return {"lr": float(lr), "lr_per_n": float(lr) / max(len(x), 1),
            "p": (1.0 + ge) / (n_boot + 1.0),
            "n_boot": n_boot, "ll1": ll1, "ll2": ll2}


# --- the probe -------------------------------------------------------------

def main(steps: int = 20000, seed: int = 0, trait: str = "forage_pref",
         lineage: str = "herb", overrides: dict | None = None,
         dump: str | None = None, n_boot: int = 199, chunk: int = 500) -> None:
    if trait not in TRAITS:
        raise SystemExit(f"--trait must be one of {sorted(TRAITS)}")
    cfg = dataclasses.replace(Config(), seed=seed, **(overrides or {}))
    print(f"device: {jax.devices()[0]}  seed={seed}  steps={steps}  "
          f"trait={trait}  lineage={lineage}")

    state, key, _step, scan_fn, _terrain = new_world(cfg)
    done = 0
    while done < steps:
        take = min(chunk, steps - done)
        state, key, _ms = scan_fn(state, key, take)
        done += take
    jax.block_until_ready(state.genome)

    decode = getattr(state_mod, TRAITS[trait])
    values = np.asarray(decode(state.genome, cfg))
    alive = np.asarray(state.alive)
    diet = np.asarray(state.diet)
    # The functional carriers only. A gene that is diet-gated (or simply not used by
    # carnivores) looks flat if the whole population is pooled -- `herb_forage_pref`
    # exists in Metrics for the same reason.
    mask = alive.copy()
    if lineage == "herb":
        mask &= diet < 0.35
    elif lineage == "carn":
        mask &= diet > 0.65
    elif lineage != "all":
        raise SystemExit("--lineage must be herb, carn or all")
    x = values[mask].astype(np.float64)

    print(f"n_alive={int(alive.sum())}  n_in_lineage={len(x)}")
    if len(x) < 30:
        print("!! too few carriers for a shape test")
        print("JSON " + json.dumps({"seed": seed, "steps": steps, "trait": trait,
                                    "lineage": lineage, "n": len(x),
                                    "overrides": overrides or {}}))
        return

    bc = bimodality_coefficient(x)
    test = blrt_two_components(x, n_boot=n_boot, seed=seed)
    hist, edges = np.histogram(x, bins=40, range=(0.0, 1.0))

    print(f"\n{trait} over {len(x)} {lineage} carriers:")
    print(f"  mean={x.mean():.4f}  sd={x.std():.4f}  "
          f"min={x.min():.4f}  max={x.max():.4f}")
    print(f"  bimodality coefficient = {bc:.4f}   (>0.555 leans bimodal)")
    print(f"  BLRT 1-vs-2 components: LR={test['lr']:.2f}  "
          f"LR/n={test['lr_per_n']:.5f}   <-- COMPARE THIS BETWEEN PAIRED ARMS")
    print(f"  blrt_p={test['p']:.4f} (n_boot={test['n_boot']}, floor "
          f"{1.0/(n_boot+1):.4f})  !! INVALID as an absolute test: kin structure "
          f"breaks iid; the inert-gene control floors it too. See module docstring.")
    print("  histogram (40 bins over [0,1]):")
    print("   " + " ".join(str(int(h)) for h in hist))

    if dump:
        np.save(dump, x)
        print(f"  raw carrier values -> {dump}  ({len(x)} values)")

    print("JSON " + json.dumps({
        "seed": seed, "steps": steps, "trait": trait, "lineage": lineage,
        "n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std()),
        "bimodality_coefficient": float(bc),
        "blrt_lr": test["lr"], "blrt_lr_per_n": test["lr_per_n"],
        "blrt_p": test["p"], "blrt_n_boot": test["n_boot"],
        "hist": [int(h) for h in hist], "hist_lo": 0.0, "hist_hi": 1.0,
        "overrides": overrides or {}}))
    print("\n[PROBE -- one seed decides nothing (conventions.md §5). Compare arms "
          "over the 6-seed paired protocol.]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="?", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trait", default="forage_pref", choices=sorted(TRAITS))
    ap.add_argument("--lineage", default="herb", choices=("herb", "carn", "all"))
    ap.add_argument("--dump", default=None, metavar="PATH.npy",
                    help="also write the raw per-carrier values for later re-analysis")
    ap.add_argument("--n-boot", type=int, default=199,
                    help="bootstrap replicates for the BLRT null (p floor = 1/(n+1))")
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--set", action="append", metavar="FIELD=VALUE", dest="sets")
    args = ap.parse_args()
    main(args.steps, args.seed, args.trait, args.lineage,
         parse_overrides(args.sets), args.dump, args.n_boot, args.chunk)
