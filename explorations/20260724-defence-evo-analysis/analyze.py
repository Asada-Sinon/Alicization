# Question: In the defence-evo ablation (armor/spike heritable ON vs OFF, 6 paired
# seeds, 20000 steps, same map), are armor & spike selected in the herbivore lineage
# (P1), do they reduce predation/carnivore_frac (P2), and are carn defence genes
# neutral (no ON/OFF diff) as a control (P3)?
#
# Reads: outputs/20260724-defence-evo/{on,off}_seed{0..5}.log  -- last line "JSON {...}"
# Outputs: prints per-seed table + paired Wilcoxon + bootstrap 95% CI + effect sizes.
# Every number here comes from the .log files; nothing is hand-computed.
import json, glob, os, numpy as np
from scipy.stats import wilcoxon

BASE = "/home/michael/workspace/pi05/temp/alicization/outputs/20260724-defence-evo"
FIELDS = ["herb_armor","herb_spike","carn_armor","carn_spike","mean_armor",
          "mean_spike","death_predation_frac","carnivore_frac","population","hunt_success"]

def load(arm, seed):
    path = f"{BASE}/{arm}_seed{seed}.log"
    line = [l for l in open(path) if l.startswith("JSON ")][-1]
    return json.loads(line[len("JSON "):])

on  = {s: load("on", s)  for s in range(6)}
off = {s: load("off", s) for s in range(6)}

def col(d, f): return np.array([d[s][f] for s in range(6)])

def boot_ci(diff, n=10000, seed=1234):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])

print("=== Per-seed values (ON / OFF / diff=ON-OFF) ===")
for f in FIELDS:
    o, fo = col(on, f), col(off, f)
    print(f"\n-- {f} --")
    for s in range(6):
        print(f"  seed{s}: ON={o[s]:.5f}  OFF={fo[s]:.5f}  diff={o[s]-fo[s]:+.5f}  {'ON>OFF' if o[s]>fo[s] else 'ON<=OFF'}")
    print(f"  mean ON={o.mean():.5f} OFF={fo.mean():.5f}  n(ON>OFF)={int((o>fo).sum())}/6")

print("\n\n=== Paired Wilcoxon signed-rank + effect + bootstrap 95% CI on paired diff ===")
def report(f, alt="two-sided"):
    o, fo = col(on, f), col(off, f)
    d = o - fo
    same = int((d>0).sum())
    # wilcoxon two-sided; guard zeros
    try:
        W, p = wilcoxon(o, fo, alternative="two-sided")
        Wp = W
    except Exception as e:
        Wp, p = float('nan'), float('nan')
    ci = boot_ci(d)
    # effect size: matched-pairs rank-biserial from wilcoxon; also median diff
    med = np.median(d)
    print(f"{f:22s} meandiff={d.mean():+.5f} med={med:+.5f} "
          f"ndir={same}/6  W={Wp} p={p:.4f}  boot95%CI=[{ci[0]:+.5f},{ci[1]:+.5f}]")
    return p

for f in FIELDS:
    report(f)

# rank-biserial effect size for the two P1 metrics
print("\n=== rank-biserial effect size (P1 metrics) ===")
for f in ["herb_armor","herb_spike"]:
    o, fo = col(on, f), col(off, f)
    d = o - fo
    r = np.abs(d)
    ranks = r.argsort().argsort()+1  # simple rank of |d|
    Rpos = ranks[d>0].sum(); Rneg = ranks[d<0].sum()
    T = Rpos+Rneg
    rb = (Rpos-Rneg)/T if T>0 else float('nan')
    print(f"{f}: rank-biserial={rb:+.3f}  (Rpos={Rpos}, Rneg={Rneg})")

# absolute levels of ON herb defence (peg-at-0 diagnostic)
print("\n=== ON herb absolute defence levels (peg-at-0 check) ===")
for f in ["herb_armor","herb_spike"]:
    o = col(on, f)
    print(f"{f} ON: min={o.min():.5f} mean={o.mean():.5f} max={o.max():.5f}")
