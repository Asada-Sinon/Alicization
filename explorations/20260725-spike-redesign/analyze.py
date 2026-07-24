# Q: Did the spike redesign (offense for carn, venom-defense for herb) revive spike?
# Reads: outputs/20260724-spike-redesign/{on,off}_seed{0..5}.log, last line starting "JSON ".
# ON = default (spike_heritable=True); OFF = spike_heritable=False (armor unchanged).
# Paired by seed, n=6, 20000 steps, same map. Prints per-seed tables, paired Wilcoxon,
# rank-biserial effect size, and 10000-rep bootstrap 95% CI on the paired difference.
import json, glob, os
import numpy as np
from scipy.stats import wilcoxon

D = "outputs/20260724-spike-redesign"
def load(arm, s):
    path = f"{D}/{arm}_seed{s}.log"
    line = [l for l in open(path) if l.startswith("JSON ")][-1]
    d = json.loads(line[len("JSON "):])
    d["_path"] = path
    return d

seeds = list(range(6))
ON = {s: load("on", s) for s in seeds}
OFF = {s: load("off", s) for s in seeds}

# sanity: overrides
print("=== overrides sanity ===")
for s in seeds:
    print(f"seed{s}: ON.overrides={ON[s]['overrides']}  OFF.overrides={OFF[s]['overrides']}  ON.steps={ON[s]['steps']} OFF.steps={OFF[s]['steps']}")

fields = ["carn_spike","herb_spike","mean_spike","mean_venom","carnivore_frac",
          "population","herb_armor","carn_armor","death_predation_frac","hunt_success"]

print("\n=== per-seed all fields ===")
hdr = "seed | arm | " + " | ".join(fields)
print(hdr)
for s in seeds:
    for arm,dd in (("ON",ON),("OFF",OFF)):
        vals = " | ".join(f"{dd[s][f]:.5f}" for f in fields)
        print(f"{s} | {arm} | {vals}")

def boot_ci(diffs, reps=10000, seed=42):
    rng = np.random.default_rng(seed)
    n=len(diffs)
    means=[np.mean(rng.choice(diffs,size=n,replace=True)) for _ in range(reps)]
    return np.percentile(means,2.5), np.percentile(means,97.5)

def rank_biserial(a,b):
    # paired: proportion favorable minus unfavorable among nonzero diffs
    d=np.array(a)-np.array(b)
    nz=d[d!=0]
    if len(nz)==0: return 0.0
    from scipy.stats import rankdata
    r=rankdata(np.abs(nz))
    pos=r[nz>0].sum(); neg=r[nz<0].sum()
    T=pos+neg
    return (pos-neg)/T

print("\n=== paired tests (ON vs OFF) ===")
for f in fields:
    a=np.array([ON[s][f] for s in seeds])
    b=np.array([OFF[s][f] for s in seeds])
    diff=a-b
    same_dir=int(np.sum(np.sign(diff)==np.sign(np.mean(diff)))) if np.mean(diff)!=0 else 0
    npos=int(np.sum(diff>0)); nneg=int(np.sum(diff<0)); nzero=int(np.sum(diff==0))
    try:
        w,p=wilcoxon(a,b)
        wstr=f"W={w:.1f} p={p:.4f}"
    except Exception as e:
        wstr=f"wilcoxon-err({e})"
    rb=rank_biserial(a,b)
    lo,hi=boot_ci(diff)
    print(f"{f:22s} ON_mean={a.mean():.5f} OFF_mean={b.mean():.5f} diff_mean={diff.mean():+.5f} "
          f"pos/neg/zero={npos}/{nneg}/{nzero} {wstr} rb={rb:+.3f} bootCI=[{lo:+.5f},{hi:+.5f}]")
