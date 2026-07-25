"""腐食第二版（接 retina 主动觅食）ON/OFF 判决 —— 全 18 配对种子。
判据：docs/multispecies_feasibility.md §7 第二版。
读 outputs/20260725-carrion-v2/results.jsonl（36 行 = arm ON/OFF × seed 0..17）。
ON=carrion_enabled true, OFF=false（唯一差异，overrides 已校验）；同步数 20000。
输出：逐种子 18 行表 + 配对 Wilcoxon 双侧 p + rank-biserial + 10000 次 bootstrap 95% CI
      + min 抬高判断 + carrion_total 分布 + 护栏。所有数字来自 stdout，非手算。
重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python explorations/20260725-carrion-v2/analyze.py
"""
import json, statistics as st
import numpy as np
from scipy.stats import wilcoxon, rankdata

rows = [json.loads(l) for l in open('outputs/20260725-carrion-v2/results.jsonl')]
ON  = {r['seed']: r for r in rows if r['arm'] == 'ON'}
OFF = {r['seed']: r for r in rows if r['arm'] == 'OFF'}
seeds = sorted(ON)
N = len(seeds)
assert N == 18, N

def boot_ci_diff(diffs, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs, float)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)

def rank_biserial_paired(on, off):
    diffs = np.array(on, float) - np.array(off, float)
    nz = diffs[diffs != 0]
    if len(nz) == 0:
        return float('nan')
    ranks = rankdata(np.abs(nz))
    Rpos = ranks[nz > 0].sum(); Rneg = ranks[nz < 0].sum()
    return (Rpos - Rneg) / ranks.sum()

print("="*92)
print(f"全 {N} 配对种子   ON=carrion_enabled true  OFF=false   steps=20000")
print("="*92)
hdr = f"{'metric':22s} {'OFF均':>10s} {'ON均':>10s} {'diff':>10s} {'boot95CI':>22s} {'↑/N':>6s} {'p(双侧)':>9s} {'r_rb':>7s}"
print(hdr)
for k in ['carnivore_frac', 'late_carn', 'population', 'min_pop',
          'death_thirst_frac', 'death_predation_frac', 'death_starvation_frac',
          'carrion_total', 'mean_age']:
    o = [OFF[s][k] for s in seeds]
    n = [ON[s][k]  for s in seeds]
    diffs = [a-b for a,b in zip(n,o)]
    up = sum(1 for d in diffs if d > 0)
    try:
        w, p = wilcoxon(n, o)      # 双侧
        ps = f"{p:.4f}"
    except Exception as e:
        ps = f"[{e}]"
    lo, hi = boot_ci_diff(diffs)
    r = rank_biserial_paired(n, o)
    print(f"{k:22s} {st.mean(o):10.4f} {st.mean(n):10.4f} {st.mean(diffs):+10.4f} "
          f"[{lo:+.4f},{hi:+.4f}] {up:3d}/{N} {ps:>9s} {r:+7.3f}")

print("="*92)
print("主判据逐种子 carnivore_frac（末态）:")
print(f"{'seed':>5s} {'OFF':>9s} {'ON':>9s} {'diff':>9s} {'同向':>5s}")
o = [OFF[s]['carnivore_frac'] for s in seeds]
n = [ON[s]['carnivore_frac'] for s in seeds]
for s in seeds:
    d = ON[s]['carnivore_frac'] - OFF[s]['carnivore_frac']
    print(f"{s:5d} {OFF[s]['carnivore_frac']:9.4f} {ON[s]['carnivore_frac']:9.4f} {d:+9.4f} {'↑' if d>0 else ('↓' if d<0 else '='):>5s}")
print(f"  同向(ON>OFF): {sum(1 for a,b in zip(n,o) if a>b)}/{N}")
print(f"  min: OFF={min(o):.4f} -> ON={min(n):.4f}  ({'抬高' if min(n)>min(o) else '未抬/反降'})")
print(f"  种子间 SD: OFF={st.pstdev(o):.4f}  ON={st.pstdev(n):.4f}")

print("-"*92)
print("late_carn 逐种子同向数 & min:")
lo_o=[OFF[s]['late_carn'] for s in seeds]; lo_n=[ON[s]['late_carn'] for s in seeds]
print(f"  同向(ON>OFF): {sum(1 for a,b in zip(lo_n,lo_o) if a>b)}/{N}  "
      f"min OFF={min(lo_o):.4f} -> ON={min(lo_n):.4f} ({'抬高' if min(lo_n)>min(lo_o) else '未抬/反降'})")

print("="*92)
ct = [ON[s]['carrion_total'] for s in seeds]
print(f"carrion_total ON: mean={st.mean(ct):.1f} median={st.median(ct):.1f} "
      f"min={min(ct):.1f} max={max(ct):.1f}  (首版≈204, 第二版6种子≈198)")

print("="*92)
print("护栏 population 逐种子 diff(ON-OFF):")
pdiff=[ON[s]['population']-OFF[s]['population'] for s in seeds]
print("  " + " ".join(f"{d:+.0f}" for d in pdiff))
print(f"  负向(ON<OFF)种子数: {sum(1 for d in pdiff if d<0)}/{N}  均值 diff={st.mean(pdiff):+.1f}")
