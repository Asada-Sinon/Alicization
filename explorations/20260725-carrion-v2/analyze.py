"""腐食第二版（接 retina 主动觅食）6 配对种子 ON/OFF 判决。
判据：docs/multispecies_feasibility.md §7 第二版（先写后跑）。
读 outputs/20260725-carrion-v2/results.jsonl（12 行，arm ON/OFF × seed 0..5）。
输出：逐种子表 + 配对 Wilcoxon + 效应量(rank-biserial) + bootstrap CI + min 抬高判断
      + carrion_total 对比首版 204 + 护栏。所有数字来自 stdout，非手算。
重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python explorations/20260725-carrion-v2/analyze.py
"""
import json, statistics as st
import numpy as np
from scipy.stats import wilcoxon

rows = [json.loads(l) for l in open('outputs/20260725-carrion-v2/results.jsonl')]
ON  = {r['seed']: r for r in rows if r['arm'] == 'ON'}
OFF = {r['seed']: r for r in rows if r['arm'] == 'OFF'}
seeds = sorted(ON)

def boot_ci(d, n=20000, seed=42):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)

def rank_biserial(on, off):
    # 配对：r = (正号秩和-负号秩和)/总秩和；等价用 Wilcoxon 的 T
    diffs = np.array(on) - np.array(off)
    nz = diffs[diffs != 0]
    ranks = st_rankdata(np.abs(nz))
    Rpos = ranks[nz > 0].sum(); Rneg = ranks[nz < 0].sum()
    return (Rpos - Rneg) / ranks.sum()

def st_rankdata(a):
    from scipy.stats import rankdata
    return rankdata(a)

print("="*80)
for k in ['carnivore_frac', 'late_carn', 'population', 'death_thirst_frac',
          'carrion_total', 'min_pop', 'mean_age']:
    o = [OFF[s][k] for s in seeds]
    n = [ON[s][k]  for s in seeds]
    diffs = [a-b for a,b in zip(n,o)]
    same = sum(1 for d in diffs if d > 0)
    try:
        w, p = wilcoxon(n, o)
        ps = f"W={w:.1f} p={p:.4f}"
    except Exception as e:
        ps = f"[{e}]"
    lo, hi = boot_ci(diffs)
    try:
        r = rank_biserial(n, o)
    except Exception:
        r = float('nan')
    print(f"{k:20s} OFF {st.mean(o):9.4f}  ON {st.mean(n):9.4f}  "
          f"diff {st.mean(diffs):+8.4f}  CI[{lo:+.4f},{hi:+.4f}]  "
          f"{same}/6↑  {ps}  r={r:+.3f}")

print("="*80)
# 主判据逐种子
for k in ['carnivore_frac', 'late_carn']:
    o = [OFF[s][k] for s in seeds]; n = [ON[s][k] for s in seeds]
    print(f"\n[{k}] 逐种子:")
    print("  seed:  " + " ".join(f"{s:7d}" for s in seeds))
    print("  OFF :  " + " ".join(f"{x:7.4f}" for x in o))
    print("  ON  :  " + " ".join(f"{x:7.4f}" for x in n))
    print("  diff:  " + " ".join(f"{a-b:+7.4f}" for a,b in zip(n,o)))
    print(f"  min OFF = {min(o):.4f}  ->  min ON = {min(n):.4f}   "
          f"({'抬高' if min(n)>min(o) else '未抬/反降'})")
    print(f"  种子间 SD: OFF {st.pstdev(o):.4f}  ON {st.pstdev(n):.4f}")

print("="*80)
# carrion_total 对比首版 204
ct = [ON[s]['carrion_total'] for s in seeds]
print(f"\ncarrion_total ON 逐种子: {[round(x,1) for x in ct]}")
print(f"carrion_total ON 均值 = {st.mean(ct):.1f}   (首版 204)   "
      f"{'更低(更快消费)' if st.mean(ct)<204 else '更高(更积压)'}")
