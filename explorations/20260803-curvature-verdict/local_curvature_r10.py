"""R10 的最后两个检查：手术**即使完美交付**够不够翻号，以及全局 vs 局部曲率。

回答什么
--------
1. `W''_增量 = (1−k)·(A+B)`（对 c(s) 在 s=0 展开得到，`_forage_pref_scale` 的 Levins
   前沿）。把它跟**实测的 K10 基线曲率**比：若增量 < |基线|，那么即使手术 100% 交付，
   K05 的实现曲面仍然是凹的 —— 分歧选择在设计上就不可能出现。
2. `quad_intake` 是**全局**二次拟合，而分歧选择只关心**种群均值处的局部曲率**。
   实测曲线是「左侧陡升 + 右侧平台」的饱和形，全局二次拟合对这种形状**必然**读负，
   与均值处的局部曲率无关。分别拟合全域与均值 ±1.5 SD 的窗口，看两者差多少。

读哪些文件
----------
outputs/20260803-curvature/r10/*.log 的 `JSON ` 行（bin_centers / bin_n / intake）。

输出怎么读
----------
第 1 节最后一行是判决要用的数；第 2 节比较「全域二次项」与「局部二次项」。
"""
import math
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, "/home/xrl/intern/Alicization")
sys.path.insert(0, "/home/xrl/intern/Alicization/scripts")
from exp_stats import RunSet, paired

DIR = "/home/xrl/intern/Alicization/outputs/20260803-curvature/r10"
ARMS = ["N", "K10", "K05", "K035"]
SEEDS, REPS = list(range(12)), (1, 2)
KOF = {"N": 1.0, "K10": 1.0, "K05": 0.5, "K035": 0.35}
rs = RunSet.load(DIR, ARMS, SEEDS, REPS)
assert not rs.problems, rs.problems

print("=" * 92)
print("1. 即使 100% 交付，(1−k)(A+B) 够不够盖过实测基线曲率？")
print("=" * 92)


def raw_curv(rec):
    ctr, w = np.array(rec["bin_centers"]), np.array(rec["bin_n"], float)
    mu = (ctr * w).sum() / w.sum()
    sd = math.sqrt(((ctr - mu) ** 2 * w).sum() / w.sum())
    return rec["quad_intake"] / (2.0 * sd * sd)


def AB(rec):
    n = max(rec["population"] * (1 - rec["carnivore_frac"]), 1.0)
    return rec["graze_gain"] / n, rec["fruit_gain"] / n


base = rs.cell_means("K10", raw_curv)
print(f"  实测基线 W''(s) @ k=1（§11.1 要求恰为 0）= {base.mean():+.4f} "
      f"± {base.std(ddof=1)/math.sqrt(12):.4f}  (t={stats.ttest_1samp(base,0)[0]:+.2f}, "
      f"p={stats.ttest_1samp(base,0)[1]:.4f})")
for a in ["K05", "K035"]:
    ab = np.array([AB(rs.records[(a, s, r)]) for s in SEEDS for r in REPS])
    inc = (1 - KOF[a]) * ab.sum(axis=1)
    obs = rs.cell_means(a, raw_curv)
    print(f"  {a:>5} k={KOF[a]}: A+B={ab.sum(axis=1).mean():.4f}  "
          f"解析增量 (1−k)(A+B) = {inc.mean():+.4f}")
    print(f"          完美交付则 W'' = 基线 {base.mean():+.4f} + {inc.mean():+.4f} "
          f"= {base.mean()+inc.mean():+.4f}  ⇒ "
          f"{'**仍为凹（不可能分歧）**' if base.mean()+inc.mean() < 0 else '可翻正'}")
    print(f"          实测 W'' = {obs.mean():+.4f}；实测−基线 = {obs.mean()-base.mean():+.4f} "
          f"vs 解析增量 {inc.mean():+.4f}")

print()
print("=" * 92)
print("2. 全局二次拟合 vs 均值 ±1.5 SD 的局部二次拟合")
print("=" * 92)


def fit(rec, half=None):
    ctr, w = np.array(rec["bin_centers"]), np.array(rec["bin_n"], float)
    y = np.array([np.nan if v is None else v for v in rec["intake"]], float)
    mu = (ctr * w).sum() / w.sum()
    sd = math.sqrt(((ctr - mu) ** 2 * w).sum() / w.sum())
    m = np.isfinite(y) & (w > 0)
    if half is not None:
        m &= np.abs(ctr - mu) <= half * sd
    if m.sum() < 4:
        return np.nan, np.nan, 0
    z = (ctr[m] - mu) / sd
    sw = np.sqrt(w[m])
    X = np.stack([np.ones_like(z), z, z * z], 1) * sw[:, None]
    b = np.linalg.lstsq(X, y[m] * sw, rcond=None)[0]
    return float(b[1]), float(b[2]), int(m.sum())


for a in ARMS:
    for half, name in [(None, "全域"), (2.5, "±2.5SD"), (2.0, "±2.0SD"), (1.5, "±1.5SD")]:
        b1 = rs.cell_means(a, lambda r: fit(r, half)[0])
        b2 = rs.cell_means(a, lambda r: fit(r, half)[1])
        nb = rs.cell_means(a, lambda r: fit(r, half)[2])
        t1, p1 = stats.ttest_1samp(b1, 0.0)
        t2, p2 = stats.ttest_1samp(b2, 0.0)
        print(f"  {a:>5} {name:<7} 一阶={b1.mean():+.5f}(t={t1:+.2f},p={p1:.4f})  "
              f"二阶={b2.mean():+.5f}(t={t2:+.2f},p={p2:.4f})  箱={nb.mean():.1f}")
print()
print("  局部窗口的 K05−K10（配对，与 H1 同口径但只看均值邻域）:")
print(paired(rs, lambda r: fit(r, 2.5)[1], "K05", "K10", metric_name="局部二次项 ±2.5SD").format())

print()
print("=" * 92)
print("3. 要把 §14.4 的两道闸打到预测效应 +0.001839 上，需要多少 run")
print("=" * 92)
noise = rs.pair_noise("quad_intake")
sd_d = float((rs.cell_means("K05", "quad_intake") - rs.cell_means("K10", "quad_intake")).std(ddof=1))
pred = 0.001839
r_need = 2.0 * (noise / pred) ** 2
n_t = ((1.96 + 0.8416) * sd_d / pred) ** 2
print(f"  自估配对噪声 = {noise:.6f}（r=2）；实测配对差 SD = {sd_d:.6f}；预测效应 = {pred:.6f}")
print(f"  闸一「效应/噪声 ≥1」：噪声 ∝ 1/√r，需 r ≥ {r_need:.1f} 次重复/格 "
      f"⇒ 每臂 12×{math.ceil(r_need)} = {12*math.ceil(r_need)} run（**加种子救不了这道闸**）")
print(f"  闸二「p ≤ 0.05」（配对 t 近似，80% 功效）：需 s ≈ {n_t:.0f} 个种子")
print(f"  本轮 s=12, r=2 ⇒ 闸一实际能测到的最小效应 = {noise:.6f}（预测值的 "
      f"{noise/pred:.1f} 倍）")
