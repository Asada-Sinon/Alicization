"""R10 的三个追加检查：预测值检验、曲面形状、估计量偏倚。

回答什么
--------
1. 实测的 `K05−K10` 与 §14.1 **自己预测的 +0.00184** 相差多少个噪声 SD？
   （原分析只测「对 0」，没测「对预测值」——那才是证伪该假设要的检验。）
2. 种群实际走到的 `s = 2·pref−1` 范围有多宽？曲率旋钮只在 |s|→1 附近才咬得动，
   若种群从不离开 s≈0 的邻域，前沿在实测范围内近似还是直线。
3. `quad_demog` 在中性臂 N（真值必须恰为 0，因为基因被编译期断开）读出多少？
   非 0 即为估计量偏倚。
4. 分箱摄入曲线的形状（平台+两端塌陷 = 供给饱和；光滑倒抛物线 = 别的）。

读哪些文件
----------
outputs/20260803-curvature/r10/*.log 的 `JSON ` 行。

输出怎么读
----------
四节，节标题即问题。第 2 节的「到 s=±1 要几个 SD」是关键数。
"""
import math
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, "/home/xrl/intern/Alicization")
sys.path.insert(0, "/home/xrl/intern/Alicization/scripts")
from exp_stats import RunSet, paired, one_sample

DIR = "/home/xrl/intern/Alicization/outputs/20260803-curvature/r10"
ARMS = ["N", "K10", "K05", "K035"]
SEEDS = list(range(12))
REPS = (1, 2)
rs = RunSet.load(DIR, ARMS, SEEDS, REPS)
assert not rs.problems, rs.problems

print("=" * 92)
print("1. 实测 K05−K10 对**假设自己的预测值** μ=+0.001839 的检验")
print("=" * 92)
d = rs.cell_means("K05", "quad_intake") - rs.cell_means("K10", "quad_intake")
noise = rs.pair_noise("quad_intake")
for mu, name in [(0.0, "对 0（原分析做的）"), (0.001839, "对 §14.1 预测值（该做的）")]:
    t, p = stats.ttest_1samp(d, mu)
    W, pw = stats.wilcoxon(d - mu, alternative="two-sided")
    print(f"  {name:<28} 差−μ = {d.mean()-mu:+.5f}  = {(d.mean()-mu)/noise:+.2f} 个噪声SD  "
          f"t={t:+.2f} p_t={p:.4f}  符号秩 p={pw:.4f}  ({(d>mu).sum()}+/{(d<mu).sum()}−)")
print("  ⇒ 若对预测值的检验不显著，本轮**没有证伪** §14.1，只是分辨不出预测值与 0。")

print()
print("=" * 92)
print("2. 种群走到的 s 范围 vs 曲率旋钮起作用的范围")
print("=" * 92)


def moments(rec):
    ctr = np.array(rec["bin_centers"])
    w = np.array(rec["bin_n"], float)
    mu = (ctr * w).sum() / w.sum()
    sd = math.sqrt(((ctr - mu) ** 2 * w).sum() / w.sum())
    return mu, sd


def frontier(p, k):
    s = 2.0 * p - 1.0
    g = np.clip(1.0 + s, 0.0, None)
    f = np.clip(1.0 - s, 0.0, None)
    if k != 1.0:
        n = np.power(g, k) + np.power(f, k)
        c = np.power(2.0 / np.maximum(n, 1e-12), 1.0 / k)
        g, f = g * c, f * c
    return g, f


KOF = {"N": 1.0, "K10": 1.0, "K05": 0.5, "K035": 0.35}
for a in ARMS:
    mus = np.array([moments(rs.records[(a, s, r)]) for s in SEEDS for r in REPS])
    p_mu, p_sd = mus[:, 0].mean(), mus[:, 1].mean()
    s_mu, s_sd = 2 * p_mu - 1, 2 * p_sd
    k = KOF[a]
    g2, f2 = frontier(np.array([p_mu + 2 * p_sd]), k)
    g1, f1 = frontier(np.array([1.0]), k)
    print(f"  {a:>5} k={k:<5} pref={p_mu:.4f}±{p_sd:.4f}  s={s_mu:+.4f}±{s_sd:.4f}  "
          f"到 s=+1 需 {(1-s_mu)/s_sd:.1f} 个 SD")
    print(f"          总乘子 g+f: s=0 → 2.000, +2SD(s={s_mu+2*s_sd:+.3f}) → {(g2+f2)[0]:.4f} "
          f"(+{100*((g2+f2)[0]-2)/2:.2f}%), s=+1 → {(g1+f1)[0]:.3f}")

print()
print("=" * 92)
print("3. quad_demog 的估计量偏倚：中性臂 N 的真值必须恰为 0")
print("=" * 92)
for met in ["quad_demog", "quadrel_demog", "quad_intake"]:
    r = one_sample(rs, met, "N")
    x = rs.cell_means("N", met)
    t, p = stats.ttest_1samp(x, 0.0)
    print(f"  {met:<16} N 臂 = {x.mean():+.5f} ± {x.std(ddof=1)/math.sqrt(12):.5f}  "
          f"t={t:+.2f} p_t={p:.4f}  符号秩 p={r.p:.4f}  ({r.n_pos}+/{r.n_neg}−)")
print("  基因在 N 臂被编译期断开（forage_tradeoff=0），pref 是纯中性标记 ⇒ 真值恰为 0。")

print()
print("=" * 92)
print("4. 分箱摄入曲线的形状（跨 24 run 平均，仅取全臂都有效的箱）")
print("=" * 92)
ctr = np.array(rs.records[("N", 0, 1)]["bin_centers"])
print("   pref   " + "".join(f"{a:>12}" for a in ARMS) + "        N计数")
curves = {}
for a in ARMS:
    M = np.full((len(SEEDS) * len(REPS), len(ctr)), np.nan)
    C = np.zeros(len(ctr))
    for i, (s, r) in enumerate([(s, r) for s in SEEDS for r in REPS]):
        rec = rs.records[(a, s, r)]
        M[i] = [np.nan if v is None else v for v in rec["intake"]]
        C += np.array(rec["bin_n"], float)
    curves[a] = (np.nanmean(M, axis=0), C / len(M), np.isfinite(M).sum(axis=0))
for j, c in enumerate(ctr):
    row = f"  {c:>6.3f} "
    for a in ARMS:
        v = curves[a][0][j]
        row += f"{v:>12.4f}" if np.isfinite(v) else f"{'--':>12}"
    row += f"   {curves['N'][1][j]:>9.0f}"
    print(row)
print("   （峰在中部、两端下塌 = 凹；平台+两端悬崖 = 供给饱和；单调 = 只有一阶项）")
for a in ARMS:
    y, w, n = curves[a]
    m = np.isfinite(y) & (n >= 20)
    print(f"  {a:>5} 峰值箱 pref={ctr[m][np.argmax(y[m])]:.3f} "
          f"(摄入 {y[m].max():.4f})，最低有效箱 {y[m].min():.4f}，有效箱 {m.sum()}")
