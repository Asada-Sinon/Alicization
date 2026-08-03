"""R10 判决的**独立复算**：不引用 `outputs/.../r10_analysis.txt` 里的任何数字，
全部从 96 个原始 log 的 `JSON ` 行重算，并加三项原分析没做的检查。

回答什么
--------
1. 复核 §14.4 H1–H4 与 §14.5 护栏的每一个数（口径走 `scripts/exp_stats.py`）。
2. **§14.1 假设本身在本轮设计下的预测效应量是多大？** 用实测的 `forage_pref` 分布
   与实测的两层单位回报 (A, B) 直接算 `beta2_pred = A·q_g(k) + B·q_f(k)`，
   再跟本轮自估的 MDE 比。这是原分析缺的一步——不知道预测效应量就无法区分
   「证伪」与「功效不足」。
3. **z 标准化的尺度混杂**：`quad_intake` 是 z 单位的系数，z 用的是各臂自己的
   `sd(pref)`。臂间 sd 不同 ⇒ 同一条原始曲面会读出不同的 beta2。换算回原始
   `s = 2·pref − 1` 单位后重测。

读哪些文件
----------
outputs/20260803-curvature/r10/{N,K10,K05,K035}_s{0..11}_r{1,2}.log 的 `JSON ` 行。

输出怎么读
----------
分节打印；每节标题下第一行就是结论要用的数。全部 p 值都打印，不做 Bonferroni。
"""
import math
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, "/home/xrl/intern/Alicization")
sys.path.insert(0, "/home/xrl/intern/Alicization/scripts")
from exp_stats import RunSet, paired, one_sample, mde_sign_consistent, wilcoxon_p_floor

DIR = "/home/xrl/intern/Alicization/outputs/20260803-curvature/r10"
ARMS = ["N", "K10", "K05", "K035"]
SEEDS = list(range(12))
REPS = (1, 2)


def check(rec, arm, s, r):
    want = {"N": (0.0, 1.0), "K10": (1.0, 1.0), "K05": (1.0, 0.5), "K035": (1.0, 0.35)}[arm]
    o = rec.get("overrides", {})
    got = (o.get("forage_tradeoff"), o.get("forage_curvature"))
    return None if got == want else f"overrides {got} != 臂名要求 {want}"


rs = RunSet.load(DIR, ARMS, SEEDS, REPS, check=check)
print("=" * 92)
print("0. 载入自检")
print("=" * 92)
print(f"  run 数 = {len(rs.records)}  problems = {rs.problems if rs.problems else '无'}")
print(f"  全臂 overrides 差异字段 = {rs.overrides_diff()}")
for a, b in [("K05", "K10"), ("K035", "K10"), ("K035", "K05"), ("K05", "N"), ("K10", "N")]:
    # **必须先过滤 records**：`overrides_diff` 扫的是 `self.records` 全量，
    # 只换 `arms` 列表不会把别的臂剔掉（第一版就是这么错的，读出的是全臂差异）。
    sel = {k: v for k, v in rs.records.items() if k[0] in (a, b)}
    sub = RunSet(sel, {k: rs.sources[k] for k in sel}, [a, b], SEEDS, list(REPS))
    print(f"  {a} vs {b} 的 overrides 差异字段 = {sub.overrides_diff()}")

print()
print("=" * 92)
print("1. 各臂 quad_intake（H3：N 与 K10 都应 ≈0）—— 单样本 t 与符号秩，两种都报")
print("=" * 92)
cm = {a: rs.cell_means(a, "quad_intake") for a in ARMS}
for a in ARMS:
    x = cm[a]
    t, p_t = stats.ttest_1samp(x, 0.0)
    w = one_sample(rs, "quad_intake", a)
    icc, sb, sw = rs.icc("quad_intake", a)
    print(f"  {a:>5}  均值={x.mean():+.5f}  跨种子SE={x.std(ddof=1)/math.sqrt(len(x)):.5f}  "
          f"t={t:+.2f} p_t={p_t:.5f}   符号秩 p={w.p:.5f} ({w.n_neg}/12 为负)  "
          f"ICC={icc:.3f} σ_w={sw:.5f}")
    print(f"         逐种子 = {np.array2string(x, precision=5, sign='+')}")

print()
print("=" * 92)
print("2. H1 主判据 = K05 − K10（§14.4 原文：`K05 − K10 > 0`, p≤0.05, 方向≥10/12, 比值≥1）")
print("=" * 92)
CONTRASTS = [("K05", "K10"), ("K035", "K10"), ("K035", "K05"), ("K10", "N"),
             ("K05", "N"), ("K035", "N")]
for a, b in CONTRASTS:
    print(paired(rs, "quad_intake", a, b).format())
    print()

sd_pair = float((cm["K05"] - cm["K10"]).std(ddof=1))
noise = rs.pair_noise("quad_intake")
print(f"  本轮自估配对差噪声 √2·σ̂_W/√r = {noise:.6f}")
print(f"  MDE(同向, n=12, 80%) 用噪声      = {mde_sign_consistent(noise, 12):.6f}")
print(f"  MDE(同向, n=12, 80%) 用实测差SD  = {mde_sign_consistent(sd_pair, 12):.6f}")
print(f"  §14.2 预注册的 MDE（pilot σ_w=0.00274）= 0.0057")

print()
print("=" * 92)
print("3. §14.1 假设**自己**预测的效应量：beta2_pred = A·q_g(k) + B·q_f(k)")
print("=" * 92)
print("   A,B 由实测 graze_gain/fruit_gain ÷ 食草者数得到（W=A·g+B·f 的两层单位回报）；")
print("   q_g,q_f 由**实测 pref 直方图**上对 [1,z,z²] 的加权回归得到。k=1 时恒为 0。")


def frontier(p, k):
    s = 2.0 * p - 1.0
    g = np.clip(1.0 + 1.0 * s, 0.0, None)
    f = np.clip(1.0 - 1.0 * s, 0.0, None)
    if k != 1.0:
        norm = np.power(g, k) + np.power(f, k)
        c = np.power(2.0 / np.maximum(norm, 1e-12), 1.0 / k)
        g, f = g * c, f * c
    return g, f


def quad_coefs(ctr, w, k):
    """在实测 pref 分布上，把 g_k、f_k 各自对 [1,z,z²] 做加权回归，返回两个二次项系数。"""
    m = w.sum()
    mu = (ctr * w).sum() / m
    sd = math.sqrt(max(((ctr - mu) ** 2 * w).sum() / m, 1e-18))
    z = (ctr - mu) / sd
    X = np.stack([np.ones_like(z), z, z * z], 1) * np.sqrt(w)[:, None]
    g, f = frontier(ctr, k)
    qg = np.linalg.lstsq(X, g * np.sqrt(w), rcond=None)[0][2]
    qf = np.linalg.lstsq(X, f * np.sqrt(w), rcond=None)[0][2]
    return float(qg), float(qf), mu, sd


KOF = {"N": 1.0, "K10": 1.0, "K05": 0.5, "K035": 0.35}
pred = {}
for a in ARMS:
    rows = []
    for s in SEEDS:
        for r in REPS:
            rec = rs.records[(a, s, r)]
            ctr = np.array(rec["bin_centers"])
            w = np.array(rec["bin_n"], float)
            nherb = rec["population"] * (1.0 - rec["carnivore_frac"])
            A = rec["graze_gain"] / max(nherb, 1.0)
            B = rec["fruit_gain"] / max(nherb, 1.0)
            qg, qf, mu, sd = quad_coefs(ctr, w, KOF[a])
            rows.append((A * qg + B * qf, A, B, mu, sd, float(w.sum() / rec["n_samples"])))
    arr = np.array(rows)
    pred[a] = arr
    print(f"  {a:>5}  预测 beta2 = {arr[:,0].mean():+.6f} ± {arr[:,0].std(ddof=1):.6f}   "
          f"A={arr[:,1].mean():.4f} B={arr[:,2].mean():.4f}  "
          f"pref 均值={arr[:,3].mean():.4f} sd={arr[:,4].mean():.4f}  "
          f"落箱率={arr[:,5].mean():.3f}")
dpred = pred["K05"][:, 0].mean() - pred["K10"][:, 0].mean()
print(f"\n  ** 假设预测的 K05−K10 = {dpred:+.6f}，本轮 MDE = {mde_sign_consistent(noise,12):.6f}"
      f"，预测/MDE = {dpred / mde_sign_consistent(noise, 12):.3f} **")
print(f"  实测 K05−K10 = {(cm['K05']-cm['K10']).mean():+.6f}；"
      f"实测/预测 = {(cm['K05']-cm['K10']).mean()/dpred:+.2f}")
dpred35 = pred["K035"][:, 0].mean() - pred["K10"][:, 0].mean()
print(f"  假设预测的 K035−K10 = {dpred35:+.6f}，实测 = {(cm['K035']-cm['K10']).mean():+.6f}")

print()
print("=" * 92)
print("4. z 单位的尺度混杂：换算成原始 s = 2·pref−1 单位后重测")
print("=" * 92)
print("   beta2_z = (1/2)·W''(s)·(2·sd_p)²  ⇒  W'' = beta2_z / (2·sd_p²)。sd_p 用实测直方图。")


def curv_raw(rec):
    w = np.array(rec["bin_n"], float)
    ctr = np.array(rec["bin_centers"])
    mu = (ctr * w).sum() / w.sum()
    sd = math.sqrt(((ctr - mu) ** 2 * w).sum() / w.sum())
    return rec["quad_intake"] / (2.0 * sd * sd)


for a in ARMS:
    x = rs.cell_means(a, curv_raw)
    t, p_t = stats.ttest_1samp(x, 0.0)
    print(f"  {a:>5}  W''(s) = {x.mean():+.5f} ± {x.std(ddof=1)/math.sqrt(12):.5f}  t={t:+.2f} p={p_t:.5f}")
for a, b in [("K05", "K10"), ("K035", "K10"), ("K05", "N")]:
    print(paired(rs, curv_raw, a, b, metric_name="W''(s) 原始单位").format())
    print()

print("=" * 92)
print("5. 一阶项：曲面被人口学平衡拉平了吗？（用分箱曲线自算 beta1，原分析没存）")
print("=" * 92)
print("   若平衡把实现适应度拉平，一阶项也应 ≈0。分箱加权回归 intake ~ 1+z+z²。")


def binned_fit(rec):
    ctr = np.array(rec["bin_centers"])
    w = np.array(rec["bin_n"], float)
    y = np.array([np.nan if v is None else v for v in rec["intake"]], float)
    m = np.isfinite(y) & (w > 0)
    if m.sum() < 4:
        return np.array([np.nan] * 3), 0
    mu = (ctr * w).sum() / w.sum()
    sd = math.sqrt(((ctr - mu) ** 2 * w).sum() / w.sum())
    z = (ctr[m] - mu) / sd
    sw = np.sqrt(w[m])
    X = np.stack([np.ones_like(z), z, z * z], 1) * sw[:, None]
    return np.linalg.lstsq(X, y[m] * sw, rcond=None)[0], int(m.sum())


for a in ARMS:
    b0 = rs.cell_means(a, lambda r: binned_fit(r)[0][0])
    b1 = rs.cell_means(a, lambda r: binned_fit(r)[0][1])
    b2 = rs.cell_means(a, lambda r: binned_fit(r)[0][2])
    nb = rs.cell_means(a, lambda r: binned_fit(r)[1])
    t1, p1 = stats.ttest_1samp(b1, 0.0)
    t2, p2 = stats.ttest_1samp(b2, 0.0)
    print(f"  {a:>5}  截距={b0.mean():.5f}  一阶={b1.mean():+.5f} (t={t1:+.2f} p={p1:.4f})  "
          f"二阶={b2.mean():+.5f} (t={t2:+.2f} p={p2:.4f})  有效箱={nb.mean():.1f}")
print("   一阶/二阶之比越大，说明曲面越没被拉平（梯度仍在）。")

print()
print("=" * 92)
print("6. 次判据 / H4 / mean_pref / 护栏（全部 p 值都报）")
print("=" * 92)
for met in ["quadrel_intake", "quad_demog", "quadrel_demog", "sd",
            "bimodality_coefficient", "blrt_lr_per_n", "mean_pref"]:
    for a, b in [("K05", "K10"), ("K10", "N")]:
        r = paired(rs, met, a, b)
        flag = " **功效不足**" if r.underpowered else ""
        print(f"  {met:<24} {a}−{b} = {r.diff.mean():+.5f}  "
              f"({r.n_pos}+/{r.n_neg}−) p={r.p:.4f} 比值={r.ratio:+.2f}"
              f" CI=[{r.ci[0]:+.4f},{r.ci[1]:+.4f}]{flag}")
print()
print("  护栏（§14.5，容差 = 2×本轮自估配对噪声）:")
for met in ["population", "min_pop", "carnivore_frac", "carn_speed", "frugivory_frac",
            "death_thirst_frac", "graze_gain", "fruit_gain", "herb_speed",
            "death_starvation_frac", "death_predation_frac"]:
    r = paired(rs, met, "K05", "K10")
    tol = 2 * r.noise
    print(f"    {met:<22} K10={r.mean_b:>10.4f} K05={r.mean_a:>10.4f} Δ={r.diff.mean():+10.4f} "
          f"±{tol:>9.4f}  {'ok' if abs(r.diff.mean()) <= tol else '**破**'}  "
          f"比值={r.ratio:+.2f} p={r.p:.4f}")

print()
print("=" * 92)
print("7. 总通量：k<1 注入的乘子有多少变成了实际摄入？")
print("=" * 92)
for a in ARMS:
    tot = rs.cell_means(a, lambda r: r["graze_gain"] + r["fruit_gain"])
    per = rs.cell_means(a, lambda r: (r["graze_gain"] + r["fruit_gain"])
                        / max(r["population"] * (1 - r["carnivore_frac"]), 1.0))
    mult = pred[a][:, 0] * 0 + [
        float((np.array(rs.records[(a, s, rr)]["bin_n"], float) @
               (lambda gf: gf[0] + gf[1])(frontier(np.array(rs.records[(a, s, rr)]["bin_centers"]), KOF[a])))
              / np.array(rs.records[(a, s, rr)]["bin_n"], float).sum())
        for s in SEEDS for rr in REPS]
    print(f"  {a:>5}  总摄入={tot.mean():8.2f}  人均={per.mean():.5f}  "
          f"实测 pref 分布上的平均总乘子 g+f = {mult.mean():.5f} (k=1 恒为 2)")
r = paired(rs, lambda r: r["graze_gain"] + r["fruit_gain"], "K05", "K10", metric_name="总摄入")
print(f"  K05−K10 总摄入 = {r.diff.mean():+.3f} ({100*r.diff.mean()/r.mean_b:+.2f}%) "
      f"p={r.p:.4f} 比值={r.ratio:+.2f}")
