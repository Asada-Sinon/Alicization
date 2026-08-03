"""R9 判决的独立复核（result-analyst）。**不改判据**，只回答四个问题：

1. `forage_pref` 的符号：高 = 草专精还是果专精？(代码 + 跨 run 经验旁证)
2. 护栏按**预注册的口径**（§13.5「对着本轮 N1 臂自估的 √2·σ̂_W/√r」）重算一遍，
   并**逐种子**读（§7.1「护栏按逐种子读」+ §13.5「渴死若再次逐种子破半数」）——
   `analyze_r9.py` 用的是四臂池化噪声、且只报了臂均值。
3. 零结果的 MDE（H1/H2 必须报，否则会被当成「证明了没有效应」）。
4. 「平世界塌缩在结构世界消失了」是不是就是交互项本身。

读：outputs/20260803-shade/r9/*.log（96）。输出：stdout（见 output/audit.txt）。
重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260803-shade-audit/audit_r9.py
"""
import math
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

import numpy as np
from scipy.stats import wilcoxon, pearsonr

from exp_stats import (RunSet, paired, one_sample, mde_sign_consistent,
                       power_paired_wilcoxon, required_seeds, _paired_core)

DIR = "outputs/20260803-shade/r9"
SEEDS, REPS = list(range(12)), [1, 2]
ARMS = ["N0", "T0", "N1", "T1"]

rs = RunSet.load(DIR, ARMS, SEEDS, REPS)
print(f"载入 {len(rs.records)} run, problems={rs.problems}")
print(f"overrides_diff (跨臂取值不同的字段) = {rs.overrides_diff()}")
print(f"一个臂的 overrides 样本: T1 = {rs.records[('T1',0,1)]['overrides']}")
print(f"出处样例: T1_s3_r1 -> {rs.sources[('T1',3,1)]}")

print("\n" + "="*90)
print("[Q1] 符号：跨 run 的 corr(herb_forage_pref, frugivory_frac)，每臂 24 个 run")
print("  高 forage_pref = 草专精 ⇒ 应为负；高 = 果专精 ⇒ 应为正；中性臂应 ≈ 0")
print("="*90)
for a in ARMS:
    x = rs.raw(a, "herb_forage_pref"); y = rs.raw(a, "frugivory_frac")
    r, p = pearsonr(x, y)
    print(f"  {a}: r={r:+.4f}  p={p:.5f}  n={len(x)}  "
          f"herb_forage_pref 均值={x.mean():.4f} sd={x.std(ddof=1):.4f}")

print("\n[Q1b] 同一批 run 内 corr(mean_forage_pref, frugivory_frac)")
for a in ARMS:
    x = rs.raw(a, "mean_forage_pref"); y = rs.raw(a, "frugivory_frac")
    r, p = pearsonr(x, y)
    print(f"  {a}: r={r:+.4f}  p={p:.5f}")

print("\n" + "="*90)
print("[Q2a] rho_space 的原始 r（不是 z）、每 run 的 herb 数 n、置换零模型 sd")
print("      z ≈ r·√n，所以臂间 n 不同会污染 z 的跨臂比较")
print("="*90)
for a in ARMS:
    rho = rs.raw(a, "rho_space"); n = rs.raw(a, "n"); sd = rs.raw(a, "rho_null_sd")
    z = rs.raw(a, "rho_space_z")
    print(f"  {a}: rho 均值={rho.mean():+.5f} (min {rho.min():+.4f}, max {rho.max():+.4f})  "
          f"n 均值={n.mean():7.1f}  null_sd 均值={sd.mean():.5f}  z 均值={z.mean():+.3f}")
print("  --- 原始 r 上的配对检验（去掉 √n 放大） ---")
for a, b in (("T1","N1"), ("T0","N0")):
    r = paired(rs, "rho_space", a, b)
    print(f"  {a}-{b}: 差={r.diff.mean():+.5f}  {r.n_neg}/12 为负  p={r.p:.5f}  "
          f"效应/噪声={r.ratio:+.3f}  CI=[{r.ci[0]:+.5f},{r.ci[1]:+.5f}]")
ix = paired(rs, "rho_space", "T1", "N1").diff - paired(rs, "rho_space", "T0", "N0").diff
sw = rs.pooled_within_sd("rho_space")
core = _paired_core(paired(rs,"rho_space","T1","N1").diff,
                    paired(rs,"rho_space","T0","N0").diff,
                    2.0*sw/math.sqrt(len(REPS)), "rho_space(原始r)",
                    "(T1-N1)-(T0-N0)", "T1-N1", "T0-N0")
print("  --- 原始 r 的交互项 ---")
print(core.format())

print("\n" + "="*90)
print("[Q2b] H3 第一半：T1 臂自己的 rho_space_z 对 0 的单样本符号秩（预注册写「显著 >0」）")
print("="*90)
for a in ARMS:
    r = one_sample(rs, "rho_space_z", a, mu=0.0)
    print(f"  {a}: 均值={r.mean_a:+.4f}  正 {r.n_pos}/12 负 {r.n_neg}/12  p={r.p:.5f}  "
          f"CI=[{r.ci[0]:+.3f},{r.ci[1]:+.3f}]  效应/噪声={r.ratio:+.2f}")

print("\n" + "="*90)
print("[Q3] 护栏：预注册口径 = 只用 N1 臂自估 σ̂_W（analyze_r9.py 用的是四臂池化）")
print("="*90)
GUARDS = ["death_thirst_frac", "frugivory_frac", "carnivore_frac", "carn_speed",
          "herb_speed", "population", "min_pop", "death_starvation_frac"]
for label, a, b in (("T1 vs N1", "T1", "N1"), ("N1 vs N0", "N1", "N0"),
                    ("T1 vs N0", "T1", "N0")):
    print(f"\n  --- {label} ---")
    print(f'  {"指标":<22}{"Δ":>11}{"2SE(N1自估)":>13}{"破?":>5}{"2SE(四臂池)":>13}{"破?":>5}')
    for m in GUARDS:
        r_n1 = paired(rs, m, a, b, noise_arms=["N1"])
        r_all = paired(rs, m, a, b)
        d = r_n1.diff.mean()
        print(f"  {m:<22}{d:>+11.5f}{2*r_n1.noise:>13.5f}"
              f"{'破' if abs(d) > 2*r_n1.noise else 'ok':>5}"
              f"{2*r_all.noise:>13.5f}{'破' if abs(d) > 2*r_all.noise else 'ok':>5}")

print("\n" + "="*90)
print("[Q4] 护栏逐种子读（§7.1『护栏按逐种子读』；§13.5『渴死若再次逐种子破半数』）")
print("     每个种子的格均值之差 vs 2SE 容差（N1 自估）")
print("="*90)
for m in ("death_thirst_frac", "carnivore_frac", "frugivory_frac", "population"):
    for a, b in (("T1","N1"), ("N1","N0"), ("T1","N0")):
        r = paired(rs, m, a, b, noise_arms=["N1"])
        tol = 2*r.noise
        brk = np.abs(r.diff) > tol
        print(f"  {m:<20} {a}-{b}: 破 {int(brk.sum())}/12 个种子 (容差 {tol:.5f})  "
              f"破的种子={[SEEDS[i] for i in np.where(brk)[0]]}")

print("\n  死渴 death_thirst_frac 的绝对值（格均值，四臂）")
print(f'    {"seed":>4}{"N0":>9}{"T0":>9}{"N1":>9}{"T1":>9}')
cm = {a: rs.cell_means(a, "death_thirst_frac") for a in ARMS}
for i, s in enumerate(SEEDS):
    print(f"    {s:>4}{cm['N0'][i]:>9.4f}{cm['T0'][i]:>9.4f}{cm['N1'][i]:>9.4f}{cm['T1'][i]:>9.4f}")
print(f"    {'均值':>4}" + "".join(f"{cm[a].mean():>9.4f}" for a in ARMS))

print("\n" + "="*90)
print("[Q5] 捕食者灭绝：逐 run 的 carnivore_frac == 0 计数（臂均值会藏住灭绝）")
print("="*90)
for a in ARMS:
    v = rs.raw(a, "carnivore_frac")
    cmn = rs.cell_means(a, "carnivore_frac")
    print(f"  {a}: 24 run 中 carn=0 的有 {int((v==0).sum())} 个；"
          f"12 个格里格均值=0 的有 {int((cmn==0).sum())} 个；臂均值={v.mean():.5f}")
    print(f"       逐种子格均值 = {np.array2string(cmn, precision=4)}")
print("\n  min_pop 逐种子（T1 / N1），看有没有近灭绝")
for a in ("N0","N1","T1"):
    print(f"  {a}: {np.array2string(rs.cell_means(a,'min_pop'), precision=0)}")

print("\n" + "="*90)
print("[Q6] 零结果的 MDE（H1/H2 必须报）与所需种子数")
print("="*90)
for m in ("sd", "blrt_lr_per_n", "rho_space_z"):
    t1, n1 = rs.cell_means("T1", m), rs.cell_means("N1", m)
    t0, n0 = rs.cell_means("T0", m), rs.cell_means("N0", m)
    d = (t1-n1)-(t0-n0)
    sd_d = float(d.std(ddof=1))
    sw = rs.pooled_within_sd(m)
    noise = 2.0*sw/math.sqrt(len(REPS))
    mde_floor = mde_sign_consistent(sd_d, len(SEEDS))
    pw = power_paired_wilcoxon(abs(d.mean()), sd_d, len(SEEDS))
    print(f"  {m}: 观测交互={d.mean():+.5f}  实测配对差SD={sd_d:.5f}  纯噪声预测={noise:.5f}")
    print(f"     MDE(全同向口径, s=12)={mde_floor:+.5f}   "
          f"当前效应在 s=12 下 p≤0.05 的功效={pw:.3f}")
    need = required_seeds(abs(d.mean()), sd_d/math.sqrt(2)*math.sqrt(len(REPS)), max_seeds=200)
    print(f"     要把当前观测效应检出到 p≤0.05@80% 功效需要 s≈{need}")

print("\n" + "="*90)
print("[Q7] 「平世界塌缩在结构世界消失了」是不是就是交互项")
print("="*90)
for m in ("sd",):
    t1, n1 = rs.cell_means("T1", m), rs.cell_means("N1", m)
    t0, n0 = rs.cell_means("T0", m), rs.cell_means("N0", m)
    a = paired(rs, m, "T0", "N0"); b = paired(rs, m, "T1", "N1")
    print(f"  T0-N0: Δ={a.diff.mean():+.5f} p={a.p:.4f} 负 {a.n_neg}/12 效应/噪声={a.ratio:+.2f}")
    print(f"  T1-N1: Δ={b.diff.mean():+.5f} p={b.p:.4f} 正 {b.n_pos}/12 效应/噪声={b.ratio:+.2f}")
    print(f"  两者之差(=交互项) = {((t1-n1)-(t0-n0)).mean():+.5f}, "
          f"Wilcoxon p={wilcoxon(t1-n1, t0-n0)[1]:.5f}")
    print(f"  T1-N1 的 95% CI = [{b.ci[0]:+.5f}, {b.ci[1]:+.5f}]  "
          f"→ 这个区间没有排除掉多大的塌缩？T0-N0 的塌缩是 {a.diff.mean():+.5f}")
    print(f"  T0-N0 的 CI = [{a.ci[0]:+.5f}, {a.ci[1]:+.5f}]")
    print(f"  相对量: T0-N0 = {100*a.diff.mean()/a.mean_b:+.1f}% of N0；"
          f"T1-N1 = {100*b.diff.mean()/b.mean_b:+.1f}% of N1")
