"""A 阶段判决的补充计算：C/D 的**判据容差数值**与**本设计的分辨率**。

回答什么（`analyze.py` 没算的三件）：
1. C/D 护栏容差的具体数值 —— 对着 `√2·σ̂_W/√r`（`docs/run_to_run_variance.md` §7.1 口径 3）
   给出 1×/2× 两条线，并同时给出「只用 niche 臂自估」与「两臂池化」两个版本。
2. 本设计的分辨率 —— 主口径下降 20% / 50% 各需多少种子；以及 s=12,r=2 在
   **p≤0.05, 80% 功效** 口径下的 MDE（`analyze.py` 报的是 `mde_sign_consistent`
   的「顶到地板 p=2/2¹²」口径，对 n=12 偏保守，两个都要有）。
3. base 臂主口径的「上行空间」检查 —— base 的 sel_ratio_water 与零模型 1.0 的
   95% CI 是否含 0。含 0 ⇒ base 已经贴着零模型，C 阶段的「重叠度下去了没有」
   在 base 世界里没有可下降的量（地板效应）。

读什么：`outputs/20260803-overlapA/{base,niche}_s{0..11}_r{1,2}.log`（48 个）。
统计一律走 `scripts/exp_stats.py`，本文件不重推任何算术。

怎么读输出：三节各自带表头；「建议容差」列即可直接抄进 C/D 的判据。

重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260803-overlapA/resolution_and_tolerances.py
"""
import math
import sys

sys.path.insert(0, "scripts")

import numpy as np

from exp_stats import (RunSet, bootstrap_ci, mde_sign_consistent, one_sample,
                       power_paired_wilcoxon, required_seeds)

DIR = "outputs/20260803-overlapA"
SEEDS, REPS = list(range(12)), [1, 2]
ARMS = ["base", "niche"]
PRIMARY = "sel_ratio_water"
SECONDARY = ["sel_ratio_land", "frac_in_fruit", "schoener_d"]
GUARDS = ["population", "carnivore_frac", "frugivory_frac", "herb_water_dist", "forest_frac"]

rs = RunSet.load(DIR, ARMS, SEEDS, REPS)
print(f"载入 {len(rs.records)} 个 run; problems = {rs.problems or '无'}")
assert not rs.problems, rs.problems
print(f"overrides_diff() = {sorted(rs.overrides_diff())}  （{len(rs.overrides_diff())} 项）")


# ---------------------------------------------------------------- §A 容差
print("\n" + "=" * 108)
print("§A C/D 判据容差（口径 3：对着 √2·σ̂_W/√r 定，r=2 ⇒ 该量 == σ̂_W）")
print("=" * 108)
print(f'  {"metric":<20} {"臂":<6} {"均值":>11} {"σ̂_W":>10} {"1×噪声":>10} '
      f'{"2×噪声(建议容差)":>16} {"2×噪声占均值":>12}')
for m in [PRIMARY] + SECONDARY + GUARDS:
    for arm in ARMS + ["池化"]:
        which = [arm] if arm in ARMS else ARMS
        pn = rs.pair_noise(m, which)
        mu = float(np.mean([rs.cell_means(a, m).mean() for a in which]))
        sw = rs.pooled_within_sd(m, which)
        rel = 100 * 2 * pn / abs(mu) if mu else float("nan")
        print(f"  {m:<20} {arm:<6} {mu:>11.4f} {sw:>10.4f} {pn:>10.4f} "
              f"{2 * pn:>16.4f} {rel:>11.1f}%")
    print()


# ------------------------------------------------------- §B 分辨率 / MDE
def mde_alpha05(noise: float, s: int, power: float = 0.80,
                lo: float = 0.0, hi_mult: float = 6.0) -> float:
    """二分求 p≤0.05、`power` 功效下的最小可检出配对差（用 power_paired_wilcoxon 反解）。"""
    hi = hi_mult * noise
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if power_paired_wilcoxon(mid, noise, s) >= power:
            hi = mid
        else:
            lo = mid
    return hi


print("=" * 108)
print("§B 分辨率：s=12,r=2 的 MDE，以及检出 −20% / −50% 所需的种子数")
print("=" * 108)
print(f'  {"metric":<20} {"臂":<6} {"基线":>10} {"MDE(地板p口径)":>15} '
      f'{"MDE(p≤.05,80%)":>15} {"占基线":>8} {"−20%需":>7} {"−50%需":>7}')
for m in [PRIMARY] + SECONDARY:
    for arm in ARMS:
        base_v = float(rs.cell_means(arm, m).mean())
        pn = rs.pair_noise(m, [arm])
        sw = rs.pooled_within_sd(m, [arm])
        m_floor = mde_sign_consistent(pn, len(SEEDS))
        m_05 = mde_alpha05(pn, len(SEEDS))
        n20 = required_seeds(0.20 * abs(base_v), sw, reps=len(REPS))
        n50 = required_seeds(0.50 * abs(base_v), sw, reps=len(REPS))
        print(f"  {m:<20} {arm:<6} {base_v:>10.4f} {m_floor:>15.4f} {m_05:>15.4f} "
              f"{100 * m_05 / abs(base_v):>7.1f}% {str(n20):>7} {str(n50):>7}")
    print()

# 主口径若从 niche 基线下降 20%/50%，本设计(s=12,r=2)的实际功效是多少
niche_base = float(rs.cell_means("niche", PRIMARY).mean())
pn_n = rs.pair_noise(PRIMARY, ["niche"])
print(f"  niche 主口径基线 {niche_base:.4f}，配对差噪声 {pn_n:.4f}")
for frac in (0.10, 0.20, 0.30, 0.50):
    eff = frac * niche_base
    pw = power_paired_wilcoxon(eff, pn_n, 12)
    print(f"    下降 {100*frac:>4.0f}% (= {eff:.4f})  → s=12,r=2 的功效 = {pw:.3f}")


# ------------------------------------------------- §C base 臂有没有下行空间
print("\n" + "=" * 108)
print("§C 哪个臂能当 C/D 的基线：主口径离零模型还有多少可下降的量")
print("=" * 108)
for arm in ARMS:
    r = one_sample(rs, PRIMARY, arm, mu=1.0)
    head = r.mean_a - 1.0
    print(f"  {arm:<6} sel_ratio_water = {r.mean_a:.4f}   离零模型 1.0 的量 = {head:+.4f}   "
          f"95% CI [{r.ci[0]:+.4f}, {r.ci[1]:+.4f}]  {'含 0' if r.ci[0]*r.ci[1] <= 0 else '不含 0'}")
    print(f"         p={r.p:.5f}  {r.n_pos}/12 高于零模型  效应/噪声={r.ratio:+.3f}"
          f"{'  ** <1 功效不足 **' if r.underpowered else ''}")
    print(f"         「可下降的量」÷ 本设计 MDE(p≤.05) = "
          f"{head / mde_alpha05(rs.pair_noise(PRIMARY, [arm]), 12):.2f}")
    fg = rs.cell_means(arm, "frugivory_frac")
    print(f"         frugivory_frac = {fg.mean():.5f}  [{fg.min():.5f}, {fg.max():.5f}]")

# 两臂 frugivory 的比值：果层「值不值钱」差多少倍
fb, fn = rs.cell_means("base", "frugivory_frac"), rs.cell_means("niche", "frugivory_frac")
print(f"\n  frugivory_frac niche/base 倍数 = {fn.mean() / fb.mean():.1f}×")
lo, hi = bootstrap_ci(fn - fb)
print(f"  frugivory_frac 配对差 = {(fn - fb).mean():+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]")
