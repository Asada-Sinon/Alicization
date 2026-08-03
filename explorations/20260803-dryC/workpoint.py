"""C 阶段工作点判读：`fruit_dry_weight` 该取哪个值进 D，还是「曲线上没有可用点」。

回答什么（`docs/multispecies_program.md` §12.3.C）：
  1. 推荐哪个 `fruit_dry_weight` 作 D 的工作点，或判「没有可用点」；
  2. 护栏（`carnivore_frac` / `death_thirst_frac`）到底崩没崩；
  3. `sel_ratio_water` 能不能当跨臂的「分离」判据（`analyze_c.py` §3 用了它）；
  4. 哪些读数被 `analyze_c.py` 丢掉了。

读哪些文件：
  - `outputs/20260803-dryC/w{00,025,05,075,10}_s{0,1,2}_r1.log`   （本轮探针，15 run）
  - `outputs/20260803-overlapA/niche_s{0..11}_r{1,2}.log`          （A 的 niche 臂，借噪声）
  - `outputs/20260803-partition/P_tradeoff0_s{0..5}_r{1,2,3}.log`  （**同一个 niche 世界**、
      `forage_tradeoff=0.0`＝默认、20000 步、`probe_trait_dist.py` 与 `measure_overlap.py`
      的护栏口径逐字相同：`row = {k: v[-1]}` 末步快照 + `death_*_frac` 全程累加。
      这是 `death_thirst_frac` 在本世界唯一存在的 σ̂_W 来源。）
  - `terrain.build(Config(fruit_dry_weight=w))`（无 RNG，确定值）

输出怎么读：§4 是本脚本的核心——`sel_ratio_water` 的**可达上限随臂改变**，所以它的跨臂
比较（`analyze_c.py` §3 的「分离」列）不成立。§3 给的是逐种子配对差 + 借来的噪声尺，
n=3 的双侧 p 地板是 0.25，**任何 p 都不可能过线**，报出来只是为了满足「算过的都要报」。

重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260803-dryC/workpoint.py
"""
import math
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")

import numpy as np

from exp_stats import RunSet, mde_sign_consistent, paired, wilcoxon_p_floor

SEEDS = [0, 1, 2]
ARMS = ["w00", "w025", "w05", "w075", "w10"]
W_OF = {"w00": 0.0, "w025": 0.25, "w05": 0.5, "w075": 0.75, "w10": 1.0}

# ---------------------------------------------------------------- §0 载入与自检
print("=" * 100)
print("§0 载入与自检")
print("=" * 100)
rs = RunSet.load("outputs/20260803-dryC", ARMS, SEEDS, [1])
print(f"  dryC: {len(rs.records)} run   problems={rs.problems}")
print(f"  dryC overrides_diff() = {sorted(rs.overrides_diff())}")
for a in ARMS:
    ov = rs.records[(a, 0, 1)]["overrides"]
    print(f"    {a:>5}  fruit_dry_weight={ov['fruit_dry_weight']}  "
          f"steps={rs.records[(a,0,1)]['steps']}  src={rs.sources[(a,0,1)]}  overrides={ov}")

a_niche = RunSet.load("outputs/20260803-overlapA", ["niche"], range(12), [1, 2])
print(f"\n  A/niche: {len(a_niche.records)} run   problems={a_niche.problems}")
p_part = RunSet.load("outputs/20260803-partition", ["P_tradeoff0"], range(6), [1, 2, 3])
print(f"  partition/P_tradeoff0: {len(p_part.records)} run   problems={p_part.problems}")
print(f"    P_tradeoff0 overrides = {p_part.records[('P_tradeoff0',0,1)]['overrides']}")
print(f"    dryC w00    overrides = {rs.records[('w00',0,1)]['overrides']}")
print("    -> 两者的世界参数集合相同（forage_tradeoff=0.0 与 fruit_dry_weight=0.0 都是 Config 默认）")

# ------------------------------------------------------- §1 逐种子读数（全部字段）
READ = ["sel_ratio_water", "sel_ratio_land", "frac_in_fruit", "frac_in_fruit_null_land",
        "schoener_d", "schoener_d_null_land", "frugivory_frac", "fruit_cells_eff",
        "herb_water_dist", "forest_frac", "death_thirst_frac", "death_predation_frac",
        "death_starvation_frac", "total_deaths", "population", "carnivore_frac",
        "fruit_cap_total"]
print()
print("=" * 100)
print("§1 逐种子读数（n=3 种子 × 1 重复；每个数 = 对应日志的 JSON 行）")
print("=" * 100)
for m in READ:
    print(f"  {m}")
    hdr = "      seed  " + "".join(f"{a:>13}" for a in ARMS)
    print(hdr)
    for s in SEEDS:
        row = f"      {s:>4}  "
        for a in ARMS:
            row += f"{rs.records[(a, s, 1)][m]:>13.4f}"
        print(row)
    row = "      均值  "
    for a in ARMS:
        row += f"{rs.cell_means(a, m).mean():>13.4f}"
    print(row)
    row = "      跨SD  "
    for a in ARMS:
        row += f"{rs.cell_means(a, m).std(ddof=1):>13.4f}"
    print(row)

# ------------------------------------------------------------- §2 借来的噪声尺
print()
print("=" * 100)
print("§2 噪声尺：本轮 r=1 估不出 σ̂_W，从同世界的两批数据借")
print("=" * 100)
print("  σ̂_W = 同种子复跑的池化 SD（口径 2）；ICC = 创始者方差占比（0 = 配对种子买不到东西）")
print(f'  {"指标":<24}{"A/niche 12x2":>28}{"partition/P 6x3":>28}')
print(f'  {"":<24}{"σ̂_W      ICC   跨种子SD":>28}{"σ̂_W      ICC   跨种子SD":>28}')
NOISE = {}
for m in ["sel_ratio_water", "sel_ratio_land", "frac_in_fruit", "schoener_d",
          "frugivory_frac", "herb_water_dist", "forest_frac", "population",
          "carnivore_frac", "death_thirst_frac", "death_predation_frac", "total_deaths"]:
    cells = []
    for rsx, tag in ((a_niche, "niche"), (p_part, "P_tradeoff0")):
        try:
            sw = rsx.pooled_within_sd(m, [tag])
            icc = rsx.icc(m, tag)[0]
            bsd = rsx.cell_means(tag, m).std(ddof=1)
            cells.append((sw, icc, bsd))
        except KeyError:
            cells.append(None)
    txt = f"  {m:<24}"
    for c in cells:
        txt += ("        --  (无此字段)  " if c is None
                else f"{c[0]:>10.4f}{c[1]:>8.3f}{c[2]:>10.4f}")
    print(txt)
    NOISE[m] = cells

print()
print("  ** 本轮 r=1，所以一个格 = 一个 run，格内噪声进不了均值。")
print("     配对差（同种子、两臂、各 1 run）的纯噪声 SD = sqrt(2)*σ̂_W。")
print("     3 个种子的配对差均值的噪声 SE = sqrt(2)*σ̂_W/sqrt(3)。")

# --------------------------------------- §3 逐种子配对差 vs w00 + 借来的噪声 + p
print()
print("=" * 100)
print("§3 各臂 vs w00：逐种子配对差、借来的噪声尺、MDE、以及全部算过的 p")
print("=" * 100)
print(f"  n=3 配对符号秩的双侧 p 地板 = {wilcoxon_p_floor(3):.4f}  -> 本阶段任何 p 都不可能 <=0.05")


def sigma_w(metric, prefer_partition=False):
    """借来的 σ̂_W：默认取 A/niche（口径与本轮同源），A 没有的字段取 partition/P。"""
    a, p = NOISE[metric]
    if prefer_partition and p is not None:
        return p[0], "partition/P 6x3"
    if a is not None:
        return a[0], "A/niche 12x2"
    return p[0], "partition/P 6x3"


TARGETS = [("sel_ratio_water", False), ("sel_ratio_land", False), ("frac_in_fruit", False),
           ("schoener_d", False), ("frugivory_frac", False), ("herb_water_dist", False),
           ("forest_frac", False), ("population", False), ("carnivore_frac", False),
           ("death_thirst_frac", True), ("death_predation_frac", True),
           ("total_deaths", True)]
ALL_P = []
for m, pref in TARGETS:
    sw, src = sigma_w(m, pref)
    pair_noise = math.sqrt(2.0) * sw               # r=1
    se3 = pair_noise / math.sqrt(3.0)
    base = rs.cell_means("w00", m)
    print(f"\n  --- {m} ---   借来 σ̂_W={sw:.4f} ({src})；配对差纯噪声 SD={pair_noise:.4f}，"
          f"3 种子均值 SE={se3:.4f}")
    print(f"      w00 逐种子 = {np.array2string(base, precision=4)}   均值={base.mean():.4f}")
    for a in ARMS[1:]:
        r = paired(rs, m, a, "w00")
        d = r.diff
        mde = mde_sign_consistent(pair_noise, 3)
        ALL_P.append((m, a, r.p))
        print(f"      w={W_OF[a]:<5} 均值={r.mean_a:>10.4f}  Δ={d.mean():>+10.4f} "
              f"({100*d.mean()/r.mean_b:>+7.1f}%)  逐种子Δ={np.array2string(d, precision=4, sign='+')}"
              f"  同向 {max(r.n_pos, r.n_neg)}/3")
        print(f"              Δ/纯噪声SD={d.mean()/pair_noise:>+6.2f}  "
              f"|Δ|/SE3={abs(d.mean())/se3:>6.2f}  实测ΔSD={d.std(ddof=1):.4f}"
              f"  Wilcoxon p={r.p:.4f}  CI95={r.ci[0]:+.4f},{r.ci[1]:+.4f}"
              f"  MDE(3种子全同向,80%)={mde:.4f}")

# ------------------------- §4 sel_ratio_water 的跨臂可比性（纯地形，决定性诊断）
print()
print("=" * 100)
print("§4 `sel_ratio_water` 能不能跨臂比？——它的**可达上限**随臂改变（纯地形计算）")
print("=" * 100)
from underworld import Config                       # noqa: E402
from underworld import terrain as terrain_mod       # noqa: E402

base_t = terrain_mod.build(Config())
wd = np.asarray(base_t.water_dist, dtype=np.float64)
is_land = np.asarray(base_t.capacity, dtype=np.float64) > 0.0
grass = np.asarray(base_t.capacity, dtype=np.float64)

N_BINS = 24                                          # 与 measure_overlap.water_matched_null 相同
land_wd = wd[is_land]
edges = np.linspace(0.0, float(land_wd.max()) + 1e-6, N_BINS + 1)
which = np.clip(np.digitize(wd, edges) - 1, 0, N_BINS - 1)


def marginal_for_mean(target_mean):
    """在陆地格上构造 m ∝ exp(-beta*wd)，二分 beta 使质量加权平均水距 = target_mean。"""
    lo, hi = -0.5, 0.5
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        w = np.where(is_land, np.exp(-mid * (wd - land_wd.mean())), 0.0)
        mu = (w * wd).sum() / w.sum()
        if mu > target_mean:
            lo = mid
        else:
            hi = mid
    w = np.where(is_land, np.exp(-0.5 * (lo + hi) * (wd - land_wd.mean())), 0.0)
    return w / w.sum(), (w * wd).sum() / w.sum()


def ratio_bounds(cap, mass):
    """给定水距边缘分布 mass（逐格），sel_ratio_water 的可达 [下限, 1(零模型), 上限]。"""
    num_hi = num_lo = den = 0.0
    for b in range(N_BINS):
        sel = (which == b) & is_land
        if not sel.any():
            continue
        mb = float(mass[sel].sum())
        if mb <= 0:
            continue
        num_hi += mb * float(cap[sel].max())
        num_lo += mb * float(cap[sel].min())
        den += mb * float(cap[sel].mean())
    return num_lo / den, num_hi / den


obs_wd = {a: float(rs.cell_means(a, "herb_water_dist").mean()) for a in ARMS}
obs_sel = {a: float(rs.cell_means(a, "sel_ratio_water").mean()) for a in ARMS}
caps = {a: np.asarray(terrain_mod.build(Config(fruit_dry_weight=W_OF[a])).fruit_capacity,
                      dtype=np.float64) for a in ARMS}

m_fixed, mu_fixed = marginal_for_mean(obs_wd["w00"])
print(f"  上限 = 在同一水距边缘分布下，把全部个体放到每个水距箱里 fruit_cap 最大的那格所得的")
print(f"  sel_ratio_water。它只依赖地形，是这个臂上该指标的**量纲**。")
print(f'\n  {"臂":>6} {"实测 sel_r_w":>13} {"实测 herb_wd":>13} {"该臂上限(自身wd)":>18}'
      f' {"实测/上限":>10} {"该臂上限(固定wd=%.1f)":>22}' % mu_fixed)
scaled = {}
for a in ARMS:
    m_own, mu_own = marginal_for_mean(obs_wd[a])
    lo_own, hi_own = ratio_bounds(caps[a], m_own)
    lo_fix, hi_fix = ratio_bounds(caps[a], m_fixed)
    scaled[a] = obs_sel[a] / hi_own
    print(f"  {a:>6} {obs_sel[a]:>13.4f} {obs_wd[a]:>13.2f} {hi_own:>18.2f}"
          f" {obs_sel[a]/hi_own:>10.4f} {hi_fix:>22.2f}")
print("\n  同一列上限从 w=0 到 w=1 变了多少倍："
      f" 自身wd口径 {ratio_bounds(caps['w10'], marginal_for_mean(obs_wd['w10'])[0])[1] / ratio_bounds(caps['w00'], marginal_for_mean(obs_wd['w00'])[0])[1]:.2f}x")

# ------------------------------------------------ §5 地形级分离度（含 |间距| 一列）
print()
print("=" * 100)
print("§5 地形级分离度（terrain.build，无 RNG）——加一列 |间距|，这是 §12 立项目标本身")
print("=" * 100)
grass_c = float((wd * grass).sum() / grass.sum())
print(f'  {"w":>6} {"果层重心wd":>12} {"草层重心wd":>12} {"间距":>9} {"|间距|":>9}'
      f' {"|间距|/基线":>11} {"果层总承载":>11}')
base_gap = None
for a in ARMS:
    fc = caps[a]
    fc_c = float((wd * fc).sum() / fc.sum())
    gap = fc_c - grass_c
    if base_gap is None:
        base_gap = abs(gap)
    print(f"  {W_OF[a]:>6} {fc_c:>12.2f} {grass_c:>12.2f} {gap:>+9.2f} {abs(gap):>9.2f}"
          f" {abs(gap)/base_gap:>11.2f} {fc.sum():>11.2f}")

# ------------------------------------------ §6 被 analyze_c.py 丢掉的读数：超额量
print()
print("=" * 100)
print("§6 `analyze_c.py` 没报的读数：与**处理无关的零模型**（陆地均匀）之差")
print("=" * 100)
print("  这三个的零模型不随处理移动，所以跨臂可比性强于 sel_ratio_water。")
for name, obs, null in [("frac_in_fruit", "frac_in_fruit", "frac_in_fruit_null_land"),
                        ("schoener_d", "schoener_d", "schoener_d_null_land")]:
    print(f"\n  --- {name} 实测 − N-land 零模型 ---")
    for a in ARMS:
        o = rs.cell_means(a, obs)
        n = rs.cell_means(a, null)
        print(f"      w={W_OF[a]:<5} 实测={o.mean():.4f}  零模型={n.mean():.4f}  "
              f"超额={o.mean()-n.mean():+.4f}  逐种子超额="
              f"{np.array2string(o-n, precision=4, sign='+')}")
print("\n  --- sel_ratio_land（零模型 = 陆地均匀，不随处理移动）---")
for a in ARMS:
    v = rs.cell_means(a, "sel_ratio_land")
    print(f"      w={W_OF[a]:<5} 均值={v.mean():.4f}  逐种子={np.array2string(v, precision=4)}")

print("\n  --- 死因预算（全程累加）---")
print(f'  {"w":>6} {"total_deaths":>13} {"thirst":>9} {"predation":>11} {"starv":>9} {"senesc":>9}')
for a in ARMS:
    td = rs.cell_means(a, "total_deaths").mean()
    print(f"  {W_OF[a]:>6} {td:>13.0f}"
          f" {rs.cell_means(a,'death_thirst_frac').mean():>9.4f}"
          f" {rs.cell_means(a,'death_predation_frac').mean():>11.4f}"
          f" {rs.cell_means(a,'death_starvation_frac').mean():>9.4f}"
          f" {rs.cell_means(a,'death_senescence_frac').mean():>9.4f}")
print("\n  --- 渴死绝对量（total_deaths × thirst_frac）---")
for a in ARMS:
    n = rs.cell_means(a, lambda r: r["total_deaths"] * r["death_thirst_frac"])
    print(f"      w={W_OF[a]:<5} 均值={n.mean():>9.0f}  逐种子={np.array2string(n, precision=0)}")

print()
print("=" * 100)
print("§7 全部算过的 p（n=3，地板 0.25，未做 Bonferroni）")
print("=" * 100)
for m, a, p in ALL_P:
    print(f"  {m:<24} {a:>6} vs w00 : p={p:.4f}")

# --------------------------------------------- §8 w00 锚点本身可靠吗（对 A 的 niche）
print()
print("=" * 100)
print("§8 w00 锚点检验：本轮 3 种子的 w00 与 A 的 niche 臂（12 种子 × 2 重复）对得上吗")
print("=" * 100)
print("  两批的世界参数相同（overrides 逐字相同，除了 A 没传 fruit_dry_weight=0.0，那是默认值）；")
print("  测量脚本同一个 measure_overlap.py。差别只有：A 是 r=2、12 种子，本轮 r=1、3 种子。")
print(f'  {"指标":<22}{"dryC w00 (s0-2,r1)":>20}{"A niche 同 3 种子":>20}'
      f'{"A niche 全 12 种子":>20}{"w00 偏离/A跨种子SD":>20}')
for m in ["sel_ratio_water", "sel_ratio_land", "frac_in_fruit", "schoener_d",
          "frugivory_frac", "herb_water_dist", "forest_frac", "population",
          "carnivore_frac"]:
    c = rs.cell_means("w00", m).mean()
    a3 = np.mean([a_niche.cell("niche", s, m)[0] for s in SEEDS])
    a12 = a_niche.cell_means("niche", m)
    z = (c - a12.mean()) / a12.std(ddof=1) if a12.std(ddof=1) > 0 else float("nan")
    print(f"  {m:<22}{c:>20.4f}{a3:>20.4f}{a12.mean():>20.4f}{z:>20.2f}")
print("\n  A niche 全 12 种子的 carnivore_frac 逐种子格均值：")
print("    " + np.array2string(a_niche.cell_means("niche", "carnivore_frac"), precision=4))
print("  dryC w00 的 carnivore_frac 逐种子：" +
      np.array2string(rs.cell_means("w00", "carnivore_frac"), precision=4))
raw = a_niche.raw("niche", "carnivore_frac")
print(f"  A niche **逐 run**（24 个，非格均值）carnivore_frac: min={raw.min():.4f} "
      f"p25={np.percentile(raw,25):.4f} 中位={np.median(raw):.4f} "
      f"p75={np.percentile(raw,75):.4f} max={raw.max():.4f}")
print(f"  -> dryC w00 s2 的 0.4015 落在 A 的 24 run 分布的第 "
      f"{100*(raw < 0.4015).mean():.1f} 百分位")

# ------------------------- §9 护栏改用 12x2 / 6x3 的稳基线，而不是本轮 3 种子的 w00
print()
print("=" * 100)
print("§9 护栏重锚：拿 A/niche(12x2) 与 partition/P(6x3) 当基线，而不是本轮 3 种子的 w00")
print("=" * 100)
print("  为什么要重锚：§8 显示本轮 w00 的 carnivore_frac 被单个 run（s2=0.4015，超出 A 全部")
print("  24 个 run 的最大值 0.2536）抬高了。用它当分母会把「回落到 A 的常态」误读成「崩了」。")
print("  非配对（两批种子集合不同）SE：")
print("    SE_C = sqrt(σ_b²+σ_w²)/sqrt(3)   （本轮 r=1）")
print("    SE_A = sqrt(σ_b²+σ_w²/r_A)/sqrt(s_A)")
print("    SE_diff = sqrt(SE_C²+SE_A²)      （假设处理不改变方差分量——这是个假设）")
BASELINES = [
    ("sel_ratio_water", a_niche, "niche", 12, 2), ("frugivory_frac", a_niche, "niche", 12, 2),
    ("herb_water_dist", a_niche, "niche", 12, 2), ("forest_frac", a_niche, "niche", 12, 2),
    ("population", a_niche, "niche", 12, 2), ("carnivore_frac", a_niche, "niche", 12, 2),
    ("death_thirst_frac", p_part, "P_tradeoff0", 6, 3),
    ("death_predation_frac", p_part, "P_tradeoff0", 6, 3),
]
for m, rsx, tag, sA, rA in BASELINES:
    _, sb, sw = rsx.icc(m, tag)
    ref = rsx.cell_means(tag, m).mean()
    se_c = math.sqrt(sb ** 2 + sw ** 2) / math.sqrt(3)
    se_a = math.sqrt(sb ** 2 + sw ** 2 / rA) / math.sqrt(sA)
    se_d = math.sqrt(se_c ** 2 + se_a ** 2)
    src = f"{tag} {sA}x{rA}"
    print(f"\n  --- {m} ---  基线={ref:.4f} ({src})  σ_b={sb:.4f} σ_w={sw:.4f}  "
          f"SE_diff={se_d:.4f}  (2×SE_diff={2*se_d:.4f})")
    for a in ARMS:
        v = rs.cell_means(a, m)
        d = v.mean() - ref
        print(f"      w={W_OF[a]:<5} 均值={v.mean():>10.4f}  Δ={d:>+10.4f}  "
              f"Δ/SE_diff={d/se_d:>+6.2f}  {'** 越 2SE **' if abs(d) > 2*se_d else '(2SE 内)'}")
