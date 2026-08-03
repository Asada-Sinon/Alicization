"""A 阶段判决：果层 × 种群 重叠度的基线数字 + 它的噪声。

回答什么：`docs/multispecies_program.md` §12.3 A / §12.3.A —— 基线重叠度是多少，
它离零模型有多远，以及**它的复现噪声有多大**（C/D 的判据容差要对着这个数定）。

读什么：`outputs/20260803-overlapA/base_s{0..11}_r{1,2}.log`（24 个）+ `provenance.txt`。

统计走 `scripts/exp_stats.py`，口径见 `docs/conventions.md` §5.2。A 阶段只有一个臂，
所以「实测 vs 零模型」用 `one_sample()`——零模型是**同一个 run 内部**算出来的，
不存在第二个臂，但口径 1（先取格均值）与口径 2（效应 ÷ 自估噪声）照样适用。

重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260803-overlapA/analyze.py
"""
import sys

sys.path.insert(0, "scripts")

import numpy as np

from exp_stats import RunSet, mde_sign_consistent, one_sample, required_seeds

DIR = "outputs/20260803-overlapA"
ARM = "base"
SEEDS, REPS = list(range(12)), [1, 2]

# 预注册的主口径（§12.3.A，commit 57a49a7）。次口径与护栏跟在后面。
PRIMARY = "sel_ratio_water"
SECONDARY = ["sel_ratio_land", "frac_in_fruit", "schoener_d"]
GUARDS = ["population", "carnivore_frac", "frugivory_frac", "herb_water_dist", "forest_frac"]
SHAPE = ["fruit_cells_eff", "fruit_cells_pos", "fruit_cap_total", "land_cells"]

print("=" * 96)
print("§0 载入与自检")
print("=" * 96)
rs = RunSet.load(DIR, [ARM], SEEDS, REPS)
print(f"  期望 {len(SEEDS) * len(REPS)} 个 run，实际载入 {len(rs.records)} 个")
print(f"  problems = {rs.problems if rs.problems else '无'}")
print(f"  overrides 差异字段 = {rs.overrides_diff()}  (单臂基线，期望 [])")
assert not rs.problems, rs.problems

# 地形不吃 RNG，所以形状哨兵必须在 24 个 run 里逐位相同。不同就说明配置漏了。
print("\n  形状哨兵（terrain.build 不吃 RNG，24 个 run 必须完全一致）:")
for m in SHAPE:
    vals = rs.raw(ARM, m)
    same = np.allclose(vals, vals[0])
    print(f"    {m:<18} = {vals[0]:.4f}   {'[24 个 run 一致]' if same else '[**不一致!**]'}")
    assert same, f"{m} 在不同 run 之间不一致，地形不应随种子变化"

print()
print("=" * 96)
print(f"§1 主口径 {PRIMARY}：逐种子格均值（口径 1：2 次重复先取均值）")
print("=" * 96)
cm = rs.cell_means(ARM, PRIMARY)
print(f'  {"seed":>4}  {"r1":>9} {"r2":>9}   {"格均值":>9} {"格内SD":>9}   来源')
for i, s in enumerate(SEEDS):
    mu, sd, v = rs.cell(ARM, s, PRIMARY)
    src = " ".join(rs.sources[(ARM, s, r)] for r in REPS)
    print(f"  {s:>4}  {v[0]:>9.5f} {v[1]:>9.5f}   {mu:>9.5f} {sd:>9.5f}   {src}")
print(f"  {'均值':>4}  {'':>9} {'':>9}   {cm.mean():>9.5f}")
print(f"  跨种子（创始者）SD = {cm.std(ddof=1):.5f}   最小 {cm.min():.5f}   最大 {cm.max():.5f}")

print()
print("=" * 96)
print("§2 实测 vs 零模型：基线到底偏不偏果层（one_sample，比值对 μ=1.0）")
print("=" * 96)
res = {}
for m, mu in [("sel_ratio_water", 1.0), ("sel_ratio_land", 1.0), ("sel_ratio_uniform", 1.0)]:
    r = one_sample(rs, m, ARM, mu=mu, label=f"{m} vs 零模型(=1.0)")
    res[m] = r
    print()
    print(r.format())

# 派生量：实测份额 − 同 run 内零模型份额。派生之后再估噪声，才吃得到两者的相关性。
print()
for a, b in [("frac_in_fruit", "frac_in_fruit_null_water"),
             ("frac_in_fruit", "frac_in_fruit_null_land"),
             ("schoener_d", "schoener_d_null_water")]:
    r = one_sample(rs, lambda rec, a=a, b=b: rec[a] - rec[b], ARM, mu=0.0,
                   metric_name=f"{a} − {b}", label=f"{a} 减同 run 零模型 {b}（对 μ=0）")
    res[f"{a}-{b}"] = r
    print(r.format())
    print()

print("=" * 96)
print("§3 噪声与方差分解：C/D 的判据容差要对着这里的数定（口径 3）")
print("=" * 96)
print(f'  {"metric":<26} {"均值":>10} {"σ̂_W(同种子)":>13} {"σ_B(创始者)":>12} '
      f'{"ICC":>7} {"√2·σ̂_W/√2":>11}')
for m in [PRIMARY] + SECONDARY + GUARDS:
    icc, sb, sw = rs.icc(m, ARM)
    print(f"  {m:<26} {rs.cell_means(ARM, m).mean():>10.5f} {sw:>13.5f} {sb:>12.5f} "
          f"{icc:>7.3f} {rs.pair_noise(m, [ARM]):>11.5f}")
print("\n  读法：ICC 是创始者方差占比。ICC≈0 的指标**配对完全无用**（配对只消掉创始者那份，")
print("        降幅 1/(1−ICC)）。最后一列是两臂配对差的噪声，C/D 的护栏容差按它定。")

print()
print("=" * 96)
print("§4 本设计能看见多大的变化：MDE 与所需种子数")
print("=" * 96)
for m in [PRIMARY] + SECONDARY:
    icc, sb, sw = rs.icc(m, ARM)
    base = float(rs.cell_means(ARM, m).mean())
    pair_noise = rs.pair_noise(m, [ARM])
    # 两臂比较时配对差的 SD 上界≈√2·σ̂_W/√r（保守：假设创始者那份被配对完全消掉）
    mde = mde_sign_consistent(pair_noise, 12)
    need_20 = required_seeds(0.20 * base, sw, reps=2)
    need_10 = required_seeds(0.10 * base, sw, reps=2)
    print(f"  {m:<26} 基线 {base:>8.5f}  12×2 的 MDE(顶到地板口径) = {mde:>8.5f} "
          f"({100 * mde / base:+.1f}%)")
    print(f"  {'':<26} 检出 −20% 需 {need_20} 种子，−10% 需 {need_10} 种子 (r=2, p≤0.05)")

print()
print("=" * 96)
print("§5 生态护栏基线（C 阶段的「不许崩」对着这里读）")
print("=" * 96)
for m in GUARDS:
    v = rs.cell_means(ARM, m)
    icc, sb, sw = rs.icc(m, ARM)
    print(f"  {m:<20} 均值 {v.mean():>10.4f}   逐种子 [{v.min():.4f}, {v.max():.4f}]   "
          f"配对差噪声 √2·σ̂_W/√2 = {rs.pair_noise(m, [ARM]):.4f}")

print()
print("=" * 96)
print("§6 判决摘要")
print("=" * 96)
pw = res["sel_ratio_water"]
pl = res["sel_ratio_land"]
print(f"  主口径 {PRIMARY} = {pw.mean_a:.4f}（零模型 1.0）")
print(f"    差 {pw.diff.mean():+.4f}，{pw.n_pos}/12 高于零模型，p={pw.p:.5f}，"
      f"CI [{pw.ci[0]:+.4f}, {pw.ci[1]:+.4f}]，效应/噪声 {pw.ratio:+.2f}"
      f"{'（<1，功效不足）' if pw.underpowered else ''}")
print(f"  次口径 sel_ratio_land = {pl.mean_a:.4f}，差 {pl.diff.mean():+.4f}，"
      f"{pl.n_pos}/12，p={pl.p:.5f}，效应/噪声 {pl.ratio:+.2f}")
print(f"\n  两者之差正是 §12.1 要测的东西：sel_ratio_land 把「果层和种群都靠水」算进了重叠，")
print(f"  sel_ratio_water 把它扣掉。扣掉之后剩下的 {pw.mean_a:.4f} 才是「额外的果层偏好」。")
