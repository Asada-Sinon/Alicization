"""两个饭碗真的拉开了吗？——**格子级**的分离度，不是重心距离。**只依赖地形，无 RNG。**

回答什么：§12 的目标是「让专精果实与专精草的个体落在不同的地方」。§12.3.B 补的
「重心间距」是一阶矩，**它可以在两个场高度重合时依然很大**（果层集中、草层摊满全图，
两个重心自然分得开）。个体感受到的是**它站的那一格上两种食物各有多少**，所以要问的是：

  1. `corr(grass_capacity, fruit_capacity)`——一格对果实好，是不是也对草好？
  2. **果层所在处的草有多少**：`Σ(fruit·grass)/Σfruit ÷ 陆地平均 grass`。
     若 ≈1，跑去吃果实**不用放弃草**——那就没有饭碗可分，只有一次搬家。
  3. 草层在干旱带（`water_dist > fruit_dry_d0=45`）里占多大份额。
  4. 草层承载力沿水距的剖面——它到底是不是「河边高、内陆低」。

**诚实标注：这不是 §12.3.C 的预注册读数，是在看到 C 数据之后补的。** 补它的理由与
`terrain_separation.py` 相同：完全不依赖任何 run，地形表在跑任何 run 之前就已确定，
没有「看了结果再挑口径」的空间。

重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260803-dryC/bowl_separation.py
"""
import sys

sys.path.insert(0, ".")

import numpy as np

from underworld import Config
from underworld import terrain as terrain_mod

WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
# niche 捆绑里唯一影响 terrain 的字段是 plant_max=2.0（默认 2.2）。它只给 `capacity`
# 乘一个常数，所以相关系数/归一化比值/分位排名都不受影响，但**绝对草承载值受影响**，
# 所以这里照 niche 建，不用默认。`fruit_max` 未被 niche 覆盖，果层场与默认一致。
NICHE = dict(plant_max=2.0)
D0 = Config().fruit_dry_d0
D1 = Config().fruit_dry_d1

base = terrain_mod.build(Config(**NICHE))
grass = np.asarray(base.capacity, dtype=np.float64)
wd = np.asarray(base.water_dist, dtype=np.float64)
land = grass > 0.0
g = grass[land]
w = wd[land]

print("=" * 96)
print(f"格子级饭碗分离度（terrain.build，无 RNG）  d0={D0} d1={D1}  陆地格 {land.sum()}")
print("=" * 96)
print(f'  {"w":>5} {"corr(草,果)":>12} {"果处草量/陆地均值":>18} {"草处果量/陆地均值":>18}'
      f' {"果层在干旱带份额":>18}')
for ww in WEIGHTS:
    fc = np.asarray(terrain_mod.build(Config(fruit_dry_weight=ww, **NICHE)).fruit_capacity,
                    dtype=np.float64)[land]
    r = float(np.corrcoef(g, fc)[0, 1])
    grass_at_fruit = float((fc * g).sum() / fc.sum()) / float(g.mean())
    fruit_at_grass = float((g * fc).sum() / g.sum()) / float(fc.mean())
    dry_share = float(fc[w > D0].sum() / fc.sum())
    print(f"  {ww:>5} {r:>12.4f} {grass_at_fruit:>18.4f} {fruit_at_grass:>18.4f}"
          f" {dry_share:>18.4f}")

print(f"\n  草层本身在干旱带 (wd>{D0}) 的份额 = {float(g[w > D0].sum() / g.sum()):.4f}"
      f"   （干旱带占陆地格 {float((w > D0).mean()):.4f}）")

print("\n  草层承载力沿水距的剖面（陆地格，等宽 8 个箱）：")
edges = np.linspace(0.0, w.max() + 1e-6, 9)
print(f'  {"水距区间":>16} {"格数":>8} {"平均草承载":>12} {"占草层总量":>12}')
for i in range(8):
    sel = (w >= edges[i]) & (w < edges[i + 1])
    if not sel.any():
        continue
    print(f"  {edges[i]:>7.1f}-{edges[i+1]:<8.1f} {int(sel.sum()):>8}"
          f" {g[sel].mean():>12.4f} {g[sel].sum()/g.sum():>12.4f}")

print("\n  读法：若「果处草量/陆地均值」≈1，跑去吃果实不用放弃草——两个饭碗在格子级并没有分开，")
print("  分开的只是重心。这时把果层推远只是给全体加了一份水成本，不是造第二个生态位。")

# ------------------------------------------------------------------------------------
# 草层能不能提供「果富草贫」的格子？——这是几何的硬上限，与 fruit_dry_weight 无关。
# capacity = plant_max * (grass_base + forest_bonus*forest) * (1-rock)
# fertility 只能在 [grass_base, grass_base+forest_bonus] 之间动，所以草层的动态范围是
# 结构性封顶的。没有草贫的格子，就没有能惩罚「两样都吃」的地方。
cfg = Config(**NICHE)
print()
print("=" * 96)
print("草层的结构性动态范围（决定「果富草贫」的格子存不存在）")
print("=" * 96)
print(f"  fertility = grass_base({cfg.grass_base}) + forest_bonus({cfg.forest_bonus})*forest"
      f"  ->  [{cfg.grass_base}, {cfg.grass_base + cfg.forest_bonus}]，比值上限 "
      f"{(cfg.grass_base + cfg.forest_bonus) / cfg.grass_base:.3f}x")
qs = [0, 5, 10, 25, 50, 75, 90, 95, 100]
print(f"  陆地草承载分位（plant_max=2.0 的 niche 世界，本轮 15 个 run 全用它）：")
print("    " + "  ".join(f"p{q}={np.percentile(g, q):.3f}" for q in qs))
print(f"    均值={g.mean():.4f}  SD={g.std(ddof=1):.4f}  CV={g.std(ddof=1)/g.mean():.4f}")
print("\n  各臂的果层落在草层的哪个分位上（果层承载加权的草量 -> 对应陆地分位）：")
for ww in WEIGHTS:
    fc = np.asarray(terrain_mod.build(Config(fruit_dry_weight=ww, **NICHE)).fruit_capacity,
                    dtype=np.float64)[land]
    gm = float((fc * g).sum() / fc.sum())
    pct = float((g < gm).mean() * 100)
    print(f"    w={ww:<5} 果层加权草量={gm:.4f}  = 陆地草分布的第 {pct:.1f} 百分位")
print("\n  读法：果层加权草量始终坐在陆地草分布的中位数附近 -> **不存在「果富草贫」的栖息地**。")
print("  手术能造出的唯一差别成本是**水**，不是**食物**。")
