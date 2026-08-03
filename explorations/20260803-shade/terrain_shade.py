"""**先算地形，再跑种群**（`MEMORY.md [LEARN:method]`）：草层若被林冠遮荫，两个饭碗
在空间上分得开吗？以及 `water_dist` 到底是不是它自己声明的那个量？

回答什么
--------
§12.3.C 判负时给出的决定性一条是**地形级**的：`fertility = grass_base +
forest_bonus·forest`、`grass_base=0.65` ⇒ 草层动态范围结构性封顶 1.538×、铺满整张地图，
把果层推到最远，它所在格子的草量仍是陆地均值的 **0.966 倍——没有一处「有果实、没有草」**。

本脚本问三件在花掉任何 GPU 时间之前就能回答的事：

1. **诊断**：当前世界里，林冠既是果层的家、又是**全世界草最肥的地方**吗？
   （`config.py` `fruit_max` 那段注释宣称的设计意图恰好相反：
   "the canopy shading out the herb layer … canopy is not simply more food here;
   it is a low grazing floor with a high-value exception scattered through it"。）
2. **遮荫**：给 `fertility` 减去一项 `shade · forest^p`（等量归一化，只挪不加），
   「有果实、没有草」的格子会出现吗？代价是把草层推到离水多远？
3. **`water_dist` 的语义**：`Terrain.water_dist` 注释写的是 "distance to nearest
   river/sea"，但 `build()` 算的是 `where(is_sea, 0, d_river)`——**只把海格自己归零，
   从没把「到海的距离」传播到陆地上**。差多少？海岸线是不是被当成了旱地？

全部只读 `terrain.build`，不吃 RNG，不动 `underworld/`。算一次就是答案。

重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260803-shade/terrain_shade.py
"""
import sys

sys.path.insert(0, ".")

import jax
import jax.numpy as jnp
import numpy as np

from underworld import Config
from underworld import terrain as terrain_mod

cfg = Config()
T = terrain_mod.build(cfg)

height = np.asarray(T.height)
forest = np.asarray(T.forest)
rock = np.asarray(T.rock)
grass0 = np.asarray(T.capacity)
fruit0 = np.asarray(T.fruit_capacity)
wd = np.asarray(T.water_dist)

is_sea = height < cfg.sea_level
land = ~is_sea


# ---------------------------------------------------------------- helpers
def schoener_d(p: np.ndarray, q: np.ndarray) -> float:
    p = p / p.sum()
    q = q / q.sum()
    return float(1.0 - 0.5 * np.abs(p - q).sum())


def wmean(x: np.ndarray, w: np.ndarray) -> float:
    return float((x * w).sum() / w.sum())


def wquant(x: np.ndarray, w: np.ndarray, q: float) -> float:
    """Weighted quantile — the capacity-weighted water distance is what an agent
    that settles in proportion to food would actually experience."""
    o = np.argsort(x)
    xs, ws = x[o], w[o]
    c = np.cumsum(ws) / ws.sum()
    return float(np.interp(q, c, xs))


def shaded_capacity(shade: float, power: float) -> np.ndarray:
    """What `terrain.build` would produce with a canopy-shading term, **total
    capacity held constant** (equal-supply redistribution, the same discipline
    `fruit_dry_weight` already follows: separate *where* from *how much*)."""
    fert = cfg.grass_base + cfg.forest_bonus * forest - shade * forest ** power
    fert = np.clip(fert, 0.0, None)
    cap = cfg.plant_max * fert * (1.0 - rock)
    cap = np.where(is_sea, 0.0, cap)
    return cap * (grass0.sum() / cap.sum())


# ---------------------------------------------------------------- 1. diagnosis
print("=" * 92)
print("1. 诊断：当前世界里，林冠是不是「果层的家」兼「全世界草最肥的地方」？")
print("=" * 92)

# fruit-rich = top decile of fruit capacity among land cells carrying any fruit
fr = fruit0[land]
gr = grass0[land]
fo = forest[land]
thr = np.quantile(fr[fr > 0], 0.90) if (fr > 0).any() else np.inf
rich = fr >= thr
print(f"  陆地格数                       {land.sum():>10d}")
print(f"  果层有效格数 (fruit_cap>0.01)  {(fruit0 > 0.01).sum():>10d}")
print(f"  草层承载总量                   {grass0.sum():>10.1f}")
print(f"  果层承载总量                   {fruit0.sum():>10.1f}")
print()
print(f"  草层动态范围 max/min (陆地)    {gr.max() / max(gr.min(), 1e-9):>10.3f}")
print(f"  草层 P90/P10 (陆地)            {np.quantile(gr, .9) / np.quantile(gr, .1):>10.3f}")
print(f"  fertility 区间                 "
      f"[{cfg.grass_base:.2f}, {cfg.grass_base + cfg.forest_bonus:.2f}]"
      f"  ⇒ 封顶 {1 + cfg.forest_bonus / cfg.grass_base:.3f}×")
print()
print(f"  【关键】果层最富十分位处的草量 ÷ 陆地均草   "
      f"{gr[rich].mean() / gr.mean():>8.3f}   ← §12.3.C 报的是 0.966")
print(f"  r(草, 果)  Pearson  (陆地)                  "
      f"{np.corrcoef(gr, fr)[0, 1]:>8.3f}")
print(f"  r(草, forest) Pearson (陆地)                "
      f"{np.corrcoef(gr, fo)[0, 1]:>8.3f}")
print(f"  Schoener D(草, 果)                          {schoener_d(grass0, fruit0):>8.4f}")
print()
print("  意图 vs 实现：`config.py` 的 `fruit_max` 注释写的是「林冠遮蔽草本层 ⇒ 林下是")
print("  *低*放牧底 + 高值例外」，而 `terrain.py:192` 写的是 `+ forest_bonus·forest`——")
print(f"  满冠层的草承载是空地的 {1 + cfg.forest_bonus / cfg.grass_base:.3f} 倍，**林冠是全世界草最肥的地方**。")
print("  三轮资源分割实验死在的那个几何，就是这一个正负号。")


# ---------------------------------------------------------------- 2. shade sweep
print()
print("=" * 92)
print("2. 遮荫扫描：fertility = grass_base + forest_bonus·forest − shade·forest^p（等量归一化）")
print("=" * 92)
print(f'  {"p":>3} {"shade":>6} {"果富处草/均草":>13} {"r(草,果)":>9} {"D(草,果)":>9} '
      f'{"草重心水距":>11} {"草P75水距":>10} {"草P90水距":>10} {"零草格%":>8} {"草动态范围":>10}')

land_wd = wd[land]
for power in (1.0, 2.0):
    for shade in (0.0, 0.2, 0.35, 0.5, 0.7, 1.0):
        cap = shaded_capacity(shade, power)
        c = cap[land]
        if c.sum() <= 0:
            print(f"  {power:>3.0f} {shade:>6.2f}   草层被清零，跳过")
            continue
        ratio = c[rich].mean() / c.mean()
        r = np.corrcoef(c, fr)[0, 1]
        d = schoener_d(cap, fruit0)
        cw = wmean(land_wd, c)
        p75 = wquant(land_wd, c, 0.75)
        p90 = wquant(land_wd, c, 0.90)
        zero = 100.0 * float((c < 1e-6).mean())
        dyn = c.max() / max(c[c > 1e-9].min(), 1e-9)
        print(f"  {power:>3.0f} {shade:>6.2f} {ratio:>13.3f} {r:>9.3f} {d:>9.4f} "
              f"{cw:>11.2f} {p75:>10.2f} {p90:>10.2f} {zero:>8.1f} {dyn:>10.2f}")

print()
print("  基线（shade=0）的草重心水距 = "
      f"{wmean(land_wd, grass0[land]):.2f}，陆地水距中位数 = {np.median(land_wd):.2f}。")
print("  §12.3.C 的教训：`herb_water_dist` 基线 45.4，w=0.25 时 +4.93 SE 就足以破渴死护栏。")
print("  ⇒ 草重心被推出多远，是本方案会不会撞上水墙的**唯一**先验读数。")


# ---------------------------------------------------------------- 3. water_dist
print()
print("=" * 92)
print("3. `water_dist` 的语义：声明是「到最近的河或海」，实现是「到最近的河，海格归零」")
print("=" * 92)

centers = np.asarray(terrain_mod._cell_centers(cfg))
sea_pts = jnp.asarray(centers[is_sea])
pts = jnp.asarray(centers)


def _wrap(d, size):
    return (d + size / 2.0) % size - size / 2.0


@jax.jit
def dist_to_sea(points, sea):
    def body(i, best):
        blk = jax.lax.dynamic_slice(sea, (i * 256, 0), (256, 2))
        dd = _wrap(points[:, None, :] - blk[None, :, :], cfg.world_size)
        return jnp.minimum(best, jnp.min(jnp.sqrt(jnp.sum(dd * dd, 2) + 1e-12), 1))
    n_blk = sea.shape[0] // 256
    return jax.lax.fori_loop(0, n_blk, body, jnp.full((points.shape[0],), jnp.inf))


n_sea = int(is_sea.sum())
if n_sea >= 256:
    trim = (n_sea // 256) * 256
    d_sea = np.asarray(dist_to_sea(pts, sea_pts[:trim]))
    true_wd = np.minimum(wd, np.where(is_sea, 0.0, d_sea))
    gap = wd - true_wd
    print(f"  海格数 {n_sea}（用了前 {trim} 个作距离源，误差 ≤ 一个格 {cfg.cell_size:.1f}）")
    print(f"  受影响陆地格（水距被高估 >1 单位）      {int((gap[land] > 1).sum()):>8d}"
          f"  ({100.0 * (gap[land] > 1).mean():.1f}%)")
    print(f"  被高估的中位幅度（仅受影响格）          "
          f"{np.median(gap[land][gap[land] > 1]) if (gap[land] > 1).any() else 0:>8.1f}")
    print(f"  最大高估                                {gap[land].max():>8.1f}")
    print(f"  陆地水距中位数  现状 {np.median(wd[land]):>6.1f} → 按声明语义 "
          f"{np.median(true_wd[land]):>6.1f}")
    drinkable_now = (wd[land] < cfg.river_half_width).mean()
    drinkable_fix = (true_wd[land] < cfg.river_half_width).mean()
    print(f"  可直接饮水的陆地格占比  现状 {100 * drinkable_now:>5.2f}%  → "
          f"{100 * drinkable_fix:>5.2f}%")
    # what the coast looks like: land cells within one cell of the sea
    coast = land & (d_sea < 2.0 * cfg.cell_size)
    print()
    print(f"  海岸带（离海 < {2 * cfg.cell_size:.0f} 单位的陆地格）{int(coast.sum())} 格：")
    print(f"    现状 water_dist 中位数   {np.median(wd[coast]):>8.1f}"
          f"   ← 站在海边，被判成离水这么远")
    print(f"    forest 均值              {forest[coast].mean():>8.3f}"
          f"   (全陆地 {forest[land].mean():.3f})")
    print(f"    草承载均值               {grass0[coast].mean():>8.3f}"
          f"   (全陆地 {grass0[land].mean():.3f})")
    print(f"    果承载均值               {fruit0[coast].mean():>8.4f}"
          f"   (全陆地 {fruit0[land].mean():.4f})")
else:
    print(f"  海格只有 {n_sea} 个，分块最小 256，跳过。")

print()
print("=" * 92)
