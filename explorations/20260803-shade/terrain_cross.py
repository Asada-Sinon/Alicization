"""把两个改动交叉起来算地形：**林下遮荫** × **`water_dist` 认海**。

为什么必须交叉，不能分开看
--------------------------
`terrain_shade.py` 单独测出两件事：

- 遮荫能把「果富处草量 ÷ 陆地均草」从 **1.307 压到 0.491**（p=1, shade=1.0），
  即第一次造出「有果实、没有草」的格子——§12.3.C 判负时点名缺的就是这个前提。
  **代价是草层重心被推离水 49.75 → 59.27（+19%）**，而水墙是这个世界最硬的一堵。
- `water_dist` 声明是「到最近的河**或海**」，实现是「到最近的河，海格归零」。
  27.2% 的陆地格水距被高估，中位高估 43.3 单位；**站在海边的陆地格中位水距 54.7**。

这两件事的方向是相反的：遮荫把草推离河，认海把水加到海岸线上——而海岸带恰好
`forest` 0.123（陆地均值 0.274）、果承载 0.0163（陆地 0.0396），**本来就是开阔草地**。
所以要问的不是「遮荫会不会撞水墙」，而是「**认海之后还撞不撞**」。

`water_dist` 进 `forest = elev_band · exp(−water_dist/scale)`，所以认海会同时改
`capacity` 与 `fruit_capacity`——**必须整张地形重算，不能只换水距那一列。**

全部只读、无 RNG、不动 `underworld/`。

重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260803-shade/terrain_cross.py
"""
import sys

sys.path.insert(0, ".")

import jax
import jax.numpy as jnp
import numpy as np

from underworld import Config
from underworld import terrain as terrain_mod

# 接 `--set` —— 见 `terrain_criterion.py` 顶部记的同一处勘误：第一版写死 `Config()`
# （`plant_max=2.2`），而 R9 跑的是 `plant_max=2.0`。比值不受影响，绝对草数差 10%。
import dataclasses

from underworld.config import parse_overrides

_sets = [a for i, a in enumerate(sys.argv) if i > 0 and sys.argv[i - 1] == "--set"]
cfg = dataclasses.replace(Config(), **parse_overrides(_sets))
print(f"config overrides: {parse_overrides(_sets) or '(none — 默认世界)'}\n")
T0 = terrain_mod.build(cfg)

centers = np.asarray(terrain_mod._cell_centers(cfg))
height = np.asarray(T0.height)
rock = np.asarray(T0.rock)
wd_river = np.asarray(T0.water_dist)          # = where(is_sea, 0, d_river)
is_sea = height < cfg.sea_level
land = ~is_sea


def _wrap(d, size):
    return (d + size / 2.0) % size - size / 2.0


@jax.jit
def _dist_to(points, src):
    def body(i, best):
        blk = jax.lax.dynamic_slice(src, (i * 256, 0), (256, 2))
        dd = _wrap(points[:, None, :] - blk[None, :, :], cfg.world_size)
        return jnp.minimum(best, jnp.min(jnp.sqrt(jnp.sum(dd * dd, 2) + 1e-12), 1))
    return jax.lax.fori_loop(0, src.shape[0] // 256, body,
                             jnp.full((points.shape[0],), jnp.inf))


n_trim = (int(is_sea.sum()) // 256) * 256
d_sea = np.asarray(_dist_to(jnp.asarray(centers), jnp.asarray(centers[is_sea][:n_trim])))
wd_full = np.minimum(wd_river, np.where(is_sea, 0.0, d_sea))

# `patch` is the fruit sine lattice -- terrain.build computes it inline, so redo it here
px = np.sin(2.0 * np.pi * cfg.fruit_wavenumber_x * centers[:, 0] / cfg.world_size)
py = np.sin(2.0 * np.pi * cfg.fruit_wavenumber_y * centers[:, 1] / cfg.world_size)
patch = np.clip((px * py - cfg.fruit_patch_threshold) / (1.0 - cfg.fruit_patch_threshold),
                0.0, 1.0)


def build_variant(sea_fix: bool, shade: float, power: float, renorm: bool = True):
    """Recompute the whole terrain the way `terrain.build` would under both knobs."""
    wd = wd_full if sea_fix else wd_river
    elev_band = np.exp(-((height - cfg.forest_elev) ** 2)
                       / (2.0 * cfg.forest_elev_sigma ** 2))
    forest = np.where(is_sea, 0.0, elev_band * np.exp(-wd / cfg.forest_water_scale))

    fert = np.clip(cfg.grass_base + cfg.forest_bonus * forest - shade * forest ** power,
                   0.0, None)
    grass = np.where(is_sea, 0.0, cfg.plant_max * fert * (1.0 - rock))
    fruit = np.where(is_sea, 0.0, cfg.fruit_max * patch * forest ** 2 * (1.0 - rock))
    if renorm and shade > 0 and grass.sum() > 0:
        # equal-supply redistribution: the shade knob moves grass, never adds or
        # removes it. Referenced against the SAME sea_fix arm, so the two knobs
        # stay attributable separately.
        ref = build_variant(sea_fix, 0.0, power, renorm=False)[1]
        grass = grass * (ref.sum() / grass.sum())
    return wd, grass, fruit, forest


def schoener_d(p, q):
    p, q = p / p.sum(), q / q.sum()
    return float(1.0 - 0.5 * np.abs(p - q).sum())


def wmean(x, w):
    return float((x * w).sum() / w.sum())


def wq(x, w, q):
    o = np.argsort(x)
    return float(np.interp(q, np.cumsum(w[o]) / w[o].sum(), x[o]))


# fruit-rich decile is defined per-variant (the fruit field itself moves with sea_fix)
def report(sea_fix, shade, power):
    wd, grass, fruit, forest = build_variant(sea_fix, shade, power)
    g, f, w = grass[land], fruit[land], wd[land]
    thr = np.quantile(f[f > 0], 0.90)
    rich = f >= thr
    poor = f <= 1e-4                     # cells with no fruit at all
    return dict(
        # how much of the map the shade term zeroes outright, and how far the
        # renormalisation then has to scale the rest up
        zero_pct=100.0 * float((g < 1e-6).mean()),
        peak=g.max(),
        ratio=g[rich].mean() / g.mean(),
        r=np.corrcoef(g, f)[0, 1],
        D=schoener_d(grass, fruit),
        gwd=wmean(w, g),
        gp75=wq(w, g, 0.75),
        fwd=wmean(w, f),
        gsum=grass.sum(),
        fsum=fruit.sum(),
        # what fraction of the grass supply sits where there is NO fruit at all,
        # i.e. how much of the world is a genuine second bowl
        grass_in_fruitfree=100.0 * g[poor].sum() / g.sum(),
        drink=100.0 * (w < cfg.river_half_width).mean(),
    )


print("=" * 108)
print("交叉表：sea_fix × shade（p = 遮荫门控指数）。草层等量归一化，果层不动。")
print("=" * 108)
print(f'  {"sea":>4} {"shade":>6} | {"果富处草/均草":>13} {"r(草,果)":>8} '
      f'{"D":>7} | {"草重心水距":>11} {"草P75水距":>10} {"果重心水距":>11} '
      f'{"可饮格%":>8} | {"零草格%":>8} {"草峰值":>8} {"草总量":>9} {"果总量":>8}')
for sea_fix in (False, True):
    for shade in (0.0, 0.5, 0.7, 1.0, 1.3, 1.6):
        m = report(sea_fix, shade, 1.0)
        print(f'  {"ON" if sea_fix else "off":>4} {shade:>6.2f} | '
              f'{m["ratio"]:>13.3f} {m["r"]:>8.3f} {m["D"]:>7.4f} | '
              f'{m["gwd"]:>11.2f} {m["gp75"]:>10.2f} {m["fwd"]:>11.2f} '
              f'{m["drink"]:>8.2f} | {m["zero_pct"]:>8.1f} {m["peak"]:>8.3f} '
              f'{m["gsum"]:>9.1f} {m["fsum"]:>8.1f}')
    print()

print("读法")
print("-" * 108)
b = report(False, 0.0, 1.0)
for sea_fix, shade in ((False, 1.0), (True, 0.0), (True, 1.0), (True, 1.3), (True, 1.6)):
    m = report(sea_fix, shade, 1.0)
    tag = f'sea={"ON" if sea_fix else "off"} shade={shade}'
    print(f'  {tag:<22} 果富处草量 {b["ratio"]:.3f}→{m["ratio"]:.3f}   '
          f'草重心水距 {b["gwd"]:.2f}→{m["gwd"]:.2f}   '
          f'可饮格 {b["drink"]:.2f}%→{m["drink"]:.2f}%   '
          f'果总量 {b["fsum"]:.1f}→{m["fsum"]:.1f}')
print()
print("  判据（§12.3.C 判负时缺的那个前提）：**要存在「有果实、没有草」的格子。**")
print("  单靠遮荫能做到，但把草推离水；认海把水铺到海岸线，而海岸带本来就是开阔草地。")
print("  两个数一起看，才知道这条路是不是真的能走。")
