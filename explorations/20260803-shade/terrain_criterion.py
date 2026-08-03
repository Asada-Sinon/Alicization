"""§12.3.C 判负时点名缺的那个前提，逐字量出来：**存不存在「有果实、没有草」的格子。**

`terrain_cross.py` 报的是「果富十分位处的草量 ÷ 陆地均草」，那是个比值，会被均值抹平。
这里换成三个不带均值的读数：

1. **计数**：果承载在陆地前 10%、同时草承载低于陆地均值 30% 的格子，有几个。
   基线是 **0**——这就是 §12.3.C 那句「没有一处」的字面意思。
2. **能量份额**：站在果富格上，可取用能量里果实占多少。
   演化看见的不是承载量而是能量：草 1.0/单位，果 `fruit_energy`/单位
   （`dynamics.graze` vs `dynamics.eat_fruit`，两者同吃 `eat_efficiency`）。
   专精划不划算，取决于这个份额，不取决于承载量之比。
   两个 `fruit_energy` 都报：**2.0**（默认世界）与 **4.0**（R2–P4 一直用的 `niche` 臂，
   出处 `outputs/20260803-partition/provenance.txt`）——§12.3.A' 已判定 C/D 的基线必须
   是 `niche` 臂，所以后者才是要看的那一列。
3. **反向**：草最富十分位处，果实的能量份额（应当 →0，说明草区是干净的第二个碗）。

重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260803-shade/terrain_criterion.py
"""
import sys

sys.path.insert(0, ".")

import jax
import jax.numpy as jnp
import numpy as np

from underworld import Config
from underworld import terrain as terrain_mod

# 接 `--set`，因为默认 `Config()` 不是实验跑的那个世界。
# **勘误（result-analyst 复核时发现）**：本脚本第一版写死 `Config()`（`plant_max=2.2`），
# 而 R9 的 96 个 run 跑的是 `--set plant_max=2.0`。全部**比值**读数不受影响（`plant_max`
# 是全局标量，等量归一化又保总量，所以逐条复核全对），但两个**绝对**草数差 10%：
# 草层总承载 20711.5 → 18828.7，草峰值 2.1946 → 1.9951（遮荫后 3.0511 → 2.7737）。
# 教训：地形脚本必须能跑在实验的配置上，否则「先算地形」算的是另一个世界。
import dataclasses

from underworld.config import parse_overrides

_sets = [a for i, a in enumerate(sys.argv) if i > 0 and sys.argv[i - 1] == "--set"]
cfg = dataclasses.replace(Config(), **parse_overrides(_sets))
print(f"config overrides: {parse_overrides(_sets) or '(none — 默认世界)'}\n")
T0 = terrain_mod.build(cfg)
centers = np.asarray(terrain_mod._cell_centers(cfg))
height, rock = np.asarray(T0.height), np.asarray(T0.rock)
wd_river = np.asarray(T0.water_dist)
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

px = np.sin(2.0 * np.pi * cfg.fruit_wavenumber_x * centers[:, 0] / cfg.world_size)
py = np.sin(2.0 * np.pi * cfg.fruit_wavenumber_y * centers[:, 1] / cfg.world_size)
patch = np.clip((px * py - cfg.fruit_patch_threshold) / (1.0 - cfg.fruit_patch_threshold),
                0.0, 1.0)


def build_variant(sea_fix: bool, shade: float, renorm=True):
    wd = wd_full if sea_fix else wd_river
    eb = np.exp(-((height - cfg.forest_elev) ** 2) / (2.0 * cfg.forest_elev_sigma ** 2))
    forest = np.where(is_sea, 0.0, eb * np.exp(-wd / cfg.forest_water_scale))
    fert = np.clip(cfg.grass_base + cfg.forest_bonus * forest - shade * forest, 0.0, None)
    grass = np.where(is_sea, 0.0, cfg.plant_max * fert * (1.0 - rock))
    fruit = np.where(is_sea, 0.0, cfg.fruit_max * patch * forest ** 2 * (1.0 - rock))
    if renorm and shade > 0 and grass.sum() > 0:
        grass = grass * (build_variant(sea_fix, 0.0, False)[1].sum() / grass.sum())
    return wd, grass, fruit


ARMS = [(False, 0.0), (False, 1.0), (True, 0.0), (True, 0.7), (True, 1.0),
        (True, 1.3), (True, 1.6)]

print("=" * 104)
print("判据 1：「有果实、没有草」的格子计数（果承载陆地前 10% ∩ 草承载 < 30% 陆地均值）")
print("=" * 104)
print(f'  {"sea":>4} {"shade":>6} | {"果富格数":>9} {"其中草<30%均值":>15} {"占比%":>8} '
      f'{"草<50%均值":>11} {"占比%":>8} | {"果富处草量中位/均草":>20}')
for sea_fix, shade in ARMS:
    _, grass, fruit = build_variant(sea_fix, shade)
    g, f = grass[land], fruit[land]
    rich = f >= np.quantile(f[f > 0], 0.90)
    gm = g.mean()
    n30 = int((rich & (g < 0.30 * gm)).sum())
    n50 = int((rich & (g < 0.50 * gm)).sum())
    nr = int(rich.sum())
    print(f'  {"ON" if sea_fix else "off":>4} {shade:>6.2f} | {nr:>9d} {n30:>15d} '
          f'{100 * n30 / nr:>8.1f} {n50:>11d} {100 * n50 / nr:>8.1f} | '
          f'{np.median(g[rich]) / gm:>20.3f}')

print()
print("=" * 104)
print("判据 2：能量份额。演化看见的是能量，不是承载量。")
print("=" * 104)
for fe in (2.0, 4.0):
    print(f'  fruit_energy = {fe}  ({"默认世界" if fe == 2.0 else "niche 臂 —— §12.3.A′ 定的 C/D 基线"})')
    print(f'    {"sea":>4} {"shade":>6} | {"果富处果能份额":>14} {"草富处果能份额":>14} '
          f'{"两者之差":>10} | {"全陆地果能份额":>15}')
    for sea_fix, shade in ARMS:
        _, grass, fruit = build_variant(sea_fix, shade)
        g, f = grass[land], fruit[land] * fe
        rich = f >= np.quantile(f[f > 0], 0.90)
        grich = g >= np.quantile(g, 0.90)
        s_f = f[rich].sum() / (f[rich].sum() + g[rich].sum())
        s_g = f[grich].sum() / (f[grich].sum() + g[grich].sum())
        s_all = f.sum() / (f.sum() + g.sum())
        print(f'    {"ON" if sea_fix else "off":>4} {shade:>6.2f} | {s_f:>14.3f} '
              f'{s_g:>14.3f} {s_f - s_g:>10.3f} | {s_all:>15.4f}')
    print()

print("=" * 104)
print("读法")
print("-" * 104)
print("  判据 1 的基线是 0 —— 那正是 §12.3.C「没有一处「有果实、没有草」」的字面意思。")
print("  判据 2 的「两者之差」是**专精的可得回报**：一个个体从草区搬到果区，")
print("  它食谱里果实的份额能变多少。差为 0 ⇒ 搬家不改变食性 ⇒ 中间型最优（§9.9 实测）。")
