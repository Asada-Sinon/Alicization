"""草层的「林下遮荫」旋钮，以及 `water_dist` 的「认海」旋钮。

设计与理由见 `docs/multispecies_program.md` §13。一句话：`config.py` 在 `fruit_max` 上方
声明林冠应当是「低放牧底 + 高值例外」，而 `terrain.py` 给了满冠层 1.538× 的草承载——
**林冠是全世界草最肥的地方**，而果层又恰好由 `forest**2` 门控长在那里。两个饭碗完全共址，
「两样都能吃」于是严格优于专精，三轮资源分割实验死在这个几何上。

**两个旋钮默认都关，且关的时候逐位等于旧实现**，所以 `scripts/golden.json` 不动。
本文件钉住这一点，外加三条只有真旋钮才做得到的行为。
"""
import sys

import numpy as np
import pytest

sys.path.insert(0, ".")

from underworld import Config
from underworld import terrain as terrain_mod


# --------------------------------------------------------------- 默认关
def test_grass_shade_default_is_truly_off():
    """默认 0.0 时 `capacity` 必须逐位等于「没有这个旋钮」的算式。

    独立重算，不引用新代码里的任何中间量——引用了就只能测出「新路径等于它自己」。
    """
    cfg = Config()
    t = terrain_mod.build(cfg)
    fert = cfg.grass_base + cfg.forest_bonus * np.asarray(t.forest)
    is_sea = np.asarray(t.height) < cfg.sea_level
    expect = np.where(is_sea, 0.0,
                      cfg.plant_max * fert * (1.0 - np.asarray(t.rock)))
    assert np.array_equal(np.asarray(t.capacity), expect.astype(np.float32))


def test_water_sea_dist_default_is_truly_off():
    """默认 False 时整张地形逐位不变——包括由 `water_dist` 派生的 forest / 两个承载。"""
    a = terrain_mod.build(Config())
    b = terrain_mod.build(Config(water_sea_dist=False))
    for name in ("water_dist", "forest", "capacity", "fruit_capacity"):
        assert np.array_equal(np.asarray(getattr(a, name)),
                              np.asarray(getattr(b, name))), name


# --------------------------------------------------------------- 只挪不加
@pytest.mark.parametrize("s", [0.35, 0.7, 1.0, 1.3, 1.6])
def test_grass_shade_is_capacity_neutral(s):
    """旋钮只**挪动**草层，不改**总量**——这是效应可归因的前提。

    没有归一化时，遮荫同时是「结构变了」和「食物少了」，而承载力本身已经被 §8.2 判定为
    这个世界种群量的主导项（超调与渴死增量 ρ=+0.90）。那样的臂什么都归因不了。
    """
    c0 = np.asarray(terrain_mod.build(Config()).capacity)
    cs = np.asarray(terrain_mod.build(Config(grass_shade=s)).capacity)
    assert cs.sum() == pytest.approx(c0.sum(), rel=1e-5), (
        f"grass_shade={s} 改变了草层总承载 {c0.sum():.3f} -> {cs.sum():.3f}"
    )


# --------------------------------------------------------------- 独立实现对照
def test_grass_shade_below_one_equals_negative_forest_bonus():
    """`grass_shade <= 1.0` 有一个**零代码等价实现**，拿它当独立对照。

    `forest_bonus` 没有 clamp，全仓唯一的消费者就是 `terrain.py` 那一行，所以
    `grass_base + (forest_bonus - s)·forest ≡ grass_base + forest_bonus'·forest`。
    此时 `clip(...,0,None)` 还没生效（`forest<=1` ⇒ `0.65-0.65·forest >= 0`），
    两条路径只差一个全局标量，而那个标量正是归一化要除掉的东西。

    这条测试的价值不在于「旋钮能用」，在于它是**另一个人写的同一个算式**：
    实现写错了，这里会红，而 capacity-neutral 那条不会。
    """
    a = np.asarray(terrain_mod.build(Config(grass_shade=1.0)).capacity)
    b = np.asarray(terrain_mod.build(Config(forest_bonus=-0.65)).capacity)
    b = b * (a.sum() / b.sum())          # 归一化掉那个全局标量
    assert np.allclose(a, b, rtol=1e-5, atol=1e-6), (
        f"grass_shade=1.0 与 forest_bonus=-0.65 不一致，max|Δ|={np.abs(a - b).max():.3e}"
    )


# --------------------------------------------------------------- 行为
def test_grass_shade_creates_fruit_cells_without_grass():
    """§12.3.C 判负时点名缺的那个前提：**存在「有果实、没有草」的格子。**

    默认世界里这个数是 **0**——不是「很少」，是一个也没有。这条测试就是那句
    「没有一处」的可执行形式，同时钉住旋钮必须开到 1.0 以上才有用（`clip` 才起作用）。
    """
    def count(cfg):
        t = terrain_mod.build(cfg)
        land = np.asarray(t.height) >= cfg.sea_level
        g = np.asarray(t.capacity)[land]
        f = np.asarray(t.fruit_capacity)[land]
        rich = f >= np.quantile(f[f > 0], 0.90)
        return int((rich & (g < 0.30 * g.mean())).sum())

    assert count(Config()) == 0, "基线本应一个「有果实、没有草」的格子都没有"
    assert count(Config(grass_shade=1.0)) > 0
    assert count(Config(grass_shade=1.3)) > 100, "1.3 应当把大部分果富格清成无草"


def test_water_sea_dist_reaches_the_coast():
    """认海之后，紧贴大海的陆地不再被判成旱地；且水距只许下降，不许上升。"""
    cfg = Config(water_sea_dist=True)
    base = terrain_mod.build(Config())
    fixed = terrain_mod.build(cfg)
    w0 = np.asarray(base.water_dist)
    w1 = np.asarray(fixed.water_dist)

    assert np.all(w1 <= w0 + 1e-5), "min(d_river, d_sea) 不可能比 d_river 大"

    # 海岸带 = 与海格相邻的陆地。用 4-邻域在网格上找，避免再算一次距离场。
    g = cfg.grid
    is_sea = (np.asarray(base.height) < cfg.sea_level).reshape(g, g)
    nbr = (np.roll(is_sea, 1, 0) | np.roll(is_sea, -1, 0)
           | np.roll(is_sea, 1, 1) | np.roll(is_sea, -1, 1))
    coast = (nbr & ~is_sea).reshape(-1)
    assert coast.sum() > 0

    assert np.median(w0[coast]) > 3.0 * cfg.cell_size, (
        "前提不成立：海岸带在旧场里本应读作离水很远")
    assert np.all(w1[coast] <= cfg.cell_size * 1.5), (
        f"海岸带认海后仍偏远：max={w1[coast].max():.2f}")


def test_dist_to_sea_matches_brute_force():
    """`_dist_to_sea` 的分块 + 掩码 + 环面包裹，对着朴素全对矩阵逐格核对。

    这是这两个旋钮里唯一有「算错了不报错」风险的一段：分块偏移错一格、掩码取反、
    忘了 `_wrap`——三者都只会返回一个看起来合理的距离场。默认网格的全对矩阵是 1 GiB，
    所以在小网格上核对（小网格只是为了让朴素算法跑得动，被测的是同一段代码）。

    **不用 `river_half_width` 那类「河边格不该动」的判据**：河口的格子既近河又近海，
    认海之后它合法地变小了。第一版测试就是这么写错的。
    """
    cfg = Config(grid=48, water_sea_dist=True)
    centers = np.asarray(terrain_mod._cell_centers(cfg))
    height = np.asarray(terrain_mod.height_at(terrain_mod._cell_centers(cfg), cfg))
    is_sea = height < cfg.sea_level
    assert is_sea.sum() > 0, "小网格上没有海格，这条测试就没在测东西"

    mine = np.asarray(terrain_mod._dist_to_sea(
        terrain_mod._cell_centers(cfg), terrain_mod._cell_centers(cfg),
        height < cfg.sea_level, cfg))

    d = centers[:, None, :] - centers[None, is_sea, :]
    d = (d + cfg.world_size / 2.0) % cfg.world_size - cfg.world_size / 2.0
    brute = np.sqrt(np.sum(d * d, axis=2)).min(axis=1)

    assert np.allclose(mine, brute, atol=1e-3), (
        f"max|Δ|={np.abs(mine - brute).max():.3e}")
    # 海格到自己的距离是 sqrt(1e-12)=1e-6，不是 0——那个 eps 是 `_dist_to_rivers`
    # 就有的写法（保住 sqrt 在 0 处的梯度）。`build` 里 `min(0.0, 1e-6)` 仍然是 0，
    # 所以行为上没有差别，但断言要照实写。
    assert np.all(mine[is_sea] <= 1e-5), "海格到海的距离必须是 0（容 sqrt 的 eps）"
