"""果层「干旱栖息地项」旋钮：默认关时必须逐位等于旧实现。

设计与理由见 `docs/multispecies_program.md` §12.2。一句话：果层原本的栖息地项是
`forest ** 2`，而 `forest = elev_band * water_prox` 本身就是「离水近」做出来的，于是果实
必然长在所有个体本来就挤着的河岸林带里——两种食物空间交织，「两样都能吃」严格优于专精。
`fruit_dry_weight` 把果层的栖息地项从「近水的林」往「离水远」挪。

**默认 0.0 = 完全关闭**，此时 `terrain.build` 走的是与改动前逐字相同的代码路径，
所以世界逐位不变、`scripts/golden.json` 不动。本文件就是钉住这一点的。
"""
import sys

import numpy as np
import pytest

sys.path.insert(0, ".")

import jax.numpy as jnp

from underworld import Config
from underworld import terrain as terrain_mod


def _old_fruit_capacity(cfg: Config, t) -> np.ndarray:
    """改动前 `terrain.py` 那一行的独立重算，只用 `cfg` 与 `Terrain` 已导出的字段。

    刻意不引用新代码里的任何中间量——引用了就测不出「新路径等于旧路径」，只能测出
    「新路径等于它自己」。`patch` 从格心与 cfg 常量重建，`is_sea` 从 `height` 重建。
    """
    g = cfg.grid
    c = (jnp.arange(g) + 0.5) * cfg.cell_size
    cx = jnp.tile(c, g)                      # 格心 x，与 terrain.py 的 centers 同序
    cy = jnp.repeat(c, g)                    # 格心 y（cell 索引 iy*grid+ix）
    px = jnp.sin(2.0 * jnp.pi * cfg.fruit_wavenumber_x * cx / cfg.world_size)
    py = jnp.sin(2.0 * jnp.pi * cfg.fruit_wavenumber_y * cy / cfg.world_size)
    patch = jnp.clip(
        (px * py - cfg.fruit_patch_threshold) / (1.0 - cfg.fruit_patch_threshold),
        0.0, 1.0,
    )
    old = cfg.fruit_max * patch * (t.forest ** 2) * (1.0 - t.rock)
    old = jnp.where(t.height < cfg.sea_level, 0.0, old)
    return np.asarray(old)


def test_fruit_dry_weight_defaults_off():
    """默认必须是 0.0。这个旋钮一旦默认打开，全部既有结论的世界就变了。"""
    assert Config().fruit_dry_weight == 0.0


def test_fruit_dry_zero_is_bit_exact_old_formula():
    """`fruit_dry_weight=0` 时 `fruit_capacity` 与旧公式**逐位**相等（§12.2 要求的单测）。"""
    cfg = Config()
    t = terrain_mod.build(cfg)
    got = np.asarray(t.fruit_capacity)
    want = _old_fruit_capacity(cfg, t)
    assert got.shape == want.shape
    assert np.array_equal(got, want), (
        f"默认配置下果层已经偏离旧公式：最大绝对差 {np.abs(got - want).max()}"
    )


def test_fruit_dry_weight_one_decouples_from_forest():
    """`w=1` 时果层的栖息地项完全由水距决定，与 `forest` 脱钩。

    检验方式不看具体数值，而看**相关结构**：w=0 时果层承载与 `forest` 高度相关，
    w=1 时该相关应当显著下降，且果层质心的水距应当**变远**——那正是手术的目的。
    """
    t0 = terrain_mod.build(Config())
    t1 = terrain_mod.build(Config(fruit_dry_weight=1.0))
    f = np.asarray(t0.forest)
    wd = np.asarray(t0.water_dist)
    c0, c1 = np.asarray(t0.fruit_capacity), np.asarray(t1.fruit_capacity)

    assert c1.sum() > 0, "w=1 把果层整个抹掉了，那不是挪开是删除"
    corr0 = np.corrcoef(f, c0)[0, 1]
    corr1 = np.corrcoef(f, c1)[0, 1]
    assert corr1 < corr0, f"与 forest 的相关没有下降：{corr0:.3f} -> {corr1:.3f}"

    # 果层承载量加权的平均水距：挪开之后必须变远
    wd0 = float((wd * c0).sum() / c0.sum())
    wd1 = float((wd * c1).sum() / c1.sum())
    assert wd1 > wd0, f"果层没有往离水远处挪：加权水距 {wd0:.2f} -> {wd1:.2f}"


@pytest.mark.parametrize("w", [0.25, 0.5, 0.75, 1.0])
def test_fruit_dry_weight_is_capacity_neutral(w):
    """旋钮只**挪动**果层，不改变**总量**——这是 C/D 能归因的前提。

    没有这条归一化时，实测 w=1 把总承载从 536.8 抬到 1026.3（+91%）：`forest**2` 在
    几乎所有格上都很小（窄带再平方），而 `dry` 在四分之一张图上饱和到 1，所以任何朝
    `dry` 的混合都会顺带把果层加厚。那样的臂同时是「挪开了」和「厚了一倍」，效应无法
    归因给任何一个——而「加厚果层」本身已是被证伪的关闭路线（§8.3 / experiments.md §5），
    也是 §12 明写的非目标。
    """
    c0 = np.asarray(terrain_mod.build(Config()).fruit_capacity)
    cw = np.asarray(terrain_mod.build(Config(fruit_dry_weight=w)).fruit_capacity)
    assert cw.sum() == pytest.approx(c0.sum(), rel=1e-5), (
        f"w={w} 改变了果层总承载 {c0.sum():.3f} -> {cw.sum():.3f}；"
        "这个旋钮只许挪动，不许加厚"
    )


def test_fruit_dry_band_edges_are_ordered():
    """`fruit_dry_d0 < fruit_dry_d1`，否则 smoothstep 的分母为 0。"""
    cfg = Config()
    assert cfg.fruit_dry_d0 < cfg.fruit_dry_d1


@pytest.mark.parametrize("w", [0.25, 0.5, 0.75])
def test_fruit_dry_weight_is_monotone_in_water_distance(w):
    """中间剂量：果层的加权平均水距应随 `w` 单调变远，不能非单调地乱跳。"""
    wd = np.asarray(terrain_mod.build(Config()).water_dist)

    def weighted_wd(weight):
        c = np.asarray(terrain_mod.build(Config(fruit_dry_weight=weight)).fruit_capacity)
        return float((wd * c).sum() / c.sum())

    assert weighted_wd(0.0) <= weighted_wd(w) <= weighted_wd(1.0) + 1e-4
