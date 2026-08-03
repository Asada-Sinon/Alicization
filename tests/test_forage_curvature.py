"""权衡前沿的**形状**：`forage_curvature`。

设计与理由见 `docs/multispecies_feasibility.md` §11。一句话：现在的
`grass + fruit ≡ 2` 是一条**直线前沿**，而直线前沿对入侵适应度的贡献曲率**恒为 0**——
`W(s) = A·grass(s) + B·fruit(s)` 对任何 `A`、`B` 都是 `s` 的线性函数。所以中间型
**永远不可能**是适应度极小值，分歧选择没有一阶项可以来自；果层份额多大、交配怎么组织，
都改变不了这一点。五条归档阴性（§9.6 / §9.10 / §9.11 / §11 / §12 / R9）共用这一个成因。

`forage_curvature = k` 把两个乘子重标到前沿 `grass**k + fruit**k = 2` 上：
**k=1 是今天的直线且逐位等价**（分支编译期跳过），**k<1 让专精划算**，k>1 反而偏袒通才。
本文件钉住这四件事。
"""
import sys

import numpy as np
import pytest

sys.path.insert(0, ".")

import jax
import jax.numpy as jnp

from underworld import Config
from underworld.dynamics import _forage_pref_scale


class _FakeState:
    """只带 `genome` 的最小 state——`_forage_pref_scale` 只读这一个字段。"""

    def __init__(self, pref, cfg):
        # forage_pref_of 是 sigmoid(genome[:, idx])，反解出想要的 pref
        p = np.clip(np.asarray(pref, dtype=np.float64), 1e-6, 1 - 1e-6)
        g = np.zeros((len(p), cfg.genome_size), dtype=np.float32)
        g[:, cfg.forage_pref_index] = np.log(p / (1 - p))
        self.genome = jnp.asarray(g)


PREF = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])


def _mults(k, t=1.0):
    cfg = Config(forage_tradeoff=t, forage_curvature=k)
    g, f = _forage_pref_scale(_FakeState(PREF, cfg), cfg)
    return np.asarray(g, dtype=np.float64), np.asarray(f, dtype=np.float64)


def test_curvature_default_is_truly_off():
    """k=1.0 必须逐位等于「没有这个旋钮」的算式。

    独立重算，不引用新代码的任何中间量。
    """
    cfg = Config(forage_tradeoff=1.0)
    assert cfg.forage_curvature == 1.0
    st = _FakeState(PREF, cfg)
    g, f = _forage_pref_scale(st, cfg)
    s = 2.0 * jax.nn.sigmoid(st.genome[:, cfg.forage_pref_index]) - 1.0
    assert np.array_equal(np.asarray(g),
                          np.asarray(jnp.clip(1.0 + cfg.forage_tradeoff * s, 0.0, None)))
    assert np.array_equal(np.asarray(f),
                          np.asarray(jnp.clip(1.0 - cfg.forage_tradeoff * s, 0.0, None)))


def test_unbiased_forager_is_unchanged_for_every_k():
    """`pref=0.5`（基因 0，无偏采食者）在任何 `k` 下都必须是 (1, 1)。

    这是「起点世界不变」的保证——没有它，改 k 就同时改了初始种群的采食效率，
    效应无法归因给曲率（与 `fruit_dry_weight` / `grass_shade` 的等量归一化同一条纪律）。
    """
    for k in (0.3, 0.5, 0.8, 1.0, 1.5, 2.0):
        g, f = _mults(k)
        mid = len(PREF) // 2                     # pref=0.5
        assert g[mid] == pytest.approx(1.0, abs=1e-5), k
        assert f[mid] == pytest.approx(1.0, abs=1e-5), k


@pytest.mark.parametrize("k", [0.3, 0.5, 0.8, 1.5, 2.0])
def test_multipliers_land_on_the_frontier(k):
    """`grass**k + fruit**k == 2` —— 这就是这个旋钮的定义。"""
    g, f = _mults(k)
    assert np.allclose(g ** k + f ** k, 2.0, rtol=1e-4), (
        f"k={k} 不在前沿上: {g ** k + f ** k}")


def test_below_one_rewards_specialists_above_one_punishes_them():
    """曲率方向。**这条是本文件的重点**——审查意见把方向写反过一次。

    对称回报（A=B=1）下的总回报 `g+f`：k<1 时专精者 > 通才（中间型是适应度极小 ⇒
    分歧选择可能），k>1 时反过来（中间型是极大 ⇒ 稳定化），k=1 处处相等（刀刃）。
    """
    mid = len(PREF) // 2
    for k, expect in ((0.3, "reward"), (0.5, "reward"), (1.0, "flat"), (2.0, "punish")):
        g, f = _mults(k)
        spec, gen = g[-1] + f[-1], g[mid] + f[mid]
        if expect == "reward":
            assert spec > gen * 1.05, f"k={k} 本应奖励专精: {spec:.4f} vs {gen:.4f}"
        elif expect == "punish":
            assert spec < gen * 0.95, f"k={k} 本应惩罚专精: {spec:.4f} vs {gen:.4f}"
        else:
            assert spec == pytest.approx(gen, rel=1e-4)


def test_linear_frontier_has_exactly_zero_fitness_curvature():
    """立论本身，写成可执行的：直线前沿下 `W(s)=A·g+B·f` 的二阶导对任何 A、B 恒为 0。

    这条不测代码，测的是**为什么要有这个旋钮**。它红了说明上面那段论证垮了。
    """
    cfg = Config(forage_tradeoff=1.0)
    s = np.linspace(-1.0, 1.0, 9)
    pref = 0.5 * (s + 1.0)
    g, f = _forage_pref_scale(_FakeState(pref, cfg), cfg)
    g, f = np.asarray(g, dtype=np.float64), np.asarray(f, dtype=np.float64)
    # k<1 下同一个量必须为正（中间型是极小值）。先算它，好拿来当**尺度**。
    cfg2 = Config(forage_tradeoff=1.0, forage_curvature=0.5)
    g2, f2 = _forage_pref_scale(_FakeState(pref, cfg2), cfg2)
    W2 = np.asarray(g2, dtype=np.float64) + np.asarray(f2, dtype=np.float64)
    curved = np.gradient(np.gradient(W2, s), s)[len(s) // 2]
    assert curved > 0.1, f"k=0.5 本应把中间型变成适应度极小: {curved}"

    # 判据对着**信号量级**定，不拍一个 epsilon：直线前沿的残余曲率必须比 k=0.5 的曲率
    # 小三个数量级以上。残余不是恒等 0 是因为 `_FakeState` 用 logit 编码 pref、
    # `forage_pref_of` 再 sigmoid 回来，中间过一次 f32（观测残余 ~1e-5，而
    # `np.gradient` 求两次导会把误差放大 1/h² = 16 倍）。
    for A, B in ((1.0, 0.3), (0.6, 0.6), (0.2, 1.4)):
        W = A * g + B * f
        d2 = np.abs(np.gradient(np.gradient(W, s), s)).max()
        assert d2 < curved * 1e-3, f"A={A} B={B} 残余曲率 {d2:.2e} 不是 f32 噪声量级"


def test_curvature_is_read_in_exactly_one_place():
    """`forage_curvature` 只许被 `_forage_pref_scale` 读到。

    **为什么用这条代替「整个世界逐位相等」**：第一版写的是后者，它在 GPU 上**测不了**——
    每格 scatter-add 是原子的、会重排，200 步后位置就有 1e-6 的漂移
    （`CLAUDE.md`「Determinism is not bit-exact on GPU」，`test_determinism` 用的是
    「结构相同 + 数值容差」）。第一版单独跑侥幸过了、在 `check.py --full` 里红，
    正是那种「时红时绿」的坏测试。

    而「默认关是否 bit-exact」这个断言本来就不需要跑世界：`forage_curvature` 只在一个
    函数里被读，那个函数的逐位等价上面已经钉住了（`test_curvature_default_is_truly_off`）。
    真正剩下的风险只有「将来有人在别处也读它」，这条就是钉那个的。
    """
    import pathlib
    # 只找**读取**（`cfg.forage_curvature`），不是提到名字——第一版按名字找，
    # 于是把 docstring 里解释这个旋钮的那一行也算了进去。
    reads = [(p.name, i, line.strip())
             for p in sorted(pathlib.Path("underworld").glob("*.py"))
             for i, line in enumerate(p.read_text().splitlines(), 1)
             if "cfg.forage_curvature" in line and not line.lstrip().startswith("#")]
    assert {h[0] for h in reads} == {"dynamics.py"}, (
        f"forage_curvature 被 dynamics.py 以外的地方读了: {reads}")
    assert len(reads) == 2, (          # `!= 1.0` 的门控 + 取值那一行
        f"dynamics.py 里读 forage_curvature 的行数变了，重新确认门控仍是编译期的: {reads}")
