"""`explorations/20260804-readouts/neutral_null.py` 的快检。

这个模块是 `feasibility.md` §17.9/§18 那条判决的**唯一零假设实现**，而它踩过的坑
（漏掉复发突变、抄错二倍体方差公式、取样边界≠分箱边界、`p` 估计量能取到 0）
每一条都会让结论朝对自己有利的方向偏。所以它的不变量值得进 `pytest`。

**只收纯 numpy 的那半。** 「numpy 分箱与设备端 JAX 分箱逐位一致」那条留在模块自己的
`_self_check()` 里——它要建一个 4096 agent 的世界，不该进单元测试。
"""
import sys

import numpy as np
import pytest

sys.path.insert(0, "explorations/20260804-readouts")

nn = pytest.importorskip("neutral_null")


def test_sampling_edges_identity_on_full_range():
    """全量程分箱下取样边界 == 分箱边界；只有饱和的窄分箱才该被加宽。"""
    for bins in (np.linspace(0.0, 1.0, 21), np.linspace(0.0, 1.0, 15)):
        assert np.array_equal(nn.sampling_edges(bins), bins)


def test_sampling_edges_widens_saturated_ends():
    """`linspace(0.15,0.85,15)` 的 bin0 实际覆盖 (−∞,0.20)、bin13 覆盖 [0.80,+∞)，
    所以取样必须从 [0,0.20) 与 [0.80,1.0] 抽，而中间的边界一个都不许动。"""
    bins = np.linspace(0.15, 0.85, 15)
    e = nn.sampling_edges(bins)
    assert e[0] == 0.0 and e[-1] == 1.0
    assert np.array_equal(e[1:-1], bins[1:-1])


def test_pava_is_monotone_and_mean_preserving():
    """保序回归：输出单调不减，且总和守恒（块内取均值）。"""
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(1.0, 5.0, size=200))     # 有大量违序
    z = nn._pava(y)
    assert np.all(np.diff(z) >= -1e-9), "保序回归的输出必须单调不减"
    assert z.sum() == pytest.approx(y.sum(), rel=1e-9)


def test_gen_weights_modes_are_nonnegative_and_nonzero():
    rng = np.random.default_rng(1)
    g = np.cumsum(np.abs(rng.normal(1.0, 0.3, size=100)))
    g[40:50] -= 20.0                                   # 制造一段违序
    for mode in ("clip", "iso", "unif"):
        w = nn.gen_weights(g, mode)
        assert len(w) == len(g)
        assert np.all(w >= 0.0)
        assert w.sum() > 0.0


def test_iso_weights_sum_to_the_true_generation_span():
    """保序化的**真正理由**：权重总量必须等于真实经历的世代数。

    `clip(gradient, 0, None)` 把每一次「向下抖动 + 回升」都记成一次净增长，
    同一段时间数了两遍——R13 实测 Σw/跨代 = **2.24×**，多造了 141% 的权重，
    而且全堆在 boom 帧上。保序拟合的比值是 1.02。
    （注意**不能**改测「零权重帧更少」：PAVA 把违序块合并成平台，平台梯度仍是 0，
    R13 上零权重帧反而从 39.9% 升到 40.8%。第一版测试就是这么写错的。）
    """
    rng = np.random.default_rng(2)
    g = np.cumsum(np.abs(rng.normal(1.0, 0.2, size=300)))
    g[::3] -= 6.0                                      # 每三帧一次向下的抖动
    span = g[-1] - g[0]
    assert nn.gen_weights(g, "clip").sum() > 1.5 * span, "构造的序列本该让 clip 严重超量"
    assert nn.gen_weights(g, "iso").sum() == pytest.approx(span, rel=0.05)


def test_sim_run_returns_exactly_n_frames():
    """采样点数必须恒等于请求的帧数，且末帧就是末代——尾部补齐不该被触发。"""
    rng = np.random.default_rng(3)
    bins = np.linspace(0.0, 1.0, 21)
    for n_gen, n_frames in ((5, 50), (50, 5), (162, 162), (1, 10)):
        out = nn.sim_run(rng.normal(0, 1, 200), n_gen, n_frames, 200, bins, rng)
        assert len(out) == n_frames
        assert all(len(h) == len(bins) - 1 for h in out)
        take = np.linspace(0, max(n_gen, 1), n_frames).astype(int)
        assert take[-1] == max(n_gen, 1), "末帧必须落在末代上，否则尾部会被复制"


def test_mean_gene_is_a_martingale():
    """**无选择的可操作定义**：群体平均基因是鞅。

    阈值用跨重复自估的 SE 定，不拍脑袋（`CLAUDE.md` 的护栏规矩）。
    """
    rng = np.random.default_rng(4)
    g0 = rng.normal(0.0, 1.0, size=400)
    ends = np.array([nn.sim_genes(g0, 30, 400, rng).mean() for _ in range(120)])
    se = ends.std(ddof=1) / np.sqrt(len(ends))
    assert abs(ends.mean() - g0.mean()) < 4.0 * se


def test_hist_of_counts_every_individual():
    """clip 分箱**一个个体都不丢**——这正是 R13 轨迹能用而检查点不能用的原因。"""
    rng = np.random.default_rng(5)
    bins = np.linspace(0.15, 0.85, 15)
    pref = rng.random(5000)                            # 大量落在 [0,0.15) 与 (0.85,1]
    h = nn.hist_of(pref, bins)
    assert h.sum() == len(pref)
    assert h[0] > 0 and h[-1] > 0, "端点箱必须吸收窗外个体"
