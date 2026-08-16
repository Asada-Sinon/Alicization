"""中性零假设的**唯一实现**：漂变 + 复发突变，作用在 `forage_pref` 这一个位点上。

`retained_vs_null.py`（R13，14 箱 clip）与 `retained_stage2.py`（Stage 2，20 箱全量程）
本来各抄了一份，抄的东西一模一样只有 `BINS` 不同。抄第三次之前抽出来。
判据与全部论证见 `docs/multispecies_feasibility.md` §17.9 / §18。

零假设照着内核里真正发生的事写，**没有选择**：

- `state.py:200`  `pref = sigmoid(gene)`，**纯 sigmoid，无 clip**；
- `genome.py:56`  每次出生 `gene += normal(0, forage_pref_mutation_sigma)`，默认 **0.02**；
- `genome.py` 的 `crossover` 是**均匀交叉**且 `forage_pref` **不在豁免名单**里
  （只有 `diet` 和 `size` 豁免）⇒ 单个位点上等价于「从两个亲本里随机取一个」，
  所以「均匀有放回抽亲本」就是对的漂变模型。

**两次被审查拦下的错，都记在这里，别再犯**：

1. **纯二项 Wright-Fisher（`x = binomial(2Ne, x)/2Ne`）漏掉复发突变** ⇒ 0 变成吸收态
   ⇒ 中性衰减太快。`feasibility.md §14` 的解析零假设漏掉同一件事、方向相反
   （中性太宽松）。**两次都错在对当时结论有利的方向**（`MEMORY.md [LEARN:stats]`）。
2. **单倍体的平衡方差是 `V* = N·σ_m²`**，不是教科书那条二倍体加性方差 `2·N_e·V_m`；
   照抄会把中性方差高估 41%，而高估中性方差正好让「实测的簇是中性堆出来的」更容易成立。

**`N` 的定标**：用 §14.3 从**中性臂**（`forage_tradeoff=0`，`dynamics.py` 编译期断开、
基因构造上零效应）的平稳基因方差定出的 `Ne=297–400`（本模块默认扫 105/340/600）。
**不要用 `Var(Δx)` 定标**——它被要检验的信号污染，同一批数据换个稳健口径 `N` 摆动 4 倍。

自检：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260804-readouts/neutral_null.py
"""
import sys

sys.path.insert(0, "explorations/20260804-readouts")

import numpy as np
from split_score import dip_ratio, retained

MUT_SIGMA = 0.02            # config.py:263 forage_pref_mutation_sigma
OCC_THRESH = 0.5            # 「后半程大部分时间保住分裂」的门槛
DEFAULT_NS = (105, 340, 600, 2000)   # **四个都要报**，预注册写死的，不许静默删格子


def hist_of(pref, bins):
    """把 `pref` 分箱。**必须与 `trajectory.py:_hist_fn` 的设备端实现逐位一致**：

        idx = jnp.clip(((p - lo) / w).astype(jnp.int32), 0, nb - 1)

    这是一条**跨实现契约**：零假设与观测必须过同一把尺子，否则分箱本身可能造出的
    假双峰只会出现在一边。`__main__` 的自检会拿真的 JAX 版本比对。
    """
    lo, w, nb = float(bins[0]), float(bins[1] - bins[0]), len(bins) - 1
    idx = np.clip(((np.asarray(pref) - lo) / w).astype(np.int32), 0, nb - 1)
    return np.bincount(idx, minlength=nb).astype(float)


def sampling_edges(bins):
    """取样边界 ≠ 分箱边界，**这是一个会静默改数的坑**。

    clip 分箱下端点箱是**饱和**的：`BINS = linspace(0.15,0.85,15)` 时
    bin0 实际覆盖 (−∞, 0.20)、bin13 覆盖 [0.80, +∞)，而 `pref = sigmoid(...)` 值域是 [0,1]。
    所以逆变换抽样时端点箱要从 **[0, 0.20)** 和 **[0.80, 1.0]** 抽，
    不是从名义边界 [0.15,0.20) 和 [0.80,0.85) 抽。
    全量程分箱（`linspace(0,1,21)`）下两者恰好重合，所以这个函数对它是恒等的。
    """
    e = np.asarray(bins, float).copy()
    e[0] = min(e[0], 0.0)
    e[-1] = max(e[-1], 1.0)
    return e


def genes_from_hist(h, bins, rng, edges=None):
    """从观测直方图逆变换抽初始基因（箱内均匀），再取 logit 回基因空间。

    ⚠️ **只在第 0 帧用。** 那里端点箱的质量可忽略（R13 实测 bin0 占比均值 0.00006、
    最大 0.00036；Stage 2 的 96 个 run 在帧 0 全部不是 `retained`），
    所以「箱内均匀」这个假设无害。**换成后期帧就不成立了**——R13 末帧的 bin0 占比
    最高到 0.517，那时候必须在 logit 空间拟合截尾正态，不能箱内均匀。
    """
    counts = np.asarray(h).astype(int)
    edges = sampling_edges(bins) if edges is None else np.asarray(edges, float)
    out = [rng.uniform(edges[k], edges[k + 1], size=counts[k])
           for k in range(len(counts)) if counts[k] > 0]
    p = np.concatenate(out) if out else np.full(1, 0.5)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def sim_genes(genes0, n_gen, N, rng):
    """跑 `n_gen` 代中性漂变 + 突变，只返回末代基因。`sim_run` 与自检共用这一份演化规则。"""
    g = np.asarray(genes0, float)
    if len(g) > N:
        g = g[rng.integers(0, len(g), size=N)]
    for _ in range(max(int(n_gen), 1)):
        g = g[rng.integers(0, len(g), size=N)] + rng.normal(0.0, MUT_SIGMA, size=N)
    return g


def sim_run(genes0, n_gen, n_frames, N, bins, rng):
    """跑 `n_gen` 代中性漂变 + 突变，返回 `n_frames` 个直方图（等代距取样）。"""
    g = np.asarray(genes0, float)
    if len(g) > N:
        g = g[rng.integers(0, len(g), size=N)]
    n_gen = max(int(n_gen), 1)
    take = np.linspace(0, n_gen, n_frames).astype(int)
    out, j = [], 0
    for t in range(n_gen + 1):
        while j < n_frames and take[j] == t:
            out.append(hist_of(1.0 / (1.0 + np.exp(-g)), bins))
            j += 1
        g = g[rng.integers(0, len(g), size=N)] + rng.normal(0.0, MUT_SIGMA, size=N)
    while len(out) < n_frames:
        out.append(out[-1])
    return out


def _pava(y):
    """保序回归（pool-adjacent-violators）：`generation` 对帧号的最小二乘单调拟合。

    `generation` 是**存活个体的平均世代数**，不是单调的——一批新生儿到来会把它拉低
    （R13 实测逐帧 Δ 有 36.6% 为负，范围 −270…+280）。
    直接 `clip(gradient, 0, None)` 的毛病**不是「丢帧」**（保序化并不总能减少零权重帧：
    Stage 2 从 12.2% 降到 0.26%，R13 反而 39.9%→40.8%，因为 PAVA 把违序块合并成平台、
    平台的梯度还是 0）。真正的毛病是**权重总量对不上**：

        数据      口径     Σ权重    真实跨代    比
        R13       clip    1096.3    455.0    **2.24×**
        R13       iso      465.7    455.0      1.021
        Stage 2   clip     175.4    139.3      1.155
        Stage 2   iso      142.8    139.3      1.028

    **`clip` 在 R13 上凭空多造了 141% 的权重，而且全堆在 boom 帧上**——每一次
    「新生儿潮把平均世代数拉低、随后回升」都被记成一次净增长，同一段时间数了两遍。
    保序拟合的权重**加起来等于真实经历的世代数**（误差 2–3%），这才是它该被选中的理由。
    """
    lvl = []
    for v in np.asarray(y, float):
        lvl.append([v, 1.0])
        while len(lvl) > 1 and lvl[-2][0] > lvl[-1][0]:
            v2, w2 = lvl.pop()
            v1, w1 = lvl.pop()
            lvl.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2])
    out = []
    for v, c in lvl:
        out.extend([v] * int(c))
    return np.array(out)


def gen_weights(gen, mode="iso"):
    """逐帧权重。`iso` = 保序拟合后的世代增量（默认）；`clip` = 旧口径；`unif` = 等权。

    三种口径的敏感性实测在 `feasibility.md` §18.4c：**Stage 2 的四臂计数三种口径完全相同**，
    R13 是 7/8/9（clip/iso/unif）。**报结论时三种都要给，不许挑一个用。**
    """
    g = np.asarray(gen, float)
    # **未知模式必须报错，不许静默兜底成 iso。** 原来 `return` 兜住了一切拼写错误：
    # 我用 `flat`/`late` 做「三种口径都给」的敏感性检查，两个名字都不存在，
    # 于是三条曲线全是 iso，分歧「0.0000」——一个看起来最漂亮的假阴性。
    if mode not in ("iso", "clip", "unif"):
        raise ValueError(f"未知的世代权重口径 {mode!r}；只有 'iso' / 'clip' / 'unif'")
    if mode == "unif":
        return np.ones(len(g))
    if mode == "clip":
        return np.clip(np.gradient(g), 0.0, None)
    return np.clip(np.gradient(_pava(g)), 0.0, None)


def weighted_second_half(values, gen, mode="iso"):
    """后半程的加权均值。**所有「后半程」读数都走这一个函数**，否则会出现
    `retained_occ` 用 `iso`、`low_mass_second` 用内联 `clip` 这种**两把尺子**的情形
    ——而它们恰恰是被要求「必须一起看」的三个读数（§18.4）。"""
    v = np.asarray(values, float)
    k = len(v) // 2
    w = gen_weights(gen, mode)[k:]
    return float((w * v[k:]).sum() / max(w.sum(), 1e-12))


def occupancy(hists, gen, ctr, second_half=True, mass_thresh=0.03, dip_thresh=0.5,
              mode="iso"):
    """`retained()` 的**世代加权**占空比。

    加权用世代不用帧，因为关掉捕食者会让种群翻倍、世代钟慢 2–3 倍
    （`feasibility.md §16.3`），按帧加权等于给慢世界更多权重。
    """
    ret = np.array([retained(h, ctr, mass_thresh, dip_thresh) for h in hists], float)
    w = gen_weights(gen, mode)
    k = len(ret) // 2 if second_half else 0
    end = len(ret) if second_half else len(ret) // 2
    ww = w[k:end]
    return float((ww * ret[k:end]).sum() / max(ww.sum(), 1e-12))


def null_counts(runs, bins, ctr, N, reps, rng, thresh=OCC_THRESH, mode="iso"):
    """零分布：每次重复把**全部** run 各模拟一遍，数有多少个占空比 > `thresh`。

    单位是「跨 run 的计数」，**不是 per-run 的 p 值**——判据问的是
    「这么多个 run 同时保住分裂，中性出得来吗」。

    `runs` 是 `[{"hist": [n_frames, n_bins], "gen": [n_frames]}, ...]`。
    """
    out = []
    for _ in range(reps):
        c = 0
        for r in runs:
            g0 = genes_from_hist(r["hist"][0], bins, rng)
            span = int(max(r["gen"][-1] - r["gen"][0], 5))
            hs = sim_run(g0, span, len(r["hist"]), N, bins, rng)
            if occupancy(hs, r["gen"], ctr, mode=mode) > thresh:
                c += 1
        out.append(c)
    return np.array(out)


def report(runs, bins, ctr, label, reps=200, seed=20260804, ns=DEFAULT_NS, mode="iso"):
    """打印一组 run 的实测占空比与各 `N` 下的零分布分位。返回逐 `N` 的 (均值, p)。

    **观测与零假设必须用同一个 `mode`**，否则加权口径的差异会被记成效应。
    """
    obs = np.array([occupancy(r["hist"], r["gen"], ctr, mode=mode) for r in runs])
    T = int((obs > OCC_THRESH).sum())
    print(f'  {label}: 后半程 `retained` 占空比 > {OCC_THRESH} 的 run = **{T}/{len(runs)}**'
          f'   （占空比均值 {obs.mean():.4f}，加权口径 {mode}）')
    rng = np.random.default_rng(seed)
    res = {}
    floor = 1.0 / (reps + 1)
    for N in ns:
        cnt = null_counts(runs, bins, ctr, N, reps, rng, mode=mode)
        # **`(cnt >= T).mean()` 会取到恰好 0，那不是有效的蒙特卡洛 p 值。**
        # 正确估计量是 `(1 + k) / (reps + 1)`，它的可达下限就是 `1/(reps+1)`。
        # 本项目已经在另一个壳里栽过同一个错：n=6 的配对符号秩 p 地板永远是 0.031
        # （`run_to_run_variance.md §7.0`）。**顶在地板上的 p 不是强证据，是分辨率耗尽。**
        # 所以同时报零分布的**最大值**——「200 次重复里最多只到 3，而实测 24」
        # 比一个假的 0.0000 信息量大得多。
        k = int((cnt >= T).sum())
        p = (1.0 + k) / (reps + 1.0)
        res[N] = (float(cnt.mean()), p)
        print(f'    N={N:<5} 零均值 {cnt.mean():>5.2f}   5–95% [{np.percentile(cnt, 5):.0f},'
              f' {np.percentile(cnt, 95):.0f}]   最大 {cnt.max():>3.0f}   '
              f'p={"<" if k == 0 else "="}{p:.4f}'
              f'{"（= 分辨率下限）" if k == 0 else ""}'
              f'{"   ** 不拒绝 **" if p >= 0.05 else ""}')
    print(f'    （reps={reps} ⇒ p 的分辨率下限 {floor:.4f}；要主张更小必须加 reps）')
    return res


def _self_check():
    """**跨实现契约**：numpy 的 `hist_of` 必须与设备端 JAX 的分箱逐位一致。

    这一条如果破了，零假设与观测就过了两把不同的尺子，而**分箱本身可能造的假双峰
    只会出现在一边**——那正是整个判据赖以成立的抵消机制。
    """
    import dataclasses

    import jax
    import jax.numpy as jnp

    from trajectory import BINS, _hist_fn
    from underworld import Config

    cfg = dataclasses.replace(Config(), n_max=4096)
    fn = _hist_fn(cfg)
    rng = np.random.default_rng(0)
    bad = 0
    for trial in range(5):
        # 覆盖端点：故意让一部分 pref 落在 0 和 1 附近
        gene = rng.normal(0.0, 3.0, size=cfg.n_max).astype(np.float32)
        genome = np.zeros((cfg.n_max, cfg.genome_size), np.float32)
        genome[:, cfg.forage_pref_index] = gene
        alive = rng.random(cfg.n_max) < 0.7
        diet = rng.random(cfg.n_max).astype(np.float32) * 0.3   # 全是食草者
        h_jax, nh, _g = fn(jnp.asarray(genome), jnp.asarray(alive),
                           jnp.asarray(diet), jnp.zeros(cfg.n_max, np.float32))
        pref = 1.0 / (1.0 + np.exp(-gene[alive].astype(np.float64)))
        h_np = hist_of(pref, BINS)
        d = int(np.abs(np.asarray(h_jax) - h_np).sum())
        bad += d
        print(f"    trial {trial}: n_herb={int(nh)}  两个实现的逐箱总差异 = {d}"
              f"{'  ← 不一致！' if d else ''}")
    assert bad == 0, f"分箱实现不一致，总差异 {bad}"
    print("  ✓ numpy 与 JAX 的分箱逐位一致（含端点饱和箱）")

    # 第二条：**群体平均基因是鞅**——这正是「无选择」的可操作定义。
    # 注意不能检验「单次实现的分布不变」：N=4000、50 代的纯 WF 重抽样本身就有漂变，
    # 20 个箱各自 SD ≈ 0.112·√(p(1−p))，总变差期望就有 ~0.39。**要检验的是期望，
    # 不是实现。** 阈值也不能拍脑袋定，用跨重复的自估 SE 定（`CLAUDE.md` 的护栏规矩）。
    rng2 = np.random.default_rng(1)
    g0 = rng2.normal(0.0, 1.0, size=800)
    REP = 300
    ends = np.array([sim_genes(g0, 60, 800, rng2).mean() for _ in range(REP)])
    se = float(ends.std(ddof=1) / np.sqrt(REP))
    bias = float(ends.mean() - g0.mean())
    print(f"  平均基因：起点 {g0.mean():+.5f} → {REP} 次重复的终点均值 {ends.mean():+.5f}")
    print(f"  ✓ 偏移 {bias:+.5f}，自估 SE {se:.5f} ⇒ {abs(bias) / se:.2f} 个 SE（鞅要求 <3）")
    assert abs(bias) < 3.0 * se, (bias, se)


if __name__ == "__main__":
    print("=" * 88)
    print("neutral_null.py 自检")
    print("=" * 88)
    _self_check()
