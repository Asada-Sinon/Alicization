"""中性零假设模拟器：性状层面、**带复发突变**。R15-A 用。

为什么必须自己写一个
--------------------
`feasibility.md §14` 的解析零假设漏掉复发突变 ⇒ 中性**太宽松**；
而 R15-A 第一版审查里用的纯二项 Wright-Fisher（`x = binomial(2Ne, x)/2Ne`）
**同样漏掉复发突变** ⇒ 0 变成吸收态 ⇒ 中性**衰减太快**。
两次的错都指向当时那个结论有利的方向，这正是 `MEMORY.md [LEARN:stats]` 那一条。

本模拟器忠实复刻内核里真正发生的事：

- `state.py:200`  `pref = sigmoid(gene)`，**纯 sigmoid，无 clip**；
- `genome.py:56`  每次出生 `gene += normal(0, forage_pref_mutation_sigma)`，默认 **0.02**；
- `genome.py` 的 `crossover` 是**均匀交叉**且 `forage_pref` **不在豁免名单**里
  （只有 diet 和 size 豁免）⇒ 单个位点上等价于「从两个亲本里随机取一个」，
  所以「均匀有放回抽亲本」就是对的漂变模型。

**没有选择。** 这就是零假设。

跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
      explorations/20260804-readouts/null_sim.py
"""
import numpy as np

MUT_SIGMA = 0.02          # config.py:263 forage_pref_mutation_sigma
THRESH = 0.35             # low_mass 的阈值（pref 空间）
GENE_THRESH = float(np.log(THRESH / (1.0 - THRESH)))   # logit(0.35) = -0.6190


def simulate(genes0, n_gen, N, rng, record_every=1):
    """跑 `n_gen` 代中性漂变 + 突变，返回逐代的 `x = mean(sigmoid(gene) < 0.35)`。

    `genes0` 是初始基因数组；每代把种群重抽到 `N` 个。
    """
    g = np.asarray(genes0, float)
    out = []
    for t in range(n_gen + 1):
        if t % record_every == 0:
            out.append(float((g < GENE_THRESH).mean()))
        idx = rng.integers(0, len(g), size=N)
        g = g[idx] + rng.normal(0.0, MUT_SIGMA, size=N)
    return np.array(out)


def equilibrium_probe():
    """**回答一个决定整件事怎么解读的问题**：中性突变—漂变平衡下 `x` 停在哪？

    如果那个平衡水平本身就落在实测的 0.15–0.46 里，那么「少数簇在衰减」这个提法
    就问错了——该问的是「实测水平比中性平衡高还是低」。

    解析预期：这个模型是**单倍体**（每个位点从一个亲本取），方差递推是
    `V_{t+1} = V_t·(1 − 1/N) + σ_m²` ⇒ 平衡 **`V* = N·σ_m²`**。
    **不是教科书上的 `2·N_e·V_m`** —— 那条是二倍体加性方差的结果，
    照抄过来会把中性方差高估一倍（对应 √V* 高估 41%），
    而高估中性方差正好让「实测的簇是中性堆出来的」更容易成立。
    基因分布 ≈ `N(μ, V*)` 时 `x = Φ((logit(0.35) − μ)/√V*)`。
    **解析式只用来定标模拟，不当零假设**（`MEMORY.md [LEARN:stats]`）。

    ⚠️ 单次实现的离散极大：`μ` 本身在做无回复力的随机游走，所以单个 run 的 `x`
    主要由「均值漂到哪」决定，而不是由平衡方差决定。下表逐行只是**一次**实现，
    正式判决必须用多次重复的分位数。
    """
    rng = np.random.default_rng(0)
    print("=" * 92)
    print("中性突变—漂变平衡下 `low_mass` 停在哪（σ_m=0.02，起点全 0，即 pref 全 0.5）")
    print("=" * 92)
    print(f"  基因阈值 logit(0.35) = {GENE_THRESH:.4f}")
    print()
    print(f'  {"N":>6}{"弛豫~N 代":>11}{"解析 √V*":>11}{"解析 x*":>10}'
          f'{"x@400 中位":>13}{"x@400 5–95%":>20}{"SD(基因)@400 中位":>19}')
    from scipy.stats import norm
    REP = 200
    for N in (105, 149, 297, 400, 600, 1000):
        v = N * MUT_SIGMA ** 2                     # 单倍体：V* = N·σ_m²
        xa = float(norm.cdf(GENE_THRESH / np.sqrt(v)))
        xs, sds = [], []
        for _ in range(REP):
            g = np.zeros(N)
            for _t in range(400):
                g = g[rng.integers(0, N, size=N)] + rng.normal(0.0, MUT_SIGMA, size=N)
            xs.append(float((g < GENE_THRESH).mean()))
            sds.append(float(g.std()))
        xs, sds = np.array(xs), np.array(sds)
        print(f'  {N:>6}{N:>11}{np.sqrt(v):>11.4f}{xa:>10.4f}'
              f'{np.median(xs):>13.4f}   [{np.percentile(xs, 5):.4f}, {np.percentile(xs, 95):.4f}]'
              f'{np.median(sds):>19.4f}')
    print()
    print(f"  （每格 {REP} 次重复，起点 pref=0.5 全同）")
    print("  读法：`解析 x*` 是**平衡**水平；`x@400` 是从 pref=0.5 起跑 400 代后的实现分布。")
    print("  R13 的 run 只跨 118–613 代，且起点不是 pref=0.5（已跑了 22000 步）")
    print("  ⇒ 正式判决的零假设必须从**该 run 第 0 帧的实测分布**起跑，本表只定量级。")


if __name__ == "__main__":
    equilibrium_probe()
