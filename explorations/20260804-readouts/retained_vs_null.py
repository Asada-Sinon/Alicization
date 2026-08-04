"""R15-A 的推断判据：**「中间被掏空的双峰」在中性零假设下出得来吗。**

为什么是这个判据而不是「x 降不降」
----------------------------------
两次 `plan-critic` 审查把前面的候选逐个否掉，剩下的路只有这一条：

1. **「对 x 回归 Δx 估衰减率」** —— 两趟制算出预测效应 ÷ MDE ≤ 0.21（欠功效约 5 倍），
   且审查证明它的 β=0 零假设**在估计量层面**就是坏的（含截距 OLS + 有界 + 0 处吸收
   ⇒ 零分布均值 −0.085，实测 −0.028 高于整个零分布，**假阳率 100% 且符号反过来**）。
2. **「聚合的 x 降不降」** —— traj 第 0 帧是 **t=250、第 1.33 代**，那是创始者抽样本身
   （实测 0.0464 ≈ Φ(logit(0.35)/`genome_init_scale`) = 0.061）。跨 run 逐帧均值是
   **先升后降**：0.0464 → 峰 0.1691 @ 37 代 → 末帧 0.0941。
   **末帧相对创始者 +0.048，而带复发突变的中性零假设预测 +0.04 ⇒ 聚合上看不出选择。**
3. **「群体平均基因」** —— 在本零假设下确是严格鞅，但**从分箱直方图算不出来**：
   clip 口径的 bin0=(−∞,0.20)、bin13=[0.80,+∞) 会饱和，末帧 bin0 占比最高到 0.517，
   取箱中点的 logit 会把真实均值系统性拉向中间。**同一类截断失误的第四次，不踩。**

**剩下的是 run 之间的双峰结构**：少数 run 停在 x≈0.5，多数停在 0。均值检验看不见它。
而 `retained()` 直接问那个生物学问题：

    retained = (低端有质量 `low_mass > 0.03`) AND (中间被掏空 `dip_ratio < 0.5`)

**关键的不对称**：纯漂变 + 复发突变会**移动**整个分布，但**不会把中间掏空**——
它没有任何机制去劈开一个单峰。所以「中间被掏空」是中性零假设**结构上难以伪造**的特征。

`retained()` 是 R13 就定义、验证并提交的（`803dc60`，在 120 个检查点上 0 假阳 0 假阴），
**不是本轮事后挑的统计量**。

判据（跑之前写死在这个 docstring 里）
------------------------------------
**统计量 T** = 后半程（世代加权）`retained()` 占空比 > 0.5 的 run 数，分母 24。
**零假设**：每个 run 从**它自己第 0 帧的实测直方图**起跑中性模拟
（帧 0 的 bin0 占比均值 0.00006、最大 0.00036 ⇒ 逆变换抽样的边界偏倚在此处可忽略，
这是必须用帧 0 而不是后期帧当起点的理由），跑该 run 实测的代数，
在同样的帧位上取样，套同一个 `retained()`。
**N 的敏感性**：105 / 340 / 600 / 2000。340 来自 §14.3 用**中性臂**
（`forage_tradeoff=0`，基因构造上零效应）的平稳基因方差定出的 Ne=297–400，
**那是构造上无选择的定标物**；不用 `Var(Δx)` 定标，因为它被要检验的信号污染
（同一批数据均值口径给 N≈150、中位口径给 N≈600，摆动 4 倍）。
**判定**：实测 T 在零分布中的分位 < 0.05（单侧）才算拒绝，且**四个 N 都要报**。
若最保守的 N 下不过线，一律判 undecidable。

跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
      explorations/20260804-readouts/retained_vs_null.py
"""
import glob
import json
import sys

sys.path.insert(0, "explorations/20260804-readouts")

import numpy as np
from split_score import dip_ratio, retained

BINS = np.linspace(0.15, 0.85, 15)
CTR = 0.5 * (BINS[:-1] + BINS[1:])
LOW = CTR < 0.35
MUT_SIGMA = 0.02
OCC_THRESH = 0.5


def clip_hist(pref):
    """复刻 R13 版 `trajectory.py:79` 的 clip 分箱——**必须逐位一致，否则零假设不可比**。"""
    lo, w, nb = float(BINS[0]), float(BINS[1] - BINS[0]), len(CTR)
    idx = np.clip(((pref - lo) / w).astype(np.int32), 0, nb - 1)
    return np.bincount(idx, minlength=nb).astype(float)


def genes_from_hist(h, rng):
    """从观测直方图逆变换抽初始基因。箱内均匀，端点箱按其名义宽度抽。

    **只在帧 0 用**：那里 bin0/bin13 的质量可忽略（实测 bin0 均值 0.00006），
    所以「箱内均匀」这个假设在此处无害。换到后期帧就不成立了。
    """
    n = int(h.sum())
    edges = np.concatenate([[0.0], BINS[1:-1], [1.0]])   # bin0=[0,0.20)、bin13=[0.80,1]
    counts = h.astype(int)
    out = []
    for k in range(len(CTR)):
        if counts[k] <= 0:
            continue
        out.append(rng.uniform(edges[k], edges[k + 1], size=counts[k]))
    p = np.concatenate(out) if out else np.full(max(n, 1), 0.5)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def occupancy_second_half(hists, gen):
    """后半程（世代加权）`retained()` 的占空比。"""
    ret = np.array([retained(h, CTR) for h in hists], float)
    w = np.clip(np.gradient(gen), 0.0, None)
    half = len(ret) // 2
    wb = w[half:]
    return float((wb * ret[half:]).sum() / max(wb.sum(), 1e-12))


def sim_run(genes0, n_gen, n_frames, N, rng):
    """中性漂变 + 复发突变，返回 `n_frames` 个 clip 直方图（等代距取样）。"""
    g = np.asarray(genes0, float)
    if len(g) > N:                    # 起始群体大于模拟群体时先降采样
        g = g[rng.integers(0, len(g), size=N)]
    take = np.linspace(0, n_gen, n_frames).astype(int)
    out, j = [], 0
    for t in range(n_gen + 1):
        while j < n_frames and take[j] == t:
            out.append(clip_hist(1.0 / (1.0 + np.exp(-g))))
            j += 1
        g = g[rng.integers(0, len(g), size=N)] + rng.normal(0.0, MUT_SIGMA, size=N)
    while len(out) < n_frames:
        out.append(clip_hist(1.0 / (1.0 + np.exp(-g))))
    return out


def main():
    runs = []
    for f in sorted(glob.glob("outputs/20260804-traj/*.log")):
        txt = open(f).read()
        if "JSON " not in txt:
            continue
        d = json.loads(txt.split("JSON ")[1].split("\n")[0])
        tr = d["traj"]
        h = np.array([q["hist"] for q in tr], float)
        assert np.allclose(h.sum(1), [q["n_herb"] for q in tr], rtol=1e-6, atol=1e-3)
        runs.append({"name": f.split("/")[-1][:-4], "hist": h,
                     "gen": np.array([q["generation"] for q in tr], float)})

    print("=" * 96)
    print("R15-A 推断判据：**中间被掏空的双峰**在中性零假设下出得来吗")
    print("=" * 96)
    obs = np.array([occupancy_second_half(r["hist"], r["gen"]) for r in runs])
    T_obs = int((obs > OCC_THRESH).sum())
    print(f'  实测：后半程 `retained` 占空比 > {OCC_THRESH} 的 run = **{T_obs}/24**')
    hits = [(r["name"][:-3], o) for r, o in zip(runs, obs) if o > OCC_THRESH]
    print("    逐 run: " + "  ".join(f"{n}={o:.2f}" for n, o in hits))

    REP = 200
    rng = np.random.default_rng(20260804)
    print()
    print(f'  {"N":>7}{"零分布 T 均值":>14}{"5–95%":>16}{"P(null>=obs)":>15}{"判定":>12}')
    for N in (105, 340, 600, 2000):
        ts = []
        for _ in range(REP):
            c = 0
            for r in runs:
                g0 = genes_from_hist(r["hist"][0], rng)
                span = int(max(r["gen"][-1] - r["gen"][0], 10))
                hs = sim_run(g0, span, len(r["hist"]), N, rng)
                if occupancy_second_half(hs, r["gen"]) > OCC_THRESH:
                    c += 1
            ts.append(c)
        ts = np.array(ts)
        p = float((ts >= T_obs).mean())
        print(f'  {N:>7}{ts.mean():>14.2f}   [{np.percentile(ts, 5):.0f}, {np.percentile(ts, 95):.0f}]'
              f'{p:>15.4f}{"拒绝" if p < 0.05 else "不拒绝":>12}')
    print()
    print(f"  （每个 N 跑 {REP} 次完整的 24-run 重复；判定按 §19 预注册：四个 N 都要报，")
    print("    最保守的那个不过线就一律判 undecidable）")


if __name__ == "__main__":
    main()
