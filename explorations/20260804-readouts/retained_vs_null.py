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
from neutral_null import OCC_THRESH, occupancy, report

BINS = np.linspace(0.15, 0.85, 15)      # R13 版的 clip 分箱，**不是现版的全量程分箱**
CTR = 0.5 * (BINS[:-1] + BINS[1:])


def load():
    runs = []
    for f in sorted(glob.glob("outputs/20260804-traj/*.log")):
        txt = open(f).read()
        if "JSON " not in txt:
            continue
        d = json.loads(txt.split("JSON ")[1].split("\n")[0])
        tr = d["traj"]
        h = np.array([q["hist"] for q in tr], float)
        assert np.allclose(h.sum(1), [q["n_herb"] for q in tr], rtol=1e-6, atol=1e-3), f
        runs.append({"name": f.split("/")[-1][:-4], "seed": d["seed"], "hist": h,
                     "gen": np.array([q["generation"] for q in tr], float),
                     "carn": float(np.nanmean([q["carnivore_frac"] for q in tr]))})
    return runs


def main():
    runs = load()
    print("=" * 96)
    print("R15-A 推断判据：**中间被掏空的双峰**在中性零假设下出得来吗（R13 的 24 run）")
    print("=" * 96)
    obs = np.array([occupancy(r["hist"], r["gen"], CTR) for r in runs])
    hits = [(r["name"][:-3], o) for r, o in zip(runs, obs) if o > OCC_THRESH]
    print("  逐 run 命中: " + "  ".join(f"{n}={o:.2f}" for n, o in hits))
    print()
    report(runs, BINS, CTR, "全 24 run（主判据）")

    # 敏感性一：种子级（`wn1_sN` 与 `wn2_sN` 是同一次 founder 抽样，24 个不是 24 个独立单位）
    # 敏感性二：去掉 3 个自发丢失捕食者的 run
    # 两者都在 §17.9 里报了；这里只重跑主判据，分层版见 docs 的表。
    print()
    print("  （种子级与「去掉丢失捕食者」的敏感性见 `feasibility.md` §17.9 的表；")
    print("    去掉那 3 个 run 后在最宽松的 N 上 p=0.060 不过线 ⇒ 那一条判 undecidable）")


if __name__ == "__main__":
    main()
