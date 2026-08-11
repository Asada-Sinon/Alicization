#!/usr/bin/env python
"""按 §22.4b 的同一套规则重算中性零分布,但用**实测代数**。

§22.4b 的 MDE=0.176 取自 (Ne=105, 669 代) 格。R19 实测只有 ~153 代
(gen_total, outputs/20260807-r19/*: 均值 153),40 代处探针曾读 C=-0.007。
规则(§22.4b 原文):单倍体 WF 均匀重抽 + N(0,0.02) 复发突变,初始 N(0,0.4),
读数 = 「24 个 run 的 mean gene 之均值」的零分布分位。原口径 60 次重抽×24 run;
这里 2000 次×24 run,分位更稳。同时复算 (105,669) 一格验证与 0.176 一致。

输出:每 (Ne, G) 的 95 分位(单侧)与 97.5 分位,及 C 臂实测 +0.1772 的零假设 P。
"""
import numpy as np

rng = np.random.default_rng(20260811)
REPS, RUNS = 2000, 24
MUT, INIT = 0.02, 0.4

def zero_dist(N, G, reps=REPS, runs=RUNS, chunk=2000):
    """返回 reps 个「runs 个独立 WF run 的 mean gene 均值」样本。"""
    M = reps * runs
    means = np.empty(M)
    for lo in range(0, M, chunk):
        m = min(chunk, M - lo)
        g = rng.normal(0.0, INIT, size=(m, N))
        for _ in range(G):
            idx = rng.integers(0, N, size=(m, N))
            g = np.take_along_axis(g, idx, axis=1) + rng.normal(0.0, MUT, size=(m, N))
        means[lo:lo + m] = g.mean(axis=1)
    return means.reshape(reps, runs).mean(axis=1)

for N, G in [(105, 40), (105, 153), (340, 40), (340, 153), (105, 669)]:
    z = zero_dist(N, G)
    q95, q975 = np.percentile(z, [95, 97.5])
    for obs, name in [(0.1772, 'C'), (0.4082, 'A')]:
        pass
    pC = float((np.abs(z) >= 0.1772).mean())   # 双侧
    pA = float((np.abs(z) >= 0.4082).mean())
    print(f'Ne={N:4d} G={G:3d}: 95%={q95:.4f}  97.5%={q975:.4f}  SD={z.std():.4f}  '
          f'P(|zero|>=0.1772)={pC:.4f}  P(|zero|>=0.4082)={pA:.4f}')
