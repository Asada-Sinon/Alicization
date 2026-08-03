"""**直接实测适应度曲面** W(forage_pref)——`§9.9` 与 `§10.5` 都点名说「从未测过」的那件事。

为什么现在必须先测它
--------------------
`multispecies_feasibility.md §11` 的立论是纯代数的：直线权衡前沿 ⇒ `W(s)` 对 `s` 线性 ⇒
`d²W/ds² ≡ 0` ⇒ 中间型永远不可能是适应度极小值。据此装了 `forage_curvature`，
`k<1` 时前沿朝原点内凹，中间型**应当**变成极小值。

但单种子操控检查（`outputs/20260803-curvature/k*_s0.log`）里，四个剂量都**没有**出现双峰，
`sd` 也不单调。在盲跑 120 个 run 之前，要先分清两种截然不同的失败：

- **(A) 曲率没到达适应度曲面** —— 实测 `W(s)` 仍然是凹的或平的。
  那么 §11.4 的手术无效，问题在别处（例如摄入被别的项主导）。
- **(B) 曲率到达了，但基因流把两个模态杂交回中间** —— 实测 `W(s)` 是 U 形（中间是极小），
  分布却仍单峰。那么手术有效，缺的是**降低基因流**（局部交配），
  两者**互补**而非二选一。

**这两种情形要的下一步完全相反**，所以这个读数值得先花 10 分钟。

怎么测
------
`state.last_food = food_gain + fruit_gain` 是**逐个体、上一步的实际摄入**
（`step.py`），零内核改动。烧完机之后每隔几步抓一次快照，把 (forage_pref, last_food)
汇总，按基因分箱求均值——那就是 `W(s)` 本身，相差一个常数。

两条独立估计，互为交叉核对：

1. **摄入曲线**：每箱的平均 `last_food`。直接、但只统计**活着的**个体（存活偏差）。
2. **人口学曲线**：同一批箱在 Δt 步内的密度**对数变化率**。这是选择的定义式
   （`d log n_i / dt` 就是相对适应度），自动把存活与繁殖都算进去，代价是更吵。

判据：对 `k` 拟合二次项，看**二阶系数的符号**。§11.1 预测 k=1 时 ≈0、k<1 时 >0。

    XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260803-curvature/fitness_surface.py 20000 --seed 0 \
        --set forage_curvature=0.5 ...

**单个 run 不判决**。这是诊断读数，用来在两条互斥的下一步之间做选择。
"""
import argparse
import dataclasses
import json
import sys

sys.path.insert(0, ".")

import jax
import numpy as np

from underworld import Config, new_world
from underworld import state as state_mod

SAMPLE_EVERY = 10          # 抓快照的间隔
SAMPLE_STEPS = 2000        # 烧机后再跑这么多步做统计
BINS = np.linspace(0.15, 0.85, 15)


def main(steps: int, seed: int, overrides: dict, as_json: bool) -> None:
    cfg = dataclasses.replace(Config(), seed=seed, **(overrides or {}))
    state, key, _step, scan_fn, _terrain = new_world(cfg)
    done = 0
    while done < steps:                       # 烧机
        take = min(500, steps - done)
        state, key, _ = scan_fn(state, key, take)
        done += take

    pref_all, food_all = [], []
    hist0 = None
    for i in range(SAMPLE_STEPS // SAMPLE_EVERY):
        state, key, _ = scan_fn(state, key, SAMPLE_EVERY)
        alive = np.asarray(state.alive)
        diet = np.asarray(state.diet)
        herb = alive & (diet < 0.35)
        p = np.asarray(state_mod.forage_pref_of(state.genome, cfg))[herb]
        f = np.asarray(state.last_food)[herb]
        pref_all.append(p)
        food_all.append(f)
        if i == 0:
            hist0 = np.histogram(p, bins=BINS)[0].astype(float)
    hist1 = np.histogram(pref_all[-1], bins=BINS)[0].astype(float)
    jax.block_until_ready(state.genome)

    p = np.concatenate(pref_all)
    f = np.concatenate(food_all)
    idx = np.digitize(p, BINS) - 1
    ctr = 0.5 * (BINS[:-1] + BINS[1:])
    n = np.array([(idx == i).sum() for i in range(len(ctr))])
    intake = np.array([f[idx == i].mean() if (idx == i).sum() >= 30 else np.nan
                       for i in range(len(ctr))])

    ok = ~np.isnan(intake)
    out = {"seed": seed, "steps": steps, "n_samples": int(len(p)),
           "bin_centers": ctr.tolist(), "bin_n": n.tolist(),
           "intake": np.where(ok, intake, None).tolist(), "overrides": overrides or {}}

    print(f"样本 {len(p)} 个体·快照（{SAMPLE_STEPS // SAMPLE_EVERY} 帧 × 平均 "
          f"{len(p) * SAMPLE_EVERY // SAMPLE_STEPS} 食草者）")
    print(f'\n  {"pref":>6} {"n":>8} {"平均摄入":>10} {"密度变化率":>12}')
    dlog = np.where((hist0 > 20) & (hist1 > 20), np.log(np.maximum(hist1, 1) /
                                                        np.maximum(hist0, 1)), np.nan)
    for i, c in enumerate(ctr):
        print(f"  {c:>6.3f} {n[i]:>8d} "
              f"{intake[i] if ok[i] else float('nan'):>10.4f} "
              f"{dlog[i]:>12.4f}")

    # 二次拟合的二阶系数——§11.1 预测 k=1 时 ≈0、k<1 时 >0（中间是极小）
    for name, y in (("摄入", intake), ("密度变化率", dlog)):
        m = ~np.isnan(y)
        if m.sum() >= 5:
            # 以基因均值为中心、按方差归一，让不同臂的二阶系数可比
            x = (ctr[m] - ctr[m].mean()) / max(ctr[m].std(), 1e-9)
            c2, c1, c0 = np.polyfit(x, y[m], 2)
            rel = c2 / max(abs(c0), 1e-12)
            out[f"quad_{'intake' if name == '摄入' else 'demog'}"] = float(c2)
            out[f"quadrel_{'intake' if name == '摄入' else 'demog'}"] = float(rel)
            print(f"\n  {name}曲线二次拟合: 二阶系数 {c2:+.5f} "
                  f"(÷截距 {rel:+.4f}) ⇒ 中间型是{'**极小(分歧)**' if c2 > 0 else '极大(稳定化)'}")

    if as_json:
        print("JSON " + json.dumps(out))
    print("\n[诊断读数，单 run 不判决。用途：在「曲率没到达曲面」与「到达了但基因流抹平」"
          "之间做选择——两者要的下一步相反。]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="?", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="F=V")
    a = ap.parse_args()
    from underworld.config import parse_overrides
    main(a.steps, a.seed, parse_overrides(a.sets), a.json)
