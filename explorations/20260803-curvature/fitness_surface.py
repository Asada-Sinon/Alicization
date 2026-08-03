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
import math
import sys

sys.path.insert(0, ".")

import jax
import numpy as np

from underworld import Config, new_world
from underworld import state as state_mod

SAMPLE_EVERY = 10          # 抓快照的间隔
SAMPLE_STEPS = 6000        # 烧机后再跑这么多步做统计。**从 2000 提到 6000**：
#                            标定 pilot 实测 quad_intake 的 σ_within=0.00274，那是生态
#                            混沌不是采样噪声，但把窗口拉长仍能多平均掉一部分。
#                            代价是每个 run 约 +1 分钟，相对 96 个 run 的判决很便宜。
BINS = np.linspace(0.15, 0.85, 15)


def main(steps: int, seed: int, overrides: dict, as_json: bool) -> None:
    cfg = dataclasses.replace(Config(), seed=seed, **(overrides or {}))
    state, key, _step, scan_fn, _terrain = new_world(cfg)
    CAUSES = ("predation", "starvation", "thirst", "senescence")
    toll = {c: 0.0 for c in CAUSES}
    pops, row = [], {}

    def _absorb(ms):
        d = ms._asdict()
        for c in CAUSES:
            toll[c] += float(np.asarray(d[f"death_{c}"]).sum())
        pops.append(float(np.asarray(d["population"])[-1]))
        return {k: float(np.asarray(v)[-1]) for k, v in d.items()}

    done = 0
    while done < steps:                       # 烧机
        take = min(500, steps - done)
        state, key, ms = scan_fn(state, key, take)
        row = _absorb(ms)
        done += take

    n_frames = SAMPLE_STEPS // SAMPLE_EVERY
    pref_all, food_all = [], []
    for _ in range(n_frames):
        state, key, ms = scan_fn(state, key, SAMPLE_EVERY)
        row = _absorb(ms)
        alive = np.asarray(state.alive)
        diet = np.asarray(state.diet)
        herb = alive & (diet < 0.35)
        pref_all.append(np.asarray(state_mod.forage_pref_of(state.genome, cfg))[herb])
        food_all.append(np.asarray(state.last_food)[herb])
    jax.block_until_ready(state.genome)

    # 人口学侧：**前 1/4 帧与后 1/4 帧各自合并**再做直方图，不是拿首尾各一帧。
    # 第一版用的是单帧，实测 4 个格里只有 2 个能凑够有效箱——单帧的计数太稀，
    # 而这是要当主判据的读数，估计量不稳比功效不足更糟。合并把计数放大 50 倍。
    q = max(n_frames // 4, 1)
    hist0 = np.histogram(np.concatenate(pref_all[:q]), bins=BINS)[0].astype(float) / q
    hist1 = np.histogram(np.concatenate(pref_all[-q:]), bins=BINS)[0].astype(float) / q

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

    # 分布形状与护栏也一并给出，这样一个 run 同时回答「曲面什么形状」和「分布分开了吗」，
    # 不必为了拿 sd 再跑一遍 probe_trait_dist（同一条纪律：一个 run 答完一次判决要的全部）。
    last = pref_all[-1]
    out["sd"] = float(last.std())
    out["mean_pref"] = float(last.mean())
    sys.path.insert(0, "scripts")
    from probe_trait_dist import bimodality_coefficient, blrt_two_components
    if len(last) >= 30:
        out["bimodality_coefficient"] = float(bimodality_coefficient(last.astype(np.float64)))
        t = blrt_two_components(last.astype(np.float64), n_boot=199, seed=seed)
        out["blrt_lr_per_n"] = t["lr_per_n"]
    # --- dip_ratio：把「真分裂」和「整体移走」分开 ---
    #
    # **为什么不用 frac_mid（中间带 [0.40,0.60] 的占比）**：饱和探针里 S1（真分裂）读 0.0006、
    # S2（只是右移）读 0.058——两者都近乎空，`frac_mid` **分不出**它们。图看出来的：
    # S2 是整个分布移到 0.70 去了，中间带因此空；S1 是中间真的裂开。
    #
    # `dip_ratio` = 实测中间带占比 ÷ **同均值同 sd 的高斯**在该带上的期望质量。
    # 位置与尺度都被除掉了：任何单峰分布读 ~1，真分裂读 ~0。实测 S0 1.049 / S1 0.0031 /
    # S2 0.862——`S0` 那个 1.049 是它自带的估计量对照（构造上单峰必须读 1）。
    # 均值与 sd 都取自**占据分布本身**，与分子同源，不用末帧的 `sd`（那是另一个量）。
    from scipy.stats import norm as _norm
    _n = np.array(n, float)
    _n = _n / max(_n.sum(), 1.0)
    _mid = (ctr >= 0.40) & (ctr <= 0.60)
    _m = float((_n * ctr).sum())
    _sd = math.sqrt(max(float((_n * (ctr - _m) ** 2).sum()), 1e-12))
    _obs = float(_n[_mid].sum())
    _exp = float(_norm.cdf(0.60, _m, _sd) - _norm.cdf(0.40, _m, _sd))
    out["frac_mid"] = _obs
    out["dip_ratio"] = _obs / max(_exp, 1e-9)
    out["occ_mean"] = _m
    out["occ_sd"] = _sd
    print(f"  dip_ratio = {out['dip_ratio']:.4f}  (中间带实测 {_obs:.4f} ÷ 同均值同sd高斯期望 "
          f"{_exp:.4f})  ⇒ {'**真分裂**' if out['dip_ratio'] < 0.3 else '单峰'}")

    total = max(sum(toll.values()), 1.0)
    out.update({f"death_{k}_frac": toll[k] / total for k in CAUSES})
    out["min_pop"] = float(np.min(pops)) if pops else float("nan")
    for g in ("population", "carnivore_frac", "carn_speed", "herb_speed",
              "frugivory_frac", "graze_gain", "fruit_gain"):
        out[g] = row.get(g, float("nan"))

    print(f"样本 {len(p)} 个体·快照（{SAMPLE_STEPS // SAMPLE_EVERY} 帧 × 平均 "
          f"{len(p) * SAMPLE_EVERY // SAMPLE_STEPS} 食草者）")
    print(f'\n  {"pref":>6} {"n":>8} {"平均摄入":>10} {"密度变化率":>12}')
    # 门槛必须与下面拟合用的掩码**同一个**——第一版这里写 >20、拟合写 >5，
    # 于是拟合把 dlog 为 NaN 的箱也纳进去，整条曲线读出 NaN。
    MIN_CT = 5.0
    dmask = (hist0 > MIN_CT) & (hist1 > MIN_CT)
    dlog = np.where(dmask, np.log(np.maximum(hist1, 1e-9) / np.maximum(hist0, 1e-9)),
                    np.nan)
    for i, c in enumerate(ctr):
        print(f"  {c:>6.3f} {n[i]:>8d} "
              f"{intake[i] if ok[i] else float('nan'):>10.4f} "
              f"{dlog[i]:>12.4f}")

    # --- 二次系数：§11.1 预测 k=1 时 ≈0、k<1 时 >0（中间型是极小值）---
    #
    # **第一版把两条曲线都建在「先分箱求均值、再对 4–5 个箱心做 polyfit」上，那个估计量
    # 太脆**：有效箱数随臂变化（k=1 有 5 个、k=0.5 有 4 个），每个箱心的噪声又差几个数量级
    # （n 从 40 到 90939）。主判据不能建在这种东西上。两条都改成加权，权重就是各自的样本量。
    #
    # 摄入侧干脆不分箱：直接在**个体级**上回归 last_food ~ 1 + z + z²。分箱只会丢掉信息，
    # 而个体级回归的自由度是 3×10^5 而不是 5。
    zc, zs = p.mean(), max(p.std(), 1e-9)
    z = (p - zc) / zs
    X = np.stack([np.ones_like(z), z, z * z], axis=1)
    beta, *_ = np.linalg.lstsq(X, f, rcond=None)
    resid = f - X @ beta
    # 系数的标准误——用来判「二次项是不是真的不为 0」，不是拿来当跨臂噪声
    dof = max(len(f) - 3, 1)
    cov = np.linalg.pinv(X.T @ X) * float(resid @ resid) / dof
    se2 = float(np.sqrt(max(cov[2, 2], 0.0)))
    # 一阶项与 z 的尺度也存下来。**判决时靠分箱重建这两个量，精度低于个体级回归**，
    # 而多存两个 float 的成本是零。一阶项是 §11.1 的一阶预测（A>B ⇒ 选择推向草）的
    # 直接读数，本轮判决最有力的证据之一就出自它。
    out["lin_intake"] = float(beta[1])
    out["pref_sd"] = float(zs)
    out["pref_mean"] = float(zc)
    out["quad_intake"] = float(beta[2])
    out["quadrel_intake"] = float(beta[2] / max(abs(beta[0]), 1e-12))
    out["quad_intake_se"] = se2
    print(f"\n  摄入曲面（个体级回归，n={len(f)}）: 二次项 {beta[2]:+.5f} ± {se2:.5f}"
          f"  (÷截距 {beta[2] / max(abs(beta[0]), 1e-12):+.4f})"
          f"  ⇒ 中间型是{'**极小(分歧)**' if beta[2] > 0 else '极大(稳定化)'}")

    # 人口学侧仍需分箱（密度变化率本来就是箱量），但改成**按样本量加权**的二次拟合，
    # 并把「有效箱」的门槛从 20 降到 5 同时靠权重自动压低稀疏箱的影响。
    m = dmask
    if m.sum() >= 4:
        zb = (ctr[m] - zc) / zs
        w = np.sqrt(np.minimum(hist0[m], hist1[m]))     # ~ dlog 的精度 ∝ √n
        Xb = np.stack([np.ones_like(zb), zb, zb * zb], axis=1) * w[:, None]
        bb, *_ = np.linalg.lstsq(Xb, dlog[m] * w, rcond=None)
        out["quad_demog"] = float(bb[2])
        out["quadrel_demog"] = float(bb[2] / max(abs(bb[0]), 1e-12))
        out["demog_bins"] = int(m.sum())
        print(f"  人口学曲面（{int(m.sum())} 箱，按 √n 加权）: 二次项 {bb[2]:+.5f}"
              f"  (÷截距 {bb[2] / max(abs(bb[0]), 1e-12):+.4f})"
              f"  ⇒ 中间型是{'**极小(分歧)**' if bb[2] > 0 else '极大(稳定化)'}")
    else:
        out["quad_demog"] = float("nan")
        print(f"  人口学曲面: 有效箱只有 {int(m.sum())} 个，不拟合")

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
