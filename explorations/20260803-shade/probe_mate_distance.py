"""**前提检查：交配到底有多「远」？** 先量前提，再决定要不要造机制。

为什么问这个
------------
R9（`docs/multispecies_feasibility.md` §10）测到了本纲领第一个空间生态型信号——
遮荫世界里个体按基因在空间上分选（原始 `r` 的 `T1−N1` = −0.2697，12/12，p 触地板）。
但**基因均值动了、方差没动**：那是一条 cline，不是两个物种。

而 `experiments.md §5.4` 早就写过这条负结果的方法学教训：

> 频率依赖让「份额不够所以专精不划算」这个论证失效……**真正卡住分化的是选择梯度 vs
> 基因流，不是份额。**

读 `reproduction._assortative_mate` 会发现：它把**全部**想繁殖的个体按 `diet`（或消融时按
均匀随机数）**全局**排序，取相邻名次配对——**配偶选择里没有任何空间项**。
所以每一代都把两个生境的基因池完全混匀。空间分选 + 全局交配 = cline，教科书结论。

但「读代码觉得它是全局的」和「实测配偶距离等于随机配对」是两回事。本脚本量后者。

量什么
------
在跑完的世界末态上，照 `reproduce` 的原样重算 `want`，调用**真正的**
`_assortative_mate`，然后量配对个体之间的环面距离，对上两个零模型：

1. **随机配对零模型**：在同一批 wanter 里随机置换配对。若实测 ≈ 这个，交配就是空间随机的。
2. **理论值**：边长 L 的环面上两点平均距离 ≈ 0.5214·L（每轴独立均匀，
   `E|Δ| = L/4`，再取二维欧氏期望）。给一个不依赖本次样本的锚。

顺带量**空间分选的绝对尺度**：把食草者按 `forage_pref` 分成上下三分位，量两组重心间距
与各自到「果富格」的距离。**没有这个尺度，就不知道「局部交配」该局部到什么程度才有意义。**

    XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260803-shade/probe_mate_distance.py 20000 --seed 0 \
        --set ...（R9 的世界）

**单个 run 什么都不判决**（`conventions.md §5`）。这是**前提检查**，不是结论：
它要回答的是「有没有必要造那个机制」这个是非题，而不是「机制有多大效应」。
"""
import argparse
import dataclasses
import json
import sys

sys.path.insert(0, ".")

import jax
import jax.numpy as jnp
import numpy as np

from underworld import Config, new_world
from underworld import state as state_mod
from underworld.reproduction import _assortative_mate
from underworld.spatial import pos_to_cell

CHUNK = 500


def torus_dist(a: np.ndarray, b: np.ndarray, L: float) -> np.ndarray:
    d = np.abs(a - b)
    d = np.minimum(d, L - d)
    return np.sqrt((d ** 2).sum(axis=1))


def main(steps: int, seed: int, overrides: dict, as_json: bool) -> None:
    cfg = dataclasses.replace(Config(), seed=seed, **(overrides or {}))
    state, key, _step, scan_fn, terrain = new_world(cfg)
    done = 0
    while done < steps:
        take = min(CHUNK, steps - done)
        state, key, _ms = scan_fn(state, key, take)
        done += take
    jax.block_until_ready(state.genome)

    L = cfg.world_size
    alive = np.asarray(state.alive)
    pos = np.asarray(state.pos)
    diet = np.asarray(state.diet)
    pref = np.asarray(state_mod.forage_pref_of(state.genome, cfg))

    # **候选集用 `alive`，不用 `want`——这是第一版的一个方法学缺陷，记在这里。**
    # 第一版照 `reproduce` 原样算 `want = alive & (energy > repro_threshold)`，实测
    # `n_want = 0`：`scan` 的每一步末尾都在繁殖**之后**，想繁殖的刚付完 `invest` 全部掉到
    # 阈值以下，所以步边界上的快照系统性地看不到 wanter。这不是世界的性质，是采样点的性质。
    #
    # 换成 `alive` 是合法的，因为**本探针问的是那条规则的空间性质**：
    # `_assortative_mate` 是「按 diet 全局排序、取相邻名次配对」，它的空间局部性来自
    # 「diet 与位置有没有关系」，与候选集是全体还是其中能量最高的一部分无关。
    # 用 `alive` 还顺带把样本量放大了一个量级。
    assert cfg.density_repro_penalty == 0.0, "此探针只覆盖 density_repro_penalty=0 的默认支"
    want = alive.copy()
    n_want = int(want.sum())
    n_over = int((alive & (np.asarray(state.energy) > cfg.repro_threshold)).sum())
    print(f"（候选集 = 全部存活个体；步边界上真正 energy>repro_threshold 的只有 "
          f"{n_over} 个，见脚本内注释）")

    out = {"seed": seed, "steps": steps, "n_alive": int(alive.sum()), "n_want": n_want,
           "overrides": overrides or {}}
    print(f"n_alive={int(alive.sum())}  n_want={n_want}")

    if n_want >= 20:
        mate = np.asarray(_assortative_mate(
            jnp.asarray(want), jnp.asarray(diet), cfg, jax.random.PRNGKey(seed)))
        w = np.where(want)[0]
        # 自配对（wanter 数为奇数时那一个）不计入距离统计
        paired = w[mate[w] != w]
        d_obs = torus_dist(pos[paired], pos[mate[paired]], L)

        rng = np.random.default_rng(seed + 99)
        d_null = np.concatenate([
            torus_dist(pos[w], pos[rng.permutation(w)], L) for _ in range(20)])

        theory = 0.5214 * L
        out.update(mate_dist_mean=float(d_obs.mean()),
                   mate_dist_median=float(np.median(d_obs)),
                   mate_dist_p10=float(np.percentile(d_obs, 10)),
                   null_dist_mean=float(d_null.mean()),
                   theory_random_mean=theory,
                   ratio_obs_over_null=float(d_obs.mean() / d_null.mean()),
                   frac_within_cell=float((d_obs < cfg.cell_size).mean()),
                   frac_within_vision=float((d_obs < cfg.vision_radius).mean()))
        print(f"\n配偶距离（{len(paired)} 对，世界边长 {L:.0f}，格边长 {cfg.cell_size:.1f}）")
        print(f"  实测       均值 {d_obs.mean():>7.2f}  中位 {np.median(d_obs):>7.2f}  "
              f"P10 {np.percentile(d_obs, 10):>7.2f}")
        print(f"  随机配对零模型 均值 {d_null.mean():>7.2f}")
        print(f"  环面理论随机   均值 {theory:>7.2f}")
        print(f"  实测/零模型 = {d_obs.mean() / d_null.mean():.4f}   "
              f"(1.00 = 交配在空间上完全随机)")
        print(f"  配偶落在同一个植被格内的比例 {100 * (d_obs < cfg.cell_size).mean():.2f}%")
        print(f"  配偶落在一个视野半径内的比例 {100 * (d_obs < cfg.vision_radius).mean():.2f}%")

    # 空间分选的绝对尺度：食草者按 forage_pref 分上下三分位
    herb = alive & (diet < 0.35)
    if herb.sum() >= 60:
        p = pref[herb]
        px = pos[herb]
        lo, hi = np.percentile(p, [33.3, 66.7])
        grass_side = px[p >= hi]      # forage_pref 高 = 草专精（dynamics._forage_pref_scale）
        fruit_side = px[p <= lo]
        fcap = np.asarray(terrain.fruit_capacity)
        cells = np.asarray(pos_to_cell(jnp.asarray(px), cfg))
        share_hi = fcap[cells[p >= hi]].mean()
        share_lo = fcap[cells[p <= lo]].mean()

        def centroid(q):     # 环面重心用圆均值，直接取算术平均在环面上是错的
            ang = 2 * np.pi * q / L
            return L * np.mod(np.arctan2(np.sin(ang).mean(0), np.cos(ang).mean(0)),
                              2 * np.pi) / (2 * np.pi)
        c_g, c_f = centroid(grass_side), centroid(fruit_side)
        sep = float(torus_dist(c_g[None, :], c_f[None, :], L)[0])
        out.update(ecotype_centroid_sep=sep,
                   fruitcap_at_grass_specialists=float(share_hi),
                   fruitcap_at_fruit_specialists=float(share_lo))
        print(f"\n空间分选的绝对尺度（食草者上/下三分位，n={int(herb.sum())}）")
        print(f"  两个生态型的重心间距 {sep:>7.2f} 世界单位")
        print(f"  草专精者所在格的果承载 {share_hi:.4f}  "
              f"果专精者 {share_lo:.4f}  比值 {share_lo / max(share_hi, 1e-9):.3f}")

    if as_json:
        print("JSON " + json.dumps(out))
    print("\n[前提检查，非结论。它回答「有没有必要造那个机制」，不回答「效应多大」。]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="?", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="F=V")
    a = ap.parse_args()
    from underworld.config import parse_overrides
    main(a.steps, a.seed, parse_overrides(a.sets), a.json)
