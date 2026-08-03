"""A 阶段：量出「果层与种群的空间重叠度」的基线数字。

回答什么：`docs/multispecies_program.md` §12.3 A —— 现在果层和种群到底重叠多少？
后面 C 阶段扫 `fruit_dry_weight` 时，「挪开了没有」全部对着这里产出的数读。
没有这个数，「分离」无从判定。

§12.1 的病根在 `underworld/terrain.py:207`：
    fruit_capacity = fruit_max * patch * (forest ** 2) * (1 - rock)
而 `:185` 的 `forest = elev_band * water_prox`——果层的栖息地项是「离水近」做出来的，
于是果实必然长在所有个体本来就挤着的河岸林带里。

怎么量（三个口径，一起报，因为它们会分开动）：

1. **选择比 `sel_ratio`** = 个体加权的 `fruit_capacity` 均值 ÷ 陆地格的 `fruit_capacity`
   均值。1.0 = 与随机放置无异，>1 = 种群坐在果层上。
2. **Schoener's D** = `1 − ½Σ|p_i − q_i|`，`p` 是归一化的个体密度、`q` 是归一化的
   `fruit_capacity`。生态学里标准的生态位重叠指数，0=不相交、1=完全同分布。
   **⚠️ 它带一个混杂，读之前必须知道**：D 同时在惩罚「两个分布的集中度不同」。种群沿河
   高度聚集、果层摊在 2000+ 格上，所以**实测 D 会低于均匀零模型的 D**（实测 0.100 vs
   N-land 0.118），这**不代表种群避开果层**——同一批数据的选择比是 1.36（偏好果层）。
   D 在这里主要是「形状是否分开」的指标，**不要拿它单独当重叠度读**。主口径是选择比。
3. **果层份额 `frac_in_fruit`** = 落在「`fruit_capacity ≥ 5% 场最大值`」的格里的个体质量份额。

**三个零模型，缺一不可**（`CLAUDE.md`：空间指标在有零假设之前没有意义）：

- **N-land**：均匀撒在所有非海格上。这是最朴素的「随机」。
- **N-water**：**按水距分层匹配**——保留实测个体在各 `water_dist` 分箱里的份额，
  但在每个分箱内部把个体均匀摊到该分箱的所有陆地格上。这个零模型问的是：
  **「扣掉『渴死把所有人钉在河边』这一条之后，种群还额外偏好果层吗？」**
  这是本轮真正该读的那个对照——因为手术要动的正是果层与水的耦合。
- **N-uniform**：均匀撒在全部格上（含海），只作量级参照。

**同时报「果层有效格数」**（§12.4 预标风险 3）：`patch` 是正弦格、将来的 `dry` 是环河带，
两者相乘后果层可能碎得只剩零星几格。B 阶段之后要立刻复查这个数——**「挪开了」和
「把果层删了」在选择比上长得一模一样，只有靠它和 `frugivory_frac` 分辨。**

个体密度不取末帧快照，而是在**后 25% 的 run 里每 `sample_every` 步采一次**再累加：
单帧快照的空间噪声远大于我们要测的效应。

重跑（一个 run）：
    XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
      explorations/20260803-overlapA/measure_overlap.py 20000 --seed 0 --json
"""
import argparse
import dataclasses
import json
import sys

sys.path.insert(0, ".")

import jax.numpy as jnp
import numpy as np

from underworld import Config, new_world
from underworld import config as config_mod
from underworld.state import pos_to_cell

SAMPLE_FRACTION = 0.25   # 只用后 25% 的 run（前面是瞬态）
SAMPLE_EVERY = 200       # 每 200 步采一帧密度
FRUIT_EFFECTIVE_FRAC = 0.05   # 「有效果格」的阈值：达到场最大值的 5%


def schoener_d(p: np.ndarray, q: np.ndarray) -> float:
    """生态位重叠指数 D = 1 − ½Σ|p−q|，p/q 都是归一化到和为 1 的密度。"""
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    return float(1.0 - 0.5 * np.abs(p - q).sum())


def water_matched_null(occ: np.ndarray, water_dist: np.ndarray, is_land: np.ndarray,
                       n_bins: int = 24) -> np.ndarray:
    """按水距分层匹配的零模型密度。

    把陆地格按 `water_dist` 分成等宽箱；实测个体在某箱里占了多少质量，就把这些质量
    均匀摊回该箱的所有陆地格。于是零模型与实测**共享同一个水距边缘分布**，两者的差别
    只剩「在同样的水距上，是否偏好果层」。
    """
    land_wd = water_dist[is_land]
    edges = np.linspace(0.0, float(land_wd.max()) + 1e-6, n_bins + 1)
    which = np.digitize(water_dist, edges) - 1
    which = np.clip(which, 0, n_bins - 1)
    null = np.zeros_like(occ)
    for b in range(n_bins):
        sel = (which == b) & is_land
        n_cells_b = int(sel.sum())
        if n_cells_b == 0:
            continue
        mass_b = float(occ[sel].sum())
        if mass_b > 0:
            null[sel] = mass_b / n_cells_b
    return null


def main(steps: int, seed: int, overrides: dict, as_json: bool) -> None:
    cfg = dataclasses.replace(Config(), seed=seed, **(overrides or {}))
    state, key, step_fn, scan_fn, terrain = new_world(cfg)

    fruit_cap = np.asarray(terrain.fruit_capacity, dtype=np.float64)
    water_dist = np.asarray(terrain.water_dist, dtype=np.float64)
    # 海=水距 0 且无果层承载；陆地 = 有草承载力的格（与 terrain 的 is_sea 同源判据）
    capacity = np.asarray(terrain.capacity, dtype=np.float64)
    is_land = capacity > 0.0

    occ = np.zeros(cfg.n_cells, dtype=np.float64)   # 累加的个体密度
    frames = 0
    start_sampling = int(steps * (1.0 - SAMPLE_FRACTION))

    done = 0
    last_m = None
    while done < steps:
        chunk = min(SAMPLE_EVERY, steps - done)
        state, key, ms = scan_fn(state, key, chunk)
        done += chunk
        last_m = ms
        pop = int(np.asarray(state.alive).sum())
        if pop < 1:
            print("!! population collapsed to zero")
            break
        if done >= start_sampling:
            cell = np.asarray(pos_to_cell(state.pos, cfg))
            alive = np.asarray(state.alive)
            np.add.at(occ, cell[alive], 1.0)
            frames += 1

    if occ.sum() <= 0:
        print("!! 没有采到任何个体，无法计算重叠度")
        sys.exit(1)

    # ---- 三个零模型 ----
    null_uniform = np.ones_like(occ)
    null_land = is_land.astype(np.float64)
    null_water = water_matched_null(occ, water_dist, is_land)

    def sel_ratio(null: np.ndarray) -> float:
        """个体加权的 fruit_capacity 均值 ÷ 零模型加权的均值。1.0 = 与该零模型无异。"""
        w_obs = occ / occ.sum()
        w_nul = null / null.sum()
        denom = float((w_nul * fruit_cap).sum())
        return float((w_obs * fruit_cap).sum() / denom) if denom > 0 else float("nan")

    fmax = float(fruit_cap.max())
    eff = fruit_cap >= FRUIT_EFFECTIVE_FRAC * fmax if fmax > 0 else np.zeros_like(is_land)

    def frac_in(mass: np.ndarray) -> float:
        return float(mass[eff].sum() / max(mass.sum(), 1e-12))

    out = {
        "seed": seed,
        "steps": done,
        "frames": frames,
        # --- 主口径：种群坐得离果层多近 ---
        "sel_ratio_land": sel_ratio(null_land),
        "sel_ratio_water": sel_ratio(null_water),
        "sel_ratio_uniform": sel_ratio(null_uniform),
        "schoener_d": schoener_d(occ, fruit_cap),
        "schoener_d_null_land": schoener_d(null_land, fruit_cap),
        "schoener_d_null_water": schoener_d(null_water, fruit_cap),
        "frac_in_fruit": frac_in(occ),
        "frac_in_fruit_null_land": frac_in(null_land),
        "frac_in_fruit_null_water": frac_in(null_water),
        # --- 果层本身的形状（§12.4 风险 3：挪开 vs 删掉，只能靠这几个数分辨）---
        "fruit_cells_pos": int((fruit_cap > 0).sum()),
        "fruit_cells_eff": int(eff.sum()),
        "fruit_cap_total": float(fruit_cap.sum()),
        "fruit_cap_max": fmax,
        "land_cells": int(is_land.sum()),
        # --- 生态护栏 ---
        "population": float(np.asarray(last_m.population)[-1]),
        "carnivore_frac": float(np.asarray(last_m.carnivore_frac)[-1]),
        "frugivory_frac": float(np.asarray(last_m.frugivory_frac)[-1]),
        "herb_water_dist": float(np.asarray(last_m.herb_water_dist)[-1]),
        "forest_frac": float(np.asarray(last_m.forest_frac)[-1]),
        "overrides": overrides or {},
    }

    print(f"\n=== 果层 × 种群 重叠度（seed {seed}, {done} 步, {frames} 帧密度）===")
    print(f"  选择比 sel_ratio   vs N-land  = {out['sel_ratio_land']:.4f}   "
          f"vs N-water = {out['sel_ratio_water']:.4f}   (1.0 = 与零模型无异)")
    print(f"  Schoener D         实测 = {out['schoener_d']:.4f}   "
          f"N-land = {out['schoener_d_null_land']:.4f}   N-water = {out['schoener_d_null_water']:.4f}")
    print(f"  果层份额           实测 = {out['frac_in_fruit']:.4f}   "
          f"N-land = {out['frac_in_fruit_null_land']:.4f}   N-water = {out['frac_in_fruit_null_water']:.4f}")
    print(f"  果层形状           有效格 {out['fruit_cells_eff']} / 正值格 "
          f"{out['fruit_cells_pos']} / 陆地格 {out['land_cells']}   总承载 "
          f"{out['fruit_cap_total']:.1f}")
    print(f"  护栏               pop={out['population']:.0f}  carn={out['carnivore_frac']:.3f}  "
          f"frugivory={out['frugivory_frac']:.4f}  herb_wd={out['herb_water_dist']:.1f}")
    if as_json:
        print("JSON " + json.dumps(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", type=int, nargs="?", default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--set", dest="sets", action="append", default=[])
    a = ap.parse_args()
    main(a.steps, a.seed, config_mod.parse_overrides(a.sets), a.json)
