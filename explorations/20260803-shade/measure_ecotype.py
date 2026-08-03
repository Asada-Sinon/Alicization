"""R9 的测量脚本：一个 run 同时给出**形状**、**空间**和全部护栏。

判据见 `docs/multispecies_program.md` §13.5（先写后跑，commit 已落）。三个主判据：

- **H1** `forage_pref` 在食草世系上的 `sd`
- **H2** `blrt_lr_per_n`（1-vs-2 高斯成分的自助似然比，除以 n）
- **H3** `rho_space` —— 个体的 `forage_pref` 与它所在格「果实能量份额」的相关，
  外加**位置置换零模型**。**H1/H2 说基因分开了，H3 说它们住在不同地方，
  只有 H3 成立才叫生态位分化。**

为什么不直接用 `scripts/probe_trait_dist.py`：它给不出 H3，而 H3 才是「**局部**多物种
生态箱」的字面含义（§1.1 空间轴那条明写「需要生态型 × 空间的联合分布这个新读数」）。
形状那两个统计**直接 import 它的实现**，不重新推导——`exp_stats.py` 的教训同理。

    XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260803-shade/measure_ecotype.py 20000 --seed 0 --json \
        --set grass_shade=1.3 --set forage_tradeoff=1.0 ...

重跑整轮：`bash explorations/20260803-shade/sweep_r9.sh`
"""
import argparse
import dataclasses
import json
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

import jax
import numpy as np

from probe_trait_dist import blrt_two_components
from underworld import Config, new_world
from underworld import state as state_mod
from underworld.spatial import pos_to_cell

N_PERM = 200          # 位置置换零模型的重复数
CHUNK = 500


def rho_space(pref: np.ndarray, cell: np.ndarray, share: np.ndarray,
              rng: np.random.Generator) -> dict:
    """生态型 × 空间的联合分布：`forage_pref` 与所在格果实能量份额的 Pearson 相关。

    零模型**置换个体的位置**而不是重采样格子：这样基因值的边缘分布和被占据格的边缘
    分布都原样保留，被打断的只有「谁站在哪」这一层配对。§12.3.A 的教训是零模型用错了
    变量就什么也回答不了——那里的 `sel_ratio_water` 零模型用的是处理后变量，于是它
    回答不了「种群有没有离开河岸」。

    返回观测值、置换零分布的均值/标准差，以及去量纲的 z。**z 才是可跨臂比较的那个数**，
    因为不同臂的果实份额场本身就不同（遮荫臂上份额更极端），r 的可达上限随之改变——
    这正是 §12.3.C 里 `sel_ratio_water` 失效的那种方式。
    """
    s = share[cell]
    if np.std(pref) < 1e-9 or np.std(s) < 1e-9:
        return {"rho_space": float("nan"), "rho_null_mean": float("nan"),
                "rho_null_sd": float("nan"), "rho_space_z": float("nan")}
    obs = float(np.corrcoef(pref, s)[0, 1])
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        null[i] = np.corrcoef(pref, s[rng.permutation(len(s))])[0, 1]
    m, sd = float(null.mean()), float(null.std())
    return {"rho_space": obs, "rho_null_mean": m, "rho_null_sd": sd,
            "rho_space_z": (obs - m) / max(sd, 1e-12)}


def main(steps: int, seed: int, overrides: dict, as_json: bool) -> None:
    cfg = dataclasses.replace(Config(), seed=seed, **(overrides or {}))
    state, key, _step, scan_fn, terrain = new_world(cfg)

    # 「果实能量份额」= 这个格上果实能提供的能量 ÷ (果实 + 草)。用**承载力**而不是
    # 当前存量：存量被种群自己吃出来的坑污染，那会把「个体在哪」和「资源在哪」搅在一起。
    # 草每单位生物量 1.0 能量（dynamics.graze），果每单位 fruit_energy（dynamics.eat_fruit），
    # 两者同吃 eat_efficiency，所以比值里它约掉。
    cap = np.asarray(terrain.capacity, dtype=np.float64)
    fcap = np.asarray(terrain.fruit_capacity, dtype=np.float64) * cfg.fruit_energy
    share = fcap / np.maximum(fcap + cap, 1e-12)

    CAUSES = ("predation", "starvation", "thirst", "senescence")
    toll = {f"death_{c}": 0.0 for c in CAUSES}
    row: dict = {}
    pops: list[float] = []
    done = 0
    while done < steps:
        take = min(CHUNK, steps - done)
        state, key, ms = scan_fn(state, key, take)
        d = ms._asdict()
        for c in CAUSES:
            toll[f"death_{c}"] += float(np.asarray(d[f"death_{c}"]).sum())
        row = {k: float(np.asarray(v)[-1]) for k, v in d.items()}
        pops.append(row["population"])
        done += take
        if row["population"] < 1:
            print("!! population collapsed to zero")
            break
    jax.block_until_ready(state.genome)

    total = max(sum(toll.values()), 1.0)
    guards = {f"death_{c}_frac": toll[f"death_{c}"] / total for c in CAUSES}
    guards["min_pop"] = float(np.min(pops)) if pops else float("nan")
    guards["total_deaths"] = total

    pref = np.asarray(state_mod.forage_pref_of(state.genome, cfg))
    alive = np.asarray(state.alive)
    diet = np.asarray(state.diet)
    herb = alive & (diet < 0.35)          # 与 probe_trait_dist 同口径
    cell = np.asarray(pos_to_cell(state.pos, cfg))

    out = {"seed": seed, "steps": steps, "n": int(herb.sum()),
           **row, **guards, "overrides": overrides or {}}

    if herb.sum() < 30:
        out["note"] = "too few herbivore carriers"
        print("!! 食草世系携带者不足 30，形状与空间统计都不做")
    else:
        x = pref[herb].astype(np.float64)
        t = blrt_two_components(x, n_boot=199, seed=seed)
        out.update(sd=float(x.std()), mean=float(x.mean()),
                   blrt_lr=t["lr"], blrt_lr_per_n=t["lr_per_n"])
        out.update(rho_space(x, cell[herb], share,
                             np.random.default_rng(seed + 10007)))
        print(f"n_herb={int(herb.sum())}  sd={x.std():.5f}  "
              f"blrt_lr_per_n={t['lr_per_n']:.5f}")
        print(f"rho_space={out['rho_space']:+.4f}  null={out['rho_null_mean']:+.4f}"
              f"±{out['rho_null_sd']:.4f}  z={out['rho_space_z']:+.2f}")
        print(f"guards: pop={row.get('population', float('nan')):.0f} "
              f"min_pop={guards['min_pop']:.0f} "
              f"carn={row.get('carnivore_frac', float('nan')):.4f} "
              f"carn_speed={row.get('carn_speed', float('nan')):.3f} "
              f"herb_speed={row.get('herb_speed', float('nan')):.3f} "
              f"frugivory={row.get('frugivory_frac', float('nan')):.4f} "
              f"thirst={guards['death_thirst_frac']:.4f}")

    if as_json:
        print("JSON " + json.dumps(out))
    print("[PROBE -- 单个 run 什么都不判决。判据是 §13.5 的 12 种子 × 2 重复交互项。]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="?", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="F=V")
    a = ap.parse_args()
    from underworld.config import parse_overrides
    main(a.steps, a.seed, parse_overrides(a.sets), a.json)
