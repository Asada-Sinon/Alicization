"""录一段带**个体位置**的回放数据——现有 `traj` 只有直方图，看不到谁在地图哪儿。

**这是可视化的下一步**：直方图能说明「分成了两群」，但看不见
「吃果子的聚在林子边、吃草的散在草地上」。要看到那个就得存每个个体的坐标。

数据量的处理
------------
朴素存法是 `n_max × 4 个 float32 × 帧数`，一个 300 帧的 run 就 ~10 MB，
塞进网页太重。这里做三件压缩：
- **只存活着的**（`alive` 掩码），死的不占位；
- **坐标量化成 uint16**（世界 512 单位 ⇒ 精度 0.008 单位，远超肉眼可辨）；
- **`forage_pref` 与 `diet` 量化成 uint8**（1/255 ≈ 0.004，直方图才 20 箱）。
⇒ 每个体 6 字节，2000 个体 300 帧约 **3.6 MB**，可接受。

同时存地形（一次，不随时间变）：高度与可饮水掩码，让页面能画出河和山。

跑（**要占卡，约 20 分钟**）：
    XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260816-replay/record_map.py --steps 100000 --frames 300
"""
import argparse
import dataclasses
import json
import os
import sys

sys.path.insert(0, ".")

import numpy as np

from underworld import Config, new_world
from underworld import state as state_mod

# 与 R18-B / R20 完全相同的世界（无捕食那一臂），这样录出来的和判决里说的是同一个东西
WORLD = dict(fruit_energy=4.0, fruit_water_frac=0.40, plant_max=2.0, water_sea_dist=1,
             grass_shade=1.3, forage_tradeoff=1.0, forage_curvature=1.0, eat_rate=0.5,
             ridge_wavenumber=1, fruit_regrow_baseline=0.25, regrow_baseline=0.010,
             diet_delta=1.5)
OUT_DIR = "explorations/20260816-replay"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pred", action="store_true", help="录有捕食者的那一臂（diet_delta 用默认）")
    a = ap.parse_args()

    w = dict(WORLD)
    if a.pred:
        w.pop("diet_delta")            # 用默认 0.15 = 有捕食
    cfg = dataclasses.replace(Config(), seed=a.seed, **w)
    state, key, _step, scan_fn, terrain = new_world(cfg)

    every = max(a.steps // a.frames, 1)
    S = float(cfg.world_size)
    frames, done = [], 0
    print(f"录制 {a.steps} 步、每 {every} 步一帧（目标 {a.frames} 帧），"
          f"{'有' if a.pred else '无'}捕食者")
    while done < a.steps:
        n = min(every, a.steps - done)
        state, key, ms = scan_fn(state, key, n)
        done += n
        alive = np.asarray(state.alive)
        pos = np.asarray(state.pos)[alive]
        pref = np.asarray(state_mod.forage_pref_of(state.genome, cfg))[alive]
        diet = np.asarray(state.diet)[alive]
        gen = float(np.asarray(ms.mean_age)[-1]) if hasattr(ms, "mean_age") else 0.0
        frames.append({
            "t": done,
            "g": round(float(np.asarray(state.generation)[alive].mean()), 1),
            # 量化：坐标 uint16、性状 uint8
            "x": np.clip(pos[:, 0] / S * 65535, 0, 65535).astype(np.uint16).tolist(),
            "y": np.clip(pos[:, 1] / S * 65535, 0, 65535).astype(np.uint16).tolist(),
            "p": np.clip(pref * 255, 0, 255).astype(np.uint8).tolist(),
            "d": np.clip(diet * 255, 0, 255).astype(np.uint8).tolist(),
        })
        if len(frames) % 50 == 0:
            print(f"  {done}/{a.steps} 步，第 {frames[-1]['g']:.0f} 代，"
                  f"{len(frames[-1]['x'])} 个存活")
    # 地形只存一次
    g = cfg.grid
    terr = {
        "grid": g,
        "height": np.clip(np.asarray(terrain.height).reshape(g, g) * 255, 0, 255)
                    .astype(np.uint8).ravel().tolist(),
        "water": (np.asarray(terrain.water_dist).reshape(g, g) < cfg.river_half_width)
                    .astype(np.uint8).ravel().tolist(),
    }
    out = {"world": S, "terrain": terr, "frames": frames,
           "pred": bool(a.pred), "seed": a.seed, "steps": a.steps}
    name = f"map_{'pred' if a.pred else 'nopred'}_s{a.seed}.json"
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"\n✓ {path}（{os.path.getsize(path)/1e6:.1f} MB，{len(frames)} 帧）")


if __name__ == "__main__":
    main()
