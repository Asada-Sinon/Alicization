"""R20 三张地形到底差多少（复核指令第 1 点）。

关键怀疑：`ridge_base_frac` 0.5→0.35 在环面上是不是山脊/河系的**刚性平移**——
`height_at` 里 d = |_wrap(y − center_y)|（terrain.py:68，全程环面 wrap），
若是，则 b35 的「新河系」不成立（CLAUDE.md：刚性平移不算变地形）。
唯一不跟着平移的是果补丁格 patch = sin(7x)·sin(11y)（绝对坐标，terrain.py:269-270）。

用 run 时冻结代码（git archive 9439d3d，underworld/ 与 HEAD 无 diff）建三张图 +
b35 的解析平移检验 + 三张图的生态聚合量。
跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
      explorations/20260815-r20-audit/terrain_stats.py <frozen_dir>
"""
import sys
FROZEN = sys.argv[1]
sys.path.insert(0, FROZEN)
import dataclasses
import numpy as np
import jax.numpy as jnp
from underworld.config import Config
from underworld import terrain

BASE = dict(fruit_energy=4.0, fruit_water_frac=0.40, plant_max=2.0, water_sea_dist=True,
            grass_shade=1.3, forage_tradeoff=1.0, forage_curvature=1.0, eat_rate=0.5,
            fruit_regrow_baseline=0.25, regrow_baseline=0.010)
ARMS = {"wn1": dict(ridge_wavenumber=1), "wn2": dict(ridge_wavenumber=2),
        "b35": dict(ridge_wavenumber=1, ridge_base_frac=0.35)}
T, C = {}, {}
for k, ov in ARMS.items():
    cfg = dataclasses.replace(Config(), **BASE, **ov)
    T[k] = terrain.build(cfg); C[k] = cfg
    g = cfg.grid
    h = np.asarray(T[k].height); wd = np.asarray(T[k].water_dist)
    fo = np.asarray(T[k].forest); cap = np.asarray(T[k].capacity)
    fc = np.asarray(T[k].fruit_capacity)
    sea = h < cfg.sea_level
    drink = wd < cfg.river_half_width
    print(f"{k}: sea_frac={sea.mean():.4f} 可饮水格频={drink.mean():.4f} ({int(drink.sum())}格)"
          f" forest均={fo.mean():.4f} forest>0.25占={np.mean(fo>0.25):.4f}"
          f" capacity总={cap.sum():.1f} fruit_cap总={fc.sum():.2f}"
          f" fruit>0格={int((fc>1e-6).sum())} 高程均={h.mean():.4f}")
print()
# ---- b35 vs wn1 平移检验（解析：把 wn1 的连续场在 y 方向平移 −0.15·S 采样） ----
cfg1 = C["wn1"]; S = cfg1.world_size
centers = np.asarray(terrain._cell_centers(cfg1))
shift = np.array([0.0, (0.5 - 0.35) * S])   # b35 的 center_y 低 0.15·S ⇒ wn1 场在 y+0.15S 处取值
h_wn1_shifted = np.asarray(terrain.height_at(jnp.asarray((centers + shift) % S), cfg1))
h_b35 = np.asarray(T["b35"].height)
print(f"b35 高程场 vs wn1 高程场平移 0.15·S：max|Δ|={np.abs(h_b35 - h_wn1_shifted).max():.2e}"
      f"  （~0 ⇒ 山脊/盆地/海在环面上是精确刚性平移）")
# 离散场的整数格 roll 相关（grid=128，0.15·128=19.2 格 ⇒ 相位差 0.2 格）
g = cfg1.grid
h1 = np.asarray(T["wn1"].height).reshape(g, g)   # 行 = iy
hb = h_b35.reshape(g, g)
for name, f1, f2 in (("height", h1, hb),
                     ("water_dist", np.asarray(T["wn1"].water_dist).reshape(g,g), np.asarray(T["b35"].water_dist).reshape(g,g)),
                     ("fruit_cap", np.asarray(T["wn1"].fruit_capacity).reshape(g,g), np.asarray(T["b35"].fruit_capacity).reshape(g,g)),
                     ("capacity", np.asarray(T["wn1"].capacity).reshape(g,g), np.asarray(T["b35"].capacity).reshape(g,g))):
    cors = [np.corrcoef(np.roll(f1, -kk, axis=0).ravel(), f2.ravel())[0,1] for kk in range(g)]
    kbest = int(np.argmax(cors))
    print(f"  {name:10s} 整数roll最大相关 r={max(cors):.4f} @ shift={kbest}格（19.2格=预期平移）"
          f"   r@19={cors[19]:.4f} r@20={cors[20]:.4f} 无平移 r@0={cors[0]:.4f}")
# wn2 与 wn1 的任意 y-roll 最大相关（wn2 是不是也「接近同一张图」）
h2 = np.asarray(T["wn2"].height).reshape(g, g)
cors2 = [np.corrcoef(np.roll(h1, -kk, axis=0).ravel(), h2.ravel())[0,1] for kk in range(g)]
print(f"  wn2 height vs wn1 任意y-roll最大相关 r={max(cors2):.4f}（低 ⇒ 几何真的不同）")
# 果格相对山脊的相位：0.15·S / (S/11) = 1.65 周期 ⇒ 相对相位差 0.65 周期
print(f"\n果补丁 y 周期 = S/{cfg1.fruit_wavenumber_y} = {S/cfg1.fruit_wavenumber_y:.1f}，"
      f"平移 {0.15*S:.1f} = {0.15*S/(S/cfg1.fruit_wavenumber_y):.2f} 周期 ⇒ 相对相位 0.65 周期")
# 果层与河岸带的共位性：fruit 质量的 water_dist 分布（三图）
for k in ARMS:
    fc = np.asarray(T[k].fruit_capacity); wd = np.asarray(T[k].water_dist)
    w = fc / fc.sum()
    print(f"  {k}: fruit 质量加权的 water_dist 均值={float((w*wd).sum()):.2f}"
          f"  P50={float(np.interp(0.5, np.cumsum(w[np.argsort(wd)]), np.sort(wd))):.2f}")
