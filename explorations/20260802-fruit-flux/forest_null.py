"""forest_frac 的零模型：随机放置的种群会有多少比例站在林冠下？

为什么需要: metrics.py:232 的 forest_frac = 站在 (terrain.forest[cell] > 0.5) 的活体占比,
          是一条空间指标。conventions.md §7 的规矩: 没有零模型之前 0.207 / 0.238 不能
          叫「高」或「低」。terrain.build(cfg) 不用 RNG，三个臂共用同一张地图，
          所以零模型对 A/B/C 完全相同（fruit_regrow_baseline / fruit_energy 不进 terrain）。
读哪些文件: 无 run 产物；直接从 underworld.config.Config() 默认值 build 地形。
输出怎么读: 三个零模型 = 全格均匀 / 可居住格(capacity>0)均匀 / 按 capacity 加权。
重跑: XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python explorations/20260802-fruit-flux/forest_null.py
"""
import sys; sys.path.insert(0, '.')
import numpy as np
from underworld.config import Config
from underworld import terrain as T

cfg = Config()
ter = T.build(cfg)
forest = np.asarray(ter.forest)
cap = np.asarray(ter.capacity)
fcap = np.asarray(ter.fruit_capacity)
canopy = forest > 0.5                      # metrics.py:232 用的正是这个阈值
hab = cap > 0.0

print(f'grid={cfg.grid}  n_cells={forest.size}')
print(f'canopy 格数 (forest>0.5) = {canopy.sum()}   可居住格 (capacity>0) = {hab.sum()}')
print()
print('forest_frac 的零模型（随机放置时的期望值）:')
print(f'  [N1] 全格均匀                 = {canopy.mean():.4f}')
print(f'  [N2] 可居住格 (capacity>0) 均匀 = {(canopy & hab).sum() / hab.sum():.4f}')
print(f'  [N3] 按草层承载力 capacity 加权 = {cap[canopy].sum() / cap.sum():.4f}')
print(f'  [N4] 按果层承载力 fruit_capacity 加权 = {fcap[canopy].sum() / fcap.sum():.4f}')
print()
print('实测（explorations/20260802-fruit-flux/analyze.py §1 的臂均值）:')
for arm, v in [('A_base', 0.20731), ('B_frb10_fe6', 0.23835), ('C_frb25_fe4', 0.23937)]:
    print(f'  {arm:14s} forest_frac={v:.5f}   相对 N2 = {v - (canopy & hab).sum()/hab.sum():+.4f}'
          f'   相对 N3 = {v - cap[canopy].sum()/cap.sum():+.4f}')
print()
print('注: 上面三个实测值是从 analyze.py 的 stdout 抄进来的常量，只用于并排显示；'
      '零模型 N1..N4 是本脚本算的。')
