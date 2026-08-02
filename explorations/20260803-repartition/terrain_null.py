"""forest_frac 的零假设: 随机放置的个体有多大比例站在树冠下 (forest>0.5)?

回答什么: §3 里 F/G 两臂的 forest_frac 相对 A_base 6/6 上升 +0.065/+0.051。
          但 forest_frac 是「站在树冠下的种群比例」(underworld/metrics.py:29,232)，
          是一条空间 claim，脱离零假设读不出高/低 (docs/conventions.md §7 的做法)。
          本脚本从 terrain 直接算三个零假设: 全格均匀 / 可居住格均匀 / 按承载力加权。
读哪些文件: underworld/{config,terrain}.py (terrain.build 不用 RNG，与种子无关)。
          注意 plant_max 会改 capacity，所以按承载力加权的零假设逐臂算一遍。
输出怎么读: stdout 三个零假设值 + 三臂实测均值，报告中不得手算。
重跑: XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python explorations/20260803-repartition/terrain_null.py
"""
import sys; sys.path.insert(0, ".")
import numpy as np
from dataclasses import replace
from underworld.config import Config
from underworld import terrain as T

for label, pm in [('默认 plant_max (A_base)', None), ('plant_max=2.0 (F/G 两臂)', 2.0)]:
    cfg = Config() if pm is None else replace(Config(), plant_max=pm)
    tr = T.build(cfg)
    forest = np.asarray(tr.forest)
    cap = np.asarray(tr.capacity)
    fcap = np.asarray(tr.fruit_capacity)
    canopy = forest > 0.5
    habitable = cap > 0.0
    print(f'--- {label}  (plant_max={cfg.plant_max}, n_cells={forest.size}) ---')
    print(f'  forest>0.5 的格子数 = {canopy.sum()} / {forest.size}')
    print(f'  零假设1 全格均匀放置          -> forest_frac = {canopy.mean():.4f}')
    print(f'  零假设2 可居住格(capacity>0)均匀 -> forest_frac = '
          f'{canopy[habitable].mean():.4f}   (可居住格 {habitable.sum()})')
    print(f'  零假设3 按草承载力加权         -> forest_frac = '
          f'{(canopy * cap).sum() / cap.sum():.4f}')
    print(f'  参考 零假设4 按果承载力加权     -> forest_frac = '
          f'{(canopy * fcap).sum() / fcap.sum():.4f}  '
          f'(fruit_capacity 只在 forest**2 上，是果层所在地)')

# 把实测 forest_frac 放到零假设刻度上
import json, os, statistics as st
DIR = 'outputs/20260803-repartition/verdict'
null_grass, null_fruit = 0.2190, 0.6123   # 上面刚打印的零假设3 / 零假设4
print('\n--- 实测 forest_frac 落在零假设刻度的哪里 ---')
print(f'  刻度: 按草承载力加权 = {null_grass:.4f}  ->  按果承载力加权 = {null_fruit:.4f}')
for arm in ['A_base', 'F_grb010_pm20', 'G_grb012_pm20']:
    v = []
    for s in range(6):
        for l in open(f'{DIR}/{arm}_seed{s}.log'):
            if l.startswith('JSON '):
                v.append(json.loads(l[5:])['forest_frac'])
    m = st.mean(v)
    print(f'  {arm:16s} 均值={m:.4f}  逐种子=' + ','.join(f'{x:.4f}' for x in v) +
          f'  距草零假设 {m-null_grass:+.4f}'
          f'  走完草->果全程的 {(m-null_grass)/(null_fruit-null_grass)*100:.1f}%')
