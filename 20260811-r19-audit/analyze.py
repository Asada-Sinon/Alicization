#!/usr/bin/env python
"""R19 判决复核主脚本。

回答:
 1. 判决表(feasibility §25.1)的四臂 Δ 与三个对比能否复现(exp_stats 口径);
 2. C 臂的涨是「系统性偏置」还是「漂变运气」——逐 run 符号 + one_sample;
 3. 「偏差随时间累积」是否成立——正式 run 自己在 gen≈40 时 C 臂的值
    (探针3 的 C=-0.007 只有 4 run,SD 0.37,几乎无信息);
 4. mf_above(gene>0.847 占比)四臂 + 臂间对比——判决没讨论的读数;
 5. overrides_diff:每个对比差几个变量。

读:outputs/20260807-r19/{A,B,C,D}_s{0..11}_r{1,2}.log 的 JSON 行。
Δ 口径与判决一致:traj[-1].mf_mean - traj[0].mf_mean(t=102000 减 t=250)。
"""
import sys
sys.path.insert(0, '/home/xrl/intern/Alicization/scripts')
import numpy as np
from exp_stats import RunSet, paired, one_sample

RUN = '/home/xrl/intern/Alicization/outputs/20260807-r19'
rs = RunSet.load(RUN, arms=['A', 'B', 'C', 'D'], seeds=range(12), reps=(1, 2))
assert not rs.problems, rs.problems

def d_mf(rec):
    return rec['traj'][-1]['mf_mean'] - rec['traj'][0]['mf_mean']

def d_above(rec):
    return rec['traj'][-1]['mf_above'] - rec['traj'][0]['mf_above']

def final_above(rec):
    return rec['traj'][-1]['mf_above']

print('=== overrides 差异(全臂):', rs.overrides_diff())
for a in 'ABCD':
    for b in 'ABCD':
        if a < b:
            recs = {k: v for k, v in rs.records.items() if k[0] in (a, b)}
            sub = RunSet(recs, rs.sources, [a, b], rs.seeds, rs.reps, [])
            print(f'  {a} vs {b}: {sub.overrides_diff()}')

print('\n=== 1. 四臂 Δ mean gene(one_sample vs 0,12 格均值)===')
for arm in 'ABCD':
    r = one_sample(rs, d_mf, arm, mu=0.0, metric_name=f'd_mf[{arm}]')
    print(r.format())
    raw = rs.raw(arm, d_mf)
    print(f'  逐 run(24 个): >0 的有 {int((raw>0).sum())}/24, '
          f'min={raw.min():+.4f}, max={raw.max():+.4f}\n')

print('=== 2. 臂间对比(paired,noise 用全臂自估)===')
for a, b, label in [('A','C','总效应'), ('A','B','收益净效应'), ('B','C','搭车')]:
    r = paired(rs, d_mf, a, b, label=f'{a}-{b} {label}')
    print(r.format()); print()

print('=== 3. mf_above:Δ 与末值 ===')
for arm in 'ABCD':
    r = one_sample(rs, d_above, arm, mu=0.0, metric_name=f'd_above[{arm}]')
    fin = rs.cell_means(arm, final_above)
    print(f'{arm}: Δabove 均值={r.diff.mean():+.4f}  CI[{r.ci[0]:+.4f},{r.ci[1]:+.4f}] '
          f'p={r.p:.4f}  末 above 均值={fin.mean():.4f}')
for a, b in [('A','C'), ('A','B'), ('B','C')]:
    r = paired(rs, d_above, a, b)
    print(f'{a}-{b} Δabove: {r.diff.mean():+.4f}  CI[{r.ci[0]:+.4f},{r.ci[1]:+.4f}] '
          f'p={r.p:.4f}  同向 {r.n_pos}/{len(r.diff)}  效应/噪声={r.ratio:+.2f}')

print('\n=== 4. 按世代对齐的时间曲线(检验「累积」)===')
# 每个 run:在 gen 网格上线性插值 mf_mean(相对 traj[0]),再对 24 run 平均
gens_grid = [20, 40, 60, 80, 100, 120, 140, 150]
for arm in 'ABC':
    rows = []
    for s in rs.seeds:
        for rep in rs.reps:
            tr = rs.records[(arm, s, rep)]['traj']
            g = np.array([p['generation'] for p in tr]) - tr[0]['generation']
            m = np.array([p['mf_mean'] for p in tr]) - tr[0]['mf_mean']
            rows.append(np.interp(gens_grid, g, m))
    rows = np.array(rows)
    mean, se = rows.mean(0), rows.std(0, ddof=1) / np.sqrt(len(rows))
    print(f'{arm}: ' + '  '.join(f'g{gg}={mm:+.3f}±{ss:.3f}'
                                 for gg, mm, ss in zip(gens_grid, mean, se)))
    if arm == 'C':
        at40 = rows[:, 1]
        print(f'   C 在 gen40: 均值 {at40.mean():+.4f}, >0 的 run {int((at40>0).sum())}/24, '
              f'SD {at40.std(ddof=1):.4f}')

print('\n=== 5. 探针3 的 C 臂散布(4 run,复核「40 代时 C=-0.007」)===')
import json, glob
vals = []
for f in sorted(glob.glob('/home/xrl/intern/Alicization/outputs/20260807-r19probe3/C_s*.log')):
    for ln in open(f):
        if ln.startswith('JSON '):
            r = json.loads(ln[5:])
            vals.append(r['traj'][-1]['mf_mean'] - r['traj'][0]['mf_mean'])
vals = np.array(vals)
print(f'C 探针3 逐 run Δ: {np.array2string(vals, precision=4, sign="+")}')
print(f'均值 {vals.mean():+.4f}, SD {vals.std(ddof=1):.4f}, SE {vals.std(ddof=1)/2:.4f}')
