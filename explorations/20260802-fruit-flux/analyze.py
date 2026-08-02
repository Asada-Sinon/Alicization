"""R1b 果层供给扫描（流量口径）—— 6 配对种子正式判决的全部数字。

回答什么: docs/multispecies_program.md §6 预注册判据下，B_frb10_fe6 与 C_frb25_fe4
          是否各自通过（中位 frugivory_frac>=0.20 且 carnivore_frac>=0.05 且 population
          在同种子基线 [0.7,1.3]x 内 且 death_thirst_frac <= 同种子基线 +5pp）。
读哪些文件: outputs/20260802-fruit-flux/verdict/{A_base,B_frb10_fe6,C_frb25_fe4}_seed{0..5}.log
          每个 log 末尾一行 "JSON {...}"。provenance 见同目录 provenance.txt。
输出怎么读: 全部数字均由本脚本 stdout 打出，报告中不得手算。
          §0 overrides 归因核对 / §1 逐种子表 / §2 判据逐条 / §3 配对 Wilcoxon(B-A, C-A)
          / §4 B vs C / §5 异常与方差。
重跑: XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python explorations/20260802-fruit-flux/analyze.py
"""
import json, glob, os, statistics as st
import numpy as np
from scipy.stats import wilcoxon, rankdata

DIR = 'outputs/20260802-fruit-flux/verdict'
ARMS = ['A_base', 'B_frb10_fe6', 'C_frb25_fe4']
SEEDS = [0, 1, 2, 3, 4, 5]

# ---------- 载入：每个 log 恰好一行 "JSON " ----------
D, SRC = {}, {}
for arm in ARMS:
    for s in SEEDS:
        p = f'{DIR}/{arm}_seed{s}.log'
        js = [l for l in open(p) if l.startswith('JSON ')]
        assert len(js) == 1, f'{p}: 找到 {len(js)} 行 JSON，期望 1'
        r = json.loads(js[0][5:])
        assert r['seed'] == s, f'{p}: JSON seed={r["seed"]} != 文件名 seed{s}'
        D[(arm, s)] = r
        SRC[(arm, s)] = f'{os.path.basename(p)}:{sum(1 for _ in open(p))-1}'  # JSON 行号(1-based)
print('=' * 100)
print(f'载入 {len(D)} 个 run（{len(ARMS)} 臂 x {len(SEEDS)} 种子），每个文件恰好 1 行 JSON。'
      f' steps 集合={sorted({r["steps"] for r in D.values()})}')

# ---------- §0 overrides 归因核对 ----------
print('\n' + '=' * 100)
print('§0 overrides 归因核对（取自 --json 的 overrides key，不是命令行文本）')
for arm in ARMS:
    ovs = {json.dumps(D[(arm, s)]['overrides'], sort_keys=True) for s in SEEDS}
    print(f'  {arm:14s} 臂内 6 种子 overrides 一致={len(ovs)==1}  ->  {sorted(ovs)[0]}')
base_ov = D[('A_base', 0)]['overrides']
for arm in ARMS[1:]:
    ov = D[(arm, 0)]['overrides']
    keys = sorted(set(base_ov) | set(ov))
    diff = [(k, base_ov.get(k, '默认'), ov.get(k, '默认')) for k in keys
            if base_ov.get(k, '默认') != ov.get(k, '默认')]
    print(f'  {arm} vs A_base: {len(diff)} 处差异 -> ' +
          '; '.join(f'{k}: {a} -> {b}' for k, a, b in diff))
ovB, ovC = D[('B_frb10_fe6', 0)]['overrides'], D[('C_frb25_fe4', 0)]['overrides']
diffBC = [(k, ovB.get(k, '默认'), ovC.get(k, '默认')) for k in sorted(set(ovB) | set(ovC))
          if ovB.get(k, '默认') != ovC.get(k, '默认')]
print(f'  C vs B: {len(diffBC)} 处差异 -> ' + '; '.join(f'{k}: {a} -> {b}' for k, a, b in diffBC))

# ---------- §1 逐种子表 ----------
METRICS = ['frugivory_frac', 'graze_gain', 'fruit_gain', 'population', 'carnivore_frac',
           'late_carn', 'min_pop', 'death_thirst_frac', 'death_starvation_frac',
           'death_predation_frac', 'forest_frac', 'plant_total', 'fruit_total']
print('\n' + '=' * 100)
print('§1 逐种子全量表（每格出处 = <arm>_seed<N>.log 的 JSON 行 -> key）')
for k in METRICS:
    print(f'\n  -- {k} --')
    print('    ' + f'{"seed":>5s}' + ''.join(f'{a:>16s}' for a in ARMS) +
          f'{"B-A":>13s}{"C-A":>13s}')
    for s in SEEDS:
        v = [D[(a, s)][k] for a in ARMS]
        print('    ' + f'{s:>5d}' + ''.join(f'{x:>16.5g}' for x in v) +
              f'{v[1]-v[0]:>+13.5g}{v[2]-v[0]:>+13.5g}')
    med = [st.median([D[(a, s)][k] for s in SEEDS]) for a in ARMS]
    mean = [st.mean([D[(a, s)][k] for s in SEEDS]) for a in ARMS]
    sd = [st.stdev([D[(a, s)][k] for s in SEEDS]) for a in ARMS]
    print('    ' + f'{"中位":>4s} ' + ''.join(f'{x:>16.5g}' for x in med))
    print('    ' + f'{"均值":>4s} ' + ''.join(f'{x:>16.5g}' for x in mean))
    print('    ' + f'{"臂内SD":>3s} ' + ''.join(f'{x:>16.5g}' for x in sd))

# ---------- §2 预注册判据逐条 ----------
print('\n' + '=' * 100)
print('§2 预注册判据逐条判定（docs/multispecies_program.md:211-213，原文未改写）')
for arm in ARMS[1:]:
    print(f'\n  ##### 臂 {arm} #####')
    fr = [D[(arm, s)]['frugivory_frac'] for s in SEEDS]
    med_fr = st.median(fr)
    n_fr = sum(1 for x in fr if x >= 0.20)
    print(f'  [1] 中位 frugivory_frac >= 0.20 : 中位={med_fr:.5f} -> '
          f'{"达成" if med_fr >= 0.20 else "未达成"}   （逐种子 >=0.20 的有 {n_fr}/6）')
    print('      逐种子: ' + '  '.join(f's{s}={D[(arm,s)]["frugivory_frac"]:.4f}' for s in SEEDS))
    cf = [D[(arm, s)]['carnivore_frac'] for s in SEEDS]
    med_cf = st.median(cf)
    n_cf = sum(1 for x in cf if x >= 0.05)
    print(f'  [2] carnivore_frac >= 0.05 : 中位={med_cf:.5f} -> '
          f'{"达成" if med_cf >= 0.05 else "未达成"}   （逐种子 >=0.05 的有 {n_cf}/6）')
    print('      逐种子: ' + '  '.join(f's{s}={D[(arm,s)]["carnivore_frac"]:.4f}' for s in SEEDS))
    n_pop = 0
    print('  [3] population 在同种子 A_base 的 [0.7,1.3]x 内（逐种子配对）:')
    for s in SEEDS:
        a, b = D[('A_base', s)]['population'], D[(arm, s)]['population']
        ratio = b / a
        ok = 0.7 <= ratio <= 1.3
        n_pop += ok
        print(f'      seed{s}: A={a:.0f} {arm[0]}={b:.0f} 比值={ratio:.4f} '
              f'窗口=[{0.7*a:.0f},{1.3*a:.0f}] -> {"OK" if ok else "破"}')
    print(f'      -> {n_pop}/6 满足 => {"达成" if n_pop == 6 else "未达成(非全部种子)"}')
    n_th = 0
    print('  [4] death_thirst_frac <= 同种子 A_base + 5pp（逐种子配对）:')
    for s in SEEDS:
        a, b = D[('A_base', s)]['death_thirst_frac'], D[(arm, s)]['death_thirst_frac']
        d_pp = (b - a) * 100
        ok = d_pp <= 5.0
        n_th += ok
        print(f'      seed{s}: A={a*100:.2f}% {arm[0]}={b*100:.2f}% 差={d_pp:+.2f}pp -> '
              f'{"OK" if ok else "破"}')
    print(f'      -> {n_th}/6 满足 => {"达成" if n_th == 6 else "未达成(非全部种子)"}')
    allok = (med_fr >= 0.20) and (med_cf >= 0.05) and n_pop == 6 and n_th == 6
    print(f'  ===> {arm} 四条全满足(中位读法+护栏逐种子全过): {"通过" if allok else "不通过"}')

# ---------- 统计工具 ----------
def boot_ci_diff(diffs, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs, float)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def rank_biserial(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    nz = d[d != 0]
    if len(nz) == 0:
        return float('nan')
    r = rankdata(np.abs(nz))
    return float((r[nz > 0].sum() - r[nz < 0].sum()) / r.sum())

TEST_METRICS = ['frugivory_frac', 'population', 'carnivore_frac',
                'death_thirst_frac', 'forest_frac']

def compare(lab, hi, lo):
    print(f'\n  ##### {lab}（配对 Wilcoxon 符号秩，双侧，n=6；未做 Bonferroni）#####')
    print(f'  {"metric":22s}{lo+"均":>13s}{hi+"均":>13s}{"差(hi-lo)":>13s}'
          f'{"boot95%CI":>26s}{"W":>7s}{"p双侧":>9s}{"r_rb":>8s}{"同向":>7s}')
    for k in TEST_METRICS:
        a = [D[(lo, s)][k] for s in SEEDS]
        b = [D[(hi, s)][k] for s in SEEDS]
        d = [x - y for x, y in zip(b, a)]
        w, p = wilcoxon(b, a)
        ci = boot_ci_diff(d)
        r = rank_biserial(b, a)
        up = sum(1 for x in d if x > 0)
        same = max(up, 6 - up)
        print(f'  {k:22s}{st.mean(a):13.5f}{st.mean(b):13.5f}{st.mean(d):+13.5f}'
              f'  [{ci[0]:+.5f},{ci[1]:+.5f}]{w:7.1f}{p:9.4f}{r:+8.3f}{same:5d}/6')
    print(f'  n=6 配对符号秩双侧 p 的理论下限 = {2/2**6:.4f}（=2/64）'
          f'，全同向时必然取到；0.031 是地板不是强证据。')

print('\n' + '=' * 100)
print('§3 与配对基线 A_base 的检验')
compare('B_frb10_fe6 vs A_base', 'B_frb10_fe6', 'A_base')
compare('C_frb25_fe4 vs A_base', 'C_frb25_fe4', 'A_base')
print('\n' + '=' * 100)
print('§4 B vs C 直接配对比较（同一 6 配对种子）')
compare('C_frb25_fe4 vs B_frb10_fe6', 'C_frb25_fe4', 'B_frb10_fe6')
print('\n  两臂各自的护栏余量（逐种子对 A_base）:')
for arm in ARMS[1:]:
    pr = [D[(arm, s)]['population'] / D[('A_base', s)]['population'] for s in SEEDS]
    th = [(D[(arm, s)]['death_thirst_frac'] - D[('A_base', s)]['death_thirst_frac']) * 100
          for s in SEEDS]
    print(f'    {arm:14s} pop比值 min={min(pr):.4f} max={max(pr):.4f}（窗口0.70~1.30，'
          f'最紧余量={min(min(pr)-0.7, 1.3-max(pr)):.4f}）  '
          f'渴死差 max={max(th):+.2f}pp（上限+5.00pp，余量={5.0-max(th):+.2f}pp）')

# ---------- §5 异常与方差 ----------
print('\n' + '=' * 100)
print('§5 异常、反例与臂内方差')
print('  (a) min_pop 逐种子（n_max=16384；崩溃判据 min_pop 接近 0）:')
for arm in ARMS:
    v = [D[(arm, s)]['min_pop'] for s in SEEDS]
    print(f'    {arm:14s} ' + '  '.join(f's{s}={x:.0f}' for s, x in zip(SEEDS, v)) +
          f'   最小={min(v):.0f}')
print('  (b) 逐种子反向（相对该臂自身的均值方向）:')
for lab, hi, lo in [('B-A', 'B_frb10_fe6', 'A_base'), ('C-A', 'C_frb25_fe4', 'A_base'),
                    ('C-B', 'C_frb25_fe4', 'B_frb10_fe6')]:
    for k in METRICS:
        d = [D[(hi, s)][k] - D[(lo, s)][k] for s in SEEDS]
        mean_d = st.mean(d)
        rev = [s for s, x in zip(SEEDS, d) if (x > 0) != (mean_d > 0) and x != 0]
        if rev:
            print(f'    {lab} {k:22s} 均值差={mean_d:+.5g} 但反向种子={rev} '
                  f'(值 ' + ', '.join(f's{s}:{D[(hi,s)][k]-D[(lo,s)][k]:+.5g}' for s in rev) + ')')
print('  (c) 臂内变异系数 CV=SD/|均值|（>0.20 标注为方差大）:')
for k in METRICS:
    line = []
    for arm in ARMS:
        v = [D[(arm, s)][k] for s in SEEDS]
        m, sd = st.mean(v), st.stdev(v)
        cv = sd / abs(m) if m != 0 else float('nan')
        line.append(f'{arm[0]}:CV={cv:.3f}{"*" if cv > 0.20 else " "}')
    print(f'    {k:22s} ' + '  '.join(line))
print('  (d) 配对差值 vs 该指标的臂内种子间 SD（差值小于 SD 者标注）:')
for lab, hi, lo in [('B-A', 'B_frb10_fe6', 'A_base'), ('C-A', 'C_frb25_fe4', 'A_base'),
                    ('C-B', 'C_frb25_fe4', 'B_frb10_fe6')]:
    for k in TEST_METRICS:
        d = st.mean([D[(hi, s)][k] - D[(lo, s)][k] for s in SEEDS])
        sd_lo = st.stdev([D[(lo, s)][k] for s in SEEDS])
        sd_hi = st.stdev([D[(hi, s)][k] for s in SEEDS])
        flag = '  <-- 差值 < 两臂SD' if abs(d) < max(sd_lo, sd_hi) else ''
        print(f'    {lab} {k:22s} 差={d:+.5f}  SD({lo[0]})={sd_lo:.5f} SD({hi[0]})={sd_hi:.5f}{flag}')
print('  (e) 果层供给是否被吃满: fruit_total（现存量）与 fruit_gain（流量）:')
for arm in ARMS:
    ft = [D[(arm, s)]['fruit_total'] for s in SEEDS]
    fg = [D[(arm, s)]['fruit_gain'] for s in SEEDS]
    print(f'    {arm:14s} fruit_total 均={st.mean(ft):9.3f} SD={st.stdev(ft):8.3f}   '
          f'fruit_gain 均={st.mean(fg):8.3f} SD={st.stdev(fg):7.3f}')
print('\n完。所有数字见上；报告中任何数字都应能在本 stdout 中原样找到。')

# ---------- §6 判据读法歧义：护栏按「中位」读 vs 按「逐种子全过」读 ----------
print('\n' + '=' * 100)
print('§6 判据读法歧义（原文「中位 frugivory_frac>=0.20 且 carnivore_frac>=0.05 且 population')
print('   落在同种子基线[0.7,1.3]倍内 且 death_thirst_frac 不高于基线+5pp」——「中位」是否')
print('   分配到后三条，原文未明说。两种读法都算出来，不替判据做选择。）')
for arm in ARMS[1:]:
    print(f'\n  ##### {arm} #####')
    pr = [D[(arm, s)]['population'] / D[('A_base', s)]['population'] for s in SEEDS]
    th = [(D[(arm, s)]['death_thirst_frac'] - D[('A_base', s)]['death_thirst_frac']) * 100
          for s in SEEDS]
    fr = [D[(arm, s)]['frugivory_frac'] for s in SEEDS]
    cf = [D[(arm, s)]['carnivore_frac'] for s in SEEDS]
    print(f'    读法甲（护栏取中位）: pop比值中位={st.median(pr):.4f} '
          f'{"在" if 0.7 <= st.median(pr) <= 1.3 else "不在"}[0.7,1.3]; '
          f'渴死差中位={st.median(th):+.2f}pp {"<=" if st.median(th) <= 5 else ">"}+5pp')
    ok_a = (st.median(fr) >= 0.20 and st.median(cf) >= 0.05
            and 0.7 <= st.median(pr) <= 1.3 and st.median(th) <= 5.0)
    print(f'    读法甲结论: {"通过" if ok_a else "不通过"}')
    n_pop = sum(1 for x in pr if 0.7 <= x <= 1.3)
    n_th = sum(1 for x in th if x <= 5.0)
    print(f'    读法乙（护栏逐种子全过）: pop {n_pop}/6, 渴死 {n_th}/6 -> '
          f'{"通过" if (st.median(fr) >= 0.20 and st.median(cf) >= 0.05 and n_pop == 6 and n_th == 6) else "不通过"}')
    joint = sum(1 for i, s in enumerate(SEEDS)
                if fr[i] >= 0.20 and cf[i] >= 0.05 and 0.7 <= pr[i] <= 1.3 and th[i] <= 5.0)
    print(f'    参考：单个种子同时满足全部四条的有 {joint}/6 个 -> ' +
          ', '.join(f's{s}={"OK" if (fr[i]>=0.20 and cf[i]>=0.05 and 0.7<=pr[i]<=1.3 and th[i]<=5.0) else "x"}'
                    for i, s in enumerate(SEEDS)))
