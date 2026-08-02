"""R3 「等量再分配」6 配对种子正式判决的全部数字。

回答什么: docs/multispecies_program.md §11.2 (docs/multispecies_program.md:579-584) 的
          预注册判据下, F_grb010_pm20 / G_grb012_pm20 是否各自通过 ——
          中位 frugivory_frac>=0.20
          且 总植物通量 (graze_gain+fruit_gain) 在同种子 A_base [0.90,1.10]x  6/6
          且 carnivore_frac>=0.05  6/6
          且 population 在同种子 A_base [0.7,1.3]x  6/6
          且 death_thirst_frac <= 同种子 A_base +5pp  6/6。
          护栏一律逐种子, 不做中位读法, 不放宽。
          核心机制问题(§4): 前三回合的失败机制是「种群超调驱动渴死」(R1c Spearman ρ=+0.90,
          docs/multispecies_program.md:458)。本轮既然守恒了总生产力, 超调应当消失 ——
          直接用同一公式在本轮数据上重算 ρ, 并在 R1c 原始 log 上用同一公式重算作对照。
读哪些文件: outputs/20260803-repartition/verdict/{A_base,F_grb010_pm20,G_grb012_pm20}_seed{0..5}.log
          每个 log 末尾恰好一行 "JSON {...}"; 表格行用于 late-window 口径稳健性核对。
          对照: outputs/20260803-fruit-water/{A_base,C_fwf10,D_fwf40,E_fwf60}_seed{0..5}.log (R1c)
          provenance: outputs/20260803-repartition/verdict/provenance.txt
输出怎么读: 全部数字均由本脚本 stdout 打出, 报告中不得手算。
          §0 载入/归因核对 / §1 逐种子全量表 / §2 判据逐条 / §3 配对 Wilcoxon
          / §4 种群超调-渴死耦合 (本轮最重要的数) / §5 预标风险(捕食者)
          / §6 异常、反例、方差、口径稳健性 / §7 一句话判决
重跑: XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python explorations/20260803-repartition/analyze.py
"""
import json, os, re, statistics as st
import numpy as np
from scipy.stats import wilcoxon, rankdata, spearmanr

DIR = 'outputs/20260803-repartition/verdict'
ARMS = ['A_base', 'F_grb010_pm20', 'G_grb012_pm20']
TREAT = ['F_grb010_pm20', 'G_grb012_pm20']
BASE = 'A_base'
SEEDS = [0, 1, 2, 3, 4, 5]

# ---------- §0 载入 ----------
def load(d, arms, seeds):
    D, SRC, TAB, bad = {}, {}, {}, []
    for arm in arms:
        for s in seeds:
            p = f'{d}/{arm}_seed{s}.log'
            if not os.path.exists(p):
                bad.append(f'{p}: 文件不存在'); continue
            lines = open(p).read().splitlines()
            js = [(i + 1, l) for i, l in enumerate(lines) if l.startswith('JSON ')]
            if len(js) != 1:
                bad.append(f'{p}: 找到 {len(js)} 行 JSON'); continue
            ln, l = js[0]
            r = json.loads(l[5:])
            if r['seed'] != s:
                bad.append(f'{p}: JSON seed={r["seed"]} != 文件名 seed{s}'); continue
            if 'collapsed to zero' in '\n'.join(lines):
                bad.append(f'{p}: 出现 population collapsed to zero')
            D[(arm, s)] = r
            SRC[(arm, s)] = f'{os.path.basename(p)}:{ln}'
            # 表格行: step pop energy water age plant diet dietSD carn%
            rows = []
            for L in lines:
                m = re.match(r'^\s*(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+'
                             r'(\d+)\s+([\d.-]+)\s+([\d.]+)\s+([\d.]+)%', L)
                if m:
                    rows.append((int(m.group(1)), float(m.group(2)),
                                 float(m.group(6)), float(m.group(9)) / 100.0))
            TAB[(arm, s)] = rows
    return D, SRC, TAB, bad

D, SRC, TAB, bad = load(DIR, ARMS, SEEDS)
print('=' * 118)
print(f'§0 载入 {len(D)}/{len(ARMS)*len(SEEDS)} 个 run。缺失/异常/灭绝: {bad if bad else "无"}')
print(f'   steps 集合={sorted({r["steps"] for r in D.values()})} '
      f'(全一致={len({r["steps"] for r in D.values()})==1})')
print('   JSON 行号: ' + '; '.join(
    f'{a}_seed*=' + ','.join(sorted({SRC[(a, s)].split(":")[1] for s in SEEDS})) for a in ARMS))
print(f'   表格行数(每 run 应为 40 = 20000/500): '
      f'{sorted({len(TAB[(a,s)]) for a in ARMS for s in SEEDS})}')
print('   overrides 归因核对（取自 --json 的 overrides key，不是命令行文本）:')
for arm in ARMS:
    ovs = {json.dumps(D[(arm, s)]['overrides'], sort_keys=True) for s in SEEDS}
    print(f'     {arm:15s} 臂内 6 种子 overrides 一致={len(ovs) == 1}  ->  {sorted(ovs)[0]}')

def ovdiff(lo, hi):
    a, b = D[(lo, 0)]['overrides'], D[(hi, 0)]['overrides']
    ks = sorted(set(a) | set(b))
    return [(k, a.get(k, '默认'), b.get(k, '默认')) for k in ks if a.get(k) != b.get(k)]

print('   臂间 config 差异项数（>1 项则不可归因到单一变量）:')
for lo, hi in [(BASE, 'F_grb010_pm20'), (BASE, 'G_grb012_pm20'),
               ('G_grb012_pm20', 'F_grb010_pm20')]:
    dd = ovdiff(lo, hi)
    print(f'     {hi} vs {lo}: {len(dd)} 处 -> ' +
          '; '.join(f'{k}: {x} -> {y}' for k, x, y in dd))

# ---------- 派生量 ----------
def flux(arm, s):
    r = D[(arm, s)]
    return r['graze_gain'] + r['fruit_gain']

for arm in ARMS:
    for s in SEEDS:
        D[(arm, s)]['total_flux'] = flux(arm, s)
        D[(arm, s)]['flux_ratio'] = flux(arm, s) / flux(BASE, s)
        D[(arm, s)]['pop_ratio'] = D[(arm, s)]['population'] / D[(BASE, s)]['population']
        D[(arm, s)]['thirst_pp'] = 100.0 * (D[(arm, s)]['death_thirst_frac']
                                            - D[(BASE, s)]['death_thirst_frac'])

# ---------- §1 逐种子全量表 ----------
print('\n' + '=' * 118)
print('§1 逐种子全量表。出处: outputs/20260803-repartition/verdict/<arm>_seed<N>.log 的 "JSON " 行。')
print('   注意口径: graze_gain/fruit_gain/frugivory_frac/population/carnivore_frac/plant_total/')
print('   forest_frac 取自 run_headless.py:58 —— 末个 chunk 的**最后一步瞬时值**（单步快照），')
print('   非全程均值; death_*_frac 是全程累计; late_carn/min_pop 是 chunk 边界聚合。')
COLS = ['graze_gain', 'fruit_gain', 'total_flux', 'flux_ratio', 'frugivory_frac',
        'population', 'pop_ratio', 'carnivore_frac', 'late_carn', 'min_pop',
        'death_thirst_frac', 'thirst_pp', 'death_starvation_frac',
        'death_predation_frac', 'forest_frac', 'plant_total']
for arm in ARMS:
    print(f'\n  --- {arm} ---')
    print('  ' + f'{"seed":>4}' + ''.join(f'{c[:13]:>14s}' for c in COLS))
    for s in SEEDS:
        r = D[(arm, s)]
        print('  ' + f'{s:>4}' + ''.join(f'{r[c]:14.4f}' for c in COLS))
    print('  ' + f'{"均值":>3}' + ''.join(
        f'{st.mean([D[(arm,s)][c] for s in SEEDS]):14.4f}' for c in COLS))
    print('  ' + f'{"中位":>3}' + ''.join(
        f'{st.median([D[(arm,s)][c] for s in SEEDS]):14.4f}' for c in COLS))
    print('  ' + f'{"SD":>4}' + ''.join(
        f'{st.stdev([D[(arm,s)][c] for s in SEEDS]):14.4f}' for c in COLS))

# ---------- §2 预注册判据 ----------
print('\n' + '=' * 118)
print('§2 预注册判据逐条判定（docs/multispecies_program.md:579-584，护栏一律逐种子 6/6，不放宽）')
verdict = {}
for arm in TREAT:
    print(f'\n  ##### {arm} #####')
    ok = {}
    med = st.median([D[(arm, s)]['frugivory_frac'] for s in SEEDS])
    ok['C1 中位 frugivory_frac>=0.20'] = (med >= 0.20, f'中位={med:.4f}',
                                          sum(1 for s in SEEDS if D[(arm, s)]['frugivory_frac'] >= 0.20))
    n2 = [s for s in SEEDS if 0.90 <= D[(arm, s)]['flux_ratio'] <= 1.10]
    ok['C2 总通量 in [0.90,1.10]x 6/6'] = (len(n2) == 6,
        '逐种子比值=' + ','.join(f'{D[(arm,s)]["flux_ratio"]:.3f}' for s in SEEDS), len(n2))
    n3 = [s for s in SEEDS if D[(arm, s)]['carnivore_frac'] >= 0.05]
    ok['C3 carnivore_frac>=0.05 6/6'] = (len(n3) == 6,
        '逐种子=' + ','.join(f'{D[(arm,s)]["carnivore_frac"]:.3f}' for s in SEEDS), len(n3))
    n4 = [s for s in SEEDS if 0.7 <= D[(arm, s)]['pop_ratio'] <= 1.3]
    ok['C4 population in [0.7,1.3]x 6/6'] = (len(n4) == 6,
        '逐种子比值=' + ','.join(f'{D[(arm,s)]["pop_ratio"]:.3f}' for s in SEEDS), len(n4))
    n5 = [s for s in SEEDS if D[(arm, s)]['thirst_pp'] <= 5.0]
    ok['C5 death_thirst<=base+5pp 6/6'] = (len(n5) == 6,
        '逐种子 pp=' + ','.join(f'{D[(arm,s)]["thirst_pp"]:+.2f}' for s in SEEDS), len(n5))
    for k, (passed, detail, cnt) in ok.items():
        print(f'    {k:34s} {"达成" if passed else "未达成"}  {cnt}/6   {detail}')
    allp = all(v[0] for v in ok.values())
    verdict[arm] = allp
    print(f'    => {arm}: {"五条全过 -> 通过" if allp else "有条未达成 -> 不通过"}')
    if not allp:
        print(f'       破的条: ' + '; '.join(k for k, v in ok.items() if not v[0]))

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

TEST_METRICS = ['total_flux', 'frugivory_frac', 'population', 'carnivore_frac',
                'death_thirst_frac', 'forest_frac']

def compare(hi, lo):
    print(f'\n  ##### {hi} − {lo}（配对 Wilcoxon 符号秩，双侧，n=6；未做 Bonferroni） #####')
    print(f'  {"metric":20s}{lo[:9]+"均":>13s}{hi[:9]+"均":>13s}{"差(hi-lo)":>13s}'
          f'{"boot95%CI":>28s}{"W":>7s}{"p双侧":>9s}{"r_rb":>8s}{"同向":>8s}')
    for k in TEST_METRICS:
        a = [D[(lo, s)][k] for s in SEEDS]
        b = [D[(hi, s)][k] for s in SEEDS]
        d = [x - y for x, y in zip(b, a)]
        w, p = wilcoxon(b, a)
        ci = boot_ci_diff(d); r = rank_biserial(b, a)
        up = sum(1 for x in d if x > 0)
        print(f'  {k:20s}{st.mean(a):13.5f}{st.mean(b):13.5f}{st.mean(d):+13.5f}'
              f'  [{ci[0]:+.5f},{ci[1]:+.5f}]{w:7.1f}{p:9.4f}{r:+8.3f}'
              f'{max(up, 6-up):6d}/6')

print('\n' + '=' * 118)
print(f'§3 配对 Wilcoxon。n=6 配对符号秩双侧 p 的理论下限 = {2/2**6:.4f} (=2/64)，'
      '全同向时必然取到；0.031 是地板不是强证据。')
for hi, lo in [('F_grb010_pm20', BASE), ('G_grb012_pm20', BASE),
               ('F_grb010_pm20', 'G_grb012_pm20')]:
    compare(hi, lo)

# ---------- §4 种群超调 × 渴死（本轮最重要的数） ----------
print('\n' + '=' * 118)
print('§4 关键交叉检验: 守恒后「种群超调驱动渴死」是否消失')
print('   R1c 记载值 Spearman ρ=+0.9030 (n=18 配对点), docs/multispecies_program.md:458 —— 二手。')
print('   下面用同一公式在两批原始 log 上分别重算，保证同口径。')
print('   公式: x = pop_treat/pop_base（同种子）, y = 100*(thirst_frac_treat − thirst_frac_base) pp')

def coupling(D_, treats, seeds, label):
    xs, ys = [], []
    for arm in treats:
        for s in seeds:
            xs.append(D_[(arm, s)]['population'] / D_[(BASE, s)]['population'])
            ys.append(100.0 * (D_[(arm, s)]['death_thirst_frac']
                               - D_[(BASE, s)]['death_thirst_frac']))
    rho, p = spearmanr(xs, ys)
    pr, pp_ = np.corrcoef(xs, ys)[0, 1], None
    print(f'   {label:46s} n={len(xs):2d}  Spearman ρ={rho:+.4f}  p={p:.4f}  '
          f'Pearson r={pr:+.4f}')
    return rho, p, xs, ys

print('\n  (a) 本轮 (R3) pop 比基线的分布:')
print(f'  {"arm":16s}{"min":>9s}{"中位":>9s}{"均值":>9s}{"max":>9s}{">1.0 的种子数":>15s}'
      f'{"逐种子":>44s}')
for arm in TREAT:
    v = [D[(arm, s)]['pop_ratio'] for s in SEEDS]
    print(f'  {arm:16s}{min(v):9.3f}{st.median(v):9.3f}{st.mean(v):9.3f}{max(v):9.3f}'
          f'{sum(1 for x in v if x > 1.0):15d}    ' + ','.join(f'{x:.3f}' for x in v))
print('  R1c 对照（同一 pop_ratio 公式，原始 log outputs/20260803-fruit-water/）:')
R, RSRC, _, rbad = load('outputs/20260803-fruit-water',
                        ['A_base', 'C_fwf10', 'D_fwf40', 'E_fwf60'], SEEDS)
print(f'    R1c 载入 {len(R)}/24, 异常={rbad if rbad else "无"}')
for arm in ['C_fwf10', 'D_fwf40', 'E_fwf60']:
    v = [R[(arm, s)]['population'] / R[('A_base', s)]['population'] for s in SEEDS]
    print(f'  {arm:16s}{min(v):9.3f}{st.median(v):9.3f}{st.mean(v):9.3f}{max(v):9.3f}'
          f'{sum(1 for x in v if x > 1.0):15d}    ' + ','.join(f'{x:.3f}' for x in v))

print('\n  (b)(c) 耦合强度 Spearman ρ:')
r3_rho, r3_p, r3x, r3y = coupling(D, TREAT, SEEDS, 'R3 本轮 (F+G, 12 配对点，非独立)')
for arm in TREAT:
    coupling(D, [arm], SEEDS, f'R3 仅 {arm} (6 点)')
r1_rho, r1_p, r1x, r1y = coupling(R, ['C_fwf10', 'D_fwf40', 'E_fwf60'], SEEDS,
                                  'R1c 重算 (C+D+E, 18 配对点，非独立)')
coupling(R, ['D_fwf40', 'E_fwf60'], SEEDS, 'R1c 重算 仅 D+E (12 点)')
print(f'   => R3 − R1c 的 ρ 差 = {r3_rho - r1_rho:+.4f}')
print(f'   R3 pop_ratio 离 1.0 的平均绝对偏离 = '
      f'{np.mean([abs(x-1) for x in r3x]):.4f}; R1c = {np.mean([abs(x-1) for x in r1x]):.4f}')
print(f'   R3 pop_ratio 均值 = {np.mean(r3x):.4f}; R1c = {np.mean(r1x):.4f}')
print(f'   R3 thirst pp 均值 = {np.mean(r3y):+.4f}; R1c = {np.mean(r1y):+.4f}')

# ---------- §5 预标风险: 捕食者 ----------
print('\n' + '=' * 118)
print('§5 预标风险: 砍草是否压死捕食者（provenance.txt:19-21 预标）')
print(f'  {"arm":16s}{"seed":>6s}{"carnivore_frac":>16s}{"late_carn":>11s}'
      f'{"对基线 carn 差":>15s}{"对基线 late 差":>15s}')
for arm in ARMS:
    for s in SEEDS:
        r = D[(arm, s)]
        print(f'  {arm:16s}{s:6d}{r["carnivore_frac"]:16.4f}{r["late_carn"]:11.4f}'
              f'{r["carnivore_frac"]-D[(BASE,s)]["carnivore_frac"]:+15.4f}'
              f'{r["late_carn"]-D[(BASE,s)]["late_carn"]:+15.4f}')
for arm in ARMS:
    c = [D[(arm, s)]['carnivore_frac'] for s in SEEDS]
    lc = [D[(arm, s)]['late_carn'] for s in SEEDS]
    print(f'  {arm:16s} carn 均={st.mean(c):.4f} 中位={st.median(c):.4f} min={min(c):.4f} '
          f'max={max(c):.4f} | late_carn 均={st.mean(lc):.4f} min={min(lc):.4f} '
          f'max={max(lc):.4f} | late<0.05 的种子={[s for s in SEEDS if D[(arm,s)]["late_carn"]<0.05]}')

# ---------- §6 异常、反例、方差、口径稳健性 ----------
print('\n' + '=' * 118)
print('§6 异常、反例、臂内方差、口径稳健性')
print('  (i) min_pop（崩溃距离）与终末 population:')
for arm in ARMS:
    mp = [D[(arm, s)]['min_pop'] for s in SEEDS]
    print(f'  {arm:16s} min_pop 逐种子=' + ','.join(f'{x:.0f}' for x in mp) +
          f'  最小={min(mp):.0f}  <200 的种子={[s for s in SEEDS if D[(arm,s)]["min_pop"]<200]}')
print('\n  (ii) 臂内变异系数 CV = SD/均值（臂内方差是否异常放大）:')
CVK = ['total_flux', 'frugivory_frac', 'population', 'carnivore_frac',
       'death_thirst_frac', 'forest_frac', 'plant_total']
print('  ' + f'{"arm":16s}' + ''.join(f'{k[:13]:>15s}' for k in CVK))
for arm in ARMS:
    print('  ' + f'{arm:16s}' + ''.join(
        f'{st.stdev([D[(arm,s)][k] for s in SEEDS])/abs(st.mean([D[(arm,s)][k] for s in SEEDS])):15.4f}'
        for k in CVK))
print('\n  (iii) 差值 vs 基线臂的种子间 SD（差值是否小于噪声）:')
print(f'  {"metric":20s}{"A 臂 SD":>12s}{"F−A 均差":>12s}{"|差|/SD":>10s}'
      f'{"G−A 均差":>12s}{"|差|/SD":>10s}')
for k in TEST_METRICS:
    sd = st.stdev([D[(BASE, s)][k] for s in SEEDS])
    row = f'  {k:20s}{sd:12.5f}'
    for arm in TREAT:
        d = st.mean([D[(arm, s)][k] - D[(BASE, s)][k] for s in SEEDS])
        row += f'{d:+12.5f}{abs(d)/sd if sd else float("nan"):10.2f}'
    print(row)
print('\n  (iv) 逐种子反向核对（每个指标 hi−lo 的符号分布）:')
for hi in TREAT:
    for k in TEST_METRICS:
        d = [D[(hi, s)][k] - D[(BASE, s)][k] for s in SEEDS]
        sg = ''.join('+' if x > 0 else ('-' if x < 0 else '0') for x in d)
        print(f'    {hi:16s}{k:20s} 符号={sg}  同向={max(sg.count("+"), sg.count("-"))}/6'
              f'  反向种子={[SEEDS[i] for i,x in enumerate(d) if (x>0)!=(st.mean(d)>0)]}')
print('\n  (v) 口径稳健性: 终末单步快照 vs 末 5000 步(后 10 个 chunk)均值')
print(f'  {"arm":16s}{"seed":>5s}{"pop 终末":>10s}{"pop 末10均":>11s}{"比值":>8s}'
      f'{"carn 终末":>11s}{"carn 末10均":>12s}')
lw = {}
for arm in ARMS:
    for s in SEEDS:
        rows = TAB[(arm, s)][-10:]
        lw[(arm, s)] = (float(np.mean([r[1] for r in rows])),
                        float(np.mean([r[3] for r in rows])))
for arm in ARMS:
    for s in SEEDS:
        p10, c10 = lw[(arm, s)]
        r = D[(arm, s)]
        print(f'  {arm:16s}{s:5d}{r["population"]:10.0f}{p10:11.1f}'
              f'{r["population"]/p10:8.3f}{r["carnivore_frac"]:11.4f}{c10:12.4f}')
print('  末 10 chunk 均值口径下的 pop 比基线（护栏 C4 的稳健性重算，非预注册口径）:')
for arm in TREAT:
    v = [lw[(arm, s)][0] / lw[(BASE, s)][0] for s in SEEDS]
    print(f'    {arm:16s} ' + ','.join(f'{x:.3f}' for x in v) +
          f'   in[0.7,1.3] = {sum(1 for x in v if 0.7<=x<=1.3)}/6')
print('  末 10 chunk 均值口径下 carnivore_frac >= 0.05（护栏 C3 稳健性重算，非预注册口径）:')
for arm in ARMS:
    v = [lw[(arm, s)][1] for s in SEEDS]
    print(f'    {arm:16s} ' + ','.join(f'{x:.4f}' for x in v) +
          f'   >=0.05 = {sum(1 for x in v if x >= 0.05)}/6')

# ---------- §7 判决 ----------
print('\n' + '=' * 118)
print('§7 一句话判决')
for arm in TREAT:
    print(f'  {arm}: {"通过（五条 6/6 全过）" if verdict[arm] else "不通过"}')
print(f'  §11.2 失败判据（无臂全过 -> 资源轴整条关闭）触发 = {not any(verdict.values())}')

# ---------- §8 复现噪声: 探针 vs 判决 同种子同配置同代码 ----------
# 探针 outputs/20260803-repartition/probe/ 跑在 git 3e4f147 (dirty:true)，判决跑在 2f1e032；
# `git diff --stat 3e4f147..2f1e032 -- underworld/ scripts/` 为空 => 代码一致。
# 因此 t0 vs A_base_seed0、t5 vs F_grb010_pm20_seed0 是**同种子同配置的复现对**，
# 差异即 20000 步尺度上的 GPU 原子重排混沌放大（CLAUDE.md: 确定性只在短程成立）。
print('\n' + '=' * 118)
print('§8 复现噪声实测: 探针(probe) vs 判决(verdict) 的同种子同配置复现对')
print('   代码一致性: git diff 3e4f147..2f1e032 -- underworld/ scripts/ 为空（仅 docs 变动）')
PAIRS = [('probe/t0_A_base.log', ('A_base', 0)),
         ('probe/t5_grb010_pm20.log', ('F_grb010_pm20', 0))]
print(f'  {"复现对":42s}{"指标":20s}{"probe":>12s}{"verdict":>12s}{"相对差":>10s}')
for pf, key in PAIRS:
    p = f'outputs/20260803-repartition/{pf}'
    lines = open(p).read().splitlines()
    js = [(i + 1, l) for i, l in enumerate(lines) if l.startswith('JSON ')]
    r = json.loads(js[0][1][5:])
    r['total_flux'] = r['graze_gain'] + r['fruit_gain']
    v = D[key]
    for k in ['total_flux', 'graze_gain', 'fruit_gain', 'frugivory_frac', 'population',
              'carnivore_frac', 'death_thirst_frac', 'plant_total']:
        print(f'  {pf+" vs "+key[0]+"_seed0":42s}{k:20s}{r[k]:12.4f}{v[k]:12.4f}'
              f'{(v[k]-r[k])/abs(r[k])*100 if r[k] else float("nan"):9.1f}%')
    print(f'  {"":42s}{"(JSON 行号)":20s}{js[0][0]:12d}{SRC[key].split(":")[1]:>12s}')
print('  含义: 若同种子同配置的 total_flux 复现差已接近 ±10%，C2 守恒窗就窄于复现噪声。')

# ---------- §9 复现噪声正式测量: A_base seed0 跑 6 次 ----------
# 命令完全相同、无任何 --set、同 seed、同代码。差异只可能来自 GPU 逐格 scatter-add 的
# 原子重排在 20000 步上的混沌放大。用来回答: C2 的 ±10% 守恒窗有没有窄于复现噪声。
print('\n' + '=' * 118)
print('§9 复现噪声正式测量: A_base seed0 同命令重复 6 次'
      ' (outputs/20260803-repartition/replicate/, provenance 同目录)')
REPS = []
for r in range(1, 7):
    p = f'outputs/20260803-repartition/replicate/A_base_seed0_rep{r}.log'
    lines = open(p).read().splitlines()
    js = [(i + 1, l) for i, l in enumerate(lines) if l.startswith('JSON ')]
    assert len(js) == 1, f'{p}: {len(js)} 行 JSON'
    d = json.loads(js[0][1][5:])
    assert d['overrides'] == {} and d['seed'] == 0 and d['steps'] == 20000
    d['total_flux'] = d['graze_gain'] + d['fruit_gain']
    REPS.append((f'rep{r}:{js[0][0]}', d))
# 把 verdict 与 probe 的同配置同种子 run 一并纳入同一个复现样本
_pl = open('outputs/20260803-repartition/probe/t0_A_base.log').read().splitlines()
_pj = [(i + 1, l) for i, l in enumerate(_pl) if l.startswith('JSON ')][0]
_pr = json.loads(_pj[1][5:]); _pr['total_flux'] = _pr['graze_gain'] + _pr['fruit_gain']
ALLREP = REPS + [(f'verdict:{SRC[(BASE,0)]}', D[(BASE, 0)]), (f'probe/t0:{_pj[0]}', _pr)]
RK = ['total_flux', 'graze_gain', 'fruit_gain', 'population', 'carnivore_frac',
      'late_carn', 'min_pop', 'death_thirst_frac', 'forest_frac', 'plant_total']
print('  ' + f'{"run":22s}' + ''.join(f'{k[:13]:>14s}' for k in RK))
for tag, d in ALLREP:
    print('  ' + f'{tag:22s}' + ''.join(f'{d[k]:14.4f}' for k in RK))
for lbl, grp in [('6 次 replicate', REPS), ('8 次 (含 verdict+probe)', ALLREP)]:
    print(f'  --- {lbl} n={len(grp)} ---')
    print('  ' + f'{"均值":20s}' + ''.join(f'{st.mean([d[k] for _,d in grp]):14.4f}' for k in RK))
    print('  ' + f'{"SD":21s}' + ''.join(f'{st.stdev([d[k] for _,d in grp]):14.4f}' for k in RK))
    print('  ' + f'{"CV=SD/均值":18s}' + ''.join(
        f'{st.stdev([d[k] for _,d in grp])/abs(st.mean([d[k] for _,d in grp])):14.4f}' for k in RK))
    print('  ' + f'{"极差/均值":19s}' + ''.join(
        f'{(max(d[k] for _,d in grp)-min(d[k] for _,d in grp))/abs(st.mean([d[k] for _,d in grp])):14.4f}'
        for k in RK))
tf = [d['total_flux'] for _, d in ALLREP]
mn = st.mean(tf)
inw = sum(1 for x in tf if 0.90 <= x / mn <= 1.10)
print(f'\n  C2 直接检验: 把这 {len(tf)} 个**同配置同种子**的 total_flux 各自除以它们自己的均值，'
      f'落在 [0.90,1.10] 的有 {inw}/{len(tf)}。')
print(f'    比值范围 = [{min(tf)/mn:.3f}, {max(tf)/mn:.3f}]，'
      f'复现 CV = {st.stdev(tf)/mn:.4f}')
print(f'  跨种子对照: A_base 6 个不同种子的 total_flux CV = '
      f'{st.stdev([D[(BASE,s)]["total_flux"] for s in SEEDS])/st.mean([D[(BASE,s)]["total_flux"] for s in SEEDS]):.4f}'
      '  (换种子带来的变异 vs 同种子重跑带来的变异)')
print('  比值噪声估计: C2 用的是 treat/base 的比值，分子分母各带一份复现噪声，'
      f'\n    独立近似下比值 CV ≈ sqrt(2)*{st.stdev(tf)/mn:.4f} = {(2**0.5)*st.stdev(tf)/mn:.4f}，'
      f'即 ±10% 窗 ≈ ±{0.10/((2**0.5)*st.stdev(tf)/mn):.2f} 个复现 SD。')

# ---------- §10 用复现噪声(而非种子间SD)当尺子重新丈量每个效应 ----------
print('\n' + '=' * 118)
print('§10 以 §9 实测的复现噪声为尺子重新丈量效应量'
      '（种子间 SD 会低估噪声：换种子的变异比同种子重跑还小，见下 F 检验）')
from scipy.stats import f as fdist, levene
REPD = [d for _, d in REPS]
print(f'  {"metric":20s}{"复现SD(n=6)":>13s}{"种子间SD(n=6)":>14s}{"F=rep/seed":>12s}'
      f'{"p(双侧F)":>10s}{"Levene p":>10s}{"F−A 均差":>12s}{"/复现SD":>9s}'
      f'{"G−A 均差":>12s}{"/复现SD":>9s}')
for k in ['total_flux', 'frugivory_frac', 'population', 'carnivore_frac',
          'late_carn', 'death_thirst_frac', 'forest_frac', 'min_pop']:
    rsd = st.stdev([d[k] for d in REPD])
    ssd = st.stdev([D[(BASE, s)][k] for s in SEEDS])
    F = (rsd ** 2) / (ssd ** 2) if ssd > 0 else float('inf')
    pF = 2 * min(fdist.cdf(F, 5, 5), 1 - fdist.cdf(F, 5, 5)) if ssd > 0 else 0.0
    try:
        lv = levene([d[k] for d in REPD], [D[(BASE, s)][k] for s in SEEDS])[1]
    except Exception:
        lv = float('nan')
    row = f'  {k:20s}{rsd:13.5f}{ssd:14.5f}{F:12.2f}{pF:10.4f}{lv:10.4f}'
    for arm in TREAT:
        d = st.mean([D[(arm, s)][k] - D[(BASE, s)][k] for s in SEEDS])
        row += f'{d:+12.5f}{abs(d)/rsd if rsd else float("nan"):9.2f}'
    print(row)
print('  读法: |均差|/复现SD < 1 表示该效应小于「同种子重跑同一配置」自身的抖动。')

# ---------- §11 生态位分化本身的直接读数（不是通量，是行为分化） ----------
print('\n' + '=' * 118)
print('§11 「第二个生态位」的直接读数: 觅食偏好的分化程度'
      '（docs/multispecies_program.md §9 记 forage_pref_std 是分区信号）')
NICHE = ['mean_forage_pref', 'forage_pref_std', 'herb_forage_pref', 'diet_std',
         'mean_diet', 'herb_water_dist', 'carn_water_dist', 'inland_frac']
for arm in ARMS:
    print(f'  {arm:16s}' + '  '.join(
        f'{k}={st.mean([D[(arm,s)][k] for s in SEEDS]):.4f}' for k in NICHE))
print('  配对 Wilcoxon vs A_base（未做 Bonferroni；空间指标见伪重复警告）:')
for hi in TREAT:
    for k in NICHE:
        a = [D[(BASE, s)][k] for s in SEEDS]; b = [D[(hi, s)][k] for s in SEEDS]
        d = [x - y for x, y in zip(b, a)]
        try:
            w, p = wilcoxon(b, a)
        except ValueError:
            w, p = float('nan'), float('nan')
        up = sum(1 for x in d if x > 0)
        ci = boot_ci_diff(d)
        print(f'    {hi:16s}{k:18s} 差={st.mean(d):+9.5f}  '
              f'CI[{ci[0]:+.5f},{ci[1]:+.5f}]  p={p:.4f}  同向={max(up,6-up)}/6')
