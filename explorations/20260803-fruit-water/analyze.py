"""R1c 果实的预成水 —— 6 配对种子正式判决的全部数字。

回答什么: docs/multispecies_program.md §8 (docs/multispecies_program.md:407-411) 的预注册判据下,
          C_fwf10 / D_fwf40 / E_fwf60 是否各自通过 —— 中位 frugivory_frac>=0.20
          且 carnivore_frac>=0.05 6/6 且 population 在同种子 A_base [0.7,1.3]x 6/6
          且 death_thirst_frac <= 同种子 A_base +5pp 6/6（护栏一律逐种子，不做中位读法）。
          核心机制问题: fruit_water_frac 0.10->0.40->0.60 时渴死超出量是否单调下降、在哪个剂量上修好。
读哪些文件: outputs/20260803-fruit-water/{A_base,C_fwf10,D_fwf40,E_fwf60}_seed{0..5}.log
          每个 log 末尾恰好一行 "JSON {...}"。provenance 见同目录 provenance.txt。
输出怎么读: 全部数字均由本脚本 stdout 打出，报告中不得手算。
          §0 载入与 overrides 归因核对 / §1 逐种子全量表 / §2 判据逐条判定
          / §3 剂量-响应: 渴死超出量单调性 / §4 配对 Wilcoxon (D-A,E-A,D-C,E-C)
          / §5 预标风险(水杠杆副作用)核对 / §6 异常、反例与方差 / §7 一句话判决
重跑: XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python explorations/20260803-fruit-water/analyze.py
"""
import json, os, statistics as st
import numpy as np
from scipy.stats import wilcoxon, rankdata

DIR = 'outputs/20260803-fruit-water'
ARMS = ['A_base', 'C_fwf10', 'D_fwf40', 'E_fwf60']
TREAT = ['C_fwf10', 'D_fwf40', 'E_fwf60']
SEEDS = [0, 1, 2, 3, 4, 5]

# ---------- §0 载入 ----------
D, SRC = {}, {}
missing = []
for arm in ARMS:
    for s in SEEDS:
        p = f'{DIR}/{arm}_seed{s}.log'
        if not os.path.exists(p):
            missing.append(p); continue
        lines = open(p).read().splitlines()
        js = [(i + 1, l) for i, l in enumerate(lines) if l.startswith('JSON ')]
        if len(js) != 1:
            missing.append(f'{p}: 找到 {len(js)} 行 JSON'); continue
        ln, l = js[0]
        r = json.loads(l[5:])
        assert r['seed'] == s, f'{p}: JSON seed={r["seed"]} != 文件名 seed{s}'
        D[(arm, s)] = r
        SRC[(arm, s)] = f'{os.path.basename(p)}:{ln}'
print('=' * 108)
print(f'§0 载入 {len(D)}/{len(ARMS)*len(SEEDS)} 个 run。缺失/异常: {missing if missing else "无"}')
print(f'   steps 集合={sorted({r["steps"] for r in D.values()})}  '
      f'(全一致={len({r["steps"] for r in D.values()})==1})')
print('   JSON 行号（报数字时的出处）: ' + '; '.join(
    f'{a}_seed*={sorted({SRC[(a,s)].split(":")[1] for s in SEEDS})}' for a in ARMS))
# 灭绝 / 崩溃扫描
print('   灭绝扫描 (population==0 或 min_pop==0):')
for arm in ARMS:
    bad = [s for s in SEEDS if D[(arm, s)]['population'] <= 0 or D[(arm, s)]['min_pop'] <= 0]
    print(f'     {arm:10s} -> {"无" if not bad else bad}')
print('   overrides 归因核对（取自 --json 的 overrides key，不是命令行文本）:')
for arm in ARMS:
    ovs = {json.dumps(D[(arm, s)]['overrides'], sort_keys=True) for s in SEEDS}
    print(f'     {arm:10s} 臂内 6 种子 overrides 一致={len(ovs) == 1}  ->  {sorted(ovs)[0]}')

def ovdiff(lo, hi):
    a, b = D[(lo, 0)]['overrides'], D[(hi, 0)]['overrides']
    keys = sorted(set(a) | set(b))
    return [(k, a.get(k, '默认'), b.get(k, '默认')) for k in keys
            if a.get(k, '默认') != b.get(k, '默认')]

print('   臂间 config 差异处数（>1 处则不可归因到单一变量）:')
for lo, hi in [('A_base', 'C_fwf10'), ('A_base', 'D_fwf40'), ('A_base', 'E_fwf60'),
               ('C_fwf10', 'D_fwf40'), ('C_fwf10', 'E_fwf60'), ('D_fwf40', 'E_fwf60')]:
    d = ovdiff(lo, hi)
    print(f'     {hi} vs {lo}: {len(d)} 处 -> ' + '; '.join(f'{k}: {x} -> {y}' for k, x, y in d))

# ---------- §1 逐种子全量表 ----------
METRICS = ['frugivory_frac', 'graze_gain', 'fruit_gain', 'population', 'carnivore_frac',
           'late_carn', 'min_pop', 'death_thirst_frac', 'death_starvation_frac',
           'death_predation_frac', 'forest_frac', 'mean_water', 'plant_total', 'fruit_total']
print('\n' + '=' * 108)
print('§1 逐种子全量表（每格出处 = <arm>_seed<N>.log 的 JSON 行 -> key）')
for k in METRICS:
    print(f'\n  -- {k} --')
    print('    ' + f'{"seed":>5s}' + ''.join(f'{a:>13s}' for a in ARMS) +
          f'{"C-A":>12s}{"D-A":>12s}{"E-A":>12s}')
    for s in SEEDS:
        v = [D[(a, s)][k] for a in ARMS]
        print('    ' + f'{s:>5d}' + ''.join(f'{x:>13.5g}' for x in v) +
              ''.join(f'{v[i] - v[0]:>+12.5g}' for i in (1, 2, 3)))
    for lab, fn in [('中位', st.median), ('均值', st.mean), ('臂内SD', st.stdev)]:
        print('    ' + f'{lab:>5s}' + ''.join(f'{fn([D[(a, s)][k] for s in SEEDS]):>13.5g}'
                                              for a in ARMS))

# ---------- §2 预注册判据逐条 ----------
print('\n' + '=' * 108)
print('§2 预注册判据逐条判定（docs/multispecies_program.md:407-411 原文，护栏一律逐种子 6/6，'
      '不引入中位读法）')
verdict = {}
for arm in TREAT:
    print(f'\n  ##### 臂 {arm}（overrides={json.dumps(D[(arm,0)]["overrides"], sort_keys=True)}） #####')
    fr = [D[(arm, s)]['frugivory_frac'] for s in SEEDS]
    med_fr = st.median(fr)
    c1 = med_fr >= 0.20
    print(f'  [1] 中位 frugivory_frac >= 0.20 : 中位={med_fr:.5f} -> {"达成" if c1 else "未达成"}'
          f'   （逐种子 >=0.20 的有 {sum(1 for x in fr if x >= 0.20)}/6）')
    print('      逐种子: ' + '  '.join(f's{s}={x:.4f}' for s, x in zip(SEEDS, fr)))
    cf = [D[(arm, s)]['carnivore_frac'] for s in SEEDS]
    n_cf = sum(1 for x in cf if x >= 0.05)
    c2 = n_cf == 6
    print(f'  [2] carnivore_frac >= 0.05 逐种子 6/6 : {n_cf}/6 -> {"达成" if c2 else "未达成"}'
          f'   (中位={st.median(cf):.5f})')
    print('      逐种子: ' + '  '.join(f's{s}={x:.4f}{"" if x >= 0.05 else "<-破"}'
                                     for s, x in zip(SEEDS, cf)))
    n_pop = 0
    print('  [3] population 在同种子 A_base 的 [0.7,1.3]x 内 6/6:')
    for s in SEEDS:
        a, b = D[('A_base', s)]['population'], D[(arm, s)]['population']
        r = b / a; ok = 0.7 <= r <= 1.3; n_pop += ok
        print(f'      seed{s}: A={a:.0f} {arm}={b:.0f} 比值={r:.4f} '
              f'窗口=[{0.7*a:.0f},{1.3*a:.0f}] -> {"OK" if ok else "破"}')
    c3 = n_pop == 6
    print(f'      -> {n_pop}/6 => {"达成" if c3 else "未达成"}')
    n_th = 0
    print('  [4] death_thirst_frac <= 同种子 A_base + 5pp 6/6:')
    for s in SEEDS:
        a, b = D[('A_base', s)]['death_thirst_frac'], D[(arm, s)]['death_thirst_frac']
        dpp = (b - a) * 100; ok = dpp <= 5.0; n_th += ok
        print(f'      seed{s}: A={a*100:6.2f}% {arm}={b*100:6.2f}% 差={dpp:+7.2f}pp '
              f'-> {"OK" if ok else "破"}')
    c4 = n_th == 6
    print(f'      -> {n_th}/6 => {"达成" if c4 else "未达成"}')
    allok = c1 and c2 and c3 and c4
    verdict[arm] = (c1, c2, c3, c4, allok, med_fr, n_cf, n_pop, n_th)
    print(f'  ===> {arm} 四条全过: {"通过" if allok else "不通过"}  '
          f'(条件1={c1}, 条件2={n_cf}/6, 条件3={n_pop}/6, 条件4={n_th}/6)')

# ---------- §3 剂量-响应 ----------
print('\n' + '=' * 108)
print('§3 核心问题: fruit_water_frac 0.10 -> 0.40 -> 0.60 时，death_thirst_frac 相对同种子'
      '基线的超出量是否单调下降；渴死护栏在哪个剂量上被修好（上限 +5.00pp）')
print(f'  {"seed":>5s}{"A基线%":>10s}' + ''.join(f'{a+"%":>10s}' for a in TREAT) +
      ''.join(f'{"Δ"+a[0]+"(pp)":>12s}' for a in TREAT) + f'{"单调递减?":>11s}')
mono = 0
for s in SEEDS:
    base = D[('A_base', s)]['death_thirst_frac']
    vals = [D[(a, s)]['death_thirst_frac'] for a in TREAT]
    dd = [(v - base) * 100 for v in vals]
    m = dd[0] > dd[1] > dd[2]; mono += m
    print(f'  {s:>5d}{base*100:>10.2f}' + ''.join(f'{v*100:>10.2f}' for v in vals) +
          ''.join(f'{x:>+12.2f}' for x in dd) + f'{"是" if m else "否":>10s}')
print(f'  单调递减(ΔC>ΔD>ΔE)的种子数 = {mono}/6')
for i, a in enumerate(TREAT):
    dd = [(D[(a, s)]['death_thirst_frac'] - D[('A_base', s)]['death_thirst_frac']) * 100
          for s in SEEDS]
    n_ok = sum(1 for x in dd if x <= 5.0)
    print(f'  {a}: Δ 均值={st.mean(dd):+7.2f}pp 中位={st.median(dd):+7.2f}pp '
          f'min={min(dd):+7.2f} max={max(dd):+7.2f}  <=+5pp 的种子 {n_ok}/6  '
          f'-> 护栏{"守住" if n_ok == 6 else "破裂"}')
print('  绝对渴死率均值（不减基线）: ' + '  '.join(
    f'{a}={st.mean([D[(a,s)]["death_thirst_frac"] for s in SEEDS])*100:.2f}%' for a in ARMS))
print('  mean_water 均值（世界里的水是否真的变多）: ' + '  '.join(
    f'{a}={st.mean([D[(a,s)]["mean_water"] for s in SEEDS]):.4f}' for a in ARMS))

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

TEST_METRICS = ['frugivory_frac', 'population', 'carnivore_frac', 'death_thirst_frac',
                'mean_water', 'forest_frac']

def compare(hi, lo):
    print(f'\n  ##### {hi} vs {lo}（配对 Wilcoxon 符号秩，双侧，n=6；未做 Bonferroni） #####')
    print(f'  {"metric":20s}{lo+"均":>13s}{hi+"均":>13s}{"差(hi-lo)":>12s}'
          f'{"boot95%CI":>26s}{"W":>7s}{"p双侧":>9s}{"r_rb":>8s}{"同向":>8s}')
    for k in TEST_METRICS:
        a = [D[(lo, s)][k] for s in SEEDS]
        b = [D[(hi, s)][k] for s in SEEDS]
        d = [x - y for x, y in zip(b, a)]
        w, p = wilcoxon(b, a)
        ci = boot_ci_diff(d); r = rank_biserial(b, a)
        up = sum(1 for x in d if x > 0)
        print(f'  {k:20s}{st.mean(a):13.5f}{st.mean(b):13.5f}{st.mean(d):+12.5f}'
              f'  [{ci[0]:+.5f},{ci[1]:+.5f}]{w:7.1f}{p:9.4f}{r:+8.3f}'
              f'{max(up, 6-up):6d}/6')

print('\n' + '=' * 108)
print('§4 配对 Wilcoxon。n=6 配对符号秩双侧 p 的理论下限 = '
      f'{2/2**6:.4f}（=2/64），全同向时必然取到；0.031 是地板不是强证据。'
      '\n   注意: D/E/C vs A_base 是 3 处 config 差异，不可归因到 fruit_water_frac 单独一项；'
      '\n   只有 D vs C 与 E vs C 是单变量对照（仅 fruit_water_frac 不同）。')
for hi, lo in [('D_fwf40', 'A_base'), ('E_fwf60', 'A_base'),
               ('D_fwf40', 'C_fwf10'), ('E_fwf60', 'C_fwf10'),
               ('C_fwf10', 'A_base'), ('E_fwf60', 'D_fwf40')]:
    compare(hi, lo)

# ---------- §5 预标风险 ----------
print('\n' + '=' * 108)
print('§5 预标风险核对（docs/water_system.md:30-37 记: 有效水杠杆让种群 +27%~+72%、'
      'carnivore_frac 升到 33%~41%）')
print(f'  {"arm":10s}{"pop均":>9s}{"pop比A均":>10s}{"pop比min":>10s}{"pop比max":>10s}'
      f'{"carn均":>9s}{"carn中位":>10s}{"carn max":>10s}{"late_carn均":>12s}')
for arm in ARMS:
    pop = [D[(arm, s)]['population'] for s in SEEDS]
    pr = [D[(arm, s)]['population'] / D[('A_base', s)]['population'] for s in SEEDS]
    cf = [D[(arm, s)]['carnivore_frac'] for s in SEEDS]
    lc = [D[(arm, s)]['late_carn'] for s in SEEDS]
    print(f'  {arm:10s}{st.mean(pop):9.1f}{st.mean(pr):10.4f}{min(pr):10.4f}{max(pr):10.4f}'
          f'{st.mean(cf):9.4f}{st.median(cf):10.4f}{max(cf):10.4f}{st.mean(lc):12.4f}')
print('  -> 判断: 种群涨幅是否进入 +27%~+72% 区间、carnivore_frac 是否升到 0.33~0.41 区间，'
      '直接读上表的 pop比A均 与 carn max。')
print('  死因结构（均值，看是否从渴死主导反转为捕食主导）:')
print(f'  {"arm":10s}{"thirst":>10s}{"starv":>10s}{"predation":>11s}{"senesc":>10s}{"和":>8s}')
for arm in ARMS:
    t = [st.mean([D[(arm, s)][f'death_{c}_frac'] for s in SEEDS])
         for c in ('thirst', 'starvation', 'predation', 'senescence')]
    print(f'  {arm:10s}{t[0]:10.4f}{t[1]:10.4f}{t[2]:11.4f}{t[3]:10.4f}{sum(t):8.4f}')

# ---------- §6 异常与方差 ----------
print('\n' + '=' * 108)
print('§6 异常、反例与臂内方差')
print(f'  (a) min_pop 逐种子（n_init=2000, n_max=16384；崩溃判据 min_pop 接近 0）:')
for arm in ARMS:
    v = [D[(arm, s)]['min_pop'] for s in SEEDS]
    print(f'    {arm:10s} ' + '  '.join(f's{s}={x:.0f}' for s, x in zip(SEEDS, v)) +
          f'   最小={min(v):.0f}  最小/n_init={min(v)/2000:.3f}')
print('  (b) 逐种子反向（配对差值与该对比的均值方向相反的种子）:')
any_rev = False
for hi, lo in [('C_fwf10', 'A_base'), ('D_fwf40', 'A_base'), ('E_fwf60', 'A_base'),
               ('D_fwf40', 'C_fwf10'), ('E_fwf60', 'C_fwf10'), ('E_fwf60', 'D_fwf40')]:
    for k in METRICS:
        d = [D[(hi, s)][k] - D[(lo, s)][k] for s in SEEDS]
        md = st.mean(d)
        rev = [(s, x) for s, x in zip(SEEDS, d) if (x > 0) != (md > 0) and x != 0]
        if rev:
            any_rev = True
            print(f'    {hi}-{lo:8s} {k:22s} 均值差={md:+.5g} 反向种子=' +
                  ', '.join(f's{s}:{x:+.5g}' for s, x in rev))
if not any_rev:
    print('    无')
print('  (c) 臂内变异系数 CV=SD/|均值|（>0.20 标 *）:')
for k in METRICS:
    line = []
    for arm in ARMS:
        v = [D[(arm, s)][k] for s in SEEDS]
        m, sd = st.mean(v), st.stdev(v)
        cv = sd / abs(m) if m != 0 else float('nan')
        line.append(f'{arm[0]}:{cv:.3f}{"*" if cv > 0.20 else " "}')
    print(f'    {k:22s} ' + '  '.join(line))
print('  (d) 配对差值 vs 两臂的种子间 SD（|差| < max(SD) 者标注）:')
for hi, lo in [('D_fwf40', 'A_base'), ('E_fwf60', 'A_base'),
               ('D_fwf40', 'C_fwf10'), ('E_fwf60', 'C_fwf10')]:
    for k in TEST_METRICS:
        d = st.mean([D[(hi, s)][k] - D[(lo, s)][k] for s in SEEDS])
        s_lo = st.stdev([D[(lo, s)][k] for s in SEEDS])
        s_hi = st.stdev([D[(hi, s)][k] for s in SEEDS])
        flag = '  <-- 差值 < 两臂SD' if abs(d) < max(s_lo, s_hi) else ''
        print(f'    {hi}-{lo:8s} {k:20s} 差={d:+.5f}  SD({lo})={s_lo:.5f} '
              f'SD({hi})={s_hi:.5f}{flag}')
print('  (e) 果层是否被吃满 / 能量来源结构:')
print(f'  {"arm":10s}{"fruit_total均":>14s}{"fruit_gain均":>13s}{"graze_gain均":>13s}'
      f'{"plant_total均":>14s}{"frug均":>9s}')
for arm in ARMS:
    g = lambda k: st.mean([D[(arm, s)][k] for s in SEEDS])
    print(f'  {arm:10s}{g("fruit_total"):14.4f}{g("fruit_gain"):13.4f}{g("graze_gain"):13.4f}'
          f'{g("plant_total"):14.2f}{g("frugivory_frac"):9.4f}')

# ---------- §7 判决 ----------
print('\n' + '=' * 108)
print('§7 一句话判决')
passed = [a for a in TREAT if verdict[a][4]]
for a in TREAT:
    c1, c2, c3, c4, allok, med_fr, n_cf, n_pop, n_th = verdict[a]
    print(f'  {a}: 条件1(中位frug>=0.20)={"达成" if c1 else "未达成"}({med_fr:.4f}) | '
          f'条件2(carn>=0.05)={n_cf}/6 | 条件3(pop窗口)={n_pop}/6 | 条件4(渴死<=+5pp)={n_th}/6 '
          f'=> {"通过" if allok else "不通过"}')
print(f'  通过预注册判据的臂: {passed if passed else "无 -> 触发 §8 失败判据"}')
print('\n完。所有数字见上；报告中任何数字都应能在本 stdout 中原样找到。')

# ---------- §8 补充: 渴死破裂与种群超调是否同源 ----------
print('\n' + '=' * 108)
print('§8 补充: 护栏破裂集中在 s3/s5 —— 渴死超出量是否由种群超调（密度依赖的水竞争）解释，'
      '而非水口径没修好')
from scipy.stats import spearmanr
print('  (a) 逐 (臂,种子) 的 pop 比值 vs 渴死Δ(pp)，跨 C/D/E 共 18 对:')
xs, ys, labs = [], [], []
for arm in TREAT:
    for s in SEEDS:
        pr = D[(arm, s)]['population'] / D[('A_base', s)]['population']
        dth = (D[(arm, s)]['death_thirst_frac'] - D[('A_base', s)]['death_thirst_frac']) * 100
        xs.append(pr); ys.append(dth); labs.append(f'{arm[0]}s{s}')
rho, p = spearmanr(xs, ys)
print('     ' + '  '.join(f'{l}:({x:.3f},{y:+.2f})' for l, x, y in zip(labs, xs, ys)))
print(f'     Spearman rho={rho:+.4f}  p={p:.4f}  (n=18 对；不是独立样本，仅作描述)')
print('  (b) 只在 D/E 两个高水臂内部（n=12）:')
x2 = [x for l, x in zip(labs, xs) if l[0] in 'DE']
y2 = [y for l, y in zip(labs, ys) if l[0] in 'DE']
rho2, p2 = spearmanr(x2, y2)
print(f'     Spearman rho={rho2:+.4f}  p={p2:.4f}')
print('  (c) 每个种子在三个剂量上的渴死Δ符号模式（看是否有恒定的「坏种子」）:')
for s in SEEDS:
    dd = [(D[(a, s)]['death_thirst_frac'] - D[('A_base', s)]['death_thirst_frac']) * 100
          for a in TREAT]
    pr = [D[(a, s)]['population'] / D[('A_base', s)]['population'] for a in TREAT]
    print(f'     seed{s}: ΔC={dd[0]:+6.2f} ΔD={dd[1]:+6.2f} ΔE={dd[2]:+6.2f}  '
          f'popr C/D/E={pr[0]:.3f}/{pr[1]:.3f}/{pr[2]:.3f}  '
          f'三档全破={"是" if all(x > 5 for x in dd) else "否"}')
print('  (d) 把 s3,s5 剔除后的 D/E 渴死Δ（仅作敏感性描述，不改判据）:')
for a in TREAT:
    dd = [(D[(a, s)]['death_thirst_frac'] - D[('A_base', s)]['death_thirst_frac']) * 100
          for s in SEEDS if s not in (3, 5)]
    print(f'     {a}: n=4 均值={st.mean(dd):+.2f}pp max={max(dd):+.2f}pp '
          f'<=+5pp 的 {sum(1 for x in dd if x <= 5)}/4')
