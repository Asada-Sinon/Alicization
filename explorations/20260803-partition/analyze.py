"""R2「厚果层下的资源分割」P2 判决的全部数字（带每格 3 次重复）。

回答什么: docs/multispecies_program.md:539-540 的预注册 P2 ——
          厚果层下, forage_tradeoff 高选择强度臂 (Q) 的 forage_pref 分布 `sd`
          是否高于中性对照 (P, 基因编译期断开=纯漂变), 且 `blrt_lr_per_n` 同向更高;
          两项各 >=5/6 同向 + 配对 Wilcoxon p<=0.05。
          注意: 预注册写的是 tradeoff=0.5, 实跑是 1.0 (最大选择强度) —— 见 stdout §0 的偏差标注。
          本轮设计有重复: 每 (arm,seed) 格 3 次 repeat, 先取格均值再做 6 配对种子检验。
          重复的第二个用途: 直接从本批数据估「同种子复现噪声」, 不用别的配置下测的旧数。
          blrt_p 字段无效不得使用 (scripts/probe_trait_dist.py 模块 docstring)。
读哪些文件: outputs/20260803-partition/{P_tradeoff0,Q_tradeoff1}_s{0..5}_r{1..3}.log
          每个 log 末尾恰好一行 "JSON {...}"; provenance: 同目录 provenance.txt
输出怎么读: 全部数字由本脚本 stdout 打出, 报告中不得手算。
          §0 载入与归因核对 / §1 逐种子逐臂全量表(格均值 + 格内 3 次 SD)
          / §2 自估复现噪声(池化同种子 SD -> 取3次均值后的配对差噪声)
          / §3 P2 判据: 同向数 + 配对 Wilcoxon + 效应量 + bootstrap CI + 效应/噪声比
          / §4 直方图按臂求和 (40 桶) / §5 护栏 / §6 异常 / §7 一句话判决
重跑: XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python explorations/20260803-partition/analyze.py
"""
import json, os
import numpy as np
from scipy.stats import wilcoxon

DIR = 'outputs/20260803-partition'
ARMS = ['P_tradeoff0', 'Q_tradeoff1']
CTRL, TREAT = 'P_tradeoff0', 'Q_tradeoff1'
SEEDS = [0, 1, 2, 3, 4, 5]
REPS = [1, 2, 3]
RNG = np.random.default_rng(20260803)

# 报告要的全部指标
METRICS = ['sd', 'mean', 'blrt_lr_per_n', 'bimodality_coefficient', 'n',
           'population', 'carnivore_frac', 'frugivory_frac',
           'death_thirst_frac', 'min_pop']
# P2 的两个主判据量
PRIMARY = ['sd', 'blrt_lr_per_n']

# ---------------- §0 载入 ----------------
D, SRC, bad = {}, {}, []
for arm in ARMS:
    for s in SEEDS:
        for r in REPS:
            p = f'{DIR}/{arm}_s{s}_r{r}.log'
            if not os.path.exists(p):
                bad.append(f'{p}: 文件不存在'); continue
            lines = open(p).read().splitlines()
            js = [(i + 1, l) for i, l in enumerate(lines) if l.startswith('JSON ')]
            if len(js) != 1:
                bad.append(f'{p}: 找到 {len(js)} 行 JSON'); continue
            ln, l = js[0]
            rec = json.loads(l[5:])
            if rec['seed'] != s:
                bad.append(f'{p}: JSON seed={rec["seed"]} != 文件名 s{s}')
            if 'collapsed to zero' in '\n'.join(lines):
                bad.append(f'{p}: 出现 population collapsed to zero')
            D[(arm, s, r)] = rec
            SRC[(arm, s, r)] = f'{os.path.basename(p)}:{ln}'

print('=' * 78)
print('§0 载入与归因核对')
print('=' * 78)
print(f'期望 {len(ARMS)*len(SEEDS)*len(REPS)} 个 run, 实际载入 {len(D)} 个')
ov = {}
for k, rec in D.items():
    ov.setdefault(json.dumps(rec['overrides'], sort_keys=True), []).append(k)
for o, ks in sorted(ov.items()):
    print(f'  n={len(ks):2d}  overrides={o}')
sets = [set(json.loads(o).items()) for o in ov]
if len(sets) == 2:
    diff = sets[0] ^ sets[1]
    fields = sorted({k for k, _ in diff})
    print(f'  两臂 config 差异字段数 = {len(fields)}: {fields}   '
          f'{"[单变量, 可归因]" if len(fields)==1 else "[多变量, 不可归因!]"}')
stepset = sorted({rec['steps'] for rec in D.values()})
print(f'  steps 取值 = {stepset}   {"[一致]" if len(stepset)==1 else "[不一致!]"}')
print('  [偏差] 预注册 P2 (docs/multispecies_program.md:539) 写的是 forage_tradeoff=0.5;')
print('         本批实跑是 1.0。1.0 是更强选择, 未跑 0.5 => P3 剂量项无法检验。')

# ---------------- §1 逐种子表 ----------------
def cell(arm, s, m):
    """(arm,seed) 格: 3 次重复的 (均值, 样本SD(ddof=1), 三个原值)"""
    v = np.array([D[(arm, s, r)][m] for r in REPS], float)
    return v.mean(), v.std(ddof=1), v

print()
print('=' * 78)
print('§1 逐种子逐臂全量表  (格均值 ± 格内 3 次重复的样本 SD, ddof=1)')
print('=' * 78)
for m in METRICS:
    print(f'\n--- {m} ---')
    print(f'{"seed":>4} | {CTRL+" mean":>16} {"(rep SD)":>10} | '
          f'{TREAT+" mean":>16} {"(rep SD)":>10} | {"Q-P":>12} {"同向(Q>P)":>9}')
    for s in SEEDS:
        pm, ps, _ = cell(CTRL, s, m)
        qm, qs, _ = cell(TREAT, s, m)
        print(f'{s:>4} | {pm:>16.5f} {ps:>10.5f} | {qm:>16.5f} {qs:>10.5f} | '
              f'{qm-pm:>+12.5f} {"Y" if qm>pm else "n":>9}')
    pmu = np.array([cell(CTRL, s, m)[0] for s in SEEDS])
    qmu = np.array([cell(TREAT, s, m)[0] for s in SEEDS])
    print(f'{"MEAN":>4} | {pmu.mean():>16.5f} {"":>10} | {qmu.mean():>16.5f} {"":>10} | '
          f'{(qmu-pmu).mean():>+12.5f} {int((qmu>pmu).sum()):>7}/6')
    print(f'{"SD":>4} | {pmu.std(ddof=1):>16.5f} {"":>10} | {qmu.std(ddof=1):>16.5f}'
          f'   <- 跨种子 SD (格均值层面)')

print('\n每格 3 次重复的原始值 (追溯用, 文件名 -> 值):')
for m in PRIMARY:
    print(f'  [{m}]')
    for arm in ARMS:
        for s in SEEDS:
            vals = [(SRC[(arm, s, r)], D[(arm, s, r)][m]) for r in REPS]
            print('    ' + '  '.join(f'{f}={v:.5f}' for f, v in vals))

# ---------------- §2 自估复现噪声 ----------------
print()
print('=' * 78)
print('§2 自估同种子复现噪声 (只用本批 36 个 run, 不用旧配置下测的数)')
print('=' * 78)
noise = {}
for m in PRIMARY + ['population', 'carnivore_frac']:
    print(f'\n--- {m} ---')
    for arm in ARMS:
        within = []
        for s in SEEDS:
            _, sd_, v = cell(arm, s, m)
            within.append(sd_)
            print(f'  {arm} seed{s}: reps={np.array2string(v, precision=5)}  '
                  f'格内SD={sd_:.5f}')
        w = np.array(within)
        # 池化: sqrt(mean of variances), 每格 dof=2, 共 12 dof
        pooled = float(np.sqrt((w ** 2).mean()))
        print(f'  {arm} 池化同种子 SD (sqrt(mean var), dof={2*len(SEEDS)}) = {pooled:.5f}'
              f'   [格内SD 范围 {w.min():.5f}..{w.max():.5f}]')
        noise[(arm, m)] = pooled
    pooled_both = float(np.sqrt(np.mean(
        [cell(a, s, m)[1] ** 2 for a in ARMS for s in SEEDS])))
    noise[('BOTH', m)] = pooled_both
    # 取 3 次均值后, 配对差 (Q_bar - P_bar) 的噪声
    pd_ctrl = noise[(CTRL, m)] * np.sqrt(2) / np.sqrt(3)
    pd_both = pooled_both * np.sqrt(2) / np.sqrt(3)
    noise[('PAIRDIFF_CTRL', m)] = pd_ctrl
    noise[('PAIRDIFF_BOTH', m)] = pd_both
    print(f'  两臂合并池化 SD = {pooled_both:.5f} (dof={4*len(SEEDS)})')
    print(f'  => 取 3 次均值后的配对差噪声 sqrt(2)*SD/sqrt(3):')
    print(f'       用 P 臂 SD : {pd_ctrl:.5f}')
    print(f'       用两臂合并 : {pd_both:.5f}')

# ---------------- §3 P2 判据 ----------------
def boot_ci(d, B=20000, alpha=0.05):
    d = np.asarray(d, float)
    idx = RNG.integers(0, len(d), size=(B, len(d)))
    bs = d[idx].mean(axis=1)
    return float(np.percentile(bs, 100 * alpha / 2)), float(np.percentile(bs, 100 * (1 - alpha / 2)))

print()
print('=' * 78)
print('§3 P2 判据判定 (格均值层面, n=6 配对种子)')
print('=' * 78)
print('注: n=6 配对 Wilcoxon 双侧最小可达 p = 0.03125 (地板)。')
allp = []
for m in PRIMARY:
    pmu = np.array([cell(CTRL, s, m)[0] for s in SEEDS])
    qmu = np.array([cell(TREAT, s, m)[0] for s in SEEDS])
    d = qmu - pmu
    same = int((d > 0).sum())
    W2, p2 = wilcoxon(qmu, pmu, alternative='two-sided')
    W1, p1 = wilcoxon(qmu, pmu, alternative='greater')
    lo, hi = boot_ci(d)
    dz = float(d.mean() / d.std(ddof=1))
    # 匹配对秩双列相关
    r = np.abs(d); rk = np.argsort(np.argsort(r)) + 1.0
    # 处理并列
    from scipy.stats import rankdata
    rk = rankdata(r)
    Rp = rk[d > 0].sum(); Rm = rk[d < 0].sum()
    rrb = float((Rp - Rm) / (len(d) * (len(d) + 1) / 2))
    pdn = noise[('PAIRDIFF_BOTH', m)]
    pdn_c = noise[('PAIRDIFF_CTRL', m)]
    allp += [('%s two-sided' % m, p2), ('%s one-sided greater' % m, p1)]
    print(f'\n--- {m} ---')
    print(f'  P 均值 = {pmu.mean():.5f}   Q 均值 = {qmu.mean():.5f}   差 = {d.mean():+.5f}')
    print(f'  逐种子差 = {np.array2string(d, precision=5, sign="+")}')
    print(f'  Q>P 同向种子数 = {same}/6   (P2 要求 >=5/6)')
    print(f'  配对 Wilcoxon 双侧: W={W2:.1f}  p={p2:.5f}   (P2 要求 p<=0.05)')
    print(f'  配对 Wilcoxon 单侧(Q>P): W={W1:.1f}  p={p1:.5f}')
    print(f'  95% bootstrap CI (B=20000, 配对差均值) = [{lo:+.5f}, {hi:+.5f}]')
    print(f'  效应量: Cohen dz = {dz:+.3f}   匹配对秩双列 r_rb = {rrb:+.3f}')
    print(f'  效应/噪声比 (自估, 两臂合并): {d.mean()/pdn:+.3f}   '
          f'(差 {d.mean():+.5f} / 配对差噪声 {pdn:.5f})')
    print(f'  效应/噪声比 (自估, 仅 P 臂): {d.mean()/pdn_c:+.3f}   '
          f'(差 {d.mean():+.5f} / 配对差噪声 {pdn_c:.5f})')
    print(f'  跨种子(格均值)差的 SD = {d.std(ddof=1):.5f}  '
          f'-> 该差 |{d.mean():+.5f}| vs 种子间散度')
    ok = (same >= 5) and (p2 <= 0.05) and (d.mean() > 0)
    print(f'  => {m} 项: {"达成" if ok else "未达成"}')

# 次要指标也做一遍配对检验, 全部 p 值都要报 (不做 Bonferroni)
print('\n--- 其余指标的配对 Wilcoxon (双侧, 探索性; 全部算过的 p 都在此列出) ---')
for m in [x for x in METRICS if x not in PRIMARY]:
    pmu = np.array([cell(CTRL, s, m)[0] for s in SEEDS])
    qmu = np.array([cell(TREAT, s, m)[0] for s in SEEDS])
    d = qmu - pmu
    if np.allclose(d, 0):
        print(f'  {m:<24} 全部差为 0, 跳过检验'); continue
    W, p = wilcoxon(qmu, pmu, alternative='two-sided')
    lo, hi = boot_ci(d)
    allp.append((m + ' two-sided', p))
    print(f'  {m:<24} P={pmu.mean():>10.4f} Q={qmu.mean():>10.4f} '
          f'd={d.mean():>+10.4f} CI=[{lo:+.4f},{hi:+.4f}] p={p:.5f} 同向={int((d>0).sum())}/6')

# ---------------- §4 直方图 ----------------
print()
print('=' * 78)
print('§4 直方图: 每臂 18 个 run 的 hist 逐桶求和 (40 桶, [0,1])')
print('=' * 78)
H = {}
for arm in ARMS:
    h = np.zeros(40)
    for s in SEEDS:
        for r in REPS:
            h += np.array(D[(arm, s, r)]['hist'], float)
    H[arm] = h
    print(f'\n{arm}  总计数 = {h.sum():.0f}')
    print('  桶心   计数    占比%   条形(按该臂峰值归一, 满宽60)')
    peak = h.max()
    for i in range(40):
        c = h[i]
        if c == 0 and (i < 4 or i > 35):
            continue
        bar = '#' * int(round(60 * c / peak)) if peak > 0 else ''
        print(f'  {(i+0.5)/40:.4f} {c:>7.0f} {100*c/h.sum():>7.2f}  {bar}')
print('\n两臂并排 (占比%, 只列非零桶):')
print(f'  {"桶心":>7} {"P%":>8} {"Q%":>8} {"Q-P":>8}')
pp = 100 * H[CTRL] / H[CTRL].sum(); qq = 100 * H[TREAT] / H[TREAT].sum()
for i in range(40):
    if pp[i] == 0 and qq[i] == 0:
        continue
    print(f'  {(i+0.5)/40:>7.4f} {pp[i]:>8.2f} {qq[i]:>8.2f} {qq[i]-pp[i]:>+8.2f}')
# 形状描述量
for arm in ARMS:
    h = H[arm]; c = (np.arange(40) + 0.5) / 40
    nz = np.nonzero(h)[0]
    # 局部极大 (严格大于两侧邻居, 且计数 >=1% 峰值)
    loc = [i for i in range(1, 39) if h[i] > h[i-1] and h[i] > h[i+1] and h[i] >= 0.01*h.max()]
    print(f'\n{arm}: 非零桶范围 = [{c[nz[0]]:.4f}, {c[nz[-1]]:.4f}] ({len(nz)} 个非零桶)')
    print(f'  局部极大桶心 = {[round(float(c[i]),4) for i in loc]}  (计数 {[int(h[i]) for i in loc]})')
    print(f'  合并分布 mean={float((h*c).sum()/h.sum()):.4f} '
          f'sd={float(np.sqrt((h*(c-(h*c).sum()/h.sum())**2).sum()/h.sum())):.4f}')

# ---------------- §5 护栏 ----------------
print()
print('=' * 78)
print('§5 护栏: carnivore_frac >= 0.05 与 min_pop >= 1, 逐种子')
print('=' * 78)
for arm in ARMS:
    okc = okm = 0
    print(f'\n{arm}:')
    for s in SEEDS:
        cm, _, cv = cell(arm, s, 'carnivore_frac')
        mm, _, mv = cell(arm, s, 'min_pop')
        pc, pm_ = cm >= 0.05, mm >= 1
        okc += pc; okm += pm_
        allrep_c = bool((cv >= 0.05).all()); allrep_m = bool((mv >= 1).all())
        print(f'  seed{s}: carn_frac 格均值={cm:.4f} {"PASS" if pc else "FAIL"}'
              f' (逐 run {np.array2string(cv, precision=4)} 全过={allrep_c}) | '
              f'min_pop 格均值={mm:.1f} {"PASS" if pm_ else "FAIL"}'
              f' (逐 run {np.array2string(mv, precision=0)} 全过={allrep_m})')
    print(f'  => carnivore_frac {okc}/6, min_pop {okm}/6 (格均值口径)')

# ---------------- §6 异常 ----------------
print()
print('=' * 78)
print('§6 异常')
print('=' * 78)
print('载入异常:', bad if bad else '无')
small = [(SRC[k], D[k]['n']) for k in D if D[k]['n'] < 30]
print('n<30 (脚本会跳过形状检验) 的 run:', small if small else '无')
ns = np.array([D[k]['n'] for k in D])
print(f'n 范围 = [{ns.min()}, {ns.max()}]  中位 = {np.median(ns):.0f}')
mp = np.array([D[k]['min_pop'] for k in D])
print(f'min_pop 范围 = [{mp.min():.0f}, {mp.max():.0f}]  (0 = 灭绝)')
print('灭绝 run:', [SRC[k] for k in D if D[k]['min_pop'] < 1] or '无')
cf = np.array([D[k]['carnivore_frac'] for k in D])
print(f'carnivore_frac 逐 run 范围 = [{cf.min():.4f}, {cf.max():.4f}]; '
      f'<0.05 的 run 数 = {int((cf<0.05).sum())}/36')
print(f'blrt_p 逐 run 取值 = {sorted(set(round(D[k]["blrt_p"],4) for k in D))} '
      f'(该字段无效, 仅记录其确实打在地板上)')

# ---------------- §7 判决 ----------------
print()
print('=' * 78)
print('§7 全部算过的 p 值 (不做 Bonferroni)')
print('=' * 78)
for name, p in allp:
    print(f'  {name:<40} p={p:.5f}')

print()
print('=' * 78)
print('§7b P2 一句话判决')
print('=' * 78)
res = {}
for m in PRIMARY:
    pmu = np.array([cell(CTRL, s, m)[0] for s in SEEDS])
    qmu = np.array([cell(TREAT, s, m)[0] for s in SEEDS])
    d = qmu - pmu
    _, p2 = wilcoxon(qmu, pmu, alternative='two-sided')
    res[m] = ((d > 0).sum() >= 5) and (p2 <= 0.05) and (d.mean() > 0)
    print(f'  {m}: 同向 {int((d>0).sum())}/6, 双侧 p={p2:.5f}, 方向 '
          f'{"Q>P" if d.mean()>0 else "Q<P"} => {"达成" if res[m] else "未达成"}')
print(f'  P2 (两项都要达成) = {"成立" if all(res.values()) else "不成立"}')

# ---------------- §8 反向效应的量化 + 稳健性 ----------------
from scipy.stats import mannwhitneyu
print()
print('=' * 78)
print('§8 反向效应量化 (P2 失败的方向本身是一个结果) + 口径稳健性')
print('=' * 78)
for m in PRIMARY + ['bimodality_coefficient', 'frugivory_frac']:
    pmu = np.array([cell(CTRL, s, m)[0] for s in SEEDS])
    qmu = np.array([cell(TREAT, s, m)[0] for s in SEEDS])
    d = qmu - pmu
    rel = 100.0 * d.mean() / pmu.mean()
    rel_ps = 100.0 * (d / pmu)
    _, p_less = wilcoxon(qmu, pmu, alternative='less')
    lo, hi = boot_ci(rel_ps)
    print(f'\n--- {m} (反向) ---')
    print(f'  相对变化 = {rel:+.2f}%  (Q 均值 {qmu.mean():.5f} / P 均值 {pmu.mean():.5f})')
    print(f'  逐种子相对变化% = {np.array2string(rel_ps, precision=2, sign="+")}')
    print(f'  逐种子相对变化均值 = {rel_ps.mean():+.2f}%  95% bootstrap CI = [{lo:+.2f}%, {hi:+.2f}%]')
    print(f'  配对 Wilcoxon 单侧(Q<P) p={p_less:.5f}   Q<P 同向 = {int((d<0).sum())}/6')
    # run 级不配对稳健性 (18 vs 18) -- 非独立(同种子3次), 仅作口径核对
    pv = np.array([D[(CTRL, s, r)][m] for s in SEEDS for r in REPS])
    qv = np.array([D[(TREAT, s, r)][m] for s in SEEDS for r in REPS])
    U, pu = mannwhitneyu(qv, pv, alternative='two-sided')
    print(f'  [稳健性核对, 非主判据] run 级 18v18 Mann-Whitney U={U:.1f} p={pu:.6f} '
          f'(注意: 同种子 3 次非独立, 该 p 偏乐观)')
    print(f'  [稳健性核对] 用格内中位数代替均值: '
          f'P={np.median([np.median([D[(CTRL,s,r)][m] for r in REPS]) for s in SEEDS]):.5f} '
          f'Q={np.median([np.median([D[(TREAT,s,r)][m] for r in REPS]) for s in SEEDS]):.5f}')
    allp.append((m + ' one-sided less', p_less))
    allp.append((m + ' run-level 18v18 MWU two-sided', pu))

# 逐 run 一致性: 36 个 run 里, Q 的 sd 低于同种子 P 的三个值的次数
print('\n逐 run 一致性 (sd): 每个 Q run 与同种子 3 个 P run 比较')
tot = win = 0
for s in SEEDS:
    for r in REPS:
        q = D[(TREAT, s, r)]['sd']
        for r2 in REPS:
            tot += 1; win += q < D[(CTRL, s, r2)]['sd']
print(f'  Q_run < P_run 的成对次数 = {win}/{tot} = {100*win/tot:.1f}%  '
      f'(同种子内全部 3x3 配对)')

# ---------------- §9 低值次峰: Q 臂那个分离的小簇 ----------------
print()
print('=' * 78)
print('§9 Q 臂直方图低值端的分离小簇 (桶 0.225-0.275 vs 谷底 0.275-0.350)')
print('=' * 78)
LOW = slice(9, 11)     # 桶心 0.2375, 0.2625
VAL = slice(11, 14)    # 桶心 0.2875, 0.3125, 0.3375
for arm in ARMS:
    tot_ = H[arm].sum()
    lo_ = H[arm][LOW].sum(); va_ = H[arm][VAL].sum()
    print(f'  {arm}: 低簇 {lo_:.0f} ({100*lo_/tot_:.2f}%)  谷底 {va_:.0f} ({100*va_/tot_:.2f}%)  '
          f'簇/谷 比 = {lo_/va_ if va_ else float("inf"):.2f}')
print('  (比 >1 = 低值端与主峰之间存在真空 -> 分离的次群; 比 <1 = 只是单调的尾巴)')
print('\n  该小簇由哪些 run 贡献 (低簇计数 >= 20 的 run):')
for arm in ARMS:
    for s in SEEDS:
        for r in REPS:
            h = np.array(D[(arm, s, r)]['hist'], float)
            l_, v_ = h[LOW].sum(), h[VAL].sum()
            if l_ >= 20:
                print(f'    {SRC[(arm,s,r)]}: 低簇={l_:.0f} 谷底={v_:.0f} '
                      f'n={D[(arm,s,r)]["n"]} sd={D[(arm,s,r)]["sd"]:.4f} '
                      f'bimod={D[(arm,s,r)]["bimodality_coefficient"]:.4f}')

# ---------------- §10 全部 p 值重印 ----------------
print()
print('=' * 78)
print('§10 全部算过的 p 值 (含 §8, 不做 Bonferroni)')
print('=' * 78)
for name, p in allp:
    print(f'  {name:<48} p={p:.6f}')

# ---------------- §11 方差分解: 配对到底买到了什么 + 噪声模型自洽性 ----------------
print()
print('=' * 78)
print('§11 方差分解与噪声模型自洽性检查')
print('=' * 78)
PREREG_PAIRED_NOISE = {'sd': 0.0084}   # provenance.txt:18 的假设 sqrt(2)*0.0103/sqrt(3)
for m in PRIMARY:
    print(f'\n--- {m} ---')
    for arm in ARMS:
        cm = np.array([cell(arm, s, m)[0] for s in SEEDS])
        sw = noise[(arm, m)]                       # 池化格内 SD (单 run)
        obs = cm.std(ddof=1)                       # 格均值的跨种子 SD
        exp_noise_only = sw / np.sqrt(len(REPS))   # 只由格内噪声造成的格均值散度
        v_seed = obs ** 2 - exp_noise_only ** 2    # 真创始者方差分量 (可为负)
        icc = v_seed / (v_seed + sw ** 2) if (v_seed + sw ** 2) > 0 else float('nan')
        print(f'  {arm}: 单run格内SD={sw:.5f}  格均值跨种子SD(实测)={obs:.5f}  '
              f'仅噪声预期={exp_noise_only:.5f}')
        print(f'    -> 创始者方差分量 = {v_seed:+.6g} (SD={np.sqrt(v_seed):.5f})'
              if v_seed > 0 else f'    -> 创始者方差分量 = {v_seed:+.6g} (<=0, 种子解释不了任何东西)')
        print(f'    -> 近似 ICC = {icc:.3f}')
    # 配对差的噪声模型 vs 实测
    d = np.array([cell(TREAT, s, m)[0] - cell(CTRL, s, m)[0] for s in SEEDS])
    pred = np.sqrt(noise[(CTRL, m)] ** 2 / 3 + noise[(TREAT, m)] ** 2 / 3)
    print(f'  配对差 d 的实测 SD = {d.std(ddof=1):.5f}')
    print(f'  仅由格内噪声预测的 d 的 SD = sqrt(sP^2/3 + sQ^2/3) = {pred:.5f}')
    print(f'  实测/预测 = {d.std(ddof=1)/pred:.2f}  '
          f'(~1 表示配对差几乎全是复现噪声, 种子与 种子x臂 交互贡献很小)')
    if m in PREREG_PAIRED_NOISE:
        pre = PREREG_PAIRED_NOISE[m]
        print(f'  [对照预注册] provenance.txt:18 假设配对差噪声 = {pre:.4f}; '
              f'本批自估 = {pred:.5f}  -> 低估了 {pred/pre:.1f} 倍')
    print(f'  效应/噪声比 (用本批自估的 d 噪声 {pred:.5f}) = {d.mean()/pred:+.3f}')

# ---------------- §12 低簇是不是单个 run 的产物 ----------------
print()
print('=' * 78)
print('§12 低值小簇的 run 归属 (全部 36 个 run 的低簇计数)')
print('=' * 78)
for arm in ARMS:
    counts = [(SRC[(arm, s, r)], float(np.array(D[(arm, s, r)]['hist'], float)[LOW].sum()))
              for s in SEEDS for r in REPS]
    counts.sort(key=lambda t: -t[1])
    tot_ = sum(c for _, c in counts)
    print(f'\n  {arm}: 低簇总计 {tot_:.0f}')
    for f, c in counts[:5]:
        print(f'    {f}: {c:.0f}  ({100*c/tot_:.1f}% of arm low-cluster)' if tot_ else f'    {f}: {c:.0f}')
    print(f'    其余 {len(counts)-5} 个 run 合计 = {sum(c for _, c in counts[5:]):.0f}')
