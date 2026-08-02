"""P4「窄分化窗口」收口: forage_tradeoff 的五点剂量-响应曲线。

回答什么: docs/multispecies_program.md §9.3 / provenance.txt:37-44 —— P3 只测了
          0.0 / 0.5 / 1.0 三点, 看不到比自身间距更窄的窗口。本轮补 0.125 (L) 与
          0.25 (N)。判据: sd(L) 或 sd(N) 是否 **高于** 中性漂变对照 sd(P, 0.0)。
          都不高于 -> 五点全单调, 「无分化窗口」收口;
          任一显著高于 -> 窗口存在 (需逐种子 / 逐 run / 直方图三重核验)。
          blrt_p 字段无效不得使用 (scripts/probe_trait_dist.py 模块 docstring;
          演化种群是谱系不是 iid, 中性对照也打在 p 地板上)。
读哪些文件: outputs/20260803-partition/{P_tradeoff0,L_tradeoff0125,N_tradeoff025,
          M_tradeoff05,Q_tradeoff1}_s{0..5}_r{1..3}.log —— 90 个, 每个第 12 行
          恰好一条 "JSON {...}"; provenance: 同目录 provenance.txt (三段追加)。
          P/Q 跑在 git 9296fd7, M 跑在 7937499, L/N 跑在 c1f9527;
          `git diff --stat 9296fd7 c1f9527 -- underworld/ scripts/` 为空 => 仿真代码同一版本。
输出怎么读: 全部数字由本脚本 stdout 打出, 报告中不得手算。
          §0 载入与五臂归因核对 / §1 五点剂量-响应表(全指标) / §2 逐种子五点 sd 曲线 +
          L>P、N>P 计数 / §3 L、N 两臂自估格内噪声 / §4 配对检验 L-P、N-P (+全臂复核) +
          五臂 Friedman / §5 直方图五臂并排 + 双峰逐 run 归属 / §6 护栏 / §7 异常 /
          §8 全部 p 值 / §9 判决判定量 + 窗口大小上界
重跑: XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python explorations/20260803-partition/analyze_curve.py
"""
import json, os
import numpy as np
from scipy.stats import wilcoxon, rankdata, friedmanchisquare, chi2 as _chi2

DIR = 'outputs/20260803-partition'
# 按剂量升序
ARMS = ['P_tradeoff0', 'L_tradeoff0125', 'N_tradeoff025', 'M_tradeoff05', 'Q_tradeoff1']
DOSE = {'P_tradeoff0': 0.0, 'L_tradeoff0125': 0.125, 'N_tradeoff025': 0.25,
        'M_tradeoff05': 0.5, 'Q_tradeoff1': 1.0}
SHORT = {'P_tradeoff0': 'P(0.0)', 'L_tradeoff0125': 'L(.125)', 'N_tradeoff025': 'N(.25)',
         'M_tradeoff05': 'M(0.5)', 'Q_tradeoff1': 'Q(1.0)'}
P, L, N, M, Q = ARMS
SEEDS = [0, 1, 2, 3, 4, 5]
REPS = [1, 2, 3]
RNG = np.random.default_rng(20260803)

METRICS = ['sd', 'blrt_lr_per_n', 'bimodality_coefficient', 'mean', 'n',
           'population', 'carnivore_frac', 'min_pop', 'death_thirst_frac',
           'frugivory_frac']
PRIMARY = ['sd', 'blrt_lr_per_n']
ALLP = []

# ---------------- §0 载入 + 五臂归因 ----------------
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
            got = rec['overrides'].get('forage_tradeoff', 0.0)
            if abs(float(got) - DOSE[arm]) > 1e-12:
                bad.append(f'{p}: overrides.forage_tradeoff={got} != 臂名剂量 {DOSE[arm]}')
            D[(arm, s, r)] = rec
            SRC[(arm, s, r)] = f'{os.path.basename(p)}:{ln}'

print('=' * 100)
print('§0 载入与五臂归因核对')
print('=' * 100)
print(f'期望 {len(ARMS)*len(SEEDS)*len(REPS)} 个 run, 实际载入 {len(D)} 个')
ov = {}
for k, rec in D.items():
    ov.setdefault(json.dumps(rec['overrides'], sort_keys=True), []).append(k)
for o, ks in sorted(ov.items(), key=lambda t: json.loads(t[0]).get('forage_tradeoff', 0.0)):
    arms_here = sorted({k[0] for k in ks})
    print(f'  n={len(ks):2d}  arms={arms_here}\n        overrides={o}')
print(f'  不同 overrides 组合数 = {len(ov)} (期望 5)')
odicts = {arm: None for arm in ARMS}
for o, ks in ov.items():
    for arm in {k[0] for k in ks}:
        odicts[arm] = json.loads(o)
keys_union = sorted(set().union(*[set(d) for d in odicts.values()]))
print(f'\n  五臂 overrides 逐字段比较 (字段全集 {len(keys_union)} 项):')
diff_fields = []
hdr = ' '.join(f'{SHORT[a]:>9}' for a in ARMS)
print(f'    {"field":<24} {hdr}')
for kf in keys_union:
    vals = [odicts[a].get(kf, '<缺>') for a in ARMS]
    same = len(set(map(str, vals))) == 1
    if not same:
        diff_fields.append(kf)
    row = ' '.join(f'{v!s:>9}' for v in vals)
    print(f'    {kf:<24} {row}  {"同" if same else "**差异**"}')
print(f'  => 五臂差异字段 = {diff_fields}   '
      f'{"[单变量, 可归因]" if diff_fields == ["forage_tradeoff"] else "[多变量, 不可归因!]"}')
stepset = sorted({rec['steps'] for rec in D.values()})
print(f'  steps 取值 = {stepset}   {"[一致]" if len(stepset)==1 else "[不一致!]"}')
print(f'  (trait, lineage) 取值 = {sorted({(r["trait"], r["lineage"]) for r in D.values()})}')
print('  [跨 commit] provenance.txt:2 P/Q=9296fd7; :25 M=7937499; :38 L/N=c1f9527。')
print('              git diff --stat 9296fd7 c1f9527 -- underworld/ scripts/ 为空')
print('              => 五臂跑在同一份仿真代码上, 无跨 commit 混杂。')

# ---------------- 工具 ----------------
def cell(arm, s, m):
    v = np.array([D[(arm, s, r)][m] for r in REPS], float)
    return v.mean(), v.std(ddof=1), v

def cmu(arm, m):
    return np.array([cell(arm, s, m)[0] for s in SEEDS])

def boot_ci(d, B=20000, alpha=0.05):
    d = np.asarray(d, float)
    idx = RNG.integers(0, len(d), size=(B, len(d)))
    bs = d[idx].mean(axis=1)
    return float(np.percentile(bs, 100*alpha/2)), float(np.percentile(bs, 100*(1-alpha/2)))

# ---------------- §1 五点剂量-响应表 ----------------
print()
print('=' * 100)
print('§1 五点剂量-响应: 各指标的臂均值 (= 6 个种子格均值的均值), 格均值 = 3 次重复均值')
print('=' * 100)
print(f'  {"metric":<24} ' + ' '.join(f'{SHORT[a]:>11}' for a in ARMS)
      + f' {"L-P":>10} {"N-P":>10} {"单调递减?":>10}')
for m in METRICS:
    means = [cmu(a, m).mean() for a in ARMS]
    mono = all(means[i] >= means[i+1] for i in range(4))
    print(f'  {m:<24} ' + ' '.join(f'{v:>11.5f}' for v in means)
          + f' {means[1]-means[0]:>+10.5f} {means[2]-means[0]:>+10.5f} {("Y" if mono else "n"):>10}')

print('\n  相对中性漂变对照 P(0.0) 的百分比变化:')
for m in PRIMARY + ['bimodality_coefficient', 'mean']:
    base = cmu(P, m).mean()
    print(f'    {m:<24} ' + '  '.join(
        f'{SHORT[a]}={100*(cmu(a,m).mean()-base)/base:+7.2f}%' for a in ARMS))

print('\n  每臂 6 个种子格均值的跨种子 SD (创始者散度尺度):')
for m in PRIMARY + ['bimodality_coefficient', 'mean', 'population', 'carnivore_frac']:
    print(f'    {m:<24} ' + '  '.join(
        f'{SHORT[a]}={cmu(a,m).std(ddof=1):.5f}' for a in ARMS))

# ---------------- §2 逐种子五点曲线 ----------------
print()
print('=' * 100)
print('§2 逐种子五点曲线 (sd 为主, blrt_lr_per_n / bimodality 同法复核)')
print('=' * 100)
for m in ['sd'] + [x for x in PRIMARY if x != 'sd'] + ['bimodality_coefficient']:
    V = {a: cmu(a, m) for a in ARMS}
    print(f'\n--- {m} ---')
    print(f'  {"seed":>4} ' + ' '.join(f'{SHORT[a]:>10}' for a in ARMS)
          + f' | {"单调递减?":>9} {"高于自身P基线的点":>18} {"argmax点":>10}')
    nL = nN = nMono = 0
    above_any = 0
    for i, s in enumerate(SEEDS):
        row = [V[a][i] for a in ARMS]
        mono = all(row[j] >= row[j+1] for j in range(4))
        nMono += mono
        above = [SHORT[a] for j, a in enumerate(ARMS) if j > 0 and row[j] > row[0]]
        above_any += bool(above)
        nL += row[1] > row[0]
        nN += row[2] > row[0]
        amax = SHORT[ARMS[int(np.argmax(row))]]
        print(f'  {s:>4} ' + ' '.join(f'{v:>10.5f}' for v in row)
              + f' | {("Y" if mono else "n"):>9} {(",".join(above) if above else "无"):>18} {amax:>10}')
    print(f'  {"MEAN":>4} ' + ' '.join(f'{V[a].mean():>10.5f}' for a in ARMS))
    print(f'  严格单调递减 (P>=L>=N>=M>=Q) 的种子数 = {nMono}/6')
    print(f'  ** L(0.125) > P(0.0) 的种子数 = {nL}/6 **')
    print(f'  ** N(0.25)  > P(0.0) 的种子数 = {nN}/6 **')
    print(f'  任何一点高于自身 P 基线的种子数 = {above_any}/6')
    print(f'  逐种子 argmax 落在 P 的种子数 = '
          f'{sum(1 for i in range(6) if int(np.argmax([V[a][i] for a in ARMS])) == 0)}/6')
    # run 级(不取格均值)的 L>P / N>P 计数: 同种子 3x3 两两比
    for A, lab in [(L, 'L>P'), (N, 'N>P')]:
        tot = win = 0
        for s in SEEDS:
            for r in REPS:
                for r2 in REPS:
                    tot += 1
                    win += D[(A, s, r)][m] > D[(P, s, r2)][m]
        print(f'  run 级同种子 3x3 {lab}: {win}/{tot} = {100*win/tot:.1f}%')
    allA = np.array([D[(L, s, r)][m] for s in SEEDS for r in REPS] +
                    [D[(N, s, r)][m] for s in SEEDS for r in REPS])
    allP_ = np.array([D[(P, s, r)][m] for s in SEEDS for r in REPS])
    print(f'  单 run 极值: L∪N 最大 {allA.max():.5f} vs P 臂 18 个 run 最大 {allP_.max():.5f} '
          f'/ 最小 {allP_.min():.5f}')

print('\n  逐种子逐臂原始 run 值 (追溯用, 文件名:行号 -> sd):')
for arm in ARMS:
    for s in SEEDS:
        print(f'    ' + '  '.join(f'{SRC[(arm,s,r)]}={D[(arm,s,r)]["sd"]:.5f}' for r in REPS))

# ---------------- §3 L、N 自估格内噪声 ----------------
print()
print('=' * 100)
print('§3 自估格内(复现)噪声: 只用 L 与 N 两臂 36 个 run 的每种子 3 次重复散度')
print('=' * 100)
noise = {}
for m in PRIMARY + ['bimodality_coefficient', 'population', 'carnivore_frac']:
    print(f'\n--- {m} ---')
    ws = []
    for arm in [L, N]:
        for s in SEEDS:
            _, sd_, v = cell(arm, s, m)
            ws.append(sd_)
            print(f'  {SHORT[arm]} seed{s}: reps={np.array2string(v, precision=5)}  格内SD={sd_:.6f}')
    w = np.array(ws)
    pooled = float(np.sqrt((w ** 2).mean()))
    noise[m] = pooled
    noise[('pairdiff', m)] = pooled * np.sqrt(2) / np.sqrt(3)
    print(f'  L+N 池化格内 SD = {pooled:.6f} (dof={2*len(w)}, 范围 {w.min():.6f}..{w.max():.6f})')
    print(f'  => 取 3 次均值后配对差的噪声 sqrt(2)*SD/sqrt(3) = {noise[("pairdiff", m)]:.6f}')
print('\n  [参考] 五臂同口径池化格内 SD (核对量级一致):')
for m in PRIMARY:
    for arm in ARMS:
        w = np.array([cell(arm, s, m)[1] for s in SEEDS])
        print(f'    {m:<18} {SHORT[arm]:<9}: {float(np.sqrt((w**2).mean())):.6f}')

# ---------------- §4 配对检验 ----------------
def paired(m, A, B_, label, ci=True):
    x, y = cmu(A, m), cmu(B_, m)
    d = x - y
    if np.allclose(d, 0):
        print(f'\n--- {m} : {label} --- 全部差为 0, 跳过'); return
    W2, p2 = wilcoxon(x, y, alternative='two-sided')
    lo, hi = boot_ci(d)
    dz = float(d.mean()/d.std(ddof=1)) if d.std(ddof=1) > 0 else float('nan')
    rk = rankdata(np.abs(d))
    rrb = float((rk[d > 0].sum() - rk[d < 0].sum()) / (len(d)*(len(d)+1)/2))
    pdn = noise[('pairdiff', m)]
    ALLP.append((f'{m} {label} 配对Wilcoxon 双侧', p2))
    print(f'\n--- {m} : {label} ---')
    print(f'  {SHORT[A]} 均值 = {x.mean():.5f}   {SHORT[B_]} 均值 = {y.mean():.5f}   '
          f'差 = {d.mean():+.5f} ({100*d.mean()/y.mean():+.2f}% of {SHORT[B_]})')
    print(f'  逐种子差 = {np.array2string(d, precision=5, sign="+")}')
    print(f'  同向种子数: 差>0 {int((d>0).sum())}/6, 差<0 {int((d<0).sum())}/6')
    print(f'  配对 Wilcoxon 双侧: W={W2:.1f}  p={p2:.5f}   [n=6 地板 p=0.03125]')
    print(f'  95% bootstrap CI (B=20000, 配对差均值) = [{lo:+.5f}, {hi:+.5f}]  '
          f'{"不含 0" if lo*hi > 0 else "**含 0**"}')
    print(f'  效应量: Cohen dz = {dz:+.3f}   匹配对秩双列 r_rb = {rrb:+.3f}')
    print(f'  效应/噪声比 (L+N 自估配对差噪声 {pdn:.6f}) = {d.mean()/pdn:+.3f}')
    print(f'  配对差实测 SD = {d.std(ddof=1):.6f} vs 仅噪声预测 {pdn:.6f} '
          f'-> 实测/预测 = {d.std(ddof=1)/pdn:.2f}')

print()
print('=' * 100)
print('§4 配对 Wilcoxon (格均值层面, n=6 配对种子; 双侧最小可达 p = 0.03125)')
print('=' * 100)
for m in PRIMARY:
    paired(m, L, P, 'L(0.125) vs P(0.0)  [主判据]')
    paired(m, N, P, 'N(0.25)  vs P(0.0)  [主判据]')
    paired(m, N, L, 'N(0.25)  vs L(0.125)')
    paired(m, M, L, 'M(0.5)   vs L(0.125)')
    paired(m, M, P, 'M(0.5)   vs P(0.0)  [P3 复核]')
    paired(m, Q, P, 'Q(1.0)   vs P(0.0)  [R2 复核]')

print('\n--- 五臂 Friedman (n=6 区组, k=5 处理) ---')
for m in PRIMARY + ['bimodality_coefficient', 'mean', 'population',
                    'carnivore_frac', 'frugivory_frac', 'death_thirst_frac', 'min_pop', 'n']:
    cols = [cmu(a, m) for a in ARMS]
    chi, pf = friedmanchisquare(*cols)
    ALLP.append((f'{m} 五臂 Friedman (n=6,k=5)', pf))
    ranks = np.array([rankdata(np.array([c[i] for c in cols])) for i in range(6)])
    print(f'  {m:<24} chi2={chi:>8.4f} p={pf:.6f}  平均秩 ' +
          ' '.join(f'{SHORT[a]}={ranks[:, j].mean():.2f}' for j, a in enumerate(ARMS)))

print('\n--- 其余指标的 L-P / N-P 配对检验 (探索性; 全部 p 进 §8; 不做 Bonferroni) ---')
for m in [x for x in METRICS if x not in PRIMARY]:
    for A, lab in [(L, 'L-P'), (N, 'N-P')]:
        x, y = cmu(A, m), cmu(P, m); d = x - y
        if np.allclose(d, 0):
            print(f'  {m:<24} {lab}: 全部差为 0, 跳过'); continue
        W, p = wilcoxon(x, y, alternative='two-sided')
        lo, hi = boot_ci(d)
        ALLP.append((f'{m} {lab} 配对Wilcoxon 双侧', p))
        print(f'  {m:<24} {lab}: {x.mean():>10.4f} vs {y.mean():>10.4f} '
              f'd={d.mean():>+10.4f} CI=[{lo:+.4f},{hi:+.4f}] p={p:.5f} 差>0 {int((d>0).sum())}/6')

print('\n--- 稳健性: 格内用中位数代替均值后重做主检验 ---')
for m in PRIMARY:
    med = {a: np.array([np.median([D[(a, s, r)][m] for r in REPS]) for s in SEEDS]) for a in ARMS}
    for A, lab in [(L, 'L-P'), (N, 'N-P'), (M, 'M-P'), (Q, 'Q-P')]:
        d = med[A] - med[P]
        _, p = wilcoxon(med[A], med[P])
        ALLP.append((f'{m} {lab} 格内中位数口径 配对Wilcoxon 双侧', p))
        print(f'  {m:<18} {lab}: {med[A].mean():.5f} vs {med[P].mean():.5f} '
              f'd={d.mean():+.5f} p={p:.5f} 差>0 {int((d>0).sum())}/6')

# ---------------- §5 直方图 ----------------
print()
print('=' * 100)
print('§5 直方图: 每臂 18 个 run 的 hist 逐桶求和 (40 桶, [0,1])')
print('=' * 100)
H = {a: np.zeros(40) for a in ARMS}
for arm in ARMS:
    for s in SEEDS:
        for r in REPS:
            H[arm] += np.array(D[(arm, s, r)]['hist'], float)
ctr = (np.arange(40) + 0.5) / 40
pcts = {a: 100*H[a]/H[a].sum() for a in ARMS}
peak = max(pcts[a].max() for a in ARMS)
print(f'  {"桶心":>7} | ' + ' '.join(f'{SHORT[a]+"%":>9}' for a in ARMS) + ' | 条形 (共同标度, 峰=%.2f%%)' % peak)
for i in range(40):
    if all(pcts[a][i] == 0 for a in ARMS):
        continue
    bars = ''.join(f'{"#"*int(round(14*pcts[a][i]/peak)):<16}' for a in ARMS)
    print(f'  {ctr[i]:>7.4f} | ' + ' '.join(f'{pcts[a][i]:>9.2f}' for a in ARMS) + f' | {bars}')
print()
for arm in ARMS:
    h = H[arm]; nz = np.nonzero(h)[0]
    loc = [i for i in range(1, 39) if h[i] > h[i-1] and h[i] > h[i+1] and h[i] >= 0.01*h.max()]
    mu_ = float((h*ctr).sum()/h.sum())
    sd_ = float(np.sqrt((h*(ctr-mu_)**2).sum()/h.sum()))
    print(f'{SHORT[arm]:<9} dose={DOSE[arm]:<6} 总计数={h.sum():>8.0f}  '
          f'非零桶=[{ctr[nz[0]]:.4f},{ctr[nz[-1]]:.4f}] ({len(nz)}桶)  合并 mean={mu_:.4f} sd={sd_:.4f}')
    print(f'          局部极大桶心={[round(float(ctr[i]),4) for i in loc]} 计数={[int(h[i]) for i in loc]}')

print()
print('-' * 100)
print('逐 run 双峰核查: bimodality_coefficient 与直方图局部极大 (>0.555 倾向双峰)')
print('-' * 100)
for arm in ARMS:
    nb = sum(1 for s in SEEDS for r in REPS if D[(arm, s, r)]['bimodality_coefficient'] > 0.555)
    bl = [SRC[(arm, s, r)] for s in SEEDS for r in REPS
          if D[(arm, s, r)]['bimodality_coefficient'] > 0.555]
    print(f'  {SHORT[arm]:<9} bimod>0.555 的 run 数 = {nb}/18   {bl if bl else ""}')
print()
print(f'  {"run":<34} {"n":>5} {"sd":>8} {"bimod":>8} 局部极大桶心 (计数>=5%峰值)')
for arm in [L, N]:
    for s in SEEDS:
        for r in REPS:
            rec = D[(arm, s, r)]
            h = np.array(rec['hist'], float)
            loc = [round(float(ctr[i]), 3) for i in range(1, 39)
                   if h[i] > h[i-1] and h[i] > h[i+1] and h[i] >= 0.05*h.max()]
            print(f'  {SRC[(arm,s,r)]:<34} {rec["n"]:>5} {rec["sd"]:>8.4f} '
                  f'{rec["bimodality_coefficient"]:>8.4f} {loc}')

# 低值小簇归属 (R2/P3 口径: 桶心 0.2375-0.2625 vs 谷底 0.2875-0.3375)
LOW, VAL = slice(9, 11), slice(11, 14)
print('\n  低值小簇口径 (桶心 0.2375-0.2625 为簇, 0.2875-0.3375 为谷底) — 逐臂 + 归属:')
for arm in ARMS:
    t_, l_, v_ = H[arm].sum(), H[arm][LOW].sum(), H[arm][VAL].sum()
    contrib = sorted(((SRC[(arm, s, r)], float(np.array(D[(arm, s, r)]['hist'], float)[LOW].sum()))
                      for s in SEEDS for r in REPS), key=lambda t: -t[1])
    top = contrib[0]
    print(f'    {SHORT[arm]:<9}: 低簇 {l_:>6.0f} ({100*l_/t_:>5.2f}%)  谷底 {v_:>6.0f} '
          f'({100*v_/t_:>5.2f}%)  簇/谷={l_/v_ if v_ else float("inf"):>6.2f}')
    print(f'               最大贡献 run={top[0]}: {top[1]:.0f} '
          f'({100*top[1]/l_ if l_ else 0:.1f}% of 该臂低簇); 贡献>0 的 run 数 = '
          f'{sum(1 for _, c in contrib if c > 0)}/18')

# 每臂"最宽"的单 run 是否形成真双峰
print('\n  每臂 sd 最大的单 run (窗口若存在, 应在这里最先露头):')
for arm in ARMS:
    k = max(((s, r) for s in SEEDS for r in REPS), key=lambda t: D[(arm, t[0], t[1])]['sd'])
    rec = D[(arm, k[0], k[1])]
    h = np.array(rec['hist'], float)
    loc = [round(float(ctr[i]), 3) for i in range(1, 39)
           if h[i] > h[i-1] and h[i] > h[i+1] and h[i] >= 0.05*h.max()]
    print(f'    {SHORT[arm]:<9} {SRC[(arm,k[0],k[1])]:<34} sd={rec["sd"]:.5f} '
          f'bimod={rec["bimodality_coefficient"]:.4f} 局部极大={loc}')

# ---------------- §6 护栏 ----------------
print()
print('=' * 100)
print('§6 护栏: carnivore_frac >= 0.05 与 min_pop >= 1, 五臂逐种子')
print('=' * 100)
for arm in ARMS:
    okc = okm = 0
    print(f'\n{SHORT[arm]} (dose={DOSE[arm]}):')
    for s in SEEDS:
        cm, _, cv = cell(arm, s, 'carnivore_frac')
        mm, _, mv = cell(arm, s, 'min_pop')
        pc, pm_ = cm >= 0.05, mm >= 1
        okc += pc; okm += pm_
        print(f'  seed{s}: carn_frac={cm:.4f} {"PASS" if pc else "FAIL"} '
              f'(逐run {np.array2string(cv, precision=4)} 全过={bool((cv>=0.05).all())}) | '
              f'min_pop={mm:.1f} {"PASS" if pm_ else "FAIL"} '
              f'(逐run {np.array2string(mv, precision=0)} 全过={bool((mv>=1).all())})')
    print(f'  => carnivore_frac {okc}/6, min_pop {okm}/6 (格均值口径)')

# ---------------- §7 异常 ----------------
print()
print('=' * 100)
print('§7 异常')
print('=' * 100)
print('载入异常:', bad if bad else '无')
small = [(SRC[k], D[k]['n']) for k in D if D[k]['n'] < 30]
print('n<30 的 run:', small if small else '无')
for arm in ARMS:
    ns = np.array([D[(arm, s, r)]['n'] for s in SEEDS for r in REPS])
    mp = np.array([D[(arm, s, r)]['min_pop'] for s in SEEDS for r in REPS])
    cf = np.array([D[(arm, s, r)]['carnivore_frac'] for s in SEEDS for r in REPS])
    print(f'  {SHORT[arm]:<9}: n=[{ns.min()},{ns.max()}] 中位{np.median(ns):.0f} | '
          f'min_pop=[{mp.min():.0f},{mp.max():.0f}] 灭绝run={int((mp < 1).sum())} | '
          f'carn_frac=[{cf.min():.4f},{cf.max():.4f}] <0.05的run={int((cf < 0.05).sum())}/18')
print('灭绝 run:', [SRC[k] for k in D if D[k]['min_pop'] < 1] or '无')
bp = sorted({round(D[k]['blrt_p'], 4) for k in D})
print(f'blrt_p 逐 run 取值 = {bp} (该字段无效, 仅记录其确实打在地板上)')

# ---------------- §8 全部 p ----------------
print()
print('=' * 100)
print('§8 全部算过的 p 值 (不做 Bonferroni)')
print('=' * 100)
print('  地板: n=6 配对 Wilcoxon 双侧最小可达 p = 0.03125;')
print('        n=6,k=5 Friedman 的 chi2 近似 p 不是精确检验, 小样本下偏乐观。')
for name, p in ALLP:
    print(f'  {name:<66} p={p:.6f}')

# ---------------- §9 判决 + 窗口上界 ----------------
print()
print('=' * 100)
print('§9 判决判定量 + 「若窗口存在其最大可能幅度」上界')
print('=' * 100)
for m in PRIMARY:
    V = {a: cmu(a, m) for a in ARMS}
    means = [V[a].mean() for a in ARMS]
    print(f'\n--- {m} ---')
    print('  五点均值序: ' + ' > '.join(f'{SHORT[a]}={V[a].mean():.5f}' for a in ARMS))
    print(f'  五点是否单调非增 = {all(means[i] >= means[i+1] for i in range(4))}')
    for A, lab in [(L, 'L(0.125)'), (N, 'N(0.25)')]:
        d = V[A] - V[P]
        lo, hi = boot_ci(d)
        _, p = wilcoxon(V[A], V[P])
        print(f'  {lab} - P: 差={d.mean():+.5f}  95%CI=[{lo:+.5f},{hi:+.5f}]  p={p:.5f}  '
              f'差>0 {int((d>0).sum())}/6')
        print(f'      => 「高于中性漂变」的最大可能幅度 (CI 上界) = {hi:+.5f} '
              f'= P 均值的 {100*hi/V[P].mean():+.2f}%')
    # 归一化: L/N 走完 P->Q 全程挤压的比例
    span = V[P].mean() - V[Q].mean()
    for A, lab in [(L, 'L'), (N, 'N'), (M, 'M')]:
        print(f'  {lab} 走完 P->Q 全程挤压的比例 = (P-{lab})/(P-Q) = '
              f'{(V[P].mean()-V[A].mean())/span:.3f}')
print('\n  剂量对数尺度上的形状 (仅描述, 格均值对 log2(dose) 线性拟合, 排除 dose=0):')
for m in PRIMARY:
    xs = np.array([np.log2(DOSE[a]) for a in ARMS[1:]])
    ys = np.array([cmu(a, m).mean() for a in ARMS[1:]])
    b1, b0 = np.polyfit(xs, ys, 1)
    pred = b0 + b1*xs
    print(f'    {m:<18} y={b0:.5f}{b1:+.5f}*log2(dose); 残差={np.array2string(ys-pred, precision=5, sign="+")}'
          f'; P(dose=0) 实测={cmu(P, m).mean():.5f}')

# ---------------- §10 N-vs-P 口径敏感性 + 可检出窗口下界 ----------------
from scipy.stats import mannwhitneyu, norm
print()
print('=' * 100)
print('§10 主判据的口径敏感性 (N 的均值在 P 之下但 4/6 种子在 P 之上, 必须核这一处)')
print('=' * 100)
for m in PRIMARY:
    print(f'\n--- {m} ---')
    stats_ = {'格均值': lambda a, s: np.mean([D[(a, s, r)][m] for r in REPS]),
              '格中位数': lambda a, s: np.median([D[(a, s, r)][m] for r in REPS]),
              '格最大值': lambda a, s: max(D[(a, s, r)][m] for r in REPS),
              '格最小值': lambda a, s: min(D[(a, s, r)][m] for r in REPS)}
    for nm, f in stats_.items():
        row = {a: np.array([f(a, s) for s in SEEDS]) for a in ARMS}
        line = ' '.join(f'{SHORT[a]}={row[a].mean():.5f}' for a in ARMS)
        for A, lab in [(L, 'L-P'), (N, 'N-P')]:
            d = row[A] - row[P]
            _, p = wilcoxon(row[A], row[P])
            ALLP.append((f'{m} {lab} [{nm}口径] 配对Wilcoxon 双侧', p))
            line += f'  | {lab}: d={d.mean():+.5f} p={p:.5f} 差>0 {int((d>0).sum())}/6'
        print(f'  {nm:<8} {line}')
    # leave-one-seed-out
    print('  留一种子敏感性 (逐次剔除一个种子后重算 L-P / N-P 的均值差与同向数):')
    for A, lab in [(L, 'L-P'), (N, 'N-P')]:
        x, y = cmu(A, m), cmu(P, m)
        base = (x - y).mean()
        outs = []
        for i in range(6):
            k = [j for j in range(6) if j != i]
            d = x[k] - y[k]
            outs.append(f's{i}排除:d={d.mean():+.5f},>0 {int((d>0).sum())}/5')
        print(f'    {lab} 全量 d={base:+.5f};  ' + '  '.join(outs))
    # run 级 unpaired (18 vs 18) — 复跑是伪重复, 仅作描述
    for A, lab in [(L, 'L vs P'), (N, 'N vs P')]:
        xa = np.array([D[(A, s, r)][m] for s in SEEDS for r in REPS])
        xp = np.array([D[(P, s, r)][m] for s in SEEDS for r in REPS])
        U, p = mannwhitneyu(xa, xp, alternative='two-sided')
        ALLP.append((f'{m} {lab} run级 MannWhitney 18v18 (伪重复, 仅描述)', p))
        print(f'  run 级 18v18 Mann-Whitney {lab}: 中位 {np.median(xa):.5f} vs {np.median(xp):.5f} '
              f'U={U:.1f} p={p:.5f}  (复跑是同种子伪重复, 该 p 偏乐观, 仅作描述)')

print()
print('=' * 100)
print('§11 本设计能检出多大的「窗口」: 最小可检出效应 (MDE)')
print('=' * 100)
print('  n=6 配对 Wilcoxon 只有在 6/6 同向时才达到 p=0.03125, 所以功效 = P(6 个差同号)。')
print('  取该对比自身实测的配对差 SD 作 sigma, 令 Phi(delta/sigma)^6 = 0.80 求 delta:')
z80 = float(norm.ppf(0.80 ** (1/6)))
print(f'  0.80^(1/6) = {0.80**(1/6):.5f} -> z = {z80:.4f}')
for m in PRIMARY:
    base = cmu(P, m).mean()
    for A, lab in [(L, 'L-P'), (N, 'N-P')]:
        d = cmu(A, m) - cmu(P, m)
        sig = d.std(ddof=1)
        mde = z80 * sig
        print(f'  {m:<18} {lab}: 实测配对差 SD={sig:.5f} -> 80% 功效 MDE={mde:+.5f} '
              f'= P 均值的 {100*mde/base:+.1f}%')
print('  => 小于该幅度的「窄窗口」本设计看不见, 零结果只能排除大于它的窗口。')

print()
print('=' * 100)
print('§12 追加 p 值汇总 (含 §10; 不做 Bonferroni)')
print('=' * 100)
for name, p in ALLP:
    print(f'  {name:<74} p={p:.6f}')
