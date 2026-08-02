"""P3「权衡强度剂量项」: 稳定化挤压是不是随 forage_tradeoff 单调?

回答什么: docs/multispecies_program.md §9.3 P3 (provenance.txt:29-34) ——
          R2 只测了两端: forage_tradeoff 0.0 (P, 基因编译期断开=纯漂变) 与 1.0 (Q,
          最大选择强度), sd 从 0.09508 掉到 0.05866。本轮补中间档 0.5 (M)。
          若 M 落在 P 与 Q 之间 -> 单调, 适应度曲面单峰, 无分化窗口。
          若 M 比 P 更宽 -> 非单调, 弱权衡歧化、强权衡才稳定化 = R2 一步跨过分化窗口。
          若 M ~ Q -> 挤压快速饱和。
          blrt_p 字段无效不得使用 (scripts/probe_trait_dist.py 模块 docstring)。
读哪些文件: outputs/20260803-partition/{P_tradeoff0,M_tradeoff05,Q_tradeoff1}_s{0..5}_r{1..3}.log
          54 个, 每个末尾恰好一行 "JSON {...}"; provenance: 同目录 provenance.txt
          (P/Q 跑在 git 9296fd7, M 跑在 79374996; 两者间 underworld/ 与 scripts/ 无 diff)
输出怎么读: 全部数字由本脚本 stdout 打出, 报告中不得手算。
          §0 载入与三臂归因核对 / §1 逐种子逐臂全量表(格均值 + 格内 3 次 SD)
          / §2 剂量-响应曲线与单调性逐种子判定 / §3 M 臂自估格内噪声
          / §4 配对检验 M-vs-P, M-vs-Q (+ Q-vs-P 复核) / §5 直方图三臂并排 + M 臂双峰归属
          / §6 护栏 / §7 异常 / §8 全部 p 值 / §9 一句话判决
重跑: XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python explorations/20260803-partition/analyze_p3.py
"""
import json, os
import numpy as np
from scipy.stats import wilcoxon, rankdata, friedmanchisquare

DIR = 'outputs/20260803-partition'
ARMS = ['P_tradeoff0', 'M_tradeoff05', 'Q_tradeoff1']
DOSE = {'P_tradeoff0': 0.0, 'M_tradeoff05': 0.5, 'Q_tradeoff1': 1.0}
P, M, Q = ARMS
SEEDS = [0, 1, 2, 3, 4, 5]
REPS = [1, 2, 3]
RNG = np.random.default_rng(20260803)

METRICS = ['sd', 'blrt_lr_per_n', 'bimodality_coefficient', 'mean', 'n',
           'population', 'carnivore_frac', 'min_pop', 'death_thirst_frac',
           'frugivory_frac']
PRIMARY = ['sd', 'blrt_lr_per_n']
ALLP = []

# ---------------- §0 载入 + 三臂归因 ----------------
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

print('=' * 84)
print('§0 载入与三臂归因核对')
print('=' * 84)
print(f'期望 {len(ARMS)*len(SEEDS)*len(REPS)} 个 run, 实际载入 {len(D)} 个')
ov = {}
for k, rec in D.items():
    ov.setdefault(json.dumps(rec['overrides'], sort_keys=True), []).append(k)
for o, ks in sorted(ov.items()):
    arms_here = sorted({k[0] for k in ks})
    print(f'  n={len(ks):2d}  arms={arms_here}\n        overrides={o}')
print(f'  不同 overrides 组合数 = {len(ov)} (期望 3)')
odicts = {arm: None for arm in ARMS}
for o, ks in ov.items():
    for arm in {k[0] for k in ks}:
        odicts[arm] = json.loads(o)
keys_union = sorted(set().union(*[set(d) for d in odicts.values()]))
print(f'\n  三臂 overrides 逐字段比较 (字段全集 {len(keys_union)} 项):')
diff_fields = []
for kf in keys_union:
    vals = [odicts[a].get(kf, '<缺>') for a in ARMS]
    same = len(set(map(str, vals))) == 1
    if not same:
        diff_fields.append(kf)
    print(f'    {kf:<24} P={vals[0]!s:<8} M={vals[1]!s:<8} Q={vals[2]!s:<8} '
          f'{"同" if same else "**差异**"}')
print(f'  => 三臂差异字段 = {diff_fields}   '
      f'{"[单变量, 可归因]" if diff_fields == ["forage_tradeoff"] else "[多变量, 不可归因!]"}')
stepset = sorted({rec['steps'] for rec in D.values()})
print(f'  steps 取值 = {stepset}   {"[一致]" if len(stepset)==1 else "[不一致!]"}')
traits = sorted({(rec['trait'], rec['lineage']) for rec in D.values()})
print(f'  (trait, lineage) 取值 = {traits}')
print('  [跨 commit 注记] provenance.txt:2 P/Q 跑在 git 9296fd7; :25 M 跑在 7937499。')
print('               `git diff --stat 9296fd7 7937499 -- underworld/ scripts/` 为空,')
print('               两 commit 间只动了 docs/ 与 explorations/ => 仿真代码同一版本。')

# ---------------- §1 逐种子全量表 ----------------
def cell(arm, s, m):
    v = np.array([D[(arm, s, r)][m] for r in REPS], float)
    return v.mean(), v.std(ddof=1), v

def cmu(arm, m):
    return np.array([cell(arm, s, m)[0] for s in SEEDS])

print()
print('=' * 84)
print('§1 三臂 x 6 种子格均值表  (格均值 ± 格内 3 次重复的样本 SD, ddof=1)')
print('=' * 84)
for m in METRICS:
    print(f'\n--- {m} ---')
    print(f'{"seed":>4} | {"P(0.0) mean":>14} {"(repSD)":>9} | {"M(0.5) mean":>14} {"(repSD)":>9}'
          f' | {"Q(1.0) mean":>14} {"(repSD)":>9} | {"M-P":>11} {"M-Q":>11}')
    for s in SEEDS:
        pm, ps, _ = cell(P, s, m); mm, ms, _ = cell(M, s, m); qm, qs, _ = cell(Q, s, m)
        print(f'{s:>4} | {pm:>14.5f} {ps:>9.5f} | {mm:>14.5f} {ms:>9.5f} | '
              f'{qm:>14.5f} {qs:>9.5f} | {mm-pm:>+11.5f} {mm-qm:>+11.5f}')
    a, b, c = cmu(P, m), cmu(M, m), cmu(Q, m)
    print(f'{"MEAN":>4} | {a.mean():>14.5f} {"":>9} | {b.mean():>14.5f} {"":>9} | '
          f'{c.mean():>14.5f} {"":>9} | {(b-a).mean():>+11.5f} {(b-c).mean():>+11.5f}')
    print(f'{"sdSEED":>4} | {a.std(ddof=1):>14.5f} {"":>9} | {b.std(ddof=1):>14.5f} {"":>9} | '
          f'{c.std(ddof=1):>14.5f}   <- 格均值层面的跨种子 SD')

print('\n每格 3 次重复的原始值 (追溯用, 文件名:行号 -> 值):')
for m in PRIMARY:
    print(f'  [{m}]')
    for arm in ARMS:
        for s in SEEDS:
            vals = [(SRC[(arm, s, r)], D[(arm, s, r)][m]) for r in REPS]
            print('    ' + '  '.join(f'{f}={v:.5f}' for f, v in vals))

# ---------------- §2 剂量-响应 + 单调性 ----------------
print()
print('=' * 84)
print('§2 剂量-响应: sd 在 tradeoff = 0.0 / 0.5 / 1.0 三点, 逐种子单调性判定')
print('=' * 84)
for m in PRIMARY + ['bimodality_coefficient']:
    a, b, c = cmu(P, m), cmu(M, m), cmu(Q, m)
    print(f'\n--- {m} ---')
    print(f'  格均值: 0.0 -> {a.mean():.5f}   0.5 -> {b.mean():.5f}   1.0 -> {c.mean():.5f}')
    print(f'  相对 P: M {100*(b.mean()-a.mean())/a.mean():+.2f}%   '
          f'Q {100*(c.mean()-a.mean())/a.mean():+.2f}%')
    print(f'\n  {"seed":>4} {"P(0.0)":>10} {"M(0.5)":>10} {"Q(1.0)":>10} | '
          f'{"M>P?":>5} {"M<Q?":>5} {"介于P,Q间?":>10} 形态')
    n_between = n_wider = n_below = 0
    for i, s in enumerate(SEEDS):
        lo, hi = min(a[i], c[i]), max(a[i], c[i])
        between = lo <= b[i] <= hi
        wider = b[i] > a[i]          # 比中性漂变更宽 = 非单调(歧化)证据
        below = b[i] < c[i]          # 比最强权衡还窄 = 另一种非单调(过冲)
        n_between += between; n_wider += wider; n_below += below
        shape = '单调(P>M>Q)' if (a[i] > b[i] > c[i]) else \
                ('非单调: M>P (更宽)' if wider else
                 ('非单调: M<Q (更窄)' if below else '其它序'))
        print(f'  {s:>4} {a[i]:>10.5f} {b[i]:>10.5f} {c[i]:>10.5f} | '
              f'{"Y" if wider else "n":>5} {"Y" if below else "n":>5} '
              f'{"Y" if between else "n":>10} {shape}')
    print(f'  介于 P 与 Q 之间的种子数 = {n_between}/6')
    print(f'  M > P (比中性漂变更宽 = 歧化) 的种子数 = {n_wider}/6')
    print(f'  M < Q (比最强权衡还窄) 的种子数 = {n_below}/6')
    print(f'  => 格均值层面: {"单调 (P>M>Q)" if a.mean()>b.mean()>c.mean() else ("非单调: M 均值高于 P" if b.mean()>a.mean() else ("非单调: M 均值低于 Q" if b.mean()<c.mean() else "其它序"))}')
    # 三臂 Friedman (n=6, k=3)
    chi, pf = friedmanchisquare(a, b, c)
    ALLP.append((f'{m} 三臂 Friedman (n=6,k=3)', pf))
    print(f'  三臂 Friedman: chi2={chi:.4f} p={pf:.5f}  '
          f'(n=6,k=3 的最小可达 p 见 §8 说明)')

# ---------------- §3 M 臂自估格内噪声 ----------------
print()
print('=' * 84)
print('§3 自估格内噪声: 只用 M 臂 18 个 run (每种子 3 次重复的散度)')
print('=' * 84)
noise = {}
for m in PRIMARY + ['bimodality_coefficient', 'population', 'carnivore_frac']:
    print(f'\n--- {m} ---')
    within = []
    for s in SEEDS:
        _, sd_, v = cell(M, s, m)
        within.append(sd_)
        print(f'  M seed{s}: reps={np.array2string(v, precision=5)}  格内SD={sd_:.6f}')
    w = np.array(within)
    pooled = float(np.sqrt((w ** 2).mean()))
    noise[m] = pooled
    print(f'  M 臂池化格内 SD = sqrt(mean var) = {pooled:.6f}  (dof={2*len(SEEDS)}, '
          f'格内SD 范围 {w.min():.6f}..{w.max():.6f})')
    print(f'  => 取 3 次均值后, 配对差的噪声 sqrt(2)*SD/sqrt(3) = '
          f'{pooled*np.sqrt(2)/np.sqrt(3):.6f}')
    noise[('pairdiff', m)] = pooled * np.sqrt(2) / np.sqrt(3)
# 供参考: 另两臂的池化格内 SD
print('\n  [参考] 另两臂同口径池化格内 SD (报告用 M 臂的那个, 此处仅核对量级一致):')
for m in PRIMARY:
    for arm in [P, Q]:
        w = np.array([cell(arm, s, m)[1] for s in SEEDS])
        print(f'    {arm} {m}: {float(np.sqrt((w**2).mean())):.6f}')

# ---------------- §4 配对检验 ----------------
def boot_ci(d, B=20000, alpha=0.05):
    d = np.asarray(d, float)
    idx = RNG.integers(0, len(d), size=(B, len(d)))
    bs = d[idx].mean(axis=1)
    return float(np.percentile(bs, 100*alpha/2)), float(np.percentile(bs, 100*(1-alpha/2)))

def paired(m, A, B_, label):
    x, y = cmu(A, m), cmu(B_, m)      # 差 = A - B_
    d = x - y
    W2, p2 = wilcoxon(x, y, alternative='two-sided')
    lo, hi = boot_ci(d)
    dz = float(d.mean()/d.std(ddof=1)) if d.std(ddof=1) > 0 else float('nan')
    rk = rankdata(np.abs(d))
    rrb = float((rk[d > 0].sum() - rk[d < 0].sum()) / (len(d)*(len(d)+1)/2))
    pdn = noise[('pairdiff', m)]
    ALLP.append((f'{m} {label} 配对Wilcoxon 双侧', p2))
    print(f'\n--- {m} : {label} ---')
    print(f'  {A} 均值 = {x.mean():.5f}   {B_} 均值 = {y.mean():.5f}   差 = {d.mean():+.5f} '
          f'({100*d.mean()/y.mean():+.2f}% of {B_})')
    print(f'  逐种子差 = {np.array2string(d, precision=5, sign="+")}')
    print(f'  同向种子数: 差>0 {int((d>0).sum())}/6, 差<0 {int((d<0).sum())}/6')
    print(f'  配对 Wilcoxon 双侧: W={W2:.1f}  p={p2:.5f}   [n=6 地板 p=0.03125]')
    print(f'  95% bootstrap CI (B=20000, 配对差均值) = [{lo:+.5f}, {hi:+.5f}]  '
          f'{"不含 0" if lo*hi > 0 else "**含 0**"}')
    print(f'  效应量: Cohen dz = {dz:+.3f}   匹配对秩双列 r_rb = {rrb:+.3f}')
    print(f'  效应/噪声比 (M 臂自估配对差噪声 {pdn:.6f}) = {d.mean()/pdn:+.3f}')
    print(f'  配对差实测 SD = {d.std(ddof=1):.6f}  vs 仅噪声预测 {pdn:.6f}  '
          f'-> 实测/预测 = {d.std(ddof=1)/pdn:.2f}')

print()
print('=' * 84)
print('§4 配对 Wilcoxon (格均值层面, n=6 配对种子; 双侧最小可达 p = 0.03125)')
print('=' * 84)
for m in PRIMARY:
    paired(m, M, P, 'M(0.5) vs P(0.0)')
    paired(m, M, Q, 'M(0.5) vs Q(1.0)')
    paired(m, Q, P, 'Q(1.0) vs P(0.0)  [R2 结论复核]')

print('\n--- 其余指标的配对检验 (探索性, 全部算过的 p 都进 §8; 不做 Bonferroni) ---')
for m in [x for x in METRICS if x not in PRIMARY]:
    for A, B_, lab in [(M, P, 'M-P'), (M, Q, 'M-Q'), (Q, P, 'Q-P')]:
        x, y = cmu(A, m), cmu(B_, m); d = x - y
        if np.allclose(d, 0):
            print(f'  {m:<24} {lab}: 全部差为 0, 跳过'); continue
        W, p = wilcoxon(x, y, alternative='two-sided')
        lo, hi = boot_ci(d)
        ALLP.append((f'{m} {lab} 配对Wilcoxon 双侧', p))
        print(f'  {m:<24} {lab}: {x.mean():>10.4f} vs {y.mean():>10.4f} '
              f'd={d.mean():>+10.4f} CI=[{lo:+.4f},{hi:+.4f}] p={p:.5f} '
              f'差>0 {int((d>0).sum())}/6')

# ---------------- §5 直方图 ----------------
print()
print('=' * 84)
print('§5 直方图: 每臂 18 个 run 的 hist 逐桶求和 (40 桶, [0,1])')
print('=' * 84)
H = {}
for arm in ARMS:
    h = np.zeros(40)
    for s in SEEDS:
        for r in REPS:
            h += np.array(D[(arm, s, r)]['hist'], float)
    H[arm] = h
ctr = (np.arange(40) + 0.5) / 40
print(f'\n三臂并排 (占比%, 只列任一臂非零的桶):')
print(f'  {"桶心":>7} | {"P(0.0)%":>8} {"M(0.5)%":>8} {"Q(1.0)%":>8} | '
      f'{"P 条形":<22}{"M 条形":<22}{"Q 条形":<22}')
pcts = {a: 100*H[a]/H[a].sum() for a in ARMS}
peak = max(pcts[a].max() for a in ARMS)
for i in range(40):
    if all(pcts[a][i] == 0 for a in ARMS):
        continue
    bars = ''.join(f'{"#"*int(round(20*pcts[a][i]/peak)):<22}' for a in ARMS)
    print(f'  {ctr[i]:>7.4f} | {pcts[P][i]:>8.2f} {pcts[M][i]:>8.2f} {pcts[Q][i]:>8.2f} | {bars}')
for arm in ARMS:
    h = H[arm]; nz = np.nonzero(h)[0]
    loc = [i for i in range(1, 39) if h[i] > h[i-1] and h[i] > h[i+1] and h[i] >= 0.01*h.max()]
    mu_ = float((h*ctr).sum()/h.sum())
    print(f'\n{arm} (dose={DOSE[arm]}): 总计数={h.sum():.0f}  '
          f'非零桶范围=[{ctr[nz[0]]:.4f},{ctr[nz[-1]]:.4f}] ({len(nz)} 桶)')
    print(f'  局部极大桶心 = {[round(float(ctr[i]),4) for i in loc]}  计数 {[int(h[i]) for i in loc]}')
    print(f'  合并分布 mean={mu_:.4f} '
          f'sd={float(np.sqrt((h*(ctr-mu_)**2).sum()/h.sum())):.4f}')

# M 臂双峰归属: 对每个局部极大之外的"分离簇"做 run 归属
print()
print('-' * 84)
print('M 臂双峰核查: 逐 run 的 bimodality_coefficient 与直方图局部极大')
print('-' * 84)
print(f'  {"run":<32} {"n":>5} {"sd":>8} {"bimod":>8} {">0.555?":>8} 局部极大桶心 (计数>=5%峰值)')
for s in SEEDS:
    for r in REPS:
        rec = D[(M, s, r)]
        h = np.array(rec['hist'], float)
        loc = [round(float(ctr[i]), 3) for i in range(1, 39)
               if h[i] > h[i-1] and h[i] > h[i+1] and h[i] >= 0.05*h.max()]
        print(f'  {SRC[(M,s,r)]:<32} {rec["n"]:>5} {rec["sd"]:>8.4f} '
              f'{rec["bimodality_coefficient"]:>8.4f} '
              f'{"Y" if rec["bimodality_coefficient"]>0.555 else "n":>8} {loc}')
print(f'\n  M 臂 bimod>0.555 的 run 数 = '
      f'{sum(1 for s in SEEDS for r in REPS if D[(M,s,r)]["bimodality_coefficient"]>0.555)}/18')
for arm in [P, Q]:
    print(f'  {arm} 臂 bimod>0.555 的 run 数 = '
          f'{sum(1 for s in SEEDS for r in REPS if D[(arm,s,r)]["bimodality_coefficient"]>0.555)}/18')

# R2 那个低值小簇在三臂的占比 (桶 0.2375/0.2625 vs 谷底 0.2875-0.3375)
LOW, VAL = slice(9, 11), slice(11, 14)
print('\n  R2 §9 的低值小簇口径 (桶心 0.2375-0.2625 vs 谷底 0.2875-0.3375):')
for arm in ARMS:
    t_, l_, v_ = H[arm].sum(), H[arm][LOW].sum(), H[arm][VAL].sum()
    print(f'    {arm}: 低簇 {l_:.0f} ({100*l_/t_:.2f}%)  谷底 {v_:.0f} ({100*v_/t_:.2f}%)  '
          f'簇/谷={l_/v_ if v_ else float("inf"):.2f}')
    contrib = sorted(((SRC[(arm,s,r)], float(np.array(D[(arm,s,r)]['hist'],float)[LOW].sum()))
                      for s in SEEDS for r in REPS), key=lambda t: -t[1])
    top = contrib[0]
    print(f'      最大贡献 run = {top[0]}: {top[1]:.0f} '
          f'({100*top[1]/l_ if l_ else 0:.1f}% of 该臂低簇); '
          f'贡献>0 的 run 数 = {sum(1 for _,c in contrib if c>0)}/18')

# ---------------- §6 护栏 ----------------
print()
print('=' * 84)
print('§6 护栏: carnivore_frac >= 0.05 与 min_pop >= 1, 三臂逐种子')
print('=' * 84)
for arm in ARMS:
    okc = okm = 0
    print(f'\n{arm} (dose={DOSE[arm]}):')
    for s in SEEDS:
        cm, _, cv = cell(arm, s, 'carnivore_frac')
        mm, _, mv = cell(arm, s, 'min_pop')
        pc, pm_ = cm >= 0.05, mm >= 1
        okc += pc; okm += pm_
        print(f'  seed{s}: carn_frac 格均值={cm:.4f} {"PASS" if pc else "FAIL"} '
              f'(逐 run {np.array2string(cv, precision=4)} 全过={bool((cv>=0.05).all())}) | '
              f'min_pop 格均值={mm:.1f} {"PASS" if pm_ else "FAIL"} '
              f'(逐 run {np.array2string(mv, precision=0)} 全过={bool((mv>=1).all())})')
    print(f'  => carnivore_frac {okc}/6, min_pop {okm}/6 (格均值口径)')

# ---------------- §7 异常 ----------------
print()
print('=' * 84)
print('§7 异常')
print('=' * 84)
print('载入异常:', bad if bad else '无')
small = [(SRC[k], D[k]['n']) for k in D if D[k]['n'] < 30]
print('n<30 的 run:', small if small else '无')
for arm in ARMS:
    ns = np.array([D[(arm,s,r)]['n'] for s in SEEDS for r in REPS])
    mp = np.array([D[(arm,s,r)]['min_pop'] for s in SEEDS for r in REPS])
    cf = np.array([D[(arm,s,r)]['carnivore_frac'] for s in SEEDS for r in REPS])
    print(f'  {arm}: n=[{ns.min()},{ns.max()}] 中位{np.median(ns):.0f} | '
          f'min_pop=[{mp.min():.0f},{mp.max():.0f}] 灭绝run={int((mp<1).sum())} | '
          f'carn_frac=[{cf.min():.4f},{cf.max():.4f}] <0.05的run={int((cf<0.05).sum())}/18')
print('灭绝 run:', [SRC[k] for k in D if D[k]['min_pop'] < 1] or '无')
bp = sorted({round(D[k]['blrt_p'], 4) for k in D})
print(f'blrt_p 逐 run 取值 = {bp} (该字段无效, 仅记录其确实打在地板上)')

# ---------------- §8 全部 p ----------------
print()
print('=' * 84)
print('§8 全部算过的 p 值 (不做 Bonferroni)')
print('=' * 84)
print('  地板: n=6 配对 Wilcoxon 双侧最小可达 p = 0.03125;')
print('        n=6,k=3 Friedman 的 chi2 近似 p 不是精确检验, 小样本下偏乐观。')
for name, p in ALLP:
    print(f'  {name:<56} p={p:.6f}')

# ---------------- §9 判决 ----------------
print()
print('=' * 84)
print('§9 一句话判决所需的判定量')
print('=' * 84)
for m in PRIMARY:
    a, b, c = cmu(P, m), cmu(M, m), cmu(Q, m)
    _, p_mp = wilcoxon(b, a); _, p_mq = wilcoxon(b, c)
    order = 'P > M > Q (单调)' if a.mean() > b.mean() > c.mean() else \
            ('M > P (非单调, 弱权衡更宽)' if b.mean() > a.mean() else
             ('M < Q (非单调, 中间档过冲)' if b.mean() < c.mean() else '其它'))
    print(f'  {m}: 均值序 {a.mean():.5f} / {b.mean():.5f} / {c.mean():.5f} => {order}')
    print(f'    M-P p={p_mp:.5f} (差 {(b-a).mean():+.5f}, 差>0 {int((b>a).sum())}/6);  '
          f'M-Q p={p_mq:.5f} (差 {(b-c).mean():+.5f}, 差>0 {int((b>c).sum())}/6)')
    frac = (a.mean()-b.mean())/(a.mean()-c.mean()) if a.mean() != c.mean() else float('nan')
    print(f'    M 走完 P->Q 全程挤压的比例 = (P-M)/(P-Q) = {frac:.3f} '
          f'(0=贴 P, 1=贴 Q, >1=过冲, <0=反向)')

# ---------------- §10 重复的性质 + 稳健性 ----------------
from scipy.stats import chi2 as _chi2
print()
print('=' * 84)
print('§10 「3 次重复」到底是什么 + 稳健性核对')
print('=' * 84)
hdr_seeds = {}
for arm in ARMS:
    for s in SEEDS:
        for r in REPS:
            first = open(f'{DIR}/{arm}_s{s}_r{r}.log').readline()
            tok = [t for t in first.split() if t.startswith('seed=')]
            hdr_seeds[(arm, s, r)] = tok[0].split('=')[1] if tok else '?'
same_within_cell = all(len({hdr_seeds[(a, s, r)] for r in REPS}) == 1 for a in ARMS for s in SEEDS)
print(f'  各 log 首行 seed= 与文件名 s<seed> 一致, 且同一格 3 次重复 seed 相同 = {same_within_cell}')
print(f'  例: M_s0 三次重复的 header seed = '
      f'{[hdr_seeds[(M,0,r)] for r in REPS]}; joblist.txt 第二列同样是 seed。')
print('  => 3 次「重复」是同种子同配置的复跑, 差异只能来自 GPU 原子 scatter-add 重排')
print('     (CLAUDE.md「Determinism is not bit-exact on GPU」)。因此格内 SD 度量的是')
print('     混沌放大的复现噪声, 不是创始者差异; 跨种子 SD 才是创始者差异。')

print('\n  格内(复现)噪声 vs 跨种子(创始者)散度 — 同一指标两把尺子:')
for m in PRIMARY + ['bimodality_coefficient']:
    for arm in ARMS:
        w = np.array([cell(arm, s, m)[1] for s in SEEDS])
        pooled = float(np.sqrt((w ** 2).mean()))
        across = cmu(arm, m).std(ddof=1)
        print(f'    {m:<24} {arm:<14} 池化格内SD={pooled:.5f}  格均值跨种子SD={across:.5f}  '
              f'比={pooled/across if across else float("nan"):.2f}')

print('\n  配对差为何比噪声模型预测的紧 — 卡方核对 (自由度 5):')
for m in PRIMARY:
    for A, B_, lab in [(M, P, 'M-P'), (M, Q, 'M-Q'), (Q, P, 'Q-P')]:
        d = cmu(A, m) - cmu(B_, m)
        sA = float(np.sqrt((np.array([cell(A, s, m)[1] for s in SEEDS]) ** 2).mean()))
        sB = float(np.sqrt((np.array([cell(B_, s, m)[1] for s in SEEDS]) ** 2).mean()))
        pred = np.sqrt(sA**2/3 + sB**2/3)
        stat = 5 * d.std(ddof=1)**2 / pred**2
        pl = float(_chi2.cdf(stat, 5)); pu = float(1 - _chi2.cdf(stat, 5))
        ALLP.append((f'{m} {lab} 配对差方差 vs 噪声模型 chi2 下尾', pl))
        print(f'    {m:<18} {lab}: d 实测SD={d.std(ddof=1):.5f} 预测SD={pred:.5f} '
              f'比={d.std(ddof=1)/pred:.2f}  chi2(5)={stat:.3f} '
              f'下尾p={pl:.4f} 上尾p={pu:.4f}')

print('\n  run 级 3x3 一致性 (同种子内, A 的 3 个 run 逐一对 B 的 3 个 run 比大小):')
for m in PRIMARY:
    for A, B_, lab in [(M, P, 'M<P'), (M, Q, 'M<Q'), (Q, P, 'Q<P')]:
        tot = win = 0
        for s in SEEDS:
            for r in REPS:
                for r2 in REPS:
                    tot += 1
                    win += D[(A, s, r)][m] < D[(B_, s, r2)][m]
        print(f'    {m:<18} {lab}: {win}/{tot} = {100*win/tot:.1f}%')

print('\n  稳健性: 格内用中位数代替均值后重做主检验')
for m in PRIMARY:
    med = {a: np.array([np.median([D[(a, s, r)][m] for r in REPS]) for s in SEEDS]) for a in ARMS}
    for A, B_, lab in [(M, P, 'M-P'), (M, Q, 'M-Q'), (Q, P, 'Q-P')]:
        d = med[A] - med[B_]
        _, p = wilcoxon(med[A], med[B_])
        ALLP.append((f'{m} {lab} 格内中位数口径 配对Wilcoxon 双侧', p))
        print(f'    {m:<18} {lab}: {med[A].mean():.5f} vs {med[B_].mean():.5f} '
              f'd={d.mean():+.5f} p={p:.5f} 差<0 {int((d<0).sum())}/6')

print('\n  剂量三点是否落在一条直线上 (格均值对 dose 的线性拟合, 仅描述):')
for m in PRIMARY:
    x = np.array([DOSE[a] for a in ARMS]); y = np.array([cmu(a, m).mean() for a in ARMS])
    b1, b0 = np.polyfit(x, y, 1)
    pred_mid = b0 + b1 * 0.5
    print(f'    {m:<18} 线性拟合 y={b0:.5f}{b1:+.5f}*dose; dose=0.5 的线性预测={pred_mid:.5f}, '
          f'实测 M={y[1]:.5f}, 实测-线性={y[1]-pred_mid:+.5f} '
          f'({"凹(前段挤压更快)" if y[1] < pred_mid else "凸(后段挤压更快)"})')

print()
print('=' * 84)
print('§11 追加 p 值汇总 (含 §10; 不做 Bonferroni)')
print('=' * 84)
for name, p in ALLP:
    print(f'  {name:<62} p={p:.6f}')
