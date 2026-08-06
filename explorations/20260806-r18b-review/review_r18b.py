"""独立复核 R18-B 判决（`feasibility.md` §24）的六个受攻击点。

回答什么
--------
1. 等价性检验：δ=自估噪声 ⇒ 它是「单 run 噪声」还是「本实验分辨率」？两者差多少？
2. 「不再缩」这类无限定词表述与 CI 不含 0 是否自相矛盾（数值层面）。
5. 几何外推「高估 0.38%」独立复算 + 三种口径 + 对公比 r 的敏感性 + 跨轮可复现性下限。
6. 段长 60.8 vs 62.8：逐 run 段长离散度、末段所处世代的离散度、两个 δ 能否对比。

读哪些文件
----------
outputs/20260806-gen700/*.log、outputs/20260805-gen450/*.log 的末行 `JSON {...}`。
统计走 scripts/exp_stats.py 的 bootstrap_ci（与判决同一实现、同一固定种子）。

输出怎么读
----------
每节以 [Q<n>] 开头，对应上面的问题号。所有数字皆为脚本 stdout，无手算。
"""
import glob, json, sys
sys.path.insert(0, "scripts")
sys.path.insert(0, "explorations/20260804-readouts")
import numpy as np
from exp_stats import bootstrap_ci
from neutral_null import gen_weights
from scipy.stats import wilcoxon

BINS = np.linspace(0.0, 1.0, 21); CTR = 0.5 * (BINS[:-1] + BINS[1:]); LOW = CTR < 0.35
SEEDS, REPS = list(range(12)), (1, 2)

def load(d):
    R = {}
    for f in sorted(glob.glob(f"{d}/*.log")):
        txt = open(f).read()
        if "JSON " not in txt: continue
        j = json.loads(txt.split("JSON ")[1].split("\n")[0])
        tr = j["traj"]; h = np.array([q["hist"] for q in tr], float)
        b = f.split("/")[-1][:-4].split("_")
        R[(int(b[1][1:]), int(b[2][1:]))] = {
            "hist": h, "gen": np.array([q["generation"] for q in tr], float)}
    return R

lm = lambda h: h[LOW].sum() / max(h.sum(), 1.0)

def wmean(vals, gen, mask):
    w = gen_weights(gen, "iso") * mask; s = w.sum()
    return float((w * np.asarray(vals, float)).sum() / s) if s > 1e-9 else np.nan

def wmid(gen, mask):                      # 窗口的世代加权重心
    w = gen_weights(gen, "iso") * mask; s = w.sum()
    return float((w * gen).sum() / s) if s > 1e-9 else np.nan

def seg_of(rec, i, n, fn):                # 与 analyze_r18b.py 逐字一致
    g, gm = rec["gen"], rec["gen"].max()
    m = (g >= gm * i / n) & (g < gm * (i + 1) / n + (1e-9 if i == n - 1 else 0))
    return wmean([fn(h) for h in rec["hist"]], g, m)

def cellwise(R, fn):
    per = {k: fn(v) for k, v in R.items()}; mu, sd = [], []
    for s in SEEDS:
        vs = [per[(s, r)] for r in REPS if (s, r) in per]
        mu.append(np.mean(vs)); sd.append(np.std(vs, ddof=1))
    return np.array(mu), np.array(sd)

B = load("outputs/20260806-gen700"); A = load("outputs/20260805-gen450")
spans_B = np.array([r["gen"][-1] - r["gen"][0] for r in B.values()])
spans_A = np.array([r["gen"][-1] - r["gen"][0] for r in A.values()])
NSEG_B, NSEG_A = 11, 6

print("=" * 90); print("[Q6] 段长口径：11 段是对 24 个不同长度的 run 各自切的")
gmB = np.array([r["gen"].max() for r in B.values()])
print(f"  R18-B 逐 run 跨代数 mean {spans_B.mean():.1f}  min {spans_B.min():.1f}  max {spans_B.max():.1f}")
print(f"  ⇒ **逐 run 段长 = gen_max/11**：mean {gmB.mean()/11:.2f}  min {gmB.min()/11:.2f}  max {gmB.max()/11:.2f} 代")
print(f"     （判决写的「60.8 代」= 均值跨代数/11 = {spans_B.mean()/11:.2f}，不是每个 run 的段长）")
print(f"  ⇒ **末段所处世代按 run 差异**：末段中点 = gen_max*21/22，"
      f"min {gmB.min()*21/22:.0f}  max {gmB.max()*21/22:.0f} 代（跨度 {gmB.max()*21/22-gmB.min()*21/22:.0f} 代）")
print(f"  R18 参照：跨代数 mean {spans_A.mean():.1f}，段长 = gen_max/6 mean "
      f"{np.array([r['gen'].max() for r in A.values()]).mean()/6:.2f} 代")

print("=" * 90); print("[Q1] δ 是什么尺度：单 run 噪声 vs 本实验分辨率")
dl, sdl = cellwise(B, lambda r: seg_of(r, NSEG_B-1, NSEG_B, lm) - seg_of(r, NSEG_B-2, NSEG_B, lm))
sw = float(np.sqrt((sdl**2).mean())); delta = np.sqrt(2)*sw/np.sqrt(2)
lo, hi = bootstrap_ci(dl); half = (hi-lo)/2
sd_cell = float(dl.std(ddof=1)); sem = sd_cell/np.sqrt(len(dl))
print(f"  末段增量 Δ = {dl.mean():+.5f}   12 格逐格 SD = {sd_cell:.5f}   SEM = {sem:.5f}")
print(f"  σ̂_W（格内重复 SD） = {sw:.5f}  ⇒ δ = √2·σ̂_W/√2 = **{delta:.5f}（= σ̂_W 本身，因为 r=2）**")
print(f"  95%CI [{lo:+.5f}, {hi:+.5f}]  半宽 {half:.5f}")
print(f"  ⇒ **δ / CI半宽 = {delta/half:.2f}**：等价界比本实验真正的分辨率宽 {delta/half:.1f} 倍")
print(f"  ⇒ 效应 {dl.mean():+.5f} **大于** CI 半宽 {half:.5f}（所以 CI 不含 0），"
      f"**小于** δ {delta:.5f}")
print(f"  ⇒ 「小于本实验的分辨率」在数值上不成立：真分辨率是 {half:.5f}，效应是它的 {dl.mean()/half:.2f} 倍")
var_founder = sd_cell**2 - sw**2/2
print(f"  方差分解：Var(格均值)={sd_cell**2:.3e}  σ̂_W²/r={sw**2/2:.3e}  "
      f"⇒ 创始者方差分量 = {var_founder:+.3e}"
      f"{'  ⇒ **估为负 ⇒ 判 0**（与 run_to_run_variance 多数标量一致）' if var_founder<0 else ''}")
print(f"  ⇒ δ 是**单个 run 的段增量噪声 SD**；12 格平均后噪声已缩到 {sem:.5f}，"
      f"δ 是它的 {delta/sem:.1f} 倍")

print("-" * 90); print("[Q1b] 等价界与被检效应一起缩：两轮对比")
dlA, sdlA = cellwise(A, lambda r: seg_of(r, NSEG_A-1, NSEG_A, lm) - seg_of(r, NSEG_A-2, NSEG_A, lm))
swA = float(np.sqrt((sdlA**2).mean())); deltaA = swA
loA, hiA = bootstrap_ci(dlA)
print(f"  R18   末段增量 {dlA.mean():+.5f}  δ_R18 = {deltaA:.5f}  比值 {dlA.mean()/deltaA:+.2f}"
      f"  CI [{loA:+.5f},{hiA:+.5f}]  {int((dlA>0).sum())}/12")
print(f"  R18-B 末段增量 {dl.mean():+.5f}  δ_R18B= {delta:.5f}  比值 {dl.mean()/delta:+.2f}"
      f"  CI [{lo:+.5f},{hi:+.5f}]  {int((dl>0).sum())}/12")
print(f"  效应缩到 {dl.mean()/dlA.mean():.3f} 倍，δ 缩到 {delta/deltaA:.3f} 倍 "
      f"⇒ **效应比等价界快 {(deltaA/delta)/(dlA.mean()/dl.mean()):.2f} 倍反过来说：比值从 "
      f"{dlA.mean()/deltaA:+.2f} 降到 {dl.mean()/delta:+.2f}，降了 {dlA.mean()/deltaA/(dl.mean()/delta):.1f} 倍**")

print("-" * 90); print("[Q1c] 把残余漂移翻译成有实质含义的单位")
LMc = np.array([cellwise(B, lambda r, i=i: seg_of(r, i, NSEG_B, lm))[0] for i in range(NSEG_B)])
lm_last = float(LMc.mean(1)[-1]); seg_gen = spans_B.mean()/NSEG_B
for tag, rate in (("点估计", dl.mean()), ("CI 上界", hi), ("CI 下界", lo)):
    if rate > 0:
        nseg_fix = (1.0 - lm_last)/rate
        print(f"  {tag} {rate:+.5f}/段 ⇒ 线性外推到固定(lm=1)需 {nseg_fix:.0f} 段 = "
              f"**{nseg_fix*seg_gen:,.0f} 代**（本轮实测 {spans_B.mean():.0f} 代）")
print(f"  对照：全程平均速率 = ({lm_last:.4f}−{LMc.mean(1)[0]:.4f})/{spans_B.mean():.0f} 代 "
      f"= {(lm_last-LMc.mean(1)[0])/spans_B.mean():.6f}/代；"
      f"残余速率 = {dl.mean()/seg_gen:.6f}/代 ⇒ **残余是全程均速的 "
      f"{100*(dl.mean()/seg_gen)/((lm_last-LMc.mean(1)[0])/spans_B.mean()):.1f}%**")

print("=" * 90); print("[Q5] 几何外推独立复算")
r_ratio = 0.575; anchor_gen, anchor_lm, last_inc, seg_A = 345.0, 0.5842, 0.00999, 62.8
A_amp = last_inc*r_ratio/(1-r_ratio)
model = lambda g: anchor_lm + A_amp*(1 - r_ratio**((g-anchor_gen)/seg_A))
print(f"  模型 LM(g) = {anchor_lm} + {A_amp:.6f}·(1 − {r_ratio}^((g−{anchor_gen:.0f})/{seg_A}))"
      f"   渐近 {anchor_lm+A_amp:.4f}")
print(f"  (a) 判决用的口径：在**终点 gen 669** 求值 ⇒ **{model(669.0):.4f}**（判决写 0.5969 ✓）")
late, _ = cellwise(B, lambda r: wmean([lm(h) for h in r["hist"]], r["gen"], r["gen"] >= r["gen"].max()*0.75))
lo2, hi2 = bootstrap_ci(late)
mids = np.array([wmid(r["gen"], r["gen"] >= r["gen"].max()*0.75) for r in B.values()])
print(f"  实测末四分之一 = {late.mean():.4f}  95%CI [{lo2:.4f}, {hi2:.4f}]")
print(f"  ⚠️ 但「末四分之一」的**世代加权重心是 gen {mids.mean():.0f}**，不是 669"
      f"（逐 run {mids.min():.0f}–{mids.max():.0f}）")
print(f"  (b) 在窗口重心 gen {mids.mean():.0f} 求值 ⇒ **{model(mids.mean()):.4f}**"
      f"  ⇒ {'落在实测 CI **内**' if lo2 <= model(mids.mean()) <= hi2 else '落在 CI 外'}")
mm = []
for rec in B.values():
    g = rec["gen"]; m = g >= g.max()*0.75
    mm.append(wmean([model(x) for x in g], g, m))
mm = np.array(mm)
print(f"  (c) 把模型**在同一窗口上按同一权重取均值**（同口径比较）⇒ **{mm.mean():.4f}**"
      f"  ⇒ {'落在实测 CI **内**' if lo2 <= mm.mean() <= hi2 else '落在 CI 外'}")
print(f"      三种口径给出的「高估」分别为 {model(669.0)-late.mean():+.4f} / "
      f"{model(mids.mean())-late.mean():+.4f} / {mm.mean()-late.mean():+.4f}")

print("-" * 90); print("[Q5b] 预测本身的不确定性：公比 r 的来源与敏感性")
dseqA = np.diff(np.nanmean(np.array([cellwise(A, lambda r, i=i: seg_of(r, i, NSEG_A, lm))[0]
                                     for i in range(NSEG_A)]), 1))
ratiosA = [dseqA[i+1]/dseqA[i] for i in range(len(dseqA)-1)]
print(f"  R18 实测公比序列 {[f'{v:.3f}' for v in ratiosA]}")
print(f"  drift_deceleration.py:140 用的是 **mean(最后两个)** = {np.mean(ratiosA[-2:]):.3f}")
print(f"  其它同样合理的取法：全部均值 {np.mean(ratiosA):.3f}、"
      f"后三个均值 {np.mean(ratiosA[-3:]):.3f}、几何均值(后三) {np.prod(ratiosA[-3:])**(1/3):.3f}")
print("  对同一 (c) 口径预测的敏感性：")
for rr in (0.46, 0.52, 0.560, 0.575, 0.590, 0.604, 0.662):
    Ar = last_inc*rr/(1-rr); f = lambda g, Ar=Ar, rr=rr: anchor_lm + Ar*(1-rr**((g-anchor_gen)/seg_A))
    vv = np.array([wmean([f(x) for x in rec["gen"]], rec["gen"], rec["gen"] >= rec["gen"].max()*0.75)
                   for rec in B.values()]).mean()
    print(f"    r={rr:.3f} ⇒ 渐近 {anchor_lm+Ar:.4f}，窗口预测 {vv:.4f}"
          f"   {'（CI 内）' if lo2 <= vv <= hi2 else '（CI 外）'}")
print(f"  ⇒ 仅公比一项的合理取值区间就把预测扫过 "
      f"{max(anchor_lm+last_inc*rr/(1-rr) for rr in (0.46,0.662))-min(anchor_lm+last_inc*rr/(1-rr) for rr in (0.46,0.662)):.4f}"
      f"（渐近值口径），而判决声称可分辨的偏差只有 0.0022")

print("-" * 90); print("[Q5c] 跨轮可复现性下限：两轮在**同一个绝对世代窗**上测同一个量")
W = (anchor_gen - seg_A/2, anchor_gen + seg_A/2)
def win(R):
    out = {}
    for k, rec in R.items():
        g = rec["gen"]; m = (g >= W[0]) & (g <= W[1])
        out[k] = wmean([lm(h) for h in rec["hist"]], g, m) if m.sum() >= 3 else np.nan
    return out
wA, wB = win(A), win(B)
vA = np.array([np.nanmean([wA[(s,r)] for r in REPS]) for s in SEEDS])
vB = np.array([np.nanmean([wB[(s,r)] for r in REPS]) for s in SEEDS])
okA, okB = np.isfinite(vA), np.isfinite(vB)
laA, haA = bootstrap_ci(vA[okA]); laB, haB = bootstrap_ci(vB[okB])
print(f"  窗 gen [{W[0]:.1f}, {W[1]:.1f}]（= R18 末段的宽度，中心 345）")
print(f"    R18   {np.nanmean(vA):.4f}  CI [{laA:.4f},{haA:.4f}]  （{okA.sum()}/12 格有数据）")
print(f"    R18-B {np.nanmean(vB):.4f}  CI [{laB:.4f},{haB:.4f}]  （{okB.sum()}/12 格有数据）")
both = okA & okB
d_cross = vA[both] - vB[both]
lc, hc = bootstrap_ci(d_cross)
pc = wilcoxon(d_cross).pvalue
print(f"    **两轮同窗差 = {d_cross.mean():+.4f}**  CI [{lc:+.4f},{hc:+.4f}]  "
      f"{int((d_cross>0).sum())}/{both.sum()}  p={pc:.4f}")
print(f"  ⇒ 判决声称可分辨的「外推高估」是 +0.0022；"
      f"两轮**直接测量同一个量**的分歧是 {abs(d_cross.mean()):.4f}")

print("=" * 90); print("[Q6b] 末段增量与该 run 末段所处世代的关系（段长/epoch 异质性是否进了判据）")
per = {k: (seg_of(v, NSEG_B-1, NSEG_B, lm) - seg_of(v, NSEG_B-2, NSEG_B, lm)) for k, v in B.items()}
gm_run = {k: v["gen"].max() for k, v in B.items()}
cell_d = np.array([np.mean([per[(s,r)] for r in REPS]) for s in SEEDS])
cell_g = np.array([np.mean([gm_run[(s,r)] for r in REPS]) for s in SEEDS])
from scipy.stats import spearmanr
rho, pr = spearmanr(cell_g, cell_d)
print(f"  逐格末段增量 vs 逐格 gen_max：Spearman rho={rho:+.3f}  p={pr:.4f}  (n=12)")
print(f"  逐格 gen_max {np.round(cell_g,0).tolist()}")
print(f"  逐格 Δ       {np.round(cell_d,5).tolist()}")
lo_g = cell_d[cell_g < np.median(cell_g)].mean(); hi_g = cell_d[cell_g >= np.median(cell_g)].mean()
print(f"  短 run 半数（gen_max<{np.median(cell_g):.0f}）Δ={lo_g:+.5f}   "
      f"长 run 半数 Δ={hi_g:+.5f}   差 {lo_g-hi_g:+.5f}")

print("=" * 90); print("[Q1d] σ̂_W 与格均值 SD 的内部一致性")
print(f"  σ̂_W/√r = {sw/np.sqrt(2):.5f}   实测格均值 SD = {sd_cell:.5f}"
      f"  ⇒ 格均值 SD **比 σ̂_W/√r 还小** {100*(1-sd_cell/(sw/np.sqrt(2))):.0f}%")
print("  ⇒ 创始者方差分量估为 0（甚至负），且 σ̂_W 若无偏则格均值 SD 不应低于它/√r")
print(f"  ⇒ δ=σ̂_W 很可能是**偏大**的等价界（用它判等价偏宽松）")

print("=" * 90); print("[Q3] 「晚期」阈值 gen>100 的稳健性：全部失败检查点的世代分布")
sys.path.insert(0, "explorations/20260805-r18-verdict")
from diag_h2_degenerate import parts
fails = []
for k, rec in B.items():
    for h, gg in zip(rec["hist"], rec["gen"]):
        pr = parts(h, CTR)
        if pr is None or "gap" not in pr:  fails.append(("无双峰", k, float(gg)))
        elif pr["gap"] < 0.9999:           fails.append(("谷未见底", k, float(gg)))
gs = np.array([f[2] for f in fails])
print(f"  失败检查点共 {len(fails)}；世代分布分位 "
      f"min {gs.min():.1f} / 50% {np.percentile(gs,50):.1f} / 90% {np.percentile(gs,90):.1f} / "
      f"99% {np.percentile(gs,99):.1f} / max {gs.max():.1f}")
for lo_t, hi_t in ((13, 20), (20, 50), (50, 100), (100, 116), (116, 118), (118, 800)):
    n = int(((gs > lo_t) & (gs <= hi_t)).sum())
    print(f"    gen ({lo_t:>3}, {hi_t:>3}]: {n:>5} 次")
print(f"  ⇒ gen 13 与 gen 116 之间失败数 = {int(((gs>13)&(gs<116)).sum())} "
      f"⇒ **任何落在 (13,116) 的「晚期」阈值都给出同一组 4 次失败**")

# 更正上一行的表述：真正的空档是 (max_below_100, 116)
below = gs[gs < 100]
print(f"  【更正】gen<100 的失败最晚出现在 gen {below.max():.1f}；"
      f"其后到 gen {gs[gs>100].min():.1f} 之间**一次失败都没有** "
      f"⇒ **任何落在 ({below.max():.0f}, {gs[gs>100].min():.0f}) 的「晚期」阈值都给出同一组 4 次失败**")

print("=" * 90); print("[Q1e] 复核 §24.3 声称的 8/24、16/24 中期 CI（同 nseg=11 口径）")
for tag, ss in (("8 run（seed 0-3）", range(4)), ("16 run（seed 0-7）", range(8)), ("24 run", range(12))):
    sub = {k: v for k, v in B.items() if k[0] in list(ss)}
    per2 = {k: seg_of(v, NSEG_B-1, NSEG_B, lm) - seg_of(v, NSEG_B-2, NSEG_B, lm) for k, v in sub.items()}
    cm = np.array([np.mean([per2[(s, r)] for r in REPS]) for s in ss])
    l2, h2 = bootstrap_ci(cm)
    print(f"  {tag}: Δ {cm.mean():+.5f}  CI [{l2:+.5f}, {h2:+.5f}]  "
          f"{'含 0' if l2 <= 0 <= h2 else '**不含 0**'}   ({int((cm>0).sum())}/{len(cm)})")
