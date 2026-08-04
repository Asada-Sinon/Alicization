"""接 verify_stage2.py：§18.7 把 sigma_W 池化到四臂，但四臂的格内方差差一个数量级
——这会把 n 臂那个 12/12 同向的正 Δ 的比值人为压到 <1。这里按臂自估噪声再算一次，
并把 Δ 换算到「每代」以便和 R13 的 80k 步衰减比。
读：outputs/20260804-ratio2/*.log。重跑：
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python explorations/20260804-r14b-verdict/per_arm_noise.py
"""
import glob, json, math, sys
sys.path.insert(0, "scripts")
import numpy as np
from scipy.stats import chi2, wilcoxon, spearmanr
from exp_stats import bootstrap_ci

SEEDS, REPS = list(range(12)), [1, 2]
ARMS = ["R38p", "R50p", "R38n", "R50n"]
REC = {}
for f in sorted(glob.glob("outputs/20260804-ratio2/*.log")):
    for ln in open(f):
        if ln.startswith("JSON "):
            b = f.split("/")[-1][:-4].split("_")
            REC[(b[0], int(b[1][1:]), int(b[2][1:]))] = json.loads(ln[5:])

lm = lambda a,s,r,i: REC[(a,s,r)]["checkpoints"][i]["low_mass"]
dlm = lambda a,s,r: lm(a,s,r,1) - lm(a,s,r,0)
cell = lambda a,fn: np.array([np.mean([fn(a,s,r) for r in REPS]) for s in SEEDS])
within = lambda a,fn: np.array([np.std([fn(a,s,r) for r in REPS], ddof=1) for s in SEEDS])

def sw_ub(arms, fn):
    w = np.concatenate([within(a, fn) for a in arms]); w = w[np.isfinite(w)]
    return float(np.sqrt((w**2).mean())) * math.sqrt(len(w)/chi2.ppf(0.10, len(w)))

print("检查点时刻与世代：")
for a in ARMS:
    t0 = REC[(a,0,1)]["checkpoints"][0]["t"]; t1 = REC[(a,0,1)]["checkpoints"][1]["t"]
    g = []
    for s in SEEDS:
        for r in REPS:
            tr = REC[(a,s,r)]["traj"]
            f = lambda t: float(np.interp(t, [q["t"] for q in tr], [q["generation"] for q in tr]))
            g.append(f(t1) - f(t0))
    print(f"  {a:>5}  cp0 t={t0}  cp1 t={t1}  cp0->cp1 世代数 中位 {np.median(g):.1f} (min {min(g):.0f} max {max(g):.0f})")

print("\n" + "="*88)
print("每臂 Δlow_mass 检验：噪声按【本臂】自估（对比 §18.7 的四臂池化 0.0519）")
print("="*88)
pool = sw_ub(ARMS, dlm) / math.sqrt(len(REPS))
print(f"  四臂池化噪声 = {pool:.4f}")
for a in ARMS:
    d = cell(a, dlm); own = sw_ub([a], dlm) / math.sqrt(len(REPS))
    lo, hi = bootstrap_ci(d)
    # 种子间口径：均值的标准误
    se = d.std(ddof=1)/math.sqrt(len(d))
    print(f"  {a:>5}  Δ={d.mean():+.4f}  {(d>0).sum()}/12 正  p={wilcoxon(d).pvalue:.5f}  "
          f"CI=[{lo:+.4f},{hi:+.4f}]")
    print(f"         本臂噪声={own:.4f} 比值={d.mean()/own:+.2f} | "
          f"池化噪声={pool:.4f} 比值={d.mean()/pool:+.2f} | "
          f"种子间SD={d.std(ddof=1):.4f} SE={se:.4f} t={d.mean()/se:+.2f}")

print("\n" + "="*88)
print("Δ 换算到每代（和 R13 的 80k 步比）")
print("="*88)
for a in ARMS:
    g = np.median([ (lambda tr: float(np.interp(REC[(a,s,r)]['checkpoints'][1]['t'], [q['t'] for q in tr], [q['generation'] for q in tr])) - float(np.interp(REC[(a,s,r)]['checkpoints'][0]['t'], [q['t'] for q in tr], [q['generation'] for q in tr])))(REC[(a,s,r)]["traj"]) for s in SEEDS for r in REPS])
    d = cell(a, dlm).mean(); l0 = cell(a, lambda x,s,r: lm(x,s,r,0)).mean()
    rel = d / l0
    s_eq = -math.log((l0+d)/l0)/g
    print(f"  {a:>5}  Δ={d:+.4f}  相对起点 {rel*100:+.1f}%  跨 {g:.0f} 代  "
          f"等效 s = {s_eq:+.5f}/代 （负号=增长）")

print("\n" + "="*88)
print("总通量 vs low_mass：跨臂 vs 臂内（§18.6 的多变量担忧到底咬不咬 H1）")
print("="*88)
flux = {a: np.array([np.mean([REC[(a,s,r)]["checkpoints"][0]["graze_gain"]
                              + REC[(a,s,r)]["checkpoints"][0]["fruit_gain"] for r in REPS]) for s in SEEDS])
        for a in ARMS}
for a in ARMS:
    r1,p1 = spearmanr(flux[a], cell(a, lambda x,s,r: lm(x,s,r,0)))
    r2,p2 = spearmanr(flux[a], cell(a, dlm))
    print(f"  {a:>5} 臂内 n=12：Spearman(通量, low_mass_cp0)={r1:+.3f} p={p1:.3f}   "
          f"Spearman(通量, Δ)={r2:+.3f} p={p2:.3f}")
print("  跨 48 格（含臂间）：", end="")
allf = np.concatenate([flux[a] for a in ARMS])
print(f"vs low_mass_cp0 = {spearmanr(allf, np.concatenate([cell(a, lambda x,s,r: lm(x,s,r,0)) for a in ARMS]))[0]:+.3f}   "
      f"vs Δ = {spearmanr(allf, np.concatenate([cell(a, dlm) for a in ARMS]))[0]:+.3f}")
# 只在 n 臂内（干净对比）做通量-Δ 的关系
fn2 = np.concatenate([flux[a] for a in ("R38n","R50n")])
dn2 = np.concatenate([cell(a, dlm) for a in ("R38n","R50n")])
print(f"  仅 n 臂 24 格：Spearman(通量, Δ) = {spearmanr(fn2, dn2)[0]:+.3f} (p={spearmanr(fn2, dn2)[1]:.3f})")
