"""R14 Stage 1 独立复核：闸该不该触发，混杂能不能拆开。

回答的问题（judge 侧，非重跑预注册脚本）：
  Q1 复核 H0/H1/H2 的预注册口径读数（对照 explorations/20260804-readouts/analyze_r14.py）
  Q2 H1 的尺度无关口径：相对跌幅 Δ/lm0、log(lm1/lm0)、以及**终点** lm1 的臂间对比
  Q3 混杂：Δlow_mass ~ 丢失捕食者 / 种群量 / 起点，run 级与格级都算
  Q4 臂内对比（同臂内 lost vs kept），以及只用 kept run 的臂间对比
  Q5 H2 摄入比按 lost/kept 分层，并检验「比值 < 1」本身
  Q6 通量：实测 vs 探针预测；R50 frugivory 逐种子离散度与剂量-反应
读的文件：outputs/20260804-ratio/*.log（48 个，每个一行 "JSON "）
         outputs/20260804-dose/*.log（探针 6 个，单种子 22k）
输出：纯 stdout，见 output/verdict_r14.txt
重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260804-r14-verdict/verdict_r14.py
"""
import glob, json, math, sys
sys.path.insert(0, "scripts")
import numpy as np
from scipy.stats import chi2, wilcoxon, mannwhitneyu, spearmanr
from exp_stats import bootstrap_ci

CTR = np.linspace(0.15, 0.85, 15); CTR = 0.5 * (CTR[:-1] + CTR[1:])
LOW, HIGH = CTR < 0.35, CTR > 0.65
SEEDS, REPS, ARMS = list(range(12)), [1, 2], ["R38", "R50"]
SPEC = {"R38": (0.25, 0.010), "R50": (0.50, 0.005)}
ROOT = "/home/xrl/intern/Alicization/"

REC = {}
for f in sorted(glob.glob(ROOT + "outputs/20260804-ratio/*.log")):
    for ln in open(f):
        if ln.startswith("JSON "):
            b = f.split("/")[-1][:-4].split("_")
            REC[(b[0], int(b[1][1:]), int(b[2][1:]))] = json.loads(ln[5:])
assert len(REC) == 48, len(REC)
for k, d in REC.items():
    o = d["overrides"]
    assert float(o["fruit_regrow_baseline"]) == SPEC[k[0]][0] and float(o["regrow_baseline"]) == SPEC[k[0]][1], k
    assert d["steps"] == 42000 and not d["collapsed"] and len(d["checkpoints"]) == 2, k
# 臂间 overrides 差异必须恰好 2 项
oa = REC[("R38", 0, 1)]["overrides"]; ob = REC[("R50", 0, 1)]["overrides"]
diff = sorted(k for k in set(oa) | set(ob) if oa.get(k) != ob.get(k))
print(f"48 run 载入；overrides 差异项 = {diff}；cp t = {[c['t'] for c in REC[('R38',0,1)]['checkpoints']]}")

def rec(a, s, r): return REC[(a, s, r)]
def lm(a, s, r, i): return rec(a, s, r)["checkpoints"][i]["low_mass"]
def dlm(a, s, r): return lm(a, s, r, 1) - lm(a, s, r, 0)
def rel(a, s, r):
    x = lm(a, s, r, 0)
    return (lm(a, s, r, 1) - x) / x if x > 0 else np.nan
def lograt(a, s, r):
    x, y = lm(a, s, r, 0), lm(a, s, r, 1)
    return math.log(y / x) if x > 0 and y > 0 else np.nan
def cpv(a, s, r, i, k): return rec(a, s, r)["checkpoints"][i].get(k, np.nan)
def ratio(a, s, r, i=0):
    cp = rec(a, s, r)["checkpoints"][i]
    ik = np.array([np.nan if v is None else v for v in cp["intake"]], float)
    n = np.array(cp["bin_n"], float)
    p, q = LOW & np.isfinite(ik) & (n > 0), HIGH & np.isfinite(ik) & (n > 0)
    if not (p.any() and q.any()): return np.nan
    return float((ik[p]*n[p]).sum()/n[p].sum() / ((ik[q]*n[q]).sum()/n[q].sum()))
def secondhalf(a, s, r):
    t = rec(a, s, r)["traj"]; return t[len(t)//2:]
def lost(a, s, r): return any(p["carnivore_frac"] == 0 for p in secondhalf(a, s, r))
def lost_final(a, s, r):
    v = [p["carnivore_frac"] for p in rec(a, s, r)["traj"] if np.isfinite(p["carnivore_frac"])]
    return v[-1] == 0
def popmean(a, s, r):
    return float(np.nanmean([p["population"] for p in rec(a, s, r)["traj"]]))
def carnmean(a, s, r):
    return float(np.nanmean([p["carnivore_frac"] for p in secondhalf(a, s, r)]))
def flux(a, s, r, i=0): return cpv(a, s, r, i, "graze_gain") + cpv(a, s, r, i, "fruit_gain")

def cell(a, fn): return np.array([np.nanmean([fn(a, s, r) for r in REPS]) for s in SEEDS])
def within_sd(fn):
    w = [np.nanstd([fn(a, s, r) for r in REPS], ddof=1) for a in ARMS for s in SEEDS]
    return np.array([v for v in w if np.isfinite(v)])
def noise_ub(fn):
    w = within_sd(fn); sw = float(np.sqrt((w**2).mean())); df = max(len(w), 1)
    return sw * math.sqrt(df/chi2.ppf(0.10, df)) * math.sqrt(2)/math.sqrt(len(REPS)), sw

def paired(name, fn, pred=">0"):
    x, y = cell("R50", fn), cell("R38", fn)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]; d = x - y
    nz, sw = noise_ub(fn)
    p = wilcoxon(x, y).pvalue
    lo, hi = bootstrap_ci(d)
    print(f"\n  [{name}]  R38={y.mean():+.4f}  R50={x.mean():+.4f}  差={d.mean():+.4f}  "
          f"({len(d)} 格; 预测 {pred})")
    print(f"     方向 {int((d>0).sum())}/{len(d)} 为正   Wilcoxon p={p:.5f}   "
          f"95%CI=[{lo:+.4f},{hi:+.4f}]{'  **含0**' if lo<=0<=hi else ''}")
    print(f"     sigma_W={sw:.4f}  噪声(90%上界)={nz:.4f}  效应/噪声={d.mean()/nz:+.2f}"
          f"{'  ** <1 undecidable **' if abs(d.mean()/nz)<1 else ''}")
    print(f"     R38 逐种子 {np.round(y,4).tolist()}")
    print(f"     R50 逐种子 {np.round(x,4).tolist()}")
    return d

print("\n" + "="*94); print("Q1+Q2  H1 的四个口径（预注册用的只有第一个）"); print("="*94)
paired("H0  low_mass(cp0)", lambda a,s,r: lm(a,s,r,0), "R50>R38")
paired("H1  Δlow_mass 原始（预注册）", dlm, "R50>R38 衰减变慢")
paired("H1' 相对跌幅 Δ/lm0", rel, "R50>R38")
paired("H1'' log(lm1/lm0)", lograt, "R50>R38")
paired("终点 low_mass(cp1)", lambda a,s,r: lm(a,s,r,1), "（无预注册预测）")
paired("H2  摄入低/高比 @cp0", lambda a,s,r: ratio(a,s,r,0), "R50>R38")
paired("H2' 摄入低/高比 @cp1", lambda a,s,r: ratio(a,s,r,1), "（非预注册）")


# ---- Q2b 尺度无关口径必须在**格均值**上构造（§17.6 ①：先平均再算派生量）
print("\n" + "="*94)
print("Q2b  尺度无关口径（在格均值上构造比值，而非平均逐 run 比值）")
print("="*94)
lm0 = {a: cell(a, lambda a_,s,r: lm(a_,s,r,0)) for a in ARMS}
lm1 = {a: cell(a, lambda a_,s,r: lm(a_,s,r,1)) for a in ARMS}
relc = {a: (lm1[a]-lm0[a])/lm0[a] for a in ARMS}
EPS = 1.0/14/2   # 一个箱的半个最小可分辨质量，用于 log 的连续性修正
logc = {a: np.log((lm1[a]+EPS)/(lm0[a]+EPS)) for a in ARMS}
for nm, D in (("相对跌幅 Δ/lm0（格级）", relc), ("log((lm1+e)/(lm0+e))（格级, e=1/28）", logc)):
    d = D["R50"]-D["R38"]
    lo, hi = bootstrap_ci(d)
    print(f"\n  [{nm}]  R38={D['R38'].mean():+.4f}  R50={D['R50'].mean():+.4f}  差={d.mean():+.4f}")
    print(f"     方向 {int((d>0).sum())}/12 为正   Wilcoxon p={wilcoxon(D['R50'],D['R38']).pvalue:.5f}   "
          f"95%CI=[{lo:+.4f},{hi:+.4f}]{'  **含0**' if lo<=0<=hi else ''}")
    print(f"     R38 逐种子 {np.round(D['R38'],3).tolist()}")
    print(f"     R50 逐种子 {np.round(D['R50'],3).tolist()}")
# 臂级聚合口径（用户报的 -44% / -72%）
for a in ARMS:
    print(f"  臂级聚合 {a}: mean(Δ)/mean(lm0) = {(lm1[a]-lm0[a]).mean()/lm0[a].mean()*100:+.1f}%"
          f"   (mean lm0={lm0[a].mean():.4f} → mean lm1={lm1[a].mean():.4f})")
nz = np.log((lm1['R50']+EPS)/(lm0['R50']+EPS)) - np.log((lm1['R38']+EPS)/(lm0['R38']+EPS))
print(f"  R38 中 lm1 触底(=0.0000)的格: {[int(s) for s in SEEDS if lm1['R38'][s]==0]}  "
      f"R50: {[int(s) for s in SEEDS if lm1['R50'][s]==0]}")

# ---- Q3 混杂
print("\n" + "="*94); print("Q3  混杂：Δlow_mass ~ 丢失捕食者 / 种群量 / 起点"); print("="*94)
rows = [(a,s,r) for a in ARMS for s in SEEDS for r in REPS]
D  = np.array([dlm(*k) for k in rows]); L = np.array([1.0*lost(*k) for k in rows])
P  = np.array([popmean(*k) for k in rows]); C = np.array([carnmean(*k) for k in rows])
L0 = np.array([lm(*k,0) for k in rows]); A = np.array([1.0*(k[0]=="R50") for k in rows])
print("  run 级 (n=48，**重复非独立，p 偏乐观，只作描述**)：")
for nm, v in (("丢失捕食者(0/1)",L),("种群均值",P),("后半程 carn_frac",C),("起点 low_mass",L0),("臂(R50=1)",A)):
    rho,p = spearmanr(D, v); print(f"    Δlow_mass ~ {nm:<16} Spearman rho={rho:+.3f}  p={p:.2e}")
Dc = np.concatenate([cell(a,dlm) for a in ARMS]); Pc = np.concatenate([cell(a,popmean) for a in ARMS])
Lc = np.concatenate([cell(a,lambda a_,s,r: 1.0*lost(a_,s,r)) for a in ARMS])
L0c= np.concatenate([lm0[a] for a in ARMS]); Ac = np.concatenate([np.zeros(12), np.ones(12)])
print("  格级 (n=24 格均值)：")
for nm, v in (("丢失比例",Lc),("种群均值",Pc),("起点 low_mass",L0c),("臂",Ac)):
    rho,p = spearmanr(Dc, v); print(f"    Δlow_mass ~ {nm:<16} Spearman rho={rho:+.3f}  p={p:.2e}")
for a in ARMS:
    rho,p = spearmanr(cell(a,dlm), cell(a,popmean))
    rho2,p2 = spearmanr(cell(a,dlm), lm0[a])
    print(f"  **臂内**格级 {a}: Δ~种群 rho={rho:+.3f} p={p:.3f}   Δ~起点 rho={rho2:+.3f} p={p2:.3f}")

print("\n  臂内 lost vs kept（run 级，n 小，只作描述）：")
for a in ARMS:
    li = [k for k in rows if k[0]==a and lost(*k)]; ki = [k for k in rows if k[0]==a and not lost(*k)]
    for nm, fn in (("Δlow_mass",dlm),("lm0",lambda x,y,z: lm(x,y,z,0)),("lm1",lambda x,y,z: lm(x,y,z,1)),
                   ("种群",popmean),("摄入低/高@cp0",lambda x,y,z: ratio(x,y,z,0))):
        vl = np.array([fn(*k) for k in li]); vk = np.array([fn(*k) for k in ki])
        u = mannwhitneyu(vl, vk).pvalue if len(vl)>0 and len(vk)>0 else np.nan
        print(f"    {a} {nm:<14} lost(n={len(vl)})={np.nanmean(vl):+.4f}   "
              f"kept(n={len(vk)})={np.nanmean(vk):+.4f}   MWU p={u:.4f}")

# 两臂都 kept 的种子
clean = [s for s in SEEDS if not any(lost(a,s,r) for a in ARMS for r in REPS)]
print(f"\n  两臂全 kept 的种子 = {clean}  (n={len(clean)})")
if clean:
    for nm, fn in (("Δlow_mass",dlm),("lm0",lambda x,y,z: lm(x,y,z,0)),("lm1",lambda x,y,z: lm(x,y,z,1)),
                   ("摄入低/高@cp0",lambda x,y,z: ratio(x,y,z,0))):
        x = np.array([np.nanmean([fn("R50",s,r) for r in REPS]) for s in clean])
        y = np.array([np.nanmean([fn("R38",s,r) for r in REPS]) for s in clean])
        print(f"    {nm:<14} R38={y.mean():+.4f}  R50={x.mean():+.4f}  差={(x-y).mean():+.4f}  "
              f"同向 {int(((x-y)>0).sum())}/{len(x)}   [n={len(x)} 配对最小双侧 p={2/2**len(x):.3f}]")
# kept-only 非配对臂间
kx = [k for k in rows if k[0]=="R50" and not lost(*k)]; ky = [k for k in rows if k[0]=="R38" and not lost(*k)]
for nm, fn in (("Δlow_mass",dlm),("摄入低/高@cp0",lambda x,y,z: ratio(x,y,z,0))):
    vx = np.array([fn(*k) for k in kx]); vy = np.array([fn(*k) for k in ky])
    print(f"  kept-only 非配对(run级,伪重复) {nm}: R38(n={len(vy)})={np.nanmean(vy):+.4f}  "
          f"R50(n={len(vx)})={np.nanmean(vx):+.4f}  MWU p={mannwhitneyu(vx,vy).pvalue:.4f}")


def mwu(x, y):
    x = np.asarray(x,float); y = np.asarray(y,float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    return (np.nan if len(x)<1 or len(y)<1 else mannwhitneyu(x,y).pvalue), len(x), len(y)

print("\n" + "="*94)
print("Q4  只用保住捕食者的 run 做臂间对比（NaN 已剔除）")
print("="*94)
for nm, fn in (("lm0",lambda x,y,z: lm(x,y,z,0)),("lm1",lambda x,y,z: lm(x,y,z,1)),
               ("Δlow_mass",dlm),("摄入低/高@cp0",lambda x,y,z: ratio(x,y,z,0)),
               ("摄入低/高@cp1",lambda x,y,z: ratio(x,y,z,1)),("种群",popmean),
               ("frugivory@cp0",lambda x,y,z: cpv(x,y,z,0,"frugivory_frac"))):
    vx = [fn(*k) for k in rows if k[0]=="R50" and not lost(*k)]
    vy = [fn(*k) for k in rows if k[0]=="R38" and not lost(*k)]
    p,nx,ny = mwu(vx,vy)
    print(f"  kept-only {nm:<14} R38(n={ny})={np.nanmean(vy):+.4f}  R50(n={nx})={np.nanmean(vx):+.4f}  MWU p={p:.4f}")
print("  ** 上表是 run 级、非配对：r=2 同种子重复被当独立点，p 偏乐观，只作方向证据 **")

print("\n" + "="*94)
print("Q5  摄入低/高比本身：是否显著 < 1（R13 §14.7/§14.8 的『不对称权衡』）")
print("="*94)
allc = []
for a in ARMS:
    v = cell(a, lambda x,y,z: ratio(x,y,z,0)); v = v[np.isfinite(v)]
    st = wilcoxon(v - 1.0)
    print(f"  {a} @cp0  均值={v.mean():.4f}  范围=[{v.min():.4f},{v.max():.4f}]  "
          f"{int((v<1).sum())}/{len(v)} 格 <1   单样本 Wilcoxon vs 1: p={st.pvalue:.6f}")
    allc.append(v)
    v1 = cell(a, lambda x,y,z: ratio(x,y,z,1)); v1 = v1[np.isfinite(v1)]
    print(f"  {a} @cp1  均值={v1.mean():.4f}  范围=[{v1.min():.4f},{v1.max():.4f}]  "
          f"{int((v1<1).sum())}/{len(v1)} 格 <1   单样本 Wilcoxon vs 1: p={wilcoxon(v1-1.0).pvalue:.6f}")
v = np.concatenate(allc)
print(f"  两臂合并 @cp0 (n={len(v)} 格)  均值={v.mean():.4f}  {int((v<1).sum())}/{len(v)} <1  "
      f"p={wilcoxon(v-1.0).pvalue:.2e}")
lk = [ratio(*k,0) for k in rows if lost(*k)]; kk = [ratio(*k,0) for k in rows if not lost(*k)]
lk = [x for x in lk if np.isfinite(x)]; kk = [x for x in kk if np.isfinite(x)]
print(f"  按丢失分层（run 级，两臂合并）: lost(n={len(lk)})={np.mean(lk):.4f}  "
      f"kept(n={len(kk)})={np.mean(kk):.4f}   全 48 run 中 <1 的比例 = "
      f"{sum(1 for k in rows if np.isfinite(ratio(*k,0)) and ratio(*k,0)<1)}/"
      f"{sum(1 for k in rows if np.isfinite(ratio(*k,0)))}")

print("\n" + "="*94)
print("Q6  通量：实测 vs 探针；以及 frugivory 是不是外生的『剂量』")
print("="*94)
PROBE = {"R38": (188.925, 114.948, 0.3783), "R50": (144.972, 144.110, 0.4985)}
for a in ARMS:
    g0 = cell(a, lambda x,y,z: cpv(x,y,z,0,"graze_gain")); f0 = cell(a, lambda x,y,z: cpv(x,y,z,0,"fruit_gain"))
    g1 = cell(a, lambda x,y,z: cpv(x,y,z,1,"graze_gain")); f1 = cell(a, lambda x,y,z: cpv(x,y,z,1,"fruit_gain"))
    fr0 = cell(a, lambda x,y,z: cpv(x,y,z,0,"frugivory_frac")); fr1 = cell(a, lambda x,y,z: cpv(x,y,z,1,"frugivory_frac"))
    pg,pf,pfr = PROBE[a]
    print(f"  {a} @cp0 graze={g0.mean():.1f} fruit={f0.mean():.1f} 总={(g0+f0).mean():.1f}  "
          f"frugivory={fr0.mean():.4f} (SD={fr0.std(ddof=1):.4f}, 范围[{fr0.min():.3f},{fr0.max():.3f}])")
    print(f"       探针(单种子22k) graze={pg:.1f} fruit={pf:.1f} 总={pg+pf:.1f} frugivory={pfr:.4f}"
          f"   ⇒ 总通量实测/探针={(g0+f0).mean()/(pg+pf):.3f}  frugivory 差={fr0.mean()-pfr:+.4f}")
    print(f"  {a} @cp1 总={(g1+f1).mean():.1f}  frugivory={fr1.mean():.4f}  达标(±0.05) "
          f"{int(np.sum(np.abs(fr1-(0.38 if a=='R38' else 0.50))<=0.05))}/12")
tot = {a: cell(a, flux) for a in ARMS}
print(f"  总通量臂间: R38={tot['R38'].mean():.1f}  R50={tot['R50'].mean():.1f}  "
      f"相对={(tot['R50'].mean()/tot['R38'].mean()-1)*100:+.1f}%   "
      f"Wilcoxon p={wilcoxon(tot['R50'],tot['R38']).pvalue:.5f}  "
      f"{int((tot['R50']>tot['R38']).sum())}/12 为高")
print(f"  探针预测的相对总通量 = {(PROBE['R50'][0]+PROBE['R50'][1])/(PROBE['R38'][0]+PROBE['R38'][1])*100-100:+.1f}%")
print("\n  frugivory 是不是外生剂量？（若它随种群/捕食者变动，它就是结果不是剂量）")
for a in ARMS:
    fr = cell(a, lambda x,y,z: cpv(x,y,z,0,"frugivory_frac"))
    for nm, v in (("种群",cell(a,popmean)), ("丢失比例",cell(a,lambda x,y,z:1.0*lost(x,y,z))),
                  ("起点 low_mass",lm0[a]), ("Δlow_mass",cell(a,dlm))):
        rho,p = spearmanr(fr, v); print(f"    {a}: frugivory@cp0 ~ {nm:<14} rho={rho:+.3f} p={p:.3f}")

print("\n" + "="*94); print("Q7  低簇质量的时间轨迹（traj hist，非检查点），看 44k 是否还在跌"); print("="*94)
def traj_lm(a,s,r):
    t = rec(a,s,r)["traj"]; out=[]
    for p in t:
        h = np.array(p["hist"],float); tot=h.sum()
        out.append((p["t"], float(h[CTR<0.35].sum()/tot) if tot>0 else np.nan))
    return out
for a in ARMS:
    for grp, sel in (("kept",lambda k: not lost(*k)), ("lost",lambda k: lost(*k))):
        ks = [k for k in rows if k[0]==a and sel(k)]
        if not ks: continue
        prof=[]
        for frac in (0.25,0.5,0.75,0.9,1.0):
            vals=[]
            for k in ks:
                ser = traj_lm(*k); tt = ser[-1][0]
                cand = [v for (t,v) in ser if abs(t-frac*tt) < 900]
                if cand: vals.append(cand[0])
            prof.append(np.nanmean(vals) if vals else np.nan)
        print(f"  {a}/{grp} (n={len(ks)}) low_mass @ 11k/22k/33k/40k/44k = "
              + "  ".join(f"{v:.3f}" for v in prof))
