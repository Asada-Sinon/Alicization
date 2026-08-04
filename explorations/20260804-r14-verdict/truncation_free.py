"""截断修正后重做 H1：检查点 `low_mass` 的分母剔除了 pref∉[0.15,0.85]，而 R50 大量质量
跑到 pref<0.15。用两个不截断的读数重做：
  (a) traj `hist`（clip 口径，边缘箱收所有尾巴）算 low_mass_clip
  (b) 检查点里的 `mean_pref`（= p.mean()，对原始 pref 数组求均值，**完全不分箱**）
配对口径同 §17.6：先 r=2 平均成格均值，再 12 格配对符号秩，噪声 = √2·σ̂_W/√r，σ̂_W 90% 卡方上界。
读 outputs/20260804-ratio/*.log。重跑：
  XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python explorations/20260804-r14-verdict/truncation_free.py
"""
import glob, json, math, sys
sys.path.insert(0, "scripts")
import numpy as np
from scipy.stats import chi2, wilcoxon, spearmanr
from exp_stats import bootstrap_ci
BINS = np.linspace(0.15,0.85,15); CTR = 0.5*(BINS[:-1]+BINS[1:])
SEEDS, REPS, ARMS = list(range(12)), [1,2], ["R38","R50"]
REC = {}
for f in sorted(glob.glob("/home/xrl/intern/Alicization/outputs/20260804-ratio/*.log")):
    for ln in open(f):
        if ln.startswith("JSON "):
            b=f.split("/")[-1][:-4].split("_"); REC[(b[0],int(b[1][1:]),int(b[2][1:]))]=json.loads(ln[5:])
def cp(a,s,r,i): return REC[(a,s,r)]["checkpoints"][i]
def near(a,s,r,i):
    t = cp(a,s,r,i)["t"]
    return min(REC[(a,s,r)]["traj"], key=lambda p: abs(p["t"]-t))
def lm_clip(a,s,r,i):
    h = np.array(near(a,s,r,i)["hist"],float); return float(h[CTR<0.35].sum()/max(h.sum(),1))
def inrange(a,s,r,i):
    c = cp(a,s,r,i); return float(np.sum(c["bin_n"])/c["n_samples"])
def cell(a,fn): return np.array([np.nanmean([fn(a,s,r) for r in REPS]) for s in SEEDS])
def test(nm, fn, note=""):
    x,y = cell("R50",fn), cell("R38",fn); d = x-y
    w=[np.nanstd([fn(a,s,r) for r in REPS],ddof=1) for a in ARMS for s in SEEDS]
    w=np.array([v for v in w if np.isfinite(v)]); sw=float(np.sqrt((w**2).mean())); df=max(len(w),1)
    nz = sw*math.sqrt(df/chi2.ppf(0.10,df))*math.sqrt(2)/math.sqrt(2)
    lo,hi = bootstrap_ci(d)
    print(f"\n  [{nm}] {note}")
    print(f"    R38={y.mean():+.4f}  R50={x.mean():+.4f}  差={d.mean():+.4f}  方向 {int((d>0).sum())}/12 为正")
    print(f"    Wilcoxon p={wilcoxon(x,y).pvalue:.5f}  95%CI=[{lo:+.4f},{hi:+.4f}]"
          f"{'  **含0**' if lo<=0<=hi else ''}  效应/噪声={d.mean()/nz:+.2f}"
          f"{'  ** <1 undecidable **' if abs(d.mean()/nz)<1 else ''}")
    print(f"    R38 逐种子 {np.round(y,4).tolist()}")
    print(f"    R50 逐种子 {np.round(x,4).tolist()}")
    return d
print("="*94); print("截断诊断：检查点分箱丢掉了多少质量"); print("="*94)
for i,l in ((0,"cp0"),(1,"cp1")):
    for a in ARMS:
        v = cell(a, lambda x,y,z: inrange(x,y,z,i))
        print(f"  {a} {l} 落在[0.15,0.85]内的比例 = {v.mean():.4f}  范围[{v.min():.4f},{v.max():.4f}]")
d_in = cell("R50",lambda x,y,z: inrange(x,y,z,1)) - cell("R50",lambda x,y,z: inrange(x,y,z,0))
d_lm = cell("R50",lambda x,y,z: cp(x,y,z,1)["low_mass"]) - cell("R50",lambda x,y,z: cp(x,y,z,0)["low_mass"])
rho,p = spearmanr(d_in, d_lm)
print(f"  R50 臂内：Δ(落在范围内比例) 与 Δlow_mass 的 Spearman rho={rho:+.3f} p={p:.4f}  (n=12 格)")
allin = np.concatenate([cell(a,lambda x,y,z: inrange(x,y,z,1))-cell(a,lambda x,y,z: inrange(x,y,z,0)) for a in ARMS])
alllm = np.concatenate([cell(a,lambda x,y,z: cp(x,y,z,1)["low_mass"])-cell(a,lambda x,y,z: cp(x,y,z,0)["low_mass"]) for a in ARMS])
rho,p = spearmanr(allin, alllm); print(f"  两臂合并 24 格：rho={rho:+.3f} p={p:.2e}")

print("\n"+"="*94); print("H1 重做：三个读数并列（第一个是预注册用的、有截断）"); print("="*94)
test("H1 原始 Δlow_mass（检查点，**有截断**）", lambda a,s,r: cp(a,s,r,1)["low_mass"]-cp(a,s,r,0)["low_mass"], "预注册口径")
test("H1-clip Δlow_mass_clip（traj，尾巴压进边缘箱）", lambda a,s,r: lm_clip(a,s,r,1)-lm_clip(a,s,r,0), "无截断")
test("H1-pref Δmean_pref（检查点 mean_pref，完全不分箱）",
     lambda a,s,r: cp(a,s,r,1)["mean_pref"]-cp(a,s,r,0)["mean_pref"], "无截断；pref 低=偏果")
print("\n  水平量（非差分）：")
for i,l in ((0,"cp0"),(1,"cp1")):
    for nm,fn in (("low_mass_clip", lambda x,y,z: lm_clip(x,y,z,i)), ("mean_pref", lambda x,y,z: cp(x,y,z,i)["mean_pref"]),
                  ("pref_sd", lambda x,y,z: cp(x,y,z,i)["sd"])):
        a0,a1 = cell("R38",fn), cell("R50",fn)
        print(f"    {l} {nm:<16} R38={a0.mean():.4f}  R50={a1.mean():.4f}  "
              f"差={a1.mean()-a0.mean():+.4f}  p={wilcoxon(a1,a0).pvalue:.5f}  "
              f"{int((a1>a0).sum())}/12 为高")
