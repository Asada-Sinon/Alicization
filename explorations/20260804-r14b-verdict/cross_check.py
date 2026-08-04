"""三件收口用的数：
① R38p 的 overrides 与 R13 的 wn1 臂完全相同（diet_delta=0.15 是默认）⇒ R38p 是 R13 wn1 的
   一次重跑（12 种子 × r=2、42k 步、修好的读数）。对一下 cp0 的 low_mass 能不能对上。
② H1 的 MDE：§18.7 把 sigma_W 池化到四臂，但 H1 只涉及两个 n 臂。换成 n 臂自估再算一次，
   看 undecidable 的判定稳不稳。
③ Stage 2 的窗口够不够看到 R13 那个衰减：把 R13 反解的 s 外推到 28–40 代。
重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python explorations/20260804-r14b-verdict/cross_check.py
"""
import glob, json, math, sys
sys.path.insert(0, "scripts")
import numpy as np
from scipy.stats import chi2, wilcoxon, mannwhitneyu
from exp_stats import bootstrap_ci
B13 = np.linspace(0.15,0.85,15); C13 = 0.5*(B13[:-1]+B13[1:]); L13 = C13<0.35
tr = {}
for f in sorted(glob.glob("outputs/20260804-traj/*.log")):
    for ln in open(f):
        if ln.startswith("JSON "): tr[f.split("/")[-1][:-4]] = json.loads(ln[5:])
at = lambda r,t: [q for q in r["traj"] if q["t"]==t][-1]
clipv = lambda r,i: float((lambda h: h/max(h.sum(),1))(np.array(at(r,r["checkpoints"][i]["t"])["hist"],float))[L13].sum())
R2 = {}
for f in sorted(glob.glob("outputs/20260804-ratio2/*.log")):
    for ln in open(f):
        if ln.startswith("JSON "):
            b = f.split("/")[-1][:-4].split("_"); R2[(b[0], int(b[1][1:]), int(b[2][1:]))] = json.loads(ln[5:])

print("① R13 wn1（≡R38p 的配置）vs Stage 2 R38p：cp0 的 low_mass")
o13 = dict(tr["wn1_s0_r1"]["overrides"]); o38 = dict(R2[("R38p",0,1)]["overrides"])
print(f"   overrides 差异 = {sorted(k for k in set(o13)|set(o38) if o13.get(k)!=o38.get(k))}"
      f"  （R13 未显式设 diet_delta，Config 默认值 = {o38.get('diet_delta')}）")
a = np.array([clipv(tr[f"wn1_s{s}_r1"], 0) for s in range(12)])
b = np.array([np.mean([R2[("R38p",s,r)]["checkpoints"][0]["low_mass"] for r in (1,2)]) for s in range(12)])
print(f"   R13 wn1 cp0 (t=22000, clip 口径, r=1)  均值 {a.mean():.4f}  逐种子 {np.round(a,3).tolist()}")
print(f"   R38p   cp0 (t=23000, 修好的 20 箱, r=2) 均值 {b.mean():.4f}  逐种子 {np.round(b,3).tolist()}")
print(f"   同种子配对差 {np.mean(b-a):+.4f}  {(b>a).sum()}/12 正  配对 p={wilcoxon(b,a).pvalue:.4f}  "
      f"不配对 Mann-Whitney p={mannwhitneyu(a,b).pvalue:.4f}")
print(f"   （R13 wn1 cp0 截断口径均值 {np.mean([tr[f'wn1_s{s}_r1']['checkpoints'][0]['low_mass'] for s in range(12)]):.4f}）")

print("\n② H1 的噪声与 MDE：池化四臂 vs 只用 n 臂")
lm = lambda a,s,r,i: R2[(a,s,r)]["checkpoints"][i]["low_mass"]
dlm = lambda a,s,r: lm(a,s,r,1)-lm(a,s,r,0)
cell = lambda a: np.array([np.mean([dlm(a,s,r) for r in (1,2)]) for s in range(12)])
within = lambda a: np.array([np.std([dlm(a,s,r) for r in (1,2)], ddof=1) for s in range(12)])
def sw(arms):
    w = np.concatenate([within(a) for a in arms])
    return float(np.sqrt((w**2).mean()))*math.sqrt(len(w)/chi2.ppf(0.10,len(w)))
d = cell("R50n")-cell("R38n"); lo,hi = bootstrap_ci(d)
for arms,tag in ((["R38p","R50p","R38n","R50n"],"四臂池化（§18.7 原样）"), (["R38n","R50n"],"只用 n 臂")):
    nz = sw(arms)*math.sqrt(2.0)/math.sqrt(2)
    print(f"   {tag:<22} 噪声 {nz:.4f}   H1 效应 {d.mean():+.4f}   比值 {d.mean()/nz:+.2f}   "
          f"{'undecidable' if abs(d.mean()/nz)<1 else '过线'}")
print(f"   H1 的 95% bootstrap CI = [{lo:+.4f},{hi:+.4f}]  ⇒ 本设计排除掉的是 |资源比对 Δ 的影响| > "
      f"{max(abs(lo),abs(hi)):.4f} 的效应；而两个 n 臂各自的 Δ 是 +0.0295/+0.0252（比 CI 宽度大）")

print("\n③ Stage 2 的窗口能不能看到 R13 的衰减")
for stag, s in (("截断口径反解 s=0.00463", 0.00463), ("不截断口径反解 s=0.00136", 0.00136)):
    print(f"   {stag}:")
    for a, g in (("R38n",27.7), ("R50n",39.8), ("R38p",101.5), ("R50p",91.0)):
        l0 = np.array([np.mean([lm(a,x,r,0) for r in (1,2)]) for x in range(12)]).mean()
        pred = l0*(math.exp(-s*g)-1)
        obs = cell(a).mean()
        print(f"      {a:>5} l0={l0:.4f} 跨 {g:.0f} 代 ⇒ 预期 Δ={pred:+.4f}  实测 Δ={obs:+.4f}  "
              f"差 {obs-pred:+.4f}")
