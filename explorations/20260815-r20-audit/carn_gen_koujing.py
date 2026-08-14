"""§27.1 的 carn_frac 与「实测代数」两列用什么口径才能复现——lm/两峰已确认是 iso 世代加权，
但 carn（pub 0.1047/0.1236/0.0879）与代数（pub 481/481/433）用同口径对不上（我算 0.114/0.138/0.096 与 466/470/424）。
逐个试候选口径。另：逐 run 晚期高侧质量是否全为多数（形状问题的收尾）。"""
import glob, json, sys
sys.path.insert(0, "explorations/20260804-readouts")
sys.path.insert(0, "explorations/20260805-r18-verdict")
import numpy as np
from diag_h2_degenerate import parts
from neutral_null import gen_weights

BINS = np.linspace(0,1,21); CTR = 0.5*(BINS[:-1]+BINS[1:]); LOW = CTR<0.35
SEEDS = range(12); REPS=(1,2)
def load(d):
    R={}
    for f in sorted(glob.glob(f"{d}/*.log")):
        t=open(f).read()
        if "JSON " not in t: continue
        j=json.loads(t.split("JSON ")[1].split("\n")[0]); b=f.split("/")[-1][:-4].split("_")
        R[(b[0],int(b[1][1:]),int(b[2][1:]))]={
          "hist":np.array([q["hist"] for q in j["traj"]],float),
          "gen":np.array([q["generation"] for q in j["traj"]],float),
          "t":np.array([q["t"] for q in j["traj"]],float),
          "carn":np.array([q["carnivore_frac"] for q in j["traj"]],float),
          "gen_total":j.get("gen_total")}
    return R
R=load("outputs/20260813-r20")
def cm(per,c):
    out=[]
    for s in SEEDS:
        vs=[per[(c,s,r)] for r in REPS if np.isfinite(per.get((c,s,r),np.nan))]
        out.append(np.mean(vs) if vs else np.nan)
    return np.array(out)
PUBC={"wn1on":0.1047,"wn2on":0.1236,"b35on":0.0879}; PUBG={"wn1on":481,"wn2on":481,"b35on":433}
print("== carn_frac 候选口径（格均值） ==")
for name,fn in [
  ("末1/4按代·帧均", lambda r:np.nanmean(r["carn"][(r["gen"]>=r["gen"].max()*0.75)&np.isfinite(r["carn"])])),
  ("末1/4按代·iso加权", lambda r:(lambda m,w: float((np.nan_to_num(r["carn"][m])*w[m]*np.isfinite(r["carn"])[m]).sum()/max((w[m]*np.isfinite(r["carn"])[m]).sum(),1e-9)))(r["gen"]>=r["gen"].max()*0.75, gen_weights(r["gen"],"iso"))),
  ("全程帧均", lambda r:np.nanmean(r["carn"][np.isfinite(r["carn"])])),
  ("末1/4按步", lambda r:np.nanmean(r["carn"][(r["t"]>=r["t"].max()*0.75)&np.isfinite(r["carn"])])),
  ("末帧", lambda r:r["carn"][np.isfinite(r["carn"])][-1]),
]:
    row=[f"{np.nanmean(cm({k:fn(v) for k,v in R.items() if k[0]==c},c)):.4f}" for c in ("wn1on","wn2on","b35on")]
    print(f"  {name:14s} {row}  pub [0.1047, 0.1236, 0.0879]")
print("\n== 代数候选口径 ==")
for name,fn in [
  ("末帧gen格均", lambda r:r["gen"][-1]), ("gen.max格均", lambda r:r["gen"].max()),
  ("gen_total格均", lambda r:r["gen_total"]),
]:
    per={k:fn(v) for k,v in R.items()}
    row=[f"{np.nanmean(cm({k:v for k,v in per.items() if k[0]==c},c)):.0f}" for c in ("wn1on","wn2on","b35on","wn1off","wn2off","b35off")]
    print(f"  {name:12s} on={row[:3]} off={row[3:]}  pub on [481,481,433] off [148,123,151]")
per={k:v["gen"][-1] for k,v in R.items()}
for c in ("wn1on","wn2on","b35on"):
    vals=sorted([per[k] for k in per if k[0]==c])
    print(f"  {c} 末帧gen: min={vals[0]:.0f} max={vals[-1]:.0f} 逐run={['%.0f'%v for v in vals]}")
# 排除丢捕食者 run 后的均值
LOST={("wn1on",7,2),("b35on",1,2),("b35on",4,2),("b35on",7,1),("b35on",9,2)}
for c in ("wn1on","wn2on","b35on"):
    vs=[per[k] for k in per if k[0]==c and k not in LOST]
    print(f"  {c} 排除丢捕食者后 run 均值 {np.mean(vs):.0f}（n={len(vs)}）")
print("\n== 晚期（末1/4按代）逐 run：高侧是否恒为多数 ==")
flip=0; tot=0
for k,r in R.items():
    if not k[0].endswith("on"): continue
    late=r["gen"]>=r["gen"].max()*0.75
    H=r["hist"][late]; hn=(H/np.maximum(H.sum(1,keepdims=True),1e-9)).mean(0)
    lo=hn[LOW].sum(); hi=hn[CTR>0.65].sum(); tot+=1
    if lo>hi: flip+=1; print("  反例:",k,f"low={lo:.3f} high={hi:.3f}")
print(f"  低侧成为多数的 run：{flip}/{tot}")
print("\n== 平局种子的细节（两峰占比末1/4 on=off=1 的格） ==")
for (c,s) in [("wn1on",11),("wn2on",6),("b35on",9),("b35on",11)]:
    for r in (1,2):
        rec=R[(c,s,r)]; late=rec["gen"]>=rec["gen"].max()*0.75
        tw=np.mean([1.0 if (parts(h,CTR) and "minM" in parts(h,CTR)) else 0.0 for h in rec["hist"][late]])
        lm_=np.mean([h[LOW].sum()/max(h.sum(),1e-9) for h in rec["hist"][late]])
        cf=rec["carn"][np.isfinite(rec["carn"])][-1]
        print(f"  {c} s{s} r{r}: 末1/4两峰={tw:.3f} low_mass={lm_:.3f} 末帧carn={cf:.3f}")
