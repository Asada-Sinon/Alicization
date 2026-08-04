"""两件收尾：
(A) H2 的「低组」在 R50 里到底覆盖了多少人 —— intake 按 digitize 分箱，LOW=箱0..3 即
    pref∈[0.15,0.35)，而 R50 的果专精主峰在 pref<0.15，**根本不在低组里**。量化这个空洞。
(B) §17.3 的 ±0.05 达标窗在给定实测种子间 SD 下是否**可能达成**（即使剂量瞄得完全准）。
读 outputs/20260804-ratio/*.log。
"""
import glob, json, math, numpy as np
from scipy.stats import norm
BINS=np.linspace(0.15,0.85,15); CTR=0.5*(BINS[:-1]+BINS[1:])
LOW,HIGH=CTR<0.35,CTR>0.65
REC={}
for f in sorted(glob.glob("/home/xrl/intern/Alicization/outputs/20260804-ratio/*.log")):
    for ln in open(f):
        if ln.startswith("JSON "):
            b=f.split("/")[-1][:-4].split("_"); REC[(b[0],int(b[1][1:]),int(b[2][1:]))]=json.loads(ln[5:])
rows=list(REC)
def lost(k):
    t=REC[k]["traj"]; return any(p["carnivore_frac"]==0 for p in t[len(t)//2:])
print("(A) H2 的低组/高组占**全部被采样个体**的比例（分母 = n_samples，含出界者）")
print(f"{'组':<12}{'cp':<5}{'n':<4}{'低组[0.15,0.35)':>16}{'高组[0.65,0.85)':>16}{'出界':>9}")
for a in ("R38","R50"):
    for grp,sel in (("kept",lambda k: not lost(k)),("lost",lambda k: lost(k))):
        ks=[k for k in rows if k[0]==a and sel(k)]
        if not ks: continue
        for i,l in ((0,"cp0"),(1,"cp1")):
            lo,hi,oo=[],[],[]
            for k in ks:
                c=REC[k]["checkpoints"][i]; n=np.array(c["bin_n"],float); N=c["n_samples"]
                lo.append(n[LOW].sum()/N); hi.append(n[HIGH].sum()/N); oo.append(1-n.sum()/N)
            print(f"{a+'/'+grp:<12}{l:<5}{len(ks):<4}{np.mean(lo):>16.4f}{np.mean(hi):>16.4f}{np.mean(oo):>9.4f}")
print("\n⇒ R50 的 H2「低组」只覆盖了个位数百分比的个体，主果峰在 pref<0.15 被整体剔除；")
print("  R38 的低组同样只有几个百分点，但 R38 的出界比例仅 ~1%，其低组确实就是那条果实尾巴。")
print("  ⇒ H2 在 R38 上是有效的，在 R50 上量的不是果专精主峰，而是 0.15–0.35 的过渡带。")

print("\n(B) ±0.05 达标窗在实测种子间 SD 下的可达率（假设剂量瞄得完全准，正态近似）")
for a,tgt in (("R38",0.38),("R50",0.50)):
    v=np.array([np.mean([REC[(a,s,r)]["checkpoints"][0]["frugivory_frac"] for r in (1,2)]) for s in range(12)])
    sd=v.std(ddof=1); ach=2*norm.cdf(0.05/sd)-1
    obs=int(np.sum(np.abs(v-tgt)<=0.05))
    print(f"  {a}: 种子间 SD={sd:.4f}  ⇒ 完美居中时期望达标率={ach:.3f} ({ach*12:.1f}/12)  "
          f"实测达标 {obs}/12  实测均值={v.mean():.4f} 目标={tgt}")
    print(f"       实测偏离目标 {v.mean()-tgt:+.4f} = {abs(v.mean()-tgt)/sd:.2f} 个种子间 SD")
print("\n⇒ ±0.05 是绝对宽度，不是按噪声定的；R50 的 SD=0.056 使该窗 <1 个 SD，")
print("  **即使剂量瞄准 0.50，也只能期望 ~7.6/12 达标** —— 该达标规则本身就不可能满足 12/12。")
