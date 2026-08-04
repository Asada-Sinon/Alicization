"""检查点窗口 [0.15,0.85] 之外有大量质量时，分布形状到底是什么样。
用 traj 的 clip 直方图（边缘箱 0 = 所有 pref<=0.20，箱 13 = 所有 pref>=0.80）打印剖面。
警告：边缘箱是**无界收集器**，单峰在 0.05 也会把箱 0 堆满 —— 所以只看「两端同时有质量 + 中间空」。
读 outputs/20260804-ratio/*.log。
"""
import glob, json, numpy as np
BINS=np.linspace(0.15,0.85,15); CTR=0.5*(BINS[:-1]+BINS[1:])
REC={}
for f in sorted(glob.glob("/home/xrl/intern/Alicization/outputs/20260804-ratio/*.log")):
    for ln in open(f):
        if ln.startswith("JSON "):
            b=f.split("/")[-1][:-4].split("_"); REC[(b[0],int(b[1][1:]),int(b[2][1:]))]=json.loads(ln[5:])
rows=list(REC)
def lost(k):
    t=REC[k]["traj"]; return any(p["carnivore_frac"]==0 for p in t[len(t)//2:])
def prof(k,i):
    t=REC[k]["checkpoints"][i]["t"]
    h=np.array(min(REC[k]["traj"],key=lambda p:abs(p["t"]-t))["hist"],float)
    return h/max(h.sum(),1)
print("箱中心：", "  ".join(f"{c:.2f}" for c in CTR))
print("（箱0 收所有 pref<=0.20，箱13 收所有 pref>=0.80；pref 低=偏果，高=偏草）\n")
for a in ("R38","R50"):
    for grp,sel in (("kept",lambda k: not lost(k)),("lost",lambda k: lost(k))):
        ks=[k for k in rows if k[0]==a and sel(k)]
        if not ks: continue
        for i,l in ((0,"cp0"),(1,"cp1")):
            m=np.mean([prof(k,i) for k in ks],axis=0)
            print(f"{a}/{grp} {l} n={len(ks):<3} " + " ".join(f"{v:.3f}" for v in m)
                  + f"   |  中间带[0.40,0.60]={m[(CTR>=0.40)&(CTR<=0.60)].sum():.3f}"
                  + f"  两端和={m[0]+m[13]:.3f}")
        print()
# 每个 run 单独判：两端都 >0.15 且中间带 <0.10 记为「两端有质量+中间空」
print("逐 run：两端各 >0.15 且中间带 <0.10 的计数（cp1）")
for a in ("R38","R50"):
    for grp,sel in (("kept",lambda k: not lost(k)),("lost",lambda k: lost(k))):
        ks=[k for k in rows if k[0]==a and sel(k)]
        if not ks: continue
        c=0
        for k in ks:
            m=prof(k,1)
            if m[0]>0.15 and m[13]>0.15 and m[(CTR>=0.40)&(CTR<=0.60)].sum()<0.10: c+=1
        print(f"  {a}/{grp}: {c}/{len(ks)}")
