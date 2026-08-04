"""交叉验证「R50/kept 在 cp1 是两端双峰」这条**事后**读数：
用两个与直方图无关的量 —— 检查点里的 `mean_pref` 与 `sd`（都对原始 pref 数组算，不分箱）——
看它们是否与「0.1 / 0.85 两点各占一半」的混合自洽；再看两端质量的时间演化。
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
print("检查点的 mean_pref / sd（**不分箱**，与直方图独立）")
print(f"{'组':<12}{'cp':<5}{'n':<4}{'mean_pref':>11}{'sd':>9}   与混合模型的自洽性")
for a in ("R38","R50"):
    for grp,sel in (("kept",lambda k: not lost(k)),("lost",lambda k: lost(k))):
        ks=[k for k in rows if k[0]==a and sel(k)]
        if not ks: continue
        for i,l in ((0,"cp0"),(1,"cp1")):
            mp=np.mean([REC[k]["checkpoints"][i]["mean_pref"] for k in ks])
            sd=np.mean([REC[k]["checkpoints"][i]["sd"] for k in ks])
            # 若是 w 比例在 lo、(1-w) 在 hi 的两点混合，sd_max = sqrt(w(1-w))*(hi-lo)
            note=""
            if sd>0.30: note="sd>0.30 ⇒ 只能由两端质量产生（单峰在 [0,1] 内很难到）"
            print(f"{a+'/'+grp:<12}{l:<5}{len(ks):<4}{mp:>11.4f}{sd:>9.4f}   {note}")
print("\n参照：0.10 与 0.85 各半的两点混合 ⇒ mean=0.475, sd=0.375")
print("      单峰 Beta 在 [0,1] 上 sd 要到 0.30 需极端 U 型\n")
print("两端质量的时间演化（traj clip 直方图，箱0=pref<=0.20，箱13=pref>=0.80）")
print(f"{'组':<12}" + "".join(f"{int(f*44):>7}k" for f in (0.25,0.5,0.75,0.9,1.0)))
for a in ("R38","R50"):
    for grp,sel in (("kept",lambda k: not lost(k)),("lost",lambda k: lost(k))):
        ks=[k for k in rows if k[0]==a and sel(k)]
        if not ks: continue
        for lbl,fn in (("  低端箱0",lambda h:h[0]/max(h.sum(),1)),
                       ("  高端箱13",lambda h:h[13]/max(h.sum(),1)),
                       ("  中间.40-.60",lambda h:h[(CTR>=0.40)&(CTR<=0.60)].sum()/max(h.sum(),1))):
            vals=[]
            for f in (0.25,0.5,0.75,0.9,1.0):
                v=[]
                for k in ks:
                    tt=REC[k]["traj"][-1]["t"]
                    c=[p for p in REC[k]["traj"] if abs(p["t"]-f*tt)<900]
                    if c: v.append(fn(np.array(c[0]["hist"],float)))
                vals.append(np.mean(v) if v else np.nan)
            print(f"{a+'/'+grp+lbl:<24}" + "".join(f"{v:>8.3f}" for v in vals))
        print()
