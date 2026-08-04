"""检查 `low_mass` 的截断口径：检查点用 np.digitize(p,BINS)-1（**剔除** pref<0.15 与 >0.85），
而 traj 的 `_hist_fn` 用 clip（**把尾巴压进边缘箱**）。两者若不一致，说明有质量跑出 [0.15,0.85]。
读 outputs/20260804-ratio/*.log。输出：每个臂 × lost/kept 的 cp0/cp1 截断比例与两种口径的 low_mass。
重跑同目录 verdict_r14.py 前置。
"""
import glob, json, numpy as np
BINS = np.linspace(0.15, 0.85, 15); CTR = 0.5*(BINS[:-1]+BINS[1:])
REC = {}
for f in sorted(glob.glob("/home/xrl/intern/Alicization/outputs/20260804-ratio/*.log")):
    for ln in open(f):
        if ln.startswith("JSON "):
            b = f.split("/")[-1][:-4].split("_"); REC[(b[0],int(b[1][1:]),int(b[2][1:]))] = json.loads(ln[5:])
rows = list(REC)
def lost(k):
    t = REC[k]["traj"]; return any(p["carnivore_frac"]==0 for p in t[len(t)//2:])
print(f"{'臂/组':<12}{'cp':<4}{'n':<4}{'落在[0.15,0.85]内的比例':>22}{'cp内 low_mass':>15}{'traj(clip) low_mass':>21}")
for a in ("R38","R50"):
    for grp, sel in (("kept", lambda k: not lost(k)), ("lost", lambda k: lost(k))):
        ks = [k for k in rows if k[0]==a and sel(k)]
        if not ks: continue
        for i,lbl in ((0,"cp0"),(1,"cp1")):
            keep, lmv, tv = [], [], []
            for k in ks:
                cp = REC[k]["checkpoints"][i]
                n = np.array(cp["bin_n"],float)
                keep.append(n.sum()/cp["n_samples"]); lmv.append(cp["low_mass"])
                # 找 traj 里时间最接近该检查点的那一点（traj 用 clip 口径）
                best = min(REC[k]["traj"], key=lambda p: abs(p["t"]-cp["t"]))
                h = np.array(best["hist"],float); tv.append(h[CTR<0.35].sum()/max(h.sum(),1))
            print(f"{a+'/'+grp:<12}{lbl:<4}{len(ks):<4}{np.mean(keep):>22.4f}"
                  f"{np.mean(lmv):>15.4f}{np.mean(tv):>21.4f}")
print("\n若『落在内的比例』远小于 1 ⇒ 质量跑到 pref<0.15（更极端的果专精）或 >0.85，")
print("而检查点口径把它们从分子分母同时剔除 ⇒ 检查点 low_mass 的下跌部分是**截断**而非消失。")
# 直接看跑出下边界还是上边界：用 traj 的边缘箱占比
print(f"\n{'臂/组':<12}{'cp':<4}{'边缘箱0(pref<=0.20)占比':>24}{'边缘箱13(pref>=0.80)占比':>25}")
for a in ("R38","R50"):
    for grp, sel in (("kept", lambda k: not lost(k)), ("lost", lambda k: lost(k))):
        ks = [k for k in rows if k[0]==a and sel(k)]
        if not ks: continue
        for i,lbl in ((0,"cp0"),(1,"cp1")):
            b0,b13 = [],[]
            for k in ks:
                cp = REC[k]["checkpoints"][i]
                best = min(REC[k]["traj"], key=lambda p: abs(p["t"]-cp["t"]))
                h = np.array(best["hist"],float); s=max(h.sum(),1)
                b0.append(h[0]/s); b13.append(h[13]/s)
            print(f"{a+'/'+grp:<12}{lbl:<4}{np.mean(b0):>24.4f}{np.mean(b13):>25.4f}")
