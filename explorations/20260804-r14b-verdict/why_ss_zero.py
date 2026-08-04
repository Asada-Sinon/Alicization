"""为什么 clip 直方图上一个 51%/49% 的干净两端双峰读 split_score=0？逐步拆开看。
并用 split_score.retained()（803dc60 引入的质量+缺口判据）重算 R13 的留存曲线。
重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python explorations/20260804-r14b-verdict/why_ss_zero.py
"""
import glob, json, sys
sys.path.insert(0, "scripts"); sys.path.insert(0, "explorations/20260804-readouts")
import numpy as np
from split_score import split_score, retained, dip_ratio

BINS = np.linspace(0.15, 0.85, 15); CTR = 0.5*(BINS[:-1]+BINS[1:]); LOW = CTR < 0.35
runs = {}
for f in sorted(glob.glob("outputs/20260804-traj/*.log")):
    for ln in open(f):
        if ln.startswith("JSON "):
            runs[f.split("/")[-1][:-4]] = json.loads(ln[5:])
at = lambda r, t: [q for q in r["traj"] if q["t"] == t][-1]

h = np.array(at(runs["wn2_s7_r1"], runs["wn2_s7_r1"]["checkpoints"][4]["t"])["hist"], float)
p = h/h.sum()
print("wn2_s7_r1 cp4 clip 原始 %:", np.round(100*p, 2).tolist())
k = np.ones(3)/3
ps = np.convolve(np.pad(p, 1, mode="edge"), k, mode="valid"); ps = ps/ps.sum()
print("平滑后 %:", np.round(100*ps, 2).tolist())
loc = [i for i in range(len(ps)) if (i==0 or ps[i]>ps[i-1]) and (i==len(ps)-1 or ps[i]>ps[i+1])]
print("局部极大索引:", loc, " ⇒ len(loc)<2 ?", len(loc) < 2)
print("split_score =", split_score(h, CTR), "  low_mass =", round(p[LOW].sum(), 4),
      "  dip_ratio =", round(dip_ratio(h, CTR), 4), "  retained() =", retained(h, CTR))
h6 = np.array(at(runs["wn1_s6_r1"], runs["wn1_s6_r1"]["checkpoints"][4]["t"])["hist"], float)
p6 = h6/h6.sum()
ps6 = np.convolve(np.pad(p6,1,mode="edge"), k, mode="valid"); ps6 = ps6/ps6.sum()
loc6 = [i for i in range(len(ps6)) if (i==0 or ps6[i]>ps6[i-1]) and (i==len(ps6)-1 or ps6[i]>ps6[i+1])]
print("\nwn1_s6_r1 cp4 clip 原始 %:", np.round(100*p6,2).tolist())
print("  平滑后 %:", np.round(100*ps6,2).tolist(), " 局部极大:", loc6)
print("  split_score =", split_score(h6, CTR), " low_mass =", round(p6[LOW].sum(),4),
      " dip_ratio =", round(dip_ratio(h6,CTR),4), " retained() =", retained(h6, CTR))

print("\n" + "="*96)
print("用 retained()（质量>0.03 且 dip_ratio<0.5）重算 R13 留存曲线，两种口径")
print("="*96)
for tag, getter in (("截断 bin_n", lambda r,i: np.array(r["checkpoints"][i]["bin_n"], float)),
                    ("clip hist ", lambda r,i: np.array(at(r, r["checkpoints"][i]["t"])["hist"], float))):
    tab = {name: [retained(getter(r,i), CTR) for i in range(5)] for name, r in runs.items()}
    curve = [sum(v[i] for v in tab.values()) for i in range(5)]
    ent = {n:v for n,v in tab.items() if v[0]}
    cont = sum(1 for v in ent.values() if all(v))
    cont24 = sum(1 for v in tab.values() if all(v))
    print(f"  {tag}: 留存曲线(分母24) {curve}   入组 {len(ent)}/24   连续五点全保住 {cont}/{len(ent)}"
          f"（全体口径 {cont24}/24）")
    print("     逐 run: " + "  ".join(f"{n.replace('_r1',''):>7}:" + "".join("O" if x else "." for x in v)
                                      for n, v in sorted(tab.items())))
