"""三个补充读数：(1) 入组条件造成的回归均值污染，(2) 世代长的三个离群 run，
(3) low_mass 的连续趋势（避开 cp0 的入组选择）。

读：outputs/20260804-traj/*.log
重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260804-r13-verdict/extra_tests.py
"""
import glob, json, sys
import numpy as np
from scipy.stats import wilcoxon, spearmanr

runs = []
for f in sorted(glob.glob("outputs/20260804-traj/*.log")):
    for ln in open(f):
        if ln.startswith("JSON "):
            d = json.loads(ln[5:]); d["_file"] = f.split("/")[-1]
            d["_wn"] = int(d["_file"].split("_")[0][2:]); runs.append(d)
ent = [r for r in runs if r["checkpoints"][0]["split_score"] > 0]

lm = lambda r: np.array([cp["low_mass"] for cp in r["checkpoints"]])

print("[1] 入组条件在 cp0 上做了选择 ⇒ cp0→cp4 的下降含回归均值成分")
a = np.array([lm(r)[0] for r in ent]); b = np.array([lm(r)[4] for r in ent])
d = b - a
print(f"  入组 15：cp0 {a.mean():.4f} → cp4 {b.mean():.4f}  差 {d.mean():+.4f}  "
      f"SD(差) {d.std(ddof=1):.4f}  效应/噪声 {d.mean()/d.std(ddof=1):+.2f}  "
      f"{int((d<0).sum())}/15 降  p={wilcoxon(a,b).pvalue:.5f}")
print(f"  未入组 9 个 run 的 cp0 low_mass 均值 {np.mean([lm(r)[0] for r in runs if r not in ent]):.4f}"
      f"  ⇒ 入组把 cp0 的 low_mass 从全体 {np.mean([lm(r)[0] for r in runs]):.4f} 拉到 {a.mean():.4f}")

print("\n[2] 避开入组选择：全部 24 run，cp1 → cp4（两端都不是入组判定点）")
a1 = np.array([lm(r)[1] for r in runs]); b1 = np.array([lm(r)[4] for r in runs])
d1 = b1 - a1
print(f"  cp1 {a1.mean():.4f} → cp4 {b1.mean():.4f}  差 {d1.mean():+.4f}  "
      f"SD(差) {d1.std(ddof=1):.4f}  效应/噪声 {d1.mean()/d1.std(ddof=1):+.2f}  "
      f"{int((d1<0).sum())}/24 降 {int((d1>0).sum())}/24 升 {int((d1==0).sum())}/24 平  "
      f"p={wilcoxon(a1,b1).pvalue:.5f}")
nz = d1[d1 != 0]
print(f"  只看非零变化的 {len(nz)} 个 run：{int((nz<0).sum())} 降 / {int((nz>0).sum())} 升  "
      f"符号检验 p={2*min(np.sum(nz<0), np.sum(nz>0))/2**len(nz) if len(nz)<20 else float('nan'):.5f}(近似)")

print("\n[2b] 同法但只在入组的 15 个上（cp1→cp4）")
a2 = np.array([lm(r)[1] for r in ent]); b2 = np.array([lm(r)[4] for r in ent])
d2 = b2 - a2
print(f"  cp1 {a2.mean():.4f} → cp4 {b2.mean():.4f}  差 {d2.mean():+.4f}  "
      f"SD(差) {d2.std(ddof=1):.4f}  效应/噪声 {d2.mean()/d2.std(ddof=1):+.2f}  "
      f"{int((d2<0).sum())}/15 降  p={wilcoxon(a2,b2).pvalue:.5f}")

print("\n[3] 逐 run 的 low_mass 对 cp 的斜率（cp1..cp4，避开入组点）")
sl = np.array([np.polyfit([1,2,3,4], lm(r)[1:], 1)[0] for r in runs])
print(f"  全 24 run：均值 {sl.mean():+.5f}  中位 {np.median(sl):+.5f}  "
      f"{int((sl<0).sum())}/24 为负 {int((sl==0).sum())}/24 恒 0  "
      f"SD {sl.std(ddof=1):.5f}  效应/噪声 {sl.mean()/sl.std(ddof=1):+.2f}  "
      f"p={wilcoxon(sl).pvalue:.5f}")
sle = np.array([np.polyfit([1,2,3,4], lm(r)[1:], 1)[0] for r in ent])
print(f"  入组 15：均值 {sle.mean():+.5f}  {int((sle<0).sum())}/15 为负  "
      f"SD {sle.std(ddof=1):.5f}  效应/噪声 {sle.mean()/sle.std(ddof=1):+.2f}  "
      f"p={wilcoxon(sle).pvalue:.5f}")

print("\n[4] 世代长离群：gen_total 与 low_mass 丢失快慢")
for r in sorted(runs, key=lambda r: r["gen_total"])[:5]:
    g = [q["generation"] for q in r["traj"]]
    t = [q["t"] for q in r["traj"]]
    q1 = np.interp(25000, t, g); q3 = np.interp(75000, t, g)
    print(f"  {r['_file']:<16} gen_total {r['gen_total']:6.1f}  "
          f"gen@25k {q1:6.1f} gen@75k {q3:6.1f}  后半段速率 {(g[-1]-q3)/25000*1000:.2f} 代/千步  "
          f"前段 {q1/25000*1000:.2f} 代/千步   最小 n_herb {min(q['n_herb'] for q in r['traj'])}")
gt = np.array([r["gen_total"] for r in runs])
lost = np.array([sum(1 for cp in r["checkpoints"][1:] if cp["low_mass"] < 0.02) for r in runs])
print(f"  gen_total 与「cp1..4 中 low_mass<0.02 的个数」Spearman "
      f"rho={spearmanr(gt, lost).statistic:+.3f} p={spearmanr(gt, lost).pvalue:.4f}")
print(f"  gen_total 中位 {np.median(gt):.0f} ⇒ 步/代 {100000/np.median(gt):.1f}；"
      f"均值 {gt.mean():.0f} ⇒ {100000/gt.mean():.1f}；§16.1 预注册用的是 169")

print("\n[5] mean_pref / pref_sd 的逐 run 首末（方向性移动 vs 漂变）")
mp = np.array([[cp["mean_pref"] for cp in r["checkpoints"]] for r in runs])
sd = np.array([[cp["sd"] for cp in r["checkpoints"]] for r in runs])
for name, arr in (("mean_pref", mp), ("pref_sd", sd)):
    d = arr[:, 4] - arr[:, 0]
    print(f"  {name}: cp0 {arr[:,0].mean():.4f} → cp4 {arr[:,4].mean():.4f}  差 {d.mean():+.4f}  "
          f"{int((d>0).sum())}/24 为正  p={wilcoxon(arr[:,0], arr[:,4]).pvalue:.5f}  "
          f"效应/噪声 {d.mean()/d.std(ddof=1):+.2f}")
