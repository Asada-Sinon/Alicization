"""R11 复核第四轮：把 [E] 的 NaN 查清，并给 [F] 一个逐种子的读法。

 E'. 少数簇/主簇实测摄入，逐 run（不先按格平均，避免一个 rep 是 NaN 就整格作废）。
 F'. 逐种子的「左簇质量 vs 谷底质量」：dip_ratio 的固定带 [0.40,0.60] 之外还有没有分裂。
 I.  逐种子直方图的臂内异质性（中性臂 forage_pref 无选择、自由漂变，臂平均直方图会骗人）。

重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260804-r11-verdict/verify_r11d.py
"""
import glob, json, sys
sys.path.insert(0, "scripts")
import numpy as np
from scipy.stats import wilcoxon
from exp_stats import bootstrap_ci

R = "/home/xrl/intern/Alicization/"
SEEDS, REPS = list(range(12)), [1, 2]


def _load(pattern, remap=None):
    rec = {}
    for f in sorted(glob.glob(R + pattern)):
        for ln in open(f):
            if ln.startswith("JSON "):
                b = f.split("/")[-1][:-4].split("_")
                rec[((remap or {}).get(b[0], b[0]), int(b[1][1:]), int(b[2][1:]))] = json.loads(ln[5:])
    return rec


REC = _load("outputs/20260804-demand/*.log")
REC.update(_load("outputs/20260803-curvature/r10/[NK]*.log", remap={"N": "N15", "K10": "T15"}))
ARMS = ["N15", "T15", "N05", "T05", "N05m", "T05m"]
ctr = np.array(REC[("N05", 0, 1)]["bin_centers"])


def clumps(d):
    n = np.array(d["bin_n"], float)
    it = np.array([np.nan if v is None else v for v in d["intake"]], float)
    out = []
    for m0 in ((ctr < 0.35), (ctr > 0.65)):
        m = m0 & np.isfinite(it) & (n > 0)
        out += [float(n[m0].sum() / max(n.sum(), 1)), float((it[m] * n[m]).sum() / n[m].sum()) if n[m].sum() else np.nan]
    return out            # [低簇质量, 低簇摄入, 高簇质量, 高簇摄入]


print("[E'] 逐 run 的低簇/高簇实测摄入（last_food，能量/步；不含果实带的水）")
print(f"  {'臂':>5} {'可用run':>7} {'低簇摄入':>9} {'高簇摄入':>9} {'低/高':>7}  低<高 的 run 数")
paired = {}
for a in ARMS:
    rows = [clumps(REC[(a, s, r)]) for s in SEEDS for r in REPS]
    v = [(x[1], x[3]) for x in rows if np.isfinite(x[1]) and np.isfinite(x[3])]
    lo = np.array([x[0] for x in v]); hi = np.array([x[1] for x in v])
    paired[a] = (lo, hi)
    print(f"  {a:>5} {len(v):>7} {lo.mean():>9.4f} {hi.mean():>9.4f} {(lo/hi).mean():>7.3f}  "
          f"{int((lo < hi).sum())}/{len(v)}")
lo, hi = paired["T05"]
print(f"  T05: 低簇摄入 = 高簇的 {100*(lo/hi).mean():.1f}%（逐 run 比值范围 {(lo/hi).min():.2f}–{(lo/hi).max():.2f}）"
      f"  Wilcoxon(run级，非独立，仅描述) p={wilcoxon(lo, hi).pvalue:.2e}")
lo, hi = paired["T05m"]
print(f"  T05m: {100*(lo/hi).mean():.1f}%  逐 run 比值范围 {(lo/hi).min():.2f}–{(lo/hi).max():.2f}")

print("\n[F'] 逐种子：左簇质量(pref<0.30) vs 谷底质量(0.30≤pref<0.40)。谷底 < 左簇 ⇒ 左侧有一个分开的簇")
for a in ARMS:
    L, V = [], []
    for s in SEEDS:
        n = np.mean([np.array(REC[(a, s, r)]["bin_n"], float) / max(sum(REC[(a, s, r)]["bin_n"]), 1) for r in REPS], axis=0)
        L.append(n[ctr < 0.30].sum()); V.append(n[(ctr >= 0.30) & (ctr < 0.40)].sum())
    L, V = np.array(L), np.array(V)
    print(f"  {a:>5}  左簇均 {L.mean():.4f}  谷均 {V.mean():.4f}   谷<左簇 的种子 {int((V < L).sum())}/12   "
          f"逐种子左簇 {np.round(L,3).tolist()}")

print("\n[I] 臂内异质性：逐种子 mean_pref（中性臂的 forage_pref 无选择，自由漂变）")
for a in ARMS:
    v = np.array([np.mean([REC[(a, s, r)]["mean_pref"] for r in REPS]) for s in SEEDS])
    print(f"  {a:>5}  {v.mean():.4f}  种子间SD {v.std(ddof=1):.4f}  范围 [{v.min():.3f},{v.max():.3f}]")
