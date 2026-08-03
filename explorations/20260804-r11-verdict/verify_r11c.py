"""R11 复核第三轮：四件在前两轮里冒出来的事。

 E. 少数簇 vs 主簇的**实测摄入**（bin_n 加权的 intake），逐臂。
    T05 的 s0r1 raw 读数：低簇 0.078–0.097、高簇 0.148–0.153 —— 若普遍成立，
    「两个等价生计」就说不出口。注意 last_food = food_gain + fruit_gain 是**能量**
    （state.py:31），**不含果实带的水**（fruit_water_frac=0.40），而本世界 55–62% 的死
    是渴死 ⇒ 能量不是适应度。
 F. T15 自己有没有一个 dip_ratio 的固定带 [0.40,0.60] 看不见的分裂（平均直方图提示有）。
    这决定参照对比 (T15−N15)=+0.129 能不能读成「饱和世界里没分裂」。
 G. T05m 的「部分塌」到底是「深度打折」还是「有的种子塌有的不塌」。
 H. 单臂对比 (T15−N15) 与 (T05−N05) 各自的配对检验（主分析只报了交互）。

重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260804-r11-verdict/verify_r11c.py
"""
import glob, json, math, sys
sys.path.insert(0, "scripts")
import numpy as np
from scipy.stats import norm, wilcoxon
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
cell = lambda a, g: np.array([np.mean([g(REC[(a, s, r)]) for r in REPS]) for s in SEEDS])


def clump_intake(d, lo):
    n = np.array(d["bin_n"], float)
    it = np.array([np.nan if v is None else v for v in d["intake"]], float)
    m = (ctr < 0.35) if lo else (ctr > 0.65)
    m = m & np.isfinite(it) & (n > 0)
    return float((it[m] * n[m]).sum() / n[m].sum()) if n[m].sum() > 0 else float("nan")


print("[E] 少数簇（pref<0.35）与主簇（pref>0.65）的 bin_n 加权实测摄入 last_food（能量/步）")
print(f"  {'臂':>5} {'低簇摄入':>10} {'高簇摄入':>10} {'低/高':>8} {'低簇质量':>10} {'高簇质量':>10}  逐种子低/高比")
for a in ARMS:
    lo, hi = cell(a, lambda d: clump_intake(d, True)), cell(a, lambda d: clump_intake(d, False))
    lm = cell(a, lambda d: float(np.array(d["bin_n"], float)[ctr < 0.35].sum() / max(sum(d["bin_n"]), 1)))
    hm = cell(a, lambda d: float(np.array(d["bin_n"], float)[ctr > 0.65].sum() / max(sum(d["bin_n"]), 1)))
    ratio = lo / hi
    print(f"  {a:>5} {np.nanmean(lo):>10.4f} {np.nanmean(hi):>10.4f} {np.nanmean(ratio):>8.3f} "
          f"{lm.mean():>10.4f} {hm.mean():>10.4f}  {np.round(ratio,2).tolist()}")
lo05, hi05 = cell("T05", lambda d: clump_intake(d, True)), cell("T05", lambda d: clump_intake(d, False))
ok = np.isfinite(lo05) & np.isfinite(hi05)
print(f"  T05 低簇 < 高簇 的种子数 {int((lo05[ok] < hi05[ok]).sum())}/{int(ok.sum())}  "
      f"配对 p={wilcoxon(lo05[ok], hi05[ok]).pvalue:.5f}  差 CI={np.round(bootstrap_ci(lo05[ok]-hi05[ok]),4).tolist()}")

print("\n[F] T15 的分裂在 dip_ratio 的固定带外面吗")
print("  各臂：低簇质量(pref<0.30) / 谷底质量(0.30≤pref<0.40) / 谷深 = 谷底 ÷ 两侧较小者")
for a in ARMS:
    def parts(d):
        n = np.array(d["bin_n"], float); n = n / max(n.sum(), 1.0)
        left = n[ctr < 0.30].sum(); vall = n[(ctr >= 0.30) & (ctr < 0.40)].sum(); right = n[ctr >= 0.40].sum()
        return left, vall, right, vall / max(min(left, right), 1e-9)
    L = cell(a, lambda d: parts(d)[0]); V = cell(a, lambda d: parts(d)[1]); D = cell(a, lambda d: parts(d)[3])
    print(f"  {a:>5}  左簇 {L.mean():.4f}  谷 {V.mean():.4f}  谷深比 {D.mean():.3f}   "
          f"谷深比<0.5 的种子 {int((D < 0.5).sum())}/12")
print("  （dip_ratio 的带是 [0.40,0.60]；上面这个谷在 [0.30,0.40)，**在带外**。post-hoc 诊断，非判据。）")

print("\n[G] T05m 的「部分塌」：深度打折 还是 种子分两类")
dip = lambda a: cell(a, lambda d: d["dip_ratio"] if "dip_ratio" in d else np.nan)


def dip_bins(d):
    n = np.array(d["bin_n"], float); n = n / max(n.sum(), 1.0)
    m = float((n * ctr).sum()); sd = math.sqrt(max(float((n * (ctr - m) ** 2).sum()), 1e-12))
    return float(n[(ctr >= 0.40) & (ctr <= 0.60)].sum()) / max(norm.cdf(0.60, m, sd) - norm.cdf(0.40, m, sd), 1e-9)


for a in ("T05", "T05m"):
    v = cell(a, dip_bins)
    runs = np.array([dip_bins(REC[(a, s, r)]) for s in SEEDS for r in REPS])
    print(f"  {a:>5}  格均值<0.3: {int((v<0.3).sum())}/12 种子   run<0.3: {int((runs<0.3).sum())}/24   "
          f"未塌的格均值 {np.round(np.sort(v)[::-1][:4],3).tolist()}")
print("  T05m 的 12 个格均值升序：", np.round(np.sort(cell("T05m", dip_bins)), 3).tolist())

print("\n[H] 单臂对比的配对检验（主分析只报交互）")
for ta, na in (("T15", "N15"), ("T05", "N05"), ("T05m", "N05m")):
    x, y = cell(ta, dip_bins), cell(na, dip_bins)
    d = x - y
    print(f"  ({ta}−{na}) = {d.mean():+.4f}  符号 {int((d<0).sum())}−/12  p={wilcoxon(x,y).pvalue:.5f}  "
          f"CI={np.round(bootstrap_ci(d),4).tolist()}  经验SD={d.std(ddof=1):.4f}")
for ta, na in (("T15", "N15"), ("T05", "N05")):
    x, y = cell(ta, lambda d: d["mean_pref"]), cell(na, lambda d: d["mean_pref"])
    d = x - y
    print(f"  mean_pref ({ta}−{na}) = {d.mean():+.4f}  符号 {int((d>0).sum())}+/12  p={wilcoxon(x,y).pvalue:.5f}")
