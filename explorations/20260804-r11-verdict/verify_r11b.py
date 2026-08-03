"""R11 复核第二轮：把三个技术问号钉死。判据 = docs/multispecies_program.md §15。

 A. 平均占据直方图（逐箱），看「第二个模态」到底有没有质量、在哪。
 B. §15.3 预注册但因 R10 臂缺字段而在主分析里读出 nan 的 frac_mid —— 从 bin_n 补算。
 C. quad_intake 到底为什么翻号：
    - 它是**个体级**回归 last_food ~ 1 + z + z²，z=(pref−mean)/pref_sd（fitness_surface.py:185-200），
      不是分箱回归；分裂后 x 变成两个团簇 ⇒ 二次项近乎不可辨识（条件数）。
    - 更硬的一条：z 用**本臂自己的 sd** 标准化，quad ∝ sd²，跨臂不可比。这里把它换算回 x 单位。
    - 饱和探针 S1 的 8 个 run 逐个摊开（n=4 种子，低于地板）。
 D. §15.6 声称「population 改窗口时均能救护栏」——直接量两个读数的格内 σ_W（同臂、同 df）。

重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260804-r11-verdict/verify_r11b.py
"""
import glob, json, math, sys
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

print("[A] 平均占据直方图（每臂 24 run 的归一化 bin_n 平均，单位 %）")
print("      箱心 " + " ".join(f"{c:>5.2f}" for c in ctr))
for a in ARMS:
    h = np.mean([np.array(REC[(a, s, r)]["bin_n"], float) / max(sum(REC[(a, s, r)]["bin_n"]), 1)
                 for s in SEEDS for r in REPS], axis=0)
    print(f"  {a:>5}  " + " ".join(f"{100*x:>5.1f}" for x in h))
print("  （高 pref = 草专精；低 pref = 果专精。中间带 = 箱心 0.425/0.475/0.525/0.575）")

print("\n[B] frac_mid（§15.3 预注册，主分析因 R10 臂缺字段读出 nan）—— 从 bin_n 补算")
fm = lambda d: float(np.array(d["bin_n"], float)[(ctr >= 0.40) & (ctr <= 0.60)].sum()
                     / max(sum(d["bin_n"]), 1))
cell = lambda a, g: np.array([np.mean([g(REC[(a, s, r)]) for r in REPS]) for s in SEEDS])
for a in ARMS:
    v = cell(a, fm)
    print(f"  {a:>5}  {v.mean():.4f} ± {v.std(ddof=1)/math.sqrt(12):.4f}")
x, y = cell("T05", fm) - cell("N05", fm), cell("T15", fm) - cell("N15", fm)
d = x - y
print(f"  交互 = {d.mean():+.4f}  符号 {int((d<0).sum())}−/12  p={wilcoxon(x,y).pvalue:.5f}  "
      f"CI={np.round(bootstrap_ci(d),4).tolist()}  经验逐种子SD={d.std(ddof=1):.4f} ⇒ 比值 {d.mean()/d.std(ddof=1):+.2f}")

print("\n[C] quad_intake 的可比性")
print(f"  {'臂':>5} {'quad(z单位)':>13} {'pref_sd':>9} {'occ_sd(分箱)':>12} {'quad/occ_sd²(x单位)':>20} {'quadrel':>10}")
for a in ARMS:
    q = cell(a, lambda d: d["quad_intake"])
    osd = cell(a, lambda d: float(np.sqrt(max(np.sum(np.array(d["bin_n"], float) / max(sum(d["bin_n"]), 1)
                * (ctr - np.sum(np.array(d["bin_n"], float) / max(sum(d["bin_n"]),1) * ctr))**2), 1e-12))))
    psd = cell(a, lambda d: d.get("pref_sd", float("nan")))
    qx = cell(a, lambda d: d["quad_intake"]) / osd**2
    qr = cell(a, lambda d: d["quadrel_intake"])
    print(f"  {a:>5} {q.mean():>+13.5f} {np.nanmean(psd):>9.4f} {osd.mean():>12.4f} {qx.mean():>+20.4f} {qr.mean():>+10.4f}")
qxa = {a: cell(a, lambda d: d["quad_intake"]) / cell(a, lambda d: float(np.sqrt(max(np.sum(
        np.array(d["bin_n"],float)/max(sum(d["bin_n"]),1)*(ctr-np.sum(np.array(d["bin_n"],float)/max(sum(d["bin_n"]),1)*ctr))**2),1e-12))))**2 for a in ARMS}
dx = (qxa["T05"] - qxa["N05"]) - (qxa["T15"] - qxa["N15"])
print(f"  x 单位下的交互 = {dx.mean():+.4f}  符号 {int((dx<0).sum())}−/12  "
      f"p={wilcoxon(qxa['T05']-qxa['N05'], qxa['T15']-qxa['N15']).pvalue:.5f}  "
      f"CI={np.round(bootstrap_ci(dx),3).tolist()}  经验SD={dx.std(ddof=1):.4f} ⇒ 比值 {dx.mean()/dx.std(ddof=1):+.2f}")

print("\n  饱和探针 S1（= T05 的条件，但 n=4 种子 × 2 重复，低于协议地板）")
sat = {}
for f in sorted(glob.glob(R + "outputs/20260803-curvature/sat/S*.log")):
    for ln in open(f):
        if ln.startswith("JSON "):
            b = f.split("/")[-1][:-4].split("_")
            sat[(b[0], int(b[1][1:]), int(b[2][1:]))] = json.loads(ln[5:])
for arm in ("S0", "S1", "S2"):
    v = np.array([np.mean([sat[(arm, s, r)]["quad_intake"] for r in (1, 2)]) for s in range(4)])
    runs = [sat[(arm, s, r)]["quad_intake"] for s in range(4) for r in (1, 2)]
    print(f"  {arm}  格均值 {v.mean():+.5f}  逐种子 {np.round(v,5).tolist()}  正号 {int((v>0).sum())}/4 种子, "
          f"{sum(1 for x in runs if x>0)}/8 run")
print("  n=4 配对符号秩的最小双侧 p = 2/2^4 = 0.125 ⇒ **S1 的正曲率在它自己的设计里就不可能显著**")

print("\n[D] §15.6 的护栏修复是否成立：同臂同 df 比 population 与 population_mean 的格内 σ_W")
for m in ("population", "population_mean"):
    w = np.concatenate([[np.std([REC[(a, s, r)][m] for r in REPS], ddof=1) for s in SEEDS]
                        for a in ("N05", "T05", "N05m", "T05m")])
    mu = np.mean([REC[(a, s, r)][m] for a in ("N05", "T05", "N05m", "T05m") for s in SEEDS for r in REPS])
    sw = float(np.sqrt((w**2).mean()))
    print(f"  {m:<16} σ̂_W={sw:8.1f}  四臂总均值={mu:8.1f}  σ̂_W/均值={100*sw/mu:5.1f}%  "
          f"（√2σ̂_W/√r 点估容差 ±{math.sqrt(2)*sw/math.sqrt(2):.0f}）")
