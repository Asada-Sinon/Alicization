"""R14 Stage 2（方案 B）判决。判据见 `docs/multispecies_program.md` §18（跑之前已提交 `df4b3ed`）。

四臂：`R38p`/`R50p`（有捕食者）、`R38n`/`R50n`（`diet_delta=1.5`，**由构造**无捕食者）。
- **H0 操作检查**：`low_mass(cp0)`，`R50n` vs `R38n`。不成立 ⇒ 整轮读数作废。
- **H1 主判据（干净对比）**：`Δlow_mass`，`R50n` vs `R38n`。预测 `R50n > R38n`。
  **§18.4 已预声明：比值<1 判 undecidable，既不许读成「衰减停了」也不许读成「衰减照旧」。**
- **H2 交互**：`(R50n−R38n) − (R50p−R38p)`，噪声按四臂算成 `2·σ̂_W/√r`。
- **有效性闸（§18.5）**：任一 run `in_window<0.95` ⇒ 该 run 分箱读数不可用；
  某臂 >2/24 失效 ⇒ 该臂分箱判据整体作废。
- **总通量当协变量（§18.6）**：若臂间差 >1 个种子间 SD，归因判为多变量。

重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260804-readouts/analyze_r14b.py
"""
import glob
import json
import math
import sys

sys.path.insert(0, "scripts")

import numpy as np
from scipy.stats import chi2, wilcoxon

from exp_stats import bootstrap_ci

CTR = np.linspace(0.0, 1.0, 21)
CTR = 0.5 * (CTR[:-1] + CTR[1:])
LOW, HIGH = CTR < 0.35, CTR > 0.65
SEEDS, REPS = list(range(12)), [1, 2]
ARMS = ["R38p", "R50p", "R38n", "R50n"]
SPEC = {"R38p": (0.25, 0.010, 0.15), "R50p": (0.50, 0.005, 0.15),
        "R38n": (0.25, 0.010, 1.5), "R50n": (0.50, 0.005, 1.5)}

REC = {}
for f in sorted(glob.glob("outputs/20260804-ratio2/*.log")):
    for ln in open(f):
        if ln.startswith("JSON "):
            b = f.split("/")[-1][:-4].split("_")
            REC[(b[0], int(b[1][1:]), int(b[2][1:]))] = json.loads(ln[5:])
miss = [(a, s, r) for a in ARMS for s in SEEDS for r in REPS if (a, s, r) not in REC]
if miss:
    print(f"!! 缺 {len(miss)} 个 run: {miss[:6]}")
    sys.exit(1)
bad = [(a, s, r) for a in ARMS for s in SEEDS for r in REPS
       if tuple(float(REC[(a, s, r)]["overrides"].get(k, -1))
                for k in ("fruit_regrow_baseline", "regrow_baseline", "diet_delta")) != SPEC[a]]
if bad:
    print(f"!! 臂归因自检不通过: {bad[:5]}")
    sys.exit(1)
print(f"载入 {len(REC)} run，臂归因自检通过\n")

print("=" * 92)
print("§18.5 有效性闸：in_window < 0.95 ⇒ 该 run 分箱读数不可用")
print("=" * 92)
for a in ARMS:
    iw = [cp["in_window"] for s in SEEDS for r in REPS for cp in REC[(a, s, r)]["checkpoints"]]
    inv = sum(1 for s in SEEDS for r in REPS
              if any(not cp["readout_valid"] for cp in REC[(a, s, r)]["checkpoints"]))
    print(f"  {a:>6}  窗内比例 min={min(iw):.4f} 均值={np.mean(iw):.4f}   失效 run {inv}/24"
          f"{'   ** >2 ⇒ 该臂分箱判据作废 **' if inv > 2 else ''}")

print("\n" + "=" * 92)
print("§18.6 总通量当协变量：若臂间差 > 1 个种子间 SD，归因判为多变量")
print("=" * 92)
flux, frug, pop = {}, {}, {}
for a in ARMS:
    flux[a] = np.array([np.mean([REC[(a, s, r)]["checkpoints"][0].get("graze_gain", 0)
                                 + REC[(a, s, r)]["checkpoints"][0].get("fruit_gain", 0)
                                 for r in REPS]) for s in SEEDS])
    frug[a] = np.array([np.mean([REC[(a, s, r)]["checkpoints"][0].get("frugivory_frac", np.nan)
                                 for r in REPS]) for s in SEEDS])
    pop[a] = np.array([np.mean([p["population"] for p in REC[(a, s, r)]["traj"]
                                if np.isfinite(p["population"])]) for r in REPS for s in SEEDS])
    print(f"  {a:>6}  总通量 {flux[a].mean():>7.1f} (SD {flux[a].std(ddof=1):>6.1f})   "
          f"frugivory {frug[a].mean():.4f}   种群 {pop[a].mean():>6.0f}")
for pair in (("R50n", "R38n"), ("R50p", "R38p")):
    d = flux[pair[0]].mean() - flux[pair[1]].mean()
    sd = max(flux[pair[0]].std(ddof=1), flux[pair[1]].std(ddof=1))
    print(f"  {pair[0]}−{pair[1]} 总通量差 {d:+.1f}，1 个种子间 SD = {sd:.1f}   "
          f"{'** >1 SD ⇒ 归因判为多变量 **' if abs(d) > sd else 'ok（配比可单独归因）'}")


def val(a, s, r, key):
    cps = REC[(a, s, r)]["checkpoints"]
    if key == "lm0":
        return cps[0]["low_mass"]
    if key == "dlm":
        return cps[1]["low_mass"] - cps[0]["low_mass"]
    if key == "ratio":
        ik = np.array([np.nan if v is None else v for v in cps[0]["intake"]], float)
        n = np.array(cps[0]["bin_n"], float)
        x, y = LOW & np.isfinite(ik) & (n > 0), HIGH & np.isfinite(ik) & (n > 0)
        return np.nan if not (x.any() and y.any()) else float(
            (ik[x] * n[x]).sum() / n[x].sum() / ((ik[y] * n[y]).sum() / n[y].sum()))
    return cps[0].get(key, np.nan)


cell = lambda a, k: np.array([np.nanmean([val(a, s, r, k) for r in REPS]) for s in SEEDS])
within = lambda a, k: np.array([np.nanstd([val(a, s, r, k) for r in REPS], ddof=1) for s in SEEDS])


def report(key, a, b, label, pred, noise_arms=None, interaction=None):
    if interaction:
        x = cell(interaction[0], key) - cell(interaction[1], key)
        y = cell(interaction[2], key) - cell(interaction[3], key)
        k = 2.0
    else:
        x, y, k = cell(a, key), cell(b, key), math.sqrt(2.0)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    d = x - y
    w = np.concatenate([within(z, key) for z in (noise_arms or ARMS)])
    w = w[np.isfinite(w)]
    sw = float(np.sqrt((w ** 2).mean()))
    sw_ub = sw * math.sqrt(len(w) / chi2.ppf(0.10, len(w)))
    noise = sw_ub * k / math.sqrt(len(REPS))
    lo, hi = bootstrap_ci(d)
    p = wilcoxon(x, y).pvalue
    print(f"\n  --- {label}（预测 {pred}）---")
    print(f"    {'左' if interaction else a} = {x.mean():+.4f}   "
          f"{'右' if interaction else b} = {y.mean():+.4f}   差 = {d.mean():+.4f}")
    print(f"    方向 {int((d > 0).sum())}/{len(d)} 为正   p={p:.5f}   "
          f"95%CI=[{lo:+.4f}, {hi:+.4f}]{'  **含 0**' if lo <= 0 <= hi else ''}")
    print(f"    噪声(90%上界) {noise:.4f}   效应/噪声 {d.mean() / max(noise, 1e-12):+.2f}"
          f"{'   ** <1，按 §18.4 判 undecidable **' if abs(d.mean() / max(noise, 1e-12)) < 1 else ''}")
    print(f"    逐种子差 = {np.round(d, 4).tolist()}")


print("\n" + "=" * 92)
print("H0 操作检查 / H1 主判据（干净对比）/ H2 交互")
print("=" * 92)
report("lm0", "R50n", "R38n", "H0  low_mass(cp0)  无捕食者臂", "R50n > R38n")
report("dlm", "R50n", "R38n", "H1  Δlow_mass  **干净对比**", "R50n > R38n（衰减变慢/停止）")
report("dlm", None, None, "H2  交互 (R50n−R38n) − (R50p−R38p)", "≠0 则资源比效应依赖捕食者",
       interaction=("R50n", "R38n", "R50p", "R38p"))

print("\n  参照：有捕食者臂的同一对比")
report("lm0", "R50p", "R38p", "H0'  low_mass(cp0)  有捕食者臂", "R50p > R38p")
report("dlm", "R50p", "R38p", "H1'  Δlow_mass  有捕食者臂（含混杂）", "—")
report("ratio", "R50n", "R38n", "摄入低/高比（无捕食者臂）", "—")

print("\n" + "=" * 92)
print("护栏：p 臂报捕食者丢失（不做事后筛）；n 臂按构造无捕食者")
print("=" * 92)
for a in ARMS:
    lost = sum(1 for s in SEEDS for r in REPS
               if any(p["carnivore_frac"] == 0
                      for p in REC[(a, s, r)]["traj"][len(REC[(a, s, r)]["traj"]) // 2:]))
    print(f"  {a:>6}  后半程 carnivore_frac=0 的 run {lost}/24   种群均值 {pop[a].mean():.0f}")

print("\n[判决由 result-analyst 出，本脚本只把数摆出来。]")
