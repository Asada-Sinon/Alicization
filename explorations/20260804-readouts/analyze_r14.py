"""R14 Stage 1 判决：把资源供给比抬到 50:50，衰减停不停。

判据见 `docs/multispecies_program.md` §17（跑之前已提交，`c26520a`），一字不许改。

- **操作检查 H0**：`low_mass(cp0)` 的 `R50` vs `R38`。**不成立 ⇒ 整轮读数作废。**
  另加臂达标条件：实测 `frugivory_frac` 落在目标 ±0.05 内。
- **主判据 H1**：`Δlow_mass = low_mass(cp1) − low_mass(cp0)`，`R50` vs `R38` 配对。
  预测 `R50 > R38`。**本轮对它无先验数据，MDE 自估，比值 <1 判 undecidable。**
- **次判据 H2**：摄入低/高比。**§17.4 已预先声明功效不足（预测÷MDE=0.35）**，
  不成立不构成「摄入劣势没变」的证据。

分析口径（§17.6）：先 r=2 平均成格均值 → 12 个格均值配对符号秩 → 比值用
`√2·σ̂_W/√r`、`σ̂_W` 取 90% 卡方上界 → 比值<1 判 undecidable。
**护栏：任一 run 丢失捕食者即分开报，不许池化。**

重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260804-readouts/analyze_r14.py
"""
import glob
import json
import math
import sys

sys.path.insert(0, "scripts")

import numpy as np
from scipy.stats import chi2, wilcoxon

from exp_stats import bootstrap_ci

CTR = np.linspace(0.15, 0.85, 15)
CTR = 0.5 * (CTR[:-1] + CTR[1:])
LOW, HIGH = CTR < 0.35, CTR > 0.65
SEEDS, REPS, ARMS = list(range(12)), [1, 2], ["R38", "R50"]
SPEC = {"R38": (0.25, 0.010), "R50": (0.50, 0.005)}

REC = {}
for f in sorted(glob.glob("outputs/20260804-ratio/*.log")):
    for ln in open(f):
        if ln.startswith("JSON "):
            b = f.split("/")[-1][:-4].split("_")
            REC[(b[0], int(b[1][1:]), int(b[2][1:]))] = json.loads(ln[5:])
missing = [(a, s, r) for a in ARMS for s in SEEDS for r in REPS if (a, s, r) not in REC]
if missing:
    print(f"!! 缺 {len(missing)} 个 run: {missing[:6]}")
    sys.exit(1)
bad = [(a, s, r) for a in ARMS for s in SEEDS for r in REPS
       if (float(REC[(a, s, r)]["overrides"].get("fruit_regrow_baseline", -1)) != SPEC[a][0]
           or float(REC[(a, s, r)]["overrides"].get("regrow_baseline", -1)) != SPEC[a][1])]
if bad:
    print(f"!! 臂归因自检不通过: {bad[:5]}")
    sys.exit(1)
print(f"载入 {len(REC)} run，臂归因自检通过\n")


def intake_ratio(cp):
    ik = np.array([np.nan if v is None else v for v in cp["intake"]], float)
    n = np.array(cp["bin_n"], float)
    a = LOW & np.isfinite(ik) & (n > 0)
    b = HIGH & np.isfinite(ik) & (n > 0)
    if not (a.any() and b.any()):
        return np.nan
    return float((ik[a] * n[a]).sum() / n[a].sum() / ((ik[b] * n[b]).sum() / n[b].sum()))


def val(a, s, r, key):
    d = REC[(a, s, r)]
    cps = d["checkpoints"]
    if key == "lm0":
        return cps[0]["low_mass"] if cps else np.nan
    if key == "dlm":
        return (cps[1]["low_mass"] - cps[0]["low_mass"]) if len(cps) > 1 else np.nan
    if key == "ratio":
        return intake_ratio(cps[0]) if cps else np.nan
    if key == "frug":
        return cps[0].get("frugivory_frac", np.nan) if cps else np.nan
    if key == "flux":
        return (cps[0].get("graze_gain", 0) + cps[0].get("fruit_gain", 0)) if cps else np.nan
    return cps[0].get(key, np.nan) if cps else np.nan


def cell(a, key):
    return np.array([np.nanmean([val(a, s, r, key) for r in REPS]) for s in SEEDS])


def within(a, key):
    return np.array([np.nanstd([val(a, s, r, key) for r in REPS], ddof=1) for s in SEEDS])


def report(key, label, pred):
    x, y = cell("R50", key), cell("R38", key)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    d = x - y
    w = np.concatenate([within(a, key) for a in ARMS])
    w = w[np.isfinite(w)]
    sw = float(np.sqrt((w ** 2).mean()))
    df = max(len(w), 1)
    sw_ub = sw * math.sqrt(df / chi2.ppf(0.10, df))
    noise = sw_ub * math.sqrt(2) / math.sqrt(len(REPS))
    p = wilcoxon(x, y).pvalue if len(d) >= 6 else float("nan")
    lo, hi = bootstrap_ci(d)
    print(f"\n  --- {label}（预测 {pred}）---")
    print(f"    R38 = {y.mean():+.4f}   R50 = {x.mean():+.4f}   差 = {d.mean():+.4f}")
    print(f"    方向 {int((d > 0).sum())}/{len(d)} 为正   p={p:.5f}   "
          f"95%CI=[{lo:+.4f}, {hi:+.4f}]{'  **含 0**' if lo <= 0 <= hi else ''}")
    print(f"    噪声(90%上界) {noise:.4f}   效应/噪声 {d.mean() / max(noise, 1e-12):+.2f}"
          f"{'   ** <1，判 undecidable **' if abs(d.mean() / max(noise, 1e-12)) < 1 else ''}")
    print(f"    逐种子差 = {np.round(d, 4).tolist()}")


print("=" * 92)
print("操作检查：臂达标条件 = 实测 frugivory_frac 落在目标 ±0.05 内（§17.3）")
print("=" * 92)
for a, tgt in (("R38", 0.38), ("R50", 0.50)):
    v = cell(a, "frug")
    fl = cell(a, "flux")
    ok = int(np.sum(np.abs(v - tgt) <= 0.05))
    print(f"  {a}  frugivory {v.mean():.4f} ± {v.std(ddof=1) / math.sqrt(len(v)):.4f}   "
          f"目标 {tgt}  达标 {ok}/12   总通量 {fl.mean():.1f}")
    print(f"       逐种子 {np.round(v, 3).tolist()}")

print("\n" + "=" * 92)
print("H0 操作检查（不成立 ⇒ 整轮读数作废）/ H1 主判据 / H2 次判据（已预声明功效不足）")
print("=" * 92)
report("lm0", "H0  low_mass(cp0)", "R50 > R38")
report("dlm", "H1  Δlow_mass(cp0→cp1)", "R50 > R38（衰减变慢）")
report("ratio", "H2  摄入低/高比", "R50 > R38")

print("\n" + "=" * 92)
print("护栏：任一 run 丢失捕食者即分开报（§17.6）")
print("=" * 92)
for a in ARMS:
    lost = [(s, r) for s in SEEDS for r in REPS
            if any(p["carnivore_frac"] == 0 for p in REC[(a, s, r)]["traj"][len(REC[(a, s, r)]["traj"]) // 2:])]
    pops = [np.nanmean([p["population"] for p in REC[(a, s, r)]["traj"]])
            for s in SEEDS for r in REPS]
    print(f"  {a}  丢失捕食者的 run {len(lost)}/24 {lost if lost else ''}   "
          f"种群均值 {np.nanmean(pops):.0f}")
if any(True for _ in [0]):
    print("\n  分开报（去掉丢失捕食者的 run 后重算 H1）：")
    keep = [s for s in SEEDS
            if not any(any(p["carnivore_frac"] == 0
                           for p in REC[(a, s, r)]["traj"][len(REC[(a, s, r)]["traj"]) // 2:])
                       for a in ARMS for r in REPS)]
    if len(keep) >= 6:
        x = np.array([np.nanmean([val("R50", s, r, "dlm") for r in REPS]) for s in keep])
        y = np.array([np.nanmean([val("R38", s, r, "dlm") for r in REPS]) for s in keep])
        d = x - y
        print(f"    保留 {len(keep)}/12 个种子   差 = {d.mean():+.4f}   "
              f"{int((d > 0).sum())}/{len(d)} 为正   p={wilcoxon(x, y).pvalue:.5f}")
    else:
        print(f"    只剩 {len(keep)} 个种子，不重算")

print("\n[判决由 result-analyst 出，本脚本只把数摆出来。]")
