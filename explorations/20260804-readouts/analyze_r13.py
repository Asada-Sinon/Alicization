"""R13 判决：少数簇是稳定多态还是走向固定的暂态。

判据见 `docs/multispecies_program.md` §16（跑之前已提交，`d3c3029`），一字不许改。

三个互斥的可证伪预测：
    H_stable   逐 run「少数簇 dlog 对少数簇频率」的斜率 < 0，且 mean(dlog) ≈ 0
    H_neutral  斜率 ≈ 0 且 mean(dlog) ≈ 0
    H_transient mean(dlog) 显著 < 0

**入组条件（§16.3，写在跑之前）**：第一个检查点（t≈20k）的 `split_score > 0`。
不满足的 run 不进主判据，但单独报「在 20k 之前即丢失」并计入留存曲线。

**§16.4 已预先声明：本轮无法预先算 MDE**（逐箱 dlog 无先验数据），
所以 MDE 与效应/噪声比都从本实验自估，**比值 <1 一律判 undecidable，
不许读成「稳定多态」**。

重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260804-readouts/analyze_r13.py
"""
import glob
import json
import math
import sys

sys.path.insert(0, "scripts")

import numpy as np
from scipy.stats import wilcoxon

CTR = np.array(np.linspace(0.15, 0.85, 15))
CTR = 0.5 * (CTR[:-1] + CTR[1:])
LOW = CTR < 0.35                      # 少数簇（果实专精侧），边界与 §15.3 一致

runs = []
for f in sorted(glob.glob("outputs/20260804-traj/*.log")):
    for ln in open(f):
        if ln.startswith("JSON "):
            d = json.loads(ln[5:])
            d["_file"] = f.split("/")[-1]
            d["_wn"] = int(d["_file"].split("_")[0][2:])
            d["_seed"] = int(d["_file"].split("_")[1][1:])
            runs.append(d)
print(f"载入 {len(runs)} 个 run（{sum(r['collapsed'] for r in runs)} 个崩溃）\n")

print("=" * 92)
print("入组：第一个检查点的 split_score > 0（§16.3，写在跑之前）")
print("=" * 92)
entered = [r for r in runs if r.get("entry_split_score", 0.0) > 0.0]
lost_early = [r for r in runs if r.get("entry_split_score", 0.0) <= 0.0]
print(f"  入组 {len(entered)}/{len(runs)}；在 20k 之前即丢失 {len(lost_early)}/{len(runs)}")
print(f"  逐 run 的 entry_split_score: "
      f"{sorted(round(r.get('entry_split_score', float('nan')), 4) for r in runs)}")

print("\n" + "=" * 92)
print("留存曲线：每个检查点上 split_score>0 的 run 数（分母 = 入组数）")
print("=" * 92)
ncp = max(len(r["checkpoints"]) for r in runs)
print(f'  {"检查点":>8} {"t":>8} ' + " ".join(f'{"入组留存":>10}' for _ in range(1)))
for i in range(ncp):
    have = [r for r in entered if len(r["checkpoints"]) > i]
    ret = sum(1 for r in have if r["checkpoints"][i]["split_score"] > 0)
    t = np.mean([r["checkpoints"][i]["t"] for r in have]) if have else float("nan")
    lm = np.mean([r["checkpoints"][i]["low_mass"] for r in have]) if have else float("nan")
    print(f"  {i:>8} {t:>8.0f}  {ret:>3}/{len(have):<3}  低簇质量均值 {lm:.4f}")


def minority_stats(r):
    """每个检查点：少数簇频率与它的 dlog（按箱质量加权）。"""
    out = []
    for cp in r["checkpoints"]:
        n = np.array(cp["bin_n"], float)
        n = n / max(n.sum(), 1)
        dl = np.array([np.nan if v is None else v for v in cp["dlog_per_bin"]], float)
        m = LOW & np.isfinite(dl) & (n > 0)
        if not m.any():
            continue
        freq = float(n[LOW].sum())
        dlog = float((dl[m] * n[m]).sum() / max(n[m].sum(), 1e-12))
        out.append((freq, dlog))
    return out


print("\n" + "=" * 92)
print("主判据：回复力（逐 run「少数簇 dlog 对少数簇频率」的斜率）")
print("=" * 92)
slopes, meandlogs, detail = [], [], []
for r in entered:
    pts = minority_stats(r)
    if len(pts) < 3:
        detail.append((r["_file"], len(pts), None, None))
        continue
    fr = np.array([p[0] for p in pts])
    dl = np.array([p[1] for p in pts])
    if fr.std() < 1e-9:
        detail.append((r["_file"], len(pts), None, float(dl.mean())))
        continue
    sl = float(np.polyfit(fr, dl, 1)[0])
    slopes.append(sl)
    meandlogs.append(float(dl.mean()))
    detail.append((r["_file"], len(pts), sl, float(dl.mean())))

print(f"  可算斜率的 run: {len(slopes)}/{len(entered)}")
for name, npt, sl, md in detail:
    print(f"    {name:<20} 点数 {npt}  斜率 "
          f"{'--' if sl is None else f'{sl:+.4f}'}  mean(dlog) "
          f"{'--' if md is None else f'{md:+.4f}'}")

for label, arr, pred in (("斜率（H_stable 预测 <0）", slopes, "<0"),
                         ("mean(dlog)（H_transient 预测 <0）", meandlogs, "<0")):
    a = np.array(arr, float)
    a = a[np.isfinite(a)]
    if len(a) < 6:
        print(f"\n  {label}: 有效 run 只有 {len(a)} 个，不做检验")
        continue
    p = wilcoxon(a).pvalue
    noise = float(a.std(ddof=1))
    print(f"\n  {label}  n={len(a)}")
    print(f"    均值 {a.mean():+.4f}   中位 {np.median(a):+.4f}   "
          f"{int((a < 0).sum())}/{len(a)} 为负   符号秩 p={p:.5f}")
    print(f"    自估噪声（跨 run SD）{noise:.4f}   效应/噪声 {a.mean() / max(noise, 1e-12):+.2f}"
          f"{'   ** <1，按 §16.4 判 undecidable **' if abs(a.mean() / max(noise, 1e-12)) < 1 else ''}")

print("\n" + "=" * 92)
print("护栏轨迹（§16.5：不设臂均容差，报轨迹）")
print("=" * 92)
print(f'  {"检查点":>6} {"population":>12} {"carn_frac":>11} {"min n_herb":>11}')
for i in range(ncp):
    have = [r for r in runs if len(r["checkpoints"]) > i]
    tr = [[p for p in r["traj"] if abs(p["t"] - r["checkpoints"][i]["t"]) < 3000]
          for r in have]
    pop = [np.nanmean([p["population"] for p in t]) for t in tr if t]
    cf = [np.nanmean([p["carnivore_frac"] for p in t]) for t in tr if t]
    nh = [min(p["n_herb"] for p in t) for t in tr if t]
    print(f"  {i:>6} {np.nanmean(pop):>12.0f} {np.nanmean(cf):>11.4f} {np.min(nh):>11.0f}")
lost_carn = sum(1 for r in runs
                if any(p["carnivore_frac"] == 0 for p in r["traj"][len(r["traj"]) // 2:]))
print(f"\n  后半程出现过 carnivore_frac=0 的 run: {lost_carn}/{len(runs)}"
      f"   （§16.5 预注册：>4/24 则晚期检查点测的是另一个世界，分开报不许池化）")
print("\n[判决由 result-analyst 出，本脚本只把数摆出来。]")
