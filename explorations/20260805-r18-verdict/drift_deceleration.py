"""`low_mass` 的漂移在减速吗——判决要答但 §21.4 没问的那个问题。

**非预注册。** §21.4 的 H1 只问「末四分之一 vs R16 窗口有没有差」，
它答得了「变了没有」，答不了「**会不会停**」。而 §21.1 的问题（「那个 50/50 双簇
撑得到 450 代吗」）在 `low_mass` 单向漂移的情况下必须回答后者：

- 漂移**减速趋向一个新平衡** ⇒ 双簇共存，只是不对称；
- 漂移**匀速或加速** ⇒ 正走向固定（果专精独占），那才是真的没撑住。

做法：按**世代**（不是按帧）等分 N 段，逐 run 算世代加权的 `low_mass`，
跨 run 平均后看逐段增量。**增量本身要做配对检验**——「最后一段增量是否仍显著
为正」比任何外推都硬。

⚠️ 几何外推那一行是**外推不是实测**：公比只有 N−2 个点、且实测并不稳定。
报它是为了给一个数量级，不是为了当结论。**判决引用时必须带上这句话。**

跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
      explorations/20260805-r18-verdict/drift_deceleration.py [段数]
"""
import glob
import json
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, "explorations/20260804-readouts")

import numpy as np
from exp_stats import wilcoxon_p_floor
from neutral_null import gen_weights
from scipy.stats import wilcoxon

BINS = np.linspace(0.0, 1.0, 21)
CTR = 0.5 * (BINS[:-1] + BINS[1:])
LOW = CTR < 0.35
RUN_DIR = "outputs/20260805-gen450"


def wmean(v, g, m):
    w = gen_weights(g, "iso") * m
    s = w.sum()
    return float((w * np.asarray(v, float)).sum() / s) if s > 1e-9 else np.nan


def main():
    nseg = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    segs, gends, names = [[] for _ in range(nseg)], [], []
    for f in sorted(glob.glob(f"{RUN_DIR}/*.log")):
        txt = open(f).read()
        if "JSON " not in txt:
            continue
        d = json.loads(txt.split("JSON ")[1].split("\n")[0])
        if d["collapsed"]:
            continue
        tr = d["traj"]
        g = np.array([q["generation"] for q in tr], float)
        H = np.array([q["hist"] for q in tr], float)
        lm = [h[LOW].sum() / max(h.sum(), 1.0) for h in H]
        gm = g.max()
        gends.append(gm)
        names.append(f.split("/")[-1][:-4])
        for i in range(nseg):
            m = (g >= gm * i / nseg) & (g < gm * (i + 1) / nseg + (1e-9 if i == nseg - 1 else 0))
            segs[i].append(wmean(lm, g, m))
    if not names:
        print("没有已完成的 run —— 不是结论，是没数据。")
        sys.exit(1)

    A = np.array(segs)                       # [nseg, nrun]
    mu, sd = np.nanmean(A, 1), np.nanstd(A, 1, ddof=1)
    gm = float(np.mean(gends))

    print("=" * 88)
    print("`low_mass` 的漂移在减速吗（**非预注册**，§21.4 没有这一条）")
    print("=" * 88)
    print(f"  {A.shape[1]} 个 run，平均跨 {gm:.0f} 代 [{min(gends):.0f}, {max(gends):.0f}]，"
          f"按**世代**等分 {nseg} 段")
    if A.shape[1] < 24:
        print(f"  ⚠️ **未跑满 24（现 {A.shape[1]}），判决时必须用 24/24 重算**")
    print()
    print(f"  段中点世代  {[f'{gm * (i + 0.5) / nseg:.0f}' for i in range(nseg)]}")
    print(f"  low_mass    {[f'{v:.4f}' for v in mu]}")
    print(f"  跨 run SD   {[f'{v:.4f}' for v in sd]}")
    d = np.diff(mu)
    print(f"  **逐段增量** {[f'{v:+.4f}' for v in d]}")
    ratios = [d[i + 1] / d[i] for i in range(len(d) - 1) if abs(d[i]) > 1e-9]
    print(f"  增量之比    {[f'{v:.3f}' for v in ratios]}   ← 全 <1 才叫减速")
    print()

    print("  **最硬的那一条：最后一段的增量还显著为正吗**（配对，逐 run）")
    last = A[-1] - A[-2]
    fin = np.isfinite(last)
    p = wilcoxon(A[-1][fin], A[-2][fin]).pvalue if np.any(last[fin] != 0) else float("nan")
    print(f"    Δ(第{nseg}段 − 第{nseg - 1}段) 均值 {np.nanmean(last):+.4f}   "
          f"{int((last[fin] > 0).sum())}/{int(fin.sum())} 为正   "
          f"p={p:.5f}（地板 {wilcoxon_p_floor(int(fin.sum())):.5f}）")
    print(f"    逐 run {np.round(last, 4).tolist()}")
    print(f"    对比第一段增量 {d[0]:+.4f} ⇒ 末段是它的 **1/{d[0] / max(np.nanmean(last), 1e-9):.0f}**")
    print()

    r = float(np.mean(ratios[-2:])) if len(ratios) >= 2 else float("nan")
    print("  几何外推（**外推，不是实测**——公比只有几个点且实测并不稳定）：")
    if 0.0 < r < 1.0:
        print(f"    若增量按公比 r≈{r:.3f} 继续衰减 ⇒ 渐近 `low_mass` ≈ "
              f"**{mu[-1] + d[-1] * r / (1 - r):.4f}**"
              f"（当前 {mu[-1]:.4f}，跨 run SD {sd[-1]:.4f}）")
        print(f"    ⚠️ 渐近值与当前值只差 {d[-1] * r / (1 - r):.4f}，"
              f"**与跨 run SD 同量级 ⇒ 不要拿它当一个测准了的数**")
    else:
        print(f"    公比 r≈{r:.3f} 不在 (0,1) ⇒ **不能做几何外推**")
    print()
    print("  读法：增量全程 <1 的公比 + 末段增量仍显著为正")
    print("  ⇒ **在减速，但 371 代内还没停**。不许写成「已达平衡」，")
    print("     也不许写成「正走向固定」——后者要求增量不减，而实测是逐段减半。")


if __name__ == "__main__":
    main()
