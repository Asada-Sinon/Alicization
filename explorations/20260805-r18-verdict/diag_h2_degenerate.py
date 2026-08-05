"""H2 是不是退化成了 H1 的恒等变换——判决之前必须问清楚的一件事。

起因
----
8/24 的管路空跑上，H1 与 H2 方向相反且量级几乎相同：

    low_mass     晚 0.5797 vs 早 0.5132   Δ **+0.0665**   4/4 为正
    split_score  晚 0.4203 vs 早 0.4837   Δ **−0.0634**   0/4 为正

而 `split_score` 晚 = **0.4203** 与 `1 − low_mass` 晚 = 1 − 0.5797 = **0.4203**
四位小数完全相等。

回忆 `split_score` 的构造（`split_score.py:110`）：

    score = min(mL, mR) · (1 − valley / min(peak1, peak2))

**当谷见底（valley/peak → 0）时，后一个因子 → 1，score 退化成 min(mL, mR)。**
而若谷位恰在 `low_mass` 的分界 0.35 附近、且低侧是多数侧，则
`min(mL, mR) = 1 − low_mass`——**H2 就成了 H1 的恒等变换，不是独立证据。**

这与 R16 的 H2 是**完全同型的失误**（`feasibility.md` §19：那个 12/12 是恒等式
`Δ ≡ 1 − 首1/4`）。所以在判决之前把三件事量出来：

1. `1 − valley/min(peak)` 到底有多接近 1（谷见底了没有）
2. 谷位 `ctr[v]` 落在哪（离 0.35 多远）
3. `split_score` 与 `1 − low_mass` 的逐点残差有多大

跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
      explorations/20260805-r18-verdict/diag_h2_degenerate.py
"""
import glob
import json
import sys

sys.path.insert(0, "explorations/20260804-readouts")

import numpy as np
from split_score import split_score

BINS = np.linspace(0.0, 1.0, 21)
CTR = 0.5 * (BINS[:-1] + BINS[1:])
LOW = CTR < 0.35
RUN_DIR = "outputs/20260805-gen450"


def parts(n, ctr, smooth=1):
    """把 `split_score` 拆成它的两个因子，构造与 split_score.py:62-110 逐行同构。

    **不改 split_score.py**（它在 sweep 的冻结名单里）——这里复刻它的内部量，
    并在末尾用官方函数交叉核对，残差必须为 0。
    """
    p = np.asarray(n, float)
    s = p.sum()
    if s <= 0:
        return None
    p = p / s
    if smooth:
        k = np.ones(2 * smooth + 1) / (2 * smooth + 1)
        p = np.convolve(np.pad(p, smooth, mode="edge"), k, mode="valid")
        p = p / max(p.sum(), 1e-12)
    loc, i, m = [], 0, len(p)
    while i < m:
        j = i
        while j + 1 < m and p[j + 1] == p[i]:
            j += 1
        if (i == 0 or p[i] > p[i - 1]) and (j == m - 1 or p[j] > p[j + 1]):
            loc.append((i + j) // 2)
        i = j + 1
    if len(loc) < 2:
        return {"npeak": len(loc), "score": 0.0}
    loc = sorted(sorted(loc, key=lambda q: -p[q])[:2])
    i, j = loc
    v = int(np.argmin(p[i:j + 1])) + i
    lo = min(p[i], p[j])
    if lo <= 0:
        return {"npeak": len(loc), "score": 0.0}
    mL, mR = float(p[:v].sum()), float(p[v:].sum())
    gap = 1.0 - p[v] / lo
    return {"npeak": len(loc), "mL": mL, "mR": mR, "minM": min(mL, mR),
            "gap": float(gap), "valley_pos": float(ctr[v]),
            "score": float(min(mL, mR) * gap)}


def main():
    fs = sorted(glob.glob(f"{RUN_DIR}/*.log"))
    rows = []
    for f in fs:
        txt = open(f).read()
        if "JSON " not in txt:
            continue
        d = json.loads(txt.split("JSON ")[1].split("\n")[0])
        tr = d["traj"]
        g = np.array([q["generation"] for q in tr], float)
        H = np.array([q["hist"] for q in tr], float)
        late = g >= g.max() * 0.75
        early = (g >= 110.0) & (g <= 148.0)
        for tag, mask in (("早(gen110-148)", early), ("晚(末1/4)", late)):
            for h in H[mask]:
                pr = parts(h, CTR)
                if pr is None or "minM" not in pr:
                    rows.append({"tag": tag, "npeak": pr["npeak"] if pr else 0,
                                 "degenerate": False, "score": 0.0,
                                 "lm": float(h[LOW].sum() / max(h.sum(), 1.0)),
                                 "gap": np.nan, "minM": np.nan,
                                 "valley_pos": np.nan, "resid": np.nan})
                    continue
                lm = float(h[LOW].sum() / max(h.sum(), 1.0))
                off = float(split_score(h, CTR)[0])
                assert abs(off - pr["score"]) < 1e-12, (off, pr["score"], f)
                rows.append({"tag": tag, "npeak": pr["npeak"], "score": pr["score"],
                             "lm": lm, "gap": pr["gap"], "minM": pr["minM"],
                             "valley_pos": pr["valley_pos"],
                             "resid": pr["score"] - (1.0 - lm)})

    if not rows:
        print("没有已完成的 run（找不到 `JSON ` 行）—— 不是结论，是没数据。")
        sys.exit(1)

    print("=" * 92)
    print("H2 退化诊断：`split_score` 是不是 `1 − low_mass` 的恒等变换")
    print("=" * 92)
    print(f"  逐检查点样本 {len(rows)} 个（来自 "
          f"{len([f for f in fs if 'JSON ' in open(f).read()])} 个已完成 run）")
    print()
    for tag in ("早(gen110-148)", "晚(末1/4)"):
        R = [r for r in rows if r["tag"] == tag]
        if not R:
            continue
        gap = np.array([r["gap"] for r in R], float)
        vp = np.array([r["valley_pos"] for r in R], float)
        res = np.array([r["resid"] for r in R], float)
        mm = np.array([r["minM"] for r in R], float)
        lm = np.array([r["lm"] for r in R], float)
        fin = np.isfinite(gap)
        print(f"  {tag}   n={len(R)}   两峰都找到的 {int(fin.sum())}/{len(R)}")
        print(f"    (1 − valley/min(peak))  均值 {np.nanmean(gap):.4f}  "
              f"[{np.nanmin(gap):.4f}, {np.nanmax(gap):.4f}]"
              f"   ← 越接近 1，score 越退化成 min(mL,mR)")
        print(f"    谷位 ctr[v]              均值 {np.nanmean(vp):.4f}  "
              f"[{np.nanmin(vp):.4f}, {np.nanmax(vp):.4f}]"
              f"   ← 与 low_mass 的分界 0.35 比")
        print(f"    min(mL,mR)               均值 {np.nanmean(mm):.4f}   "
              f"1 − low_mass 均值 {np.nanmean(1 - lm):.4f}")
        print(f"    残差 score − (1 − low_mass)  均值 {np.nanmean(res):+.5f}  "
              f"|残差| 最大 {np.nanmax(np.abs(res)):.5f}")
        print()

    res_all = np.array([r["resid"] for r in rows], float)
    gap_all = np.array([r["gap"] for r in rows], float)
    fin = np.isfinite(res_all)
    print("  结论所需的三个数：")
    print(f"    ① 谷见底程度 (1 − valley/peak)：全样本均值 "
          f"{np.nanmean(gap_all):.4f}，最小 {np.nanmin(gap_all):.4f}")
    print(f"    ② |score − (1 − low_mass)| 的最大值：{np.nanmax(np.abs(res_all)):.5f}")
    print(f"    ③ 两者相关：r = "
          f"{np.corrcoef(np.array([r['score'] for r in rows])[fin], (1 - np.array([r['lm'] for r in rows]))[fin])[0, 1]:.6f}")
    print()
    # ---- Δ 层面才是判据实际检验的东西 -------------------------------------------
    #
    # H1/H2 检验的不是水平，是 **Δ = 晚 − 早**。若 score ≡ 1 − low_mass 在两个窗上
    # 都成立，则 Δ_H2 ≡ −Δ_H1，H2 一个自由度都不剩。所以真正要量的是
    #
    #     独立信息 = Δ_H2 + Δ_H1        （恒等 ⇒ 恒为 0）
    #
    # 并把它与 H1 自己的效应、与本轮噪声比。**逐 run 算，不先池化。**
    print("  ---- Δ 层面（判据实际检验的是这个，不是水平）----")
    d1, d2 = [], []
    for f in fs:
        txt = open(f).read()
        if "JSON " not in txt:
            continue
        d = json.loads(txt.split("JSON ")[1].split("\n")[0])
        tr = d["traj"]
        g = np.array([q["generation"] for q in tr], float)
        H = np.array([q["hist"] for q in tr], float)
        late, early = g >= g.max() * 0.75, (g >= 110.0) & (g <= 148.0)
        lm = np.array([h[LOW].sum() / max(h.sum(), 1.0) for h in H], float)
        ss = np.array([split_score(h, CTR)[0] for h in H], float)
        d1.append(lm[late].mean() - lm[early].mean())
        d2.append(ss[late].mean() - ss[early].mean())
    d1, d2 = np.array(d1), np.array(d2)
    indep = d2 + d1
    print(f"    Δ_H1 (low_mass)     逐 run {np.round(d1, 4).tolist()}")
    print(f"    Δ_H2 (split_score)  逐 run {np.round(d2, 4).tolist()}")
    print(f"    Δ_H2 + Δ_H1         逐 run {np.round(indep, 4).tolist()}")
    nexact = int((np.abs(indep) < 1e-9).sum())
    print(f"      **精确恒等（|Δ_H2+Δ_H1| < 1e-9）的 run: {nexact}/{len(indep)}**")
    print(f"      均值 {indep.mean():+.5f}   |最大| {np.abs(indep).max():.5f}"
          f"   （恒等 ⇒ 恒为 0）")

    # 与本轮**自估**的配对差噪声比——这是 CLAUDE.md 指定的标准口径。
    # `indep` 自己的 within-cell SD ⇒ 它自己的噪声，问「这点剩余自由度分辨得出来吗」。
    ns = len(indep) // 2
    if ns >= 2 and len(indep) % 2 == 0:
        cell = indep.reshape(ns, 2)
        sw = float(np.sqrt((cell.std(axis=1, ddof=1) ** 2).mean()))
        noise = np.sqrt(2) * sw / np.sqrt(2)
    else:
        noise = float("nan")
    print()
    print("      三个口径（**都报，不挑**）：")
    print(f"        ① |最大| ÷ H1 效应   = {np.abs(indep).max() / max(abs(d1.mean()), 1e-12):.3f}"
          f"   ← 单点最极端，非本项目标准口径")
    print(f"        ② 均值   ÷ H1 效应   = {abs(indep.mean()) / max(abs(d1.mean()), 1e-12):.3f}")
    print(f"        ③ 均值   ÷ 配对差噪声 = "
          f"{abs(indep.mean()) / noise if np.isfinite(noise) and noise > 1e-9 else float('nan'):.3f}"
          f"   (噪声 {noise:.5f})   ← **CLAUDE.md 指定的标准口径**")
    print()
    if np.isfinite(noise) and noise > 1e-9 and abs(indep.mean()) < noise:
        print("  ⇒ **H2 的独立自由度低于本实验自己的分辨率**（口径③ <1）。")
        print("     Δ_H2 ≈ −Δ_H1 逐 run 成立，"
              f"其中 **{nexact}/{len(indep)} 个精确成立**。")
        print("     **不许把「H1 上升 + H2 下降」报成两条互相矛盾的发现——那是把一个数报了两遍。**")
        print("     与 R16 的 H2 同型（`feasibility.md` §19：那个 12/12 是恒等式 Δ ≡ 1 − 首1/4）。")
    else:
        print("  ⇒ H2 在 Δ 层面的剩余自由度高于本实验噪声；可按两条判据分别报。")
    print()
    print("  ⚠️ 以上全部基于**已完成的 run**，不是判决。判据 §21.4，判决等 24/24。")


if __name__ == "__main__":
    main()
