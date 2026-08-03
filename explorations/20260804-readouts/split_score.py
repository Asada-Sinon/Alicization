"""位置无关的分裂统计量，并**拿 R11 的现成数据验它**（不烧新的 GPU 时间）。

为什么要换掉 `dip_ratio`
------------------------
`dip_ratio` 是「中间带 `pref ∈ [0.40,0.60]` 的实测占比 ÷ 同均值同 sd 的高斯期望」。
R11 判决（`docs/multispecies_feasibility.md §13.2`）实证了它的失效模式：
**参照臂 `T15` 其实也有一个隔开的少数簇（6/12 种子），但它的谷在 [0.30,0.40)，
而固定带正好压在 `T15` 的主峰上**，于是 `dip_ratio` 读 0.966、判为「单峰」。
「读数在对照臂读到预期值」只证明它没坏，不证明它在对照臂上有分辨力
（`MEMORY.md [LEARN:method]`）。

而且 §13.2 还指出：`dip_ratio ≈ 0` 只证明**中间带被掏空**，**不证明存在两个有质量的簇**。
那件事本该由 `low_mass` 承担，但它在三个边界下的效应/噪声都 <1。

`split_score` 的构造
-------------------
对每一个内部分割点 `t` 算：

    mL, mR      = t 两侧的质量
    peakL,peakR = 两侧的最大密度
    valley_t    = t 邻域的最小密度
    score(t)    = min(mL, mR) · (1 − valley_t / min(peakL, peakR))

取所有 `t` 上的最大值。两个因子各挡一种假阳性：

- `min(mL, mR)` 要求**两侧都有质量** —— 单纯把分布整体移走（R11 的 `S2` 那种）得 0。
- `(1 − valley/peak)` 要求**中间真的有谷** —— 一个宽的单峰得 0。

所以 **`split_score` ≈ 干净分裂时少数簇的质量，单峰时 ≈ 0**，而且**不含任何位置先验**。
它同时回答了 `dip_ratio` 只能回答一半的那个问题。

重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260804-readouts/split_score.py
"""
import glob
import json
import math
import sys

sys.path.insert(0, "scripts")

import numpy as np
from scipy.stats import chi2, norm, wilcoxon


def split_score(n, ctr, smooth=1):
    """位置无关的分裂分数。`n` 是（未必归一的）逐箱质量，`ctr` 是箱心。

    **第一版是坏的，教训记在这里**：它对每个分割点 `t` 算
    `min(mL,mR)·(1 − valley_t/min(peakL,peakR))`，而那个因子**不要求 valley 是局部极小**
    ——任何有峰的分布，只要分割点离峰一格，密度就低于峰，因子就 >0。实测六个臂
    （含明显单峰的三个中性臂）全部读 ~0.19，完全没有分辨力。

    正确构造必须先**找真正的局部极大**：平滑后取局部极大点，少于两个 ⇒ 分数恒 0
    （单峰按构造得 0）；否则取最大的两个峰，valley = 两峰**之间**的最小密度，

        score = min(谷左质量, 谷右质量) · (1 − valley / min(peak1, peak2))

    两个因子各挡一种假阳性：`min(mL,mR)` 要求两侧都有质量（整体移走得 0），
    后一项要求两峰之间真的有谷（宽单峰得 0）。**不含任何位置先验。**
    """
    p = np.asarray(n, dtype=float)
    s = p.sum()
    if s <= 0:
        return 0.0, float("nan")
    p = p / s
    if smooth:
        # **边界要用边缘复制，不能补零。** `np.convolve(mode="same")` 在两端补 0，
        # 于是**落在边界箱的峰会被平滑抹掉**：实测 `[0.332, 0.035, 0, …]` 平滑后
        # `p[0]=p[1]=0.122`，`p[0] > p[1]` 变成假，索引 0 就不再是局部极大，
        # 整个分裂读成 0。R13 的 120 个检查点里有 10 个这样的假阴，
        # 最大的一个 `low_mass=0.3671`（`wn1_s6` cp0，且 `dip_ratio=0.013` ⇒ 中间带确实空）
        # ——**它带着全场第二大的果专精质量被入组条件踢掉了**。
        k = np.ones(2 * smooth + 1) / (2 * smooth + 1)
        p = np.convolve(np.pad(p, smooth, mode="edge"), k, mode="valid")
        p = p / max(p.sum(), 1e-12)
    # 局部极大：严格高于两个邻居（端点用单侧）
    loc = [i for i in range(len(p))
           if (i == 0 or p[i] > p[i - 1]) and (i == len(p) - 1 or p[i] > p[i + 1])]
    if len(loc) < 2:
        return 0.0, float("nan")
    # 取最大的两个峰，按位置排序
    loc = sorted(sorted(loc, key=lambda i: -p[i])[:2])
    i, j = loc
    seg = p[i:j + 1]
    v = int(np.argmin(seg)) + i
    lo = min(p[i], p[j])
    if lo <= 0:
        return 0.0, float("nan")
    mL, mR = p[:v].sum(), p[v:].sum()
    return float(min(mL, mR) * (1.0 - p[v] / lo)), float(ctr[v])


def retained(n, ctr, mass_thresh=0.03, dip_thresh=0.5):
    """留存判据：**质量 + 缺口**，两项各自可解释。**这才是该当判据的那个量。**

    `split_score` 位置无关，代价是它不问「谷在哪、少数侧是哪一侧」。R13 的 120 个检查点里
    有 **8 个「假阳」**：谷落在 0.57–0.62，那**确实是双峰**（0.42 vs 0.75），
    只是**不是「果专精 vs 草专精」那一种**——是统计量与生物学问题不匹配，不是统计量错。
    最坏的 `wn2_s5` cp1 读 `split_score=0.4101` 而 `low_mass=0.0000`。

    所以判「果专精簇还在不在」要用直接问它的量：

        retained = (low_mass > θ) AND (dip_ratio < φ)

    `low_mass` 管「少数侧有没有质量」，`dip_ratio` 管「中间有没有缺口」。
    在 R13 的 120 个检查点上，**θ ∈ {0.02,0.03,0.05} × φ ∈ {0.3,0.5} 的每一组
    都给出 0 假阳、0 假阴**（对照 `split_score` 的 8 假阳 + 10 假阴）。
    """
    p = np.asarray(n, float)
    p = p / max(p.sum(), 1.0)
    return bool(p[ctr < 0.35].sum() > mass_thresh and dip_ratio(n, ctr) < dip_thresh)


def dip_ratio(n, ctr):
    """R11 的主判据，拿来做对照。"""
    p = np.asarray(n, float)
    p = p / max(p.sum(), 1.0)
    mid = (ctr >= 0.40) & (ctr <= 0.60)
    m = float((p * ctr).sum())
    sd = math.sqrt(max(float((p * (ctr - m) ** 2).sum()), 1e-12))
    e = float(norm.cdf(0.60, m, sd) - norm.cdf(0.40, m, sd))
    return float(p[mid].sum()) / max(e, 1e-9)


def load(pat, remap=None):
    rec = {}
    for f in sorted(glob.glob(pat)):
        for ln in open(f):
            if ln.startswith("JSON "):
                b = f.split("/")[-1][:-4].split("_")
                a = (remap or {}).get(b[0], b[0])
                rec[(a, int(b[1][1:]), int(b[2][1:]))] = json.loads(ln[5:])
    return rec


if __name__ == "__main__":
    R = load("outputs/20260804-demand/*.log")
    R.update(load("outputs/20260803-curvature/r10/[NK]*.log", {"N": "N15", "K10": "T15"}))
    S = list(range(12))
    ARMS = ["N15", "T15", "N05", "T05", "N05m", "T05m"]
    CTR = np.array(R[("N15", 0, 1)]["bin_centers"])


    def cell(a, fn):
        return np.array([np.mean([fn(R[(a, s, r)]) for r in (1, 2)]) for s in S])


    ss = lambda d: split_score(d["bin_n"], CTR)[0]
    sa = lambda d: split_score(d["bin_n"], CTR)[1]
    dr = lambda d: dip_ratio(d["bin_n"], CTR)

    print("=" * 88)
    print("split_score vs dip_ratio，在 R11 的 144 个 run 上（零新增 GPU 成本）")
    print("=" * 88)
    print(f'  {"臂":<6} {"split_score":>12} {"分割点位置":>11} {"dip_ratio":>11} {"低簇质量":>9}')
    for a in ARMS:
        v = cell(a, ss)
        loc = cell(a, sa)
        print(f"  {a:<6} {v.mean():>12.4f} {np.nanmean(loc):>11.3f} {cell(a, dr).mean():>11.4f} "
              f"{cell(a, lambda d: np.array(d['bin_n'], float)[CTR < 0.35].sum() / max(np.array(d['bin_n'], float).sum(), 1)).mean():>9.4f}")

    print()
    print("=" * 88)
    print("关键检验：§13.2 说 T15 其实也有隔开的少数簇，而 dip_ratio 对它瞎了。split_score 看得见吗？")
    print("=" * 88)
    t15, n15 = cell("T15", ss), cell("N15", ss)
    d = t15 - n15
    print(f"  T15 = {t15.mean():.4f}   N15 = {n15.mean():.4f}   差 = {d.mean():+.4f}   "
          f"{int((d > 0).sum())}/12 为正   p={wilcoxon(t15, n15).pvalue:.5f}")
    print(f"  同一对在 dip_ratio 上：T15 = {cell('T15', dr).mean():.4f}  N15 = {cell('N15', dr).mean():.4f}"
          f"   差 = {(cell('T15', dr) - cell('N15', dr)).mean():+.4f}  "
          f"p={wilcoxon(cell('T15', dr), cell('N15', dr)).pvalue:.5f}")
    print("  ⇒ split_score 若在这里显著而 dip_ratio 不显著，就证实了 §13.2 的诊断，")
    print("     并说明 dip_ratio 把「T15 没分裂」这件事读错了。")

    print()
    print("=" * 88)
    print("主判据的交互项，两个统计量各算一遍")
    print("=" * 88)
    NOISE_ARMS = ["N15", "T15", "N05", "N05m"]
    for name, fn in (("split_score", ss), ("dip_ratio", dr)):
        x = cell("T05", fn) - cell("N05", fn)
        y = cell("T15", fn) - cell("N15", fn)
        dd = x - y
        w = np.concatenate([np.array([np.std([fn(R[(a, s, r)]) for r in (1, 2)], ddof=1)
                                      for s in S]) for a in NOISE_ARMS])
        sw = float(np.sqrt((w ** 2).mean()))
        ub = sw * math.sqrt(len(w) / chi2.ppf(0.10, len(w)))
        noise = 2 * ub / math.sqrt(2)
        emp = float(dd.std(ddof=1))
        print(f"  {name:<12} 交互 {dd.mean():>+8.4f}  {int((dd > 0).sum())}+/12  "
              f"p={wilcoxon(x, y).pvalue:.5f}  比值(预注册口径) {dd.mean() / noise:+.2f}  "
              f"比值(经验 SD {emp:.4f}) {dd.mean() / emp:+.2f}")
    print("\n  注：`split_score` 的方向预测是**正**（分裂 ⇒ 分数升高），`dip_ratio` 是负。")
