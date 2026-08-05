"""`split_score` 在 R17 主臂上为什么掉一半：检测器失效，还是分裂真的变糙？

`verify_r17.py` 已经查明：W20 有 12/24 个 run 的**平滑后局部极大 <2 个**，
按 `split_score` 的构造直接返回 0；而在有两峰的 run 上 `min(mL,mR)` 与谷因子
与对照臂逐位可比。本脚本查明那 12 个 run 为什么找不到两个峰，并给出
**一个只放宽「严格大于」为「不小于」（容忍平台）的修补版**，重算判决。

`split_score` 的局部极大判据是**严格**不等式（`split_score.py:78`）：
    p[i] > p[i-1] and p[i] > p[i+1]
一个恰好压在两个相邻箱上、两箱质量接近的簇，平滑后是**平台**，没有严格极大点。

读：outputs/20260805-isolation/*.log。
跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
      explorations/20260805-r17-verdict/split_artifact.py
"""
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, "explorations/20260804-readouts")

import numpy as np

from exp_stats import RunSet, paired

RUN_DIR = "outputs/20260805-isolation"
ARMS = ["W20", "W00"]
SEEDS = list(range(12))
REPS = [1, 2]
BINS = np.linspace(0.0, 1.0, 21)
CTR = 0.5 * (BINS[:-1] + BINS[1:])


def last_cp(d, k):
    return d["checkpoints"][-1].get(k, float("nan"))


def smoothed(n, smooth=1):
    p = np.asarray(n, float)
    p = p / max(p.sum(), 1e-12)
    k = np.ones(2 * smooth + 1) / (2 * smooth + 1)
    p = np.convolve(np.pad(p, smooth, mode="edge"), k, mode="valid")
    return p / max(p.sum(), 1e-12)


def score(n, strict=True, smooth=1):
    """`strict=True` 是现行实现；`strict=False` 把「严格大于」放宽为「不小于」，
    并把连成一片的平台折成它的中点，其余一模一样。"""
    p = smoothed(n, smooth)
    if strict:
        loc = [i for i in range(len(p))
               if (i == 0 or p[i] > p[i - 1]) and (i == len(p) - 1 or p[i] > p[i + 1])]
    else:
        cand = [i for i in range(len(p))
                if (i == 0 or p[i] >= p[i - 1]) and (i == len(p) - 1 or p[i] >= p[i + 1])]
        loc, run = [], []
        for i in cand:                      # 把相邻的平台点折成一个代表点
            if run and i == run[-1] + 1:
                run.append(i)
            else:
                if run:
                    loc.append(run[len(run) // 2])
                run = [i]
        if run:
            loc.append(run[len(run) // 2])
        loc = [i for i in loc if p[i] > 0]
    if len(loc) < 2:
        return 0.0, len(loc)
    two = sorted(sorted(loc, key=lambda i: -p[i])[:2])
    i, j = two
    v = int(np.argmin(p[i:j + 1])) + i
    lo = min(p[i], p[j])
    if lo <= 0:
        return 0.0, len(loc)
    return float(min(p[:v].sum(), p[v:].sum()) * (1.0 - p[v] / lo)), len(loc)


def main():
    rs = RunSet.load(RUN_DIR, ARMS, SEEDS, REPS)
    print("=" * 100)
    print("① 那 12 个 `npeaks<2` 的 W20run 长什么样（平滑后，只印非零箱）")
    print("=" * 100)
    shown = 0
    for s in SEEDS:
        for rp in REPS:
            n = last_cp(rs.records[("W20", s, rp)], "bin_n")
            sc, np_ = score(n, strict=True)
            if np_ >= 2:
                continue
            p = smoothed(n)
            nz = [(i, round(float(p[i]), 4)) for i in range(len(p)) if p[i] > 1e-6]
            if shown < 4:
                print(f"  W20_s{s}_r{rp}: 严格峰数={np_}  非零箱(i, 平滑质量) = {nz}")
            shown += 1
    print(f"  共 {shown} 个。对照一个 W00：")
    p = smoothed(last_cp(rs.records[("W00", 0, 1)], "bin_n"))
    print(f"  W00_s0_r1: 非零箱 = {[(i, round(float(p[i]), 4)) for i in range(len(p)) if p[i] > 1e-6]}")

    print()
    print("=" * 100)
    print("② 只把「严格大于」放宽成「不小于」（容忍平台），其余构造不变，重算判决")
    print("=" * 100)
    for strict in (True, False):
        f = (lambda st: (lambda d: score(last_cp(d, "bin_n"), strict=st)[0]))(strict)
        r = paired(rs, f, "W20", "W00", noise_arms=ARMS,
                   metric_name=f"split_score(strict={strict})")
        z = {a: sum(1 for s in SEEDS for rp in REPS
                    if score(last_cp(rs.records[(a, s, rp)], "bin_n"), strict=strict)[0] == 0.0)
             for a in ARMS}
        print(f"  strict={strict}:  W20 {r.mean_a:.4f}  W00 {r.mean_b:.4f}  Δ {r.diff.mean():+.4f}  "
              f"{r.n_pos}/12 正  p={r.p:.5f}  比值 {r.ratio:+.2f}   "
              f"读到 0 的 run: W20 {z['W20']}/24, W00 {z['W00']}/24")

    print()
    print("=" * 100)
    print("③ 两个簇各自的位置与宽度（从末检查点直方图算，不是 mean_pref 的整体均值）")
    print("=" * 100)

    def cl(n, mask):
        p = np.asarray(n, float)
        w = p[mask]
        if w.sum() <= 0:
            return np.nan, np.nan
        m = float((w * CTR[mask]).sum() / w.sum())
        sd = float(np.sqrt(max((w * (CTR[mask] - m) ** 2).sum() / w.sum(), 0)))
        return m, sd

    LO, HI = CTR < 0.35, CTR > 0.65
    for nm, fn in (
        ("lo 簇均值 pref", lambda d: cl(last_cp(d, "bin_n"), LO)[0]),
        ("lo 簇 SD", lambda d: cl(last_cp(d, "bin_n"), LO)[1]),
        ("hi 簇均值 pref", lambda d: cl(last_cp(d, "bin_n"), HI)[0]),
        ("hi 簇 SD", lambda d: cl(last_cp(d, "bin_n"), HI)[1]),
        ("簇间距 (hi−lo)", lambda d: cl(last_cp(d, "bin_n"), HI)[0] - cl(last_cp(d, "bin_n"), LO)[0]),
    ):
        r = paired(rs, fn, "W20", "W00", noise_arms=ARMS, metric_name=nm)
        print(f"  {nm:<16} W20 {r.mean_a:7.4f}  W00 {r.mean_b:7.4f}  Δ {r.diff.mean():+.4f}  "
              f"{r.n_pos}/12 正  p={r.p:.5f}  比值 {r.ratio:+.2f}")

    print()
    print("=" * 100)
    print("④ mean|d| 的抽样噪声地板：两簇计数变小会不会自己把 mean|d| 抬上去")
    print("=" * 100)
    print("  零位点的 d 抽样 SE ≈ sqrt(1/n_lo + 1/n_hi)，E|d| = SE·sqrt(2/π)。")
    for a in ARMS:
        nlo = rs.raw(a, lambda d: float(last_cp(d, "ld_n_lo")))
        nhi = rs.raw(a, lambda d: float(last_cp(d, "ld_n_hi")))
        se = np.sqrt(1.0 / nlo + 1.0 / nhi)
        e = se * np.sqrt(2.0 / np.pi)
        print(f"  {a}: n_lo 均 {nlo.mean():.0f}  n_hi 均 {nhi.mean():.0f}  "
              f"SE 均 {se.mean():.5f}  E|d| 地板均 {e.mean():.5f}")
    nlo_a = rs.raw("W20", lambda d: float(last_cp(d, "ld_n_lo")))
    nhi_a = rs.raw("W20", lambda d: float(last_cp(d, "ld_n_hi")))
    nlo_b = rs.raw("W00", lambda d: float(last_cp(d, "ld_n_lo")))
    nhi_b = rs.raw("W00", lambda d: float(last_cp(d, "ld_n_hi")))
    da = (np.sqrt(1 / nlo_a + 1 / nhi_a) - np.sqrt(1 / nlo_b + 1 / nhi_b)).mean() * np.sqrt(2 / np.pi)
    print(f"  ⇒ 两臂地板之差 = {da:+.6f}，实测效应 +0.25506，占比 {100 * da / 0.25506:.3f}%")


if __name__ == "__main__":
    main()
