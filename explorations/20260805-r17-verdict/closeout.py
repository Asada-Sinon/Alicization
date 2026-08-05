"""R17 收口：三件还没读的数。

1. **`mean_pref` 上升的分解**：是质量在两簇之间搬家，还是簇自己往外挪？
   用 `Δmean_pref ≈ w_hi·Δμ_hi + w_lo·Δμ_lo`（质量权重不变时）核对。
2. **臂内跨种子散度**：W20 的 12 个格均值散度比 W00 小多少（「触顶」的直接读数）。
3. **护栏**：种群总量、`carnivore_frac`（`diet_delta=1.5` 由构造应为 0）、两簇计数占比。

跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
      explorations/20260805-r17-verdict/closeout.py
"""
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, "explorations/20260804-readouts")

import numpy as np

from exp_stats import RunSet, paired

RUN_DIR, ARMS, SEEDS, REPS = "outputs/20260805-isolation", ["W20", "W00"], list(range(12)), [1, 2]
CTR = 0.5 * (np.linspace(0, 1, 21)[:-1] + np.linspace(0, 1, 21)[1:])
LO, HI = CTR < 0.35, CTR > 0.65


def last_cp(d, k):
    return d["checkpoints"][-1].get(k, float("nan"))


def cmean(n, mask):
    w = np.asarray(n, float)[mask]
    return float((w * CTR[mask]).sum() / w.sum()) if w.sum() > 0 else np.nan


def main():
    rs = RunSet.load(RUN_DIR, ARMS, SEEDS, REPS)
    md = lambda d: last_cp(d, "ld_mean_abs_d")

    print("=" * 96)
    print("① mean_pref 上升的分解：质量搬家 vs 簇自己往外挪")
    print("=" * 96)
    for a in ARMS:
        lm = rs.cell_means(a, lambda d: last_cp(d, "low_mass")).mean()
        hm = rs.cell_means(a, lambda d: last_cp(d, "high_mass")).mean()
        ml = rs.cell_means(a, lambda d: cmean(last_cp(d, "bin_n"), LO)).mean()
        mh = rs.cell_means(a, lambda d: cmean(last_cp(d, "bin_n"), HI)).mean()
        mp = rs.cell_means(a, lambda d: last_cp(d, "mean_pref")).mean()
        print(f"  {a}: 质量 lo={lm:.4f} hi={hm:.4f}   簇均值 lo={ml:.4f} hi={mh:.4f}   "
              f"mean_pref(实测)={mp:.4f}   重构 lo·μlo+hi·μhi={lm * ml + hm * mh:.4f}")
    d_mp = (rs.cell_means("W20", lambda d: last_cp(d, "mean_pref"))
            - rs.cell_means("W00", lambda d: last_cp(d, "mean_pref"))).mean()
    w_lo = rs.cell_means("W00", lambda d: last_cp(d, "low_mass")).mean()
    w_hi = rs.cell_means("W00", lambda d: last_cp(d, "high_mass")).mean()
    d_lo = (rs.cell_means("W20", lambda d: cmean(last_cp(d, "bin_n"), LO))
            - rs.cell_means("W00", lambda d: cmean(last_cp(d, "bin_n"), LO))).mean()
    d_hi = (rs.cell_means("W20", lambda d: cmean(last_cp(d, "bin_n"), HI))
            - rs.cell_means("W00", lambda d: cmean(last_cp(d, "bin_n"), HI))).mean()
    print(f"  Δmean_pref 实测 = {d_mp:+.4f}；"
          f"按「质量不变、簇自己挪」预测 = {w_lo:.3f}·({d_lo:+.4f}) + {w_hi:.3f}·({d_hi:+.4f}) "
          f"= {w_lo * d_lo + w_hi * d_hi:+.4f}")

    print()
    print("=" * 96)
    print("② 臂内跨种子散度（W20 是不是触了顶）")
    print("=" * 96)
    for a in ARMS:
        v = rs.cell_means(a, md)
        print(f"  {a}: 12 个格均值 SD = {v.std(ddof=1):.4f}   范围 [{v.min():.4f}, {v.max():.4f}]   "
              f"极差 {v.max() - v.min():.4f}")
    print("  两个反向种子（s4/s6）的 W00 值在 12 个对照格里的排名（1=最大）：")
    v = rs.cell_means("W00", md)
    order = np.argsort(-v)
    rank = {int(s): int(np.where(order == i)[0][0]) + 1 for i, s in enumerate(SEEDS)}
    print(f"    s4 = {v[4]:.4f}（第 {rank[4]} 大）   s6 = {v[6]:.4f}（第 {rank[6]} 大）"
          f"   对照臂中位 {np.median(v):.4f}")

    print()
    print("=" * 96)
    print("③ 护栏：种群、carnivore_frac、簇计数")
    print("=" * 96)
    for nm, fn in (
        ("末轨迹点 population", lambda d: next(q["population"] for q in reversed(d["traj"])
                                              if q["population"] == q["population"])),
        ("末轨迹点 carnivore_frac", lambda d: next(q["carnivore_frac"] for q in reversed(d["traj"])
                                                 if q["carnivore_frac"] == q["carnivore_frac"])),
        ("末检查点 n_herb(窗内/帧)", lambda d: last_cp(d, "n_samples") / 200.0),
        ("frugivory_frac", lambda d: last_cp(d, "frugivory_frac")),
        ("graze_gain", lambda d: last_cp(d, "graze_gain")),
        ("fruit_gain", lambda d: last_cp(d, "fruit_gain")),
    ):
        r = paired(rs, fn, "W20", "W00", noise_arms=ARMS, metric_name=nm)
        print(f"  {nm:<24} W20 {r.mean_a:10.4f}  W00 {r.mean_b:10.4f}  Δ {r.diff.mean():+10.4f}  "
              f"{r.n_pos}/12 正  p={r.p:.5f}  比值 {r.ratio:+.2f}")


if __name__ == "__main__":
    main()


def traits():
    """追加：7 个具名性状 d 的**臂均值**（不只是差），用来区分两种机制。

    `d = |Δμ| / σ_pooled`。若 LD 上升纯粹来自**分母**（同型交配把簇内方差压小），
    那么**每个位点的 d 都会按同一个比例上升**。若来自**分子**（两簇真的在分化），
    上升应当**逐位点异质**。这是当前存档数据里唯一能区分两者的读数。
    """
    rs = RunSet.load(RUN_DIR, ARMS, SEEDS, REPS)
    T = ["diet", "invest", "size", "attack", "escape", "armor", "spike"]
    print()
    print("=" * 96)
    print("④ 7 个具名性状 d 的臂均值与**相对增幅**（区分分母效应 vs 分子效应）")
    print("=" * 96)
    print(f"  {'性状':<8} {'W20 d':>8} {'W00 d':>8} {'Δ':>9} {'相对增幅':>9} {'同向':>6}")
    for i, t in enumerate(T):
        f = (lambda ii: (lambda d: d["checkpoints"][-1]["ld_trait_d"][ii]))(i)
        r = paired(rs, f, "W20", "W00", noise_arms=ARMS, metric_name=t)
        rel = 100 * r.diff.mean() / r.mean_b if r.mean_b else float("nan")
        print(f"  {t:<8} {r.mean_a:>8.4f} {r.mean_b:>8.4f} {r.diff.mean():>+9.4f} "
              f"{rel:>+8.1f}% {r.n_pos:>3}/12")
    md = lambda d: last_cp(d, "ld_mean_abs_d")
    r = paired(rs, md, "W20", "W00", noise_arms=ARMS, metric_name="全基因组")
    print(f"  {'全基因组':<6} {r.mean_a:>8.4f} {r.mean_b:>8.4f} {r.diff.mean():>+9.4f} "
          f"{100 * r.diff.mean() / r.mean_b:>+8.1f}% {r.n_pos:>3}/12")
    print("  ⇒ 若是纯分母效应，这一列「相对增幅」应当**彼此接近**；异质则支持分子（真分化）。")


if __name__ == "__main__":
    traits()
