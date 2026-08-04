"""R17 判决用的读数表。判据 `docs/multispecies_program.md` §20 + §20.4b（勘误）。

**本脚本只把数摆出来，判决由 `result-analyst` 出。**

与 `analyze_r16.py` 的三个区别，每个都是那一轮的代价换来的
------------------------------------------------------------
1. **统计一律走 `scripts/exp_stats.py`。** R16 的 `analyze_r16.paired` 自己实现了一版
   并**额外加了 90% 卡方上界**，比 §19.5 指定的共享实现更严——H3 那个「比值恰好
   1.0035、刀刃上」就是这么来的（共享口径是 +1.38）。**判决被一个实现偏差绑架过一次，
   不能有第二次。**
2. **逐格 `σ̂_W` 必须打印，不能只报池化值。** 对照臂实测逐格内 SD 是
   0.001–0.315（**300 倍**），池化值由少数几个格主导（§20.4b②）。
3. **主判据是 LD（`ld_mean_abs_d`），不是「双峰更尖」。** 后者按构造是 null：
   `forage_pref` 是单基因座、均匀交叉从不产生中间值，重组本来就压不平那个双峰。

跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
      explorations/20260804-readouts/analyze_r17.py
"""
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, "explorations/20260804-readouts")

import numpy as np
from exp_stats import RunSet, mde_sign_consistent, paired

RUN_DIR = "outputs/20260805-isolation"
ARMS = ["W20", "W00"]                 # a − b：处理 − 对照
SEEDS = list(range(12))
REPS = [1, 2]
TRAITS = ["diet", "invest", "size", "attack", "escape", "armor", "spike"]


def last_cp(d, key):
    return d["checkpoints"][-1].get(key, float("nan"))


def trait_d(i):
    return lambda d: d["checkpoints"][-1]["ld_trait_d"][i]


def main():
    rs = RunSet.load(RUN_DIR, ARMS, SEEDS, REPS)
    print("=" * 96)
    print("R17 读数表（判据 `multispecies_program.md` §20 + §20.4b）")
    print("=" * 96)
    n = {a: sum(1 for k in rs.records if k[0] == a) for a in ARMS}
    print(f"  载入 {len(rs.records)} run（{', '.join(f'{a}={n[a]}' for a in ARMS)}）"
          + (f"；⚠️ {len(rs.problems)} 条问题（前 3：{rs.problems[:3]}）"
             if rs.problems else "；无崩溃/缺失"))
    if len(rs.records) < len(ARMS) * len(SEEDS) * len(REPS):
        # `RunSet.cell_means` 要求每个格都齐，缺格会 traceback。跑满之前优雅退出，
        # 免得「脚本崩了」被误读成「数据有问题」。
        print("  ⚠️ **未跑满 48，配对检验需要每个格都齐 —— 先不算，等跑完再来。**")
        print("     （已验通过的部分：`RunSet.load` 正确报出缺失/崩溃的 run）")
        return

    # ---- 主判据 ----
    print()
    print("  H1 主判据：两簇在**非 forage_pref 基因**上的 mean|Cohen's d|（W20 − W00）")
    r = paired(rs, lambda d: last_cp(d, "ld_mean_abs_d"), "W20", "W00",
               label="ld_mean_abs_d", noise_arms=ARMS, metric_name="ld_mean_abs_d")
    print("    " + r.format().replace("\n", "\n    "))
    print(f"    §20.4b① 预先算过的 MDE：对照臂 σ̂_W≈0.177 ⇒ 需要 ≥33% 的对照均值。")
    print(f"    mde_sign_consistent(σ_d={r.observed_sd:.4f}, s=12) = "
          f"{mde_sign_consistent(r.observed_sd, 12):.4f}")

    # ---- §20.4b②：逐格 σ̂_W，不能只报池化值 ----
    print()
    print("  逐格内 SD（§20.4b②：高度异方差，池化值由少数格主导）")
    for a in ARMS:
        sds = []
        for s in SEEDS:
            v = [rs.records[(a, s, rp)] for rp in REPS if (a, s, rp) in rs.records]
            sds.append(np.std([last_cp(x, "ld_mean_abs_d") for x in v], ddof=1)
                       if len(v) == 2 else np.nan)
        sds = np.array(sds)
        print(f"    {a}: 池化 {np.sqrt(np.nanmean(sds ** 2)):.4f}   "
              f"中位 {np.nanmedian(sds):.4f}   逐格 {np.round(sds, 3).tolist()}")

    # ---- 次判据 ----
    print()
    print("  次判据（§20.4，预先声明功效不足，失败一律读 undecidable）")
    for i, t in enumerate(TRAITS):
        rr = paired(rs, trait_d(i), "W20", "W00", label=f"d[{t}]",
                    noise_arms=ARMS, metric_name=f"ld_trait_d[{t}]")
        flag = "  ** <1，功效不足 **" if rr.underpowered else ""
        print(f"    {t:<8} {rr.diff.mean():+.4f}   {rr.n_pos}/{len(rr.diff)} 为正   "
              f"p={rr.p:.5f}   比值 {rr.ratio:+.2f}{flag}")
    for m, nm in ((lambda d: d.get("retained_occ", float("nan")), "retained_occ"),
                  (lambda d: last_cp(d, "low_mass"), "low_mass"),
                  (lambda d: last_cp(d, "split_score"), "split_score"),
                  (lambda d: last_cp(d, "mean_pref"), "mean_pref"),
                  (lambda d: d.get("gen_total", float("nan")), "gen_total")):
        rr = paired(rs, m, "W20", "W00", label=nm, noise_arms=ARMS, metric_name=nm)
        flag = "  ** <1 **" if rr.underpowered else ""
        print(f"    {nm:<14} {rr.mean_a:.4f} vs {rr.mean_b:.4f}   Δ {rr.diff.mean():+.4f}   "
              f"{rr.n_pos}/{len(rr.diff)} 为正   p={rr.p:.5f}   比值 {rr.ratio:+.2f}{flag}")

    # ---- 护栏 ----
    print()
    print("  护栏：两簇计数（LD 的 d 在任一簇 <30 时按 NaN 处理）")
    for a in ARMS:
        lo = [last_cp(rs.records[k], "ld_n_lo") for k in rs.records if k[0] == a]
        hi = [last_cp(rs.records[k], "ld_n_hi") for k in rs.records if k[0] == a]
        nan = sum(1 for k in rs.records if k[0] == a
                  and last_cp(rs.records[k], "ld_mean_abs_d") != last_cp(rs.records[k], "ld_mean_abs_d"))
        print(f"    {a}: lo {min(lo)}–{max(lo)}   hi {min(hi)}–{max(hi)}   NaN 的 run {nan}/{n[a]}")

    print()
    print("  [判决由 result-analyst 出，本脚本只把数摆出来。]")


if __name__ == "__main__":
    main()
