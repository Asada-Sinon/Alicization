"""R17 判决的独立复核：判据 `docs/multispecies_program.md` §20 + §20.4b。

回答三件 `analyze_r17.py` 没有回答的事（它只把预注册的量摆出来）：

1. **判据实现核对**：单变量归因（`overrides_diff`）、`readout_valid` 护栏（`trajectory.py`
   自己写死「<0.95 判读数失效」，读数表没查）、三种噪声口径、H1 的簇定义。
2. **`split_score` 反向的分解**：`split_score = min(mL,mR) · (1 − valley/min(peak))`，
   且**局部极大 <2 时恒为 0**。从末检查点的 `bin_n` 逐 run 重算三个因子，
   判断降一半是「少数侧质量掉了」「谷被填了」还是「一部分 run 掉到 0」。
3. **轨迹口径**：§20.4 次判据写的是「按代的轨迹」，读数表用的是末检查点。
   `split_score_second` / `low_mass_second` / `retained_occ_first` 在 JSON 里现成。

读：outputs/20260805-isolation/{W00,W20}_s{0..11}_r{1,2}.log 的 `JSON ` 行。
跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
      explorations/20260805-r17-verdict/verify_r17.py
"""
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, "explorations/20260804-readouts")

import numpy as np
from scipy.stats import spearmanr, wilcoxon

from exp_stats import RunSet, mde_sign_consistent, paired
from split_score import split_score, dip_ratio

RUN_DIR = "outputs/20260805-isolation"
ARMS = ["W20", "W00"]
SEEDS = list(range(12))
REPS = [1, 2]
BINS = np.linspace(0.0, 1.0, 21)
CTR = 0.5 * (BINS[:-1] + BINS[1:])


def last_cp(d, k):
    return d["checkpoints"][-1].get(k, float("nan"))


def decompose(n, smooth=1):
    """把 `split_score` 拆成它的三个组成：峰数、min(mL,mR)、谷因子。"""
    p = np.asarray(n, float)
    s = p.sum()
    if s <= 0:
        return dict(npeaks=0, minmass=np.nan, valley_fac=np.nan, score=0.0, at=np.nan)
    p = p / s
    k = np.ones(2 * smooth + 1) / (2 * smooth + 1)
    p = np.convolve(np.pad(p, smooth, mode="edge"), k, mode="valid")
    p = p / max(p.sum(), 1e-12)
    loc = [i for i in range(len(p))
           if (i == 0 or p[i] > p[i - 1]) and (i == len(p) - 1 or p[i] > p[i + 1])]
    if len(loc) < 2:
        return dict(npeaks=len(loc), minmass=np.nan, valley_fac=np.nan, score=0.0, at=np.nan)
    two = sorted(sorted(loc, key=lambda i: -p[i])[:2])
    i, j = two
    v = int(np.argmin(p[i:j + 1])) + i
    lo = min(p[i], p[j])
    mL, mR = p[:v].sum(), p[v:].sum()
    fac = 1.0 - p[v] / lo if lo > 0 else np.nan
    return dict(npeaks=len(loc), minmass=float(min(mL, mR)), valley_fac=float(fac),
                score=float(min(mL, mR) * fac), at=float(CTR[v]))


def main():
    rs = RunSet.load(RUN_DIR, ARMS, SEEDS, REPS)
    print("=" * 100)
    print("R17 独立复核（判据 §20 + §20.4b）")
    print("=" * 100)
    print(f"  载入 {len(rs.records)} run；problems = {rs.problems}")
    print(f"  overrides_diff()（应当**只有一项**）= {rs.overrides_diff()}")
    ov = {a: rs.records[(a, 0, 1)]["overrides"] for a in ARMS}
    print(f"  W20 overrides 项数 {len(ov['W20'])}；W00 {len(ov['W00'])}；"
          f"mate_forage_weight = {ov['W20'].get('mate_forage_weight')} vs "
          f"{ov['W00'].get('mate_forage_weight')}")
    print(f"  steps 一致？ {set(rs.records[k]['steps'] for k in rs.records)}")
    print(f"  checkpoints 数 {set(len(rs.records[k]['checkpoints']) for k in rs.records)}；"
          f"末检查点 t = {sorted(set(last_cp(rs.records[k], 't') for k in rs.records))}")

    # ---- 护栏：readout_valid / in_window（trajectory.py 自己写死的失效判据）----
    print()
    print("  [护栏] readout_valid（`trajectory.py:230`：in_window < 0.95 判读数失效）")
    for a in ARMS:
        iw = [c["in_window"] for k in rs.records if k[0] == a for c in rs.records[k]["checkpoints"]]
        rv = [c["readout_valid"] for k in rs.records if k[0] == a
              for c in rs.records[k]["checkpoints"]]
        print(f"    {a}: in_window 范围 [{min(iw):.4f}, {max(iw):.4f}]   "
              f"readout_valid False 的检查点 {sum(1 for x in rv if not x)}/{len(rv)}")

    # ---- H1 三种噪声口径 ----
    print()
    print("  [H1] 三种 σ̂_W 口径（§20.4 指定「只在这两臂上池化」= 口径 B）")
    md = lambda d: last_cp(d, "ld_mean_abs_d")
    for tag, arms in (("A 只用 W00", ["W00"]), ("B 两臂池化", ARMS), ("C 只用 W20", ["W20"])):
        r = paired(rs, md, "W20", "W00", noise_arms=arms, metric_name="ld_mean_abs_d")
        print(f"    {tag:<12} σ̂_W={rs.pooled_within_sd(md, arms):.4f}  "
              f"配对差噪声={r.noise:.4f}  比值={r.ratio:+.3f}")
    r = paired(rs, md, "W20", "W00", noise_arms=ARMS, metric_name="ld_mean_abs_d")
    print(f"    效应 {r.diff.mean():+.5f}   MDE(§20.4b①)=0.1726   "
          f"mde_sign_consistent={mde_sign_consistent(r.observed_sd, 12):.4f}")
    print(f"    ICC(W00)={rs.icc(md, 'W00')[0]:.3f}  ICC(W20)={rs.icc(md, 'W20')[0]:.3f}")

    # ---- H1 逐检查点轨迹（探索性，非预注册）----
    print()
    print("  [探索性，非预注册] ld_mean_abs_d 按检查点（差是否随时间累积）")
    for i in range(4):
        f = (lambda ii: (lambda d: d["checkpoints"][ii]["ld_mean_abs_d"]))(i)
        rr = paired(rs, f, "W20", "W00", noise_arms=ARMS, metric_name=f"ld@cp{i}")
        t = rs.records[("W20", 0, 1)]["checkpoints"][i]["t"]
        print(f"    cp{i} (t≈{t}): W20 {rr.mean_a:.4f}  W00 {rr.mean_b:.4f}  "
              f"Δ {rr.diff.mean():+.4f}  {rr.n_pos}/12 正  p={rr.p:.5f}  比值 {rr.ratio:+.2f}")

    # ---- split_score 分解 ----
    print()
    print("  [split_score 分解] 末检查点 bin_n 重算：峰数 / min(mL,mR) / 谷因子")
    dec = {}
    for a in ARMS:
        rows = [decompose(last_cp(rs.records[(a, s, rp)], "bin_n")) for s in SEEDS for rp in REPS]
        dec[a] = rows
        z = sum(1 for x in rows if x["npeaks"] < 2)
        print(f"    {a}: 峰数<2（分数按构造=0）的 run {z}/24；"
              f"峰数分布 {np.bincount([x['npeaks'] for x in rows], minlength=5)[:5].tolist()}")
        ok = [x for x in rows if x["npeaks"] >= 2]
        if ok:
            print(f"        仅在有两峰的 run 上：min(mL,mR) 均值 "
                  f"{np.mean([x['minmass'] for x in ok]):.4f}  谷因子均值 "
                  f"{np.mean([x['valley_fac'] for x in ok]):.4f}  "
                  f"分割点 {np.mean([x['at'] for x in ok]):.3f}  score 均值 "
                  f"{np.mean([x['score'] for x in ok]):.4f}")
        print(f"        重算 score 臂均 {np.mean([x['score'] for x in rows]):.4f}（对照日志里的值）")
    # 三个因子各自做配对检验
    print()
    for nm, fn in (("npeaks>=2", lambda d: float(decompose(last_cp(d, "bin_n"))["npeaks"] >= 2)),
                   ("min(mL,mR)", lambda d: decompose(last_cp(d, "bin_n"))["minmass"]),
                   ("valley_fac", lambda d: decompose(last_cp(d, "bin_n"))["valley_fac"])):
        try:
            rr = paired(rs, fn, "W20", "W00", noise_arms=ARMS, metric_name=nm)
            print(f"    {nm:<12} W20 {rr.mean_a:.4f}  W00 {rr.mean_b:.4f}  Δ {rr.diff.mean():+.4f}  "
                  f"{rr.n_pos}/12 正  p={rr.p:.5f}  比值 {rr.ratio:+.2f}")
        except Exception as e:                                    # NaN（无两峰）会传染
            print(f"    {nm:<12} 无法配对检验：{type(e).__name__} {e}")

    # ---- 形状：分区质量 + 独立的双峰性读数 ----
    print()
    print("  [形状] 末检查点归一化直方图的分区质量 + 两个独立的双峰性读数")
    for nm, fn in (
        ("low_mass(<0.35)", lambda d: last_cp(d, "low_mass")),
        ("high_mass(>0.65)", lambda d: last_cp(d, "high_mass")),
        ("mid_mass", lambda d: 1.0 - last_cp(d, "low_mass") - last_cp(d, "high_mass")),
        ("dip_ratio", lambda d: last_cp(d, "dip_ratio")),
        ("blrt_lr_per_n", lambda d: last_cp(d, "blrt_lr_per_n")),
        ("sd(pref)", lambda d: last_cp(d, "sd")),
        ("split_at", lambda d: last_cp(d, "split_at")),
    ):
        try:
            rr = paired(rs, fn, "W20", "W00", noise_arms=ARMS, metric_name=nm)
            print(f"    {nm:<17} W20 {rr.mean_a:8.4f}  W00 {rr.mean_b:8.4f}  Δ {rr.diff.mean():+.4f}  "
                  f"{rr.n_pos}/12 正  p={rr.p:.5f}  比值 {rr.ratio:+.2f}")
        except Exception as e:
            print(f"    {nm:<17} 无法检验：{type(e).__name__} {e}")
    for a in ARMS:
        h = np.array([np.array(last_cp(rs.records[(a, s, rp)], "bin_n"), float)
                      for s in SEEDS for rp in REPS])
        h = h / h.sum(1, keepdims=True)
        print(f"    {a} 平均直方图 = {np.round(h.mean(0), 3).tolist()}")

    # ---- 轨迹口径（§20.4 次判据写的是「按代的轨迹」）----
    print()
    print("  [轨迹口径] §20.4 次判据要的是按代轨迹，读数表用的是末检查点")
    for nm in ("split_score_second", "low_mass_second", "retained_occ_first", "retained_occ"):
        fn = (lambda k: (lambda d: float(d[k]) if d[k] is not None else float("nan")))(nm)
        try:
            rr = paired(rs, fn, "W20", "W00", noise_arms=ARMS, metric_name=nm)
            print(f"    {nm:<20} W20 {rr.mean_a:.4f}  W00 {rr.mean_b:.4f}  Δ {rr.diff.mean():+.4f}  "
                  f"{rr.n_pos}/12 正  p={rr.p:.5f}  比值 {rr.ratio:+.2f}")
        except Exception as e:
            print(f"    {nm:<20} 无法检验：{type(e).__name__} {e}")

    # ---- 簇计数与簇构成 ----
    print()
    print("  [簇构成] ld_n_lo / ld_n_hi（H1 的两簇），以及 lo 占比")
    for nm, fn in (("n_lo", lambda d: float(last_cp(d, "ld_n_lo"))),
                   ("n_hi", lambda d: float(last_cp(d, "ld_n_hi"))),
                   ("n_lo+n_hi", lambda d: float(last_cp(d, "ld_n_lo") + last_cp(d, "ld_n_hi"))),
                   ("lo 占比", lambda d: last_cp(d, "ld_n_lo")
                    / max(last_cp(d, "ld_n_lo") + last_cp(d, "ld_n_hi"), 1))):
        rr = paired(rs, fn, "W20", "W00", noise_arms=ARMS, metric_name=nm)
        print(f"    {nm:<10} W20 {rr.mean_a:9.3f}  W00 {rr.mean_b:9.3f}  Δ {rr.diff.mean():+.3f}  "
              f"{rr.n_pos}/12 正  p={rr.p:.5f}  比值 {rr.ratio:+.2f}")

    # ---- 中介/混杂：Δld 与 Δgen / Δsplit / Δmean_pref 的跨种子相关 ----
    print()
    print("  [中介检查] 逐种子 Δ 之间的 Spearman（12 个种子，探索性）")
    dl = rs.cell_means("W20", md) - rs.cell_means("W00", md)
    others = {
        "Δgen_total": lambda d: d["gen_total"],
        "Δsplit_score": lambda d: last_cp(d, "split_score"),
        "Δmean_pref": lambda d: last_cp(d, "mean_pref"),
        "Δlow_mass": lambda d: last_cp(d, "low_mass"),
        "Δn_lo+n_hi": lambda d: float(last_cp(d, "ld_n_lo") + last_cp(d, "ld_n_hi")),
    }
    for nm, fn in others.items():
        dx = rs.cell_means("W20", fn) - rs.cell_means("W00", fn)
        rho, p = spearmanr(dl, dx)
        print(f"    Δld vs {nm:<14} rho={rho:+.3f}  p={p:.4f}")
    # 臂内：ld 与 gen 的关系（若世代钟是中介，臂内也该有正相关）
    for a in ARMS:
        rho, p = spearmanr(rs.cell_means(a, md), rs.cell_means(a, lambda d: d["gen_total"]))
        print(f"    臂内 {a}: ld vs gen_total  rho={rho:+.3f}  p={p:.4f}")

    # ---- 逐种子主表 ----
    print()
    print("  [逐种子] ld_mean_abs_d 格均值 / split_score / mean_pref / gen_total")
    print(f"    {'seed':>4} {'W20 ld':>8} {'W00 ld':>8} {'Δld':>8} {'同向':>4} "
          f"{'W20 ss':>7} {'W00 ss':>7} {'Δss':>8} {'Δpref':>7} {'Δgen':>7}")
    ss = lambda d: last_cp(d, "split_score")
    mp = lambda d: last_cp(d, "mean_pref")
    gt = lambda d: d["gen_total"]
    for i, s in enumerate(SEEDS):
        a_ld, b_ld = rs.cell(("W20"), s, md)[0], rs.cell("W00", s, md)[0]
        a_ss, b_ss = rs.cell("W20", s, ss)[0], rs.cell("W00", s, ss)[0]
        print(f"    {s:>4} {a_ld:>8.4f} {b_ld:>8.4f} {a_ld - b_ld:>+8.4f} "
              f"{'✓' if a_ld > b_ld else '✗':>4} {a_ss:>7.4f} {b_ss:>7.4f} "
              f"{a_ss - b_ss:>+8.4f} {rs.cell('W20', s, mp)[0] - rs.cell('W00', s, mp)[0]:>+7.4f} "
              f"{rs.cell('W20', s, gt)[0] - rs.cell('W00', s, gt)[0]:>+7.2f}")

    # ---- §20.4b③ 复现验证：W00 vs R16 的 R38n ----
    print()
    print("  [§20.4b③] W00 的三个量（与 R16 的 R38n 应当对得上：low_mass 0.512 / 占空比 1.0 / 148 代）")
    for nm, fn in (("low_mass", lambda d: last_cp(d, "low_mass")),
                   ("retained_occ", lambda d: float(d["retained_occ"])),
                   ("gen_total", gt)):
        v = rs.raw("W00", fn)
        print(f"    W00 {nm:<13} 均值 {v.mean():.4f}  范围 [{v.min():.4f}, {v.max():.4f}]")


if __name__ == "__main__":
    main()
