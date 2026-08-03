"""R10 判决用的分析脚本。判据见 `docs/multispecies_program.md` §14（跑之前已提交）。

**这个文件本来只以内联 heredoc 的形式存在过**，于是 `outputs/.../r10_analysis.txt` 无法重跑
——`result-analyst` 在存疑点里点了这条。分析脚本必须在版本库里，否则那份 txt 只是一堆
无法复现的数字。统计一律走 `scripts/exp_stats.py`。

重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260803-curvature/analyze_r10.py
"""
import math
import sys

sys.path.insert(0, "scripts")

import numpy as np

from exp_stats import RunSet, paired

DIR = "outputs/20260803-curvature/r10"
SEEDS, REPS = list(range(12)), [1, 2]
ARMS = ["N", "K10", "K05", "K035"]
# 臂名 → (forage_tradeoff, forage_curvature)
DOSE = {"N": (0.0, 1.0), "K10": (1.0, 1.0), "K05": (1.0, 0.5), "K035": (1.0, 0.35)}


def check(rec, arm, seed, rep):
    """按臂归因：日志记的 overrides 必须与臂名一致。臂是字符串拼的 `--set`，
    拼错一位不报错，只会安静地把两个臂跑成同一个。"""
    o = rec.get("overrides", {})
    t, k = DOSE[arm]
    if float(o.get("forage_tradeoff", -1)) != t:
        return f"forage_tradeoff={o.get('forage_tradeoff')} ≠ {t}"
    if float(o.get("forage_curvature", -1)) != k:
        return f"forage_curvature={o.get('forage_curvature')} ≠ {k}"
    if float(o.get("grass_shade", -1)) != 1.3:
        return "grass_shade 不是 1.3"
    return None


rs = RunSet.load(DIR, ARMS, SEEDS, REPS, check=check)
if rs.problems:
    print("!! 自检不通过：")
    for p in rs.problems[:10]:
        print("   " + p)
    sys.exit(1)
print(f"载入 {len(rs.records)} run，自检通过\n")

print("=" * 92)
print("各臂的 quad_intake（H3 要求 N 与 K10 都 ≈0）")
print("=" * 92)
for a in ARMS:
    v = rs.cell_means(a, "quad_intake")
    se = v.std(ddof=1) / math.sqrt(len(SEEDS))
    print(f"  {a:>5}  {v.mean():+.5f} ± {se:.5f} (跨种子SE)  t={v.mean() / max(se, 1e-12):+.2f}"
          f"  σ_within={rs.pooled_within_sd('quad_intake', [a]):.5f}")
    print(f"         逐种子 = {np.round(v, 5).tolist()}")

print("\n" + "=" * 92)
print("H1 / H2。⚠️ K05−N 与 K035−N 是**双变量**对比（曲率 + tradeoff 同时变），不可归因给曲率")
print("=" * 92)
for a, b in (("K05", "K10"), ("K035", "K10"), ("K035", "K05"), ("K10", "N")):
    print(f"\n--- {a} − {b} ---")
    print(paired(rs, "quad_intake", a, b).format())

print("\n" + "=" * 92)
print("次判据与 H4（§14.4 已预声明本设计对 H4 功效不足）")
print("=" * 92)
for m in ("lin_intake", "quadrel_intake", "quad_demog", "sd",
          "bimodality_coefficient", "blrt_lr_per_n", "mean_pref"):
    try:
        r = paired(rs, m, "K05", "K10")
        n_arm = rs.cell_means("N", m).mean()
        print(f"  {m:<24} N={n_arm:>9.4f} K10={r.mean_b:>9.4f} K05={r.mean_a:>9.4f} "
              f"Δ={r.mean_a - r.mean_b:>+9.4f} {r.n_pos}/12 p={r.p:.4f} 比值={r.ratio:+.2f}")
    except Exception as e:                                   # noqa: BLE001
        print(f"  {m:<24} 不可用（旧 run 未记录该字段）: {e}")

print("\n  `mean_pref` 的 K10−N 是本轮最干净的阳性，单列：")
r = paired(rs, "mean_pref", "K10", "N")
print("  " + r.format().replace("\n", "\n  "))

print("\n" + "=" * 92)
print("护栏 K05 vs K10（容差 = 2 × 配对噪声）")
print("=" * 92)
for m in ("population", "min_pop", "carnivore_frac", "carn_speed", "frugivory_frac",
          "death_thirst_frac", "graze_gain", "fruit_gain"):
    r = paired(rs, m, "K05", "K10")
    tol, d = 2 * r.noise, r.mean_a - r.mean_b
    print(f"  {m:<20} K10={r.mean_b:>10.4f} K05={r.mean_a:>10.4f} Δ={d:>+10.4f} "
          f"±{tol:>9.4f} {'破' if abs(d) > tol else 'ok':>3} 比值={r.ratio:+.2f}")

print("\n" + "=" * 92)
print("ICC")
print("=" * 92)
for m in ("quad_intake", "sd", "bimodality_coefficient", "population"):
    icc, sb, sw = rs.icc(m, "K10")
    print(f"  {m:<24} ICC={icc:.3f}  σ_between={sb:.5f}  σ_within={sw:.5f}")

print("\n[判决由 result-analyst 出，本脚本只把数摆出来。]")
