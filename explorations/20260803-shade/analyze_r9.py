"""R9 判决：结构化草层下的资源分割。判据是 `docs/multispecies_program.md` §13.5，
**跑之前已提交**（`231c016`），本脚本一行也不许改判据。

读什么：`outputs/20260803-shade/r9/{N0,T0,N1,T1}_s{0..11}_r{1,2}.log`（96 个）+ `provenance.txt`。

四个臂（全部同世界：niche 厚果层 + `water_sea_dist=1`）：

    N0  grass_shade=0.0  forage_tradeoff=0.0   平世界 + 中性漂变
    T0  grass_shade=0.0  forage_tradeoff=1.0   平世界 + 最大选择强度（≈ §9.6 的条件）
    N1  grass_shade=1.3  forage_tradeoff=0.0   结构世界 + 中性漂变
    T1  grass_shade=1.3  forage_tradeoff=1.0   结构世界 + 最大选择强度  ← 处理臂

**主判据是交互项**，每个种子一个数：`d_s = (T1_s − N1_s) − (T0_s − N0_s)`。
为什么不是 T1 vs N1：§9.6 的基线不是「方差多大」而是「处理臂比中性漂变**窄 38.3%**」，
而不同世界的漂变方差本来就不同，所以要减掉各自世界的漂变。

**交互项的噪声是 `2·σ̂_W/√r`**，不是 `pair_noise` 给的 `√2·σ̂_W/√r`——四个臂各贡献一份
独立噪声，不是两个。这一步算错会把功效虚报 √2 倍。

统计一律走 `scripts/exp_stats.py`（口径见 `conventions.md §5`），不在这里重新推导。

重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260803-shade/analyze_r9.py
"""
import math
import sys

sys.path.insert(0, "scripts")

import numpy as np

from exp_stats import RunSet, _paired_core, paired

DIR = "outputs/20260803-shade/r9"
SEEDS, REPS = list(range(12)), [1, 2]
ARMS = ["N0", "T0", "N1", "T1"]
DOSE = {"N0": (0.0, 0.0), "T0": (0.0, 1.0), "N1": (1.3, 0.0), "T1": (1.3, 1.0)}


def check(rec, arm, seed, rep):
    """按臂归因的校验：日志里记下的 overrides 必须与臂名一致。

    这条不是形式主义——sweep 的臂是字符串拼出来的 `--set`，拼错一位不会报错，
    只会安静地把两个臂跑成同一个。`RunSet.load` 已经核了种子与文件名一致，这里补臂。
    """
    o = rec.get("overrides", {})
    want_shade, want_tr = DOSE[arm]
    if float(o.get("grass_shade", -1)) != want_shade:
        return f"grass_shade={o.get('grass_shade')} 与臂 {arm} 期望的 {want_shade} 不符"
    if float(o.get("forage_tradeoff", -1)) != want_tr:
        return f"forage_tradeoff={o.get('forage_tradeoff')} 与臂 {arm} 期望的 {want_tr} 不符"
    if int(o.get("water_sea_dist", 0)) != 1:
        return "water_sea_dist 不是 1——认海必须跑遍全部四臂"
    return None


rs = RunSet.load(DIR, ARMS, SEEDS, REPS, check=check)
if rs.problems:
    print("!! 载入自检不通过，先修数据再看结论：")
    for p in rs.problems[:20]:
        print("   " + p)
    if len(rs.problems) > 20:
        print(f"   … 另有 {len(rs.problems) - 20} 条")
    sys.exit(1)
print(f"载入 {len(rs.records)} 个 run，四臂 × 12 种子 × 2 重复，自检通过\n")


def interaction(metric: str, name: str | None = None):
    """`(T1 − N1) − (T0 − N0)`，噪声按四臂算成 `2·σ̂_W/√r`。"""
    t1, n1 = rs.cell_means("T1", metric), rs.cell_means("N1", metric)
    t0, n0 = rs.cell_means("T0", metric), rs.cell_means("N0", metric)
    sigma_w = rs.pooled_within_sd(metric)          # 全部四臂池化
    noise = 2.0 * sigma_w / math.sqrt(len(REPS))
    return _paired_core(t1 - n1, t0 - n0, noise, name or metric,
                        "(T1−N1) − (T0−N0)", "T1−N1", "T0−N0")


print("=" * 96)
print("主判据：交互项 —— 结构化草层有没有把「选择 vs 漂变」的差距推向分化")
print("=" * 96)
for m, label in (("sd", "H1  forage_pref 的 sd"),
                 ("blrt_lr_per_n", "H2  blrt_lr_per_n"),
                 ("rho_space_z", "H3  rho_space 的 z（生态型 × 空间）")):
    print(f"\n--- {label} ---")
    print(interaction(m).format())

print("\n" + "=" * 96)
print("拆开看：两个世界各自的「选择 − 漂变」。§9.6 在平世界测得 −38.3%（塌缩）")
print("=" * 96)
for m in ("sd", "blrt_lr_per_n", "rho_space_z"):
    for a, b, tag in (("T0", "N0", "平世界"), ("T1", "N1", "结构世界")):
        r = paired(rs, m, a, b, label=f"{tag} {a}−{b}")
        rel = 100.0 * (r.mean_a - r.mean_b) / abs(r.mean_b) if r.mean_b else float("nan")
        print(f"  {m:>14s} {tag:<5s} {a}={r.mean_a:>9.5f} {b}={r.mean_b:>9.5f} "
              f"Δ={r.mean_a - r.mean_b:>+9.5f} ({rel:>+7.1f}%)  "
              f"{r.n_pos}/{len(SEEDS)}  p={r.p:.4f}  效应/噪声={r.ratio:+.2f}")

print("\n" + "=" * 96)
print("护栏：容差 = 2 × 该指标的配对差噪声（对着 √2·σ̂_W/√r 定，不凭直觉）")
print("=" * 96)
GUARDS = ["death_thirst_frac", "frugivory_frac", "carnivore_frac", "carn_speed",
          "herb_speed", "population", "min_pop", "death_starvation_frac"]
print(f'  {"指标":<22} {"N1(结构·漂变)":>14} {"T1(处理)":>12} {"Δ":>10} '
      f'{"2SE 容差":>10} {"破?":>5} {"效应/噪声":>10}')
for m in GUARDS:
    r = paired(rs, m, "T1", "N1")
    tol = 2.0 * r.noise
    d = r.mean_a - r.mean_b
    print(f"  {m:<22} {r.mean_b:>14.5f} {r.mean_a:>12.5f} {d:>+10.5f} "
          f"{tol:>10.5f} {'破' if abs(d) > tol else 'ok':>5} {r.ratio:>+10.2f}")

print("\n  另一组：结构世界 vs 平世界（都在中性漂变下，量的是**地形本身**的代价）")
for m in GUARDS:
    r = paired(rs, m, "N1", "N0")
    tol = 2.0 * r.noise
    d = r.mean_a - r.mean_b
    print(f"  {m:<22} {r.mean_b:>14.5f} {r.mean_a:>12.5f} {d:>+10.5f} "
          f"{tol:>10.5f} {'破' if abs(d) > tol else 'ok':>5} {r.ratio:>+10.2f}")

print("\n" + "=" * 96)
print("ICC：配对买到了什么。ICC≈0 的指标配对完全无用（run_to_run_variance.md §5.1）")
print("=" * 96)
for m in ("sd", "blrt_lr_per_n", "rho_space_z", "death_thirst_frac",
          "carnivore_frac", "population", "frugivory_frac"):
    icc, sb, sw = rs.icc(m, "N1")
    print(f"  {m:<22} ICC={icc:>6.3f}  σ_between={sb:>10.5f}  σ_within={sw:>10.5f}")

print("\n" + "=" * 96)
print("每个种子的原始数（不要只报均值 —— CLAUDE.md）")
print("=" * 96)
for m in ("sd", "blrt_lr_per_n", "rho_space_z"):
    t1, n1 = rs.cell_means("T1", m), rs.cell_means("N1", m)
    t0, n0 = rs.cell_means("T0", m), rs.cell_means("N0", m)
    print(f"\n  {m}")
    print(f'    {"seed":>4} {"N0":>10} {"T0":>10} {"N1":>10} {"T1":>10} '
          f'{"T0−N0":>10} {"T1−N1":>10} {"交互":>10}')
    for i, s in enumerate(SEEDS):
        print(f"    {s:>4} {n0[i]:>10.5f} {t0[i]:>10.5f} {n1[i]:>10.5f} {t1[i]:>10.5f} "
              f"{t0[i] - n0[i]:>+10.5f} {t1[i] - n1[i]:>+10.5f} "
              f"{(t1[i] - n1[i]) - (t0[i] - n0[i]):>+10.5f}")

print("\n[判决由 result-analyst 出，本脚本只负责把数摆出来。]")
