"""R9 复核第二遍：预注册字面口径 vs 实现口径、护栏的方向一致性、机制核对。

1. §13.5 字面写的 H1 指标是 `forage_pref_std`（全体存活个体），而
   `measure_ecotype.py`（同属预注册 commit 231c016）实现的是 `sd`（食草世系子集）。
   两个都算一遍，看结论会不会变。
2. 护栏「没破」但方向是不是 12/12 一致——2SE 容差宽，过线不等于没动。
3. 机制核对：结构世界里「果实份额高」的格是不是就是「草少」的格（从 terrain 直接算）。

重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260803-shade-audit/audit_r9b.py
"""
import math
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

import numpy as np
from exp_stats import RunSet, paired, _paired_core

DIR = "outputs/20260803-shade/r9"
SEEDS, REPS = list(range(12)), [1, 2]
ARMS = ["N0", "T0", "N1", "T1"]
rs = RunSet.load(DIR, ARMS, SEEDS, REPS)
assert not rs.problems

def interaction(metric, name=None):
    t1, n1 = rs.cell_means("T1", metric), rs.cell_means("N1", metric)
    t0, n0 = rs.cell_means("T0", metric), rs.cell_means("N0", metric)
    noise = 2.0 * rs.pooled_within_sd(metric) / math.sqrt(len(REPS))
    return _paired_core(t1-n1, t0-n0, noise, name or metric,
                        "(T1-N1)-(T0-N0)", "T1-N1", "T0-N0")

print("="*90)
print("[A] H1 两种口径：预注册字面的 forage_pref_std（全体） vs 实现的 sd（食草世系）")
print("="*90)
for m in ("sd", "forage_pref_std"):
    r = interaction(m)
    print(f"\n  --- {m} 交互项 ---")
    print(f"  T1-N1={r.mean_a:+.5f}  T0-N0={r.mean_b:+.5f}  交互={r.diff.mean():+.5f}")
    print(f"  正 {r.n_pos}/12  p={r.p:.5f}  效应/噪声={r.ratio:+.3f}  "
          f"CI=[{r.ci[0]:+.5f},{r.ci[1]:+.5f}]")
    for a, b, tag in (("T0","N0","平世界"), ("T1","N1","结构世界")):
        q = paired(rs, m, a, b)
        print(f"    {tag} {a}-{b}: Δ={q.diff.mean():+.5f} ({100*q.diff.mean()/q.mean_b:+.1f}%) "
              f"p={q.p:.4f} 负 {q.n_neg}/12 效应/噪声={q.ratio:+.2f}")

print("\n" + "="*90)
print("[B] 护栏「过线」不等于「没动」：逐条报方向一致性与 p")
print("="*90)
GUARDS = ["death_thirst_frac", "frugivory_frac", "carnivore_frac", "carn_speed",
          "herb_speed", "population", "min_pop", "death_starvation_frac"]
for a, b in (("T1","N1"), ("N1","N0"), ("T1","N0")):
    print(f"\n  --- {a} - {b} ---")
    print(f'  {"指标":<22}{"Δ":>11}{"相对%":>9}{"正/12":>7}{"负/12":>7}{"p":>9}{"效应/噪声":>10}')
    for m in GUARDS:
        r = paired(rs, m, a, b, noise_arms=["N1"])
        rel = 100*r.diff.mean()/r.mean_b if r.mean_b else float("nan")
        print(f"  {m:<22}{r.diff.mean():>+11.5f}{rel:>+9.1f}{r.n_pos:>7}{r.n_neg:>7}"
              f"{r.p:>9.4f}{r.ratio:>+10.2f}")

print("\n" + "="*90)
print("[C] 基因均值有没有被选择推动（mean_forage_pref / herb_forage_pref）")
print("="*90)
for m in ("mean_forage_pref", "herb_forage_pref"):
    for a, b in (("T0","N0"), ("T1","N1")):
        r = paired(rs, m, a, b)
        print(f"  {m:<20} {a}-{b}: {r.mean_b:.4f} -> {r.mean_a:.4f}  Δ={r.diff.mean():+.5f} "
              f"正 {r.n_pos}/12  p={r.p:.4f}  效应/噪声={r.ratio:+.2f}")

print("\n" + "="*90)
print("[D] 机制：结构世界里「果实能量份额高」的格是不是草少的格（直接从 terrain 算）")
print("="*90)
import dataclasses
from underworld import Config
from underworld import terrain as terrain_mod
base = dict(fruit_regrow_baseline=0.25, fruit_energy=4.0, fruit_water_frac=0.40,
            regrow_baseline=0.010, plant_max=2.0, water_sea_dist=True)
for tag, shade in (("平世界 grass_shade=0.0", 0.0), ("结构世界 grass_shade=1.3", 1.3)):
    cfg = dataclasses.replace(Config(), **base, grass_shade=shade)
    t = terrain_mod.build(cfg)
    cap = np.asarray(t.capacity, dtype=np.float64)
    fcap = np.asarray(t.fruit_capacity, dtype=np.float64) * cfg.fruit_energy
    share = fcap / np.maximum(fcap + cap, 1e-12)
    land = np.asarray(t.capacity) >= 0
    sea = np.asarray(t.rock) * 0 == 0
    m = share > 0
    print(f"  {tag}: share 有果格数={int(m.sum())}  share 均值(有果格)={share[m].mean():.4f} "
          f"max={share.max():.4f}")
    print(f"     corr(share, 草承载 capacity) 全格 = {np.corrcoef(share, cap)[0,1]:+.4f}")
    print(f"     果富前10%格的草承载均值={cap[share>=np.quantile(share,0.9)].mean():.4f} "
          f"vs 全格草承载均值={cap.mean():.4f}")
    print(f"     share 的 sd（全格）={share.std():.5f}  草承载总量={cap.sum():.1f}  "
          f"果承载总量={np.asarray(t.fruit_capacity).sum():.2f}")
