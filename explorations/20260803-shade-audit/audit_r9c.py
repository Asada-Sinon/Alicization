"""R9 复核第三遍。

1. 用 `terrain.build` **直接**复核 `outputs/20260803-shade/criterion.txt` 的判据 1/2
   （那份是 numpy 重算的独立实现；这里走 jax 主路径，是第二条独立路径）。
   定义照抄 `terrain_criterion.py`：果富 = **有果格里**的前 10%，只算陆地格。
   （audit_r9b.py 的 [D] 用错了定义——按 share 的全格十分位——那个读数作废。）
2. `herb_forage_pref` / `mean_forage_pref` 的**交互项**：预注册三个假设都没覆盖它。
3. 效应量的正确讲法：遮荫把 rho 放大了多少倍。

重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260803-shade-audit/audit_r9c.py
"""
import dataclasses
import math
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

import numpy as np
from exp_stats import RunSet, paired, _paired_core

print("="*90)
print("[1] 直接用 terrain.build 复核 criterion.txt（定义照抄 terrain_criterion.py）")
print("="*90)
from underworld import Config
from underworld import terrain as terrain_mod
base = dict(fruit_regrow_baseline=0.25, fruit_energy=4.0, fruit_water_frac=0.40,
            regrow_baseline=0.010, plant_max=2.0)
for sea, shade in ((False, 0.0), (True, 0.0), (True, 1.0), (True, 1.3)):
    cfg = dataclasses.replace(Config(), **base, water_sea_dist=sea, grass_shade=shade)
    t = terrain_mod.build(cfg)
    h = np.asarray(t.height); land = h >= cfg.sea_level
    g = np.asarray(t.capacity)[land]
    f = np.asarray(t.fruit_capacity)[land]
    rich = f >= np.quantile(f[f > 0], 0.90)
    lowg = g < 0.30 * g.mean()
    fe = f * cfg.fruit_energy
    share = fe / np.maximum(fe + g, 1e-12)
    grich = g >= np.quantile(g, 0.90)
    print(f"  sea={str(sea):<5} shade={shade:<4}: 果富格数={int(rich.sum())}  "
          f"其中草<30%均值={int((rich & lowg).sum())} ({100*(rich&lowg).sum()/rich.sum():.1f}%)  "
          f"果富处草量中位/均草={np.median(g[rich])/g.mean():.3f}")
    print(f"      果能份额: 果富处={np.median(share[rich]):.3f}  草富处={np.median(share[grich]):.3f}"
          f"  之差={np.median(share[rich])-np.median(share[grich]):.3f}")

print("\n" + "="*90)
print("[2] 预注册没覆盖的那个信号：基因**均值**的交互项")
print("="*90)
DIR = "outputs/20260803-shade/r9"
SEEDS, REPS, ARMS = list(range(12)), [1, 2], ["N0", "T0", "N1", "T1"]
rs = RunSet.load(DIR, ARMS, SEEDS, REPS)
assert not rs.problems

def interaction(metric, name=None):
    t1, n1 = rs.cell_means("T1", metric), rs.cell_means("N1", metric)
    t0, n0 = rs.cell_means("T0", metric), rs.cell_means("N0", metric)
    noise = 2.0 * rs.pooled_within_sd(metric) / math.sqrt(len(REPS))
    return _paired_core(t1-n1, t0-n0, noise, name or metric,
                        "(T1-N1)-(T0-N0)", "T1-N1", "T0-N0")

for m in ("herb_forage_pref", "mean_forage_pref"):
    r = interaction(m)
    print(f"\n  {m}")
    print(f"    交互={r.diff.mean():+.5f}  正 {r.n_pos}/12  p={r.p:.5f}  "
          f"效应/噪声={r.ratio:+.3f}  CI=[{r.ci[0]:+.5f},{r.ci[1]:+.5f}]  dz={r.dz:+.3f}")
    print(f"    逐种子交互={np.array2string(r.diff, precision=4, sign='+')}")
    print(f"    四臂均值: " + "  ".join(f"{a}={rs.cell_means(a,m).mean():.4f}" for a in ARMS))
    icc, sb, sw = rs.icc(m, "N1")
    print(f"    ICC(N1)={icc:.3f}  σ_between={sb:.5f}  σ_within={sw:.5f}")

print("\n" + "="*90)
print("[3] 遮荫把 rho 放大了多少倍（原始 r，臂均值）")
print("="*90)
for a in ARMS:
    print(f"  {a}: rho={rs.cell_means(a,'rho_space').mean():+.5f}")
t1 = rs.cell_means("T1","rho_space").mean(); n1 = rs.cell_means("N1","rho_space").mean()
t0 = rs.cell_means("T0","rho_space").mean(); n0 = rs.cell_means("N0","rho_space").mean()
print(f"  选择效应 平世界 T0-N0 = {t0-n0:+.5f}")
print(f"  选择效应 结构世界 T1-N1 = {t1-n1:+.5f}")
print(f"  放大倍数 = {(t1-n1)/(t0-n0):.2f}x")

print("\n[4] H1/H2/H3 之外：sd 与 rho 在 T1 内部是不是相关（同一批种子）")
sd_t1 = rs.cell_means("T1","sd"); rho_t1 = rs.cell_means("T1","rho_space")
from scipy.stats import pearsonr, spearmanr
print(f"  T1 内 corr(sd, rho_space) = {pearsonr(sd_t1, rho_t1)[0]:+.4f} "
      f"(p={pearsonr(sd_t1, rho_t1)[1]:.4f})  spearman={spearmanr(sd_t1, rho_t1)[0]:+.4f}")
pref_t1 = rs.cell_means("T1","herb_forage_pref")
print(f"  T1 内 corr(herb_forage_pref, rho_space) = {pearsonr(pref_t1, rho_t1)[0]:+.4f} "
      f"(p={pearsonr(pref_t1, rho_t1)[1]:.4f})")

print("\n" + "="*90)
print("[5] 第二个碗有多大 + §13.4/§13.6 地形表的 config 勘误")
print("="*90)
for pm, tag in ((2.0, "实验实际 (--set plant_max=2.0)"), (2.2, "terrain_cross.py 用的 Config() 默认")):
    for shade in (0.0, 1.3):
        cfg = dataclasses.replace(Config(), **{**base, "plant_max": pm},
                                  water_sea_dist=True, grass_shade=shade)
        t = terrain_mod.build(cfg)
        g = np.asarray(t.capacity); f = np.asarray(t.fruit_capacity)
        fe = f.sum() * cfg.fruit_energy
        print(f"  plant_max={pm} shade={shade}: 草承载总量={g.sum():.1f}  草峰值={g.max():.4f}  "
              f"果承载总量={f.sum():.2f}  果能量={fe:.1f}  "
              f"果占陆地承载能量={fe/(fe+g.sum()):.4f}")
    print(f"    {tag}")
cfgA = dataclasses.replace(Config(), **{**base, "plant_max": 2.0}, water_sea_dist=True, grass_shade=0.0)
cfgB = dataclasses.replace(Config(), **{**base, "plant_max": 2.0}, water_sea_dist=True, grass_shade=1.3)
pa = np.asarray(terrain_mod.build(cfgA).capacity).max()
pb = np.asarray(terrain_mod.build(cfgB).capacity).max()
print(f"  真实草峰值 shade 0->1.3: {pa:.4f} -> {pb:.4f}  比值={pb/pa:.4f}")
print(f"  §13.6 预注册写的是 2.195 -> 3.051, 比值 {3.051/2.195:.4f}")
print(f"  两者的绝对值之比: {2.195/pa:.4f} (= 默认 plant_max 2.2 / 实验 2.0)")
