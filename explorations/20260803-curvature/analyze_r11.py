"""R11 判决：降需求造出的双峰，是「降需求量级」还是「悄悄改了两层兑换率」。

判据见 `docs/multispecies_program.md` §15（跑之前已提交，`b73b2dc`），一字不许改。

四个新臂在 `outputs/20260804-demand/`；**`N15`/`T15` 复用 R10 的 `N`/`K10`**
（同世界、同 `forage_curvature=1.0`、同步数），`dip_ratio` 从它们的 `bin_n` 逐位重算——
R10 跑的时候脚本里还没有这个字段，但 `bin_n` 与 `bin_centers` 都在，所以可以补算。

三处口径按 §15.3 写死：
- 交互项噪声 = **`2·σ̂_W/√r`**（四个格均值各一份），不是 `√2·σ̂_W/√r`。
- `σ̂_W` 取 **90% 卡方上界**（`MEMORY.md [LEARN:stats]`）。
- 噪声**只从预测不会触底的臂池化**（`N15`/`T15`/`N05`/`N05m`）。

重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260803-curvature/analyze_r11.py
"""
import glob
import json
import math
import sys

sys.path.insert(0, "scripts")

import numpy as np
from scipy.stats import chi2, norm, wilcoxon

from exp_stats import bootstrap_ci, mde_sign_consistent

SEEDS, REPS = list(range(12)), [1, 2]


def _load(pattern, remap=None):
    rec = {}
    for f in sorted(glob.glob(pattern)):
        for ln in open(f):
            if ln.startswith("JSON "):
                b = f.split("/")[-1][:-4].split("_")
                arm = (remap or {}).get(b[0], b[0])
                rec[(arm, int(b[1][1:]), int(b[2][1:]))] = json.loads(ln[5:])
    return rec


def dip_from_bins(d):
    """从 `bin_n` 重算 `dip_ratio`——R10 的 run 跑在加这个字段之前。"""
    ctr = np.array(d["bin_centers"])
    n = np.array(d["bin_n"], float)
    n = n / max(n.sum(), 1.0)
    mid = (ctr >= 0.40) & (ctr <= 0.60)
    m = float((n * ctr).sum())
    sd = math.sqrt(max(float((n * (ctr - m) ** 2).sum()), 1e-12))
    e = float(norm.cdf(0.60, m, sd) - norm.cdf(0.40, m, sd))
    return float(n[mid].sum()) / max(e, 1e-9)


REC = _load("outputs/20260804-demand/*.log")
REC.update(_load("outputs/20260803-curvature/r10/[NK]*.log",
                 remap={"N": "N15", "K10": "T15"}))
ARMS = ["N15", "T15", "N05", "T05", "N05m", "T05m"]
missing = [(a, s, r) for a in ARMS for s in SEEDS for r in REPS if (a, s, r) not in REC]
if missing:
    print(f"!! 缺 {len(missing)} 个 run：{missing[:8]}")
    sys.exit(1)
# 臂归因自检：日志里记的 overrides 必须与臂名一致
SPEC = {"N15": (0.0, 1.5, 1.0), "T15": (1.0, 1.5, 1.0), "N05": (0.0, 0.5, 1.0),
        "T05": (1.0, 0.5, 1.0), "N05m": (0.0, 0.5, 0.333), "T05m": (1.0, 0.5, 0.333)}
bad = []
for a in ARMS:
    for s in SEEDS:
        for r in REPS:
            o = REC[(a, s, r)]["overrides"]
            t, er, fer = SPEC[a]
            if (float(o.get("forage_tradeoff", 0.0)) != t
                    or float(o.get("eat_rate", 1.5)) != er
                    or float(o.get("fruit_eat_rate", 1.0)) != fer):
                bad.append((a, s, r, o.get("forage_tradeoff"), o.get("eat_rate"),
                            o.get("fruit_eat_rate")))
if bad:
    print(f"!! 臂归因自检不通过（{len(bad)} 条）：{bad[:5]}")
    sys.exit(1)
print(f"载入 {len(REC)} run（新跑 96 + 复用 R10 的 48），臂归因自检通过\n")


def val(a, s, r, m):
    d = REC[(a, s, r)]
    return dip_from_bins(d) if m == "dip_ratio" and "dip_ratio" not in d else d.get(m, float("nan"))


def cell(a, m):
    return np.array([np.mean([val(a, s, r, m) for r in REPS]) for s in SEEDS])


def within(a, m):
    return np.array([np.std([val(a, s, r, m) for r in REPS], ddof=1) for s in SEEDS])


NOISE_ARMS = ["N15", "T15", "N05", "N05m"]          # §15.3：只从不会触底的臂池化


def noise(m, interaction=True):
    w = np.concatenate([within(a, m) for a in NOISE_ARMS])
    w = w[np.isfinite(w)]
    sw = float(np.sqrt((w ** 2).mean()))
    df = max(len(w), 1)                              # 每格 r=2 ⇒ 每个格贡献 1 个自由度
    sw_ub = sw * math.sqrt(df / chi2.ppf(0.10, df))  # 90% 上界（MEMORY [LEARN:stats]）
    k = 2.0 if interaction else math.sqrt(2.0)
    return sw * k / math.sqrt(len(REPS)), sw_ub * k / math.sqrt(len(REPS))


def report(m, ta, na, tb, nb, label):
    x = cell(ta, m) - cell(na, m)
    y = cell(tb, m) - cell(nb, m)
    d = x - y
    p = wilcoxon(x, y, alternative="two-sided").pvalue
    lo, hi = bootstrap_ci(d)
    n_pt, n_ub = noise(m)
    print(f"  {label}")
    print(f"    ({ta}−{na}) = {x.mean():+.4f}   ({tb}−{nb}) = {y.mean():+.4f}   "
          f"交互 = {d.mean():+.4f}")
    print(f"    方向 {int((d < 0).sum())}/12 为负   p={p:.5f}   "
          f"95%CI=[{lo:+.4f}, {hi:+.4f}]{'  **含 0**' if lo <= 0 <= hi else ''}")
    print(f"    噪声 点估计 {n_pt:.4f} / σ̂上界 {n_ub:.4f}   "
          f"效应/噪声 {d.mean() / n_pt:+.2f} / {d.mean() / n_ub:+.2f}"
          f"{'   ** <1，功效不足 **' if abs(d.mean() / n_ub) < 1 else ''}")
    print(f"    逐种子 = {np.round(d, 4).tolist()}")
    return d.mean(), p, d.mean() / n_ub


print("=" * 96)
print("各臂的 dip_ratio（构造上：单峰读 ~1，真分裂读 ~0。两个中性臂是内置估计量对照）")
print("=" * 96)
for a in ARMS:
    v = cell(a, "dip_ratio")
    print(f"  {a:>5}  {v.mean():.4f} ± {v.std(ddof=1) / math.sqrt(12):.4f}   "
          f"逐种子 {np.round(v, 3).tolist()}")

print("\n" + "=" * 96)
print("H1 主判据：dip_ratio 的交互项 (T05−N05) − (T15−N15)，预测为**负**")
print("=" * 96)
report("dip_ratio", "T05", "N05", "T15", "N15", "H1")

print("\n" + "=" * 96)
print("§15.5 归因闸：兑换率匹配臂 (T05m−N05m) − (T15−N15)")
print("  与 H1 同号同量级 ⇒ 起作用的是**需求量级**；只有 T05 塌而 T05m 不塌 ⇒ **兑换率**")
print("=" * 96)
report("dip_ratio", "T05m", "N05m", "T15", "N15", "归因闸")

print("\n" + "=" * 96)
print("次判据（§15.3 全部预声明功效不足，失败一律读 undecidable）")
print("=" * 96)
for m in ("low_mass", "bimodality_coefficient", "blrt_lr_per_n", "sd",
          "quad_intake", "mean_pref", "frac_mid"):
    if not any(m in REC[(a, 0, 1)] for a in ARMS):
        continue
    try:
        print(f"\n  --- {m} ---")
        report(m, "T05", "N05", "T15", "N15", "主对比")
    except Exception as e:                                     # noqa: BLE001
        print(f"    不可用: {e}")

print("\n" + "=" * 96)
print("护栏（T05 vs N05，容差 2 × 配对噪声上界）")
print("=" * 96)
for m in ("death_thirst_frac", "carnivore_frac", "frugivory_frac", "min_pop",
          "population_mean", "population", "carn_speed"):
    if m not in REC[("T05", 0, 1)]:
        continue
    x, y = cell("T05", m), cell("N05", m)
    _, n_ub = noise(m, interaction=False)
    d = (x - y).mean()
    print(f"  {m:<20} N05={y.mean():>10.4f} T05={x.mean():>10.4f} Δ={d:>+10.4f} "
          f"±{2 * n_ub:>9.4f} {'破' if abs(d) > 2 * n_ub else 'ok':>3}")

print("\n[判决由 result-analyst 出，本脚本只把数摆出来。]")
