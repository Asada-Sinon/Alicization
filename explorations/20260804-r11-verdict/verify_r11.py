"""R11 判决的独立复核（result-analyst）。判据 = docs/multispecies_program.md §15（commit b73b2dc）。

回答七件事，全部只从磁盘上的 run 产物算，不引用主控给的任何数字：
 1. 复用闸：R10 的 N/K10 与 R11 四臂，overrides 全键差异 + bin_centers 是否逐位相同；
    并把新臂存档的 dip_ratio 与「从 bin_n 重算」的值对拍（复用路径的往返审计）。
 2. 各臂 dip_ratio 格均值、run 级 <0.3 计数、主判据与归因闸的交互项。
 3. §15.3 预注册但脚本里没实现的 low_mass：从 bin_n 补算（pref<0.35 的质量）。
 4. bin_n 的截断率：BINS=linspace(0.15,0.85,15)，np.digitize 之外的质量被丢弃。
    这决定 low_mass 能不能承担「两个有质量的生态型」。
 5. quad_intake 的臂均值 + 探针 S1/S0 的读数（正曲率是否复现）。
 6. 交互项的经验逐种子 SD（与预注册的 2·σ̂_W/√r 对照，看预注册噪声是保守还是激进）。
 7. 护栏：容差 ÷ 臂均，以及对同一批数做配对检验（预注册容差是否形同虚设）。

重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260804-r11-verdict/verify_r11.py
"""
import glob, json, math, sys
sys.path.insert(0, "scripts")
import numpy as np
from scipy.stats import chi2, norm, wilcoxon
from exp_stats import bootstrap_ci

SEEDS, REPS = list(range(12)), [1, 2]
R = "/home/xrl/intern/Alicization/"


def _load(pattern, remap=None):
    rec = {}
    for f in sorted(glob.glob(R + pattern)):
        for ln in open(f):
            if ln.startswith("JSON "):
                b = f.split("/")[-1][:-4].split("_")
                rec[((remap or {}).get(b[0], b[0]), int(b[1][1:]), int(b[2][1:]))] = json.loads(ln[5:])
    return rec


REC = _load("outputs/20260804-demand/*.log")
REC.update(_load("outputs/20260803-curvature/r10/[NK]*.log", remap={"N": "N15", "K10": "T15"}))
ARMS = ["N15", "T15", "N05", "T05", "N05m", "T05m"]
REC = {k: v for k, v in REC.items() if k[0] in ARMS}
print(f"[0] 载入 {len(REC)} run（应为 6×12×2=144；R10 的 K05/K035 已排除）")

# ---------- 1. 复用闸 ----------
print("\n[1] 复用闸：全键 overrides 差异")
keys = sorted({k for v in REC.values() for k in v["overrides"]})
for a in ARMS:
    o = REC[(a, 0, 1)]["overrides"]
    print(f"  {a:>5} steps={REC[(a,0,1)]['steps']}  " + " ".join(f"{k}={o.get(k,'—')}" for k in keys))
same_ov = all(REC[(a, s, r)]["overrides"] == REC[(a, 0, 1)]["overrides"] for a in ARMS for s in SEEDS for r in REPS)
print(f"  臂内 overrides 全格一致: {same_ov}")
ctr0 = REC[("N15", 0, 1)]["bin_centers"]
print(f"  bin_centers 全 144 run 逐位相同: {all(REC[k]['bin_centers'] == ctr0 for k in REC)}  (n_bins={len(ctr0)}, "
      f"范围 [{ctr0[0]:.3f},{ctr0[-1]:.3f}])")


def dip_from_bins(d):
    ctr = np.array(d["bin_centers"]); n = np.array(d["bin_n"], float)
    n = n / max(n.sum(), 1.0)
    m = float((n * ctr).sum()); sd = math.sqrt(max(float((n * (ctr - m) ** 2).sum()), 1e-12))
    e = float(norm.cdf(0.60, m, sd) - norm.cdf(0.40, m, sd))
    return float(n[(ctr >= 0.40) & (ctr <= 0.60)].sum()) / max(e, 1e-9)


err = [abs(dip_from_bins(v) - v["dip_ratio"]) for v in REC.values() if "dip_ratio" in v]
print(f"  往返审计：新臂 {len(err)} run 的「存档 dip_ratio」vs「从 bin_n 重算」最大绝对差 = {max(err):.3e}")

# ---------- 2/3/4. 读数 ----------
def low_mass(d):
    ctr = np.array(d["bin_centers"]); n = np.array(d["bin_n"], float)
    return float(n[ctr < 0.35].sum() / max(n.sum(), 1.0))


def trunc(d):
    return 1.0 - float(np.sum(d["bin_n"])) / max(d["n_samples"], 1)


def hi_mass(d):
    ctr = np.array(d["bin_centers"]); n = np.array(d["bin_n"], float)
    return float(n[ctr > 0.65].sum() / max(n.sum(), 1.0))


GET = {"dip_ratio": lambda d: dip_from_bins(d), "low_mass": low_mass, "trunc_frac": trunc, "hi_mass": hi_mass}
val = lambda a, s, r, m: GET[m](REC[(a, s, r)]) if m in GET else REC[(a, s, r)].get(m, float("nan"))
cell = lambda a, m: np.array([np.mean([val(a, s, r, m) for r in REPS]) for s in SEEDS])
within = lambda a, m: np.array([np.std([val(a, s, r, m) for r in REPS], ddof=1) for s in SEEDS])
NOISE_ARMS = ["N15", "T15", "N05", "N05m"]


def noise(m, interaction=True):
    w = np.concatenate([within(a, m) for a in NOISE_ARMS]); w = w[np.isfinite(w)]
    sw = float(np.sqrt((w ** 2).mean())); df = max(len(w), 1)
    k = 2.0 if interaction else math.sqrt(2.0)
    return sw * k / math.sqrt(len(REPS)), sw * math.sqrt(df / chi2.ppf(0.10, df)) * k / math.sqrt(len(REPS))


def report(m, ta, na, tb, nb, label):
    x, y = cell(ta, m) - cell(na, m), cell(tb, m) - cell(nb, m)
    d = x - y
    p = wilcoxon(x, y, alternative="two-sided").pvalue
    lo, hi = bootstrap_ci(d)
    n_pt, n_ub = noise(m)
    emp = float(np.std(d, ddof=1))
    print(f"  {label:<28} 交互={d.mean():+.4f}  ({ta}−{na})={x.mean():+.4f} ({tb}−{nb})={y.mean():+.4f}")
    print(f"    符号 {int((d<0).sum())}−/{int((d>0).sum())}+  p={p:.5f}  CI=[{lo:+.4f},{hi:+.4f}]"
          f"{'  **含0**' if lo <= 0 <= hi else ''}")
    print(f"    预注册噪声 2σ̂/√r: 点估 {n_pt:.4f} / 上界 {n_ub:.4f} ⇒ 比值 {d.mean()/n_pt:+.2f} / {d.mean()/n_ub:+.2f}"
          f"    [经验逐种子SD {emp:.4f} ⇒ 比值 {d.mean()/emp:+.2f}]")


print("\n[2] 各臂读数（格均值 ± SE）")
print(f"  {'臂':>5} {'dip_ratio':>18} {'low_mass':>18} {'hi_mass':>12} {'trunc_frac':>12} {'quad_intake':>14} {'mean_pref':>12}")
for a in ARMS:
    row = [a]
    for m in ("dip_ratio", "low_mass"):
        v = cell(a, m); row.append(f"{v.mean():.4f}±{v.std(ddof=1)/math.sqrt(12):.4f}")
    for m in ("hi_mass", "trunc_frac", "quad_intake", "mean_pref"):
        row.append(f"{cell(a, m).mean():.5f}")
    print(f"  {row[0]:>5} {row[1]:>18} {row[2]:>18} {row[3]:>12} {row[4]:>12} {row[5]:>14} {row[6]:>12}")

print("\n  run 级 dip_ratio<0.3 计数（每臂 24 run）与逐种子格均值")
for a in ARMS:
    runs = [val(a, s, r, "dip_ratio") for s in SEEDS for r in REPS]
    v = cell(a, "dip_ratio")
    print(f"  {a:>5}  <0.3: {sum(1 for x in runs if x < 0.3):>2}/24   格均值 min={v.min():.3f} max={v.max():.3f}   {np.round(v,3).tolist()}")

print("\n[3] 主判据 + 归因闸 + 补算的 low_mass")
report("dip_ratio", "T05", "N05", "T15", "N15", "H1 主判据 dip_ratio")
report("dip_ratio", "T05m", "N05m", "T15", "N15", "§15.5 归因闸 (m)")
report("low_mass", "T05", "N05", "T15", "N15", "low_mass（补算）")
report("low_mass", "T05m", "N05m", "T15", "N15", "low_mass (m)（补算）")
report("hi_mass", "T05", "N05", "T15", "N15", "hi_mass（非预注册）")
d05, d05m = cell("T05", "dip_ratio") - cell("N05", "dip_ratio"), cell("T05m", "dip_ratio") - cell("N05m", "dip_ratio")
print(f"\n  T05 与 T05m 的塌陷深度直接比：(T05−N05)={d05.mean():+.4f} vs (T05m−N05m)={d05m.mean():+.4f}")
print(f"    差 = {(d05m-d05).mean():+.4f}  符号 {int((d05m>d05).sum())}/12 为 T05m 更浅  "
      f"p={wilcoxon(d05m, d05, alternative='two-sided').pvalue:.5f}  CI={np.round(bootstrap_ci(d05m-d05),4).tolist()}")
print(f"    比例 (T05m−N05m)/(T05−N05) = {d05m.mean()/d05.mean():.3f}")

print("\n[4] 截断率（bin_n 丢掉的质量 = 1 − Σbin_n/n_samples）")
for a in ARMS:
    t = cell(a, "trunc_frac")
    print(f"  {a:>5}  {t.mean()*100:6.3f}%  逐种子 max={t.max()*100:.3f}%")

print("\n[5] quad_intake：本轮臂均 vs 饱和探针")
for a in ARMS:
    v = cell(a, "quad_intake")
    print(f"  {a:>5}  {v.mean():+.5f} ± {v.std(ddof=1)/math.sqrt(12):.5f}   逐种子符号 {int((v>0).sum())}+/12")
for f in sorted(glob.glob(R + "outputs/20260803-curvature/sat/*.log")):
    for ln in open(f):
        if ln.startswith("JSON "):
            d = json.loads(ln[5:])
            print(f"  探针 {f.split('/')[-1]:<18} quad={d['quad_intake']:+.5f} se={d.get('quad_intake_se',float('nan')):.5f} "
                  f"dip={d.get('dip_ratio',float('nan')):.4f} eat_rate={d['overrides'].get('eat_rate','—')} "
                  f"regrow={d['overrides'].get('regrow_baseline','—')} tradeoff={d['overrides'].get('forage_tradeoff','—')}")

print("\n[6] 次判据（全部 §15.3 预声明功效不足；mean_pref 不在 §15.3 表里）")
for m in ("bimodality_coefficient", "blrt_lr_per_n", "sd", "quad_intake", "mean_pref"):
    report(m, "T05", "N05", "T15", "N15", m)

print("\n[7] 护栏：预注册容差 vs 臂均，并做配对检验")
for m in ("death_thirst_frac", "carnivore_frac", "frugivory_frac", "min_pop", "population_mean", "population", "carn_speed"):
    if m not in REC[("T05", 0, 1)]:
        continue
    x, y = cell("T05", m), cell("N05", m)
    _, n_ub = noise(m, interaction=False)
    d = x - y
    p = wilcoxon(x, y, alternative="two-sided").pvalue
    lo, hi = bootstrap_ci(d)
    print(f"  {m:<18} N05={y.mean():>10.4f} T05={x.mean():>10.4f} Δ={d.mean():>+10.4f} ({d.mean()/max(abs(y.mean()),1e-9)*100:+6.1f}%)"
          f"  容差±{2*n_ub:<10.4f}(={2*n_ub/max(abs(y.mean()),1e-9)*100:5.1f}%臂均) {'破' if abs(d.mean())>2*n_ub else 'ok'}"
          f"   配对 p={p:.4f} CI=[{lo:+.3f},{hi:+.3f}] 符号{int((d<0).sum())}−/12")
