"""R14 Stage 2 判决的独立复核 + 判据 §18.4 里没写的「每臂 Δ 是否异于 0」。

回答：①`outputs/20260804-ratio2_analysis.txt` 的每个数能否从原始日志复现；
     ②四个臂各自的 `Δlow_mass` 是否显著异于 0（主控没做，是本轮最显著观察的直接检验）；
     ③`R38n` vs `R38p`（只动 `diet_delta` 的事后对比）到底有多大。
读：outputs/20260804-ratio2/*.log（96 run，每个一行 "JSON "）。
输出：stdout；每节标了它对应 §18 的哪一条。
重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260804-r14b-verdict/verify_stage2.py
"""
import glob, json, math, sys
sys.path.insert(0, "scripts")
import numpy as np
from scipy.stats import chi2, wilcoxon
from exp_stats import bootstrap_ci

CTR = np.linspace(0.0, 1.0, 21); CTR = 0.5 * (CTR[:-1] + CTR[1:])
SEEDS, REPS = list(range(12)), [1, 2]
ARMS = ["R38p", "R50p", "R38n", "R50n"]
SPEC = {"R38p": (0.25, 0.010, 0.15), "R50p": (0.50, 0.005, 0.15),
        "R38n": (0.25, 0.010, 1.5), "R50n": (0.50, 0.005, 1.5)}

REC = {}
for f in sorted(glob.glob("outputs/20260804-ratio2/*.log")):
    for ln in open(f):
        if ln.startswith("JSON "):
            b = f.split("/")[-1][:-4].split("_")
            REC[(b[0], int(b[1][1:]), int(b[2][1:]))] = json.loads(ln[5:])
assert len(REC) == 96, len(REC)
for a in ARMS:
    for s in SEEDS:
        for r in REPS:
            o = REC[(a, s, r)]["overrides"]
            got = tuple(float(o[k]) for k in ("fruit_regrow_baseline", "regrow_baseline", "diet_delta"))
            assert got == SPEC[a], (a, s, r, got)
            assert REC[(a, s, r)]["collapsed"] is False
print("载入 96 run；overrides 自检通过（每臂只有 fruit_regrow_baseline/regrow_baseline/diet_delta 三项按表）")

# --- 臂间 overrides 差异枚举（单变量归因闸，§18.6 之外我自己加的）------------------
print("\n[臂间 overrides 差异枚举]")
for A, B in (("R50n", "R38n"), ("R50p", "R38p"), ("R38n", "R38p"), ("R50n", "R50p")):
    oa, ob = REC[(A, 0, 1)]["overrides"], REC[(B, 0, 1)]["overrides"]
    diff = sorted(k for k in set(oa) | set(ob) if oa.get(k) != ob.get(k))
    print(f"  {A} vs {B}: {len(diff)} 项 -> {[(k, ob.get(k), oa.get(k)) for k in diff]}")

lm  = lambda a, s, r, i: REC[(a, s, r)]["checkpoints"][i]["low_mass"]
dlm = lambda a, s, r: lm(a, s, r, 1) - lm(a, s, r, 0)
cell = lambda a, fn: np.array([np.mean([fn(a, s, r) for r in REPS]) for s in SEEDS])
within = lambda a, fn: np.array([np.std([fn(a, s, r) for r in REPS], ddof=1) for s in SEEDS])

def noise(fn, k, arms=ARMS):
    """§18.7：sigma_W 从本实验自估，取 90% 卡方上界，噪声 = k * sw_ub / sqrt(r)。"""
    w = np.concatenate([within(a, fn) for a in arms]); w = w[np.isfinite(w)]
    sw = float(np.sqrt((w ** 2).mean()))
    return sw * math.sqrt(len(w) / chi2.ppf(0.10, len(w))) * k / math.sqrt(len(REPS))

print("\n" + "=" * 88)
print("复核 A：H0 / H1 / H2 是否可从原始日志复现（对照 outputs/20260804-ratio2_analysis.txt）")
print("=" * 88)
for lbl, A, B in (("H0  low_mass(cp0) R50n-R38n", "R50n", "R38n"),
                  ("H0' low_mass(cp0) R50p-R38p", "R50p", "R38p")):
    x, y = cell(A, lambda a, s, r: lm(a, s, r, 0)), cell(B, lambda a, s, r: lm(a, s, r, 0))
    d = x - y; nz = noise(lambda a, s, r: lm(a, s, r, 0), math.sqrt(2.0))
    lo, hi = bootstrap_ci(d)
    print(f"  {lbl}: {x.mean():+.4f} - {y.mean():+.4f} = {d.mean():+.4f}  "
          f"{(d>0).sum()}/12 正  p={wilcoxon(x,y).pvalue:.5f}  CI=[{lo:+.4f},{hi:+.4f}]  "
          f"噪声={nz:.4f}  比值={d.mean()/nz:+.2f}")
for lbl, A, B in (("H1  dlow_mass R50n-R38n", "R50n", "R38n"),
                  ("H1' dlow_mass R50p-R38p", "R50p", "R38p")):
    x, y = cell(A, dlm), cell(B, dlm)
    d = x - y; nz = noise(dlm, math.sqrt(2.0)); lo, hi = bootstrap_ci(d)
    print(f"  {lbl}: {x.mean():+.4f} - {y.mean():+.4f} = {d.mean():+.4f}  "
          f"{(d>0).sum()}/12 正  p={wilcoxon(x,y).pvalue:.5f}  CI=[{lo:+.4f},{hi:+.4f}]  "
          f"噪声={nz:.4f}  比值={d.mean()/nz:+.2f}")
xi = cell("R50n", dlm) - cell("R38n", dlm); yi = cell("R50p", dlm) - cell("R38p", dlm)
d = xi - yi; nz = noise(dlm, 2.0); lo, hi = bootstrap_ci(d)
print(f"  H2  交互: {xi.mean():+.4f} - {yi.mean():+.4f} = {d.mean():+.4f}  "
      f"{(d>0).sum()}/12 正  p={wilcoxon(xi,yi).pvalue:.5f}  CI=[{lo:+.4f},{hi:+.4f}]  "
      f"噪声={nz:.4f}  比值={d.mean()/nz:+.2f}")

print("\n" + "=" * 88)
print("复核 B（**新增，§18.4 没写**）：每个臂的 Δlow_mass 单独检验是否异于 0")
print("  单样本 Wilcoxon（12 个格均值 vs 0）；噪声 = 1*sw_ub/sqrt(r)，不是 sqrt(2)——单臂无配对差")
print("=" * 88)
nz1 = noise(dlm, 1.0)
for a in ARMS:
    d = cell(a, dlm); lo, hi = bootstrap_ci(d)
    p = wilcoxon(d).pvalue
    print(f"  {a:>5}  Δ={d.mean():+.4f}  {(d>0).sum()}/12 正  p={p:.5f}  "
          f"CI=[{lo:+.4f},{hi:+.4f}]{'  含0' if lo<=0<=hi else '  **不含0**'}  "
          f"噪声={nz1:.4f}  比值={d.mean()/nz1:+.2f}"
          f"{'  <1 undecidable' if abs(d.mean()/nz1)<1 else ''}")
    print(f"         逐种子 = {np.round(d,4).tolist()}")
print(f"  cp0/cp1 臂均: " + "  ".join(
    f"{a}: {cell(a, lambda x,s,r: lm(x,s,r,0)).mean():.4f}->{cell(a, lambda x,s,r: lm(x,s,r,1)).mean():.4f}"
    for a in ARMS))

print("\n" + "=" * 88)
print("复核 C（事后，非预注册）：R38n vs R38p —— 只动 diet_delta，配比不动")
print("=" * 88)
for key, fn in (("low_mass(cp0)", lambda a,s,r: lm(a,s,r,0)), ("low_mass(cp1)", lambda a,s,r: lm(a,s,r,1)),
                ("Δlow_mass", dlm)):
    for A, B in (("R38n", "R38p"), ("R50n", "R50p")):
        x, y = cell(A, fn), cell(B, fn); d = x - y
        nz = noise(fn, math.sqrt(2.0)); lo, hi = bootstrap_ci(d)
        print(f"  {key:>14} {A}-{B}: {x.mean():+.4f} - {y.mean():+.4f} = {d.mean():+.4f}  "
              f"{(d>0).sum()}/12 正  p={wilcoxon(x,y).pvalue:.5f}  CI=[{lo:+.4f},{hi:+.4f}]  比值={d.mean()/nz:+.2f}")

print("\n" + "=" * 88)
print("复核 D：总通量 / 种群 / frugivory / 捕食者丢失（§18.6 §18.7）")
print("=" * 88)
flux = {a: np.array([np.mean([REC[(a,s,r)]["checkpoints"][0]["graze_gain"]
                              + REC[(a,s,r)]["checkpoints"][0]["fruit_gain"] for r in REPS]) for s in SEEDS])
        for a in ARMS}
popm = {a: np.array([np.mean([np.nanmean([p["population"] for p in REC[(a,s,r)]["traj"]
                                          if np.isfinite(p["population"])]) for r in REPS]) for s in SEEDS])
        for a in ARMS}
for a in ARMS:
    lost = sum(1 for s in SEEDS for r in REPS
               if any(p["carnivore_frac"] == 0 for p in REC[(a,s,r)]["traj"][len(REC[(a,s,r)]["traj"])//2:]))
    print(f"  {a:>5}  通量 {flux[a].mean():7.1f} (种子间SD {flux[a].std(ddof=1):5.1f})  "
          f"种群 {popm[a].mean():6.0f} (SD {popm[a].std(ddof=1):5.0f})  后半程 carn=0 的 run {lost}/24")
for A, B in (("R50n","R38n"), ("R50p","R38p"), ("R38n","R38p")):
    d = flux[A].mean() - flux[B].mean()
    print(f"  通量差 {A}-{B} = {d:+.1f}；两臂种子间 SD = "
          f"{flux[A].std(ddof=1):.1f}/{flux[B].std(ddof=1):.1f}  => "
          f"{'多变量' if abs(d) > max(flux[A].std(ddof=1), flux[B].std(ddof=1)) else '可单独归因'}")
# 通量与 Δlow_mass 的关系：若配比效应真的被通量搬走，两者该相关
from scipy.stats import spearmanr
allf = np.concatenate([flux[a] for a in ARMS]); alld = np.concatenate([cell(a, dlm) for a in ARMS])
rho, pv = spearmanr(allf, alld)
print(f"  Spearman(总通量, Δlow_mass) 跨 48 格 = {rho:+.3f} (p={pv:.4f})")
rho2, pv2 = spearmanr(np.concatenate([flux[a] for a in ARMS]),
                      np.concatenate([cell(a, lambda x,s,r: lm(x,s,r,0)) for a in ARMS]))
print(f"  Spearman(总通量, low_mass(cp0)) 跨 48 格 = {rho2:+.3f} (p={pv2:.4f})")
