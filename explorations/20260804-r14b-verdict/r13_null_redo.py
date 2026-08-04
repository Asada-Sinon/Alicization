"""R13 判决（§14）的重算：把两个「判决之后才修好」的读数缺陷同时回滚。

缺陷 1（截断）：checkpoints[i]["bin_n"] 用 np.digitize + BINS=linspace(0.15,0.85,15)，
  窗外个体从分子分母同时剔除。修法：改用 traj 的 clip 直方图（尾巴进边缘箱）。
缺陷 2（检测器）：判决 commit 17ff01b 时 split_score 用 np.convolve(mode="same") 补零平滑，
  **落在边界箱的峰被抹掉** ⇒ 边缘双峰读 0。803dc60 已修（改 pad(mode="edge")），
  但 R13 日志里存的是旧值。修法：用当前 split_score 重算。

三件事：A 用当前检测器重算留存曲线与「连续保住」；
       B 用不截断口径重建中性零假设（复刻 null_test.py 的 WF 模拟器），复核「实测快于中性」；
       C 重算反解的每代选择系数 s。
输出：stdout。重跑：
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python explorations/20260804-r14b-verdict/r13_null_redo.py
"""
import glob, json, math, sys
sys.path.insert(0, "scripts"); sys.path.insert(0, "explorations/20260804-readouts")
import numpy as np
from scipy.stats import wilcoxon
from exp_stats import bootstrap_ci
from split_score import split_score, retained

BINS = np.linspace(0.15, 0.85, 15); CTR = 0.5*(BINS[:-1]+BINS[1:]); LOW = CTR < 0.35
SIG = 0.02
runs = []
for f in sorted(glob.glob("outputs/20260804-traj/*.log")):
    for ln in open(f):
        if ln.startswith("JSON "):
            d = json.loads(ln[5:]); d["_f"] = f.split("/")[-1][:-4]; runs.append(d)
at = lambda r,t: [q for q in r["traj"] if q["t"] == t][-1]
clip_h = lambda r,i: np.array(at(r, r["checkpoints"][i]["t"])["hist"], float)
trunc_h = lambda r,i: np.array(r["checkpoints"][i]["bin_n"], float)
gen_at = lambda r,t: float(np.interp(t, [q["t"] for q in r["traj"]], [q["generation"] for q in r["traj"]]))

print("="*98); print("A 留存曲线：两个缺陷各自的贡献（当前 split_score vs 日志里存的旧值）"); print("="*98)
for tag, getter, use_stored in (
        ("① §14.2 原样（截断 bin_n + 旧检测器，日志值）", trunc_h, True),
        ("② 只修检测器（截断 bin_n + 当前 split_score）", trunc_h, False),
        ("③ 只修截断（clip hist + 旧检测器，日志值）", clip_h, True),
        ("④ 两个都修（clip hist + 当前 split_score）", clip_h, False)):
    tab = {}
    for r in runs:
        if use_stored:
            v = [r["checkpoints"][i]["split_score"] > 0 if getter is trunc_h
                 else at(r, r["checkpoints"][i]["t"])["split_score"] > 0 for i in range(5)]
        else:
            v = [split_score(getter(r,i), CTR)[0] > 0 for i in range(5)]
        tab[r["_f"]] = v
    curve = [sum(v[i] for v in tab.values()) for i in range(5)]
    ent = {n:v for n,v in tab.items() if v[0]}
    print(f"  {tag:<46} 留存(分母24) {curve}  入组 {len(ent)}/24  "
          f"连续五点全保住 {sum(1 for v in ent.values() if all(v))}/{len(ent)}  "
          f"（全体口径 {sum(1 for v in tab.values() if all(v))}/24）")
tabR = {r["_f"]: [retained(clip_h(r,i), CTR) for i in range(5)] for r in runs}
print(f"  {'⑤ 换 803dc60 的 retained()（质量+缺口）+ clip':<46} 留存(分母24) "
      f"{[sum(v[i] for v in tabR.values()) for i in range(5)]}  "
      f"入组 {sum(1 for v in tabR.values() if v[0])}/24  连续五点全保住 "
      f"{sum(1 for v in tabR.values() if all(v))}/{sum(1 for v in tabR.values() if v[0])}")

# ---------------- B 中性零假设（复刻 null_test.py 的模拟器，只换读数口径）-------------
def wf(g, rng):
    R, Ne = g.shape
    idx = rng.integers(0, Ne, size=(R, Ne))
    return np.take_along_axis(g, idx, axis=1) + rng.normal(0.0, SIG, size=(R, Ne))

def init_from(h, R, Ne, rng, bin0_lo=0.15):
    """按箱质量抽初值。bin0_lo 控制 clip 的边缘箱铺开到哪（敏感性用）。"""
    p = np.asarray(h, float); p = p/p.sum()
    k = rng.choice(len(CTR), size=(R, Ne), p=p)
    lo = np.where(k == 0, bin0_lo, BINS[k])
    wd = np.where(k == 0, 0.20-bin0_lo, 0.05)
    pref = np.clip(lo + rng.random((R, Ne))*wd, 1e-4, 1-1e-4)
    return np.log(pref/(1-pref))

def read(g, N, rng, mode):
    R, Ne = g.shape
    idx = rng.integers(0, Ne, size=(R, N))
    pref = 1/(1+np.exp(-(np.take_along_axis(g, idx, axis=1) + rng.normal(0, SIG, (R, N)))))
    if mode == "clip":                                  # 不截断：分母是全体
        return (pref < 0.35).mean(axis=1)
    out = np.empty(R)                                   # 截断：np.histogram 丢掉窗外
    for i in range(R):
        hh = np.histogram(pref[i], bins=BINS)[0].astype(float)
        out[i] = (hh/max(hh.sum(),1))[LOW].sum()
    return out

def null_run(mode, getter, bin0_lo=0.15, R=300, seed=7):
    rng = np.random.default_rng(seed)
    obs = np.array([ (getter(r,4)/max(getter(r,4).sum(),1))[LOW].sum() if mode=="clip"
                     else r["checkpoints"][4]["low_mass"] for r in runs])
    lm0 = np.array([ (getter(r,0)/max(getter(r,0).sum(),1))[LOW].sum() if mode=="clip"
                     else r["checkpoints"][0]["low_mass"] for r in runs])
    out = {}
    for Ne in (149, 297, 600):
        L = np.empty((len(runs), R))
        for j, r in enumerate(runs):
            N = int(np.median([q["n_herb"] for q in r["traj"]]))
            g = init_from(getter(r,0), R, Ne, rng, bin0_lo)
            need = int(round(gen_at(r, r["checkpoints"][4]["t"]) - gen_at(r, r["checkpoints"][0]["t"])))
            for _ in range(max(need,0)):
                g = wf(g, rng)
            L[j] = read(g, N, rng, mode)
        rng2 = np.random.default_rng(99)
        draws = np.array([L[np.arange(len(runs)), rng2.integers(0,R,len(runs))].mean() for _ in range(4000)])
        out[Ne] = (L.mean(), float((draws <= obs.mean()).mean()),
                   np.percentile(draws,5), np.percentile(draws,95),
                   int(sum(obs[j] < np.median(L[j]) for j in range(len(runs)))))
    return lm0.mean(), obs.mean(), out

print("\n" + "="*98)
print("B 中性零假设：全 24 run，cp0->cp4，模拟器与 null_test.py 相同，只换读数口径")
print("="*98)
for mode, getter, lo, tag in (("trunc", trunc_h, 0.15, "截断（§14.3 原样，复现用）"),
                              ("clip", clip_h, 0.15, "不截断，边缘箱铺在 [0.15,0.20)"),
                              ("clip", clip_h, 0.05, "不截断，边缘箱铺在 [0.05,0.20)（敏感性）")):
    s0, so, res = null_run(mode, getter, lo)
    print(f"  {tag}")
    print(f"    起点 low_mass 均值 {s0:.4f}   实测 cp4 均值 {so:.4f}")
    for Ne,(m,p,q5,q95,below) in res.items():
        print(f"      Ne={Ne:>3}  中性 cp4 均值 {m:.4f}  零分布[5%,95%]=[{q5:.4f},{q95:.4f}]  "
              f"单侧 p={p:.4f}   实测低于本 run 中性中位的 {below}/24")

print("\n" + "="*98)
print("C 反解每代选择系数 s（池化口径，与 null_test.py [C] 同式）")
print("="*98)
dg = np.array([gen_at(r, r["checkpoints"][4]["t"]) - gen_at(r, r["checkpoints"][0]["t"]) for r in runs])
ent_tr = [r for r in runs if r["checkpoints"][0]["split_score"] > 0]
for tag, getter, subset in (("截断 / §14 入组 15 run（原样）", trunc_h, ent_tr),
                            ("截断 / 全 24 run", trunc_h, runs),
                            ("不截断 / 全 24 run", clip_h, runs)):
    if getter is trunc_h:
        a = np.array([r["checkpoints"][0]["low_mass"] for r in subset])
        b = np.array([r["checkpoints"][4]["low_mass"] for r in subset])
    else:
        a = np.array([(getter(r,0)/getter(r,0).sum())[LOW].sum() for r in subset])
        b = np.array([(getter(r,4)/getter(r,4).sum())[LOW].sum() for r in subset])
    d = np.array([gen_at(r, r["checkpoints"][4]["t"]) - gen_at(r, r["checkpoints"][0]["t"]) for r in subset])
    s = -math.log(b.mean()/a.mean())/np.median(d)
    print(f"  {tag:<32} s = -ln({b.mean():.4f}/{a.mean():.4f})/{np.median(d):.0f} 代 = {s:+.5f}/代   "
          f"Ne*s (Ne=297) = {297*s:+.2f}")
print(f"  世代数 cp0->cp4：中位 {np.median(dg):.0f}，范围 {dg.min():.0f}–{dg.max():.0f}")
