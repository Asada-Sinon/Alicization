"""最干净的一版中性检验：**全部 24 个 run，从 cp0 起算**——入组条件在此不做任何筛选，
所以完全没有回归均值/选择污染。零假设仍是「只有漂变+突变的单倍体 WF，世代钟用各 run 实测」。

鞅论证：中性下 E[少数簇质量_t | 质量_0] = 质量_0，与 Ne 和世代数无关。
所以「均值系统性下降」本身就是选择的证据；模拟只是把零分布的**宽度**给出来。

重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260804-r13-verdict/null_clean.py
"""
import glob, json, math, sys
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, "explorations/20260804-readouts")
from split_score import split_score

BINS = np.linspace(0.15, 0.85, 15); CTR = 0.5 * (BINS[:-1] + BINS[1:]); SIG = 0.02
runs = []
for f in sorted(glob.glob("outputs/20260804-traj/*.log")):
    for ln in open(f):
        if ln.startswith("JSON "):
            d = json.loads(ln[5:]); d["_file"] = f.split("/")[-1]; runs.append(d)

gen_at = lambda r, t: float(np.interp(t, [q["t"] for q in r["traj"]],
                                      [q["generation"] for q in r["traj"]]))

def wf(g, rng):
    R, Ne = g.shape
    return np.take_along_axis(g, rng.integers(0, Ne, (R, Ne)), 1) + rng.normal(0, SIG, (R, Ne))

def init(n, R, Ne, rng):
    p = np.asarray(n, float)
    if p.sum() <= 0:
        return None
    p = p / p.sum()
    pref = np.clip(BINS[rng.choice(len(CTR), (R, Ne), p=p)] + rng.random((R, Ne)) * 0.05,
                   1e-4, 1 - 1e-4)
    return np.log(pref / (1 - pref))

def read(g, N, rng):
    R, Ne = g.shape
    pref = 1 / (1 + np.exp(-(np.take_along_axis(g, rng.integers(0, Ne, (R, N)), 1)
                             + rng.normal(0, SIG, (R, N)))))
    L, S = np.empty(R), np.empty(R)
    for i in range(R):
        h = np.histogram(pref[i], bins=BINS)[0].astype(float)
        L[i] = (h / max(h.sum(), 1))[CTR < 0.35].sum(); S[i] = split_score(h, CTR)[0]
    return L, S

def binom_test_low(k, n):
    """观测到 k 个（>=k 或 <=k，取偏离方向）低于中性中位数的单侧二项 p。"""
    if k > n / 2:
        return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n


obs0 = np.array([r["checkpoints"][0]["low_mass"] for r in runs])
obs4 = np.array([r["checkpoints"][4]["low_mass"] for r in runs])
print("=" * 92)
print("[全部 24 run，cp0 → cp4，无任何筛选]")
print("=" * 92)
print(f"  实测 cp0 均值 {obs0.mean():.4f} → cp4 均值 {obs4.mean():.4f}  差 {(obs4-obs0).mean():+.4f}")
nz = (obs0 > 0) | (obs4 > 0)
print(f"  逐 run：{int((obs4<obs0).sum())}/24 降 {int((obs4>obs0).sum())}/24 升 "
      f"{int((obs4==obs0).sum())}/24 平（均为 0-0）  Wilcoxon p={wilcoxon(obs0, obs4).pvalue:.5f}")
print(f"  只看两端非全零的 {int(nz.sum())} 个：{int((obs4[nz]<obs0[nz]).sum())} 降 / "
      f"{int((obs4[nz]>obs0[nz]).sum())} 升  p={wilcoxon(obs0[nz], obs4[nz]).pvalue:.5f}")

for Ne in (149, 297, 600):
    rng = np.random.default_rng(4242); R = 300
    L = np.zeros((len(runs), R)); S = np.zeros((len(runs), R))
    for j, r in enumerate(runs):
        g = init(r["checkpoints"][0]["bin_n"], R, Ne, rng)
        need = int(round(gen_at(r, r["checkpoints"][4]["t"]) - gen_at(r, r["checkpoints"][0]["t"])))
        for _ in range(max(need, 0)):
            g = wf(g, rng)
        L[j], S[j] = read(g, int(np.median([q["n_herb"] for q in r["traj"]])), rng)
    rg = np.random.default_rng(5)
    draws = np.array([L[np.arange(len(runs)), rg.integers(0, R, len(runs))].mean()
                      for _ in range(4000)])
    drawS = np.array([(S[np.arange(len(runs)), rg.integers(0, R, len(runs))] > 0).sum()
                      for _ in range(4000)])
    obsS = int(sum(1 for r in runs if r["checkpoints"][4]["split_score"] > 0))
    print(f"\n  Ne={Ne:>4}  中性 cp4 low_mass 均值 {L.mean():.4f}  "
          f"零分布 [5%,50%,95%] = [{np.percentile(draws,5):.4f}, {np.percentile(draws,50):.4f}, "
          f"{np.percentile(draws,95):.4f}]  实测 {obs4.mean():.4f}  单侧 p={float((draws<=obs4.mean()).mean()):.4f}")
    print(f"         中性 cp4 留存 中位 {np.median(drawS):.0f}/24 "
          f"[5%,95%]=[{np.percentile(drawS,5):.0f},{np.percentile(drawS,95):.0f}]  实测 {obsS}/24  "
          f"单侧 p={float((drawS<=obsS).mean()):.4f}")
    below = int(sum(obs4[j] < np.median(L[j]) for j in range(len(runs))))
    print(f"         实测低于本 run 中性中位的 {below}/24  （中性期望 12/24，二项单侧 p="
          f"{binom_test_low(below, 24):.5f}）")
