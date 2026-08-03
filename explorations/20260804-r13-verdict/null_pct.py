"""null_clean.py 的逐 run 分位数修正：那里用严格 `<` 数「低于中性中位」，
而 7/24 个 run 的实测与中性都恒为 0（平局），被算成「不低于」，给出方向相反的假象。
改用**中位分位**：pct = P(中性 < 实测) + 0.5·P(中性 = 实测)，中性下应均匀分布在 0.5 附近。

重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260804-r13-verdict/null_pct.py
"""
import glob, json, sys
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
    p = np.asarray(n, float); p = p / p.sum()
    pref = np.clip(BINS[rng.choice(len(CTR), (R, Ne), p=p)] + rng.random((R, Ne)) * 0.05,
                   1e-4, 1 - 1e-4)
    return np.log(pref / (1 - pref))

def read_lm(g, N, rng):
    R, Ne = g.shape
    pref = 1 / (1 + np.exp(-(np.take_along_axis(g, rng.integers(0, Ne, (R, N)), 1)
                             + rng.normal(0, SIG, (R, N)))))
    return np.array([(np.histogram(pref[i], bins=BINS)[0]
                      / max(np.histogram(pref[i], bins=BINS)[0].sum(), 1))[CTR < 0.35].sum()
                     for i in range(R)])

for Ne in (149, 297, 600):
    rng = np.random.default_rng(4242); R = 300
    pct, names = [], []
    for r in runs:
        g = init(r["checkpoints"][0]["bin_n"], R, Ne, rng)
        need = int(round(gen_at(r, r["checkpoints"][4]["t"]) - gen_at(r, r["checkpoints"][0]["t"])))
        for _ in range(max(need, 0)):
            g = wf(g, rng)
        L = read_lm(g, int(np.median([q["n_herb"] for q in r["traj"]])), rng)
        o = r["checkpoints"][4]["low_mass"]
        pct.append(float((L < o).mean() + 0.5 * (np.isclose(L, o, atol=1e-9)).mean()))
        names.append(r["_file"])
    pct = np.array(pct)
    print(f"Ne={Ne:>4}  逐 run 分位（中性下应 ~0.5）：均值 {pct.mean():.3f}  中位 {np.median(pct):.3f}  "
          f"{int((pct<0.5).sum())}/24 低于 0.5  Wilcoxon(vs 0.5) p={wilcoxon(pct-0.5).pvalue:.5f}")
    print(f"        分位 <0.05 的 run 数 {int((pct<0.05).sum())}/24；"
          f"逐 run 分位 {dict(zip([n[:-4] for n in names], np.round(pct,3)))}")
