"""R11 复核第五轮：§15.3 的「预声明功效不足」有多少是测量代码自己造成的。

fitness_surface.py:117 `last = pref_all[-1]` —— `sd` / `mean_pref` /
`bimodality_coefficient` / `blrt_lr_per_n` 都算在**单帧**上；而主判据 `dip_ratio`
（以及 `frac_mid` / `low_mass`）算在 `bin_n` 上，那是 **600 帧池化**的。
这里把 sd 与 mean 换成 bin_n 池化版（= 存档字段 occ_sd / occ_mean，R10 臂可从 bin_n 补算），
在同一口径下重跑交互，看功效差多少。同时给 low_mass 换一个边界做敏感性。

重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260804-r11-verdict/verify_r11e.py
"""
import glob, json, math, sys
sys.path.insert(0, "scripts")
import numpy as np
from scipy.stats import chi2, wilcoxon
from exp_stats import bootstrap_ci

R = "/home/xrl/intern/Alicization/"
SEEDS, REPS = list(range(12)), [1, 2]


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
ctr = np.array(REC[("N05", 0, 1)]["bin_centers"])
NOISE_ARMS = ["N15", "T15", "N05", "N05m"]


def w_(d):
    n = np.array(d["bin_n"], float)
    return n / max(n.sum(), 1.0)


G = {
    "sd_单帧":      lambda d: d["sd"],
    "sd_池化600帧": lambda d: math.sqrt(max(float((w_(d) * (ctr - float((w_(d) * ctr).sum())) ** 2).sum()), 1e-12)),
    "mean_单帧":    lambda d: d["mean_pref"],
    "mean_池化":    lambda d: float((w_(d) * ctr).sum()),
    "low<0.35":     lambda d: float(w_(d)[ctr < 0.35].sum()),
    "low<0.30":     lambda d: float(w_(d)[ctr < 0.30].sum()),
    "low<0.25":     lambda d: float(w_(d)[ctr < 0.25].sum()),
}
cell = lambda a, g: np.array([np.mean([g(REC[(a, s, r)]) for r in REPS]) for s in SEEDS])
within = lambda a, g: np.array([np.std([g(REC[(a, s, r)]) for r in REPS], ddof=1) for s in SEEDS])

print(f"  {'读数':<14} {'N15':>8} {'T15':>8} {'N05':>8} {'T05':>8} {'N05m':>8} {'T05m':>8} |"
      f" {'交互':>9} {'符号':>6} {'p':>9} {'预注册比值':>10} {'经验比值':>8}")
for name, g in G.items():
    mus = [cell(a, g).mean() for a in ARMS]
    x, y = cell("T05", g) - cell("N05", g), cell("T15", g) - cell("N15", g)
    d = x - y
    w = np.concatenate([within(a, g) for a in NOISE_ARMS]); w = w[np.isfinite(w)]
    sw = float(np.sqrt((w ** 2).mean())); df = max(len(w), 1)
    n_ub = sw * math.sqrt(df / chi2.ppf(0.10, df)) * 2.0 / math.sqrt(2)
    p = wilcoxon(x, y).pvalue
    sgn = f"{int((d>0).sum())}+/12"
    print(f"  {name:<14} " + " ".join(f"{v:>8.4f}" for v in mus) +
          f" | {d.mean():>+9.4f} {sgn:>6} {p:>9.5f} {d.mean()/n_ub:>+10.2f} {d.mean()/d.std(ddof=1):>+8.2f}")
print("\n  （比值 = 交互 ÷ 噪声；预注册噪声 = 2·σ̂_W(90%上界)/√r，只从 N15/T15/N05/N05m 池化，§15.3）")
