"""把中性零假设做成一个真正的检验，并算出「主判据本来需要多大功效」。

三件事：
 A 鞅论证：中性下 E[少数簇频率_t] = 频率_0（与 Ne、世代数无关）。用模拟给出
   「15 个 run 的 cp4 low_mass 均值」在中性下的分布，读实测 0.0446 的分位。
 B 避开入组选择：从 cp1 起算，全部 24 个 run 重做一遍（cp1 不是入组判定点）。
 C 主判据的功效：由留存轨迹反解出的每代选择系数 s，再看 2000 步窗的 dlog 该有多大，
   与实测 dlog 噪声比。这决定 §16.4 的主判据是不是**设计上**就测不到。

重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260804-r13-verdict/null_test.py
"""
import glob, json, sys
import numpy as np

sys.path.insert(0, "explorations/20260804-readouts")
from split_score import split_score

BINS = np.linspace(0.15, 0.85, 15)
CTR = 0.5 * (BINS[:-1] + BINS[1:])
SIG = 0.02

runs = []
for f in sorted(glob.glob("outputs/20260804-traj/*.log")):
    for ln in open(f):
        if ln.startswith("JSON "):
            d = json.loads(ln[5:]); d["_file"] = f.split("/")[-1]; runs.append(d)
ent = [r for r in runs if r["checkpoints"][0]["split_score"] > 0]


def gen_at(r, t):
    return float(np.interp(t, [q["t"] for q in r["traj"]], [q["generation"] for q in r["traj"]]))


def wf(g, rng):
    R, Ne = g.shape
    idx = rng.integers(0, Ne, size=(R, Ne))
    return np.take_along_axis(g, idx, axis=1) + rng.normal(0.0, SIG, size=(R, Ne))


def init(n, R, Ne, rng):
    p = np.asarray(n, float); p = p / p.sum()
    k = rng.choice(len(CTR), size=(R, Ne), p=p)
    pref = np.clip(BINS[k] + rng.random((R, Ne)) * 0.05, 1e-4, 1 - 1e-4)
    return np.log(pref / (1 - pref))


def read_lm(g, N, rng):
    R, Ne = g.shape
    idx = rng.integers(0, Ne, size=(R, N))
    pref = 1 / (1 + np.exp(-(np.take_along_axis(g, idx, axis=1) + rng.normal(0, SIG, (R, N)))))
    out = np.empty(R); ss = np.empty(R)
    for i in range(R):
        h = np.histogram(pref[i], bins=BINS)[0].astype(float)
        hn = h / max(h.sum(), 1)
        out[i] = hn[CTR < 0.35].sum(); ss[i] = split_score(h, CTR)[0]
    return out, ss


def null_panel(subset, i0, i1, Ne, R, seed=7):
    """从 cp[i0] 起算到 cp[i1]，返回 [len(subset), R] 的中性 low_mass 与 split_score。"""
    rng = np.random.default_rng(seed)
    L = np.empty((len(subset), R)); S = np.empty((len(subset), R))
    for j, r in enumerate(subset):
        N = int(np.median([q["n_herb"] for q in r["traj"]]))
        g = init(r["checkpoints"][i0]["bin_n"], R, Ne, rng)
        need = int(round(gen_at(r, r["checkpoints"][i1]["t"]) - gen_at(r, r["checkpoints"][i0]["t"])))
        for _ in range(max(need, 0)):
            g = wf(g, rng)
        L[j], S[j] = read_lm(g, N, rng)
    return L, S


for tag, subset, i0 in (("A 入组 15，cp0→cp4（起点即入组判定点）", ent, 0),
                        ("B 全部 24，cp1→cp4（避开入组选择）", runs, 1)):
    print("=" * 92); print(f"[{tag}]"); print("=" * 92)
    obs_lm = np.array([r["checkpoints"][4]["low_mass"] for r in subset])
    obs_ss = np.array([r["checkpoints"][4]["split_score"] for r in subset])
    lm0 = np.array([r["checkpoints"][i0]["low_mass"] for r in subset])
    print(f"  起点 low_mass 均值 {lm0.mean():.4f}   实测 cp4 均值 {obs_lm.mean():.4f}   "
          f"实测 cp4 留存 {(obs_ss > 0).sum()}/{len(subset)}")
    for Ne in (149, 297, 600):
        R = 300
        L, S = null_panel(subset, i0, 4, Ne, R)
        rng = np.random.default_rng(99)
        # 零分布：每个 run 独立抽一个重复，取 15(24) 个的均值
        draws = np.array([L[np.arange(len(subset)), rng.integers(0, R, len(subset))].mean()
                          for _ in range(4000)])
        drawS = np.array([(S[np.arange(len(subset)), rng.integers(0, R, len(subset))] > 0).sum()
                          for _ in range(4000)])
        p_lm = float((draws <= obs_lm.mean()).mean())
        p_ss = float((drawS <= (obs_ss > 0).sum()).mean())
        below = int(sum(obs_lm[j] < np.median(L[j]) for j in range(len(subset))))
        print(f"  Ne={Ne:>4}  中性 cp4 low_mass 均值 {L.mean():.4f} "
              f"[零分布 5%,50%,95% = {np.percentile(draws,5):.4f}, {np.percentile(draws,50):.4f}, "
              f"{np.percentile(draws,95):.4f}]  单侧 p={p_lm:.4f}")
        print(f"         中性 cp4 留存 中位 {np.median(drawS):.0f}/{len(subset)} "
              f"[5%,95% = {np.percentile(drawS,5):.0f}, {np.percentile(drawS,95):.0f}]  "
              f"单侧 p={p_ss:.4f}   实测低于本 run 中性中位的 {below}/{len(subset)}")

print("\n" + "=" * 92)
print("[C] 由留存轨迹反解每代选择系数 s，并算主判据 dlog 在 2000 步窗上该有多大")
print("=" * 92)
dg = np.array([gen_at(r, r["checkpoints"][4]["t"]) - gen_at(r, r["checkpoints"][0]["t"]) for r in ent])
lm0 = np.array([r["checkpoints"][0]["low_mass"] for r in ent])
lm4 = np.array([r["checkpoints"][4]["low_mass"] for r in ent])
m = lm4 > 0
s_run = -np.log(lm4[m] / lm0[m]) / dg[m]
print(f"  逐 run 可解的 {m.sum()}/{len(ent)} 个（lm4>0）：s = {np.round(s_run, 5).tolist()}")
s_pool = -np.log(lm4.mean() / lm0.mean()) / np.median(dg)
print(f"  池化口径：s = -ln({lm4.mean():.4f}/{lm0.mean():.4f}) / {np.median(dg):.0f} 代 = {s_pool:.5f} /代")
print(f"  与漂变比：Ne·s（Ne=297）= {297*s_pool:.2f}   （<1 即「有效中性」，~1 即漂变与选择同量级）")
win_gen = 2000 / (100000 / np.median([r["gen_total"] for r in runs]))
print(f"  一个 2000 步窗 = {win_gen:.1f} 代  ⇒ 预期 dlog = -s·代数 = {-s_pool*win_gen:+.4f}")
dl = []
for r in ent:
    for cp in r["checkpoints"]:
        n = np.array(cp["bin_n"], float); n = n / max(n.sum(), 1)
        d = np.array([np.nan if v is None else v for v in cp["dlog_per_bin"]], float)
        mm = (CTR < 0.35) & np.isfinite(d) & (n > 0)
        if mm.any():
            dl.append(float((d[mm] * n[mm]).sum() / n[mm].sum()))
dl = np.array(dl)
print(f"  实测低箱 dlog 的单格 SD = {dl.std(ddof=1):.4f}（n={len(dl)} 格）")
print(f"  ⇒ 单格 信号/噪声 = {abs(-s_pool*win_gen)/dl.std(ddof=1):.4f}；"
      f"n=9 个 run 平均后 = {abs(-s_pool*win_gen)/(dl.std(ddof=1)/np.sqrt(9)):.4f}")
need = (dl.std(ddof=1) / abs(-s_pool * win_gen)) ** 2
print(f"  要让效应/噪声达到 1，需要 n ≈ {need:.0f} 个独立 run（§16.4 的比值门槛）")
