"""§16.1 的中性零假设，独立重算一遍——而且不用两等位基因的 WF 类比。

§16.1 用「p=0.13 的等位基因、2Ne·p/(1-p)·|ln p| ≈ 91 代」算丢失期望。
`forage_pref` 不是两等位基因：它是**连续性状 + 每次生育都加 N(0, 0.02²) 的复发突变**，
`forage_pref_of` 是 sigmoid(基因)，uniform crossover 让该位点从两亲之一取值 ⇒ 单倍体单位点。
在复发突变下等位基因**永不永久丢失**；会丢的是分布的一个**模态**。
所以正确的零假设是：把 cp0 实测的分布放进一个**只有漂变+突变、没有选择**的 WF 里，
跑该 run **自己实测的**世代数，再用**同一个** `split_score` / `low_mass` 读它。

三件事在这里定：
  1. Ne 的定标不靠解析式（§16.1 用 V*=2Ne·σ_m² 得 149；单倍体递推 V*=Ne·σ_m² 得 297）——
     直接在模拟器里跑到平稳，找哪个 Ne 给出中性臂实测的基因位方差 0.1189。
  2. 世代钟用每个 run 自己的 `traj.generation` 插值，不用全局均值。
  3. 读数用同一个 `split_score(bin_n, CTR)` 与同一套箱（0.15..0.85，14 箱，越界丢弃）。

输出：模拟 vs 实测的留存曲线与 low_mass 曲线。实测**快于**中性 ⇒ 主动淘汰，不是中性丢失。
重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260804-r13-verdict/neutral_null.py
"""
import glob, json, sys
import numpy as np

sys.path.insert(0, "explorations/20260804-readouts")
from split_score import split_score

BINS = np.linspace(0.15, 0.85, 15)
CTR = 0.5 * (BINS[:-1] + BINS[1:])
SIG = 0.02              # cfg.forage_pref_mutation_sigma
RNG = np.random.default_rng(0)


def wf_step(g, rng):
    """单倍体 WF 一代：每个后代随机取一个亲本的基因 + N(0, SIG²)。g: [R, Ne]"""
    R, Ne = g.shape
    idx = rng.integers(0, Ne, size=(R, Ne))
    return np.take_along_axis(g, idx, axis=1) + rng.normal(0.0, SIG, size=(R, Ne))


print("[定标] 平稳基因位方差 vs Ne（中性臂 N05 实测 0.1189，§16.1）")
for Ne in (75, 149, 200, 297, 400, 600):
    g = RNG.normal(0.0, 0.35, size=(24, Ne))
    for _ in range(int(12 * Ne)):
        g = wf_step(g, RNG)
    v = np.mean([np.var(row) for row in g])
    print(f"  Ne={Ne:>4}   平稳 var(gene)={v:.4f}   （解析 Ne·σ² = {Ne*SIG**2:.4f}）")

# ------------------------------------------------------------------ 读实测
runs = []
for f in sorted(glob.glob("outputs/20260804-traj/*.log")):
    for ln in open(f):
        if ln.startswith("JSON "):
            d = json.loads(ln[5:]); d["_file"] = f.split("/")[-1]; runs.append(d)
ent = [r for r in runs if r["checkpoints"][0]["split_score"] > 0]


def gen_at(r, t):
    tt = [q["t"] for q in r["traj"]]; gg = [q["generation"] for q in r["traj"]]
    return float(np.interp(t, tt, gg))


def sample_from_hist(n, R, N, rng):
    """把实测 bin_n 当密度，抽 R×N 个 pref，再反 sigmoid 回基因尺度。"""
    p = np.asarray(n, float); p = p / p.sum()
    k = rng.choice(len(CTR), size=(R, N), p=p)
    pref = BINS[k] + rng.random((R, N)) * (BINS[1] - BINS[0])
    pref = np.clip(pref, 1e-4, 1 - 1e-4)
    return np.log(pref / (1 - pref))


def readout(g, N_read, rng):
    """从基因池抽 N_read 个后代（含一代突变），做与实测同口径的直方图。"""
    R, Ne = g.shape
    idx = rng.integers(0, Ne, size=(R, N_read))
    gg = np.take_along_axis(g, idx, axis=1) + rng.normal(0.0, SIG, size=(R, N_read))
    pref = 1.0 / (1.0 + np.exp(-gg))
    ss, lm = np.empty(R), np.empty(R)
    for i in range(R):
        h = np.histogram(pref[i], bins=BINS)[0].astype(float)   # 越界丢弃，与 trajectory.py 同
        ss[i] = split_score(h, CTR)[0]
        hn = h / max(h.sum(), 1)
        lm[i] = hn[CTR < 0.35].sum()
    return ss, lm


for NE in (149, 297, 600):
    rng = np.random.default_rng(12345)
    R = 120
    sim_ret = np.zeros(5); sim_lm = np.zeros(5); nrun = 0
    per_run = []
    for r in ent:
        n_read = int(np.median([q["n_herb"] for q in r["traj"]]))
        g = sample_from_hist(r["checkpoints"][0]["bin_n"], R, NE, rng)
        g0 = gen_at(r, r["checkpoints"][0]["t"])
        ss, lm = readout(g, n_read, rng)
        row = [(float((ss > 0).mean()), float(lm.mean()))]
        done = 0
        for i in range(1, 5):
            need = int(round(gen_at(r, r["checkpoints"][i]["t"]) - g0))
            for _ in range(max(need - done, 0)):
                g = wf_step(g, rng)
            done = max(need, done)
            ss, lm = readout(g, n_read, rng)
            row.append((float((ss > 0).mean()), float(lm.mean())))
        per_run.append((r["_file"], row, int(round(gen_at(r, r["checkpoints"][4]["t"]) - g0))))
        sim_ret += np.array([x[0] for x in row]); sim_lm += np.array([x[1] for x in row])
        nrun += 1
    sim_ret /= nrun; sim_lm /= nrun

    obs_ret = np.array([sum(1 for r in ent if r["checkpoints"][i]["split_score"] > 0) / len(ent)
                        for i in range(5)])
    obs_lm = np.array([np.mean([r["checkpoints"][i]["low_mass"] for r in ent]) for i in range(5)])
    print(f"\n{'='*88}\n[中性零假设] Ne={NE}，R={R} 次重复/run，15 个入组 run，世代钟用各 run 实测")
    print(f"{'='*88}")
    print(f'  {"cp":>3} {"实测留存":>10} {"中性留存":>10} {"实测low_mass":>13} {"中性low_mass":>13}')
    for i in range(5):
        print(f"  {i:>3} {obs_ret[i]*len(ent):>7.0f}/15 {sim_ret[i]*len(ent):>7.1f}/15 "
              f"{obs_lm[i]:>13.4f} {sim_lm[i]:>13.4f}")
    gens = [x[2] for x in per_run]
    print(f"  实测 cp0→cp4 世代数：中位 {np.median(gens):.0f}  范围 [{min(gens)}, {max(gens)}]")
    # 逐 run：实测 cp4 low_mass 落在中性分布的哪个分位
    print(f'\n  逐 run（Ne={NE}）：cp0→cp4 世代数、实测 cp4 low_mass、中性 cp4 low_mass 均值')
    print(f'  {"file":<16} {"Δgen":>6} {"实测lm4":>9} {"中性lm4":>9} {"实测ss4":>9} {"中性留存率":>11}')
    for (f, row, dg), r in zip(per_run, ent):
        print(f"  {f:<16} {dg:>6} {r['checkpoints'][4]['low_mass']:>9.4f} {row[4][1]:>9.4f} "
              f"{r['checkpoints'][4]['split_score']:>9.4f} {row[4][0]:>11.3f}")
