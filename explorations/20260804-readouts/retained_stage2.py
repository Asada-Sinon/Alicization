"""把 §17.9 的判据原样套到 R14 Stage 2 的 96 个 run 上。**零 GPU。**

为什么这比新跑一轮更有力
------------------------
§17.9 在 R13 的 24 个 run 上拒绝了中性，但**敏感性检验（去掉 3 个丢失捕食者的 run）
在最宽松的 `N` 上 p=0.060 不过线**，所以「有捕食者时也会分裂」判了 undecidable。

Stage 2 恰好是为此设计的：**4 臂 × 12 种子 × 2 重复 = 96 run**，
`diet_delta=1.5` 由**构造**关掉捕食（不是靠自发灭绝，没有 collider 风险），
且它跑在**修好的 20 箱读数**上（`BINS = linspace(0,1,21)`，`in_window` 四臂全 1.0000）。

于是同一个判据在这里有：真正的实验操纵、12 个配对种子、r=2 自估噪声。

代价：Stage 2 只跑 42k 步 ⇒ `n` 臂 28–40 代、`p` 臂 91–102 代
（**世代钟不等长，必须写明，不许按步比**）。R13 是 118–613 代。

跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
      explorations/20260804-readouts/retained_stage2.py
"""
import glob
import json
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, "explorations/20260804-readouts")

import numpy as np
from scipy.stats import wilcoxon
from split_score import retained

BINS = np.linspace(0.0, 1.0, 21)          # Stage 2 用的是修好的全量程分箱
CTR = 0.5 * (BINS[:-1] + BINS[1:])
MUT_SIGMA = 0.02
OCC_THRESH = 0.5
ARMS = ["R38p", "R50p", "R38n", "R50n"]
SEEDS = list(range(12))


def clip_hist(pref):
    lo, w, nb = float(BINS[0]), float(BINS[1] - BINS[0]), len(CTR)
    idx = np.clip(((pref - lo) / w).astype(np.int32), 0, nb - 1)
    return np.bincount(idx, minlength=nb).astype(float)


def genes_from_hist(h, rng):
    counts = h.astype(int)
    out = [rng.uniform(BINS[k], BINS[k + 1], size=counts[k])
           for k in range(len(CTR)) if counts[k] > 0]
    p = np.concatenate(out) if out else np.full(1, 0.5)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def occupancy_second_half(hists, gen):
    ret = np.array([retained(h, CTR) for h in hists], float)
    w = np.clip(np.gradient(gen), 0.0, None)
    half = len(ret) // 2
    return float((w[half:] * ret[half:]).sum() / max(w[half:].sum(), 1e-12))


def sim_run(genes0, n_gen, n_frames, N, rng):
    g = np.asarray(genes0, float)
    if len(g) > N:
        g = g[rng.integers(0, len(g), size=N)]
    take = np.linspace(0, max(n_gen, 1), n_frames).astype(int)
    out, j = [], 0
    for t in range(max(n_gen, 1) + 1):
        while j < n_frames and take[j] == t:
            out.append(clip_hist(1.0 / (1.0 + np.exp(-g))))
            j += 1
        g = g[rng.integers(0, len(g), size=N)] + rng.normal(0.0, MUT_SIGMA, size=N)
    while len(out) < n_frames:
        out.append(clip_hist(1.0 / (1.0 + np.exp(-g))))
    return out


def main():
    R = {}
    for f in sorted(glob.glob("outputs/20260804-ratio2/*.log")):
        txt = open(f).read()
        if "JSON " not in txt:
            continue
        d = json.loads(txt.split("JSON ")[1].split("\n")[0])
        b = f.split("/")[-1][:-4].split("_")
        tr = d["traj"]
        h = np.array([q["hist"] for q in tr], float)
        assert np.allclose(h.sum(1), [q["n_herb"] for q in tr], rtol=1e-6, atol=1e-3), f
        R[(b[0], int(b[1][1:]), int(b[2][1:]))] = {
            "hist": h, "gen": np.array([q["generation"] for q in tr], float)}
    print("=" * 96)
    print("§17.9 的判据套到 Stage 2 的 96 run 上（4 臂 × 12 种子 × 2 重复，修好的 20 箱读数）")
    print("=" * 96)
    print(f"  载入 {len(R)} run")

    occ = {a: np.array([[occupancy_second_half(R[(a, s, r)]["hist"], R[(a, s, r)]["gen"])
                         for r in (1, 2)] for s in SEEDS]) for a in ARMS}
    gen = {a: np.array([[R[(a, s, r)]["gen"][-1] - R[(a, s, r)]["gen"][0]
                         for r in (1, 2)] for s in SEEDS]) for a in ARMS}

    print()
    print(f'  {"臂":<6}{"占空比格均值":>13}{"格均>0.5 的种子":>16}{"逐 run >0.5":>13}{"跨代":>9}')
    for a in ARMS:
        cell = occ[a].mean(1)
        print(f'  {a:<6}{cell.mean():>13.4f}{int((cell > OCC_THRESH).sum()):>16}/12'
              f'{int((occ[a] > OCC_THRESH).sum()):>10}/24{gen[a].mean():>9.1f}')

    print()
    print("  配对对比（先格均值 → 12 格配对符号秩；σ̂_W 只在参与对比的臂上池化）")
    for x, y, why in (("R38n", "R38p", "只动 diet_delta：关掉捕食通道"),
                      ("R50n", "R50p", "同上，在高果供给下"),
                      ("R50n", "R38n", "只动资源比：果供给翻倍（无捕食者）"),
                      ("R50p", "R38p", "同上（有捕食者）")):
        cx, cy = occ[x].mean(1), occ[y].mean(1)
        d = cx - cy
        w = np.concatenate([occ[x].std(1, ddof=1), occ[y].std(1, ddof=1)])
        sw = float(np.sqrt((w ** 2).mean()))
        noise = np.sqrt(2) * sw / np.sqrt(2)
        p = wilcoxon(cx, cy).pvalue if np.any(d != 0) else float("nan")
        print(f'    {x}−{y}: {d.mean():+.4f}   {int((d > 0).sum())}/12 为正   p={p:.5f}   '
              f'比值 {d.mean() / max(noise, 1e-12):+.2f}   ({why})')

    REP = 200
    rng = np.random.default_rng(20260804)
    print()
    print("  中性零假设（每 run 从自己帧 0 的实测直方图起跑，跑该 run 实测代数，同一 `retained`）")
    print(f'  {"N":>7}' + "".join(f'{a + " 零均值":>16}' for a in ARMS))
    null = {a: {} for a in ARMS}
    for N in (105, 340, 600):
        line = f'  {N:>7}'
        for a in ARMS:
            cnt = []
            for _ in range(REP):
                c = 0
                for s in SEEDS:
                    for r in (1, 2):
                        d = R[(a, s, r)]
                        span = int(max(d["gen"][-1] - d["gen"][0], 5))
                        hs = sim_run(genes_from_hist(d["hist"][0], rng), span,
                                     len(d["hist"]), N, rng)
                        if occupancy_second_half(hs, d["gen"]) > OCC_THRESH:
                            c += 1
                cnt.append(c)
            cnt = np.array(cnt)
            obs = int((occ[a] > OCC_THRESH).sum())
            null[a][N] = (float(cnt.mean()), float((cnt >= obs).mean()))
            line += f'{cnt.mean():>8.2f} p={float((cnt >= obs).mean()):<6.4f}'
        print(line)
    print(f"  （每格 {REP} 次 24-run 重复；`obs` 是该臂 24 个 run 里占空比>0.5 的个数）")


if __name__ == "__main__":
    main()
