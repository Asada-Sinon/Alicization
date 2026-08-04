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
from neutral_null import OCC_THRESH, occupancy, report
from scipy.stats import wilcoxon
from split_score import split_score

BINS = np.linspace(0.0, 1.0, 21)        # Stage 2 用的是修好的全量程分箱
CTR = 0.5 * (BINS[:-1] + BINS[1:])
LOW = CTR < 0.35
ARMS = ["R38p", "R50p", "R38n", "R50n"]
SEEDS = list(range(12))


def load():
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
    return R


def second_half_mean(d, fn):
    """后半程的世代加权均值。"""
    h, g = d["hist"], d["gen"]
    half = len(h) // 2
    w = np.clip(np.gradient(g), 0.0, None)[half:]
    v = np.array([fn(h[i]) for i in range(half, len(h))])
    return float((w * v).sum() / max(w.sum(), 1e-12))


def main():
    R = load()
    print("=" * 96)
    print("§17.9 的判据套到 Stage 2 的 96 run 上（4 臂 × 12 种子 × 2 重复，修好的 20 箱读数）")
    print("=" * 96)
    print(f"  载入 {len(R)} run")

    occ = {a: np.array([[occupancy(R[(a, s, r)]["hist"], R[(a, s, r)]["gen"], CTR)
                         for r in (1, 2)] for s in SEEDS]) for a in ARMS}
    gen = {a: np.array([[R[(a, s, r)]["gen"][-1] - R[(a, s, r)]["gen"][0]
                         for r in (1, 2)] for s in SEEDS]) for a in ARMS}

    # **只看占空比会把四个臂看成一样的**（§18.4）：必须与量级读数一起报
    lm = {a: np.array([second_half_mean(R[(a, s, r)],
                                        lambda h: h[LOW].sum() / max(h.sum(), 1.0))
                       for s in SEEDS for r in (1, 2)]) for a in ARMS}
    ss = {a: np.array([second_half_mean(R[(a, s, r)], lambda h: split_score(h, CTR)[0])
                       for s in SEEDS for r in (1, 2)]) for a in ARMS}

    print()
    print(f'  {"臂":<6}{"占空比":>9}{"格均>0.5":>11}{"逐run>0.5":>11}'
          f'{"后半low_mass":>14}{"后半split":>11}{"跨代":>8}')
    for a in ARMS:
        cell = occ[a].mean(1)
        print(f'  {a:<6}{cell.mean():>9.4f}{int((cell > OCC_THRESH).sum()):>8}/12'
              f'{int((occ[a] > OCC_THRESH).sum()):>8}/24{lm[a].mean():>14.4f}'
              f'{ss[a].mean():>11.4f}{gen[a].mean():>8.1f}')

    print()
    print("  配对对比（先格均值 → 12 格配对符号秩；σ̂_W 只在参与对比的臂上池化）")
    for x, y, why in (("R38n", "R38p", "只动 diet_delta：关掉捕食通道"),
                      ("R50n", "R50p", "同上，在高果供给下"),
                      ("R50n", "R38n", "只动资源比：果供给翻倍（无捕食者）"),
                      ("R50p", "R38p", "同上（有捕食者）")):
        cx, cy = occ[x].mean(1), occ[y].mean(1)
        d = cx - cy
        w = np.concatenate([occ[x].std(1, ddof=1), occ[y].std(1, ddof=1)])
        noise = float(np.sqrt((w ** 2).mean()))
        p = wilcoxon(cx, cy).pvalue if np.any(d != 0) else float("nan")
        print(f'    {x}-{y}: {d.mean():+.4f}   {int((d > 0).sum())}/12 为正   p={p:.5f}   '
              f'比值 {d.mean() / max(noise, 1e-12):+.2f}   ({why})')

    print()
    print("  中性零假设（每 run 从自己帧 0 的实测直方图起跑，跑该 run 实测代数，同一 `retained`）")
    for a in ARMS:
        report([R[(a, s, r)] for s in SEEDS for r in (1, 2)], BINS, CTR, a)


if __name__ == "__main__":
    main()
