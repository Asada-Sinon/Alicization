"""用**修复后**的 `split_score` 重算 R16 §19 的四分位表。

为什么
------
`docs/multispecies_feasibility.md` §19（R16 判决）里有这张表：

    | | Q1 | Q2 | Q3 | Q4 |
    | low_mass    | 0.2858 | 0.4574 | 0.4873 | 0.5120 |
    | split_score | 0.2832 | 0.4433 | 0.4135 | 0.4826 |
    Q4−Q3 的 low_mass +0.0246（12/12、比值 +1.92）、split_score +0.0691（+3.73）

**Q3 的 `split_score` 对不上它自己的构造**。在谷见底的干净双峰上
`split_score ≡ min(low_mass, 1 − low_mass)`（`feasibility.md` §18.4 记作
「干净分裂时 ≈ 少数簇的质量」），于是预测值是

    Q1 0.2858 / Q2 0.4574 / Q3 0.4873 / Q4 0.4880

实测与预测的差：Q1 **0.0026**、Q2 **0.0141**、Q4 **0.0054**，而 **Q3 差 0.0738**
——是其余三个的 5–28 倍。而 Q4−Q3 = +0.0691 几乎**全部**来自 Q3 这个低值：
若 Q3 取预测的 0.4873，Q4−Q3 变成 **−0.0047**。

这是 `split_score` **平台失效**的签名（严格不等号让被压进相邻箱的簇找不到局部极大，
函数按构造返回 0），而那个修复的时间是：

    R16 判决      fe54d49  2026-08-05 04:50:40
    平台修复      88b8f44  2026-08-05 09:50:22   ← **晚 5 小时**

R17 那次只就地更正了 Stage 2 的一个值（`R38n` 0.4423→0.4501），**四分位表没重算**。

本脚本用**当前**的 `split_score` 重算，并把新旧并排。**不改 `split_score.py`**
（R18 在跑，它在冻结名单里）——只调用它。

跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
      explorations/20260805-r18-verdict/recheck_r16_quartiles.py
"""
import glob
import json
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, "explorations/20260804-readouts")

import numpy as np
from exp_stats import wilcoxon_p_floor
from neutral_null import gen_weights
from scipy.stats import wilcoxon
from split_score import split_score

BINS = np.linspace(0.0, 1.0, 21)
CTR = 0.5 * (BINS[:-1] + BINS[1:])
LOW = CTR < 0.35
PUBLISHED = {"low_mass": [0.2858, 0.4574, 0.4873, 0.5120],
             "split_score": [0.2832, 0.4433, 0.4135, 0.4826]}


def wmean(vals, gen, mask):
    w = gen_weights(gen, "iso") * mask
    return float((w * np.asarray(vals, float)).sum() / max(w.sum(), 1e-12))


def main():
    runs = {}
    for f in sorted(glob.glob("outputs/20260805-longrun/R38n_*.log")):
        txt = open(f).read()
        if "JSON " not in txt:
            continue
        d = json.loads(txt.split("JSON ")[1].split("\n")[0])
        if d["collapsed"]:
            continue
        tr = d["traj"]
        b = f.split("/")[-1][:-4].split("_")
        runs[(int(b[1][1:]), int(b[2][1:]))] = {
            "gen": np.array([q["generation"] for q in tr], float),
            "hist": np.array([q["hist"] for q in tr], float)}
    if not runs:
        print("找不到 R16 的 R38n run —— 不是结论，是没数据。")
        sys.exit(1)

    print("=" * 88)
    print("R16 §19 四分位表：用**修复后**的 split_score 重算")
    print("=" * 88)
    print(f"  载入 {len(runs)} 个 R38n run")

    lm_q, ss_q, plateau = {}, {}, 0
    for key, r in runs.items():
        g, H = r["gen"], r["hist"]
        gmax = g.max()
        lm_row, ss_row = [], []
        for q in range(4):
            mask = (g >= gmax * 0.25 * q) & (g < gmax * 0.25 * (q + 1) + (1e-9 if q == 3 else 0))
            lmv = [h[LOW].sum() / max(h.sum(), 1.0) for h in H]
            ssv = [split_score(h, CTR)[0] for h in H]
            lm_row.append(wmean(lmv, g, mask))
            ss_row.append(wmean(ssv, g, mask))
        lm_q[key], ss_q[key] = lm_row, ss_row
        plateau += int(sum(1 for h in H if split_score(h, CTR)[0] == 0.0))

    seeds = sorted({s for s, _ in runs})
    LM = np.array([[np.mean([lm_q[(s, rr)][q] for rr in (1, 2) if (s, rr) in lm_q])
                    for q in range(4)] for s in seeds])
    SS = np.array([[np.mean([ss_q[(s, rr)][q] for rr in (1, 2) if (s, rr) in ss_q])
                    for q in range(4)] for s in seeds])

    print(f"  格数 {len(seeds)}   split_score 恰为 0 的检查点 {plateau}"
          f"（平台失效的直接计数；修复后应该很少）")
    print()
    print(f"  {'':13s}{'Q1':>10}{'Q2':>10}{'Q3':>10}{'Q4':>10}")
    for name, M in (("low_mass", LM), ("split_score", SS)):
        pub = PUBLISHED[name]
        print(f"  {name:13s}" + "".join(f"{v:>10.4f}" for v in M.mean(0)) + "   ← 重算")
        print(f"  {'（已发表）':11s}" + "".join(f"{v:>10.4f}" for v in pub) + "   ← §19")
        print(f"  {'差':13s}" + "".join(f"{M.mean(0)[q] - pub[q]:>+10.4f}" for q in range(4)))
    print()

    print("  `split_score ≡ min(low_mass, 1 − low_mass)` 的核对（谷见底时应成立）：")
    pred = np.minimum(LM.mean(0), 1 - LM.mean(0))
    print(f"  {'预测':13s}" + "".join(f"{v:>10.4f}" for v in pred))
    print(f"  {'重算−预测':11s}" + "".join(f"{SS.mean(0)[q] - pred[q]:>+10.4f}" for q in range(4)))
    print(f"  {'已发表−预测':10s}"
          + "".join(f"{PUBLISHED['split_score'][q] - pred[q]:>+10.4f}" for q in range(4)))
    print()

    print("  Q4 − Q3（§19 报的是 low_mass +0.0246 比值+1.92、split_score +0.0691 比值+3.73）：")
    for name, M, oldd in (("low_mass", LM, 0.0246), ("split_score", SS, 0.0691)):
        d = M[:, 3] - M[:, 2]
        p = wilcoxon(M[:, 3], M[:, 2]).pvalue if np.any(d != 0) else float("nan")
        print(f"    {name:13s} Δ {d.mean():+.4f}   {int((d > 0).sum())}/{len(d)} 为正   "
              f"p={p:.5f}（地板 {wilcoxon_p_floor(len(d)):.5f}）"
              f"   §19 报的是 {oldd:+.4f}")
        print(f"      逐格 {np.round(d, 4).tolist()}")
    print()
    print("  ⚠️ 这是对**已发表数字**的重算，不是新实验。R16 的原始 run 一个字节都没动。")


if __name__ == "__main__":
    main()
