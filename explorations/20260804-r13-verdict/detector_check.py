"""留存判据 `split_score>0` 到底在检测什么？

§16 把「留存」定义为 `split_score>0`，并把它当作「果专精少数簇仍在」的代理。
`split_score` 按构造是**位置无关**的（`split_score.py` 的设计目标），
所以它对「主峰被劈成两半」和「低端还有一个果专精簇」给同一个阳性。
本脚本用 `split_at`（谷底位置）与 `low_mass` 把两者拆开。

读：outputs/20260804-traj/*.log
重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260804-r13-verdict/detector_check.py
"""
import glob
import json
import sys

sys.path.insert(0, "explorations/20260804-readouts")
import numpy as np
from scipy.stats import wilcoxon
from split_score import split_score

CTR = np.linspace(0.15, 0.85, 15)
CTR = 0.5 * (CTR[:-1] + CTR[1:])

runs = []
for f in sorted(glob.glob("outputs/20260804-traj/*.log")):
    for ln in open(f):
        if ln.startswith("JSON "):
            d = json.loads(ln[5:]); d["_file"] = f.split("/")[-1]; runs.append(d)

cells = [(r["_file"], i, cp) for r in runs for i, cp in enumerate(r["checkpoints"])]
pos = [(f, i, cp["split_score"], cp["split_at"], cp["low_mass"], cp["high_mass"])
       for f, i, cp in cells if cp["split_score"] > 0]
print(f"全部 (run,cp) 格 {len(cells)} 个；split_score>0 的 {len(pos)} 个\n")
print("split_score>0 的每一格：谷底位置 split_at 与 low_mass")
print(f'  {"file":<16} {"cp":>3} {"split_score":>12} {"split_at":>9} {"low_mass":>9} {"high_mass":>10}')
for f, i, ss, sa, lm, hm in sorted(pos, key=lambda x: x[3]):
    flag = "  <-- 谷在主峰内部，不是果专精簇" if sa >= 0.55 else ""
    print(f"  {f:<16} {i:>3} {ss:>12.4f} {sa:>9.3f} {lm:>9.4f} {hm:>10.4f}{flag}")

sa = np.array([p[3] for p in pos]); lm = np.array([p[4] for p in pos])
print(f"\n  split_at 分布：中位 {np.median(sa):.3f}  范围 [{sa.min():.3f}, {sa.max():.3f}]")
print(f"  其中 split_at >= 0.55（谷落在草专精一侧）：{int((sa >= 0.55).sum())}/{len(sa)}")
print(f"  其中 low_mass < 0.01（阳性但几乎没有果专精质量）：{int((lm < 0.01).sum())}/{len(lm)}")

print("\n反向：low_mass 很大但 split_score=0 的格（有质量、没有谷 ⇒ 连续肩部）")
neg = [(f, i, cp["low_mass"], cp["dip_ratio"]) for f, i, cp in cells
       if cp["split_score"] == 0 and cp["low_mass"] > 0.05]
for f, i, l, dr in sorted(neg, key=lambda x: -x[2]):
    print(f"  {f:<16} cp{i}  low_mass {l:.4f}  dip_ratio {dr:.3f}")
print(f"  共 {len(neg)} 格")

print("\n一致性：把留存判据换成 low_mass>0.05 会得到什么留存曲线？")
print(f'  {"cp":>3} {"ss>0(入组15)":>14} {"lm>0.05(入组15)":>17} {"lm>0.02(入组15)":>17}')
ent = [r for r in runs if r["checkpoints"][0]["split_score"] > 0]
for i in range(5):
    a = sum(1 for r in ent if r["checkpoints"][i]["split_score"] > 0)
    b = sum(1 for r in ent if r["checkpoints"][i]["low_mass"] > 0.05)
    c = sum(1 for r in ent if r["checkpoints"][i]["low_mass"] > 0.02)
    print(f"  {i:>3} {a:>10}/15 {b:>13}/15 {c:>13}/15")

print("\n连续留存（5 个 cp 全部 split_score>0）的 run 数：")
cont = [r["_file"] for r in ent if all(cp["split_score"] > 0 for cp in r["checkpoints"])]
print(f"  {len(cont)}/15   {cont}")
cont2 = [r["_file"] for r in ent if all(cp["low_mass"] > 0.02 for cp in r["checkpoints"])]
print(f"  改用 low_mass>0.02 连续留存：{len(cont2)}/15   {cont2}")

print("\n低簇质量 low_mass 的逐 run 首末配对（cp0 → cp4，入组 15 个）")
a = np.array([r["checkpoints"][0]["low_mass"] for r in ent])
b = np.array([r["checkpoints"][4]["low_mass"] for r in ent])
print(f"  cp0 均值 {a.mean():.4f}  cp4 均值 {b.mean():.4f}  差 {(b-a).mean():+.4f}  "
      f"{int((b < a).sum())}/15 下降  符号秩 p={wilcoxon(a, b).pvalue:.5f}")
print(f"  逐 run 差：{np.round(b - a, 4).tolist()}")
