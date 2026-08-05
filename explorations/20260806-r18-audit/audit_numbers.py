"""独立复核 R18 判决里被引用最多的几个数，全部用脚本算，不手算。

回答什么
--------
① §19 勘误块的「其余三格偏离 0.0026/0.0050/0.0063，Q3 偏离 0.0696，是它们的 10 倍」
   —— 在**每一个单一口径**下这三个数分别是多少，10 倍站不站得住。
② §22.2 的 H1 主判据用 `scripts/exp_stats` 的口径复算（Δ、CI、比值）。
③ Δ_H2 有多少比例被 −Δ_H1 解释掉。
④ §22.4 的两个窗覆盖了全程的百分之多少。
⑤ 少数簇质量 min(lm,1−lm) 的峰值→末段相对跌幅。

读哪些文件：outputs/20260805-longrun/R38n_*.log、outputs/20260805-gen450/*.log
跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
      explorations/20260806-r18-audit/audit_numbers.py
"""
import glob
import json
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, "explorations/20260804-readouts")

import numpy as np
from neutral_null import gen_weights
from scipy.stats import wilcoxon
from split_score import split_score

BINS = np.linspace(0.0, 1.0, 21)
CTR = 0.5 * (BINS[:-1] + BINS[1:])
LOW = CTR < 0.35


def load(pat):
    out = {}
    for f in sorted(glob.glob(pat)):
        txt = open(f).read()
        if "JSON " not in txt:
            continue
        d = json.loads(txt.split("JSON ")[1].split("\n")[0])
        if d["collapsed"]:
            continue
        tr = d["traj"]
        b = f.split("/")[-1][:-4].split("_")
        out[(int(b[1][1:]), int(b[2][1:]))] = {
            "gen": np.array([q["generation"] for q in tr], float),
            "hist": np.array([q["hist"] for q in tr], float)}
    return out


PUB_LM = np.array([0.2858, 0.4574, 0.4873, 0.5120])
PUB_SS = np.array([0.2832, 0.4433, 0.4135, 0.4826])
# 来自 audit_retraction.py 的实测重算（同一份数据、同一份代码）
GEN_LM = np.array([0.2494, 0.4483, 0.4831, 0.5111])

print("=" * 92)
print("【① §19 勘误块「10 倍」的核对：每个口径下的偏离】")
print("=" * 92)
pred_frame = np.minimum(PUB_LM, 1 - PUB_LM)
pred_gen = np.minimum(GEN_LM, 1 - GEN_LM)
for name, pred in (("frame 口径（= 已发表表自己的口径）", pred_frame),
                   ("gen 口径（= recheck 脚本的口径）", pred_gen)):
    dev = np.abs(PUB_SS - pred)
    others = np.sort(np.delete(dev, 2))[::-1]
    print(f"  {name}")
    print(f"    预测 min(lm,1−lm) = {np.round(pred,4).tolist()}")
    print(f"    |已发表 ss − 预测| = {np.round(dev,4).tolist()}")
    print(f"    Q3 偏离 {dev[2]:.4f}；其余三格最大 {others[0]:.4f}"
          f"  ⇒ 倍数 = **{dev[2]/others[0]:.2f}×**")
print("  文档写的是「0.0026 / 0.0050 / 0.0063」和「Q3 偏离 0.0696、10 倍」")
print(f"  frame 三格 = {np.round(np.delete(np.abs(PUB_SS-pred_frame),2),4).tolist()}"
      f"   gen 三格 = {np.round(np.delete(np.abs(PUB_SS-pred_gen),2),4).tolist()}")

r18 = load("outputs/20260805-gen450/*.log")
print()
print("=" * 92)
print("【② §22.2 H1 主判据复算（走 scripts/exp_stats 的噪声口径）】")
print("=" * 92)
seeds = sorted({s for s, _ in r18})
early_c, late_c, wsd = [], [], []
d2_runs = []
for s in seeds:
    e, l, dd = [], [], []
    for rr in (1, 2):
        g, H = r18[(s, rr)]["gen"], r18[(s, rr)]["hist"]
        lm = np.array([h[LOW].sum() / max(h.sum(), 1.0) for h in H])
        ss = np.array([split_score(h, CTR)[0] for h in H])
        w = gen_weights(g, "iso")
        em, lmk = (g >= 110) & (g <= 148), g >= g.max() * 0.75
        ev = float((w * em * lm).sum() / (w * em).sum())
        lv = float((w * lmk * lm).sum() / (w * lmk).sum())
        e.append(ev); l.append(lv); dd.append(lv - ev)
        d2_runs.append(float((w * lmk * ss).sum() / (w * lmk).sum()
                             - (w * em * ss).sum() / (w * em).sum()))
    early_c.append(np.mean(e)); late_c.append(np.mean(l))
    wsd.append(np.std(dd, ddof=1))
early_c, late_c = np.array(early_c), np.array(late_c)
d = late_c - early_c
sw = float(np.sqrt(np.mean(np.array(wsd) ** 2)))
noise = np.sqrt(2) * sw / np.sqrt(2)
rng = np.random.default_rng(0)
bs = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(20000)])
print(f"  早窗 {early_c.mean():.4f}   末1/4 {late_c.mean():.4f}   Δ {d.mean():+.4f}")
print(f"  {int((d>0).sum())}/12 为正   p={wilcoxon(late_c, early_c).pvalue:.5f}"
      f"   95%CI [{np.percentile(bs,2.5):+.4f}, {np.percentile(bs,97.5):+.4f}]")
print(f"  σ̂_W(Δ)={sw:.4f}  配对差噪声={noise:.4f}  **比值 {d.mean()/noise:+.2f}**")
print(f"  逐格 Δ = {np.round(d,4).tolist()}   [{d.min():+.4f}, {d.max():+.4f}]")

print()
print("=" * 92)
print("【③ Δ_H2 被 −Δ_H1 解释掉的比例】")
print("=" * 92)
d2 = np.array(d2_runs)
d1_runs = []
for s in seeds:
    for rr in (1, 2):
        g, H = r18[(s, rr)]["gen"], r18[(s, rr)]["hist"]
        lm = np.array([h[LOW].sum() / max(h.sum(), 1.0) for h in H])
        w = gen_weights(g, "iso")
        em, lmk = (g >= 110) & (g <= 148), g >= g.max() * 0.75
        d1_runs.append(float((w * lmk * lm).sum() / (w * lmk).sum()
                             - (w * em * lm).sum() / (w * em).sum()))
d1 = np.array(d1_runs)
ind = d1 + d2
print(f"  |Δ_H2| 均值 {np.abs(d2).mean():.5f}   |残差| 均值 {np.abs(ind).mean():.5f}")
print(f"  ⇒ Δ_H2 里被 −Δ_H1 解释掉的比例 = **{100*(1-np.abs(ind).mean()/np.abs(d2).mean()):.1f}%**")

print()
print("=" * 92)
print("【④ §22.4 两个窗覆盖了全程多少 / ⑤ 少数簇质量跌幅】")
print("=" * 92)
ncov = ntot = 0
gspan = []
for r in r18.values():
    g = r["gen"]
    m = ((g >= 110) & (g <= 148)) | (g >= g.max() * 0.75)
    ncov += int(m.sum()); ntot += len(g)
    gspan.append((float(g.min()), float(g.max())))
print(f"  两窗覆盖检查点 {ncov}/{ntot} = **{100*ncov/ntot:.1f}%**"
      f"   （§22.7 的「全程」指的是 100%）")
print(f"  世代范围：起 {np.mean([a for a,_ in gspan]):.1f}  止 {np.mean([b for _,b in gspan]):.1f}")
peak, last = 0.4758, 0.4158
print(f"  少数簇 min(lm,1−lm)：峰值段 {peak:.4f} → 末段 {last:.4f}"
      f"   绝对 {last-peak:+.4f}   相对 **{100*(last-peak)/peak:+.1f}%**")
