"""独立复核 R18 判决的第 1、2 点：那条撤回站不站得住，恒等式的推论有没有过度。

回答什么
--------
① `split_score ≡ min(low_mass, 1−low_mass)` 这个恒等式在 R16 数据上成立吗（独立实现，
   不 import diag 脚本），并把它拆成三个必要条件分别验：谷见底、谷位在 0.35 左侧、
   平滑不跨界搬质量。
② §19 的四分位表用的是哪个口径——**用修复前的 split_score 直接复现它**。
   这是比「残差 10 倍」更硬的判据：若旧版 + 某个口径能复现全部四格，
   那么「Q3 是平台失效假值」就从推断变成了实证。
③ 两个口径下 Q4−Q3 各是多少（帧比例 vs 世代），撤回的替换数字对口径敏感吗。
④ Δ_H2+Δ_H1 的三个口径 + 一个作者没算的口径：indep 的**跨 run SD** 与噪声比。
   （问「H2 有没有独立信息」不能只问均值——均值为 0 而逐 run 抖动很大同样是信息。）

读哪些文件
----------
outputs/20260805-longrun/R38n_*.log   （R16，24 run）
outputs/20260805-gen450/*.log          （R18，24 run）
explorations/20260804-readouts/split_score.py（当前版，只 import）
git show fe54d49:...split_score.py     （修复前版本，脚本内嵌一份复刻）

输出怎么读
----------
每一节标题是它回答的问题；`【判定】` 开头的行是本脚本的结论。

跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
      explorations/20260806-r18-audit/audit_retraction.py
"""
import glob
import json
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, "explorations/20260804-readouts")

import numpy as np
from neutral_null import gen_weights
from scipy.stats import wilcoxon
from split_score import split_score as ss_new

BINS = np.linspace(0.0, 1.0, 21)
CTR = 0.5 * (BINS[:-1] + BINS[1:])
LOW = CTR < 0.35
LOW_N = int(LOW.sum())          # 分界箱下标


def ss_old(n, ctr, smooth=1):
    """**修复前**（`fe54d49`）的 split_score：局部极大用严格不等号，平台找不到峰。
    逐行复刻 git 历史里的那一版，只为复现已发表的四分位表。"""
    p = np.asarray(n, dtype=float)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    if smooth:
        k = np.ones(2 * smooth + 1) / (2 * smooth + 1)
        p = np.convolve(np.pad(p, smooth, mode="edge"), k, mode="valid")
        p = p / max(p.sum(), 1e-12)
    loc = [i for i in range(len(p))
           if (i == 0 or p[i] > p[i - 1]) and (i == len(p) - 1 or p[i] > p[i + 1])]
    if len(loc) < 2:
        return 0.0
    loc = sorted(sorted(loc, key=lambda q: -p[q])[:2])
    i, j = loc
    v = int(np.argmin(p[i:j + 1])) + i
    lo = min(p[i], p[j])
    if lo <= 0:
        return 0.0
    return float(min(p[:v].sum(), p[v:].sum()) * (1.0 - p[v] / lo))


def load(pattern):
    out = {}
    for f in sorted(glob.glob(pattern)):
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


def lm_of(h):
    return float(h[LOW].sum() / max(h.sum(), 1.0))


def wmean(vals, gen, mask):
    w = gen_weights(gen, "iso") * mask
    return float((w * np.asarray(vals, float)).sum() / max(w.sum(), 1e-12))


# ---------------------------------------------------------------- ① 恒等式
def section_identity(runs, tag):
    print(f"\n【① 恒等式 ss ≡ min(lm,1−lm)】{tag}   {len(runs)} run")
    nchk = exact = 0
    resid, gapmin, vpos, cross = [], [], [], 0
    mass_between = []          # 谷位与 0.35 分界之间的质量（恒等式的必要条件之一）
    for r in runs.values():
        for h in r["hist"]:
            p = h / max(h.sum(), 1e-12)
            lm = lm_of(h)
            s = ss_new(h, CTR)[0]
            resid.append(s - min(lm, 1 - lm))
            nchk += 1
            exact += int(abs(s - min(lm, 1 - lm)) < 1e-9)
            cross += int(lm < 0.5)
            # 拆条件：谷见底 & 谷位
            sm = np.convolve(np.pad(p, 1, mode="edge"), np.ones(3) / 3, mode="valid")
            sm = sm / max(sm.sum(), 1e-12)
            loc, i, m = [], 0, len(sm)
            while i < m:
                j = i
                while j + 1 < m and sm[j + 1] == sm[i]:
                    j += 1
                if (i == 0 or sm[i] > sm[i - 1]) and (j == m - 1 or sm[j] > sm[j + 1]):
                    loc.append((i + j) // 2)
                i = j + 1
            if len(loc) < 2:
                continue
            loc = sorted(sorted(loc, key=lambda q: -sm[q])[:2])
            a, b = loc
            v = int(np.argmin(sm[a:b + 1])) + a
            gapmin.append(1.0 - sm[v] / min(sm[a], sm[b]))
            vpos.append(CTR[v])
            mass_between.append(float(p[min(v, LOW_N):max(v, LOW_N)].sum()))
    resid = np.array(resid)
    print(f"  检查点 {nchk}   精确成立 {exact}/{nchk} = {100*exact/nchk:.1f}%"
          f"   |残差| 均值 {np.abs(resid).mean():.5f}  最大 {np.abs(resid).max():.5f}")
    print(f"  谷见底因子 1−valley/min(peak)：均值 {np.mean(gapmin):.6f}  最小 {np.min(gapmin):.6f}")
    print(f"  谷位 ctr[v]：[{np.min(vpos):.3f}, {np.max(vpos):.3f}] 均值 {np.mean(vpos):.3f}"
          f"   （low_mass 分界 0.35 = 箱 {LOW_N}）")
    print(f"  谷位与分界之间的原始质量：均值 {np.mean(mass_between):.6f}  最大 {np.max(mass_between):.6f}"
          f"   ← 非 0 就是残差的来源之一")
    print(f"  lm<0.5（尖点另一侧）的检查点 {cross}/{nchk} = {100*cross/nchk:.1f}%")
    return resid


# ---------------------------------------------------------------- ② 复现旧表
def quartiles(runs, fn, by):
    """by='gen' 用世代等分；by='frame' 用帧下标比例（analyze_r16.wmean_slice 的口径）。"""
    seeds = sorted({s for s, _ in runs})
    M = []
    for s in seeds:
        rows = []
        for rr in (1, 2):
            if (s, rr) not in runs:
                continue
            g, H = runs[(s, rr)]["gen"], runs[(s, rr)]["hist"]
            v = np.array([fn(h) for h in H], float)
            row = []
            for q in range(4):
                if by == "gen":
                    gmax = g.max()
                    m = (g >= gmax * 0.25 * q) & (g < gmax * 0.25 * (q + 1) + (1e-9 if q == 3 else 0))
                    row.append(wmean(v, g, m))
                else:
                    a, b = int(len(v) * 0.25 * q), int(len(v) * 0.25 * (q + 1))
                    w = gen_weights(g, "iso")[a:b]
                    row.append(float((w * v[a:b]).sum() / max(w.sum(), 1e-12)))
            rows.append(row)
        M.append(np.mean(rows, axis=0))
    return np.array(M)


def show(name, M, pub=None):
    mu = M.mean(0)
    line = f"  {name:26s}" + "".join(f"{v:>10.4f}" for v in mu)
    if pub is not None:
        line += "   |已发表差 " + " ".join(f"{mu[q]-pub[q]:+.4f}" for q in range(4))
    print(line)
    return mu


def q43(M, name):
    d = M[:, 3] - M[:, 2]
    p = wilcoxon(M[:, 3], M[:, 2]).pvalue if np.any(d != 0) else float("nan")
    print(f"    {name:34s} Δ {d.mean():+.4f}   {int((d>0).sum())}/{len(d)} 为正   p={p:.5f}")
    return d.mean(), p


# ---------------------------------------------------------------- ④ Δ 层面
def section_delta(runs):
    print("\n【④ Δ_H2 + Δ_H1：H2 还剩多少独立自由度】R18 24 run")
    d1, d2, keys = [], [], []
    for k in sorted(runs):
        g, H = runs[k]["gen"], runs[k]["hist"]
        late, early = g >= g.max() * 0.75, (g >= 110.0) & (g <= 148.0)
        lm = np.array([lm_of(h) for h in H])
        ss = np.array([ss_new(h, CTR)[0] for h in H])
        d1.append(lm[late].mean() - lm[early].mean())
        d2.append(ss[late].mean() - ss[early].mean())
        keys.append(k)
    d1, d2 = np.array(d1), np.array(d2)
    indep = d1 + d2
    # 格均值（先把 r=1,2 平均），这才是本项目的检验单位
    seeds = sorted({s for s, _ in keys})
    cellI = np.array([np.mean([indep[keys.index((s, r))] for r in (1, 2) if (s, r) in keys])
                      for s in seeds])
    cellD1 = np.array([np.mean([d1[keys.index((s, r))] for r in (1, 2) if (s, r) in keys])
                       for s in seeds])
    sw = float(np.sqrt(np.mean([np.std([indep[keys.index((s, r))] for r in (1, 2)], ddof=1) ** 2
                                for s in seeds])))
    noise = np.sqrt(2) * sw / np.sqrt(2)
    print(f"  Δ_H1 均值 {d1.mean():+.5f}   Δ_H2 均值 {d2.mean():+.5f}")
    print(f"  indep = Δ_H1+Δ_H2  逐 run {np.round(indep,4).tolist()}")
    print(f"  精确为 0 的 run {int((np.abs(indep)<1e-9).sum())}/{len(indep)}"
          f"   均值 {indep.mean():+.5f}   |最大| {np.abs(indep).max():.5f}")
    print(f"  格内 σ̂_W(indep) = {sw:.5f}   配对差噪声 = {noise:.5f}")
    print(f"    口径① |最大| ÷ H1效应   = {np.abs(indep).max()/abs(d1.mean()):.3f}")
    print(f"    口径② 均值   ÷ H1效应   = {abs(indep.mean())/abs(d1.mean()):.3f}")
    print(f"    口径③ 均值   ÷ 配对差噪声= {abs(indep.mean())/noise:.3f}   ← 作者用的")
    # 作者没算的：indep 的跨格 SD 与格内噪声比 —— 「逐 run 抖动」也是信息
    sd_between = float(cellI.std(ddof=1))
    print(f"    口径④ indep 的跨格 SD    = {sd_between:.5f}  ÷ 格内 σ̂_W {sw:.5f}"
          f" = {sd_between/max(sw,1e-12):.2f}   ← **本脚本新增**")
    print(f"    口径⑤ 跨格 SD ÷ H1效应   = {sd_between/abs(d1.mean()):.3f}")
    # indep 是不是显著非零（配对符号秩 vs 0）
    p = wilcoxon(cellI).pvalue
    print(f"    indep 的 12 格符号秩 vs 0：均值 {cellI.mean():+.5f}  "
          f"{int((cellI>0).sum())}/12 为正  p={p:.5f}")
    # indep 与 Δ_H1 的相关：若强相关，说明残余不是随机的
    r = float(np.corrcoef(cellD1, cellI)[0, 1])
    print(f"    corr(格均值 Δ_H1, 格均值 indep) = {r:+.3f}")
    return indep


def main():
    r16 = load("outputs/20260805-longrun/R38n_*.log")
    r18 = load("outputs/20260805-gen450/*.log")
    print("=" * 96)
    print("独立复核：R18 判决的撤回与恒等式推论")
    print("=" * 96)
    print(f"  R16(20260805-longrun R38n) {len(r16)} run    R18(20260805-gen450) {len(r18)} run")

    section_identity(r16, "R16 / R38n")
    section_identity(r18, "R18")

    PUB_LM = [0.2858, 0.4574, 0.4873, 0.5120]
    PUB_SS = [0.2832, 0.4433, 0.4135, 0.4826]
    print("\n【② 复现 §19 已发表的四分位表：哪个口径 + 哪个版本的 split_score】")
    print(f"  {'':26s}{'Q1':>10}{'Q2':>10}{'Q3':>10}{'Q4':>10}")
    print(f"  {'已发表 low_mass':24s}" + "".join(f"{v:>10.4f}" for v in PUB_LM))
    print(f"  {'已发表 split_score':23s}" + "".join(f"{v:>10.4f}" for v in PUB_SS))
    print("  " + "-" * 60)
    res = {}
    for by in ("frame", "gen"):
        res[("lm", by)] = quartiles(r16, lm_of, by)
        res[("ssN", by)] = quartiles(r16, lambda h: ss_new(h, CTR)[0], by)
        res[("ssO", by)] = quartiles(r16, lambda h: ss_old(h, CTR), by)
        show(f"low_mass [{by}]", res[("lm", by)], PUB_LM)
        show(f"split_score 旧版 [{by}]", res[("ssO", by)], PUB_SS)
        show(f"split_score 新版 [{by}]", res[("ssN", by)], PUB_SS)
        pred = np.minimum(res[("lm", by)].mean(0), 1 - res[("lm", by)].mean(0))
        print(f"  {'  恒等式预测 min(lm,1−lm)':24s}" + "".join(f"{v:>10.4f}" for v in pred))
        print("  " + "-" * 60)

    print("\n【③ Q4 − Q3：撤回的替换数字对口径敏感吗】")
    for by in ("frame", "gen"):
        print(f"  口径 = {by}")
        q43(res[("lm", by)], "low_mass")
        q43(res[("ssO", by)], "split_score 旧版（已发表用的）")
        q43(res[("ssN", by)], "split_score 新版（撤回后的替换值）")

    section_delta(r18)


if __name__ == "__main__":
    main()
