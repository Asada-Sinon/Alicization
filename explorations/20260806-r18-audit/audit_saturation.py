"""独立复核 R18 判决的第 3、5 点：饱和读数撑不撑得起「双簇撑住了」，以及减速分析。

回答什么
--------
① §22.4 说「两峰占比 1.0000、谷未见底度 0.0000」，§22.7 写成「谷**全程**全空」。
   但 §22.4 的读数只在**两个窗**（早窗 gen110–148、末 1/4）上算。
   本脚本在**全部检查点**上重算，看「全程」这个词站不站得住。
② 「两峰都找得到」这个读数**有没有可能失败**——在 R16 数据上它失败过几次？
   一个从不失败的读数不是证据。
③ §22.6 的减速表在 24/24 上复算，并补一条作者没报的：**少数簇质量**
   `min(lm,1−lm)` 的逐段轨迹——「双簇撑住」关心的是它，不是 low_mass。
④ 早期 low_mass 从 0.31 起步 ⇒ 那个 50/50 是**穿过**的一个点还是一个平衡？

读哪些文件：outputs/20260805-gen450/*.log、outputs/20260805-longrun/R38n_*.log
输出怎么读：`【判定】` 开头的是本脚本的结论。

跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
      explorations/20260806-r18-audit/audit_saturation.py
"""
import glob
import json
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, "explorations/20260804-readouts")

import numpy as np
from neutral_null import gen_weights
from scipy.stats import wilcoxon
from split_score import retained, split_score

BINS = np.linspace(0.0, 1.0, 21)
CTR = 0.5 * (BINS[:-1] + BINS[1:])
LOW = CTR < 0.35


def peaks_and_gap(h):
    """返回 (峰数, 谷见底因子)；峰数 <2 时因子为 nan。与 split_score 内部逐行同构。"""
    p = np.asarray(h, float)
    s = p.sum()
    if s <= 0:
        return 0, np.nan
    p = p / s
    p = np.convolve(np.pad(p, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    p = p / max(p.sum(), 1e-12)
    loc, i, m = [], 0, len(p)
    while i < m:
        j = i
        while j + 1 < m and p[j + 1] == p[i]:
            j += 1
        if (i == 0 or p[i] > p[i - 1]) and (j == m - 1 or p[j] > p[j + 1]):
            loc.append((i + j) // 2)
        i = j + 1
    if len(loc) < 2:
        return len(loc), np.nan
    a, b = sorted(sorted(loc, key=lambda q: -p[q])[:2])
    v = int(np.argmin(p[a:b + 1])) + a
    lo = min(p[a], p[b])
    if lo <= 0:
        return len(loc), np.nan
    return len(loc), 1.0 - p[v] / lo


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


def scan(runs, tag):
    print(f"\n【①② 全检查点扫描】{tag}   {len(runs)} run")
    for lo_gen, label in ((None, "全部检查点"), (110.0, "gen>=110（早窗起点之后）")):
        n = nfail = nnotempty = 0
        gaps = []
        worst = None
        for k, r in runs.items():
            g, H = r["gen"], r["hist"]
            m = np.ones(len(g), bool) if lo_gen is None else (g >= lo_gen)
            for gi, h in zip(g[m], H[m]):
                np_, gap = peaks_and_gap(h)
                n += 1
                if np_ < 2:
                    nfail += 1
                    continue
                gaps.append(gap)
                if gap < 0.9999:
                    nnotempty += 1
                    if worst is None or gap < worst[0]:
                        worst = (gap, k, gi)
        gaps = np.array(gaps)
        print(f"  {label:28s} n={n:6d}  找不到两峰 {nfail:5d}"
              f"  谷未见底(gap<0.9999) {nnotempty:5d} = {100*nnotempty/max(n,1):.2f}%")
        print(f"      gap 均值 {gaps.mean():.6f}  最小 {gaps.min():.6f}"
              + (f"  最差在 {worst[1]} gen={worst[2]:.0f}" if worst else ""))


def windows(runs):
    """复现 §22.4 的两窗口径，看饱和是不是窗口造成的。"""
    print("\n【① §22.4 的两窗口径（作者用的）vs 全检查点】R18")
    for tag, sel in (("早窗 gen110-148", "early"), ("末1/4", "late")):
        n = nfail = nnotempty = 0
        gaps = []
        for r in runs.values():
            g, H = r["gen"], r["hist"]
            m = (g >= 110.0) & (g <= 148.0) if sel == "early" else (g >= g.max() * 0.75)
            for h in H[m]:
                np_, gap = peaks_and_gap(h)
                n += 1
                if np_ < 2:
                    nfail += 1
                    continue
                gaps.append(gap)
                nnotempty += int(gap < 0.9999)
        gaps = np.array(gaps)
        print(f"  {tag:18s} n={n:6d}  找不到两峰 {nfail}  谷未见底 {nnotempty}"
              f"  gap 均值 {gaps.mean():.6f} 最小 {gaps.min():.6f}")


def drift(runs, nseg=6):
    print(f"\n【③④ 减速表（24/24 复算）+ 少数簇质量】按世代等分 {nseg} 段")
    segs_lm = [[] for _ in range(nseg)]
    segs_min = [[] for _ in range(nseg)]
    segs_ret = [[] for _ in range(nseg)]
    gends = []
    for k in sorted(runs):
        g, H = runs[k]["gen"], runs[k]["hist"]
        lm = np.array([h[LOW].sum() / max(h.sum(), 1.0) for h in H])
        mn = np.minimum(lm, 1 - lm)
        rt = np.array([retained(h, CTR) for h in H], float)
        gm = g.max()
        gends.append(gm)
        w = gen_weights(g, "iso")
        for i in range(nseg):
            m = (g >= gm * i / nseg) & (g < gm * (i + 1) / nseg + (1e-9 if i == nseg - 1 else 0))
            ww = w * m
            segs_lm[i].append(float((ww * lm).sum() / max(ww.sum(), 1e-12)))
            segs_min[i].append(float((ww * mn).sum() / max(ww.sum(), 1e-12)))
            segs_ret[i].append(float((ww * rt).sum() / max(ww.sum(), 1e-12)))
    A, B, C = np.array(segs_lm), np.array(segs_min), np.array(segs_ret)
    gm = float(np.mean(gends))
    print(f"  {A.shape[1]} run，平均跨 {gm:.0f} 代 [{min(gends):.0f}, {max(gends):.0f}]")
    print(f"  段中点世代      {[f'{gm*(i+0.5)/nseg:.0f}' for i in range(nseg)]}")
    for name, M in (("low_mass", A), ("min(lm,1-lm) 少数簇", B), ("retained 占空比", C)):
        print(f"  {name:22s} {[f'{v:.4f}' for v in M.mean(1)]}")
        print(f"  {'  逐段增量':20s} {[f'{v:+.4f}' for v in np.diff(M.mean(1))]}")
    # 末段增量检验（逐 run，24 个；再给格均值 12 格版本）
    for name, M in (("low_mass", A), ("min(lm,1-lm)", B)):
        last = M[-1] - M[-2]
        p = wilcoxon(M[-1], M[-2]).pvalue
        seeds = sorted({s for s, _ in runs})
        keys = sorted(runs)
        cell = np.array([np.mean([last[keys.index((s, r))] for r in (1, 2) if (s, r) in runs])
                         for s in seeds])
        pc = wilcoxon(cell).pvalue
        print(f"  末段增量 {name:14s} 逐run 均值 {last.mean():+.5f} "
              f"{int((last>0).sum())}/{len(last)} p={p:.5f}   |  "
              f"**格均值 12 格** {cell.mean():+.5f} {int((cell>0).sum())}/12 p={pc:.5f}")
    print(f"  【判定】第 1 段 low_mass = {A.mean(1)[0]:.4f} ⇒ 起点在 0.5 的"
          f"{'下' if A.mean(1)[0] < 0.5 else '上'}方，"
          f"系统是**穿过** 0.5 而不是从 0.5 出发")


def main():
    r18 = load("outputs/20260805-gen450/*.log")
    r16 = load("outputs/20260805-longrun/R38n_*.log")
    print("=" * 96)
    print("独立复核：饱和读数与减速分析")
    print("=" * 96)
    windows(r18)
    scan(r18, "R18 (20260805-gen450)")
    scan(r16, "R16 (20260805-longrun R38n)")
    drift(r18)


if __name__ == "__main__":
    main()


def last_segment_stats(nseg=6):
    """§22.6 的「末段增量」：作者按 24 个 run 做检验（21/24、p=0.00065）。
    本项目协议要求先把 r=1,2 平均成格均值再检验。两种单位都算，并给两种噪声口径。"""
    runs = load("outputs/20260805-gen450/*.log")
    keys = sorted(runs)
    inc = {}
    for k in keys:
        g, H = runs[k]["gen"], runs[k]["hist"]
        lm = np.array([h[LOW].sum() / max(h.sum(), 1.0) for h in H])
        gm = g.max()
        w = gen_weights(g, "iso")
        seg = []
        for i in range(nseg):
            m = (g >= gm * i / nseg) & (g < gm * (i + 1) / nseg + (1e-9 if i == nseg - 1 else 0))
            ww = w * m
            seg.append(float((ww * lm).sum() / max(ww.sum(), 1e-12)))
        inc[k] = seg[-1] - seg[-2]
    seeds = sorted({s for s, _ in keys})
    per_run = np.array([inc[k] for k in keys])
    cell = np.array([np.mean([inc[(s, r)] for r in (1, 2)]) for s in seeds])
    sw = float(np.sqrt(np.mean([np.std([inc[(s, r)] for r in (1, 2)], ddof=1) ** 2 for s in seeds])))
    print("\n【§22.6 末段增量：分析单位与噪声口径】")
    print(f"  逐 run（作者用的）：均值 {per_run.mean():+.5f}  "
          f"{int((per_run>0).sum())}/24  p={wilcoxon(per_run).pvalue:.5f}")
    print(f"  **格均值 12 格（协议要求）**：均值 {cell.mean():+.5f}  "
          f"{int((cell>0).sum())}/12  p={wilcoxon(cell).pvalue:.5f}")
    print(f"  格内 σ̂_W(增量) = {sw:.5f}")
    print(f"    口径 a：√2·σ̂_W/√r = {np.sqrt(2)*sw/np.sqrt(2):.5f} ⇒ 比值 "
          f"{cell.mean()/(np.sqrt(2)*sw/np.sqrt(2)):+.2f}   ← 文档补记用的")
    print(f"    口径 b：σ̂_W/√r（增量本身已是差）= {sw/np.sqrt(2):.5f} ⇒ 比值 "
          f"{cell.mean()/(sw/np.sqrt(2)):+.2f}")
    print(f"  逐格增量 {np.round(cell,4).tolist()}")


if __name__ == "__main__" and "--last" in sys.argv:
    last_segment_stats()
