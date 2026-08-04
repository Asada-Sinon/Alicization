"""R16 判决用的读数表。判据 `docs/multispecies_program.md` §19（+ §19.4b 勘误）。

**本脚本只把数摆出来，判决由 `result-analyst` 出。**

三件按 §19.4b 必须做对的事
--------------------------
1. **判据是「末四分之一」，而 `trajectory.py` 自动出的 `retained_occ` 是「后半程」。**
   两者不是一回事，所以这里从 `traj` 现算任意切片（`occ_slice`），
   自动字段只作次读数一并报。
2. **`retained_occ` 会撞 1.0 的天花板** ⇒ H2 的 Δ 被压住、跨 run 方差被人为压小
   ⇒ **比值会被高估**。表里直接标出有多少个 run 顶在 1.0。
3. **H2 没有预声明「它长了」** ⇒ 若 Δ 显著为正，只能记成「方向为正，未预声明」。

⚠️ **不修改 `neutral_null.py` / `split_score.py` / `trajectory.py`**——R16 跑的时候
它们在冻结清单上（`HANDOFF.md`），本脚本只 import。

跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
      explorations/20260804-readouts/analyze_r16.py
"""
import glob
import json
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, "explorations/20260804-readouts")

import numpy as np
from neutral_null import DEFAULT_NS, OCC_THRESH, gen_weights, genes_from_hist, sim_run
from scipy.stats import chi2, wilcoxon
from split_score import retained, split_score

BINS = np.linspace(0.0, 1.0, 21)
CTR = 0.5 * (BINS[:-1] + BINS[1:])
LOW = CTR < 0.35
ARMS = ["R38n", "R38p"]
SEEDS = list(range(12))
REPS = (1, 2)
RUN_DIR = "outputs/20260805-longrun"


def occ_slice(hists, gen, lo=0.75, hi=1.0, mode="iso"):
    """任意区间的世代加权 `retained` 占空比。**默认末四分之一，这才是 §19 的判据。**"""
    ret = np.array([retained(h, CTR) for h in hists], float)
    w = gen_weights(gen, mode)
    a, b = int(len(ret) * lo), int(len(ret) * hi)
    ww = w[a:b]
    return float((ww * ret[a:b]).sum() / max(ww.sum(), 1e-12))


def wmean_slice(vals, gen, lo=0.75, hi=1.0, mode="iso"):
    v = np.asarray(vals, float)
    w = gen_weights(gen, mode)
    a, b = int(len(v) * lo), int(len(v) * hi)
    return float((w[a:b] * v[a:b]).sum() / max(w[a:b].sum(), 1e-12))


def load():
    R, bad = {}, []
    for f in sorted(glob.glob(f"{RUN_DIR}/*.log")):
        txt = open(f).read()
        name = f.split("/")[-1][:-4]
        if "JSON " not in txt:
            continue
        d = json.loads(txt.split("JSON ")[1].split("\n")[0])
        tr = d["traj"]
        h = np.array([q["hist"] for q in tr], float)
        assert np.allclose(h.sum(1), [q["n_herb"] for q in tr], rtol=1e-6, atol=1e-3), f
        b = name.split("_")
        rec = {"name": name, "hist": h, "collapsed": bool(d["collapsed"]),
               "gen": np.array([q["generation"] for q in tr], float),
               "carn": np.array([q["carnivore_frac"] for q in tr], float),
               "auto_occ": d.get("retained_occ")}
        if rec["collapsed"]:
            bad.append(rec)          # **崩溃 run 分开报，不池化**（§19.5）
            continue
        R[(b[0], int(b[1][1:]), int(b[2][1:]))] = rec
    return R, bad


def cell(R, arm, fn):
    """格均值：先把 r=1,2 平均成 12 个格，再做检验（`s×r` 个 run 不是独立样本）。"""
    out = []
    for s in SEEDS:
        vs = [fn(R[(arm, s, r)]) for r in REPS if (arm, s, r) in R]
        out.append(np.mean(vs) if vs else np.nan)
    return np.array(out)


def within_sd(R, arm, fn):
    return np.array([np.std([fn(R[(arm, s, r)]) for r in REPS if (arm, s, r) in R], ddof=1)
                     if sum((arm, s, r) in R for r in REPS) == 2 else np.nan
                     for s in SEEDS])


def paired(x, y, sd_pool, label, why):
    """先格均值 → 12 格配对符号秩 → 比值用 `√2·σ̂_W/√r`（σ̂_W 取 90% 卡方上界）。"""
    m = np.isfinite(x) & np.isfinite(y)
    d = (x - y)[m]
    if len(d) < 2:
        print(f"    {label}: 样本不足（{len(d)} 格）")
        return
    w = sd_pool[np.isfinite(sd_pool)]
    sw = float(np.sqrt((w ** 2).mean())) if len(w) else float("nan")
    ub = sw * np.sqrt(len(w) / chi2.ppf(0.10, len(w))) if len(w) else float("nan")
    noise = np.sqrt(2) * ub / np.sqrt(len(REPS))
    p = wilcoxon(x[m], y[m]).pvalue if np.any(d != 0) else float("nan")
    # **读数饱和时比值是无定义的，不是「很大」。** 末1/4 占空比顶在 1.0 会让格内 SD
    # 恰好为 0 ⇒ 噪声除零 ⇒ 打印出 3.4e11 这种数。报一个那样的数比不报还糟：
    # 它看起来像「效应极强」，实际是「读数已经没有分辨率了」（§19.4b①）。
    if not np.isfinite(noise) or noise < 1e-9:
        ratio = "  比值 **无定义（读数饱和，σ̂_W ≈ 0）**"
    else:
        ratio = (f"  比值 {d.mean() / noise:+.2f}"
                 + ("   ** <1，判 undecidable **" if abs(d.mean()) < noise else ""))
    print(f"    {label}: {d.mean():+.4f}   {int((d > 0).sum())}/{len(d)} 为正   "
          f"p={p:.5f}{ratio}")
    print(f"      逐格 = {np.round(d, 4).tolist()}   ({why})")


def main():
    R, bad = load()
    print("=" * 96)
    print("R16 读数表（判据 `multispecies_program.md` §19 + §19.4b 勘误）")
    print("=" * 96)
    n_by = {a: sum(1 for k in R if k[0] == a) for a in ARMS}
    print(f"  载入 {len(R)} run（{', '.join(f'{a}={n_by[a]}' for a in ARMS)}），"
          f"崩溃 {len(bad)}" + (f"：{[b['name'] for b in bad]}" if bad else ""))
    if len(R) < 48:
        print(f"  ⚠️ **未跑满 48，以下只是管路空跑，不是判决**")

    occ4 = lambda d: occ_slice(d["hist"], d["gen"], 0.75, 1.0)      # 末四分之一
    occ1 = lambda d: occ_slice(d["hist"], d["gen"], 0.0, 0.25)      # 首四分之一
    lm = lambda d: wmean_slice([h[LOW].sum() / max(h.sum(), 1.0) for h in d["hist"]], d["gen"])
    ss = lambda d: wmean_slice([split_score(h, CTR)[0] for h in d["hist"]], d["gen"])
    gsp = lambda d: float(d["gen"][-1] - d["gen"][0])

    print()
    print(f'  {"臂":<7}{"n":>4}{"末1/4占空比":>12}{"顶在1.0":>9}{"首1/4":>8}'
          f'{"末1/4 low_mass":>15}{"末1/4 split":>12}{"跨代":>8}')
    for a in ARMS:
        if not n_by[a]:
            continue
        c4, c1 = cell(R, a, occ4), cell(R, a, occ1)
        runs = [R[k] for k in R if k[0] == a]
        ceil = sum(1 for r in runs if occ4(r) >= 0.9999)
        print(f'  {a:<7}{n_by[a]:>4}{np.nanmean(c4):>12.4f}{ceil:>6}/{len(runs)}'
              f'{np.nanmean(c1):>8.4f}{np.nanmean(cell(R, a, lm)):>15.4f}'
              f'{np.nanmean(cell(R, a, ss)):>12.4f}{np.mean([gsp(r) for r in runs]):>8.0f}')

    if n_by["R38n"]:
        print()
        print("  H2（稳定 vs 衰减）：R38n 的末1/4 − 首1/4，σ̂_W 只在 R38n 上自估")
        paired(cell(R, "R38n", occ4), cell(R, "R38n", occ1),
               within_sd(R, "R38n", occ4), "占空比 Δ",
               "⚠️ §19.4b①：末1/4 顶在 1.0 ⇒ 比值被高估；②方向为正未预声明")

    if n_by["R38n"] and n_by["R38p"]:
        print()
        print("  H3（捕食通道，长时程）：R38n − R38p 的末1/4 占空比")
        sdn, sdp = within_sd(R, "R38n", occ4), within_sd(R, "R38p", occ4)
        rn, rp = np.nanmean(sdn), np.nanmean(sdp)
        print(f"    两臂格内 SD：R38n {rn:.4f} / R38p {rp:.4f}   比 {max(rn,rp)/max(min(rn,rp),1e-9):.1f}×")
        both = np.concatenate([sdn, sdp])
        if max(rn, rp) / max(min(rn, rp), 1e-9) >= 10:
            print("    ⚠️ 差一个量级 ⇒ **不池化**，三种口径都报（`MEMORY.md [LEARN:stats]`）")
            # **「只在参与对比的臂上池化」这条规矩会和天花板撞上**：`R38n` 的末1/4 占空比
            # 顶在 1.0 ⇒ 它的 σ̂_W 恰好是 0 ⇒ 口径 A 无定义，口径 B 被那个 0 拉低。
            # **唯一有分辨率的噪声估计来自 `R38p`**，所以必须报口径 C。
            # 这一条是空跑时才发现的：规矩本身没错，但它假定了两臂都有方差。
            paired(cell(R, "R38n", occ4), cell(R, "R38p", occ4), sdn, "口径A：只用 R38n", "饱和臂，通常无定义")
            paired(cell(R, "R38n", occ4), cell(R, "R38p", occ4), both, "口径B：两臂池化", "被 R38n 的 0 拉低")
            paired(cell(R, "R38n", occ4), cell(R, "R38p", occ4), sdp, "口径C：只用 R38p", "**唯一有分辨率的那个**")
        else:
            paired(cell(R, "R38n", occ4), cell(R, "R38p", occ4), both, "占空比差",
                   "Stage 2 在 42k 步上是 +0.6140（12/12、比值 +1.91），预测同号")

    print()
    print("  护栏：逐 run 的 min(carnivore_frac)（**不做事后筛**，§19.5）")
    for a in ARMS:
        ks = sorted([k for k in R if k[0] == a])
        if not ks:
            continue
        mins = [float(np.nanmin(R[k]["carn"])) for k in ks]
        lost = sum(1 for m in mins if m <= 0.0)
        if a.endswith("n"):
            # **`diet_delta=1.5` 由构造无捕食者**：这里的 0 不是「丢失事件」。
            # 不标这一句，`15/15 曾触 0` 会被读成异常——那正是本 session 记过的
            # 「压缩转述层丢限定词」那类错。
            print(f"    {a}: min 恒为 {max(mins):.4f}（**由构造无捕食，`diet_delta=1.5`，"
                  f"不是丢失事件**）")
        else:
            print(f"    {a}: **曾触 0 的 run {lost}/{len(ks)}**（这一臂才是真的丢失事件，"
                  f"R13 有 3/24）；min 范围 [{min(mins):.4f}, {max(mins):.4f}]")

    print()
    print("  H1（主判据）：末1/4 占空比 > 0.5 的 run 数 vs 模拟中性零假设")
    REPS_NULL = 200
    for a in ARMS:
        ks = sorted([k for k in R if k[0] == a])
        if not ks:
            continue
        runs = [R[k] for k in ks]
        T = sum(1 for r in runs if occ4(r) > OCC_THRESH)
        print(f"    {a}: 实测 **{T}/{len(runs)}**")
        rng = np.random.default_rng(20260805)
        for N in DEFAULT_NS:
            cnt = []
            for _ in range(REPS_NULL):
                c = 0
                for r in runs:
                    g0 = genes_from_hist(r["hist"][0], BINS, rng)
                    span = int(max(r["gen"][-1] - r["gen"][0], 5))
                    hs = sim_run(g0, span, len(r["hist"]), N, BINS, rng)
                    if occ_slice(hs, r["gen"], 0.75, 1.0) > OCC_THRESH:
                        c += 1
                cnt.append(c)
            cnt = np.array(cnt)
            k = int((cnt >= T).sum())
            p = (1.0 + k) / (REPS_NULL + 1.0)
            print(f"      N={N:<5} 零均值 {cnt.mean():>5.2f}  最大 {cnt.max():>3.0f}  "
                  f"p={'<' if k == 0 else '='}{p:.4f}"
                  f"{'（= 分辨率下限）' if k == 0 else ''}")
        print(f"      （reps={REPS_NULL} ⇒ p 的分辨率下限 {1/(REPS_NULL+1):.4f}）")

    print()
    print("  [判决由 result-analyst 出，本脚本只把数摆出来。]")


if __name__ == "__main__":
    main()
