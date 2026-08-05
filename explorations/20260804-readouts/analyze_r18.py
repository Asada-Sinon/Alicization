"""R18 判决用的读数表。判据 `docs/multispecies_program.md` §21。

**本脚本只把数摆出来，判决由 `result-analyst` 出。**

三件必须做对的
--------------
1. **按世代切片，不是按帧比例。** §21.4 的 H1 是「末四分之一 vs **gen 110–148 的窗**」，
   后者是 R16 的末四分之一所覆盖的那一段——**只有按世代切，两轮才可比**。
   各 run 的世代钟不一样（R16 实测 133–164 代），按帧切会切到不同的世代年龄上。
2. **统计一律走 `scripts/exp_stats.py`。** R16 那次我自己实现并多加卡方上界，
   造出「比值恰好 1.0035、刀刃上」的假象（共享口径 +1.38）。
3. **`retained_occ` 不进主判据**（`docs/conventions.md` §11）：它在这个臂上已知饱和
   （R16 实测 24/24 顶在 1.0）。只作 H3 的计数输入，并与未饱和读数一并报。

跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
      explorations/20260804-readouts/analyze_r18.py
"""
import glob
import json
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, "explorations/20260804-readouts")

import numpy as np
from exp_stats import bootstrap_ci, wilcoxon_p_floor
from neutral_null import DEFAULT_NS, OCC_THRESH, gen_weights, genes_from_hist, sim_run
from scipy.stats import wilcoxon
from split_score import retained, split_score

BINS = np.linspace(0.0, 1.0, 21)
CTR = 0.5 * (BINS[:-1] + BINS[1:])
LOW = CTR < 0.35
SEEDS = list(range(12))
REPS = (1, 2)
RUN_DIR = "outputs/20260805-gen450"
R16_WINDOW = (110.0, 148.0)      # R16 的末四分之一覆盖的世代段（§21.4）


def gen_mask(gen, lo, hi):
    """**按世代**取窗。各 run 的世代钟不同，按帧比例切会切到不同的世代年龄上。"""
    return (gen >= lo) & (gen <= hi)


def wmean(vals, gen, mask, mode="iso"):
    v = np.asarray(vals, float)
    w = gen_weights(gen, mode) * mask
    return float((w * v).sum() / max(w.sum(), 1e-12))


def load():
    R, bad = {}, []
    for f in sorted(glob.glob(f"{RUN_DIR}/*.log")):
        txt = open(f).read()
        if "JSON " not in txt:
            continue
        d = json.loads(txt.split("JSON ")[1].split("\n")[0])
        tr = d["traj"]
        h = np.array([q["hist"] for q in tr], float)
        assert np.allclose(h.sum(1), [q["n_herb"] for q in tr], rtol=1e-6, atol=1e-3), f
        b = f.split("/")[-1][:-4].split("_")
        rec = {"name": b[0] + "_" + b[1] + "_" + b[2], "hist": h,
               "gen": np.array([q["generation"] for q in tr], float),
               "collapsed": bool(d["collapsed"])}
        if rec["collapsed"]:
            bad.append(rec)          # §21.5：崩溃 run 分开报，不池化
        else:
            R[(int(b[1][1:]), int(b[2][1:]))] = rec
    return R, bad


def cells(R, fn):
    return np.array([np.mean([fn(R[(s, r)]) for r in REPS if (s, r) in R]) for s in SEEDS])


def within(R, fn):
    return np.array([np.std([fn(R[(s, r)]) for r in REPS if (s, r) in R], ddof=1)
                     if all((s, r) in R for r in REPS) else np.nan for s in SEEDS])


def report(late, early, sd, name, why):
    """§21.4 的口径：先格均值 → 12 格配对符号秩 → 比值 `√2·σ̂_W/√r`。

    **不套卡方上界**——§21.5 指定走 `exp_stats` 的口径，R16 那次多加一个上界
    就把 +1.38 变成了刀刃上的 +1.00。
    """
    m = np.isfinite(late) & np.isfinite(early)
    d = (late - early)[m]
    w = sd[np.isfinite(sd)]
    sw = float(np.sqrt((w ** 2).mean())) if len(w) else float("nan")
    noise = np.sqrt(2) * sw / np.sqrt(len(REPS))
    p = wilcoxon(late[m], early[m]).pvalue if np.any(d != 0) else float("nan")
    lo, hi = bootstrap_ci(d)
    sat = (not np.isfinite(noise)) or noise < 1e-9
    ratio = "**无定义（读数饱和）**" if sat else f"{d.mean() / noise:+.2f}"
    flag = "" if sat or abs(d.mean()) >= noise else "   ** <1，判 undecidable **"
    print(f"  {name}: 晚 {late[m].mean():.4f} vs 早 {early[m].mean():.4f}   "
          f"Δ {d.mean():+.4f}   {int((d > 0).sum())}/{len(d)} 为正")
    print(f"    p={p:.5f}（n={len(d)} 地板 {wilcoxon_p_floor(len(d)):.5f}）   "
          f"95%CI [{lo:+.4f}, {hi:+.4f}]{'  不含 0' if lo * hi > 0 else '  **含 0**'}   "
          f"噪声 {noise:.4f}   比值 {ratio}{flag}")
    print(f"    逐格 Δ = {np.round(d, 4).tolist()}")
    print(f"    逐格内 SD = {np.round(sd, 4).tolist()}   ({why})")


def main():
    R, bad = load()
    print("=" * 96)
    print("R18 读数表（判据 `multispecies_program.md` §21）")
    print("=" * 96)
    print(f"  载入 {len(R)} run，崩溃 {len(bad)}"
          + (f"：{[b['name'] for b in bad]}" if bad else ""))
    if len(R) < 24:
        print("  ⚠️ **未跑满 24，以下只是管路空跑，不是判决**")

    spans = np.array([r["gen"][-1] - r["gen"][0] for r in R.values()])
    if len(spans):
        print(f"  **实测跨代数** {spans.mean():.0f} [{spans.min():.0f}, {spans.max():.0f}]"
              f"   （§21.2：这是读数不是假设，判决按实际达到的代数写，不许写「450 代」）")
        print(f"  参照：R16 同配置 100k 步 = 148 代 [133, 164]")

    if len(R) < 4:
        print("  样本太少，停在这里。")
        return

    def slices(rec):
        g = rec["gen"]
        gmax = g.max()
        late = gen_mask(g, gmax * 0.75, gmax)                 # 末四分之一（按世代）
        early = gen_mask(g, *R16_WINDOW)                      # R16 的窗
        return late, early

    def mk(fn):
        def late(rec):
            l, _ = slices(rec)
            return wmean([fn(h) for h in rec["hist"]], rec["gen"], l)

        def early(rec):
            _, e = slices(rec)
            return wmean([fn(h) for h in rec["hist"]], rec["gen"], e)
        return late, early

    lm = lambda h: h[LOW].sum() / max(h.sum(), 1.0)
    ss = lambda h: split_score(h, CTR)[0]
    rt = lambda h: float(retained(h, CTR))

    print()
    print("  H1 主判据：`low_mass` 的「末四分之一 − R16 窗口(gen 110–148)」")
    a, b = mk(lm)
    report(cells(R, a), cells(R, b), within(R, a), "low_mass",
           "§21.3 跑前算死：σ̂_W=0.0093、预测效应÷MDE=18.6")
    print()
    print("  H2：`split_score` 同口径")
    print("  ⚠️ **§21.4b：H2 不是 H1 之外的独立证据。** 谷见底的干净双峰上")
    print("     `split_score ≡ min(low_mass, 1 − low_mass)`（实测精确成立 94.4%），")
    print("     所以下面这条在 `low_mass > 0.5` 时必然与 H1 反号。**它测的是")
    print("     「离 50/50 有多远」，不是「分裂有多强」**——按对称性读数解释，")
    print("     不许和 H1 并排当两条互相矛盾的发现。诊断见")
    print("     `explorations/20260805-r18-verdict/diag_h2_degenerate.py`。")
    a2, b2 = mk(ss)
    report(cells(R, a2), cells(R, b2), within(R, a2), "split_score",
           "§21.3 跑前算死：σ̂_W=0.0072；⚠️ 这个 σ̂_W 比 low_mass 的 0.0093 小 23%，"
           "是 min 在尖点附近压缩了方差，不是它更精确 —— 比值会虚高")
    print()
    print("  次读数（不进主判据，§21.4 + conventions §11：已知饱和）")
    a3, b3 = mk(rt)
    report(cells(R, a3), cells(R, b3), within(R, a3), "retained 占空比",
           "**这一条已知会饱和，只作对照**")

    print()
    print("  H3：末四分之一的 `retained` 计数 vs 模拟中性零假设")
    runs = list(R.values())
    T = sum(1 for r in runs
            if wmean([rt(h) for h in r["hist"]], r["gen"], slices(r)[0]) > OCC_THRESH)
    print(f"    实测 **{T}/{len(runs)}**")
    rng = np.random.default_rng(20260805)
    REPS_NULL = 200
    for N in DEFAULT_NS:
        cnt = []
        for _ in range(REPS_NULL):
            c = 0
            for r in runs:
                g0 = genes_from_hist(r["hist"][0], BINS, rng)
                span = int(max(r["gen"][-1] - r["gen"][0], 5))
                hs = sim_run(g0, span, len(r["hist"]), N, BINS, rng)
                if wmean([rt(h) for h in hs], r["gen"], slices(r)[0]) > OCC_THRESH:
                    c += 1
            cnt.append(c)
        cnt = np.array(cnt)
        k = int((cnt >= T).sum())
        print(f"      N={N:<5} 零均值 {cnt.mean():>5.2f}  最大 {cnt.max():>3.0f}  "
              f"p={'<' if k == 0 else '='}{(1.0 + k) / (REPS_NULL + 1.0):.4f}"
              f"{'（= 分辨率下限）' if k == 0 else ''}")
    print(f"      （reps={REPS_NULL} ⇒ p 的分辨率下限 {1 / (REPS_NULL + 1):.4f}）")

    print()
    print("  [判决由 result-analyst 出，本脚本只把数摆出来。]")


if __name__ == "__main__":
    main()
