"""R18-B 判决用的读数表。判据 `docs/multispecies_program.md` §23（先写后跑）。

**本脚本只把数摆出来，判决由人/复核出。**

与 `analyze_r18.py` 的四处不同——每一处都来自 R18 判决被 fable5 复核抓到的错
----------------------------------------------------------------------------
1. **H1 是等价性检验，不是「有没有差」。** §23.4：末段增量的 95%CI 完全落在
   `[−δ, +δ]` 内 ⇒ **停了**；完全落在 `[+δ, ∞)` ⇒ **没停且推翻几何衰减**；
   否则 undecidable。**「没检出」不等于「停了」**，这是本项目第一次用等价性检验。
2. **按固定段长（~63 代，与 R18 一致）切，不按固定段数。** 若照抄 R18 的 6 段，
   681 代会切成 ~114 代一段，噪声与效应都与 R18 不可比。
3. **H3 在【全部检查点】上扫，不是只在两个窗上。** R18 的教训：那两个窗只覆盖
   41.7% 的检查点，于是「窗内全饱和」被写成了「全程都在」，而全量上有 275 个
   检查点无双峰、1248 个谷未见底（**全在 gen 63 之前**）。
4. **H3b 少数簇质量 `min(lm, 1−lm)` 是一等读数。** `gap≡1` 时它就是分裂强度本身，
   也是 §21.1 直接要的量。R18 上它从峰值 0.4803 跌到 0.4158（−13.4%），
   而判决初版只报了多数侧上升——**那让结论听起来比实际乐观**。

**分析单位一律是格均值（12 格），不是 run。** R18 的 `drift_deceleration.py`
就是直接对 24 个 run 做 wilcoxon，把 9/12 报成了 21/24。

跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
      explorations/20260806-r18b/analyze_r18b.py [--skip-h4]
"""
import glob
import json
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, "explorations/20260804-readouts")
sys.path.insert(0, "explorations/20260805-r18-verdict")

import numpy as np
from diag_h2_degenerate import parts
from exp_stats import bootstrap_ci, wilcoxon_p_floor
from neutral_null import DEFAULT_NS, OCC_THRESH, gen_weights, genes_from_hist, sim_run
from scipy.stats import wilcoxon
from split_score import retained, split_score

BINS = np.linspace(0.0, 1.0, 21)
CTR = 0.5 * (BINS[:-1] + BINS[1:])
LOW = CTR < 0.35
SEEDS = list(range(12))
REPS = (1, 2)
RUN_DIR = "outputs/20260806-gen700"
R16_WINDOW = (110.0, 148.0)
SEG_GEN = 62.8          # R18 的段长，本轮按它定段数（§23.4）
H2_BAND = (0.585, 0.610)


def wmean(vals, gen, mask):
    w = gen_weights(gen, "iso") * mask
    s = w.sum()
    return float((w * np.asarray(vals, float)).sum() / s) if s > 1e-9 else np.nan


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
        rec = {"name": "_".join(b[:3]), "hist": h,
               "gen": np.array([q["generation"] for q in tr], float),
               "collapsed": bool(d["collapsed"])}
        if rec["collapsed"]:
            bad.append(rec)                 # §21.5：崩溃 run 分开报，不池化
        else:
            R[(int(b[1][1:]), int(b[2][1:]))] = rec
    return R, bad


def cellwise(R, fn):
    """先 run→格均值（协议口径），并回传重复内 SD 用于自估噪声。"""
    per = {k: fn(v) for k, v in R.items()}
    mu, sd = [], []
    for s in SEEDS:
        vs = [per[(s, r)] for r in REPS if (s, r) in per]
        mu.append(np.mean(vs) if vs else np.nan)
        sd.append(np.std(vs, ddof=1) if len(vs) == len(REPS) else np.nan)
    return np.array(mu), np.array(sd)


def equiv(delta_cells, sd_cells, delta, name, meaning):
    """§23.4 的等价性判定。**「没检出」不等于「停了」，看的是 CI 落在哪。**"""
    m = np.isfinite(delta_cells)
    d = delta_cells[m]
    lo, hi = bootstrap_ci(d)
    p = wilcoxon(d).pvalue if np.any(d != 0) else float("nan")
    w = sd_cells[np.isfinite(sd_cells)]
    sw = float(np.sqrt((w ** 2).mean())) if len(w) else float("nan")
    noise = np.sqrt(2) * sw / np.sqrt(len(REPS))
    ratio = d.mean() / noise if np.isfinite(noise) and noise > 1e-9 else float("nan")
    if lo >= -delta and hi <= delta:
        verdict = f"**停了（等价性成立：CI 完全落在 ±{delta:.4f} 内）**"
    elif lo >= delta:
        verdict = f"**没停，且推翻几何衰减（CI 完全在 +{delta:.4f} 之上）**"
    elif hi <= -delta:
        verdict = f"**反向且超出 ±{delta:.4f}**"
    else:
        verdict = "**undecidable（CI 跨过阈值边界）**"
    print(f"  {name}: Δ {d.mean():+.5f}   {int((d > 0).sum())}/{len(d)} 为正   "
          f"p={p:.5f}（地板 {wilcoxon_p_floor(len(d)):.5f}）")
    half = (hi - lo) / 2.0
    print(f"    95%CI [{lo:+.5f}, {hi:+.5f}]   δ={delta:.5f}"
          f"（本轮自估 σ̂_W={sw:.5f}）   比值 {ratio:+.2f}"
          f"{'  ** <1，功效不足 **' if abs(ratio) < 1 else ''}")
    print(f"    ⇒ {verdict}   （{meaning}）")
    # **这个检验有多容易通过**——δ 是用同一批数据自估的噪声，不是独立的实际等价界。
    # 只要 |效应| < δ − CI半宽 就会判「停了」，所以必须把这个余量摆出来，
    # 否则「等价性成立」读起来像「测到了零」，而它只是「小于本实验的分辨率」。
    print(f"    ⚠️ 该判定的宽严：CI 半宽 {half:.5f} = {half / delta:.2f}·δ ⇒ "
          f"**|效应| < {max(delta - half, 0):.5f} 就会判「停了」**")
    print(f"       ⇒ 「停了」的含义是 **「残余漂移小于本实验的分辨率」，不是「绝对为零」**")
    print(f"    逐格 Δ = {np.round(delta_cells, 5).tolist()}")
    return d.mean(), (lo, hi), noise


def main():
    skip_h4 = "--skip-h4" in sys.argv
    R, bad = load()
    print("=" * 96)
    print("R18-B 读数表（判据 `multispecies_program.md` §23）")
    print("=" * 96)
    print(f"  载入 {len(R)} run，崩溃 {len(bad)}"
          + (f"：{[b['name'] for b in bad]}" if bad else ""))
    if len(R) < 24:
        print(f"  ⚠️ **未跑满 24（现 {len(R)}），以下只是管路空跑，不是判决**")
    if len(R) < 4:
        print("  样本太少，停在这里。")
        return

    spans = np.array([r["gen"][-1] - r["gen"][0] for r in R.values()])
    span = float(spans.mean())
    nseg = max(int(round(span / SEG_GEN)), 3)
    print(f"  **实测跨代数** {span:.0f} [{spans.min():.0f}, {spans.max():.0f}]"
          f"   （§21.2：判决按实测写，**不许写「700 代」**）")
    print(f"  参照：R18 同臂 350k 步 = 375 代 [345, 431]")
    print(f"  **按固定段长切**：{nseg} 段 × {span / nseg:.1f} 代/段"
          f"（R18 是 6 段 × {SEG_GEN} 代——段长可比，段数不可比）")

    # ---- 分段轨迹（协议口径：先 run→格） -------------------------------------
    def seg_of(rec, i, fn):
        g, gm = rec["gen"], rec["gen"].max()
        m = (g >= gm * i / nseg) & (g < gm * (i + 1) / nseg + (1e-9 if i == nseg - 1 else 0))
        return wmean([fn(h) for h in rec["hist"]], g, m)

    lm = lambda h: h[LOW].sum() / max(h.sum(), 1.0)
    minority = lambda h: min(lm(h), 1.0 - lm(h))

    LMc = np.array([cellwise(R, lambda r, i=i: seg_of(r, i, lm))[0] for i in range(nseg)])
    MNc = np.array([cellwise(R, lambda r, i=i: seg_of(r, i, minority))[0] for i in range(nseg)])
    print()
    print(f"  `low_mass` 逐段（格均值）  {[f'{v:.4f}' for v in np.nanmean(LMc, 1)]}")
    print(f"  **少数簇 min(lm,1−lm)**    {[f'{v:.4f}' for v in np.nanmean(MNc, 1)]}")
    dseq = np.diff(np.nanmean(LMc, 1))
    print(f"  逐段增量                  {[f'{v:+.4f}' for v in dseq]}")
    rr = [dseq[i + 1] / dseq[i] for i in range(len(dseq) - 1) if abs(dseq[i]) > 1e-9]
    print(f"  增量之比                  {[f'{v:.3f}' for v in rr]}   ← 全 <1 才叫减速")

    # ---- H1：末段增量的等价性检验 --------------------------------------------
    print()
    print("  **H1（主）：漂移停了没有——等价性检验**（§23.4）")
    dl, sdl = cellwise(R, lambda r: seg_of(r, nseg - 1, lm) - seg_of(r, nseg - 2, lm))
    w = sdl[np.isfinite(sdl)]
    sw = float(np.sqrt((w ** 2).mean())) if len(w) else float("nan")
    delta = np.sqrt(2) * sw / np.sqrt(len(REPS))          # §23.3：δ 用本轮自估
    print(f"    δ = 本轮自估配对差噪声 = **{delta:.5f}**（§23.3 跑前预估 0.0095）")
    equiv(dl, sdl, delta, "low_mass 末段增量", "≈0 ⇒ 漂移停了")

    # ---- H1b：少数簇质量的末段增量（§23.4 H3b） -------------------------------
    print()
    print("  **H3b：少数簇质量还在缩吗**（`gap≡1` 时它就是分裂强度本身）")
    peak = float(np.nanmax(np.nanmean(MNc, 1)))
    now = float(np.nanmean(MNc, 1)[-1])
    print(f"    峰值 {peak:.4f}（第 {int(np.nanargmax(np.nanmean(MNc, 1))) + 1} 段）"
          f"→ 末段 {now:.4f}   **绝对 {now - peak:+.4f}、相对 {100 * (now - peak) / peak:+.1f}%**"
          f"   （R18: 0.4803→0.4158、−13.4%）")
    dm, sdm = cellwise(R, lambda r: seg_of(r, nseg - 1, minority) - seg_of(r, nseg - 2, minority))
    equiv(dm, sdm, delta, "少数簇末段增量", "≈0 ⇒ 不再缩；显著为负 ⇒ 仍在缩")

    # ---- H2：几何外推验证（预先声明功效不足） --------------------------------
    print()
    print("  **H2：几何外推验证**（§23.4 **已预先声明功效不足**，落在带内只能说「不矛盾」）")
    late, sdlate = cellwise(R, lambda r: wmean([lm(h) for h in r["hist"]], r["gen"],
                                               r["gen"] >= r["gen"].max() * 0.75))
    lo, hi = bootstrap_ci(late[np.isfinite(late)])
    inb = H2_BAND[0] <= np.nanmean(late) <= H2_BAND[1]
    print(f"    末四分之一 `low_mass` = {np.nanmean(late):.4f}   95%CI [{lo:.4f}, {hi:.4f}]")
    print(f"    预注册带 {H2_BAND} ⇒ **{'落在带内（不矛盾，不算证实）' if inb else '落在带外'}**")

    # ---- H3：全检查点扫描（口径已按复核改） ----------------------------------
    print()
    print("  **H3：双簇还在不在——【全部检查点】扫描**（§23.4，口径已按 R18 的教训改）")
    tot = nopeak = nogap = 0
    gmin, last_np, last_ng = 9.0, 0.0, 0.0
    late_fail = []
    for rec in R.values():
        for h, gg in zip(rec["hist"], rec["gen"]):
            tot += 1
            pr = parts(h, CTR)
            if pr is None or "gap" not in pr:
                nopeak += 1
                last_np = max(last_np, gg)
                if gg > 100:
                    late_fail.append(("无双峰", rec["name"], float(gg)))
                continue
            gmin = min(gmin, pr["gap"])
            if pr["gap"] < 0.9999:
                nogap += 1
                last_ng = max(last_ng, gg)
                if gg > 100:
                    late_fail.append(("谷未见底", rec["name"], float(gg)))
    print(f"    全部检查点 {tot}")
    print(f"      找不到两峰 **{nopeak}**（{100 * nopeak / tot:.2f}%，最晚 gen {last_np:.0f}）")
    print(f"      谷未见底   **{nogap}**（{100 * nogap / tot:.2f}%，最晚 gen {last_ng:.0f}）")
    print(f"      `gap` 最小 **{gmin:.6f}**   两者都干净 {tot - nopeak - nogap}")
    print(f"      参照 R18：275 / 1248 / gap_min 0.002372，最晚 gen 13 / 63")
    if late_fail:
        # §23.4b：**晚期失败必须先分「暂态」还是「并合」**，否则一次 1 代的扰动
        # 会被读成「双簇开始并合」。判据：涉及几个 run、连续多少个检查点、是否恢复。
        by_run = {}
        for kind, nm, gg in late_fail:
            by_run.setdefault(nm, []).append(gg)
        print(f"    ⚠️ **gen>100 之后有 {len(late_fail)} 次失败**，涉及 {len(by_run)} 个 run")
        for nm, gs in sorted(by_run.items()):
            gs = sorted(gs)
            span_g = gs[-1] - gs[0]
            rec = next(r for r in R.values() if r["name"] == nm)
            after = [(g_, parts(h, CTR)) for h, g_ in zip(rec["hist"], rec["gen"]) if g_ > gs[-1]]
            recov = [g_ for g_, pr in after
                     if pr and "gap" in pr and pr["gap"] >= 0.9999]
            tail_bad = [g_ for g_, pr in after
                        if not pr or "gap" not in pr or pr["gap"] < 0.9999]
            print(f"      {nm}: {len(gs)} 次，gen {gs[0]:.1f}–{gs[-1]:.1f}（跨 {span_g:.1f} 代）")
            if recov and not tail_bad:
                print(f"        ⇒ **暂态**：此后 {len(recov)} 个检查点全部恢复 `gap=1`，"
                      f"到 gen {rec['gen'].max():.0f} 再无失败")
            elif tail_bad:
                print(f"        ⇒ ⚠️⚠️ **不是暂态**：其后仍有 {len(tail_bad)} 次失败"
                      f"（最晚 gen {max(tail_bad):.0f}）——**这才是「双簇开始并合」，"
                      f"§23.4 要求单独成节报**")
            else:
                print("        ⇒ 失败发生在轨迹末尾，无法判断是否恢复——**按未恢复报**")
    else:
        print(f"    ⇒ 失败**全部**落在 gen {max(last_np, last_ng):.0f} 之前 ⇒ "
              f"**双簇形成于头 {max(last_np, last_ng):.0f} 代，此后 {span - max(last_np, last_ng):.0f} 代未并合**")

    # ---- H4：中性对照 ---------------------------------------------------------
    print()
    print("  **H4：末四分之一的 `retained` 计数 vs 模拟中性零假设**")
    if skip_h4:
        print("    （--skip-h4，跳过）")
        return
    runs = list(R.values())
    rt = lambda h: float(retained(h, CTR))
    T = sum(1 for r in runs
            if wmean([rt(h) for h in r["hist"]], r["gen"], r["gen"] >= r["gen"].max() * 0.75)
            > OCC_THRESH)
    print(f"    实测 **{T}/{len(runs)}**")
    rng = np.random.default_rng(20260806)
    REPS_NULL = 200
    for N in DEFAULT_NS:
        cnt = []
        for _ in range(REPS_NULL):
            c = 0
            for r in runs:
                g0 = genes_from_hist(r["hist"][0], BINS, rng)
                sp = int(max(r["gen"][-1] - r["gen"][0], 5))
                hs = sim_run(g0, sp, len(r["hist"]), N, BINS, rng)
                if wmean([rt(h) for h in hs], r["gen"],
                         r["gen"] >= r["gen"].max() * 0.75) > OCC_THRESH:
                    c += 1
            cnt.append(c)
        cnt = np.array(cnt)
        k = int((cnt >= T).sum())
        print(f"      N={N:<5} 零均值 {cnt.mean():>5.2f}  最大 {cnt.max():>3.0f}  "
              f"p={'<' if k == 0 else '='}{(1.0 + k) / (REPS_NULL + 1.0):.4f}"
              f"{'（= 分辨率下限）' if k == 0 else ''}")
    print(f"      （reps={REPS_NULL} ⇒ p 的分辨率下限 {1 / (REPS_NULL + 1):.4f}）")


if __name__ == "__main__":
    main()
