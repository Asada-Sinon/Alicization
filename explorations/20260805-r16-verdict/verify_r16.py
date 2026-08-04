"""R16 判决前的独立复核 + 判据要求但读数表没出的几个数。

回答什么
--------
1. 复核 `analyze_r16.py` 的表头数字（占空比、H2、H3 三口径），并给出更多位数
   ——H3 口径 C 的比值印成 +1.00，恰在 §19.4「比值<1 判 undecidable」的刀刃上。
2. 单变量归因：两臂的 `overrides` 是否只差 `diet_delta`。
3. §19.4b③ 要求「自动字段作次读数一并报」——读数表载入了 `retained_occ` 却没打印。
4. §19.4 次判据里的 `mean_pref` 与「检查点的逐箱摄入」、§19.5 的 `in_window`：读数表未出。
5. §19.5 要求「报每个 run 的 min(carnivore_frac)」——读数表只报了范围；这里逐 run 报，
   并把 4 个丢失捕食者的 run **分开报**（不做事后筛）。
6. 【判据未预声明，用户 Q4】世代钟不等长对 H3 的威胁：把 R38p 截到与 R38n 相同的
   **世代年龄**（gen ∈ [111,148]）再比一次。
7. 口径敏感性：末四分之一按帧号 vs 按世代权重质量；`gen_weights` 的 iso/clip/unif。

读哪些文件：outputs/20260805-longrun/*.log（48 个，`JSON ` 行）。
输出怎么读：见每一节的标题；`[预注册]` = §19 写死的口径，`[非预注册]` = 事后补的旁证，
不得用来推翻 `[预注册]` 的闸。

跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
      explorations/20260805-r16-verdict/verify_r16.py
"""
import json
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, "explorations/20260804-readouts")
sys.path.insert(0, "explorations/20260805-r16-verdict")

import numpy as np
from analyze_r16 import ARMS, BINS, CTR, LOW, REPS, SEEDS, cell, load, occ_slice, within_sd
from neutral_null import OCC_THRESH, gen_weights
from scipy.stats import chi2, wilcoxon
from split_score import retained

RNG = np.random.default_rng(20260805)


def boot_ci(d, reps=20000):
    d = np.asarray(d, float)
    bs = [np.mean(RNG.choice(d, len(d), replace=True)) for _ in range(reps)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def noise_of(sds, r=2):
    """项目口径：√2·σ̂_W/√r，σ̂_W 取 90% 卡方上界（与 analyze_r16.paired 完全一致）。"""
    w = np.asarray(sds, float)
    w = w[np.isfinite(w)]
    if not len(w):
        return float("nan")
    sw = float(np.sqrt((w ** 2).mean()))
    ub = sw * np.sqrt(len(w) / chi2.ppf(0.10, len(w)))
    return float(np.sqrt(2) * ub / np.sqrt(r))


def occ_gen_window(rec, glo, ghi, mode="iso"):
    """[非预注册] 在**绝对世代区间** [glo,ghi] 上算世代加权占空比；无帧落入则 nan。"""
    ret = np.array([retained(h, CTR) for h in rec["hist"]], float)
    w = gen_weights(rec["gen"], mode)
    m = (rec["gen"] >= glo) & (rec["gen"] <= ghi)
    if not m.any() or w[m].sum() <= 0:
        return float("nan")
    return float((w[m] * ret[m]).sum() / w[m].sum())


def occ_by_genmass(rec, lo=0.75, hi=1.0, mode="iso"):
    """[非预注册] 末四分之一改按**世代权重累积质量**切（判据按帧号切，两者不同）。"""
    ret = np.array([retained(h, CTR) for h in rec["hist"]], float)
    w = gen_weights(rec["gen"], mode)
    c = np.cumsum(w) / max(w.sum(), 1e-12)
    m = (c > lo) & (c <= hi + 1e-12)
    if not m.any() or w[m].sum() <= 0:
        return float("nan")
    return float((w[m] * ret[m]).sum() / w[m].sum())


def main():
    R, bad = load()
    print("=" * 96)
    print("R16 独立复核（judge: result-analyst）— 判据 §19 + §19.4b")
    print("=" * 96)
    print(f"载入 {len(R)} run，崩溃 {len(bad)}")

    # ---------- 1. 单变量归因 ----------
    print("\n[1] 单变量归因：两臂 overrides 逐键比对")
    ov = {}
    for k, rec in R.items():
        txt = open(f"outputs/20260805-longrun/{rec['name']}.log").read()
        d = json.loads(txt.split("JSON ")[1].split("\n")[0])
        ov.setdefault(k[0], set()).add(json.dumps(d["overrides"], sort_keys=True))
    for a in ARMS:
        print(f"  {a}: 臂内 overrides 唯一值个数 = {len(ov[a])}")
    on, op = json.loads(list(ov["R38n"])[0]), json.loads(list(ov["R38p"])[0])
    diff = sorted(set(on) | set(op))
    diff = [k for k in diff if on.get(k) != op.get(k)]
    print(f"  两臂差异键 = {diff}   （R38n={[on.get(k) for k in diff]}, R38p={[op.get(k) for k in diff]}）")
    print(f"  世界覆盖项数（含 diet_delta）= {len(on)}；键 = {sorted(on)}")
    steps = {a: {json.loads(open(f'outputs/20260805-longrun/{R[k]["name"]}.log').read()
                            .split('JSON ')[1].split('\n')[0])["steps"]
                 for k in R if k[0] == a} for a in ARMS}
    print(f"  steps: {steps}")

    occ4 = lambda d: occ_slice(d["hist"], d["gen"], 0.75, 1.0)
    occ1 = lambda d: occ_slice(d["hist"], d["gen"], 0.0, 0.25)

    # ---------- 2. 复核表头 + 更多位数 ----------
    print("\n[2] [预注册] 末1/4 / 首1/4 占空比（帧号切片，mode=iso）")
    c4 = {a: cell(R, a, occ4) for a in ARMS}
    c1 = {a: cell(R, a, occ1) for a in ARMS}
    for a in ARMS:
        runs4 = np.array([occ4(R[k]) for k in sorted(R) if k[0] == a])
        print(f"  {a}: 末1/4 格均 {c4[a].mean():.6f}  逐run min={runs4.min():.6f} max={runs4.max():.6f} "
              f"=1.0 的 run {int((runs4 >= 0.9999).sum())}/{len(runs4)}；首1/4 格均 {c1[a].mean():.6f}")
        print(f"    逐格末1/4 = {np.round(c4[a], 4).tolist()}")
        print(f"    逐格首1/4 = {np.round(c1[a], 4).tolist()}")

    # ---------- 3. H2 ----------
    print("\n[3] H2 = R38n 末1/4 − 首1/4")
    d2 = c4["R38n"] - c1["R38n"]
    p2 = wilcoxon(c4["R38n"], c1["R38n"]).pvalue
    lo, hi = boot_ci(d2)
    sd_occ4_n = within_sd(R, "R38n", occ4)
    sd_occ1_n = within_sd(R, "R38n", occ1)
    print(f"  Δ={d2.mean():+.6f}  {int((d2>0).sum())}/{len(d2)} 为正  p={p2:.6f}  95%CI=[{lo:+.4f},{hi:+.4f}]")
    print(f"  [预注册] σ̂_W(末1/4, R38n) 逐格 = {np.round(sd_occ4_n,6).tolist()}  ⇒ 噪声={noise_of(sd_occ4_n):.3e}")
    print(f"  天花板检查：末1/4 全为 1.0 ⇒ Δ 恒等于 (1 − 首1/4)。max|Δ−(1−首1/4)| = "
          f"{np.abs(d2 - (1 - c1['R38n'])).max():.2e}")
    # [非预注册] 直接对差值估格内噪声（不是 √2 口径，因为两个窗口来自同一个 run）
    dsd = np.array([np.std([occ4(R[("R38n", s, r)]) - occ1(R[("R38n", s, r)]) for r in REPS], ddof=1)
                    for s in SEEDS])
    sw = float(np.sqrt((dsd ** 2).mean()))
    ub = sw * np.sqrt(len(dsd) / chi2.ppf(0.10, len(dsd)))
    print(f"  [非预注册·旁证] 直接对 Δ 估格内 SD：RMS={sw:.4f} 90%上界={ub:.4f} "
          f"⇒ 格均噪声={ub/np.sqrt(2):.4f}  比值={d2.mean()/(ub/np.sqrt(2)):+.2f}")
    print(f"  [非预注册·旁证] σ̂_W(首1/4) 噪声={noise_of(sd_occ1_n):.4f} ⇒ 比值={d2.mean()/noise_of(sd_occ1_n):+.2f}")

    # ---------- 4. H3 ----------
    print("\n[4] H3 = R38n − R38p 的末1/4")
    d3 = c4["R38n"] - c4["R38p"]
    p3 = wilcoxon(c4["R38n"], c4["R38p"]).pvalue
    lo3, hi3 = boot_ci(d3)
    sdn, sdp = within_sd(R, "R38n", occ4), within_sd(R, "R38p", occ4)
    print(f"  Δ={d3.mean():+.6f}  {int((d3>0).sum())}/{len(d3)} 为正  p={p3:.6f}  95%CI=[{lo3:+.4f},{hi3:+.4f}]")
    print(f"  逐格 = {np.round(d3,4).tolist()}")
    for lab, s in [("A 只用 R38n", sdn), ("B 两臂池化", np.concatenate([sdn, sdp])), ("C 只用 R38p", sdp)]:
        nz = noise_of(s)
        print(f"    口径{lab}: 噪声={nz:.6f}  比值={d3.mean()/nz if nz>1e-9 else float('nan'):+.4f}"
              + ("  ← 唯一有分辨率" if lab.startswith("C") else ""))
    print(f"  R38p 格内 SD 逐格 = {np.round(sdp,4).tolist()}   RMS={np.sqrt((sdp**2).mean()):.4f}")
    # 项目口径的「比值」= 效应 ÷ **格内重复**噪声（问「一次复跑能不能翻掉它」）。
    # 另一个不同的量是**格间**信噪：效应 ÷ 12 个配对差的标准误。两者都报，别混。
    for lab, d in [("H3", d3), ("H2", d2)]:
        sd_between = float(np.std(d, ddof=1))
        print(f"  [非预注册·旁证] {lab} 格间：12 个配对差 SD={sd_between:.4f} "
              f"SE={sd_between/np.sqrt(len(d)):.4f} ⇒ 效应/SE={d.mean()/(sd_between/np.sqrt(len(d))):+.2f}")
    # §19.5 说「统计走 scripts/exp_stats.py」，但 analyze_r16.paired 自己实现了一版，
    # 且**多加了一道 90% 卡方上界**（exp_stats.pair_noise 没有）⇒ 读数表的比值比共享实现更严。
    from exp_stats import mde_sign_consistent
    n_es = float(np.sqrt((sdp[np.isfinite(sdp)] ** 2).mean())) * np.sqrt(2) / np.sqrt(len(REPS))
    print(f"  [口径核对] 若按 `exp_stats.pair_noise`（无卡方上界）：噪声={n_es:.4f} "
          f"⇒ 口径C 比值={d3.mean()/n_es:+.4f}；读数表用的是更严的卡方上界版")
    print(f"  [参考] mde_sign_consistent(σ_d={n_es:.4f}, s=12) = {mde_sign_consistent(n_es, 12):.4f}")
    print(f"  [非预注册·旁证] 口径C 的保守性：它把**零方差的 R38n** 也按 R38p 的格内 SD 记账"
          f"（√2 因子）⇒ 若只算 R38p 一侧的格均噪声 = {noise_of(sdp)/np.sqrt(2):.4f}，"
          f"比值 = {d3.mean()/(noise_of(sdp)/np.sqrt(2)):+.2f}")

    # ---------- 5. §19.4b③ 自动字段 + 次判据 ----------
    print("\n[5] 次读数（读数表未打印的）")
    for a in ARMS:
        auto = np.array([R[k]["auto_occ"] for k in sorted(R) if k[0] == a], float)
        firsth = []
        for k in sorted(R):
            if k[0] != a:
                continue
            txt = open(f"outputs/20260805-longrun/{R[k]['name']}.log").read()
            d = json.loads(txt.split("JSON ")[1].split("\n")[0])
            firsth.append(d["retained_occ_first"])
        print(f"  {a}: 自动字段 retained_occ(后半程) 均值={auto.mean():.4f}  >0.5 的 run={int((auto>0.5).sum())}/{len(auto)}"
              f"   retained_occ_first(前半程) 均值={np.mean(firsth):.4f}")
    print("  检查点（4 个）逐臂均值：t / mean_pref / in_window / low_mass / frugivory_frac / 逐箱摄入峰位")
    for a in ARMS:
        cps = [json.loads(open(f"outputs/20260805-longrun/{R[k]['name']}.log").read()
                          .split("JSON ")[1].split("\n")[0])["checkpoints"] for k in sorted(R) if k[0] == a]
        for i in range(4):
            mp = np.mean([c[i]["mean_pref"] for c in cps])
            iw = np.mean([c[i]["in_window"] for c in cps])
            lm = np.mean([c[i]["low_mass"] for c in cps])
            fr = np.mean([c[i]["frugivory_frac"] for c in cps])
            rv = np.mean([float(c[i]["readout_valid"]) for c in cps])
            def _arr(v):
                # 逐箱读数会出现 None（该箱无样本）——**必须转 nan，不能当 0**，
                # 否则「空箱」会被读成「摄入为零」。
                return np.array([np.nan if x is None else float(x) for x in v], float)
            ik = np.array([_arr(c[i]["intake"]) for c in cps])
            intake = np.nanmean(ik, axis=0)
            dlog = np.nanmean(np.array([_arr(c[i]["dlog_per_bin"]) for c in cps]), axis=0)
            # `trajectory.py:169` 的 `intake` 是**每箱的人均摄入**（样本<30 记 None），
            # 不是总量 ⇒ 只能比人均，不能算「份额」。
            lowi = float(np.nanmean(intake[LOW]))
            print(f"    {a} cp{i} t={cps[0][i]['t']:>6} mean_pref={mp:.4f} in_window={iw:.3f} low_mass={lm:.4f} "
                  f"frugivory={fr:.4f} valid={rv:.2f} 人均摄入 低箱={lowi:.4f} 高箱={np.nanmean(intake[~LOW]):.4f} "
                  f"dlog(低箱均)={np.nanmean(dlog[LOW]):+.4f} dlog(高箱均)={np.nanmean(dlog[~LOW]):+.4f}")

    # ---------- 6. 护栏：逐 run min(carn) ----------
    print("\n[6] [预注册 §19.5] 逐 run min(carnivore_frac)，丢失捕食者的 run 分开报（不筛）")
    lost, kept = [], []
    for k in sorted(R):
        if k[0] != "R38p":
            continue
        m = float(np.nanmin(R[k]["carn"]))
        (lost if m <= 0.0 else kept).append((R[k]["name"], m, occ4(R[k]), float(R[k]["carn"][-1])))
    print(f"  丢失捕食者（min=0）的 run {len(lost)}/24：")
    for n, m, o, last in lost:
        print(f"    {n}: min={m:.4f} 末帧 carn={last:.4f} 末1/4占空比={o:.4f}")
    print(f"  其余 {len(kept)} 个 run：min 范围 [{min(x[1] for x in kept):.4f}, {max(x[1] for x in kept):.4f}]  "
          f"末1/4占空比 均值={np.mean([x[2] for x in kept]):.4f}")
    print(f"  丢失组 末1/4占空比 均值={np.mean([x[2] for x in lost]):.4f}；"
          f"占空比>0.5 的 run：丢失组 {sum(1 for x in lost if x[2]>OCC_THRESH)}/{len(lost)}，"
          f"其余 {sum(1 for x in kept if x[2]>OCC_THRESH)}/{len(kept)}")
    seeds_lost = sorted({int(n.split('_')[1][1:]) for n, *_ in lost})
    print(f"  丢失事件涉及的种子 = {seeds_lost}（格均值受影响的格）")
    keep_cells = np.array([s for s in SEEDS if s not in seeds_lost])
    if len(keep_cells) >= 2:
        m_ = np.isin(np.array(SEEDS), keep_cells)
        d3b = (c4["R38n"] - c4["R38p"])[m_]
        print(f"  [非预注册·敏感性，不作判据] 排除这些格后 H3：Δ={d3b.mean():+.4f} "
              f"{int((d3b>0).sum())}/{len(d3b)} 为正 p={wilcoxon(c4['R38n'][m_], c4['R38p'][m_]).pvalue:.5f}")

    # ---------- 7. 世代钟威胁（用户 Q4） ----------
    print("\n[7] [非预注册] 世代钟：把 R38p 截到与 R38n 相同的世代年龄")
    gn = np.array([R[k]["gen"][-1] for k in sorted(R) if k[0] == "R38n"])
    gp = np.array([R[k]["gen"][-1] for k in sorted(R) if k[0] == "R38p"])
    print(f"  末帧世代：R38n 均 {gn.mean():.1f} [{gn.min():.1f},{gn.max():.1f}]；"
          f"R38p 均 {gp.mean():.1f} [{gp.min():.1f},{gp.max():.1f}]  比 {gp.mean()/gn.mean():.2f}×")
    # R38n 末1/4 实际覆盖的世代区间
    lohi = []
    for k in sorted(R):
        if k[0] != "R38n":
            continue
        g = R[k]["gen"]
        a = int(len(g) * 0.75)
        lohi.append((g[a], g[-1]))
    GLO, GHI = float(np.mean([x[0] for x in lohi])), float(np.mean([x[1] for x in lohi]))
    print(f"  R38n 末1/4 覆盖的世代区间（均）= [{GLO:.1f}, {GHI:.1f}]")
    for a in ARMS:
        v = np.array([occ_gen_window(R[k], GLO, GHI) for k in sorted(R) if k[0] == a])
        nfr = np.mean([((R[k]["gen"] >= GLO) & (R[k]["gen"] <= GHI)).sum() for k in sorted(R) if k[0] == a])
        print(f"  {a}: 世代窗 [{GLO:.0f},{GHI:.0f}] 内占空比 均值={np.nanmean(v):.4f}  "
              f">0.5 的 run={int(np.nansum(v > OCC_THRESH))}/{len(v)}  窗内平均帧数={nfr:.0f}")
    # 配对检验（同种子），世代匹配窗
    cw = {a: np.array([np.mean([occ_gen_window(R[(a, s, r)], GLO, GHI) for r in REPS]) for s in SEEDS])
          for a in ARMS}
    dm = cw["R38n"] - cw["R38p"]
    print(f"  世代匹配后的 H3'：Δ={dm.mean():+.4f}  {int((dm>0).sum())}/{len(dm)} 为正  "
          f"p={wilcoxon(cw['R38n'], cw['R38p']).pvalue:.5f}")
    print(f"    逐格 R38n={np.round(cw['R38n'],3).tolist()}")
    print(f"    逐格 R38p={np.round(cw['R38p'],3).tolist()}")
    # R38p 沿世代的占空比曲线
    print("  R38p 沿世代的占空比（24 run 汇总，每段一个数）：")
    edges = [0, 50, 111, 148, 200, 300, 452]
    for i in range(len(edges) - 1):
        v = np.array([occ_gen_window(R[k], edges[i], edges[i + 1]) for k in sorted(R) if k[0] == "R38p"])
        print(f"    gen [{edges[i]:>3},{edges[i+1]:>3}] : 均值={np.nanmean(v):.4f}  n有效={int(np.isfinite(v).sum())}")
    print("  R38n 沿世代的占空比：")
    for i in range(4):
        v = np.array([occ_gen_window(R[k], edges[i], edges[i + 1]) for k in sorted(R) if k[0] == "R38n"])
        print(f"    gen [{edges[i]:>3},{edges[i+1]:>3}] : 均值={np.nanmean(v):.4f}  n有效={int(np.isfinite(v).sum())}")

    # ---------- 8. 口径敏感性 ----------
    print("\n[8] [非预注册] 口径敏感性")
    for mode in ("iso", "clip", "unif"):
        row = []
        for a in ARMS:
            v = np.array([occ_slice(R[k]["hist"], R[k]["gen"], 0.75, 1.0, mode) for k in sorted(R) if k[0] == a])
            row.append(f"{a} 均={v.mean():.4f} >0.5:{int((v>OCC_THRESH).sum())}/{len(v)}")
        print(f"  mode={mode:<5} " + "   ".join(row))
    for a in ARMS:
        v = np.array([occ_by_genmass(R[k]) for k in sorted(R) if k[0] == a])
        print(f"  末1/4 按**世代质量**切（非帧号）: {a} 均={np.nanmean(v):.4f} "
              f">0.5:{int(np.nansum(v>OCC_THRESH))}/{len(v)}")

    # ---------- 9. 次判据：逐四分位的 low_mass / split_score（判据 §19.4 次判据） ----------
    print("\n[9] [预注册·次判据] 逐四分位（帧号切片，世代加权）low_mass / split_score")
    from analyze_r16 import wmean_slice
    from split_score import split_score
    for a in ARMS:
        for q in range(4):
            lo_, hi_ = q * 0.25, (q + 1) * 0.25
            lm = np.array([wmean_slice([h[LOW].sum() / max(h.sum(), 1.0) for h in R[k]["hist"]],
                                       R[k]["gen"], lo_, hi_) for k in sorted(R) if k[0] == a])
            ss = np.array([wmean_slice([split_score(h, CTR)[0] for h in R[k]["hist"]],
                                       R[k]["gen"], lo_, hi_) for k in sorted(R) if k[0] == a])
            oc = np.array([occ_slice(R[k]["hist"], R[k]["gen"], lo_, hi_) for k in sorted(R) if k[0] == a])
            print(f"  {a} Q{q+1}: low_mass={lm.mean():.4f}  split={ss.mean():.4f}  占空比={oc.mean():.4f}")

    # ---------- 10. 用**未饱和**的读数回答 H2 的问题 ----------
    print("\n[10] [非预注册] H2 的问题换一个未饱和的读数：R38n 的 low_mass / split_score")
    lmq = lambda q: (lambda d: wmean_slice([h[LOW].sum() / max(h.sum(), 1.0) for h in d["hist"]],
                                           d["gen"], q * 0.25, (q + 1) * 0.25))
    ssq = lambda q: (lambda d: wmean_slice([split_score(h, CTR)[0] for h in d["hist"]],
                                           d["gen"], q * 0.25, (q + 1) * 0.25))
    for nm, f in [("low_mass", lmq), ("split_score", ssq)]:
        for a, b, lab in [(3, 2, "Q4−Q3"), (3, 0, "Q4−Q1")]:
            x, y = cell(R, "R38n", f(a)), cell(R, "R38n", f(b))
            d = x - y
            nz = noise_of(within_sd(R, "R38n", f(a)))
            lo_, hi_ = boot_ci(d)
            print(f"  R38n {nm} {lab}: Δ={d.mean():+.4f}  {int((d>0).sum())}/{len(d)} 为正  "
                  f"p={wilcoxon(x, y).pvalue:.5f}  95%CI=[{lo_:+.4f},{hi_:+.4f}]  "
                  f"噪声={nz:.4f}  比值={d.mean()/nz:+.2f}"
                  + ("  ** <1，功效不足 **" if abs(d.mean()) < nz else ""))
    # H3 的世代钟分解
    print("\n[11] [非预注册] H3 的世代钟分解")
    print(f"  100k 步末1/4（世代不匹配）Δ = {d3.mean():+.4f}")
    print(f"  世代匹配窗 [{GLO:.0f},{GHI:.0f}] 的 Δ = {dm.mean():+.4f}")
    print(f"  ⇒ 可归给「R38p 多跑了 {gp.mean()/gn.mean():.2f}× 世代」的至多 = "
          f"{d3.mean() - dm.mean():+.4f}（占总差 {(d3.mean()-dm.mean())/d3.mean()*100:.1f}%）；"
          f"世代匹配后仍剩 {dm.mean():+.4f}（{dm.mean()/d3.mean()*100:.1f}%）")

    # ---------- 12. 丢失捕食者发生在什么时候 ----------
    print("\n[12] [非预注册] R38p 丢失捕食者的 4 个 run：灭绝发生在第几步/第几代")
    for k in sorted(R):
        if k[0] != "R38p":
            continue
        c = R[k]["carn"]
        # **NaN 不是灭绝**：`trajectory.py:204` 在 4 个检查点帧上把 `population` /
        # `carnivore_frac` 写成 nan 占位（`hist`/`generation` 在那些帧是真的）。
        # 只找**有限的 0**，否则 24/24 个 run 都会被误报成「丢失捕食者」。
        idx = np.where(np.isfinite(c) & (c <= 0))[0]
        if not len(idx):
            continue
        i = int(idx[0])
        txt = open(f"outputs/20260805-longrun/{R[k]['name']}.log").read()
        tr = json.loads(txt.split("JSON ")[1].split("\n")[0])["traj"]
        print(f"    {R[k]['name']}: 首次 carn≤0 在帧 {i}/{len(c)}  t={tr[i]['t']}  "
              f"gen={R[k]['gen'][i]:.1f}/{R[k]['gen'][-1]:.1f}  "
              f"该 run 末1/4 占空比={occ4(R[k]):.4f}  首1/4={occ1(R[k]):.4f}")
    print(f"    （每个 run 有 4 个 nan 帧 = 检查点占位，`trajectory.py:204`，不是灭绝；"
          f"§19.5 的 min 用 nanmin 读，不受影响）")
    glost = [R[k]["gen"][-1] for k in sorted(R)
             if k[0] == "R38p" and np.nanmin(R[k]["carn"]) <= 0]
    gkeep = [R[k]["gen"][-1] for k in sorted(R)
             if k[0] == "R38p" and np.nanmin(R[k]["carn"]) > 0]
    print(f"    丢失组末帧世代 均={np.mean(glost):.1f} {np.round(glost,1).tolist()}；"
          f"存活组 均={np.mean(gkeep):.1f} [{min(gkeep):.1f},{max(gkeep):.1f}]；"
          f"R38n 均={gn.mean():.1f} [{gn.min():.1f},{gn.max():.1f}]")


if __name__ == "__main__":
    main()
