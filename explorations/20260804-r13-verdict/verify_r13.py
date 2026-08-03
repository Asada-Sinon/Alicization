"""R13 判决的独立复核：把 analyze_r13.py 的每个数从原始日志重算一遍，并补它没算的诊断。

读：outputs/20260804-traj/*.log（24 个 run，每个一行 "JSON "，含 5 个 checkpoint + 每 250 步 traj）
判据：docs/multispecies_program.md §16（commit d3c3029，跑之前提交）

回答的问题：
  A 复现入组 / 留存曲线 / 斜率 / mean(dlog) / 护栏（对着 analyze_r13.py 的输出逐项核）
  B 留存曲线的分母：§16.3 写的是「计入留存曲线的分子分母」⇒ 分母应为 24，脚本用了 15
  C 斜率噪声从哪来：逐 run 的频率支撑范围 vs |斜率|
  D dlog 的共模混杂：全箱质量加权 dlog（种群总量项）vs 低箱−高箱对比
  E 低簇的适应度代理：逐检查点 低箱/高箱 的 intake 与 drink（§13.8 要的含水代理）
  F 地形（wn=1/2）拆分留存；2/15 存活 run 的共同点
  G 世代长与非单调 low_mass 的来源

输出：stdout（explorations/20260804-r13-verdict/output/verify.txt）
重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260804-r13-verdict/verify_r13.py
"""
import glob
import json
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, "explorations/20260804-readouts")

import numpy as np
from scipy.stats import wilcoxon, fisher_exact, spearmanr

from split_score import split_score

CTR = np.linspace(0.15, 0.85, 15)
CTR = 0.5 * (CTR[:-1] + CTR[1:])
LOW = CTR < 0.35
HIGH = CTR > 0.65

runs = []
for f in sorted(glob.glob("outputs/20260804-traj/*.log")):
    for ln in open(f):
        if ln.startswith("JSON "):
            d = json.loads(ln[5:])
            d["_file"] = f.split("/")[-1]
            d["_wn"] = int(d["_file"].split("_")[0][2:])
            d["_seed"] = int(d["_file"].split("_")[1][1:])
            runs.append(d)
runs.sort(key=lambda r: (r["_wn"], r["_seed"]))
print(f"[A] 载入 {len(runs)} 个 run；collapsed={sum(r['collapsed'] for r in runs)}；"
      f"每个 run 的 checkpoint 数 {sorted(set(len(r['checkpoints']) for r in runs))}")

# --- 存档 split_score 与从 bin_n 重算的往返一致性（复用闸） ---
err = max(abs(split_score(cp["bin_n"], CTR)[0] - cp["split_score"])
          for r in runs for cp in r["checkpoints"])
print(f"[A] 存档 split_score vs 从 bin_n 重算：最大绝对差 {err:.3e}")

entered = [r for r in runs if r["checkpoints"][0]["split_score"] > 0.0]
print(f"[A] 入组（cp0 split_score>0）{len(entered)}/{len(runs)}；"
      f"20k 前即丢失 {len(runs) - len(entered)}/{len(runs)}")

print("\n[A/B] 留存曲线，两个分母")
print(f'  {"cp":>3} {"t":>7} {"入组留存":>10} {"全体留存":>10} {"low_mass(入组)":>16} {"low_mass(全体)":>16}')
for i in range(5):
    t = np.mean([r["checkpoints"][i]["t"] for r in runs])
    re_ = sum(1 for r in entered if r["checkpoints"][i]["split_score"] > 0)
    ra = sum(1 for r in runs if r["checkpoints"][i]["split_score"] > 0)
    lme = np.mean([r["checkpoints"][i]["low_mass"] for r in entered])
    lma = np.mean([r["checkpoints"][i]["low_mass"] for r in runs])
    print(f"  {i:>3} {t:>7.0f} {re_:>7}/{len(entered):<3} {ra:>7}/{len(runs):<3} "
          f"{lme:>16.4f} {lma:>16.4f}")

print("\n[B] 复活事件：某个 cp 上 split_score=0 而更晚的 cp 上 >0 的 run")
revive = []
for r in runs:
    ss = [cp["split_score"] for cp in r["checkpoints"]]
    for i in range(1, 5):
        if ss[i] > 0 and ss[i - 1] == 0:
            revive.append((r["_file"], i, ss))
            break
for f, i, ss in revive:
    print(f"    {f:<16} 首次复活于 cp{i}   split_score 轨迹 " + " ".join(f"{v:.4f}" for v in ss))
print(f"    复活 run 数 {len(revive)}/{len(runs)}")

print("\n[F/G] 逐 run 全表（split_score / low_mass 轨迹）")
print(f'  {"file":<16} {"wn":>2} {"seed":>4} {"入组":>4} | '
      + " ".join(f"{'ss@cp'+str(i):>8}" for i in range(5)) + " | "
      + " ".join(f"{'lm@cp'+str(i):>8}" for i in range(5)) + f' {"gen":>6}')
for r in runs:
    ss = [cp["split_score"] for cp in r["checkpoints"]]
    lm = [cp["low_mass"] for cp in r["checkpoints"]]
    print(f'  {r["_file"]:<16} {r["_wn"]:>2} {r["_seed"]:>4} '
          f'{"Y" if ss[0] > 0 else "-":>4} | '
          + " ".join(f"{v:8.4f}" for v in ss) + " | "
          + " ".join(f"{v:8.4f}" for v in lm) + f' {r.get("gen_total", float("nan")):6.0f}')

print("\n[F] 地形拆分：逐 cp 的留存（分母=该地形的入组数）")
for wn in (1, 2):
    e = [r for r in entered if r["_wn"] == wn]
    row = [sum(1 for r in e if r["checkpoints"][i]["split_score"] > 0) for i in range(5)]
    print(f"  wn={wn}  入组 {len(e)}/12   留存 " + "  ".join(f"{v}/{len(e)}" for v in row))
tab = [[sum(1 for r in runs if r["_wn"] == wn and r["checkpoints"][0]["split_score"] > 0),
        sum(1 for r in runs if r["_wn"] == wn and r["checkpoints"][0]["split_score"] == 0)]
       for wn in (1, 2)]
print(f"  入组率 2x2 {tab}  Fisher p={fisher_exact(tab)[1]:.4f}")
tab2 = [[sum(1 for r in entered if r["_wn"] == wn and r["checkpoints"][4]["split_score"] > 0),
         sum(1 for r in entered if r["_wn"] == wn and r["checkpoints"][4]["split_score"] == 0)]
        for wn in (1, 2)]
print(f"  cp4 留存 2x2 {tab2}  Fisher p={fisher_exact(tab2)[1]:.4f}")
lm4 = {wn: np.array([r["checkpoints"][4]["low_mass"] for r in runs if r["_wn"] == wn])
       for wn in (1, 2)}
print(f"  cp4 low_mass 臂均 wn1={lm4[1].mean():.4f} wn2={lm4[2].mean():.4f}")

print("\n[F] 入组 split_score 是否预测存活到 cp4")
surv = np.array([r["checkpoints"][0]["split_score"] for r in entered
                 if r["checkpoints"][4]["split_score"] > 0])
died = np.array([r["checkpoints"][0]["split_score"] for r in entered
                 if r["checkpoints"][4]["split_score"] == 0])
print(f"  存活到 cp4 的入组分 {np.round(surv, 4).tolist()}（n={len(surv)}）")
print(f"  未存活的入组分   均值 {died.mean():.4f} 中位 {np.median(died):.4f} n={len(died)}")

# ---------------------------------------------------------------- C/D: dlog 诊断
def per_cp(r):
    """每个 cp 返回 (freq_low, dlog_low, dlog_all, dlog_high, low_int, high_int, low_dr, high_dr)"""
    out = []
    for cp in r["checkpoints"]:
        n = np.array(cp["bin_n"], float)
        n = n / max(n.sum(), 1)
        dl = np.array([np.nan if v is None else v for v in cp["dlog_per_bin"]], float)
        it = np.array([np.nan if v is None else v for v in cp["intake"]], float)
        dr = np.array([np.nan if v is None else v for v in cp["drink"]], float)
        mlo = LOW & np.isfinite(dl) & (n > 0)
        mhi = HIGH & np.isfinite(dl) & (n > 0)
        mall = np.isfinite(dl) & (n > 0)
        w = lambda m, v: float((v[m] * n[m]).sum() / max(n[m].sum(), 1e-12)) if m.any() else np.nan
        out.append(dict(
            t=cp["t"], freq=float(n[LOW].sum()),
            dlow=w(mlo, dl) if mlo.any() else np.nan,
            dhigh=w(mhi, dl) if mhi.any() else np.nan,
            dall=w(mall, dl) if mall.any() else np.nan,
            ilow=w(LOW & np.isfinite(it) & (n > 0), it),
            ihigh=w(HIGH & np.isfinite(it) & (n > 0), it),
            wlow=w(LOW & np.isfinite(dr) & (n > 0), dr),
            whigh=w(HIGH & np.isfinite(dr) & (n > 0), dr),
        ))
    return out


print("\n[C] 斜率诊断：逐 run 的可用点数、少数簇频率支撑范围、斜率")
print(f'  {"file":<16} {"pts":>4} {"freq_min":>9} {"freq_max":>9} {"range":>8} '
      f'{"slope":>10} {"mean_dlog":>10} {"spearman":>9}')
rows = []
for r in entered:
    pts = [p for p in per_cp(r) if np.isfinite(p["dlow"])]
    fr = np.array([p["freq"] for p in pts])
    dl = np.array([p["dlow"] for p in pts])
    sl = float(np.polyfit(fr, dl, 1)[0]) if len(pts) >= 3 and fr.std() > 1e-9 else np.nan
    rho = spearmanr(fr, dl).statistic if len(pts) >= 3 else np.nan
    rows.append((r["_file"], len(pts), fr, dl, sl, dl.mean() if len(dl) else np.nan, rho))
    print(f'  {r["_file"]:<16} {len(pts):>4} '
          f'{fr.min() if len(fr) else np.nan:>9.4f} {fr.max() if len(fr) else np.nan:>9.4f} '
          f'{(fr.max()-fr.min()) if len(fr) else np.nan:>8.4f} '
          f'{sl:>10.3f} {dl.mean() if len(dl) else np.nan:>10.4f} {rho:>9.3f}')

ok = [r for r in rows if np.isfinite(r[4])]
S = np.array([r[4] for r in ok])
M = np.array([r[5] for r in ok])
RG = np.array([r[2].max() - r[2].min() for r in ok])
RHO = np.array([r[6] for r in ok])
print(f"\n[A] 斜率 n={len(S)} 均值 {S.mean():+.4f} 中位 {np.median(S):+.4f} "
      f"{(S<0).sum()}/{len(S)} 为负 符号秩 p={wilcoxon(S).pvalue:.5f} "
      f"SD {S.std(ddof=1):.4f} 效应/噪声 {S.mean()/S.std(ddof=1):+.2f}")
print(f"[A] mean(dlog) n={len(M)} 均值 {M.mean():+.4f} 中位 {np.median(M):+.4f} "
      f"{(M<0).sum()}/{len(M)} 为负 符号秩 p={wilcoxon(M).pvalue:.5f} "
      f"SD {M.std(ddof=1):.4f} 效应/噪声 {M.mean()/M.std(ddof=1):+.2f}")
print(f"[C] |斜率| vs 频率支撑范围：Spearman rho={spearmanr(np.abs(S), RG).statistic:+.3f} "
      f"p={spearmanr(np.abs(S), RG).pvalue:.4f}；range 逐 run {np.round(RG,4).tolist()}")
print(f"[C] 稳健替代：逐 run Spearman(freq, dlow) 均值 {RHO.mean():+.3f} 中位 {np.median(RHO):+.3f} "
      f"{(RHO<0).sum()}/{len(RHO)} 为负 符号秩 p={wilcoxon(RHO).pvalue:.5f} "
      f"效应/噪声 {RHO.mean()/RHO.std(ddof=1):+.2f}")

print("\n[D] dlog 的共模项：全箱质量加权 dlog（= 该窗内食草总量的 log 变化）")
allc = [p for r in entered for p in per_cp(r) if np.isfinite(p["dall"])]
DA = np.array([p["dall"] for p in allc])
print(f"  n={len(DA)} 个 (run,cp)  均值 {DA.mean():+.4f}  SD {DA.std(ddof=1):.4f}  "
      f"范围 [{DA.min():+.4f}, {DA.max():+.4f}]")
pair = [(p["dlow"], p["dall"], p["dhigh"]) for r in entered for p in per_cp(r)
        if np.isfinite(p["dlow"]) and np.isfinite(p["dall"])]
DL = np.array([q[0] for q in pair]); DAp = np.array([q[1] for q in pair])
print(f"  低箱 dlog 的 SD {DL.std(ddof=1):.4f}；扣掉共模后 (dlow-dall) 的 SD "
      f"{(DL-DAp).std(ddof=1):.4f}  ⇒ 共模解释了方差的 "
      f"{1 - (DL-DAp).var(ddof=1)/DL.var(ddof=1):.1%}")
c = [(p["dlow"] - p["dall"]) for r in entered for p in per_cp(r) if np.isfinite(p["dlow"])]
c = np.array(c)
print(f"  [事后] 低箱相对全体的 dlog（dlow-dall）：n={len(c)} 均值 {c.mean():+.4f} "
      f"{(c<0).sum()}/{len(c)} 为负 符号秩 p={wilcoxon(c).pvalue:.6f}")
# 按 run 取均值再检验（避免伪重复 run 内多 cp）
byrun = np.array([np.nanmean([p["dlow"] - p["dall"] for p in per_cp(r)
                              if np.isfinite(p["dlow"])]) for r in entered])
byrun = byrun[np.isfinite(byrun)]
print(f"  [事后] 同量按 run 取均值：n={len(byrun)} 均值 {byrun.mean():+.4f} "
      f"{(byrun<0).sum()}/{len(byrun)} 为负 符号秩 p={wilcoxon(byrun).pvalue:.5f} "
      f"效应/噪声 {byrun.mean()/byrun.std(ddof=1):+.2f}")

print("\n[E] 低簇 vs 高簇的两个代理（逐 cp，池化全部 24 run）")
print(f'  {"cp":>3} {"n":>4} {"intake低":>9} {"intake高":>9} {"比":>6} '
      f'{"drink低":>9} {"drink高":>9} {"比":>6}')
for i in range(5):
    P = [per_cp(r)[i] for r in runs]
    il = np.array([p["ilow"] for p in P]); ih = np.array([p["ihigh"] for p in P])
    wl = np.array([p["wlow"] for p in P]); wh = np.array([p["whigh"] for p in P])
    m = np.isfinite(il) & np.isfinite(ih)
    mw = np.isfinite(wl) & np.isfinite(wh)
    print(f"  {i:>3} {m.sum():>4} {il[m].mean():>9.4f} {ih[m].mean():>9.4f} "
          f"{il[m].mean()/ih[m].mean():>6.3f} {wl[mw].mean():>9.4f} {wh[mw].mean():>9.4f} "
          f"{wl[mw].mean()/wh[mw].mean():>6.3f}")
il0 = np.array([[p["ilow"], p["ihigh"], p["wlow"], p["whigh"]]
                for r in runs for p in per_cp(r)])
m = np.isfinite(il0).all(1)
z = il0[m]
print(f"  池化 n={m.sum()} (run,cp)：intake 低<高 {int((z[:,0]<z[:,1]).sum())}/{m.sum()} "
      f"符号秩 p={wilcoxon(z[:,0], z[:,1]).pvalue:.3e}；"
      f"drink 低<高 {int((z[:,2]<z[:,3]).sum())}/{m.sum()} "
      f"符号秩 p={wilcoxon(z[:,2], z[:,3]).pvalue:.3e}")

print("\n[G] 世代长与护栏轨迹")
gt = np.array([r["gen_total"] for r in runs])
print(f"  gen_total（100k 步走过的世代数）均值 {gt.mean():.1f} 中位 {np.median(gt):.1f} "
      f"范围 [{gt.min():.0f}, {gt.max():.0f}]  ⇒ 步/代 {100000/gt.mean():.1f}")
print(f"  22k→102k 的 80k 步 ≈ {80000/(100000/gt.mean()):.0f} 代")
for i in range(5):
    P = [[q for q in r["traj"] if abs(q["t"] - r["checkpoints"][i]["t"]) < 3000] for r in runs]
    pop = [np.nanmean([q["population"] for q in t]) for t in P if t]
    cf = [np.nanmean([q["carnivore_frac"] for q in t]) for t in P if t]
    nh = [min(q["n_herb"] for q in t) for t in P if t]
    mp = [r["checkpoints"][i]["mean_pref"] for r in runs]
    sd = [r["checkpoints"][i]["sd"] for r in runs]
    print(f"  cp{i} pop {np.nanmean(pop):7.0f}  carn {np.nanmean(cf):.4f}  "
          f"min n_herb {np.min(nh):5.0f}  mean_pref {np.mean(mp):.4f}  pref_sd {np.mean(sd):.4f}")

print("\n[G] 非单调 low_mass（cp3 0.0230 → cp4 0.0446）来源：逐 run 的 lm 变化")
d = np.array([[r["checkpoints"][3]["low_mass"], r["checkpoints"][4]["low_mass"]] for r in entered])
print(f"  入组 15 个 run 的 cp3→cp4 变化：{(d[:,1]>d[:,0]).sum()} 个上升，"
      f"{(d[:,1]<d[:,0]).sum()} 个下降；最大单 run 增量 {np.max(d[:,1]-d[:,0]):+.4f}")
top = np.argsort(-(d[:, 1] - d[:, 0]))[:3]
for k in top:
    print(f"    {entered[k]['_file']:<16} {d[k,0]:.4f} → {d[k,1]:.4f}")
print(f"  去掉增量最大的那个 run 后 cp4 均值 {np.mean(np.delete(d[:,1], top[0])):.4f} "
      f"(全体 {d[:,1].mean():.4f})")
