"""R20 判决（feasibility.md §27, commit f162a89）的独立复算。

回答的问题（对应复核指令的六点里的 2/3/4/5 与全表复现）：
  A. §27.1 全表能否从 outputs/20260813-r20 复现（低质量/两峰/carn_frac/代数/崩溃），
     并反推它用的口径（帧比例 vs 世代加权）——§19 勘误的教训：判决没留脚本，口径只能反推。
  B. H2 的配对统计：逐种子差、含并列的符号计数、精确 Wilcoxon 与精确符号检验的 p、
     效应 ÷ 配对差噪声（exp_stats 口径：√2·σ̂_W/√r）、MDE。
  C. gen100±5 对齐窗：每 run 采样帧数、两臂在窗内的步数区间（世代对齐 = 步数不对齐）、
     等步数窗 t∈[19k,23k] 的替代对齐。
  D. 捕食臂逐 run 的捕食者存活（R16 的「处理失效」：48 里 4 个丢捕食者——本轮呢）。
  E. gen700（R18-B 800k 步、无捕食）在 gen∈[433,481] 切片的 low_mass/两峰——
     「多跑三倍代数自己会不会丢双峰」的直接对照，比 gen100 截断更硬。
  F. 捕食臂的形状：低峰消失是「搬家」还是「合并」——mean_pref、两峰时的谷位/两侧质量、
     逐四分位 low_mass（先形成后被压 vs 从未形成）。

读的文件：outputs/20260813-r20/*.log、outputs/20260806-gen700/*.log 的 `JSON ` 行。
跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
      explorations/20260815-r20-audit/recompute_verdict.py
"""
import glob, json, math, sys
sys.path.insert(0, "scripts")
sys.path.insert(0, "explorations/20260804-readouts")
sys.path.insert(0, "explorations/20260805-r18-verdict")
import numpy as np
from scipy.stats import wilcoxon, binomtest
from diag_h2_degenerate import parts
from exp_stats import bootstrap_ci, wilcoxon_p_floor, mde_sign_consistent
from neutral_null import gen_weights

BINS = np.linspace(0.0, 1.0, 21); CTR = 0.5*(BINS[:-1]+BINS[1:]); LOW = CTR < 0.35
SEEDS = list(range(12)); REPS = (1, 2)
CELLS = ["wn1off","wn1on","wn2off","wn2on","b35off","b35on"]
PUB = {  # §27.1 已发表数（feasibility.md 2952-2957 行）
 "wn1off": (0.5105,1.0000,0.0000,148), "wn1on": (0.0716,0.5773,0.1047,481),
 "wn2off": (0.5171,1.0000,0.0000,123), "wn2on": (0.0233,0.3403,0.1236,481),
 "b35off": (0.4660,1.0000,0.0000,151), "b35on": (0.0918,0.4215,0.0879,433)}
PUB100 = {  # §27.3 gen100 表（2986-2988 行）：off_lm, off_pk, on_lm, on_pk
 "wn1": (0.4866,1.0000,0.1074,0.8024), "wn2": (0.5101,1.0000,0.0884,0.7274),
 "b35": (0.4418,1.0000,0.1476,0.7708)}

def load(run_dir, pat="*"):
    R = {}
    for f in sorted(glob.glob(f"{run_dir}/{pat}.log")):
        txt = open(f).read()
        if "JSON " not in txt: continue
        d = json.loads(txt.split("JSON ")[1].split("\n")[0])
        tr = d["traj"]
        b = f.split("/")[-1][:-4].split("_")
        R[(b[0], int(b[1][1:]), int(b[2][1:]))] = {
            "hist": np.array([q["hist"] for q in tr], float),
            "gen":  np.array([q["generation"] for q in tr], float),
            "t":    np.array([q["t"] for q in tr], float),
            "carn": np.array([q["carnivore_frac"] for q in tr], float),
            "nher": np.array([q["n_herb"] for q in tr], float),
            "collapsed": bool(d["collapsed"]), "ov": d["overrides"], "src": f}
    return R

def lm(h): s = h.sum(); return h[LOW].sum()/s if s > 0 else np.nan
def two(h): p = parts(h, CTR); return 1.0 if (p and "minM" in p) else 0.0

def run_stats(rec, mask, weighted):
    H, g = rec["hist"][mask], rec["gen"][mask]
    if H.shape[0] == 0: return np.nan, np.nan, 0
    w = gen_weights(rec["gen"], "iso")[mask] if weighted else np.ones(H.shape[0])
    if w.sum() <= 1e-9: w = np.ones(H.shape[0])
    lms = np.array([lm(h) for h in H]); tw = np.array([two(h) for h in H])
    return float((w*lms).sum()/w.sum()), float((w*tw).sum()/w.sum()), int(H.shape[0])

def cellmean(per, cell):
    mus = []
    for s in SEEDS:
        vs = [per[(cell,s,r)] for r in REPS if (cell,s,r) in per and np.isfinite(per[(cell,s,r)])]
        mus.append(np.mean(vs) if vs else np.nan)
    return np.array(mus)

R = load("outputs/20260813-r20")
print(f"载入 {len(R)} 个 run；崩溃 {sum(v['collapsed'] for v in R.values())} 个")
# --- 归因检查：逐格 overrides ---------------------------------------------
byc = {}
for (c,s,r),v in R.items(): byc.setdefault(c, set()).add(json.dumps(v["ov"], sort_keys=True))
for c in CELLS: assert len(byc[c]) == 1, (c, byc[c])
ov = {c: json.loads(list(byc[c])[0]) for c in CELLS}
for t in ("wn1","wn2","b35"):
    on, off = ov[t+"on"], ov[t+"off"]
    dk = {k for k in set(on)|set(off) if on.get(k) != off.get(k)}
    print(f"  {t}: on vs off 差异键 = {sorted(dk)}")
print(f"  wn1off vs wn2off 差异键 = {sorted({k for k in set(ov['wn1off'])|set(ov['wn2off']) if ov['wn1off'].get(k)!=ov['wn2off'].get(k)})}")
print(f"  wn1off vs b35off 差异键 = {sorted({k for k in set(ov['wn1off'])|set(ov['b35off']) if ov['wn1off'].get(k)!=ov['b35off'].get(k)})}")

# --- A. §27.1 复现（两种口径） --------------------------------------------
print("\n== A. §27.1 复现：格均值（帧比例口径 | 世代iso口径 | 已发表） ==")
per_fr, per_iso, per_c, per_g = {}, {}, {}, {}
for k, rec in R.items():
    late = rec["gen"] >= rec["gen"].max()*0.75
    a, b, _ = run_stats(rec, late, False); per_fr[k] = (a, b)
    a2, b2, _ = run_stats(rec, late, True); per_iso[k] = (a2, b2)
    fin = np.isfinite(rec["carn"])
    per_c[k] = float(rec["carn"][late & fin].mean())
    per_g[k] = float(rec["gen"][-1])
for c in CELLS:
    lf = cellmean({k: v[0] for k, v in per_fr.items()}, c); tf = cellmean({k: v[1] for k, v in per_fr.items()}, c)
    li = cellmean({k: v[0] for k, v in per_iso.items()}, c); ti = cellmean({k: v[1] for k, v in per_iso.items()}, c)
    cf = cellmean(per_c, c); gf = cellmean(per_g, c)
    p = PUB[c]
    print(f"  {c:7s} lm {np.nanmean(lf):.4f}|{np.nanmean(li):.4f}|{p[0]:.4f}"
          f"   两峰 {np.nanmean(tf):.4f}|{np.nanmean(ti):.4f}|{p[1]:.4f}"
          f"   carn {np.nanmean(cf):.4f}|pub {p[2]:.4f}   gen {np.nanmean(gf):.0f}|pub {p[3]}")

# --- D. 捕食臂逐 run 捕食者存活 --------------------------------------------
print("\n== D. 捕食臂逐 run：末帧 carn_frac 与「丢失捕食者」计数（R16: 4/24） ==")
for c in ("wn1on","wn2on","b35on"):
    lost, rows = [], []
    for s in SEEDS:
        for r in REPS:
            rec = R[(c,s,r)]; fin = np.isfinite(rec["carn"])
            cf = rec["carn"][fin]; final = cf[-1]
            if final < 0.005: lost.append((s,r))
            rows.append(final)
    rows = np.array(rows)
    print(f"  {c}: 末帧 carn min={rows.min():.4f} P25={np.percentile(rows,25):.4f} "
          f"中位={np.median(rows):.4f} max={rows.max():.4f}   丢失(<0.005)={len(lost)}/24 {lost}")

# --- B. H2 配对统计（末四分之一，帧比例口径） -------------------------------
def paired_block(tag, per_lm, per_tw):
    print(f"\n== B. H2 配对（{tag}）on−off，s=12 格均值 ==")
    for t in ("wn1","wn2","b35"):
        for name, per in (("low_mass", per_lm), ("两峰占比", per_tw)):
            on, off = cellmean(per, t+"on"), cellmean(per, t+"off")
            m = np.isfinite(on) & np.isfinite(off); d = (on-off)[m]
            npos, nneg, nz = int((d>0).sum()), int((d<0).sum()), int((d==0).sum())
            nnz = npos+nneg
            try:
                w_ex = wilcoxon(d[d!=0], mode="exact") if nnz else None
            except Exception:
                w_ex = wilcoxon(d[d!=0]) if nnz else None
            w_pr = wilcoxon(d, zero_method="pratt") if np.any(d!=0) else None
            sign_p = binomtest(min(npos,nneg), nnz, 0.5).pvalue if nnz else np.nan
            # 噪声：两臂池化的格内 SD（exp_stats.pair_noise 口径，r=2 ⇒ noise=σ̂_W）
            ws = []
            for cc in (t+"on", t+"off"):
                for s in SEEDS:
                    vs = [per[(cc,s,r)] for r in REPS if np.isfinite(per.get((cc,s,r), np.nan))]
                    if len(vs) == 2: ws.append(np.std(vs, ddof=1))
            sw = float(np.sqrt(np.mean(np.square(ws)))) if ws else np.nan
            noise = sw*math.sqrt(2)/math.sqrt(2)
            lo, hi = bootstrap_ci(d)
            mde = mde_sign_consistent(float(d.std(ddof=1)), int(m.sum())) if m.sum()>1 else np.nan
            print(f"  {t} {name:8s} Δ={d.mean():+.4f} CI[{lo:+.4f},{hi:+.4f}] "
              f"符号 −{nneg}/0×{nz}/+{npos}  Wilcoxon(drop0) p={w_ex.pvalue if w_ex else np.nan:.5f}"
              f"  (pratt p={w_pr.pvalue if w_pr else np.nan:.5f})  精确符号检验 p={sign_p:.2e}"
              f"  地板(n={nnz})={wilcoxon_p_floor(nnz):.5f}  效应/噪声={d.mean()/noise if noise>1e-9 else float('nan'):+.1f}"
              f"  σ̂_W={sw:.4f} MDE={mde:.4f}")
            if nz: print(f"      并列种子: {[SEEDS[i] for i in range(len(d)) if d[i]==0]} ← 这些不是「同向」")
per_lm_fr = {k: v[0] for k, v in per_fr.items()}; per_tw_fr = {k: v[1] for k, v in per_fr.items()}
paired_block("末四分之一·帧比例", per_lm_fr, per_tw_fr)

# --- C. gen100±5 对齐 -------------------------------------------------------
print("\n== C. gen100±5 窗：帧数、步数区间、格均值、配对 ==")
per_lm100, per_tw100, meta = {}, {}, {}
for k, rec in R.items():
    m = np.abs(rec["gen"] - 100.0) <= 5.0
    a, b, n = run_stats(rec, m, False)
    per_lm100[k], per_tw100[k] = a, b
    meta[k] = (n, (rec["t"][m].min(), rec["t"][m].max()) if n else (np.nan, np.nan))
for c in CELLS:
    ns = [meta[(c,s,r)][0] for s in SEEDS for r in REPS]
    t0 = np.nanmean([meta[(c,s,r)][1][0] for s in SEEDS for r in REPS])
    t1 = np.nanmean([meta[(c,s,r)][1][1] for s in SEEDS for r in REPS])
    lmc = cellmean(per_lm100, c); twc = cellmean(per_tw100, c)
    print(f"  {c:7s} 窗内帧数 min={min(ns)} 中位={int(np.median(ns))} max={max(ns)}"
          f"   平均步数区间 [{t0:.0f},{t1:.0f}]   lm={np.nanmean(lmc):.4f} 两峰={np.nanmean(twc):.4f}")
for t in ("wn1","wn2","b35"):
    p = PUB100[t]
    print(f"  {t} 已发表: off {p[0]:.4f}/{p[1]:.4f}  on {p[2]:.4f}/{p[3]:.4f}")
paired_block("gen100±5·帧比例", per_lm100, per_tw100)

# 等步数窗（on 臂到 gen100 的typical步数区间）：t∈[19000,23000]
print("\n== C2. 等步数窗 t∈[19k,23k]（世代对齐的镜像检查：步数对齐、世代不对齐） ==")
per_lm_t, per_tw_t = {}, {}
for k, rec in R.items():
    m = (rec["t"] >= 19000) & (rec["t"] <= 23000)
    a, b, n = run_stats(rec, m, False); per_lm_t[k], per_tw_t[k] = a, b
for c in CELLS:
    lmc = cellmean(per_lm_t, c); twc = cellmean(per_tw_t, c)
    g0 = np.nanmean([R[(c,s,r)]["gen"][(R[(c,s,r)]["t"]>=19000)&(R[(c,s,r)]["t"]<=23000)].mean() for s in SEEDS for r in REPS])
    print(f"  {c:7s} lm={np.nanmean(lmc):.4f} 两峰={np.nanmean(twc):.4f}  窗内平均世代 {g0:.0f}")
paired_block("等步数 t∈[19k,23k]·帧比例", per_lm_t, per_tw_t)

# --- E. gen700（R18-B 800k, 无捕食）在捕食臂代数区间上的切片 -----------------
print("\n== E. gen700 无捕食 800k：末四分之一与 gen∈[433,481] 切片 ==")
G = load("outputs/20260806-gen700")
print(f"  载入 {len(G)} run，崩溃 {sum(v['collapsed'] for v in G.values())}")
for tag, mk in (("末1/4(按代)", lambda rec: rec["gen"] >= rec["gen"].max()*0.75),
                ("gen∈[433,481]", lambda rec: (rec["gen"]>=433)&(rec["gen"]<=481)),
                ("gen∈[95,105]", lambda rec: np.abs(rec["gen"]-100)<=5)):
    per_l, per_t2, nmin = {}, {}, []
    for k, rec in G.items():
        m = mk(rec); a, b, n = run_stats(rec, m, False); per_l[k], per_t2[k] = a, b; nmin.append(n)
    lmc = cellmean(per_l, "R38n"); twc = cellmean(per_t2, "R38n")
    print(f"  {tag:14s} lm={np.nanmean(lmc):.4f} [{np.nanmin(lmc):.4f},{np.nanmax(lmc):.4f}]"
          f"  两峰={np.nanmean(twc):.4f} [{np.nanmin(twc):.4f},{np.nanmax(twc):.4f}]  窗内帧数 min={min(nmin)}")
gmax = [G[k]["gen"].max() for k in G]
print(f"  gen700 末帧代数：min={min(gmax):.0f} 中位={np.median(gmax):.0f} max={max(gmax):.0f}")

# --- F. 形状：搬家还是合并 ---------------------------------------------------
print("\n== F. 捕食臂末四分之一的形状 ==")
for c in CELLS:
    hs = []
    for s in SEEDS:
        for r in REPS:
            rec = R[(c,s,r)]; late = rec["gen"] >= rec["gen"].max()*0.75
            H = rec["hist"][late]; hn = H / np.maximum(H.sum(1, keepdims=True), 1e-9)
            hs.append(hn.mean(0))
    hb = np.mean(hs, 0)
    mp = float((hb*CTR).sum()/hb.sum()); hi_m = float(hb[CTR>0.65].sum()/hb.sum())
    mid = float(hb[(CTR>=0.35)&(CTR<=0.65)].sum()/hb.sum())
    pk = parts(hb*1000, CTR)
    ex = (f" 两峰[mL={pk['mL']:.3f},mR={pk['mR']:.3f},谷位={pk['valley_pos']:.3f},gap={pk['gap']:.3f}]"
          if pk and "minM" in pk else f" npeak={pk['npeak'] if pk else 0}")
    top = np.argsort(hb)[-2:][::-1]
    print(f"  {c:7s} mean_pref={mp:.3f} low={float(hb[LOW].sum()/hb.sum()):.3f} mid={mid:.3f} high={hi_m:.3f}"
          f" 最高两箱ctr={CTR[top[0]]:.3f},{CTR[top[1]]:.3f}{ex}")
print("\n  捕食臂 low_mass 逐四分位（格均值；「先形成后被压」vs「从未形成」）+ 逐 run 峰值")
for c in ("wn1on","wn2on","b35on","wn1off"):
    qs = []
    for qi in range(4):
        per_q = {}
        for k, rec in R.items():
            if k[0] != c: continue
            gm = rec["gen"].max(); m = (rec["gen"] >= gm*qi/4) & (rec["gen"] < gm*(qi+1)/4 + (1 if qi==3 else 0))
            per_q[k] = run_stats(rec, m, False)[0]
        qs.append(np.nanmean(cellmean(per_q, c)))
    pkm = {}
    for k, rec in R.items():
        if k[0] != c: continue
        lms = np.array([lm(h) for h in rec["hist"]])
        pkm[k] = float(np.nanmax(lms))
    pk_cells = cellmean(pkm, c)
    print(f"  {c:7s} Q1-Q4: {['%.4f'%q for q in qs]}   run内 max(low_mass) 格均值 {np.nanmean(pk_cells):.4f} [{np.nanmin(pk_cells):.4f},{np.nanmax(pk_cells):.4f}]")
