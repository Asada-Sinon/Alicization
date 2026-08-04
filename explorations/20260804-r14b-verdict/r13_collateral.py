"""R13 判决里其余建在截断 bin_n 上的读数，逐条查它们是否也失效。
①bin0 铺开位置的敏感性到底动没动（复核我自己的模拟）；②§14.2 的「6/24 复活」；
③§14.7 的「摄入低/高 = 0.59–0.69，53/53 格」——intake 是逐 bin 的，pref<0.15 的
个体根本没有 bin，所以「低组」里没有真正的果专精。
重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python explorations/20260804-r14b-verdict/r13_collateral.py
"""
import glob, json, math, sys
sys.path.insert(0, "scripts"); sys.path.insert(0, "explorations/20260804-readouts")
import numpy as np
from scipy.stats import wilcoxon, binomtest
BINS = np.linspace(0.15,0.85,15); CTR = 0.5*(BINS[:-1]+BINS[1:]); LOW = CTR<0.35; HIGH = CTR>0.65
runs = []
for f in sorted(glob.glob("outputs/20260804-traj/*.log")):
    for ln in open(f):
        if ln.startswith("JSON "):
            d = json.loads(ln[5:]); d["_f"]=f.split("/")[-1][:-4]; runs.append(d)
at = lambda r,t: [q for q in r["traj"] if q["t"]==t][-1]
clipv = lambda r,i: float((lambda h: h/max(h.sum(),1))(np.array(at(r,r["checkpoints"][i]["t"])["hist"],float))[LOW].sum())

print("① bin0 铺开位置的敏感性自检（我上一份脚本两档给出同样的数，先确认不是管道断了）")
rng = np.random.default_rng(0)
h0 = np.array(at(runs[0], runs[0]["checkpoints"][0]["t"])["hist"], float); p0 = h0/h0.sum()
for lo in (0.15, 0.05):
    k = rng.choice(len(CTR), size=(1,5000), p=p0)
    L = np.where(k==0, lo, BINS[k]); W = np.where(k==0, 0.20-lo, 0.05)
    pref = np.clip(L + rng.random((1,5000))*W, 1e-4, 1-1e-4)
    print(f"   bin0_lo={lo}: 初始 mean(pref)={pref.mean():.4f}  bin0 占比={float((k==0).mean()):.4f}  "
          f"其中 pref<0.15 的={float((pref<0.15).mean()):.4f}")
print("   ⇒ 该 run 的 cp0 bin0 占比很小，所以两档差异被冲淡；全 24 run 的 cp0 pref<0.15 均值只有 0.0055。")

print("\n② §14.2「6/24 复活」：clip 口径下还成不成立（复活 = 某检查点为 0 后又 >0.02）")
for tag, get in (("截断", lambda r,i: r["checkpoints"][i]["low_mass"]), ("clip", clipv)):
    rev = [r["_f"] for r in runs
           if any(get(r,i) <= 1e-9 and any(get(r,j) > 0.02 for j in range(i+1,5)) for i in range(4))]
    zero_ever = [r["_f"] for r in runs if any(get(r,i) <= 1e-9 for i in range(5))]
    print(f"   {tag:>4}: 曾归零 {len(zero_ever)}/24   归零后复活 {len(rev)}/24  {rev}")

print("\n③ §14.7 的摄入低/高：低组里有没有真正的果专精？")
cells, miss_low = [], 0
for r in runs:
    for i,cp in enumerate(r["checkpoints"]):
        ik = np.array([np.nan if v is None else v for v in cp["intake"]], float)
        n = np.array(cp["bin_n"], float)
        x, y = LOW & np.isfinite(ik) & (n>0), HIGH & np.isfinite(ik) & (n>0)
        below15 = 1.0 - n.sum()/cp["n_samples"] - max(0.0, 0.0)   # 近似：窗外总量
        iw = n.sum()/cp["n_samples"]
        if x.any() and y.any():
            cells.append(((ik[x]*n[x]).sum()/n[x].sum() / ((ik[y]*n[y]).sum()/n[y].sum()),
                          r["_f"], i, iw, clipv(r,i)))
        elif y.any():
            miss_low += 1
ratios = np.array([c[0] for c in cells])
print(f"   可算的格 {len(cells)}（§14.7 报 53）；低组缺失而高组在的格 {miss_low}")
print(f"   低/高比 均值 {ratios.mean():.4f}  中位 {np.median(ratios):.4f}  <1 的 {int((ratios<1).sum())}/{len(cells)}  "
      f"符号检验 p={binomtest(int((ratios<1).sum()), len(cells), 0.5).pvalue:.2e}")
hi = [c for c in cells if c[3] >= 0.99]; lo_ = [c for c in cells if c[3] < 0.95]
print(f"   仅 in_window>=0.99 的格 {len(hi)}：低/高比 均值 {np.mean([c[0] for c in hi]):.4f}  "
      f"<1 的 {sum(1 for c in hi if c[0]<1)}/{len(hi)}")
print(f"   in_window<0.95 的格 {len(lo_)}：低/高比 " +
      (f"均值 {np.mean([c[0] for c in lo_]):.4f}  <1 的 {sum(1 for c in lo_ if c[0]<1)}/{len(lo_)}" if lo_ else "（无）"))
big = [c for c in cells if c[4] > 0.30]
print(f"   clip low_mass>0.30（果专精真的很大）的格 {len(big)}：" +
      (f"低/高比 均值 {np.mean([c[0] for c in big]):.4f}" if big else "（一个都没有 ⇒ 摄入对比从未在真果专精上做过）"))
print(f"   全部 clip low_mass>0.30 的格总数（不问 intake 能不能算）= "
      f"{sum(1 for r in runs for i in range(5) if clipv(r,i) > 0.30)}")
