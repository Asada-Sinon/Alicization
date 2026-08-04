"""为什么 clip 口径下 wn1_s6 / wn2_s7 / wn1_s4 的 split_score 恒为 0，而它们的
low_mass 是全场最大且在长？—— 直接把直方图打出来看，并换一个不依赖峰形的留存判据。
读：outputs/20260804-traj/*.log。重跑：
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python explorations/20260804-r14b-verdict/r13_detector.py
"""
import glob, json, sys
sys.path.insert(0, "scripts"); sys.path.insert(0, "explorations/20260804-readouts")
import numpy as np
from scipy.stats import wilcoxon, fisher_exact
from exp_stats import bootstrap_ci

BINS = np.linspace(0.15, 0.85, 15); CTR = 0.5*(BINS[:-1]+BINS[1:]); LOW = CTR < 0.35
runs = {}
for f in sorted(glob.glob("outputs/20260804-traj/*.log")):
    for ln in open(f):
        if ln.startswith("JSON "):
            runs[f.split("/")[-1][:-4]] = json.loads(ln[5:])

def at(r, t):
    return [q for q in r["traj"] if q["t"] == t][-1]

print("clip 直方图（归一到 %），bin0=pref<0.20 ... bin13=pref>=0.80")
print("箱心:", np.round(CTR, 3).tolist())
for name in ("wn1_s6_r1", "wn2_s7_r1", "wn1_s4_r1", "wn2_s1_r1"):
    r = runs[name]
    print(f"\n  {name}  （§14.6 点名的丢失捕食者 run）" if name != "wn2_s1_r1" else f"\n  {name}  （§14.2 的「复活」样例）")
    for i, cp in enumerate(r["checkpoints"]):
        h = np.array(at(r, cp["t"])["hist"], float); hn = 100*h/max(h.sum(),1)
        n = np.array(cp["bin_n"], float); nn = 100*n/max(n.sum(),1)
        print(f"    cp{i} clip  " + " ".join(f"{v:5.1f}" for v in hn) +
              f"   low={hn[LOW].sum()/100:.3f} ss={at(r,cp['t'])['split_score']:.4f}")
        print(f"        trunc " + " ".join(f"{v:5.1f}" for v in nn) +
              f"   low={cp['low_mass']:.3f} ss={cp['split_score']:.4f} in_win={n.sum()/cp['n_samples']:.3f}")

print("\n" + "="*100)
print("换 §14.5 建议的判据：留存 = low_mass > 阈值（不依赖峰形），clip 口径")
print("="*100)
rows = []
for name, r in runs.items():
    row = {"f": name}
    for i, cp in enumerate(r["checkpoints"]):
        h = np.array(at(r, cp["t"])["hist"], float)
        row[f"clip{i}"] = float((h/max(h.sum(),1))[LOW].sum())
        row[f"trunc{i}"] = cp["low_mass"]
    rows.append(row)
for thr in (0.02, 0.05, 0.10):
    for k, tag in (("clip", "不截断"), ("trunc", "截断")):
        curve = [int(sum(1 for r in rows if r[f"{k}{i}"] > thr)) for i in range(5)]
        ent = [r for r in rows if r[f"{k}0"] > thr]
        cont = sum(1 for r in ent if all(r[f"{k}{i}"] > thr for i in range(5)))
        print(f"  阈值 {thr:.2f} {tag:>5}：留存曲线(分母24) {curve}   入组 {len(ent)}/24   "
              f"连续五点全保住 {cont}/{len(ent)}")

print("\n" + "="*100)
print("§14.6 的敏感性复核：去掉 3 个丢失捕食者的 run（wn1_s4/wn1_s6/wn2_s7）")
print("="*100)
drop = {"wn1_s4_r1", "wn1_s6_r1", "wn2_s7_r1"}
for k, tag in (("trunc", "截断（§14.6 报的）"), ("clip", "不截断")):
    for sub, lbl in ((rows, "全 24"), ([r for r in rows if r["f"] not in drop], "去掉 3 个")):
        x0 = np.array([r[f"{k}0"] for r in sub]); x4 = np.array([r[f"{k}4"] for r in sub])
        d = x4 - x0; lo, hi = bootstrap_ci(d)
        print(f"  {tag:>16} {lbl:>8}(n={len(sub)}): {x0.mean():.4f}->{x4.mean():.4f}  Δ={d.mean():+.4f}  "
              f"{int((d<0).sum())}降/{int((d>0).sum())}升/{int((d==0).sum())}平  "
              f"p={wilcoxon(x4,x0).pvalue:.5f}  CI=[{lo:+.4f},{hi:+.4f}]")
print("  那 3 个 run 自己（clip）：", [(r['f'], round(r['clip0'],3), round(r['clip4'],3))
                                     for r in rows if r['f'] in drop])
print("  那 3 个 run 自己（trunc）：", [(r['f'], round(r['trunc0'],3), round(r['trunc4'],3))
                                      for r in rows if r['f'] in drop])
