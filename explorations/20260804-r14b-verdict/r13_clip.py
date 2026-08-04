"""R13（outputs/20260804-traj）的截断诊断与不截断重算。

R13 跑在 commit d3c3029，BINS=linspace(0.15,0.85,15)。两个读数口径同时存在于日志里：
  - checkpoints[i]["bin_n"]：np.digitize，窗外个体从分子分母**同时剔除** ⇒ 截断
  - traj[j]["hist"]：_hist_fn 的 jnp.clip，尾巴进边缘箱 ⇒ 不截断（bin0 = pref<0.20，
    bin13 = pref>=0.80；low_mass = bins 0..3 = pref<0.35，分母是全体食草者）
两者时间对齐：窗口结束后补的那个 traj 点 t 与 cp["t"] 相同。

回答：①窗内比例 in_window = sum(bin_n)/n_samples 是多少，§18.5 的闸会不会触发；
     ②不截断口径下 cp0->cp4 的衰减是多少；③留存判据（split_score>0）换口径后如何。
重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260804-r14b-verdict/r13_clip.py
"""
import glob, json, math, sys
sys.path.insert(0, "scripts"); sys.path.insert(0, "explorations/20260804-readouts")
import numpy as np
from scipy.stats import wilcoxon
from exp_stats import bootstrap_ci
from split_score import split_score

BINS = np.linspace(0.15, 0.85, 15); CTR = 0.5*(BINS[:-1]+BINS[1:])
LOW = CTR < 0.35
runs = []
for f in sorted(glob.glob("outputs/20260804-traj/*.log")):
    for ln in open(f):
        if ln.startswith("JSON "):
            d = json.loads(ln[5:]); d["_f"] = f.split("/")[-1][:-4]; runs.append(d)
assert len(runs) == 24, len(runs)
print(f"载入 {len(runs)} run（R13）。BINS 长度 = {len(runs[0]['checkpoints'][0]['bin_n'])}（14 箱 ⇒ 旧的截断标尺）")

def traj_at(r, t):
    cands = [q for q in r["traj"] if q["t"] == t]
    assert len(cands) >= 1, (r["_f"], t)
    return cands[-1]

rows = []
for r in runs:
    row = {"f": r["_f"]}
    for i, cp in enumerate(r["checkpoints"]):
        n = np.array(cp["bin_n"], float)
        row[f"iw{i}"] = n.sum() / cp["n_samples"]           # 窗内比例
        row[f"trunc{i}"] = cp["low_mass"]                    # 截断口径（§14 判决用的）
        row[f"ss_trunc{i}"] = cp["split_score"]
        h = np.array(traj_at(r, cp["t"])["hist"], float)
        hn = h / max(h.sum(), 1)
        row[f"clip{i}"] = float(hn[LOW].sum())               # 不截断口径
        row[f"ss_clip{i}"] = float(traj_at(r, cp["t"])["split_score"])
        # 分解：低于 0.15 的那一段 = clip bin0 - digitize bin0（都归一到全体）
        row[f"below{i}"] = float(hn[0] - n[0]/cp["n_samples"])
        row[f"above{i}"] = float(hn[-1] - n[-1]/cp["n_samples"])
    rows.append(row)

print("\n" + "="*100)
print("① §18.5 的有效性闸回溯到 R13：in_window = sum(bin_n)/n_samples")
print("="*100)
for i in range(5):
    v = np.array([r[f"iw{i}"] for r in rows])
    print(f"  cp{i}  均值 {v.mean():.4f}  min {v.min():.4f}  <0.95 的 run {int((v<0.95).sum())}/24")
bad = sum(1 for r in rows if any(r[f"iw{i}"] < 0.95 for i in range(5)))
print(f"  ⇒ 任一检查点 in_window<0.95 的 run = {bad}/24（§18.5 阈值：某臂 >2/24 失效 ⇒ 该臂分箱判据整体作废）")
print(f"  最差的单个检查点：{min(((r[f'iw{i}'], r['f'], i) for r in rows for i in range(5)))}")
print("  窗外质量的去向（占全体食草者的比例）：")
for i in range(5):
    b = np.array([r[f"below{i}"] for r in rows]); a = np.array([r[f"above{i}"] for r in rows])
    print(f"    cp{i}  pref<0.15 均值 {b.mean():+.4f}(max {b.max():.4f})   pref>=0.85 均值 {a.mean():+.4f}(max {a.max():.4f})")

print("\n" + "="*100)
print("② cp0 -> cp4 的衰减：截断口径 vs 不截断口径")
print("="*100)
for tag, k in (("截断（§14 判决用的）", "trunc"), ("不截断（clip）", "clip")):
    x0 = np.array([r[f"{k}0"] for r in rows]); x4 = np.array([r[f"{k}4"] for r in rows])
    d = x4 - x0; lo, hi = bootstrap_ci(d)
    print(f"  {tag:>22}: cp0 {x0.mean():.4f} -> cp4 {x4.mean():.4f}  Δ={d.mean():+.4f}  "
          f"{int((d<0).sum())}/24 负 / {int((d>0).sum())} 正 / {int((d==0).sum())} 平  "
          f"p={wilcoxon(x4,x0).pvalue:.5f}  CI=[{lo:+.4f},{hi:+.4f}]")
    print(f"                          逐 run cp4 = {np.round(np.sort(x4),4).tolist()}")

print("\n" + "="*100)
print("③ 留存判据：split_score>0 —— 截断 bin_n（§14.2 用的）vs clip 直方图")
print("="*100)
for tag, k in (("截断 bin_n", "ss_trunc"), ("clip hist", "ss_clip")):
    ent = [r for r in rows if r[f"{k}0"] > 0]
    cont = [r for r in ent if all(r[f"{k}{i}"] > 0 for i in range(5))]
    cont24 = [r for r in rows if all(r[f"{k}{i}"] > 0 for i in range(5))]
    curve = [int(sum(1 for r in rows if r[f"{k}{i}"] > 0)) for i in range(5)]
    print(f"  {tag:>12}: 入组(cp0>0) {len(ent)}/24   留存曲线(分母24) {curve}   "
          f"连续五点全保住 {len(cont)}/{len(ent)}（全体口径 {len(cont24)}/24）")
print("  逐 run（clip 口径）五点 split_score>0 的模式：")
for r in sorted(rows, key=lambda z: z["f"]):
    pat = "".join("O" if r[f"ss_clip{i}"] > 0 else "." for i in range(5))
    patt = "".join("O" if r[f"ss_trunc{i}"] > 0 else "." for i in range(5))
    print(f"    {r['f']:>9}  clip {pat}   trunc {patt}   clip low_mass "
          f"{r['clip0']:.3f}->{r['clip4']:.3f}   trunc {r['trunc0']:.3f}->{r['trunc4']:.3f}")
np.save("explorations/20260804-r14b-verdict/output/r13_rows.npy", rows, allow_pickle=True)
