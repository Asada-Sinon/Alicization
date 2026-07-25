# 问题：判决「forage_pref 资源分割第二食草者」6配对种子实验。
#   主判据 forage_pref_std ON>OFF（方差=双峰必要下界），载体正确性，护栏不恶化。
# 读：outputs/20260725-forage/results.jsonl（12行，ON/OFF×seed0-5）
# 判据来源：docs/multispecies_feasibility.md §9.3
# 输出读法：stdout 直接给逐种子表、Wilcoxon p、效应量 r、bootstrap CI、护栏检验。
import json, numpy as np
from scipy import stats

rows=[json.loads(l) for l in open("outputs/20260725-forage/results.jsonl") if l.strip()]
ON ={r["seed"]:r for r in rows if r["arm"]=="ON"}
OFF={r["seed"]:r for r in rows if r["arm"]=="OFF"}
seeds=sorted(set(ON)&set(OFF))
print("配对种子:",seeds,"n=",len(seeds))

def paired(metric, higher_is_worse=None):
    on =np.array([ON[s][metric]  for s in seeds])
    off=np.array([OFF[s][metric] for s in seeds])
    d=on-off
    # 配对 Wilcoxon 双侧
    try:
        w=stats.wilcoxon(on,off)
        p=w.pvalue
    except Exception as e:
        p=float('nan')
    # 效应量 r = Z/sqrt(N)  用正态近似的 z（wilcoxon 精确p反推不稳，用符号秩的 rank-biserial）
    # rank-biserial for paired: r = (favorable - unfavorable sum)/total  —— 用简单 matched-pairs rank-biserial
    absd=np.abs(d); nz=absd>0
    ranks=stats.rankdata(absd[nz])
    pos=ranks[d[nz]>0].sum(); neg=ranks[d[nz]<0].sum()
    T=pos+neg
    rrb=(pos-neg)/T if T>0 else 0.0
    # bootstrap CI on mean difference (paired)
    rng=np.random.default_rng(0)
    bs=np.array([d[rng.integers(0,len(d),len(d))].mean() for _ in range(20000)])
    ci=np.percentile(bs,[2.5,97.5])
    samedir=np.sum(np.sign(d)==np.sign(np.median(d))) if np.median(d)!=0 else 0
    npos=int(np.sum(d>0)); nneg=int(np.sum(d<0))
    return on,off,d,p,rrb,ci,npos,nneg

print("\n==== 主判据: forage_pref_std (ON>OFF?) ====")
on,off,d,p,r,ci,npos,nneg=paired("forage_pref_std")
print("seed |   ON_std |  OFF_std |   diff")
for i,s in enumerate(seeds):
    print(f"  {s}  | {on[i]:.5f} | {off[i]:.5f} | {d[i]:+.5f}")
print(f"mean  ON={on.mean():.5f}  OFF={off.mean():.5f}  diff={d.mean():+.5f}")
print(f"ON>OFF 种子数: {npos}/{len(seeds)}   ON<OFF: {nneg}/{len(seeds)}")
print(f"配对Wilcoxon 双侧 p={p:.4f}   rank-biserial r={r:+.3f}   diff 95%CI=[{ci[0]:+.5f},{ci[1]:+.5f}]")
print(f"[种子间SD ON={on.std(ddof=1):.5f} OFF={off.std(ddof=1):.5f}]")

print("\n==== 载体正确: mean_forage_pref (ON 是否绕0.5分裂而非整体偏移?) ====")
on,off,d,p,r,ci,npos,nneg=paired("mean_forage_pref")
print("seed | ON_mean | OFF_mean |  diff")
for i,s in enumerate(seeds):
    print(f"  {s}  | {on[i]:.4f} | {off[i]:.4f} | {d[i]:+.4f}")
print(f"mean ON={on.mean():.4f} OFF={off.mean():.4f}  ON偏离0.5={on.mean()-0.5:+.4f}")
print(f"Wilcoxon p={p:.4f} r={r:+.3f} CI=[{ci[0]:+.4f},{ci[1]:+.4f}] (ON>OFF {npos}/{len(seeds)})")

print("\n==== 载体: herb_forage_pref (实际拨盘) ====")
on,off,d,p,r,ci,npos,nneg=paired("herb_forage_pref")
print(f"mean ON={on.mean():.4f} OFF={off.mean():.4f} diff={d.mean():+.4f}  Wilcoxon p={p:.4f} (ON>OFF {npos}/{len(seeds)})")

print("\n==== 护栏 (ON 不应恶化) ====")
for m,worse in [("population","low"),("carnivore_frac","low"),("late_carn","low"),
                ("death_thirst_frac","high"),("min_pop","low"),("plant_total","either")]:
    on,off,d,p,r,ci,npos,nneg=paired(m)
    tag={"low":"ON更低=恶化","high":"ON更高=恶化","either":""}[worse]
    print(f"{m:20s} ON={on.mean():.4f} OFF={off.mean():.4f} diff={d.mean():+.4f} "
          f"CI=[{ci[0]:+.4f},{ci[1]:+.4f}] Wilcoxon p={p:.4f} (ON>OFF {npos}/{len(seeds)}) {tag}")
