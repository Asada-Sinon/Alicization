"""敏感性：3 个后半程丢失捕食者的 run（wn1_s4 / wn1_s6 / wn2_s7）是不是另一个世界？

§16.5 预注册的阈值是「>4/24 才分开报」，实测 3/24，字面上不触发。
但这 3 个 run 同时是世代钟最慢的 3 个（118/133/161 代 vs 其余 ~490）
与食草种群最大的 3 个（min n_herb 2562–2852 vs 其余 173–1000+）——
去掉捕食者后食草者不再被压制，周转变慢、密度变高。它们是否在驱动结论？

重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260804-r13-verdict/sensitivity.py
"""
import glob, json, sys
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, "explorations/20260804-readouts")

NO_CARN = {"wn1_s4_r1.log", "wn1_s6_r1.log", "wn2_s7_r1.log"}
runs = []
for f in sorted(glob.glob("outputs/20260804-traj/*.log")):
    for ln in open(f):
        if ln.startswith("JSON "):
            d = json.loads(ln[5:]); d["_f"] = f.split("/")[-1]; runs.append(d)

print("三个丢失捕食者的 run 的画像")
print(f'  {"file":<16} {"gen_total":>10} {"min n_herb":>11} {"lm@cp0":>8} {"lm@cp4":>8} {"入组":>5}')
for r in runs:
    if r["_f"] in NO_CARN:
        print(f'  {r["_f"]:<16} {r["gen_total"]:>10.1f} '
              f'{min(q["n_herb"] for q in r["traj"]):>11d} '
              f'{r["checkpoints"][0]["low_mass"]:>8.4f} {r["checkpoints"][4]["low_mass"]:>8.4f} '
              f'{"Y" if r["checkpoints"][0]["split_score"]>0 else "-":>5}')
oth = [r for r in runs if r["_f"] not in NO_CARN]
print(f'  其余 21 个：gen_total 中位 {np.median([r["gen_total"] for r in oth]):.0f}，'
      f'min n_herb 中位 {np.median([min(q["n_herb"] for q in r["traj"]) for r in oth]):.0f}')

for tag, sub in (("全部 24", runs), ("去掉 3 个无捕食者 run（21）", oth)):
    ent = [r for r in sub if r["checkpoints"][0]["split_score"] > 0]
    a = np.array([r["checkpoints"][0]["low_mass"] for r in sub])
    b = np.array([r["checkpoints"][4]["low_mass"] for r in sub])
    ret = [sum(1 for r in ent if r["checkpoints"][i]["split_score"] > 0) for i in range(5)]
    lm = [np.mean([r["checkpoints"][i]["low_mass"] for r in ent]) for i in range(5)]
    print(f"\n[{tag}] 入组 {len(ent)}/{len(sub)}")
    print(f"  入组留存 " + " ".join(f"{v}/{len(ent)}" for v in ret))
    print(f"  入组 low_mass " + " ".join(f"{v:.4f}" for v in lm))
    print(f"  全体 cp0→cp4 low_mass {a.mean():.4f} → {b.mean():.4f}  差 {(b-a).mean():+.4f}  "
          f"{int((b<a).sum())} 降/{int((b>a).sum())} 升/{int((b==a).sum())} 平  "
          f"Wilcoxon p={wilcoxon(a, b).pvalue:.5f}")

print("\n捕食者压力与少数簇：末段 carnivore_frac vs cp4 low_mass（全 24）")
cf = np.array([np.nanmean([q["carnivore_frac"] for q in r["traj"][-40:]
                           if np.isfinite(q["carnivore_frac"])]) for r in runs])
lm4 = np.array([r["checkpoints"][4]["low_mass"] for r in runs])
from scipy.stats import spearmanr
print(f"  Spearman rho={spearmanr(cf, lm4).statistic:+.3f} p={spearmanr(cf, lm4).pvalue:.4f}")
nh = np.array([np.median([q["n_herb"] for q in r["traj"]]) for r in runs])
print(f"  中位 n_herb vs cp4 low_mass：rho={spearmanr(nh, lm4).statistic:+.3f} "
      f"p={spearmanr(nh, lm4).pvalue:.4f}")
print(f"  gen_total vs cp4 low_mass：rho={spearmanr([r['gen_total'] for r in runs], lm4).statistic:+.3f} "
      f"p={spearmanr([r['gen_total'] for r in runs], lm4).pvalue:.4f}")
