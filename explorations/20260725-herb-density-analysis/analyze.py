# Question: Do any of three density levers (A repro_threshold, D density_repro_penalty,
#   B water economy) lower herbivore density (population / herb_count) WITHOUT driving
#   carnivores extinct (carn_frac min->0) or (for B) pushing juvenile thirst deaths back up?
# Reads: outputs/20260725-herb-density/{rt16,rt24,rt32}_seed{0..5}.log
#        outputs/20260725-density-repro/{p0,p08,p15}_seed{0..5}.log
#        outputs/20260725-water-tighten/{w_base,w_mild,w_full}_seed{0..5}.log
#   Each log: last line beginning "JSON " -> json.loads.
# Output: per-arm summary table + per-seed population/herb_count + paired Wilcoxon
#   (treatment vs its baseline) for population and herb_count, bootstrap 95% CI of mean
#   paired diff, rank-biserial. All numbers printed; nothing hand-computed.
import json, glob, os
import numpy as np
from scipy.stats import wilcoxon

SEEDS=[0,1,2,3,4,5]
def load(d,arm):
    rows={}
    for s in SEEDS:
        p=f"outputs/{d}/{arm}_seed{s}.log"
        line=[l for l in open(p) if l.startswith("JSON ")][-1]
        rows[s]=json.loads(line[5:])
    return rows

EXPS={
 "A repro_threshold":("20260725-herb-density",[("rt16(base=16)","rt16"),("rt24","rt24"),("rt32","rt32")]),
 "D density_repro_penalty":("20260725-density-repro",[("p0(base=0)","p0"),("p08(0.8)","p08"),("p15(1.5)","p15")]),
 "B water economy":("20260725-water-tighten",[("w_base(.01/.025)","w_base"),("w_mild(.015/.0375)","w_mild"),("w_full(.02/.05)","w_full")]),
}

def herb(r): return r["population"]*(1-r["carnivore_frac"])

def paired_stats(base,treat):
    d=np.array(treat)-np.array(base)  # treatment - baseline
    # Wilcoxon signed-rank
    try:
        w=wilcoxon(treat,base)
        p=w.pvalue
    except ValueError as e:
        p=float('nan')
    # rank-biserial for wilcoxon = (sum pos ranks - sum neg ranks)/total
    nz=d[d!=0]
    ranks=np.argsort(np.argsort(np.abs(nz)))+1
    Rpos=ranks[nz>0].sum(); Rneg=ranks[nz<0].sum(); T=ranks.sum()
    rb=(Rpos-Rneg)/T if T>0 else float('nan')
    # bootstrap 95% CI of mean paired diff
    rng=np.random.default_rng(42)
    boot=[rng.choice(d,size=len(d),replace=True).mean() for _ in range(10000)]
    lo,hi=np.percentile(boot,[2.5,97.5])
    return d.mean(),p,rb,lo,hi

for name,(d,arms) in EXPS.items():
    print("="*90); print(f"EXPERIMENT {name}  dir=outputs/{d}"); print("="*90)
    data={label:load(d,arm) for label,arm in arms}
    base_label=arms[0][0]
    # per-arm summary
    print(f"{'arm':<16}{'pop_mean':>10}{'herb_mean':>11}{'carn_mean':>10}{'carn_MIN':>10}{'mean_age':>10}{'thirst_frac':>12}{'thirst_age':>11}")
    for label,arm in arms:
        r=data[label]
        pop=np.mean([r[s]["population"] for s in SEEDS])
        hc=np.mean([herb(r[s]) for s in SEEDS])
        cf=[r[s]["carnivore_frac"] for s in SEEDS]
        age=np.mean([r[s]["mean_age"] for s in SEEDS])
        tf=np.mean([r[s]["death_thirst_frac"] for s in SEEDS])
        ta=np.mean([r[s]["death_thirst_age"] for s in SEEDS])
        print(f"{label:<16}{pop:>10.1f}{hc:>11.1f}{np.mean(cf):>10.4f}{min(cf):>10.4f}{age:>10.1f}{tf:>12.4f}{ta:>11.1f}")
    # per-seed population and herb_count
    for metric,fn in [("population",lambda r:r["population"]),("herb_count",herb)]:
        print(f"\n-- per-seed {metric} --")
        hdr="seed  "+"".join(f"{lab:>16}" for lab,_ in arms)
        print(hdr)
        for s in SEEDS:
            print(f"{s:<6}"+"".join(f"{fn(data[lab][s]):>16.1f}" for lab,_ in arms))
        # paired tests treat vs base
        base_vals=[fn(data[base_label][s]) for s in SEEDS]
        for lab,_ in arms[1:]:
            tv=[fn(data[lab][s]) for s in SEEDS]
            diffs=np.array(tv)-np.array(base_vals)
            md,p,rb,lo,hi=paired_stats(base_vals,tv)
            samedir=int(np.sum(np.sign(diffs)==np.sign(md)))
            print(f"   {lab} vs {base_label}: mean_diff={md:+.1f}  Wilcoxon_p={p:.4f}  rank-biserial={rb:+.3f}  boot95%CI=[{lo:+.1f},{hi:+.1f}]  same-dir={samedir}/6")
    # carn min per seed for treatments
    print("\n-- carnivore_frac per seed (extinction check) --")
    print("seed  "+"".join(f"{lab:>16}" for lab,_ in arms))
    for s in SEEDS:
        print(f"{s:<6}"+"".join(f"{data[lab][s]['carnivore_frac']:>16.4f}" for lab,_ in arms))
    print()
