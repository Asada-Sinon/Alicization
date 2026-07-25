# 判决共享报警场 alarm 6 配对种子实验（docs/multispecies_feasibility.md §8.4）。
# 读: outputs/20260725-alarm/results.jsonl（12 行，arm ON/OFF × seed 0..5）。
# 主判据: death_predation_frac ON<OFF, 6/6 同向 + 配对 Wilcoxon p + 效应量 + 10000 bootstrap 95% CI。
# 护栏: population, death_thirst_frac, death_thirst_age, carnivore_frac。机制点火: alarm_total。
# 输出直接读 stdout。
import json, numpy as np
from scipy import stats

rows = [json.loads(l) for l in open("/home/michael/workspace/pi05/temp/alicization/outputs/20260725-alarm/results.jsonl")]
ON  = {r["seed"]: r for r in rows if r["arm"]=="ON"}
OFF = {r["seed"]: r for r in rows if r["arm"]=="OFF"}
seeds = sorted(ON)
rng = np.random.default_rng(20260725)

def paired(metric, direction):
    # direction: 'ON<OFF' means improvement is ON smaller
    on  = np.array([ON[s][metric]  for s in seeds])
    off = np.array([OFF[s][metric] for s in seeds])
    diff = on - off  # ON - OFF
    # same-direction count toward hypothesis
    if direction=='ON<OFF':
        same = int(np.sum(diff < 0))
    else:
        same = int(np.sum(diff > 0))
    # paired wilcoxon (two-sided)
    try:
        w = stats.wilcoxon(on, off)
        p = w.pvalue
    except Exception as e:
        p = float('nan')
    # effect size r = Z/sqrt(N) via normal approx of wilcoxon; use matched-pairs rank-biserial
    # rank-biserial for wilcoxon signed rank:
    d = diff[diff!=0]
    if len(d):
        r_ranks = stats.rankdata(np.abs(d))
        Rpos = r_ranks[d>0].sum(); Rneg = r_ranks[d<0].sum()
        T = Rpos+Rneg
        rrb = (Rpos-Rneg)/T
    else:
        rrb = 0.0
    # bootstrap 95% CI of mean paired diff
    boots = np.array([rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(10000)])
    ci = np.percentile(boots, [2.5, 97.5])
    return on, off, diff, same, p, rrb, ci

def report(name, metric, direction):
    on, off, diff, same, p, rrb, ci = paired(metric, direction)
    print(f"\n=== {name} ({metric}), 期望方向 {direction} ===")
    print(f"{'seed':>4} {'ON':>12} {'OFF':>12} {'diff(ON-OFF)':>14}")
    for i,s in enumerate(seeds):
        print(f"{s:>4} {on[i]:>12.5f} {off[i]:>12.5f} {diff[i]:>14.5f}")
    print(f"mean ON={on.mean():.5f}  mean OFF={off.mean():.5f}  mean diff={diff.mean():.5f}")
    print(f"同向种子(符合{direction}): {same}/6")
    print(f"配对Wilcoxon 双侧 p={p:.4f}   rank-biserial r={rrb:+.3f}")
    print(f"bootstrap(10000) 95% CI of mean diff: [{ci[0]:.5f}, {ci[1]:.5f}]")
    # seed-to-seed SD for context
    print(f"OFF臂种子间SD={off.std(ddof=1):.5f}  ON臂种子间SD={on.std(ddof=1):.5f}")

print("############ 主判据 ############")
report("被捕食死亡率", "death_predation_frac", "ON<OFF")

print("\n############ 护栏 ############")
report("总种群", "population", "ON>OFF")          # 恶化=ON<OFF
report("渴死占比", "death_thirst_frac", "ON<OFF") # 恶化=ON>OFF
report("渴死年龄", "death_thirst_age", "ON>OFF")  # 恶化=ON<OFF(更早渴死)
report("食肉者占比", "carnivore_frac", "ON>OFF")   # 崩=ON<OFF
report("min_pop", "min_pop", "ON>OFF")

print("\n############ 机制点火 alarm_total ############")
print(f"{'seed':>4} {'ON alarm_total':>16} {'OFF alarm_total':>16}")
for s in seeds:
    print(f"{s:>4} {ON[s]['alarm_total']:>16.2f} {OFF[s]['alarm_total']:>16.2f}")

print("\n############ overrides 归因核对 ############")
print("ON overrides:", set(json.dumps(ON[s]['overrides'],sort_keys=True) for s in seeds))
print("OFF overrides:", set(json.dumps(OFF[s]['overrides'],sort_keys=True) for s in seeds))
print("ON steps:", set(ON[s]['steps'] for s in seeds), " OFF steps:", set(OFF[s]['steps'] for s in seeds))
