"""Carrion scavenging 6-seed ON/OFF analysis (docs/multispecies_feasibility.md §7).
Reads outputs/20260725-carrion/{on,off}_seed{0..5}.log, prints per-metric paired
means + 6-seed direction + paired Wilcoxon. Rerun:
  XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python explorations/20260725-carrion/analyze.py
"""
import json, glob, statistics as st
from scipy.stats import wilcoxon

def load(arm):
    return {int(f.split('seed')[1].split('.')[0]):
            json.loads([x for x in open(f) if x.startswith('JSON')][0][5:])
            for f in glob.glob(f'outputs/20260725-carrion/{arm}_seed*.log')}

off, on = load('off'), load('on')
for k in ['carnivore_frac', 'population', 'death_thirst_frac', 'carrion_total',
          'mean_age', 'herb_water_dist']:
    o = [off[s][k] for s in range(6)]; n = [on[s][k] for s in range(6)]
    same = sum(1 for a, b in zip(n, o) if a > b)
    try:
        w, p = wilcoxon(n, o); ps = f"W={w:.1f} p={p:.4f}"
    except Exception as e:
        ps = str(e)
    print(f"{k:18s} OFF {st.mean(o):8.3f}  ON {st.mean(n):8.3f}  diff {st.mean(n)-st.mean(o):+8.3f}"
          f"  {same}/6 ON>OFF  {ps}")
print("per-seed carn_frac OFF:", [round(off[s]['carnivore_frac'], 3) for s in range(6)])
print("per-seed carn_frac ON :", [round(on[s]['carnivore_frac'], 3) for s in range(6)])
