#!/usr/bin/env python
"""mechanism_sim 的补充:繁殖成批(burst)时排序偏置是否放大。

真实 want=能量>阈值,日夜与进食同步 => n_want 可能远超 Poisson 均值的散布。
保持出生率 8.6/步不变,k 的分布改为「p=0.1 时 86 个,否则 0」(burst86)
与「p=0.2 时 43 个」(burst43)。其余与 mechanism_sim.py 的 r19 变体一致。
"""
import numpy as np
POP, STEPS, REPS = 5600, 100_000, 8
MUT, INIT = 0.02, 0.4
def w_of(g): return np.clip(1.0/(1.0+np.exp(-g))-0.5, 0.0, None)
def run(kmode, seed):
    rng = np.random.default_rng(seed)
    g = rng.normal(0.0, INIT, POP)
    r1 = rng.permutation(POP)/(POP-1.0)
    born=0; self_n=0; self_gain=0.0; snap={}
    for step in range(STEPS):
        p, size = kmode
        k = size if rng.random() < p else 0
        if k < 2: continue
        idx = rng.choice(POP, k, replace=False)
        w = w_of(g[idx]); r2 = rng.random(k)
        norm = np.sqrt((1-w)**2 + w**2)
        key = ((1-w)*r1[idx] + w*r2)/norm
        order = np.argsort(key)
        swap = np.arange(k)^1
        swap = np.where(swap<k, swap, np.arange(k))
        parents = idx[order]; mates = idx[order][swap]
        sm = parents==mates; self_n += int(sm.sum())
        if sm.any(): self_gain += float((g[parents[sm]]-g.mean()).sum())
        pick = rng.random(k)<0.5
        child = np.where(pick, g[parents], g[mates]) + rng.normal(0,MUT,k)
        cr1 = np.clip(np.where(pick, r1[parents], r1[mates]) + rng.normal(0,0.01,k),0,1)
        die = rng.choice(POP, k, replace=False)
        g[die]=child; r1[die]=cr1; born+=k
        gen = born/POP
        for tag,gg in (('g40',40),('g153',153)):
            if tag not in snap and gen>=gg: snap[tag]=g.mean()
    snap.setdefault('g153', g.mean())
    return snap.get('g40',np.nan), snap['g153'], self_n/max(born,1), self_gain/max(self_n,1)
for name,kmode in (('burst86',(0.1,86)), ('burst43',(0.2,43))):
    out = np.array([run(kmode, 2000+i) for i in range(REPS)])
    g40,g153,sr,sg = out.T
    print(f'{name:10s} gen40={g40.mean():+.4f}±{g40.std(ddof=1):.4f}  '
          f'gen153={g153.mean():+.4f}±{g153.std(ddof=1):.4f}  '
          f'自配率={sr.mean()*100:.2f}%  自配者Δgene={sg.mean():+.3f}', flush=True)
