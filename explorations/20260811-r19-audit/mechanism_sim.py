#!/usr/bin/env python
"""C 臂(随机轴)mean gene 上涨的机制模拟——检验判决 §25.2 的「分布形状」猜想。

复刻 d3eda56 版 `_assortative_mate` 的排序-相邻配对到单位点模型:
  key = ((1-w)*rank01(diet) + w*rank01(axis)) / sqrt((1-w)^2+w^2)   [未中心化!]
  wanter 按 key 排序,相邻配对,**奇数时队尾自配**(swap 越界 -> self)。
  后代 gene = 随机取一亲(单位点均匀交叉等价) + N(0,0.02);替换随机死者。

规模对齐真实 C 臂:POP=5600,每步 k~Poisson(8.6) 个 wanter(=每步出生数,
由 pop*gen_total/steps = 5600*153/100k 实测),100k 步 ≈ 153 代。

变体:
  r19        未中心化归一(实测代码)
  center     先减 0.5 再归一(均值对 w 恒定;残余=判决猜想的「形状」效应)
  r19_noself 未中心化,但奇数时队尾不繁殖——分离「自配」通路
  cent_noself 中心化+无自配——纯配对结构(理论上不动均值)
  rand       key=U(0,1)(D 臂,模拟自身的零校验)

读数:gen40/gen153 的 mean gene(8 rep 均值±SD),自配率,自配者 gene 超群体均值的量。
"""
import numpy as np

POP, RATE, STEPS, REPS = 5600, 8.6, 100_000, 8
GEN_STEP = POP / RATE                     # 步/代 ≈ 651
MUT, INIT = 0.02, 0.4

def w_of(g):
    return np.clip(1.0 / (1.0 + np.exp(-g)) - 0.5, 0.0, None)

def run(variant, seed):
    rng = np.random.default_rng(seed)
    g = rng.normal(0.0, INIT, POP)
    r1 = rng.permutation(POP) / (POP - 1.0)          # diet 的全体秩(固定属性,子代继承)
    born = 0
    self_n, self_gain = 0, 0.0
    snap = {}
    for step in range(STEPS):
        k = rng.poisson(RATE)
        if k < 2:
            continue
        idx = rng.choice(POP, k, replace=False)      # wanter 与 gene 无关
        w = w_of(g[idx])
        r2 = rng.random(k)                            # axis 每步重抽 => 秩 ~ U(0,1)
        norm = np.sqrt((1 - w) ** 2 + w ** 2)
        if variant == 'rand':
            key = rng.random(k)
        elif variant in ('center', 'cent_noself'):
            key = ((1 - w) * (r1[idx] - .5) + w * (r2 - .5)) / norm + .5
        else:
            key = ((1 - w) * r1[idx] + w * r2) / norm
        order = np.argsort(key)
        swap = np.arange(k) ^ 1
        drop_tail = variant.endswith('noself') and (k % 2 == 1)
        swap = np.where(swap < k, swap, np.arange(k))  # 奇数 -> 队尾自配
        parents = idx[order]
        mates = idx[order][swap]
        if drop_tail:
            parents, mates = parents[:-1], mates[:-1]
        m = len(parents)
        # 自配统计
        sm = parents == mates
        self_n += int(sm.sum())
        if sm.any():
            self_gain += float((g[parents[sm]] - g.mean()).sum())
        # 后代:随机取一亲 + 突变;替换随机死者
        pick = rng.random(m) < 0.5
        child = np.where(pick, g[parents], g[mates]) + rng.normal(0, MUT, m)
        child_r1 = np.clip(np.where(pick, r1[parents], r1[mates])
                           + rng.normal(0, 0.01, m), 0, 1)
        die = rng.choice(POP, m, replace=False)
        g[die] = child
        r1[die] = child_r1
        born += m
        gen = born / POP
        for tag, gg in (('g40', 40), ('g153', 153)):
            if tag not in snap and gen >= gg:
                snap[tag] = g.mean()
    snap.setdefault('g153', g.mean())
    return snap.get('g40', np.nan), snap['g153'], self_n / max(born, 1), \
        self_gain / max(self_n, 1)

for variant in ('r19', 'center', 'r19_noself', 'cent_noself', 'rand'):
    out = np.array([run(variant, 1000 + i) for i in range(REPS)])
    g40, g153, sr, sg = out.T
    print(f'{variant:12s} gen40 = {g40.mean():+.4f}±{g40.std(ddof=1):.4f}   '
          f'gen153 = {g153.mean():+.4f}±{g153.std(ddof=1):.4f}   '
          f'自配率={sr.mean()*100:.2f}%  自配者Δgene={sg.mean():+.3f}', flush=True)
