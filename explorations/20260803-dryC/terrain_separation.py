"""草层与果层的空间分离度，随 `fruit_dry_weight` 变化。**只依赖地形，不读任何 run。**

回答什么：§12 的目标一句话就是「把两个饭碗在空间上拉开」，而 §12.3.C 预注册的五个读数
**没有一个直接量这个**——它们量的都是「种群相对果层」，不是「果层相对草层」。
`terrain.build` 不吃 RNG，所以这张表算一次就是确定的，零成本。

**诚实标注：这个读数是在看到 C 阶段数据之后才补的，不是预注册项。** 补它的理由不是数据
好看，而是它直接对应 §12 的立项目标，且完全不依赖 run 数据——没有「看了结果再挑口径」的
空间（地形表在跑任何 run 之前就已经确定了）。

两列的读法：

- **重心间距**（`Σ w·water_dist / Σ w` 之差）是主要的那列。
- **Schoener D** 带 §12.3.A 记过的**集中度混杂**：果层集中在少数格、草层摊满全图，
  所以 D 会低估「同处一地」的程度。只作旁证。

重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260803-dryC/terrain_separation.py
"""
import sys

sys.path.insert(0, ".")

import numpy as np

from underworld import Config
from underworld import terrain as terrain_mod

WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]


def schoener_d(p: np.ndarray, q: np.ndarray) -> float:
    p = p / p.sum()
    q = q / q.sum()
    return float(1.0 - 0.5 * np.abs(p - q).sum())


base = terrain_mod.build(Config())
grass = np.asarray(base.capacity)
wd = np.asarray(base.water_dist)
grass_c = float((wd * grass).sum() / grass.sum())

print("=" * 88)
print("草层 × 果层 的空间分离度（terrain.build，无 RNG，确定值）")
print("=" * 88)
print(f'  {"w":>5} {"果层重心水距":>13} {"草层重心水距":>13} {"重心间距":>10} '
      f'{"D(草,果)":>10} {"果层总承载":>11}')
for w in WEIGHTS:
    fc = np.asarray(terrain_mod.build(Config(fruit_dry_weight=w)).fruit_capacity)
    fc_c = float((wd * fc).sum() / fc.sum())
    print(f"  {w:>5} {fc_c:>13.2f} {grass_c:>13.2f} {fc_c - grass_c:>+10.2f} "
          f"{schoener_d(grass, fc):>10.4f} {fc.sum():>11.2f}")

print()
print("  读法：曲线**非单调**，而且非单调的方式会误导工作点的选择。")
print("  草层重心并不在河边——capacity 主要是 grass_base 的均匀底噪，重心 49.75 约等于陆地")
print("  水距中位数 45.2。而 w=0 的果层重心在 25.80，**两个饭碗本来就有约 24 个世界单位的")
print("  间距，只是果层偏在河的一侧**。所以把果层往外推时，它会先**穿过**草层重心")
print("  （w=0.25，间距只剩 +4.0、D 反而升到最高），再才拉开。")
print("  => w=0.25 是曲线上最差的点，不是「弱一点的处理」。")
