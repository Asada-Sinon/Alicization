# 20260803-overlapA — §12 A 阶段：果层 × 种群 重叠度基线

回答什么：`docs/multispecies_program.md` §12.3 **A 阶段**——「现在果层和种群到底重叠多少？」
产出一个基线重叠度数字，**后面 C 阶段扫 `fruit_dry_weight` 时所有「挪开了没有」都对着它读**。
没有这个数，「分离」无从判定。

读什么：`outputs/20260803-overlapA/base_s{0..11}_r{1,2}.log`（24 个 = 12 种子 × 2 重复，
新协议 `docs/run_to_run_variance.md` §7.1）。每个末尾一行 `JSON {...}`。

**为什么 A 阶段也跑 12×2**：这一轮不只要基线的**值**，还要基线的**噪声**——C 和 D 的判据
容差必须对着 `√2·σ̂_W` 定（口径 3）。跑 12×2 就地估出重叠度指标的 `σ̂_W`，一次到位。

三个口径 + 三个零模型的定义见 `measure_overlap.py` 的模块 docstring。主口径是
**`sel_ratio_water`**（按水距分层匹配的选择比）——它问的是「扣掉『渴死把所有人钉在河边』
之后，种群还额外偏好果层吗」，正是手术要动的那个耦合。

重跑单个 run：

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
  explorations/20260803-overlapA/measure_overlap.py 20000 --seed 0 --json
```

重跑整批：`bash outputs/20260803-overlapA/sweep.sh`（并发上限 6）。
判决分析：`analyze.py`（import `scripts/exp_stats.py`）。
