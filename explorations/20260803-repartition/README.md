# 20260803-repartition — R3「等量再分配」6 配对种子判决的分析

- `analyze.py` — 主分析。读 `outputs/20260803-repartition/verdict/*.log`（18 个 run 的 `JSON` 行 + 表格行）、
  `outputs/20260803-fruit-water/*.log`（R1c 对照，重算 ρ）、`outputs/20260803-repartition/{probe,replicate}/*.log`。
  §0 归因 §1 逐种子 §2 预注册五条判据 §3 配对 Wilcoxon §4 种群超调×渴死 §5 捕食者风险
  §6 异常/方差/口径 §7 判决 §8 探针-判决复现对 §9 同种子同配置复现噪声 §10 用复现噪声当尺子 §11 生态位分化直读。
- `terrain_null.py` — `forest_frac` 的零假设（terrain 无 RNG，与种子无关），并把三臂实测放到刻度上。
- 重跑：`XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python explorations/20260803-repartition/{analyze,terrain_null}.py`
