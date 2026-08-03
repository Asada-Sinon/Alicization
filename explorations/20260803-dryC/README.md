# 20260803-dryC —— §12.3.C 定标探针的分析

`sweep_c.sh` / `sweep_c_w0.sh` 采数（5 臂 × 3 种子 × 1 重复，`outputs/20260803-dryC/`）；
`analyze_c.py` 是预注册五读数的原始读数表，`terrain_separation.py` 是地形级重心间距。
`workpoint.py` 定工作点（借同世界的 12×2/6×3 自估噪声当尺、诊断 `sel_ratio_water` 的跨臂
不可比性、重锚护栏），`bowl_separation.py` 是格子级饭碗分离度（纯地形，无 RNG）。

重跑：`XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python explorations/20260803-dryC/workpoint.py`
（输出落 `output/`，不进版本库）。
