R10（`outputs/20260803-curvature/r10`，96 run）判决的**独立复算 + 三项追加检查**，不引用 `r10_analysis.txt` 里的任何数字。

- `verify_r10.py` — 复核 §14.4 H1–H4 与 §14.5 护栏；追加：假设自己的预测效应量 `A·q_g(k)+B·q_f(k)`、z 单位尺度混杂、分箱曲线的一阶项。
- `mechanism_r10.py` — 实测对预测值的检验、种群走到的 `s` 范围、`quad_demog` 在中性臂的偏倚、分箱摄入曲线形状。
- `local_curvature_r10.py` — 「即使完美交付 `(1−k)(A+B)` 够不够翻号」、全域 vs ±2.5SD 局部曲率、补齐 §14.4 两道闸所需的 run 数。
