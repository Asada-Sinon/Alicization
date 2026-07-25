# 20260725-carrion-terrain

腐食 v2 (`carrion_enabled`) carn_frac 正效应的跨地形去伪重复分析。
`analyze.py` 读 `outputs/20260725-carrion-terrain/results.jsonl`（48 行，4 地形×ON/OFF×seed0-5，配对，仅 carrion_enabled 差异），
输出逐地形配对 Wilcoxon + 效应量、跨地形符号一致性、wn2_b0.40 反向诊断、population 代价方向、pooled 检验。
重跑：`XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python explorations/20260725-carrion-terrain/analyze.py`
