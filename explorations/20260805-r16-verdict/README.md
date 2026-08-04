# 20260805-r16-verdict — R16 判决前的独立复核

- `verify_r16.py` — 复核 `20260804-readouts/analyze_r16.py` 的表头数字（更多位数），
  并补出判据要求、读数表却没打印的几项：`retained_occ` 自动字段、`mean_pref` / `in_window` /
  逐箱摄入、逐 run 的 `min(carnivore_frac)`、逐四分位轨迹，以及两个**非预注册**的旁证
  （世代匹配后的 H3′、未饱和读数上的 H2）。输出 `output/verify_r16.txt`。
- `null_subgroup.py` — 把 R38p 的 H1 计数 7/24 按「丢失捕食者 4 / 存活 20」拆开各跑一次
  中性零假设（**事后分组，不是预注册判决**，§19.5 只准分开报不准筛）。输出 `output/null_subgroup.txt`。
- 跑法：`XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python explorations/20260805-r16-verdict/<脚本>`
