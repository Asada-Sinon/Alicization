# R18 判决的独立复核（fable5，2026-08-06）

三个脚本，全部只读 `outputs/20260805-gen450/`（R18，24 run）与
`outputs/20260805-longrun/R38n_*.log`（R16，24 run），不改任何被冻结的文件。

- `audit_retraction.py` — 恒等式独立验证 + **用修复前的 `split_score` 复现 §19 已发表的
  四分位表**（这是判定「撤回对不对」的决定性证据）+ 两个口径下的 Q4−Q3 + Δ 层面五个口径。
- `audit_saturation.py` — 在**全部**检查点上重算「两峰/谷见底/retained」（§22.4 只在两个窗上算）
  + 24/24 减速表 + 少数簇质量轨迹。`--last` 跑末段增量的分析单位对比。
- `audit_numbers.py` — 勘误块「10 倍」的口径核对、H1 复算、Δ_H2 被解释掉的比例、窗覆盖率。

跑：`XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python explorations/20260806-r18-audit/<脚本>`
