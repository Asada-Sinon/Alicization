# R13 判决的独立复核（2026-08-04）

判据 `docs/multispecies_program.md` §16（commit `d3c3029`，跑之前提交）；数据 `outputs/20260804-traj/*.log`（24 run × 100k 步 × 5 检查点）。
`verify_r13.py` 复现 `analyze_r13.py` 的每个数并补诊断；`detector_check.py` 查 `split_score>0` 到底在检测什么；`extra_tests.py` 查入组选择造成的回归均值；`neutral_null.py` / `null_test.py` / `null_clean.py` / `null_pct.py` 用单倍体 WF 模拟建中性零假设（Ne 在模拟器里自定标，不用 §16.1 的解析式）；`sensitivity.py` 查 3 个丢失捕食者的 run 是否驱动结论。
全部输出在 `output/`（已 gitignore）；重跑命令写在每个脚本的 docstring 里。
