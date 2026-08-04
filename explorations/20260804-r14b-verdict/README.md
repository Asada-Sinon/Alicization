# R14 Stage 2 判决 + R13 判决的读数回溯（2026-08-04）

判据 `docs/multispecies_program.md` §18（`df4b3ed`，跑之前提交）；数据 `outputs/20260804-ratio2/*.log`（96 run）与 `outputs/20260804-traj/*.log`（R13，24 run）。
`verify_stage2.py` 从原始日志复现 `outputs/20260804-ratio2_analysis.txt` 的每个数，并补 §18.4 没写的「每臂 Δ 是否异于 0」；`per_arm_noise.py` 查 §18.7 的四臂池化噪声是否掩盖了 n 臂信号，并把 Δ 换算到每代；`r13_clip.py` 把 §18.5 的有效性闸回溯到 R13 并用不截断口径重算衰减；`why_ss_zero.py` / `r13_detector.py` 查 `split_score` 在边缘双峰上为何读 0；`r13_null_redo.py` 用不截断口径重建中性零假设并重算 `s`；`r13_collateral.py` 查 §14.7 等其余读数是否连带失效；`cross_check.py` 把 R38p 当作 R13 wn1 的重跑对一次，并算 H1 的 MDE。
全部脚本无 GPU 需求；重跑命令写在各自 docstring 里。
