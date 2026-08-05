# R17 判决的独立复核（`outputs/20260805-isolation`，48 run × 100k 步）

`verify_r17.py` 复核判据实现（单变量归因、`readout_valid` 护栏、三种噪声口径）并补预注册
里写了但读数表没读的量（轨迹口径的 `split_score_second` / `retained_occ_first`、簇构成、
逐检查点 LD）；`split_artifact.py` 查明 `split_score` 反向是**检测器失效**（12/24 个 W20 run
的平滑直方图是平台、无严格局部极大 ⇒ 分数按构造为 0），并给出容忍平台的修补版重算；
`closeout.py` 拆 `mean_pref` 的上升、报臂内跨种子散度与护栏。

跑：`XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python explorations/20260805-r17-verdict/<脚本>.py`
（输出留在 `output/`，已被 `.gitignore` 覆盖）。
