# 20260802-fruit-flux — R1b 果层供给扫描（流量口径）6 配对种子正式判决

`analyze.py` 读 `outputs/20260802-fruit-flux/verdict/*.log`（18 个 run 的 `JSON ` 行），打印逐种子表、
预注册判据逐条判定（两种读法）、B/C 对 A_base 及 C 对 B 的配对 Wilcoxon + rank-biserial + bootstrap CI、异常与方差。

`forest_null.py` 从 `terrain.build(Config())` 算 `forest_frac` 的四个零模型（该指标是「站在林冠下的活体占比」，
没有零模型不能说高低）。两个脚本都加 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 前缀、cwd 为仓库根运行。
