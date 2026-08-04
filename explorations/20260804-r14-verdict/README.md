# R14 Stage 1 独立复核（judge 侧）

判据 `docs/multispecies_program.md` §17（预注册 `c26520a`）。读 `outputs/20260804-ratio/*.log`（48 run）。
核心发现：预注册主判据 `low_mass` 的直方图窗 [0.15,0.85] 在 `R50` 臂截断了过半质量（cp1 仅 41.9% 在窗内），
「衰减」与截断的相关 rho=+0.891 —— 该统计量在 `R50` 上不测它声称测的东西。

- `verdict_r14.py` — H0/H1/H2 复核 + 尺度无关口径 + 捕食者丢失混杂 + 通量/剂量内生性
- `check_truncation.py` — 截断诊断：检查点(digitize，剔除出界) vs traj(clip，收尾巴) 两口径对比
- `truncation_free.py` — 用 clip 直方图与不分箱的 `mean_pref` 重做 H1
- `shape.py` — 窗外的分布形状（`R50`/kept cp1 两端 0.472/0.288、中间带 0.000）
- `bimodal_check.py` — 用不分箱的 mean/sd 交叉验证双峰，并给两端质量的时间演化
- `h2_and_compliance.py` — H2 低组覆盖率空洞；±0.05 达标窗的可达率

全部 `XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python explorations/20260804-r14-verdict/<脚本>`
