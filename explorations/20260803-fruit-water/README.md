# 20260803-fruit-water — R1c 果实的预成水 6 配对种子正式判决

`analyze.py` 读 `outputs/20260803-fruit-water/{A_base,C_fwf10,D_fwf40,E_fwf60}_seed{0..5}.log` 的 `JSON ` 行，
打印逐种子全量表、预注册判据（`docs/multispecies_program.md:407-411`，护栏逐种子 6/6）逐条判定、
`fruit_water_frac` 剂量-响应下的渴死超出量单调性、六组配对 Wilcoxon + rank-biserial + bootstrap CI、
`water_system.md` 预标风险核对、异常与方差。

重跑: `XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python explorations/20260803-fruit-water/analyze.py`（cwd 仓库根）。
