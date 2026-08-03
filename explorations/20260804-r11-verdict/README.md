R11 判决的独立复核（result-analyst，2026-08-04）。判据 = `docs/multispecies_program.md` §15（commit `b73b2dc`，跑前已提交），一字未改。
读 `outputs/20260804-demand/*.log`（96 新 run）+ `outputs/20260803-curvature/r10/{N,K10}_*.log`（48 复用）+ `outputs/20260803-curvature/sat/*.log`（24 探针）。
按序跑 `verify_r11.py`（复用闸/主判据/归因闸/low_mass 补算/护栏）→ `b`（直方图/frac_mid 补算/quad_intake 单位/§15.6 护栏修复）→ `c`（簇内摄入/T15 的谷）→ `d`（逐 run 摄入/逐种子谷）→ `e`（单帧 vs 池化口径的功效）。
