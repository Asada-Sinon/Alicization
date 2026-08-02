# 20260803-partition — R2「厚果层下的资源分割」P2 判决

回答什么：厚果层（果供能约 28%）下，`forage_tradeoff=1.0`（Q，最大选择强度）的
`forage_pref` 分布 `sd` 与 `blrt_lr_per_n` 是否高于 `0.0`（P，基因编译期断开＝纯漂变对照）
——预注册 P2，`docs/multispecies_program.md:539-540`。

读什么：`outputs/20260803-partition/{P_tradeoff0,Q_tradeoff1}_s{0..5}_r{1..3}.log`（36 个，
每个末尾一行 `JSON {...}`）+ 同目录 `provenance.txt`。每格 3 次重复先取均值再做 6 配对种子检验。

重跑：`XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python explorations/20260803-partition/analyze.py`
