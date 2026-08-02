# 20260803-partition — R2「厚果层下的资源分割」P2 判决

回答什么：厚果层（果供能约 28%）下，`forage_tradeoff=1.0`（Q，最大选择强度）的
`forage_pref` 分布 `sd` 与 `blrt_lr_per_n` 是否高于 `0.0`（P，基因编译期断开＝纯漂变对照）
——预注册 P2，`docs/multispecies_program.md:539-540`。

读什么：`outputs/20260803-partition/{P_tradeoff0,Q_tradeoff1}_s{0..5}_r{1..3}.log`（36 个，
每个末尾一行 `JSON {...}`）+ 同目录 `provenance.txt`。每格 3 次重复先取均值再做 6 配对种子检验。

重跑：`XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python explorations/20260803-partition/analyze.py`

## analyze_p3.py — P3「权衡强度剂量项」

回答什么：补 `forage_tradeoff=0.5`（M 臂）这个中间档，判定 R2 测到的稳定化挤压
（sd 从 P(0.0)=0.09508 掉到 Q(1.0)=0.05866）**是不是随权衡强度单调** —— 预注册见
`outputs/20260803-partition/provenance.txt:29-34`（§9.3 P3）。

读什么：同目录 54 个 log（`{P_tradeoff0,M_tradeoff05,Q_tradeoff1}_s{0..5}_r{1..3}.log`，
每个第 12 行一条 `JSON {...}`）。注意 `_r1/_r2/_r3` 是**同种子复跑**，格内散度来自 GPU
原子重排，不是创始者差异 —— 脚本 §10 会把这一点核出来。

重跑：`XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python explorations/20260803-partition/analyze_p3.py`
