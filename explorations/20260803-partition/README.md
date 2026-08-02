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

## analyze_curve.py — P4「窄分化窗口」收口 (五点剂量-响应)

回答什么：P3 只测了 0.0/0.5/1.0，看不到比自身间距更窄的窗口。本轮补 `forage_tradeoff`
= 0.125 (L) 与 0.25 (N)，判据是 `sd(L)` 或 `sd(N)` 是否**高于**中性漂变对照 `sd(P,0.0)`。

读什么：同目录 90 个 log（`{P_tradeoff0,L_tradeoff0125,N_tradeoff025,M_tradeoff05,
Q_tradeoff1}_s{0..5}_r{1..3}.log`，每个第 12 行一条 `JSON {...}`）+ `provenance.txt`
（三段追加；P/Q=9296fd7、M=7937499、L/N=c1f9527，三者间 `underworld/`+`scripts/` 无 diff）。

重跑：`XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python explorations/20260803-partition/analyze_curve.py`
（§10 口径敏感性与 §11 MDE 是本脚本相对 analyze_p3.py 新增的两节。）
