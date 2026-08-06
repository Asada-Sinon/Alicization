# 20260806-r18b-review — 独立复核 `feasibility.md` §24（R18-B 判决）

`review_r18b.py` 只读 `outputs/20260806-gen700/` 与 `outputs/20260805-gen450/` 的 `JSON` 行，
不 import `underworld`（复核时用户正在改 `trait_dim`）。统计走 `scripts/exp_stats.bootstrap_ci`。

跑：`XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python explorations/20260806-r18b-review/review_r18b.py`
