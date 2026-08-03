# R9 判决的独立复核（result-analyst，2026-08-03）

对 `outputs/20260803-shade/r9/*.log`（96 run）重算一遍，**不改 §13.5 判据**：核 `forage_pref`
符号、护栏改用预注册的 N1 自估噪声并**逐种子**读、补 H1/H2 的 MDE、复核 `criterion.txt` 的地形判据。

`audit_r9.py` → 符号/原始 rho/护栏逐种子/MDE；`audit_r9b.py` → 两种 H1 口径 + 护栏方向一致性；
`audit_r9c.py` → 地形复核 + 预注册漏掉的基因均值交互项。输出在 `output/`。
**注意 `audit_r9b.py` 的 [D] 段用错了十分位定义（按 share 全格取），那一段读数作废，以 `audit_r9c.py` [1] 为准。**
