# defence-evo ablation analysis (2026-07-24)
- `analyze.py` — per-seed table + paired Wilcoxon + bootstrap 95% CI + effect size for 10 fields, reads outputs/20260724-defence-evo/{on,off}_seed{0..5}.log
- `carn_vs_herb_drift.py` — tests whether carn>herb defence (seen in throwaway seed99 OFF) holds across 6 formal OFF runs
Rerun: `XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python explorations/20260724-defence-evo-analysis/analyze.py`
