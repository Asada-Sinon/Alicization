# 20260725-spike-redesign
Paired ON/OFF analysis of the spike redesign (carn=offense, herb=venom-defense).
`analyze.py` reads outputs/20260724-spike-redesign/{on,off}_seed{0..5}.log (last "JSON " line),
prints per-seed fields + paired Wilcoxon + rank-biserial + 10000-rep bootstrap 95% CI.
Run: XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python explorations/20260725-spike-redesign/analyze.py
