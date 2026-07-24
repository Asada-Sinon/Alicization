# 20260725 herbivore-density lever analysis
Question: do any of 3 density levers (A repro_threshold, D density_repro_penalty,
B water economy) lower herb density without extincting carnivores / (B) without
thirst-death return? Reads the 3 outputs/20260725-* dirs' JSON lines.
Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python explorations/20260725-herb-density-analysis/analyze.py`
Paired Wilcoxon (treatment vs its baseline) for population & herb_count, bootstrap
95% CI of mean paired diff, rank-biserial. n=6 paired -> floor p=0.031.
