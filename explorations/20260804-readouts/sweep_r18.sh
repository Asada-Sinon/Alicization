#!/usr/bin/env bash
# R18：那个 50/50 双簇撑得到 450 代吗。判据见 `multispecies_program.md` §21（先写后跑）。
# 只跑 R38n 一臂（R38p 在 R16 里已到 452 代）：12 种子 × r=2 = 24 run × 350k 步 × 6 检查点。
# **用户已把卡腾空并授权全卡**，所以 CONC 提到 8（每 run 约 0.9 GiB，8 个约 7 GiB）。
# ⚠️ 跑的时候源码树冻结（它会陆续起新进程 import underworld）。
# ⚠️ python 加 `-u`：日志块缓冲会让人以为卡住了。
set -u
RUN=outputs/20260805-gen450
SCRIPT=explorations/20260804-readouts/trajectory.py
CONC=${CONC:-8}; NEED_MIB=${NEED_MIB:-2600}
W=(--set fruit_energy=4.0 --set fruit_water_frac=0.40 --set plant_max=2.0
   --set water_sea_dist=1 --set grass_shade=1.3 --set forage_tradeoff=1.0
   --set forage_curvature=1.0 --set eat_rate=0.5 --set ridge_wavenumber=1
   --set fruit_regrow_baseline=0.25 --set regrow_baseline=0.010 --set diet_delta=1.5)
SEEDS=(0 1 2 3 4 5 6 7 8 9 10 11); REPS=(1 2)
free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1; }
mkdir -p "$RUN"
{ echo "run_id: 20260805-gen450  (R18 撑得到 450 代吗)"; echo "git HEAD: $(git rev-parse HEAD)"
  echo "dirty: $([ -z "$(git status --porcelain)" ] && echo false || echo true)"
  echo "判据: docs/multispecies_program.md §21（先写后跑，commit dbf8622）"
  echo "world: ${W[*]}"; echo "arm: R38n only (diet_delta=1.5 由构造无捕食)"
  echo "⚠️ 不是默认世界：9 项与 Config() 不同（forage_tradeoff 默认 0.0）"
  echo "MDE 已跑前算死：low_mass σ̂_W=0.0093（均值 2%）；预测效应÷MDE=18.6"
  echo "seeds: ${SEEDS[*]} reps: ${REPS[*]} steps: 350000 checkpoints: 6  CONC=$CONC"
  echo "started: $(date -Is) free: $(free_mib)"; } >> "$RUN/provenance.txt"
for s in "${SEEDS[@]}"; do for r in "${REPS[@]}"; do
  out="$RUN/R38n_s${s}_r${r}.log"
  grep -q '^JSON ' "$out" 2>/dev/null && continue
  while [ "$(jobs -rp|wc -l)" -ge "$CONC" ]; do wait -n; done
  while [ "$(free_mib)" -lt "$NEED_MIB" ]; do sleep 30; done
  XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python -u "$SCRIPT" 350000 --json \
    --seed "$s" --checkpoints 6 "${W[@]}" > "$out" 2>&1 &
  sleep 5
done; done
wait
ok=$(grep -l '^JSON ' "$RUN"/*.log 2>/dev/null|wc -l)
echo "finished: $(date -Is) ok=$ok/24" >> "$RUN/provenance.txt"; echo "done: $ok/24"
