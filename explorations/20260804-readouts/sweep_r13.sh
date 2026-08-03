#!/usr/bin/env bash
# R13：少数簇是稳定多态还是暂态。判据见 docs/multispecies_program.md §16（先写后跑）。
# 2 地形 × 12 种子 × 1 重复 = 24 run，每个 100k 步 5 检查点。
# ⚠️ 共用卡：显存闸 + 断点续跑。⚠️ 跑的时候它读的文件都不许动。
set -u
RUN=outputs/20260804-traj
SCRIPT=explorations/20260804-readouts/trajectory.py
CONC=${CONC:-5}; NEED_MIB=${NEED_MIB:-2600}
W=(--set fruit_regrow_baseline=0.25 --set fruit_energy=4.0 --set fruit_water_frac=0.40
   --set regrow_baseline=0.010 --set plant_max=2.0 --set water_sea_dist=1
   --set grass_shade=1.3 --set forage_tradeoff=1.0 --set forage_curvature=1.0
   --set eat_rate=0.5)
SEEDS=(0 1 2 3 4 5 6 7 8 9 10 11); WN=(1 2)
free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1; }
mkdir -p "$RUN"
{ echo "run_id: 20260804-traj"; echo "git HEAD: $(git rev-parse HEAD)"
  echo "dirty: $([ -z "$(git status --porcelain)" ] && echo false || echo true)"
  echo "判据: docs/multispecies_program.md §16（先写后跑）"
  echo "world: ${W[*]}"; echo "terrain: ridge_wavenumber ${WN[*]}"
  echo "seeds: ${SEEDS[*]}  steps: 100000  checkpoints: 5"
  echo "started: $(date -Is) free: $(free_mib)"; } >> "$RUN/provenance.txt"
for wn in "${WN[@]}"; do for s in "${SEEDS[@]}"; do
  out="$RUN/wn${wn}_s${s}_r1.log"
  grep -q '^JSON ' "$out" 2>/dev/null && continue
  while [ "$(jobs -rp|wc -l)" -ge "$CONC" ]; do wait -n; done
  while [ "$(free_mib)" -lt "$NEED_MIB" ]; do sleep 30; done
  XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python "$SCRIPT" 100000 --json \
    --seed "$s" --checkpoints 5 "${W[@]}" --set "ridge_wavenumber=$wn" > "$out" 2>&1 &
  sleep 6
done; done
wait
ok=$(grep -l '^JSON ' "$RUN"/*.log 2>/dev/null|wc -l)
echo "finished: $(date -Is) ok=$ok/24" >> "$RUN/provenance.txt"; echo "done: $ok/24"
