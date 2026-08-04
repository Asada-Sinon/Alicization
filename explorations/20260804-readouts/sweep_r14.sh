#!/usr/bin/env bash
# R14 Stage 1：把资源供给比抬到 50:50，衰减停不停。判据见 §17（先写后跑）。
# 2 臂 × 12 种子 × r=2 = 48 run × 42k 步 2 检查点，约 2 小时。
# ⚠️ 共用卡：显存闸 + 断点续跑。⚠️ 跑的时候它读的文件都不许动。
set -u
RUN=outputs/20260804-ratio
SCRIPT=explorations/20260804-readouts/trajectory.py
CONC=${CONC:-5}; NEED_MIB=${NEED_MIB:-2600}
W=(--set fruit_energy=4.0 --set fruit_water_frac=0.40 --set plant_max=2.0
   --set water_sea_dist=1 --set grass_shade=1.3 --set forage_tradeoff=1.0
   --set forage_curvature=1.0 --set eat_rate=0.5 --set ridge_wavenumber=1)
SEEDS=(0 1 2 3 4 5 6 7 8 9 10 11); REPS=(1 2)
# 臂名:fruit_regrow_baseline:regrow_baseline
ARMS=(R38:0.25:0.010 R50:0.50:0.005)
free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1; }
mkdir -p "$RUN"
{ echo "run_id: 20260804-ratio  (R14 Stage 1)"; echo "git HEAD: $(git rev-parse HEAD)"
  echo "dirty: $([ -z "$(git status --porcelain)" ] && echo false || echo true)"
  echo "判据: docs/multispecies_program.md §17（先写后跑）"
  echo "world: ${W[*]}"; echo "arms: ${ARMS[*]}"
  echo "seeds: ${SEEDS[*]} reps: ${REPS[*]} steps: 42000 checkpoints: 2"
  echo "started: $(date -Is) free: $(free_mib)"; } >> "$RUN/provenance.txt"
for a in "${ARMS[@]}"; do
  IFS=: read -r tag frb rb <<< "$a"
  for s in "${SEEDS[@]}"; do for r in "${REPS[@]}"; do
    out="$RUN/${tag}_s${s}_r${r}.log"
    grep -q '^JSON ' "$out" 2>/dev/null && continue
    while [ "$(jobs -rp|wc -l)" -ge "$CONC" ]; do wait -n; done
    while [ "$(free_mib)" -lt "$NEED_MIB" ]; do sleep 30; done
    XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python "$SCRIPT" 42000 --json \
      --seed "$s" --checkpoints 2 "${W[@]}" \
      --set "fruit_regrow_baseline=$frb" --set "regrow_baseline=$rb" > "$out" 2>&1 &
    sleep 5
  done; done
done
wait
ok=$(grep -l '^JSON ' "$RUN"/*.log 2>/dev/null|wc -l)
echo "finished: $(date -Is) ok=$ok/48" >> "$RUN/provenance.txt"; echo "done: $ok/48"
