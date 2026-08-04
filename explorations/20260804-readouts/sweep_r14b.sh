#!/usr/bin/env bash
# R14 Stage 2 / 方案 B：把捕食者通道由构造固定住再问资源比。判据见 §18（先写后跑）。
# 4 臂 × 12 种子 × r=2 = 96 run × 42k 步 2 检查点，约 3.5 小时。
# diet_delta=1.5 ⇒ 猎物需比攻击者食草 1.5 以上，而 diet∈[0,1] ⇒ 结构性无捕食。
# ⚠️ 共用卡：显存闸 + 断点续跑。⚠️ 跑的时候它读的文件都不许动。
set -u
RUN=outputs/20260804-ratio2
SCRIPT=explorations/20260804-readouts/trajectory.py
CONC=${CONC:-5}; NEED_MIB=${NEED_MIB:-2600}
W=(--set fruit_energy=4.0 --set fruit_water_frac=0.40 --set plant_max=2.0
   --set water_sea_dist=1 --set grass_shade=1.3 --set forage_tradeoff=1.0
   --set forage_curvature=1.0 --set eat_rate=0.5 --set ridge_wavenumber=1)
SEEDS=(0 1 2 3 4 5 6 7 8 9 10 11); REPS=(1 2)
# 臂名:fruit_regrow_baseline:regrow_baseline:diet_delta
ARMS=(R38p:0.25:0.010:0.15 R50p:0.50:0.005:0.15 R38n:0.25:0.010:1.5 R50n:0.50:0.005:1.5)
free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1; }
mkdir -p "$RUN"
{ echo "run_id: 20260804-ratio2  (R14 Stage 2 / 方案 B)"; echo "git HEAD: $(git rev-parse HEAD)"
  echo "dirty: $([ -z "$(git status --porcelain)" ] && echo false || echo true)"
  echo "判据: docs/multispecies_program.md §18（先写后跑）"
  echo "world: ${W[*]}"; echo "arms: ${ARMS[*]}"
  echo "读数已修: BINS=linspace(0,1,21) + in_window/readout_valid 有效性闸"
  echo "seeds: ${SEEDS[*]} reps: ${REPS[*]} steps: 42000 checkpoints: 2"
  echo "started: $(date -Is) free: $(free_mib)"; } >> "$RUN/provenance.txt"
for a in "${ARMS[@]}"; do
  IFS=: read -r tag frb rb dd <<< "$a"
  for s in "${SEEDS[@]}"; do for r in "${REPS[@]}"; do
    out="$RUN/${tag}_s${s}_r${r}.log"
    grep -q '^JSON ' "$out" 2>/dev/null && continue
    while [ "$(jobs -rp|wc -l)" -ge "$CONC" ]; do wait -n; done
    while [ "$(free_mib)" -lt "$NEED_MIB" ]; do sleep 30; done
    XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python "$SCRIPT" 42000 --json \
      --seed "$s" --checkpoints 2 "${W[@]}" --set "fruit_regrow_baseline=$frb" \
      --set "regrow_baseline=$rb" --set "diet_delta=$dd" > "$out" 2>&1 &
    sleep 5
  done; done
done
wait
ok=$(grep -l '^JSON ' "$RUN"/*.log 2>/dev/null|wc -l)
echo "finished: $(date -Is) ok=$ok/96" >> "$RUN/provenance.txt"; echo "done: $ok/96"
