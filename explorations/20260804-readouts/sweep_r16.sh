#!/usr/bin/env bash
# R16：那个 50/50 双簇在 200+ 代上稳不稳。判据见 `multispecies_program.md` §19（先写后跑）。
# 2 臂 × 12 种子 × r=2 = 48 run × 100k 步 × 4 检查点，CONC=3 约 5 小时。
# 两臂只差 `diet_delta`：1.5 由构造无捕食（猎物需比攻击者食草 1.5 以上，而 diet∈[0,1]）。
# ⚠️ 共用卡：显存闸 + 断点续跑（`grep -q '^JSON '` 已完成的直接跳过）。
# ⚠️ 跑的时候它读的文件都不许动。
set -u
RUN=outputs/20260805-longrun
SCRIPT=explorations/20260804-readouts/trajectory.py
CONC=${CONC:-3}; NEED_MIB=${NEED_MIB:-2600}
W=(--set fruit_energy=4.0 --set fruit_water_frac=0.40 --set plant_max=2.0
   --set water_sea_dist=1 --set grass_shade=1.3 --set forage_tradeoff=1.0
   --set forage_curvature=1.0 --set eat_rate=0.5 --set ridge_wavenumber=1
   --set fruit_regrow_baseline=0.25 --set regrow_baseline=0.010)
SEEDS=(0 1 2 3 4 5 6 7 8 9 10 11); REPS=(1 2)
ARMS=(R38n:1.5 R38p:0.15)          # 臂名:diet_delta
free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1; }
mkdir -p "$RUN"
{ echo "run_id: 20260805-longrun  (R16 长时程稳定性)"; echo "git HEAD: $(git rev-parse HEAD)"
  echo "dirty: $([ -z "$(git status --porcelain)" ] && echo false || echo true)"
  echo "判据: docs/multispecies_program.md §19（先写后跑，commit b583504）"
  echo "world: ${W[*]}"; echo "arms: ${ARMS[*]}"
  echo "⚠️ 不是默认世界：以上 9 项与 Config() 不同（forage_tradeoff 默认 0.0）"
  echo "seeds: ${SEEDS[*]} reps: ${REPS[*]} steps: 100000 checkpoints: 4  CONC=$CONC"
  echo "started: $(date -Is) free: $(free_mib)"; } >> "$RUN/provenance.txt"
for a in "${ARMS[@]}"; do
  IFS=: read -r tag dd <<< "$a"
  for s in "${SEEDS[@]}"; do for r in "${REPS[@]}"; do
    out="$RUN/${tag}_s${s}_r${r}.log"
    grep -q '^JSON ' "$out" 2>/dev/null && continue
    while [ "$(jobs -rp|wc -l)" -ge "$CONC" ]; do wait -n; done
    while [ "$(free_mib)" -lt "$NEED_MIB" ]; do sleep 30; done
    XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python "$SCRIPT" 100000 --json \
      --seed "$s" --checkpoints 4 "${W[@]}" --set "diet_delta=$dd" > "$out" 2>&1 &
    sleep 5
  done; done
done
wait
ok=$(grep -l '^JSON ' "$RUN"/*.log 2>/dev/null|wc -l)
echo "finished: $(date -Is) ok=$ok/48" >> "$RUN/provenance.txt"; echo "done: $ok/48"
